"""CCA test suite — Phoenix-traced integration tests.

Each test gets its own Phoenix project (e.g. eva-code-trace) via the
PHOENIX_PROJECT_NAME env var set by GitLab CI. Test and CCA server
spans share a single trace via W3C traceparent propagation — each test
appears as one unified trace tree in Phoenix.

Every test creates spans visible in the Phoenix UI (set PHOENIX_URL env var).

Annotations are deferred until AFTER the span is closed and flushed to
Phoenix. This avoids 404 errors caused by posting annotations for spans
that haven't arrived at the server yet (BatchSpanProcessor buffers for
5 seconds by default).

Server load management:
  - Inter-test cooldown (CCA_TEST_COOLDOWN env var, default 3s) prevents
    overwhelming the CCA server and vLLM backend between tests.
  - LLM judge is opt-in (--with-judge flag or CCA_RUN_JUDGE=1 env var)
    since it doubles vLLM load by making 2 additional calls per test.
"""

from __future__ import annotations

import logging
import os
import time
import uuid

import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import StatusCode
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .cca_client import CCAClient, TIMEOUT_DIAGNOSTIC
from .evaluators import post_deferred_annotations

log = logging.getLogger(__name__)

# ==================== Configuration ====================

PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://192.168.4.204:4317")
CCA_BASE_URL = os.getenv("CCA_BASE_URL", "https://192.168.4.205:8500")

# Per-test Phoenix projects. Each test gets its own project named
# "test/{canonical_name}" where canonical_name is the file stem per
# the doc standard (see tests/_naming.py). One Phoenix project per
# test file. Functions inside the file are sub-checks; their traces
# all go to the same project. Parametrized variants land there too.
# Retries pile up — the project page IS the test's run history.
#
# The trace_test fixture below sets this per-test via using_project()
# and tells the CCAClient to send "X-Phoenix-Project: <test_project>"
# so the server routes its spans to the same project (full hierarchy
# in one place: test -> cca.request -> cca.agent -> vLLM).
#
# PHOENIX_PROJECT_NAME env var is honored only as the fallback project
# for the shared TracerProvider Resource — used by spans that aren't
# wrapped in using_project() (session-level setup spans, if any).
PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "cca-http")

from ._naming import canonical_name

# Phoenix per-scope project override (uses contextvars; propagates
# through async). Falls back to a no-op if the dependency is missing.
try:
    from openinference.instrumentation import dangerously_using_project as _using_project
except ImportError:
    from contextlib import nullcontext as _nullctx
    def _using_project(_name):  # type: ignore[misc]
        return _nullctx()

# Inter-test cooldown (seconds) to prevent server overload.
# Each test triggers LLM inference on vLLM — without cooldown,
# sequential tests pile up requests faster than vLLM can drain them.
TEST_COOLDOWN = float(os.getenv("CCA_TEST_COOLDOWN", "3"))


# ==================== Markers ====================


def pytest_addoption(parser):
    parser.addoption(
        "--with-judge",
        action="store_true",
        default=False,
        help="Enable LLM judge evaluators (doubles vLLM load, off by default)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "user: User identification and profile tests")
    config.addinivalue_line("markers", "websearch: Web search and URL fetch tests")
    config.addinivalue_line("markers", "integration: Multi-tool integration tests")
    config.addinivalue_line("markers", "coder: CODER route tool tests (file, bash, search, graph, docs, rules)")
    config.addinivalue_line("markers", "slow: Tests that take more than 60 seconds")
    config.addinivalue_line("markers", "trace: Code trace and assemble tests")
    config.addinivalue_line("markers", "knowledge: Knowledge pipeline and memory tests")
    config.addinivalue_line("markers", "isolation: Route tool isolation boundary tests")
    config.addinivalue_line("markers", "eva: EVA project real-world pipeline tests")


# ==================== Phoenix / OpenTelemetry ====================


@pytest.fixture(scope="session")
def phoenix_provider():
    """Single TracerProvider for all tests — per-test Phoenix project.

    The project name comes from PHOENIX_PROJECT_NAME env var (set by
    GitLab CI per test job). Falls back to 'cca-http' for local runs.

    W3C traceparent propagation is enabled by default (opentelemetry-sdk
    installs TraceContextTextMapPropagator globally). CCAClient's inject()
    adds traceparent headers so server spans become children of the test trace.
    """
    resource = Resource.create({
        "service.name": PROJECT_NAME,
        "openinference.project.name": PROJECT_NAME,
    })
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    yield provider
    provider.shutdown()


@pytest.fixture(scope="session")
def phoenix_tracer(phoenix_provider):
    """Single tracer for the entire test session."""
    tracer = phoenix_provider.get_tracer(PROJECT_NAME)
    return phoenix_provider, tracer


def _post_test_result(cca_url: str, headers: dict, ci_job: str,
                      pipeline_id: str, node_id: str, status: str,
                      metrics: dict, failure_reason: str,
                      trace_id: str = "", primary_session_id: str = "",
                      phoenix_project: str = ""):
    """Post individual test result to CCA DB. Fire-and-forget.

    Called in CI mode only (when CI_PIPELINE_ID and RUN_TEST are set).
    The /admin/test-result endpoint upserts the result and recomputes
    TestDefinition aggregate status (FAILED if ANY individual test failed).

    trace_id / primary_session_id / phoenix_project carry the Phoenix
    linkage so the dashboard can deep-link each TestResult to its
    exact Phoenix trace (instead of guessing a project URL by name).
    """
    try:
        import httpx as _httpx
        _httpx.post(
            f"{cca_url}/admin/test-result",
            json={
                "ci_job_name": ci_job,
                "pipeline_id": pipeline_id,
                "node_id": node_id,
                "status": status,
                "duration_ms": metrics.get("execution_time_ms", 0),
                "route": metrics.get("route", ""),
                "failure_reason": (failure_reason or "")[:500],
                "tool_iterations": metrics.get("tool_iterations", 0),
                "turns": len(metrics.get("_turns", [])),
                "trace_id": trace_id,
                "primary_session_id": primary_session_id,
                "phoenix_project": phoenix_project,
            },
            headers=headers,
            timeout=10,
            verify=False,  # internal CCA HTTPS cert is self-signed
        )
    except Exception as e:
        log.warning("Failed to post test result: %s", e)


@pytest.fixture(scope="module", autouse=True)
def trace_file(request, phoenix_tracer, cca):
    """One root span per test FILE. All functions in the file (and all
    parametrize cases of each) become child spans under this root —
    Phoenix collapses them into ONE trace per file run, regardless of
    how many pytest invocations the file contains.

    The doc standard (docs/testing/writing-tests.md) says the FILE is
    the test. Functions inside are sub-checks; parametrize cases are
    different invocations of the same sub-check. They all belong to
    one trace per file run. Retries (next pipeline run) produce a new
    trace inside the same Phoenix project.

    Also pins CCAClient.project_name for the whole file so any
    session-scoped diagnostic call (health check, cleanup) routes to
    the right per-file project instead of the global cca-http.
    """
    _provider, tracer = phoenix_tracer
    file_label = canonical_name(request.module.__file__)
    file_project = f"test/{file_label}"

    _prev_project = cca.project_name
    cca.project_name = file_project
    request.addfinalizer(lambda: setattr(cca, "project_name", _prev_project))

    # The `with` block stays open for the whole module's run because
    # pytest holds this generator paused at `yield`. OTel context
    # propagation keeps this span "current" so each per-function
    # `trace_test` span auto-parents under it — one trace, many spans.
    with _using_project(file_project), \
            tracer.start_as_current_span(f"file::{file_label}") as root:
        root.set_attribute("openinference.span.kind", "CHAIN")
        root.set_attribute("cca.test.file", file_label)
        root.set_attribute("cca.test.module", request.module.__name__)
        yield root


@pytest.fixture(autouse=True)
def trace_test(request, phoenix_tracer, cca, trace_file):
    """Wrap every test function in a child span under `trace_file`'s
    root. Each parametrize case becomes a child too — they all share
    the file's trace_id, so Phoenix shows them as one trace.

    Annotations are collected during the test via span._pending_annotations
    and posted AFTER the span closes + flushes — fixing the 404 race.
    """
    provider, tracer = phoenix_tracer

    test_path = request.node.nodeid
    if "/user/" in test_path:
        category = "user"
    elif "/websearch/" in test_path:
        category = "websearch"
    elif "/integration/" in test_path:
        category = "integration"
    elif "/coder/" in test_path:
        category = "coder"
    else:
        category = "other"

    test_name = request.node.name
    span_name = f"{category}::{test_name}"

    # File-stem project (set once by trace_file; re-asserted here for
    # the function-level span's own using_project tag — same value).
    test_project = f"test/{canonical_name(request.node)}"

    with _using_project(test_project), tracer.start_as_current_span(span_name) as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("cca.test.name", test_name)
        span.set_attribute("cca.test.category", category)
        span.set_attribute("cca.test.nodeid", test_path)

        # Evaluators append annotation dicts here during the test
        span._pending_annotations = []
        # CCA client stashes metrics here for per-test reporting
        span._test_metrics = {}

        yield span

        # Set I/O attributes from accumulated chat turns (done after yield
        # so ALL turns are collected before we write).
        #
        # cca_client.chat() promotes the root span to LLM kind on first
        # call, so we always emit llm.input_messages / llm.output_messages
        # here (both single- and multi-turn) — Phoenix then renders the
        # chat-conversation panel. The CHAIN default at line 171 stays
        # in place for tests that never call chat().
        turns = span._test_metrics.get("_turns", [])
        if turns:
            # Idempotent overwrite of the kind set in cca_client.chat()
            span.set_attribute("openinference.span.kind", "LLM")
            idx = 0
            for i, (msg, resp) in enumerate(turns):
                span.set_attribute(
                    f"llm.input_messages.{idx}.message.role", "user"
                )
                span.set_attribute(
                    f"llm.input_messages.{idx}.message.content",
                    f"[Turn {i + 1}] {msg}" if len(turns) > 1 else msg,
                )
                idx += 1
                if i < len(turns) - 1:
                    # Intermediate assistant responses go into input context
                    span.set_attribute(
                        f"llm.input_messages.{idx}.message.role", "assistant"
                    )
                    span.set_attribute(
                        f"llm.input_messages.{idx}.message.content",
                        f"[Turn {i + 1}] {resp}",
                    )
                    idx += 1
            # Final assistant response is the output
            final_resp = turns[-1][1]
            span.set_attribute(
                "llm.output_messages.0.message.role", "assistant"
            )
            span.set_attribute(
                "llm.output_messages.0.message.content",
                f"[Turn {len(turns)}] {final_resp}" if len(turns) > 1 else final_resp,
            )

        if hasattr(request.node, "rep_call"):
            rep = request.node.rep_call
            span.set_attribute("cca.test.passed", rep.passed)
            span.set_attribute("cca.test.outcome", rep.outcome)
            if rep.passed:
                span.set_status(StatusCode.OK)
            else:
                span.set_status(StatusCode.ERROR, rep.longreprtext[:500] if hasattr(rep, "longreprtext") else "test failed")
        else:
            # No rep_call means setup failed or test was skipped
            span.set_status(StatusCode.ERROR, "no test result available")

    # Span is now CLOSED — flush to Phoenix, then post annotations.
    # force_flush() sends via gRPC; Phoenix needs a moment to persist
    # to Postgres before the span is queryable for annotations.
    provider.force_flush()
    time.sleep(1)
    post_deferred_annotations(span)

    metrics = getattr(span, "_test_metrics", {})

    # Compute pytest outcome — always available regardless of metrics.
    # Non-agent tests (knowledge cmdlet checks, direct API tests) don't
    # populate metrics, but they still have a rep_call from pytest.
    # Without this, those tests' results never make it to the dashboard.
    outcome_str = "PASSED"
    failure_reason = ""
    if hasattr(request.node, "rep_call"):
        outcome_str = request.node.rep_call.outcome.upper()  # PASSED|FAILED|SKIPPED
        if request.node.rep_call.failed:
            failure_reason = str(request.node.rep_call.longrepr)[:2000]
    else:
        # No rep_call → setup error before the test body ran. Mark as
        # ERROR so the dashboard surfaces it instead of treating as PASSED.
        outcome_str = "ERROR"

    # Per-test summary line for real-time monitoring
    if metrics:
        route = metrics.get("route", "?")
        steps = metrics.get("estimated_steps", "?")
        iters = metrics.get("tool_iterations", "?")
        elapsed_s = metrics.get("execution_time_ms", 0) / 1000
        flags = []
        if metrics.get("nudge_skipped"):
            flags.append("nudge_skipped")
        if metrics.get("circuit_breaker_fired"):
            flags.append("CB_FIRED")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(
            f"\n  >> {span_name}: {outcome_str} | "
            f"route={route} steps={steps} iters={iters} "
            f"{elapsed_s:.1f}s{flag_str}",
            flush=True,
        )

    # Post individual test result to DB (CI mode only) — runs for EVERY
    # pytest test, regardless of whether it called the agent. Knowledge
    # tests like test_cmdlet_search are parametrized 100+ times and only
    # show up on the dashboard via this post. metrics may be empty dict
    # for non-agent tests; the endpoint accepts that.
    _pipeline_id = os.environ.get("CI_PIPELINE_ID", "")
    _ci_job = os.environ.get("RUN_TEST", "")
    if _pipeline_id and _ci_job:
        _api_key = os.environ.get("CCA_TEST_API_KEY", "")
        _auth_h = (
            {"Authorization": f"Bearer {_api_key}"} if _api_key else {}
        )

        # Phoenix linkage — extract trace_id from the test root span and
        # session_id from the first session the test tracked. These let
        # the dashboard deep-link to the exact Phoenix trace.
        try:
            sc = span.get_span_context()
            _trace_id_hex = f"{sc.trace_id:032x}" if sc and sc.is_valid else ""
        except Exception:
            _trace_id_hex = ""
        _sessions = metrics.get("_session_ids", []) if metrics else []
        _primary_session = _sessions[0] if _sessions else ""

        _post_test_result(
            cca_url=os.environ.get(
                "CCA_BASE_URL", "https://192.168.4.205:8500"
            ),
            headers=_auth_h,
            ci_job=_ci_job,
            pipeline_id=_pipeline_id,
            node_id=_ci_job,  # use the dash-form RUN_TEST, not the long span_name
            status=outcome_str,
            metrics=metrics,
            failure_reason=failure_reason,
            trace_id=_trace_id_hex,
            primary_session_id=_primary_session,
            phoenix_project=test_project,
        )

    # Generate .md test report for Claude Code review (agent tests only —
    # non-agent tests have no transcript/turns/iterations to render).
    #
    # Report folder name = canonical name (file stem) per the standard.
    # CI sets _ci_job = RUN_TEST = canonical name, so use that when
    # available; for local runs fall back to deriving from the pytest
    # node via the shared helper. Either way the folder is
    # P{pipeline_id}_{canonical-name} and matches the dashboard's
    # expectation.
    if metrics:
        try:
            from tests.report_generator import generate_test_report

            _report_name = _ci_job if _ci_job else canonical_name(request.node)

            test_dir = generate_test_report(
                test_name=_report_name,
                status=outcome_str,
                turns=metrics.get("_turns", []),
                metrics=metrics,
                evals_per_turn=metrics.get("_evals"),
                turn_details=metrics.get("_turn_details"),
                failure_reason=failure_reason,
                ci_job_name=_ci_job,
            )

            # Capture CCA debug logs into the test's folder
            if test_dir:
                try:
                    import httpx as _httpx
                    cca_url = os.environ.get(
                        "CCA_BASE_URL", "https://192.168.4.205:8500"
                    )
                    # Auth required (CCA_API_AUTH=1) — use same key as test client
                    api_key = os.environ.get("CCA_TEST_API_KEY", "")
                    _auth_headers = (
                        {"Authorization": f"Bearer {api_key}"} if api_key else {}
                    )
                    resp = _httpx.get(
                        f"{cca_url}/admin/debug-logs?tail=5000",
                        timeout=15,
                        headers=_auth_headers,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("status") == "ok":
                            from pathlib import Path
                            Path(test_dir / "cca_debug.log").write_text(
                                data.get("logs", "")
                            )
                            log.info(
                                "Captured %d debug log lines for %s",
                                data.get("line_count", 0), span_name,
                            )
                    else:
                        log.warning(
                            "debug-logs returned %d for %s",
                            resp.status_code, span_name,
                        )
                except Exception as e:
                    log.warning("Failed to capture CCA debug logs: %s", e)

                # Clear CCA logs so next test starts clean
                try:
                    _httpx.post(
                        f"{cca_url}/admin/clear-logs",
                        timeout=10,
                        headers=_auth_headers,
                    )
                except Exception:
                    pass  # Best effort

        except Exception as e:
            log.warning("Failed to generate test report: %s", e)

    # Inter-test cooldown: let vLLM drain its queue before the next test
    # starts a new LLM call. Without this, sequential tests pile up
    # requests and cause timeout cascades.
    if TEST_COOLDOWN > 0:
        log.debug("Cooldown %.1fs before next test", TEST_COOLDOWN)
        time.sleep(TEST_COOLDOWN)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Store test result on node for trace_test fixture to read."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


# ==================== LLM Judge (direct vLLM, NOT CCA) ====================


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://192.168.4.208:8000/v1")
VLLM_MODEL = "/models/Qwen3.5-35B-A3B-FP8"


@pytest.fixture(scope="session")
def judge_model(request):
    """Direct vLLM connection for LLM-as-judge evaluators.

    **Opt-in**: Returns None (disabled) unless --with-judge flag is passed
    or CCA_RUN_JUDGE=1 env var is set. This is because the LLM judge makes
    2 additional vLLM calls per test (response_quality + task_completion),
    doubling the load on the shared Spark2 vLLM server.

    Returns a dict with base_url and model (NOT a Phoenix OpenAIModel).
    The evaluators make direct httpx calls to vLLM with our own rail
    extraction that handles Qwen3 thinking-model output. Phoenix's
    llm_classify can't parse thinking tokens (produces NOT_PARSABLE
    for ~50% of responses).
    """
    # Check opt-in flag
    use_judge = (
        request.config.getoption("--with-judge", default=False)
        or os.getenv("CCA_RUN_JUDGE", "0") == "1"
    )
    if not use_judge:
        log.info("LLM judge disabled (use --with-judge or CCA_RUN_JUDGE=1 to enable)")
        return None

    try:
        import httpx

        resp = httpx.get(f"{VLLM_BASE_URL[:-3]}/health", timeout=5)
        if resp.status_code != 200:
            log.warning("vLLM not healthy at %s — judge disabled", VLLM_BASE_URL)
            return None
    except Exception:
        return None

    log.info("LLM judge enabled — 2 extra vLLM calls per test")
    return {
        "base_url": VLLM_BASE_URL,
        "model": VLLM_MODEL,
    }


# ==================== CCA Client ====================


@pytest.fixture(scope="session")
def cca(phoenix_tracer):
    """CCA AAAM client with Phoenix tracing.

    Session-scoped: one HTTP client for the entire test run.
    Uses streaming with idle timeout — no fixed total timeout.
    """
    _provider, tracer = phoenix_tracer
    client = CCAClient(base_url=CCA_BASE_URL, tracer=tracer, project_name=PROJECT_NAME)
    yield client
    client.close()


@pytest.fixture
def test_run(cca, request):
    """Auto-tracking test context that persists all test data.

    Creates a TestRunContext that wraps CCAClient with resource tracking.
    On completion, writes manifest.json to the reports directory (uploaded
    by CI after_script). Data is NEVER deleted — use the dashboard's
    "Clear One" or "Clear All" to clean up.

    Usage::

        def test_something(test_run):
            session_id = f"test-xxx-{uuid4().hex[:8]}"
            test_run.track_session(session_id)
            r = test_run.chat("Hello", session_id=session_id)
            assert r.content
    """
    from .cca_client import TestRunContext
    # Pass the pytest node so TestRunContext can derive the canonical
    # file-stem name (per tests/_naming.py). Passing request.node.name
    # alone would give us the function name, which violates the standard
    # for multi-function or descriptively-named tests.
    ctx = TestRunContext(cca, request.node)
    yield ctx
    # Determine if the test failed
    failed = hasattr(request.node, "rep_call") and request.node.rep_call.failed
    ctx.finalize(failed=failed)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Store test result on the item node so the test_run fixture can read it."""
    import pytest
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture(scope="session", autouse=True)
def warn_missing_api_key():
    """Warn if CCA_TEST_API_KEY is not set — tests will fail on auth-enabled servers."""
    if not os.getenv("CCA_TEST_API_KEY"):
        import warnings
        warnings.warn(
            "CCA_TEST_API_KEY not set — tests will fail against auth-enabled servers. "
            "Get the key from: docker exec cca python3 -c "
            "\"from ui.models import APIKey; print(APIKey.objects.get(name='ci-tests').key)\"",
            UserWarning,
            stacklevel=1,
        )


@pytest.fixture(scope="session", autouse=True)
def require_cca_healthy(cca):
    """Skip all tests if CCA AAAM server is unreachable OR if workspace
    indexing hasn't finished its first cycle yet.

    Session-scoped: one health check per pytest session, not per
    function. Under the GitLab dispatcher, each pipeline runs one
    file's pytest invocation, so session ≈ file. The dispatcher's
    `health-check` job already gates pipeline-level health; this
    fixture covers local `make test NAME=...` runs that skip CI.

    The workspace_indexing_ready gate prevents `test_code_intelligence`
    (and any other graph-dependent suite) from racing the indexer: a
    cold cca container parses files in Phase 1 but doesn't populate
    CALLS edges until Phase 2 (resolve_project_calls) finishes the
    project-scoped second pass. Tests that ran in the gap saw empty
    callers, e.g. "Connect-SessionVC has 0 callers" when the graph
    actually has 89. Poll 5s × 60 = 5 min ceiling; if still not ready,
    skip rather than fail (so a slow indexer doesn't masquerade as a
    product regression).
    """
    health = cca.health()
    if health.get("status") != "healthy":
        pytest.skip(
            f"CCA AAAM server not healthy: {health.get('error', 'unknown')}"
        )

    if health.get("workspace_indexing_ready"):
        return

    log.info("Workspace indexing not ready yet — polling /health (5s × 60 max)")
    for _ in range(60):
        time.sleep(5)
        health = cca.health()
        if health.get("workspace_indexing_ready"):
            log.info("Workspace indexing ready — proceeding")
            return

    pytest.skip(
        "CCA workspace indexing not ready after 5 min — slow indexer, "
        "not a product regression"
    )


# ==================== Test Helpers ====================


# ==================== Session-Level Cleanup ====================


@pytest.fixture(scope="session", autouse=True)
def session_cleanup(cca):
    """Safety net: log leaked test resources after all tests (no deletions).

    Each test cleans up its own resources via TestResourceTracker.
    This fixture only warns about resources that leaked due to test crashes,
    so developers can fix their test cleanup.
    """
    yield  # All tests run here

    log.info("=== Session cleanup: checking for leaked test resources ===")

    try:
        users_data = cca.list_users()
        test_prefixes = (
            "Onboard_", "Memory_", "CRUD_", "Lifecycle_", "NoteTest_",
            "EditFlow_", "BashTest_", "TestUser_",
            "Planner_", "Coder_", "Infra_", "Recall_", "RouteUser_",
            "InfraTest_",
        )
        stale = [
            u.get("display_name", "")
            for u in users_data.get("users", [])
            if any(
                u.get("display_name", "").startswith(p)
                for p in test_prefixes
            )
        ]
        if stale:
            log.warning("LEAKED test users (not cleaned by tests): %s", stale)
    except Exception as e:
        log.warning("Session cleanup: failed to check users: %s", e)


# ==================== Deferred Experiment + LLM Judge ====================


@pytest.fixture(scope="session", autouse=True)
def deferred_experiment(judge_model, request):
    """Create Phoenix Dataset + Experiment after all tests.

    ALWAYS creates dataset/experiment (code eval scores for all 53 calls).
    If --with-judge: also runs deferred LLM judge on eligible items.
    """
    yield  # All tests run here

    from .evaluators import _EVAL_QUEUE, run_deferred_experiment

    if not _EVAL_QUEUE:
        return

    run_judge = judge_model is not None
    label = "LLM Judge + " if run_judge else ""
    print(f"\n{'=' * 60}")
    print(f"{label}Phoenix Experiment ({len(_EVAL_QUEUE)} responses)")
    print(f"{'=' * 60}")

    results = run_deferred_experiment(run_judge=run_judge)
    request.config._judge_results = results

    failures = [r for r in results if r["label"] in ("poor", "failed")]
    request.config._judge_failures = failures

    if run_judge and results:
        passed = len(results) - len(failures)
        print(f"Judge: {passed} passed, {len(failures)} failed")
    print(f"{'=' * 60}")


def pytest_sessionfinish(session, exitstatus):
    """Report judge results as advisory — never override code assertion outcomes.

    Code evaluators are deterministic and catch real problems. The LLM judge
    is a quality signal (logged, posted to Phoenix, shown in terminal report)
    but should NOT fail the test run when all code assertions passed.

    Rationale: the 35B judge model has variance — one weak rating shouldn't
    override 20+ passing code assertions. Judge failures are surfaced
    prominently in the terminal report and Phoenix annotations for review.
    """
    results = getattr(session.config, "_judge_results", [])
    if not results or exitstatus != 0:
        return

    from collections import defaultdict

    by_test: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_test[r["test_name"]].append(r)

    test_failures = []
    for test_name, evals in by_test.items():
        n_fail = sum(1 for e in evals if e["label"] in ("poor", "failed"))
        if n_fail > len(evals) / 2:
            test_failures.append(test_name)

    # Advisory only — log failures but do NOT change exit status.
    # Code assertions are the ground truth for pass/fail.
    if test_failures:
        log.warning(
            "LLM judge flagged %d test(s) as low quality (advisory): %s",
            len(test_failures),
            ", ".join(test_failures),
        )

    # Update _judge_failures so terminal_summary shows flagged tests
    session.config._judge_failures = [
        r for r in results
        if r["test_name"] in test_failures
        and r["label"] in ("poor", "failed")
    ]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print LLM judge report with per-test/per-turn grouping (advisory)."""
    results = getattr(config, "_judge_results", [])
    if not results:
        return

    failures = getattr(config, "_judge_failures", [])
    terminalreporter.write_sep("=", "LLM Judge Report (advisory — does not affect pass/fail)")

    current_test = None
    for r in results:
        if r["test_name"] != current_test:
            current_test = r["test_name"]
            terminalreporter.write_line(f"\n  {current_test}:")

        is_fail = r["label"] in ("poor", "failed")
        marker = "WARN" if is_fail else "PASS"
        turn = r.get("turn", "?")
        terminalreporter.write_line(
            f"    {marker}  [turn {turn}] {r['name']} = {r['label']}"
        )

    passed = len(results) - len(failures)
    terminalreporter.write_line(
        f"\n  {passed} passed, {len(failures)} flagged "
        f"({len(results)} total)"
    )

    if failures:
        terminalreporter.write_sep("-", "Judge Warnings (review in Phoenix)")
        for f in failures:
            terminalreporter.write_line(
                f"  {f['test_name']} [turn {f.get('turn', '?')}] "
                f":: {f['name']}: {f['label']}"
            )
            terminalreporter.write_line(
                f"    {f.get('explanation', '')[:200]}"
            )
