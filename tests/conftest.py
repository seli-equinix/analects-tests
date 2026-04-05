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
CCA_BASE_URL = os.getenv("CCA_BASE_URL", "http://192.168.4.205:8500")

# Same Phoenix project as the CCA server — test + server spans unified
PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "cca-http")

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


@pytest.fixture(autouse=True)
def trace_test(request, phoenix_tracer):
    """Wrap every test in a Phoenix span with test metadata.

    Creates a root span like 'websearch::test_basic_search' so tests
    are easy to find and filter in the Phoenix UI.

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

    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("cca.test.name", test_name)
        span.set_attribute("cca.test.category", category)
        span.set_attribute("cca.test.nodeid", test_path)

        # Evaluators append annotation dicts here during the test
        span._pending_annotations = []
        # CCA client stashes metrics here for per-test reporting
        span._test_metrics = {}

        yield span

        # Set I/O attributes from accumulated chat turns (done after yield so
        # ALL turns are collected before we write).
        # Single-turn: input.value / output.value as plain text (CHAIN span).
        # Multi-turn: llm.input_messages / llm.output_messages so Phoenix
        # renders each turn as its own independent message block (LLM span).
        turns = span._test_metrics.get("_turns", [])
        if turns:
            if len(turns) == 1:
                span.set_attribute("input.value", turns[0][0])
                span.set_attribute("output.value", turns[0][1])
            else:
                # Switch span kind to LLM so Phoenix uses the chat/conversation
                # renderer — each message becomes an independent block.
                span.set_attribute("openinference.span.kind", "LLM")
                idx = 0
                for i, (msg, resp) in enumerate(turns):
                    span.set_attribute(
                        f"llm.input_messages.{idx}.message.role", "user"
                    )
                    span.set_attribute(
                        f"llm.input_messages.{idx}.message.content",
                        f"[Turn {i + 1}] {msg}",
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
                span.set_attribute(
                    "llm.output_messages.0.message.role", "assistant"
                )
                span.set_attribute(
                    "llm.output_messages.0.message.content",
                    f"[Turn {len(turns)}] {turns[-1][1]}",
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

    # Per-test summary line for real-time monitoring
    metrics = getattr(span, "_test_metrics", {})
    if metrics:
        outcome = "?"
        if hasattr(request.node, "rep_call"):
            outcome = request.node.rep_call.outcome.upper()
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
            f"\n  >> {span_name}: {outcome} | "
            f"route={route} steps={steps} iters={iters} "
            f"{elapsed_s:.1f}s{flag_str}",
            flush=True,
        )

    # Generate .md test report for Claude Code review
    if metrics:
        try:
            from tests.report_generator import generate_test_report

            outcome_str = "PASSED"
            failure_reason = ""
            if hasattr(request.node, "rep_call"):
                outcome_str = request.node.rep_call.outcome.upper()
                if request.node.rep_call.failed:
                    # Capture the actual assertion error / traceback
                    failure_reason = str(request.node.rep_call.longrepr)[:2000]

            test_dir = generate_test_report(
                test_name=span_name.replace("::", "-"),
                status=outcome_str,
                turns=metrics.get("_turns", []),
                metrics=metrics,
                evals_per_turn=metrics.get("_evals"),
                turn_details=metrics.get("_turn_details"),
                failure_reason=failure_reason,
            )

            # Capture CCA debug logs into the test's folder
            if test_dir:
                try:
                    import httpx as _httpx
                    cca_url = os.environ.get(
                        "CCA_BASE_URL", "http://192.168.4.205:8500"
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
    ctx = TestRunContext(cca, request.node.name)
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


@pytest.fixture(autouse=True)
def require_cca_healthy(cca):
    """Skip all tests if CCA AAAM server is unreachable."""
    health = cca.health()
    if health.get("status") != "healthy":
        pytest.skip(
            f"CCA AAAM server not healthy: {health.get('error', 'unknown')}"
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
