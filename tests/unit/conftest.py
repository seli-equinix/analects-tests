"""Unit-test conftest: pure-Python tests that don't need the live CCA server.

Overrides the parent conftest's autouse `require_cca_healthy` fixture so
tests in this directory can run without a running cca container. Unit
tests should never hit the network — anything that does belongs under
tests/integration/, tests/coder/, etc.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def require_cca_healthy():
    """No-op override of the parent autouse fixture for unit tests."""
    yield


@pytest.fixture(autouse=True)
def trace_test(request):
    """No-op override of the parent's Phoenix-wrapping fixture, BUT keep
    the per-test-result POST so the dashboard updates from per-test
    pipelines.

    The parent's `trace_test` does `provider.force_flush()` + `time.sleep(1)`
    after every test plus the gRPC export round-trip — ~4s per test that
    unit tests don't need. But it ALSO posts to /admin/test-result, which
    is what updates ui_testdefinition.last_status — without that post the
    dashboard stays stuck even after a green per-test run.

    This override drops the Phoenix bits, keeps the result POST.
    """
    yield

    import os, httpx
    _pipeline_id = os.environ.get("CI_PIPELINE_ID", "")
    _ci_job = os.environ.get("RUN_TEST", "")
    if not (_pipeline_id and _ci_job):
        return

    outcome_str = "PASSED"
    failure_reason = ""
    if hasattr(request.node, "rep_call"):
        outcome_str = request.node.rep_call.outcome.upper()
        if request.node.rep_call.failed:
            failure_reason = str(request.node.rep_call.longrepr)[:2000]
    else:
        outcome_str = "ERROR"

    cca_url = os.environ.get("CCA_BASE_URL", "https://192.168.4.205:8500")
    api_key = os.environ.get("CCA_TEST_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    # node_id = canonical file-stem name (per tests/_naming.py + the doc).
    # Pre-fix: this used f"other::{request.node.name}" which made unit-test
    # TestResult rows un-joinable to TestDefinition (which keys by file-stem).
    from tests._naming import canonical_name
    node_id = canonical_name(request.node)
    try:
        httpx.post(
            f"{cca_url}/admin/test-result",
            json={
                "ci_job_name": _ci_job,
                "pipeline_id": _pipeline_id,
                "node_id": node_id,
                "status": outcome_str,
                "duration_ms": 0,
                "route": "",
                "failure_reason": failure_reason[:500],
                "tool_iterations": 0,
                "turns": 0,
            },
            headers=headers,
            timeout=10,
            verify=False,
        )
    except Exception:
        pass  # Don't fail the test on dashboard-post errors
