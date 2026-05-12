"""Contract-test conftest: pure-Python tests that don't need the live CCA server.

Overrides the parent conftest's autouse `require_cca_healthy` fixture so
contract tests (schema drift, OpenAPI shape, etc.) can run without a
running cca container — they only need the source tree.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def require_cca_healthy():
    """No-op override of the parent autouse fixture for contract tests."""
    yield


@pytest.fixture(autouse=True)
def trace_test(request):
    """No-op override of the parent's Phoenix-wrapping fixture, BUT keep
    the per-test-result POST so the dashboard updates from per-test
    pipelines. See tests/unit/conftest.py for the same rationale.
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
    node_id = f"other::{request.node.name}".replace("::", "-")
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
        pass
