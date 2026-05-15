"""Shared HTTP + Cypher helpers for integration tests.

Lifted from tests/integration/test_gitnexus_parity.py (the original
in-line copy stays there for backwards-compat; new tests should import
from here). Reads CCA_BASE_URL + the auth keys from env so the same
helpers work locally and inside the cca-tests CI container.

Auth precedence: prefers CCA_ADMIN_API_KEY (required for /admin/cypher
and other destructive endpoints), falls back to CCA_TEST_API_KEY (the
regular API key wired into APIKeyAuthMiddleware via the ui_apikey
table). Tests that need admin-only endpoints will fail loudly if
neither is set.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import httpx


BASE_URL = os.environ.get("CCA_BASE_URL", "https://192.168.4.205:8500")

# Admin key is needed for destructive endpoints (admin/cypher when it
# writes, admin/workspace/reindex with force, etc.). Test key works
# for everything that goes through APIKeyAuthMiddleware.
_ADMIN_KEY = os.environ.get("CCA_ADMIN_API_KEY", "")
_TEST_KEY = os.environ.get("CCA_TEST_API_KEY", "")
_AUTH_KEY = _ADMIN_KEY or _TEST_KEY


def _headers() -> dict:
    """Bearer auth header — prefers admin key, falls back to test key."""
    if not _AUTH_KEY:
        return {}
    return {"Authorization": f"Bearer {_AUTH_KEY}"}


def admin_get(path: str, **params: Any) -> dict:
    """GET an admin endpoint; raise on non-2xx; return JSON."""
    r = httpx.get(
        f"{BASE_URL}{path}", params=params,
        headers=_headers(), verify=False, timeout=30,
    )
    r.raise_for_status()
    return r.json()


def admin_post(path: str, body: dict) -> dict:
    """POST a JSON body to an admin endpoint; raise on non-2xx; return JSON."""
    r = httpx.post(
        f"{BASE_URL}{path}", json=body,
        headers={**_headers(), "Content-Type": "application/json"},
        verify=False, timeout=60,
    )
    r.raise_for_status()
    return r.json()


def cypher(query: str, **params: Any) -> dict:
    """Run a read-only Cypher query via /admin/cypher."""
    return admin_post("/admin/cypher", {"query": query, "params": params})


def cypher_rows(query: str, **params: Any) -> list[dict]:
    """Cypher convenience: return just the rows list (or [] if none)."""
    return cypher(query, **params).get("rows", []) or []


def trigger_reindex(paths: Optional[list[str]] = None, force: bool = False) -> str:
    """POST /workspace/reindex with optional paths + force flag.

    Returns the job_id. Use wait_for_reindex(job_id) to block until done.
    """
    body: dict[str, Any] = {"force": force}
    if paths:
        body["paths"] = paths
    resp = admin_post("/workspace/reindex", body)
    return resp.get("job_id") or resp.get("status_url", "").rstrip("/").rsplit("/", 1)[-1]


def wait_for_reindex(job_id: str, timeout_s: int = 300, poll_s: int = 5) -> dict:
    """Poll /workspace/reindex/<job_id> until status in {completed, failed}.

    Raises TimeoutError if the job doesn't reach terminal state in
    `timeout_s` seconds. Returns the final job dict so the caller can
    inspect `result` for indexer stats.
    """
    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        last = admin_get(f"/workspace/reindex/{job_id}")
        status = last.get("status", "?")
        if status in ("completed", "failed", "error"):
            return last
        time.sleep(poll_s)
    raise TimeoutError(
        f"reindex job {job_id} didn't finish within {timeout_s}s "
        f"(last status: {last.get('status')!r})"
    )


def delete_project(project: str) -> None:
    """Cypher cleanup — DETACH DELETE every node tagged with `project`.

    Tests use this in teardown to clean up synthetic projects without
    affecting EVA or other real workspaces. Safe to call multiple times.
    """
    cypher(
        "MATCH (n {project: $proj}) DETACH DELETE n",
        proj=project,
    )
