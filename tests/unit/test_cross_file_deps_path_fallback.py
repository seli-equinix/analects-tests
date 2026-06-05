"""get_cross_file_deps / get_file_functions path-suffix fallback (Slice 2).

Regression guard for test_code_intelligence Turn 3 ("DEPENDENCY DATA INTEGRITY
ISSUE: equinix.automation.vcenter.psm1 should have many dependents"). The graph
DID have 27+ cross-file dependents, but `get_cross_file_deps` matched the File
by EXACT `path`, while the model passed the BASENAME (the user said "the
equinix.automation.vcenter.psm1 file"). Exact match on a basename → 0 rows →
"no dependencies" → test fail.

The fix tries the exact `File {path: $path}` first (unchanged for full-path
callers) and, only when that finds nothing, falls back to a path-suffix match
(`ENDS WITH '/' + basename`) so a basename / partial / relative path still
resolves.

Mocks the AsyncSession boundary (no live Memgraph), mirroring
test_resolve_edge_count_consistency.py.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from confucius.server.code_intelligence.memgraph_client import MemgraphClient


def _result(data):
    r = MagicMock()
    r.data = AsyncMock(return_value=data)
    return r


def _client(run_results):
    """MemgraphClient whose session.run yields the given results in order."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.run = AsyncMock(side_effect=run_results)
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    return MemgraphClient(driver), session


# ── get_cross_file_deps ───────────────────────────────────────────────


class TestCrossFileDepsFallback:
    def test_exact_match_hits_no_fallback(self):
        # Full path passed → exact match returns rows → fallback NOT used.
        rows = [{"dependent_file": "a.ps1", "function_count": 3,
                 "sample_functions": ["x"]}]
        client, session = _client([_result(rows)])
        res = asyncio.run(
            client.get_cross_file_deps("/workspace/EVA/code/x.psm1")
        )
        assert res == rows
        assert session.run.await_count == 1, "exact hit must not run fallback"

    def test_basename_falls_back_to_suffix(self):
        # Basename passed → exact match empty → suffix fallback returns rows.
        rows = [{"dependent_file": "VmOrchestrationAPI.psm1",
                 "function_count": 3, "sample_functions": ["a", "b"]}]
        client, session = _client([_result([]), _result(rows)])
        res = asyncio.run(
            client.get_cross_file_deps("equinix.automation.vcenter.psm1")
        )
        assert res == rows, "basename must resolve via the suffix fallback"
        assert session.run.await_count == 2
        # Second query is the suffix fallback, keyed on '/' + basename.
        second = session.run.await_args_list[1]
        assert second.kwargs.get("suffix") == "/equinix.automation.vcenter.psm1"
        assert "ENDS WITH" in second.args[0]

    def test_genuinely_no_deps_returns_empty(self):
        # Exact empty AND suffix empty → empty (leaf file, no dependents).
        client, session = _client([_result([]), _result([])])
        res = asyncio.run(client.get_cross_file_deps("leaf.py"))
        assert res == []
        assert session.run.await_count == 2


# ── get_file_functions ────────────────────────────────────────────────


class TestFileFunctionsFallback:
    def test_exact_match_hits_no_fallback(self):
        rows = [{"name": "add", "qualified_name": "x.psm1::add"}]
        client, session = _client([_result(rows)])
        res = asyncio.run(
            client.get_file_functions("/workspace/EVA/code/x.psm1")
        )
        assert res == rows
        assert session.run.await_count == 1

    def test_basename_falls_back_to_suffix(self):
        rows = [{"name": "Connect-SessionVC",
                 "qualified_name": "equinix.automation.vcenter.psm1::Connect-SessionVC"}]
        client, session = _client([_result([]), _result(rows)])
        res = asyncio.run(
            client.get_file_functions("equinix.automation.vcenter.psm1")
        )
        assert res == rows
        assert session.run.await_count == 2
        second = session.run.await_args_list[1]
        assert second.kwargs.get("suffix") == "/equinix.automation.vcenter.psm1"
        assert "ENDS WITH" in second.args[0]
