"""analyze_impact(file_path=…) returns ANY function in the file, not just __module__.

Background — 2026-05-22 fix for the code-intelligence test regression.
Pre-fix, `_handle_analyze_impact` resolved `file_path` → `<file>::__module__`
and queried callers of THAT single node. For library-style files (PowerShell
.psm1, Python module-only files with no `__main__` block), the `__module__`
synthetic has 0 callers semantically — so the tool returned empty results
even when the file has many cross-file dependents on its function defs.

The fix: when `file_path` is provided without `qualified_name`, match
`(sym:Function)` with a `WHERE sym.file_path = $file_path` clause so the
upstream/downstream walk captures every external caller of every function
in the file. Strict superset of the prior behaviour for files where
`__module__` does have callers (Python scripts with `__main__`).

Mocks at the AsyncSession boundary — same pattern as
test_graph_extension_new_tools.py. Skips on node5 where the graph
extension's runtime deps aren't installed.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

# Same skip pattern as test_graph_extension_new_tools.py — the graph
# extension transitively imports langchain_core, plotly, the orchestrator
# base classes, which aren't on node5's venv. CI runs unit tests inside
# the cca container where these deps are installed.
pytest.importorskip(
    "langchain_core",
    reason="graph_extension's runtime deps are only installed inside the "
           "cca container; this test runs in the container-based CI bucket.",
)
pytest.importorskip("plotly", reason="see langchain_core skip reason above")

from confucius.server.code_intelligence.graph_extension import (  # noqa: E402
    GraphToolsExtension,
)


def _make_session(query_results: list) -> tuple[MagicMock, list]:
    """Build a mock AsyncSession that returns the given rows in order,
    one list of rows per `await session.run(...)` call.

    Returns (session_mock, calls) where `calls` captures every
    (query_string, kwargs) so tests can assert Cypher shape + params.
    """
    calls: list = []
    results = []
    for rows in query_results:
        result_obj = MagicMock()
        result_obj.data = AsyncMock(return_value=rows)
        # `await res.single()` returns the first row (or None); some
        # handlers use it. Provide a sensible default.
        result_obj.single = AsyncMock(return_value=(rows[0] if rows else None))
        results.append(result_obj)

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    iter_results = iter(results)

    async def capture_run(query, **kwargs):
        calls.append((query, kwargs))
        try:
            return next(iter_results)
        except StopIteration:
            r = MagicMock()
            r.data = AsyncMock(return_value=[])
            r.single = AsyncMock(return_value=None)
            return r

    session.run = capture_run
    return session, calls


def _make_extension(session: MagicMock) -> GraphToolsExtension:
    """Build a GraphToolsExtension with a mocked driver/graph."""
    backend = MagicMock()
    backend.memgraph = MagicMock()
    ext = GraphToolsExtension(backend)

    graph = MagicMock()
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    graph._driver = driver
    ext._cached_graph = graph
    return ext


# Synthetic upstream row matching the live tool's RETURN shape.
def _row(qname: str, name: str, file_path: str, hops: int = 1) -> dict:
    return {
        "name": name,
        "qname": qname,
        "file_path": file_path,
        "language": "powershell",
        "is_module": False,
        "hops": hops,
        "edge_types": ["CALLS"],
        "edge_methods": ["direct"],
        "from_qname": qname,
        "to_qname": f"/workspace/EVA/code/lib.psm1::Open-DB",
    }


# ── Fix 1: file_path mode broadens to ANY function in the file ──


class TestAnalyzeImpactFilePathMode:
    """file_path WITHOUT qualified_name → matches every :Function in the file."""

    @pytest.mark.asyncio
    async def test_file_path_query_uses_file_path_where_clause(self):
        """Cypher must constrain `sym.file_path = $file_path` instead of
        narrowing to `<file>::__module__`."""
        # Two queries fire in "both" direction (upstream + downstream).
        # Return synthetic rows so the JSON-shape assertions later work.
        rows_upstream = [
            _row(qname=f"/workspace/EVA/code/caller{i}.ps1::caller_fn{i}",
                 name=f"caller_fn{i}",
                 file_path=f"/workspace/EVA/code/caller{i}.ps1")
            for i in range(3)
        ]
        rows_downstream = []
        session, calls = _make_session([rows_upstream, rows_downstream])
        ext = _make_extension(session)

        out = await ext._handle_analyze_impact({
            "file_path": "/workspace/EVA/code/lib.psm1",
            "direction": "both",
            "depth": 3,
        })
        result = json.loads(out)

        assert "error" not in result, f"got error: {result}"
        assert len(calls) == 2, "expected 2 Cypher calls (upstream + downstream)"

        upstream_query, upstream_params = calls[0]
        # The new Cypher must match ANY :Function with the file_path filter:
        assert ":Function" in upstream_query, (
            f"upstream query should match label :Function, got:\n{upstream_query}"
        )
        assert "sym.file_path = $file_path" in upstream_query, (
            f"upstream query must filter by file_path, got:\n{upstream_query}"
        )
        # And must NOT resolve to ::__module__ in this mode:
        assert "::__module__" not in str(upstream_params), (
            f"file_path mode must NOT inject ::__module__; got params: "
            f"{upstream_params}"
        )
        assert upstream_params.get("file_path") == "/workspace/EVA/code/lib.psm1"

    @pytest.mark.asyncio
    async def test_file_path_returns_callers_when_module_has_zero(self):
        """Repro of P21861 — .psm1 with 0 __module__ callers BUT 3+
        external callers of its named function defs. Pre-fix: empty
        result. Post-fix: 3 upstream rows."""
        rows_upstream = [
            _row(qname="/workspace/EVA/code/job1.ps1::Step1",
                 name="Step1", file_path="/workspace/EVA/code/job1.ps1"),
            _row(qname="/workspace/EVA/code/job2.ps1::Step2",
                 name="Step2", file_path="/workspace/EVA/code/job2.ps1"),
            _row(qname="/workspace/EVA/code/job3.ps1::Step3",
                 name="Step3", file_path="/workspace/EVA/code/job3.ps1"),
        ]
        session, _ = _make_session([rows_upstream, []])
        ext = _make_extension(session)

        out = await ext._handle_analyze_impact({
            "file_path": "/workspace/EVA/code/lib.psm1",
            "direction": "upstream",
        })
        result = json.loads(out)
        assert result["upstream_count"] >= 1, (
            f"file_path mode must surface callers when functions in the "
            f"file have external callers; got: {result}"
        )

    @pytest.mark.asyncio
    async def test_file_path_with_explicit_qualified_name_uses_qname(self):
        """Precedence: qualified_name > file_path. Explicit qualified_name
        wins, file_path is ignored."""
        session, calls = _make_session([[], []])
        ext = _make_extension(session)

        await ext._handle_analyze_impact({
            "qualified_name": "/workspace/EVA/code/lib.psm1::Open-DB",
            "file_path": "/workspace/EVA/code/lib.psm1",  # should be ignored
            "direction": "both",
        })

        upstream_query, upstream_params = calls[0]
        # Explicit qualified_name path: should NOT inject the file_path
        # WHERE clause.
        assert "sym.file_path = $file_path" not in upstream_query, (
            f"qualified_name mode must NOT use file_path filter; got query:\n"
            f"{upstream_query}"
        )
        # qname param carries the explicit qualified_name unchanged:
        assert upstream_params["qname"] == "/workspace/EVA/code/lib.psm1::Open-DB"


# ── Regression: existing qualified_name + name modes unchanged ──


class TestAnalyzeImpactBackwardCompat:
    @pytest.mark.asyncio
    async def test_qualified_name_mode_uses_qname_match(self):
        """qualified_name mode unchanged: matches `{qualified_name: $qname}`."""
        # _handle_analyze_impact ALSO does a "sym qname lookup" round-trip
        # when the input had no qualified_name. With qualified_name supplied,
        # the lookup is skipped — only the 2 main queries fire.
        session, calls = _make_session([[], []])
        ext = _make_extension(session)

        await ext._handle_analyze_impact({
            "qualified_name": "/workspace/EVA/code/lib.psm1::Open-DB",
            "direction": "both",
        })

        upstream_query = calls[0][0]
        assert "{qualified_name: $qname}" in upstream_query, (
            f"qualified_name mode should match by qname property; got:\n"
            f"{upstream_query}"
        )
        assert ":Function" not in upstream_query.split("->(sym")[1].split(")")[0], (
            "qualified_name mode should NOT use :Function label-only match"
        )

    @pytest.mark.asyncio
    async def test_name_mode_uses_name_match(self):
        """name mode unchanged: matches `{name: $name}` with sym-qname lookup."""
        # When neither qualified_name nor file_path is given, the handler
        # first does a lookup query to resolve the sym's qname (so the
        # forward-DAG filter has a uniqueness key), then the main 2.
        # Three total queries fire when direction=both.
        session, calls = _make_session([
            [{"qname": "/workspace/x.py::Open-DB"}],  # sym-qname lookup result
            [],  # upstream
            [],  # downstream
        ])
        ext = _make_extension(session)

        await ext._handle_analyze_impact({
            "name": "Open-DB",
            "direction": "both",
        })

        # The first call is the sym-qname lookup ("MATCH (sym {name: $name})").
        assert "{name: $name}" in calls[0][0] or "{name:$name}" in calls[0][0]
        # The main upstream/downstream queries use {name: $name} too.
        assert "{name: $name}" in calls[1][0]

    @pytest.mark.asyncio
    async def test_missing_all_inputs_returns_error(self):
        """No qualified_name, no file_path, no name → error JSON."""
        session, _ = _make_session([])
        ext = _make_extension(session)

        out = await ext._handle_analyze_impact({})
        result = json.loads(out)
        assert "error" in result
        assert "required" in result["error"].lower()


# ── Param wiring sanity ──


class TestAnalyzeImpactParamWiring:
    @pytest.mark.asyncio
    async def test_file_path_param_injected_in_upstream_and_downstream(self):
        """Both upstream and downstream queries get file_path param when
        match_by_file is True."""
        session, calls = _make_session([[], []])
        ext = _make_extension(session)

        await ext._handle_analyze_impact({
            "file_path": "/workspace/EVA/code/lib.psm1",
            "direction": "both",
        })

        for query, kwargs in calls:
            assert kwargs.get("file_path") == "/workspace/EVA/code/lib.psm1", (
                f"file_path param must be present in every query; got: {kwargs}"
            )

    @pytest.mark.asyncio
    async def test_file_path_param_NOT_injected_for_qname_mode(self):
        """qualified_name mode must not leak a $file_path param (it'd
        be unused but defensive: catch accidental param bloat)."""
        session, calls = _make_session([[], []])
        ext = _make_extension(session)

        await ext._handle_analyze_impact({
            "qualified_name": "/workspace/x.py::foo",
            "direction": "both",
        })

        for query, kwargs in calls:
            assert "file_path" not in kwargs, (
                f"qualified_name mode must not pass file_path param; got: "
                f"{kwargs}"
            )
