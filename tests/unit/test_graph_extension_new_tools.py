"""LLM-tool handlers for the new Route + WRAPS surfaces.

Covers `_handle_find_routes` and `_handle_get_decorator_chain` on the
GraphToolsExtension. Both tools surface indexer data (Route nodes,
WRAPS edges, Function.decorators) that previously had no agent-callable
consumer — the agent had to fall back to text search via
`search_codebase` for HTTP-route or decorator-stack questions.

Tests mock at the AsyncSession boundary so we don't need a live
Memgraph; the tool handlers are thin Cypher-orchestration wrappers,
so the assertions focus on:
  - input-shape handling (project / method / framework / path_pattern
    filter combinations; bare-name vs. qualified_name lookup)
  - WHERE clause construction (correct field-name matches, case
    normalization, optional filters absent vs. present)
  - JSON output shape (the agent / dashboard parse this)
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

# `graph_extension` transitively pulls in the full agent runtime
# (langchain_core, plotly, the orchestrator base classes, etc.) — all
# present inside the cca container's image but not on node5's local
# venv. Skip the file with a clear message when those aren't available
# rather than playing dependency-stub whack-a-mole. CI runs unit tests
# inside the cca-tests container where these deps are installed, so
# coverage is preserved.
pytest.importorskip(
    "langchain_core",
    reason="graph_extension's runtime deps (langchain_core, plotly, …) "
           "are only installed inside the cca container; this test file "
           "runs in the container-based CI bucket.",
)
pytest.importorskip("plotly", reason="see langchain_core skip reason above")

from confucius.server.code_intelligence.graph_extension import (  # noqa: E402
    GraphToolsExtension,
)


def _make_session(query_results: list) -> tuple[MagicMock, list]:
    """Build a session mock that returns the given results in order
    (one per `await session.run(...)` call). Each result is a list of
    dicts that .data() awaits to.

    Returns the session and a `calls` list capturing every (query,
    kwargs) so tests can assert WHERE-clause construction.
    """
    calls: list = []
    results = []
    for rows in query_results:
        result_obj = MagicMock()
        result_obj.data = AsyncMock(return_value=rows)
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
            return r

    session.run = capture_run
    return session, calls


def _make_extension(session: MagicMock) -> GraphToolsExtension:
    """Build a GraphToolsExtension with mocked driver/graph."""
    backend = MagicMock()
    backend.memgraph = MagicMock()
    ext = GraphToolsExtension(backend)

    graph = MagicMock()
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    graph._driver = driver
    ext._cached_graph = graph
    return ext


# ── find_routes ──────────────────────────────────────────────────────


class TestFindRoutes:
    @pytest.mark.asyncio
    async def test_no_filters_returns_all_routes(self):
        rows = [
            {"path": "/", "http_method": "GET", "framework": "fastapi",
             "project": "fastapi_user_app", "file_path": "/workspace/main.py"},
            {"path": "/users", "http_method": "POST", "framework": "fastapi",
             "project": "fastapi_user_app", "file_path": "/workspace/main.py"},
        ]
        session, calls = _make_session([rows])
        ext = _make_extension(session)

        out = await ext._handle_find_routes({})
        result = json.loads(out)
        assert result["count"] == 2
        assert result["routes"] == rows
        # No WHERE clause when no filters supplied (just `MATCH ... LIMIT`).
        assert "WHERE" not in calls[0][0]

    @pytest.mark.asyncio
    async def test_http_method_normalized_to_upper(self):
        """Lowercase 'post' or 'Post' → 'POST'. The agent's free-text
        input shouldn't have to match the indexer's storage casing."""
        session, calls = _make_session([[]])
        ext = _make_extension(session)

        await ext._handle_find_routes({"http_method": "post"})
        # Param value is upper-cased before the query
        assert calls[0][1]["http_method"] == "POST"
        assert "toUpper(r.http_method) = $http_method" in calls[0][0]

    @pytest.mark.asyncio
    async def test_framework_normalized_to_lower(self):
        session, calls = _make_session([[]])
        ext = _make_extension(session)

        await ext._handle_find_routes({"framework": "FASTAPI"})
        assert calls[0][1]["framework"] == "fastapi"
        assert "toLower(r.framework) = $framework" in calls[0][0]

    @pytest.mark.asyncio
    async def test_path_pattern_uses_contains(self):
        session, calls = _make_session([[]])
        ext = _make_extension(session)

        await ext._handle_find_routes({"path_pattern": "/api/v1"})
        assert calls[0][1]["path_pattern"] == "/api/v1"
        assert "r.path CONTAINS $path_pattern" in calls[0][0]

    @pytest.mark.asyncio
    async def test_combined_filters_AND_in_where(self):
        """All filters present → AND clause includes all four."""
        session, calls = _make_session([[]])
        ext = _make_extension(session)

        await ext._handle_find_routes({
            "project": "EVA",
            "http_method": "GET",
            "framework": "flask",
            "path_pattern": "/api",
        })
        query = calls[0][0]
        assert "r.project = $project" in query
        assert "toUpper(r.http_method) = $http_method" in query
        assert "toLower(r.framework) = $framework" in query
        assert "r.path CONTAINS $path_pattern" in query
        # ANDed, not ORed
        assert query.count(" AND ") == 3

    @pytest.mark.asyncio
    async def test_limit_clamped_to_safe_range(self):
        session, calls = _make_session([[]])
        ext = _make_extension(session)

        # Negative / zero / over-cap → fall back to default 50.
        await ext._handle_find_routes({"limit": -1})
        assert calls[0][1]["limit"] == 50
        await ext._handle_find_routes({"limit": 0})
        assert calls[1][1]["limit"] == 50
        await ext._handle_find_routes({"limit": 9999})
        assert calls[2][1]["limit"] == 50
        # Valid in-range value passes through.
        await ext._handle_find_routes({"limit": 25})
        assert calls[3][1]["limit"] == 25

    @pytest.mark.asyncio
    async def test_no_driver_returns_error_json(self):
        ext = GraphToolsExtension(MagicMock())
        ext._cached_graph = MagicMock()
        ext._cached_graph._driver = None
        out = await ext._handle_find_routes({"project": "EVA"})
        assert json.loads(out) == {"error": "Memgraph not available"}


# ── get_decorator_chain ──────────────────────────────────────────────


class TestGetDecoratorChain:
    @pytest.mark.asyncio
    async def test_returns_wrappers_wraps_and_target_decorators(self):
        # Three queries fired in order: incoming WRAPS, outgoing WRAPS,
        # target metadata.
        wrappers = [
            {"name": "timed_api", "qname": "/foo.py::timed_api",
             "file_path": "/foo.py"},
        ]
        wraps = []
        target_data = [
            {"name": "bar", "qname": "/bar.py::bar",
             "file_path": "/bar.py",
             "decorators": ["@timed_api", "@login_required"]},
        ]
        session, calls = _make_session([wrappers, wraps, target_data])
        ext = _make_extension(session)

        out = await ext._handle_get_decorator_chain({
            "qualified_name": "/bar.py::bar",
        })
        result = json.loads(out)
        assert result["wrappers"] == wrappers
        assert result["wrappers_count"] == 1
        assert result["wraps"] == []
        assert result["wraps_count"] == 0
        assert result["target"]["decorators"] == ["@timed_api", "@login_required"]

    @pytest.mark.asyncio
    async def test_qualified_name_takes_precedence_over_name(self):
        """When both qualified_name AND name are passed, the query uses
        qualified_name (more specific) — bare-name fallback is for the
        case where the agent doesn't know the qualified form."""
        session, calls = _make_session([[], [], []])
        ext = _make_extension(session)

        await ext._handle_get_decorator_chain({
            "qualified_name": "/foo.py::bar",
            "name": "bar",
        })
        # All three queries match by qualified_name when it's set.
        for query, _kwargs in calls:
            assert "{qualified_name: $qname}" in query
            assert "{name: $name}" not in query

    @pytest.mark.asyncio
    async def test_bare_name_fallback_when_no_qname(self):
        session, calls = _make_session([[], [], []])
        ext = _make_extension(session)

        await ext._handle_get_decorator_chain({"name": "bar"})
        for query, _kwargs in calls:
            assert "{name: $name}" in query
            assert "{qualified_name: $qname}" not in query

    @pytest.mark.asyncio
    async def test_depth_clamped_to_safe_range(self):
        session, calls = _make_session([[], [], []])
        ext = _make_extension(session)

        # Out-of-range → default 5
        await ext._handle_get_decorator_chain(
            {"name": "bar", "depth": 0},
        )
        assert "WRAPS*1..5" in calls[0][0]
        # Valid in-range → passed through
        session2, calls2 = _make_session([[], [], []])
        ext2 = _make_extension(session2)
        await ext2._handle_get_decorator_chain(
            {"name": "bar", "depth": 3},
        )
        assert "WRAPS*1..3" in calls2[0][0]

    @pytest.mark.asyncio
    async def test_missing_both_inputs_returns_error(self):
        session, _ = _make_session([])
        ext = _make_extension(session)
        out = await ext._handle_get_decorator_chain({})
        assert json.loads(out) == {
            "error": "qualified_name or name is required",
        }

    @pytest.mark.asyncio
    async def test_project_filter_added_to_all_three_queries(self):
        session, calls = _make_session([[], [], []])
        ext = _make_extension(session)

        await ext._handle_get_decorator_chain({
            "name": "bar", "project": "EVA",
        })
        # All three queries should filter by project.
        for query, kwargs in calls:
            assert "target.project = $project" in query
            assert kwargs["project"] == "EVA"

    @pytest.mark.asyncio
    async def test_no_driver_returns_error_json(self):
        ext = GraphToolsExtension(MagicMock())
        ext._cached_graph = MagicMock()
        ext._cached_graph._driver = None
        out = await ext._handle_get_decorator_chain({"name": "bar"})
        assert json.loads(out) == {"error": "Memgraph not available"}
