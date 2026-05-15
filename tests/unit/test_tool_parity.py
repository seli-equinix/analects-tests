"""Tool parity — verify the LLM has the same file-scope access the dashboard does.

Stage 8 of the file-level participation work surfaces `method` (call kind:
direct / subprocess / dot_source / …) and `is_module` (file-as-process
anchor flag) on every CALLS-traversing agent tool, accepts `file_path`
on tools that previously only took bare names, and adds two new
parity tools (list_groups, get_workspace_file_status) that mirror the
dashboard's metadata tabs.

Tests are schema-level (the LLM-visible input_schema must declare the
new fields) and handler-level (mocked Memgraph / sqlite so we don't
need a live backend). No live cca instance required — runs in the
unit-tests bucket on every push.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytest.importorskip(
    "langchain_core",
    reason="graph_extension's runtime deps (langchain_core, plotly, …) "
           "are only installed inside the cca container; this test file "
           "runs in the container-based CI bucket.",
)
pytest.importorskip("plotly", reason="see langchain_core skip reason above")


# ──────────────────────────────────────────────────────────────────────
# Helpers — shared session mock pattern (mirrors test_graph_extension_new_tools)
# ──────────────────────────────────────────────────────────────────────


def _make_session(query_results: list) -> tuple[MagicMock, list]:
    calls: list = []
    results = []
    for rows in query_results:
        result_obj = MagicMock()
        result_obj.data = AsyncMock(return_value=rows)
        result_obj.single = AsyncMock(
            return_value=(rows[0] if rows else None),
        )
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


# ──────────────────────────────────────────────────────────────────────
# Schema-level parity — input_schemas must declare the new fields
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_analyze_impact_accepts_qualified_name_and_file_path():
    """Stage 8c: analyze_impact must take qualified_name or file_path."""
    from confucius.server.code_intelligence.graph_extension import (
        GraphToolsExtension,
    )
    backend = MagicMock()
    backend.memgraph = MagicMock()
    ext = GraphToolsExtension(backend)
    tools = await ext.tools
    schema = next(t for t in tools if t.name == "analyze_impact").input_schema
    props = schema["properties"]
    assert "qualified_name" in props
    assert "file_path" in props
    assert "name" in props
    assert schema["required"] == []


@pytest.mark.asyncio
async def test_get_decorator_chain_accepts_file_path():
    """Stage 8d: get_decorator_chain must accept file_path."""
    from confucius.server.code_intelligence.graph_extension import (
        GraphToolsExtension,
    )
    backend = MagicMock()
    backend.memgraph = MagicMock()
    ext = GraphToolsExtension(backend)
    tools = await ext.tools
    schema = next(t for t in tools if t.name == "get_decorator_chain").input_schema
    assert "file_path" in schema["properties"]


@pytest.mark.asyncio
async def test_list_rules_accepts_file_path():
    """Stage 8f: list_rules must accept file_path."""
    from confucius.server.code_intelligence.rules_extension import (
        RulesToolsExtension,
    )
    backend = MagicMock()
    ext = RulesToolsExtension(backend, user_id="test")
    tools = await ext.tools
    schema = next(t for t in tools if t.name == "list_rules").input_schema
    assert "file_path" in schema["properties"]


@pytest.mark.asyncio
async def test_search_documents_accepts_file_path():
    """Stage 8g: search_documents must accept file_path."""
    from confucius.server.code_intelligence.document_extension import (
        DocumentToolsExtension,
    )
    backend = MagicMock()
    ext = DocumentToolsExtension(
        backend_clients=backend, session_id="t", user_id="u",
    )
    tools = await ext.tools
    schema = next(t for t in tools if t.name == "search_documents").input_schema
    assert "file_path" in schema["properties"]


@pytest.mark.asyncio
async def test_metadata_extension_exposes_two_new_tools():
    """Stage 8h/8i: list_groups + get_workspace_file_status exist."""
    from confucius.server.code_intelligence.metadata_extension import (
        MetadataToolsExtension,
    )
    backend = MagicMock()
    ext = MetadataToolsExtension(backend_clients=backend)
    tools = await ext.tools
    names = {t.name for t in tools}
    assert names == {"list_groups", "get_workspace_file_status"}

    list_groups_schema = next(
        t for t in tools if t.name == "list_groups"
    ).input_schema
    assert "file_path" in list_groups_schema["properties"]
    assert list_groups_schema.get("required", []) == []

    status_schema = next(
        t for t in tools if t.name == "get_workspace_file_status"
    ).input_schema
    assert "file_path" in status_schema["properties"]
    assert status_schema.get("required", []) == ["file_path"]


# ──────────────────────────────────────────────────────────────────────
# Handler-level parity — method + is_module surface on returned JSON
# ──────────────────────────────────────────────────────────────────────


def _make_graph_extension(session: MagicMock):
    from confucius.server.code_intelligence.graph_extension import (
        GraphToolsExtension,
    )
    backend = MagicMock()
    backend.memgraph = MagicMock()
    ext = GraphToolsExtension(backend)
    graph = MagicMock()
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    graph._driver = driver
    ext._cached_graph = graph
    return ext


@pytest.mark.asyncio
async def test_analyze_impact_resolves_file_path_to_module_entry():
    """When file_path is passed, the handler builds qualified_name as
    '<file_path>::__module__' so the Cypher matches the synthetic
    module-entry node."""
    rows = [
        {
            "name": "step_a.py::__module__",
            "qname": "/workspace/orch/step_a.py::__module__",
            "file_path": "/workspace/orch/step_a.py",
            "depth": 1,
            "edge_methods": ["subprocess"],
            "is_module": True,
        },
    ]
    session, calls = _make_session([rows, []])
    ext = _make_graph_extension(session)

    out = await ext._handle_analyze_impact({
        "file_path": "/workspace/orch/orchestrator.py",
        "direction": "downstream",
    })
    parsed = json.loads(out)
    assert "error" not in parsed
    # First Cypher call must reference qualified_name = file::__module__
    first_query, first_params = calls[0]
    assert any(
        "/workspace/orch/orchestrator.py::__module__" == v
        for v in first_params.values()
    )


@pytest.mark.asyncio
async def test_get_decorator_chain_resolves_file_path_to_module_entry():
    """Stage 8d: file_path → <file_path>::__module__ resolution."""
    session, calls = _make_session([
        [{"qname": "/workspace/foo.py::__module__"}],
        [], [], [{"name": "__module__", "qname": "/workspace/foo.py::__module__",
                  "file_path": "/workspace/foo.py", "decorators": []}],
    ])
    ext = _make_graph_extension(session)

    out = await ext._handle_get_decorator_chain({
        "file_path": "/workspace/foo.py",
    })
    parsed = json.loads(out)
    assert "error" not in parsed
    # qualified_name passed in via $qname for the target lookup
    _q, params = calls[0]
    assert params["qname"] == "/workspace/foo.py::__module__"


# ──────────────────────────────────────────────────────────────────────
# Handler-level parity — list_groups + get_workspace_file_status
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def groups_db(tmp_path: Path) -> Path:
    """Create a minimal sqlite DB with the 3 ui_repogroup* tables the
    /admin/groups endpoint reads."""
    db = tmp_path / "django.db"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE ui_repogroup (
        id INTEGER PRIMARY KEY, name TEXT, description TEXT,
        created_at TEXT, last_synced TEXT
    )""")
    conn.execute("""CREATE TABLE ui_repogroupmember (
        id INTEGER PRIMARY KEY, group_id INTEGER, repo_name TEXT, role TEXT
    )""")
    conn.execute("""CREATE TABLE ui_groupcontractlink (
        id INTEGER PRIMARY KEY, group_id INTEGER, contract_id INTEGER
    )""")
    conn.execute(
        "INSERT INTO ui_repogroup VALUES (1, 'platform', 'core stack', '2026-01', '2026-05')"
    )
    conn.execute(
        "INSERT INTO ui_repogroup VALUES (2, 'frontend', 'UI repos', '2026-01', '2026-05')"
    )
    conn.execute(
        "INSERT INTO ui_repogroupmember VALUES (1, 1, 'EVA', 'primary')"
    )
    conn.execute(
        "INSERT INTO ui_repogroupmember VALUES (2, 1, 'cca', 'primary')"
    )
    conn.execute(
        "INSERT INTO ui_repogroupmember VALUES (3, 2, 'ui-monorepo', 'primary')"
    )
    conn.commit()
    conn.close()
    return db


@pytest.mark.asyncio
async def test_list_groups_unfiltered_returns_all(groups_db: Path):
    from confucius.server.code_intelligence.metadata_extension import (
        MetadataToolsExtension,
    )
    backend = MagicMock()
    ext = MetadataToolsExtension(backend_clients=backend)

    with patch(
        "confucius.server.code_intelligence.metadata_extension._GROUPS_DB_PATH",
        str(groups_db),
    ):
        out = await ext._handle_list_groups({})

    parsed = json.loads(out)
    assert parsed["count"] == 2
    names = {g["name"] for g in parsed["groups"]}
    assert names == {"platform", "frontend"}


@pytest.mark.asyncio
async def test_list_groups_file_path_narrows_to_owning_repo(groups_db: Path):
    """file_path='/workspace/EVA/foo.py' should narrow to the 'platform'
    group only (because 'platform' has 'EVA' in its members)."""
    from confucius.server.code_intelligence.metadata_extension import (
        MetadataToolsExtension,
    )
    backend = MagicMock()
    ext = MetadataToolsExtension(backend_clients=backend)

    with patch(
        "confucius.server.code_intelligence.metadata_extension._GROUPS_DB_PATH",
        str(groups_db),
    ):
        out = await ext._handle_list_groups({
            "file_path": "/workspace/EVA/foo.py",
        })

    parsed = json.loads(out)
    assert parsed["count"] == 1
    assert parsed["groups"][0]["name"] == "platform"
    assert parsed["filter"]["owning_repo"] == "EVA"


@pytest.mark.asyncio
async def test_get_workspace_file_status_requires_file_path():
    from confucius.server.code_intelligence.metadata_extension import (
        MetadataToolsExtension,
    )
    backend = MagicMock()
    ext = MetadataToolsExtension(backend_clients=backend)
    out = await ext._handle_workspace_status({})
    parsed = json.loads(out)
    assert "error" in parsed
    assert "file_path" in parsed["error"]


@pytest.mark.asyncio
async def test_get_workspace_file_status_returns_file_block():
    """The returned JSON must surface the rich file detail block that
    the dashboard's Workspace tab shows."""
    from confucius.server.code_intelligence.metadata_extension import (
        MetadataToolsExtension,
    )

    file_row = {
        "path": "/workspace/orch/orchestrator.py",
        "project": "orch",
        "language": "python",
        "indexed_at": "2026-05-15T14:37:51",
        "folder_path": "/workspace/orch",
        "has_module_body": True,
        "top_level_call_count": 2,
        "invocation_out_count": 2,
        "function_count": 1,
        "module_entry_count": 1,
    }
    session, _calls = _make_session([[file_row]])

    backend = MagicMock()
    memgraph = MagicMock()
    memgraph.session = MagicMock(return_value=session)
    backend.memgraph = memgraph

    ext = MetadataToolsExtension(backend_clients=backend)
    out = await ext._handle_workspace_status({
        "file_path": "/workspace/orch/orchestrator.py",
    })
    parsed = json.loads(out)
    assert parsed["indexed"] is True
    assert parsed["file"]["has_module_body"] is True
    assert parsed["file"]["invocation_out_count"] == 2
    assert parsed["file"]["module_entry_count"] == 1


# ──────────────────────────────────────────────────────────────────────
# Schema-level parity — RESEARCH_TOOLS lists the new read-only tools
# ──────────────────────────────────────────────────────────────────────


def test_research_tools_includes_new_parity_tools():
    """The new read-only tools must appear in RESEARCH_TOOLS so the
    orchestrator's dual-model loop knows to keep using them on the
    small model instead of escalating to the big one."""
    from confucius.server.dual_model_orchestrator import RESEARCH_TOOLS
    assert "list_groups" in RESEARCH_TOOLS
    assert "get_workspace_file_status" in RESEARCH_TOOLS
    assert "list_rules" in RESEARCH_TOOLS
    assert "get_decorator_chain" in RESEARCH_TOOLS


# ──────────────────────────────────────────────────────────────────────
# Tool group wiring — METADATA registered for CODER + INFRA routes
# ──────────────────────────────────────────────────────────────────────


def test_metadata_toolgroup_wired_to_coder_and_infra():
    """Stage 8h/8i: METADATA must be in ROUTE_TOOL_GROUPS for CODER and
    INFRASTRUCTURE so the agent actually receives these tools in those
    routes (where file-scope metadata questions arise)."""
    from confucius.server.tool_groups import (
        ROUTE_TOOL_GROUPS,
        ToolGroup,
    )
    from confucius.server.expert_router import ExpertType

    assert ToolGroup.METADATA in ROUTE_TOOL_GROUPS[ExpertType.CODER]
    assert ToolGroup.METADATA in ROUTE_TOOL_GROUPS[ExpertType.INFRASTRUCTURE]
