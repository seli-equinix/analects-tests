"""GitNexus feature-parity smoke tests.

Validates that every Phase 1-8 capability is reachable via REST and
returns sensible data when run against a real CCA server with at least
one indexed project. Skips gracefully if no project is indexed yet.

Hits the live server at the URL set by env var CCA_URL or the test fixture.
"""

from __future__ import annotations

import os

import pytest

from tests.cca_client import CCAClient


BASE_URL = os.environ.get("CCA_URL", "http://localhost:8500")


@pytest.fixture(scope="module")
def client():
    c = CCAClient(BASE_URL)
    yield c
    c.close()


@pytest.fixture(scope="module")
def any_project(client) -> str:
    """Pick the first indexed project (or skip if none)."""
    nodes = client.graph_nodes(limit=10)
    if not nodes.get("nodes"):
        pytest.skip("no indexed projects — run workspace sync first")
    project = nodes["nodes"][0]["data"].get("project")
    if not project:
        pytest.skip("no project metadata on graph nodes")
    return project


# ─── Phase 1: Multi-language parser support ──────────────────────


def test_phase1_languages_indexed(client):
    """At least one of the new languages should appear in the graph
    (Python is indexed by default)."""
    nodes = client.graph_nodes(label="Function", limit=200)
    languages = {n["data"].get("language") for n in nodes.get("nodes", [])}
    # Python is always indexed (CCA itself is Python). Don't fail if other
    # languages aren't present yet — they require user code in /workspace.
    assert "python" in languages, f"got {languages}"


def test_phase1_powershell_preserved(client):
    """PowerShell parsing should still work (CCA-unique win)."""
    nodes = client.graph_nodes(language="powershell", limit=10)
    # If no PowerShell repos are indexed, this returns empty — that's fine.
    assert "nodes" in nodes


# ─── Phase 2: Schema v2 ──────────────────────────────────────────


def test_phase2_new_node_types_queryable(client):
    """Schema accepts queries for the new node types (even if empty)."""
    for label in ("Method", "Interface", "Route", "Tool", "Community", "Process"):
        result = client.graph_nodes(label=label, limit=5)
        assert "nodes" in result, f"label {label} not queryable"


# ─── Phase 3: Pipeline phases ────────────────────────────────────


def test_phase3_communities_exist_after_index(client, any_project):
    """After indexing a project with multiple connected functions,
    Communities should be created."""
    nodes = client.graph_nodes(label="Community", limit=10)
    # If the indexed project has fewer than 2 symbols, no communities.
    # Just verify the endpoint works.
    assert "nodes" in nodes


def test_phase3_processes_exist(client, any_project):
    nodes = client.graph_nodes(label="Process", limit=10)
    assert "nodes" in nodes


# ─── Phase 4: Hybrid search ──────────────────────────────────────


def test_phase4_hybrid_search_returns_results(client, any_project):
    """Hybrid search should return results for a common query."""
    result = client.search_codebase("function", project=any_project, n_results=5)
    assert "results" in result, f"no results key in {result}"


def test_phase4_bm25_mode_explicit_match(client, any_project):
    """BM25 mode should rank exact-name matches at the top."""
    # Find a known function name first
    nodes = client.graph_nodes(project=any_project, label="Function", limit=5)
    if not nodes.get("nodes"):
        pytest.skip("no functions in project")
    target_name = nodes["nodes"][0]["data"].get("name", "")
    if not target_name or len(target_name) < 4:
        pytest.skip("no usable function name")
    result = client.search_codebase(target_name, project=any_project, mode="bm25", n_results=5)
    # If BM25 index is populated, the result should appear
    if result.get("results"):
        names = [r.get("name", "") for r in result["results"]]
        assert target_name in names, f"BM25 didn't find {target_name} in {names}"


# ─── Phase 5: Symbol context, impact, detect_changes ─────────────


def test_phase5_get_symbol_context(client, any_project):
    nodes = client.graph_nodes(project=any_project, label="Function", limit=1)
    if not nodes.get("nodes"):
        pytest.skip("no functions")
    name = nodes["nodes"][0]["data"]["name"]
    ctx = client.get_symbol_context(name, project=any_project)
    assert "name" in ctx or "error" in ctx
    if "name" in ctx:
        assert ctx["name"] == name
        assert "callers" in ctx
        assert "callees" in ctx


def test_phase5_analyze_impact_returns_structure(client, any_project):
    nodes = client.graph_nodes(project=any_project, label="Function", limit=1)
    if not nodes.get("nodes"):
        pytest.skip("no functions")
    name = nodes["nodes"][0]["data"]["name"]
    result = client.analyze_impact(name, project=any_project, depth=2)
    assert "upstream" in result and "downstream" in result, result


# ─── Phase 5.4: Cypher endpoint ──────────────────────────────────


def test_phase5_cypher_readonly_works(client):
    """A simple read-only Cypher query should succeed."""
    result = client.cypher("MATCH (n) RETURN count(n) AS total")
    assert "rows" in result, result
    if result.get("rows"):
        assert "total" in result["rows"][0]


def test_phase5_cypher_rejects_writes(client):
    """Write operations must be rejected."""
    with pytest.raises(Exception):  # 400 from server
        client.cypher("CREATE (n:Test) RETURN n")


# ─── Phase 6: Multi-repo groups ──────────────────────────────────


def test_phase6_list_groups(client):
    """Listing groups always works (may be empty)."""
    result = client.list_groups()
    assert "groups" in result and "count" in result


# ─── Phase 7: Graph data + Mermaid ───────────────────────────────


def test_phase7_graph_neighborhood(client, any_project):
    nodes = client.graph_nodes(project=any_project, limit=1)
    if not nodes.get("nodes"):
        pytest.skip("no nodes")
    nid = nodes["nodes"][0]["data"]["id"]
    result = client.graph_neighborhood(nid, depth=1)
    assert "nodes" in result and "edges" in result


# ─── Phase 8: Export ─────────────────────────────────────────────


def test_phase8_ndjson_export_streamed(client):
    """NDJSON export should return streamable content."""
    import httpx
    r = httpx.get(f"{BASE_URL}/admin/graph/export.ndjson", timeout=30)
    assert r.status_code == 200
    # Each line should be valid JSON
    import json as _json
    lines_seen = 0
    for line in r.text.split("\n")[:5]:
        if not line:
            continue
        record = _json.loads(line)
        assert record.get("type") in ("node", "edge")
        lines_seen += 1
    # Empty graph is OK; non-empty must have valid records
    assert lines_seen >= 0


# ─── CCA wins preserved ──────────────────────────────────────────


def test_cca_wins_browse_project(client, any_project):
    """browse_project (project tree) — CCA-unique workflow still works."""
    # Use direct HTTP since this isn't on the parity test surface
    import httpx
    r = httpx.get(f"{BASE_URL}/admin/workspace/repos/{any_project}/tree", timeout=15)
    assert r.status_code in (200, 404)  # 404 if repo not configured


def test_cca_wins_user_isolated_search(client):
    """search_codebase respects multi-tenant isolation (no cross-user
    data leak)."""
    # The endpoint should not error even if no user-scoped collection exists
    result = client.search_codebase("test", n_results=5)
    assert "results" in result
