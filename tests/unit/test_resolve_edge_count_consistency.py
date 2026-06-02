"""resolve_project_calls edge-count consistency (regression-report fix).

Locks in the fix for the code-intelligence/code-trace regression: the
state-hash skip was CONTENT-only, so when an interrupted/partial reindex blew
away CALLS edges (DELETE ran, CREATE didn't finish) the content hash was
unchanged and every later run skipped on hash-match → caller edges stayed
missing → query_call_graph returned no callers and trace_execution returned
incomplete data.

The fix stores the resolved edge count on `_ResolveState` and skips ONLY when
the hash matches AND the actual edge count has NOT regressed below it. It must:
  - SKIP when hash matches and edges are intact (existing >= stored) — preserve
    the intended no-op (the devs removed an "always retry" version for churn);
  - REBUILD when the hash matches but edges regressed (existing < stored) — self-heal;
  - REBUILD when no edge count was ever stored (pre-fix _ResolveState);
  - REBUILD on force=True (unchanged).

Mocks the AsyncSession boundary (no live Memgraph), mirroring
test_memgraph_atomic_indexing.py.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from confucius.server.code_intelligence.memgraph_client import MemgraphClient


def _result(record):
    r = MagicMock()
    r.single = AsyncMock(return_value=record)
    return r


def _client_with_records(records):
    """A MemgraphClient whose session.run returns the given records in order
    (each becomes a result whose .single() yields the dict)."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.run = AsyncMock(side_effect=[_result(rec) for rec in records])
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    client = MemgraphClient(driver)
    return client, session


def _run(client, **kw):
    import asyncio
    return asyncio.run(client.resolve_project_calls("EVA", **kw))


# Records consumed in order by resolve_project_calls:
#   1 existing-count, 2 pending-count, 3 last-state (hash+edge_count),
#   [rebuild only: 4 delete, 5 rebuild-merge, 6 cleanup-stub, 7 store]
_REBUILD_TAIL = [{"deleted": 0}, {"edges_created": 100}, None, None]


@patch.object(MemgraphClient, "_compute_resolve_state_hash",
              new=AsyncMock(return_value="HASH_A"))
class TestEdgeCountConsistency:
    def test_skip_when_hash_matches_and_edges_intact(self):
        client, session = _client_with_records([
            {"existing": 100}, {"pending": 5},
            {"hash": "HASH_A", "edge_count": 100},
        ])
        res = _run(client)
        assert res.get("skipped") == "state-hash-match", res
        assert session.run.await_count == 3  # no delete/rebuild queries

    def test_rebuild_when_edges_regressed(self):
        # existing (50) dropped below the last resolved count (100) → an
        # interrupted reindex blew edges away → must rebuild despite hash match.
        client, session = _client_with_records([
            {"existing": 50}, {"pending": 5},
            {"hash": "HASH_A", "edge_count": 100},
            *_REBUILD_TAIL,
        ])
        res = _run(client)
        assert res.get("skipped") is None, res
        assert res.get("created") == 100, res
        assert session.run.await_count >= 5  # delete + rebuild ran

    def test_rebuild_when_no_stored_edge_count(self):
        # Pre-fix _ResolveState has no edge_count → can't prove edges intact → rebuild.
        client, session = _client_with_records([
            {"existing": 100}, {"pending": 5},
            {"hash": "HASH_A", "edge_count": None},
            *_REBUILD_TAIL,
        ])
        res = _run(client)
        assert res.get("skipped") is None, res
        assert res.get("created") == 100

    def test_force_always_rebuilds(self):
        client, session = _client_with_records([
            {"existing": 100}, {"pending": 5},
            {"hash": "HASH_A", "edge_count": 100},
            *_REBUILD_TAIL,
        ])
        res = _run(client, force=True)
        assert res.get("skipped") is None, res
        assert res.get("created") == 100

    def test_no_pending_skips_early(self):
        # pending==0 → nothing to resolve → early skip (no hash work).
        client, session = _client_with_records([
            {"existing": 100}, {"pending": 0},
        ])
        res = _run(client)
        assert res.get("skipped") is True
        assert session.run.await_count == 2
