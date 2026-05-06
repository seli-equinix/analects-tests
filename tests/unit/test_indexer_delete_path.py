"""Delete-path completeness for the workspace indexer.

Two pre-existing issues surfaced during the post-fix audit of the
indexer's atomic-transaction landing:

1. `MemgraphClient.clear_file` ran 16 DETACH DELETE queries with
   auto-commit each. A mid-loop Memgraph error left the file
   half-deleted (Functions gone but Class nodes still present, or
   similar). Now wrapped in an explicit transaction — same pattern
   as `_index_file_graph_once`.

2. The watcher's `_handle_file_delete` called `clear_file` directly,
   which only cleans the graph. Embedding vectors in Qdrant and
   keyword entries in BM25 stayed alive for up to 5 minutes (until
   the periodic reconciler caught up). Now both the watcher and the
   reconciler go through `WorkspaceIndexer.cleanup_deleted_file`,
   which cleans all three indexes in one call.

These tests don't need live infrastructure; they mock at the
session/transaction boundary (for clear_file) and at the
WorkspaceIndexer's clients (for the cleanup helper).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from confucius.server.code_intelligence.memgraph_client import MemgraphClient


# ── clear_file: now transactional ────────────────────────────────────


def _make_tx() -> AsyncMock:
    tx = AsyncMock()
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__ = AsyncMock(return_value=None)
    tx.run = AsyncMock()
    return tx


def _make_driver_with_tx(tx: AsyncMock) -> MagicMock:
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.begin_transaction = AsyncMock(return_value=tx)
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    return driver


class TestClearFileTransaction:
    """`clear_file` runs all 17 DETACH DELETEs (16 labels + File) inside
    ONE transaction. A mid-loop failure rolls back ALL deletions; no
    half-deleted state survives."""

    @pytest.mark.asyncio
    async def test_all_queries_use_tx_run_not_session_run(self):
        tx = _make_tx()
        driver = _make_driver_with_tx(tx)
        client = MemgraphClient(driver)

        result = await client.clear_file("/workspace/foo.py")

        assert result is True
        # 15 node-label DETACH DELETEs + 1 File DETACH DELETE = 16.
        # If a future commit adds a label to the cleanup loop and
        # forgets to update this assertion, the test catches it.
        assert tx.run.await_count == 16
        tx.__aenter__.assert_awaited_once()
        tx.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mid_loop_error_propagates_no_partial_commit(self):
        """A Memgraph TransientError on the 5th label rolls back ALL
        prior deletions. The transaction context exits with an
        exception; the driver auto-issues ROLLBACK on __aexit__."""
        tx = _make_tx()
        # First 4 succeed, 5th raises
        tx.run = AsyncMock(side_effect=[
            None, None, None, None,
            RuntimeError("Memgraph TransientError on Class label"),
        ])
        driver = _make_driver_with_tx(tx)
        client = MemgraphClient(driver)

        # `clear_file` swallows exceptions and returns False, but the
        # transaction context unwinds with the exception (verifiable
        # via `tx.run` call count: bailed at the failing query).
        result = await client.clear_file("/workspace/foo.py")
        assert result is False
        assert tx.run.await_count == 5  # bailed at the failing query
        tx.__aenter__.assert_awaited_once()
        # __aexit__ called with the exception → driver issues ROLLBACK
        tx.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_driver_returns_false(self):
        client = MemgraphClient(None)
        result = await client.clear_file("/workspace/foo.py")
        assert result is False


# ── cleanup_deleted_file: 3-index unified cleanup ────────────────────


class TestCleanupDeletedFile:
    """`WorkspaceIndexer.cleanup_deleted_file` is the single source of
    truth for "drop ALL traces of this file" — Memgraph + Qdrant + BM25.
    Used by both the realtime watcher and the 5-min reconciler so the
    delete-path is consistent across both."""

    @staticmethod
    def _make_indexer(graph_ok: bool = True):
        """Build a WorkspaceIndexer with mocked clients/graph."""
        from confucius.server.code_intelligence.workspace_indexer import (
            WorkspaceIndexer,
        )
        clients = MagicMock()
        clients.qdrant = MagicMock()
        clients.memgraph = MagicMock()
        indexer = WorkspaceIndexer(clients)

        # Mock the Memgraph graph wrapper (or none if graph_ok=False).
        graph = AsyncMock()
        graph.clear_file = AsyncMock(return_value=True)
        if graph_ok:
            indexer._graph = graph
        else:
            indexer._graph = None
            indexer._clients.memgraph = None
        return indexer, graph

    @pytest.mark.asyncio
    async def test_happy_path_calls_all_three_cleanup_paths(self):
        indexer, graph = self._make_indexer()
        # Stub Qdrant + BM25 cleanup
        indexer._delete_file_docs = AsyncMock()
        bm25_mock = AsyncMock()
        bm25_mock.remove_file = AsyncMock()
        with patch(
            "confucius.server.code_intelligence.bm25_index.get_bm25_index",
            return_value=bm25_mock,
        ):
            result = await indexer.cleanup_deleted_file("/workspace/foo.py")

        # Each cleanup happened
        graph.clear_file.assert_awaited_once_with("/workspace/foo.py")
        indexer._delete_file_docs.assert_awaited_once_with("/workspace/foo.py")
        bm25_mock.remove_file.assert_awaited_once_with("/workspace/foo.py")
        # Result reports success across all three indexes
        assert result == {"memgraph": True, "qdrant": True, "bm25": True}

    @pytest.mark.asyncio
    async def test_qdrant_failure_does_not_skip_bm25(self):
        """Independent failure modes per index — a Qdrant outage
        shouldn't block BM25 cleanup. The helper logs the failure and
        keeps going."""
        indexer, graph = self._make_indexer()
        indexer._delete_file_docs = AsyncMock(
            side_effect=RuntimeError("Qdrant unreachable"),
        )
        bm25_mock = AsyncMock()
        bm25_mock.remove_file = AsyncMock()
        with patch(
            "confucius.server.code_intelligence.bm25_index.get_bm25_index",
            return_value=bm25_mock,
        ):
            result = await indexer.cleanup_deleted_file("/workspace/foo.py")

        graph.clear_file.assert_awaited_once()  # graph still cleaned
        bm25_mock.remove_file.assert_awaited_once()  # bm25 still cleaned
        assert result["memgraph"] is True
        assert result["qdrant"] is False
        assert result["bm25"] is True

    @pytest.mark.asyncio
    async def test_graph_failure_does_not_skip_qdrant_or_bm25(self):
        indexer, graph = self._make_indexer()
        graph.clear_file = AsyncMock(side_effect=RuntimeError("Memgraph down"))
        indexer._delete_file_docs = AsyncMock()
        bm25_mock = AsyncMock()
        bm25_mock.remove_file = AsyncMock()
        with patch(
            "confucius.server.code_intelligence.bm25_index.get_bm25_index",
            return_value=bm25_mock,
        ):
            result = await indexer.cleanup_deleted_file("/workspace/foo.py")

        indexer._delete_file_docs.assert_awaited_once()
        bm25_mock.remove_file.assert_awaited_once()
        assert result == {"memgraph": False, "qdrant": True, "bm25": True}

    @pytest.mark.asyncio
    async def test_no_graph_backend_only_cleans_qdrant_and_bm25(self):
        """When Memgraph isn't configured, cleanup still proceeds for
        the other two indexes — it doesn't NOT-cleanup just because
        one backend is down/missing."""
        indexer, _ = self._make_indexer(graph_ok=False)
        indexer._delete_file_docs = AsyncMock()
        bm25_mock = AsyncMock()
        bm25_mock.remove_file = AsyncMock()
        with patch(
            "confucius.server.code_intelligence.bm25_index.get_bm25_index",
            return_value=bm25_mock,
        ):
            result = await indexer.cleanup_deleted_file("/workspace/foo.py")

        indexer._delete_file_docs.assert_awaited_once()
        bm25_mock.remove_file.assert_awaited_once()
        assert result["memgraph"] is False  # no graph → no cleanup
        assert result["qdrant"] is True
        assert result["bm25"] is True

    @pytest.mark.asyncio
    async def test_bm25_unavailable_does_not_block_other_cleanup(self):
        """If BM25 isn't initialized (get_bm25_index returns None),
        the other two indexes still get cleaned. Was a real risk
        when the indexer started before the BM25 module's lazy init
        completed — silently skipping BM25 is the right move
        rather than crashing the whole cleanup path."""
        indexer, graph = self._make_indexer()
        indexer._delete_file_docs = AsyncMock()
        with patch(
            "confucius.server.code_intelligence.bm25_index.get_bm25_index",
            return_value=None,
        ):
            result = await indexer.cleanup_deleted_file("/workspace/foo.py")

        graph.clear_file.assert_awaited_once()
        indexer._delete_file_docs.assert_awaited_once()
        assert result["memgraph"] is True
        assert result["qdrant"] is True
        assert result["bm25"] is False  # bm25 wasn't available
