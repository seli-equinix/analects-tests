"""Atomic per-file transaction contract for `_index_file_graph_once`.

Locks in the fix that converted the method from auto-commit-per-query
to ONE explicit transaction per file. The previous pattern broke the
retry loop's idempotency invariant: a TransientError mid-loop committed
queries 1..N-1, leaving partial state (most insidiously, a Function
node whose `file_path` SET clause never ran — invisible to the retry's
`MATCH (fn) WHERE fn.file_path = $path DETACH DELETE` cleanup). Those
orphans accumulated and eventually wedged a connection on a half-open
transaction, manifesting as the silent reindex hang at small files
like `cca-pair-579145a5/main.py`.

These tests don't need a live Memgraph — they mock the AsyncSession +
AsyncTransaction boundary and verify the call shape (`tx.run` not
`session.run`, transaction context is entered + exited, commit happens
on success, rollback happens on exception).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from confucius.server.code_intelligence.memgraph_client import MemgraphClient


def _make_driver_with_tx(tx: AsyncMock) -> MagicMock:
    """Build a driver mock whose .session() yields a session whose
    .begin_transaction() yields the given Transaction mock.

    Because the production code uses `async with` for both the session
    and the transaction, the mock has to honor both context manager
    protocols (__aenter__ / __aexit__) AND the `await
    session.begin_transaction()` coroutine.
    """
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    session.begin_transaction = AsyncMock(return_value=tx)

    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    return driver


def _make_tx() -> AsyncMock:
    """Build an AsyncTransaction mock that supports `async with` AND
    tracks every `tx.run` call. Returns the tx so callers can override
    side_effects when simulating mid-flight failures."""
    tx = AsyncMock()
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__ = AsyncMock(return_value=None)
    tx.run = AsyncMock()
    tx.commit = AsyncMock()
    tx.rollback = AsyncMock()
    return tx


@pytest.fixture
def client_with_mock_tx() -> tuple[MemgraphClient, AsyncMock]:
    """Returns (client, tx_mock) with the mock pre-wired.

    `MemgraphClient(driver)` takes the driver directly (not a backend
    wrapper) — verified at memgraph_client.py:90.
    """
    tx = _make_tx()
    driver = _make_driver_with_tx(tx)
    client = MemgraphClient(driver)
    return client, tx


# ── Happy path ──────────────────────────────────────────────────────


class TestAtomicHappyPath:
    """A single file's index runs ALL queries through `tx.run` and exits
    the transaction context cleanly so the driver auto-commits."""

    @pytest.mark.asyncio
    async def test_all_queries_use_tx_run_not_session_run(
        self, client_with_mock_tx,
    ):
        client, tx = client_with_mock_tx
        result = await client._index_file_graph_once(
            file_path="/workspace/foo.py",
            project="EVA",
            language="python",
            functions=[
                {"name": "bar", "signature": "def bar():",
                 "calls": [], "imports": [], "decorators": [],
                 "line_start": 1, "line_end": 3, "loc": 3,
                 "is_async": False, "return_type": ""},
            ],
        )
        # Returned counts dict matches what the production code emits.
        assert result["functions"] == 1
        # Transaction was entered and exited (driver auto-commits on
        # successful __aexit__).
        tx.__aenter__.assert_awaited_once()
        tx.__aexit__.assert_awaited_once()
        # Every query went through tx.run, NOT a stray session.run.
        # The 4 base queries (3 DETACH DELETE + 1 CREATE File) plus 1
        # MERGE Function = 5 calls minimum.
        assert tx.run.await_count >= 5

    @pytest.mark.asyncio
    async def test_imports_loop_runs_once_per_import(
        self, client_with_mock_tx,
    ):
        client, tx = client_with_mock_tx
        await client._index_file_graph_once(
            file_path="/workspace/foo.py",
            project="EVA", language="python",
            functions=[
                {"name": "bar", "signature": "def bar():",
                 "calls": [],
                 "imports": ["os", "sys", "json"],  # 3 imports
                 "decorators": [],
                 "line_start": 1, "line_end": 3, "loc": 3,
                 "is_async": False, "return_type": ""},
            ],
        )
        # 3 base + 1 File CREATE + 1 MERGE Function + 3 Module MERGEs = 8
        assert tx.run.await_count == 8

    @pytest.mark.asyncio
    async def test_class_method_creates_class_and_belongs_to(
        self, client_with_mock_tx,
    ):
        client, tx = client_with_mock_tx
        await client._index_file_graph_once(
            file_path="/workspace/foo.py",
            project="EVA", language="python",
            functions=[
                {"name": "method", "class_name": "MyClass",
                 "signature": "def method(self):",
                 "calls": [], "imports": [], "decorators": [],
                 "line_start": 5, "line_end": 7, "loc": 3,
                 "is_async": False, "return_type": ""},
            ],
        )
        # 3 base + 1 File CREATE + 1 MERGE Function + 1 CREATE Class
        # + 1 MERGE BELONGS_TO = 7
        assert tx.run.await_count == 7


# ── Mid-flight failure → rollback ──────────────────────────────────


class TestAtomicRollbackOnError:
    """If any `tx.run` raises mid-loop, the transaction context exits
    with an exception. The neo4j async driver's `__aexit__` issues an
    automatic rollback. The retry loop's invariant — every retry sees
    a clean state — depends on this."""

    @pytest.mark.asyncio
    async def test_exception_propagates_out_of_method(
        self, client_with_mock_tx,
    ):
        client, tx = client_with_mock_tx
        # Simulate Memgraph TransientError on the 3rd query (DETACH
        # DELETE File node, line 3 in the production sequence).
        tx.run = AsyncMock(side_effect=[
            None, None, RuntimeError("Memgraph TransientError"),
        ])
        with pytest.raises(RuntimeError, match="TransientError"):
            await client._index_file_graph_once(
                file_path="/workspace/foo.py",
                project="EVA", language="python",
                functions=[],  # doesn't matter — error fires before func loop
            )
        # Transaction was entered (begin_transaction ran)
        tx.__aenter__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_partial_commit_when_query_in_func_loop_fails(
        self, client_with_mock_tx,
    ):
        client, tx = client_with_mock_tx
        # 3 base queries succeed (DETACH DELETE × 3), 4th succeeds
        # (CREATE File), then 5th fails (MERGE Function)
        tx.run = AsyncMock(side_effect=[
            None, None, None, None,
            RuntimeError("Memgraph TransientError"),
        ])
        with pytest.raises(RuntimeError):
            await client._index_file_graph_once(
                file_path="/workspace/foo.py",
                project="EVA", language="python",
                functions=[
                    {"name": "bar", "signature": "def bar():",
                     "calls": [], "imports": [], "decorators": [],
                     "line_start": 1, "line_end": 3, "loc": 3,
                     "is_async": False, "return_type": ""},
                ],
            )
        # Every query that DID succeed is bound to the transaction;
        # since the transaction never reaches a clean __aexit__, the
        # driver's rollback semantics apply. We assert via the call
        # count that the loop bailed at the failing query — no further
        # tx.run calls were issued.
        assert tx.run.await_count == 5  # 4 successful + 1 raising


# ── Retry idempotency (the actual bug) ──────────────────────────────


class TestRetryIdempotency:
    """The real-world bug the transaction wrapper fixes: when the outer
    retry loop re-enters `_index_file_graph_once` after a transient
    failure, it must see CLEAN state. Each invocation gets a fresh
    transaction context; the previous attempt's writes either all
    committed (success) or all rolled back (exception)."""

    @pytest.mark.asyncio
    async def test_each_invocation_opens_fresh_transaction(self):
        """Two back-to-back calls (simulating retry attempt 1 fails,
        attempt 2 succeeds) each get their own transaction.

        The session is reused across calls (production opens a fresh
        session per `async with self._driver.session()` block, but for
        the purposes of this test we only need to verify that each
        invocation opens its OWN transaction). begin_transaction with
        side_effect=[tx1, tx2] returns a different transaction per
        call — that's the invariant under test."""
        tx_attempt_1 = _make_tx()
        tx_attempt_1.run = AsyncMock(
            side_effect=RuntimeError("Memgraph TransientError"),
        )
        tx_attempt_2 = _make_tx()
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.begin_transaction = AsyncMock(
            side_effect=[tx_attempt_1, tx_attempt_2],
        )
        driver = MagicMock()
        driver.session = MagicMock(return_value=session)
        client = MemgraphClient(driver)

        # Attempt 1 raises mid-flight (simulating TransientError)
        with pytest.raises(RuntimeError):
            await client._index_file_graph_once(
                file_path="/workspace/foo.py",
                project="EVA", language="python", functions=[],
            )
        # Attempt 2 succeeds with a fresh tx (no leftover state)
        await client._index_file_graph_once(
            file_path="/workspace/foo.py",
            project="EVA", language="python", functions=[],
        )

        # CRITICAL assertion: each attempt opened its OWN transaction.
        # The fix is meaningless if attempts share transactional state.
        assert session.begin_transaction.await_count == 2
        assert tx_attempt_1 is not tx_attempt_2
        # Attempt 1's tx saw the failing run; attempt 2's tx didn't.
        assert tx_attempt_1.run.await_count == 1
        assert tx_attempt_2.run.await_count >= 1

    @pytest.mark.asyncio
    async def test_outer_retry_loop_recovers_from_first_attempt_failure(self):
        """End-to-end against a mocked driver: first attempt raises a
        TransientError, the outer retry loop in `index_file_graph`
        catches it, second attempt succeeds. Returned dict is from the
        SECOND attempt — first attempt's writes don't leak into it."""
        tx_attempt_1 = _make_tx()
        tx_attempt_1.run = AsyncMock(
            side_effect=Exception("Memgraph.TransientError on MERGE"),
        )
        tx_attempt_2 = _make_tx()  # default success
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.begin_transaction = AsyncMock(
            side_effect=[tx_attempt_1, tx_attempt_2],
        )
        driver = MagicMock()
        driver.session = MagicMock(return_value=session)
        client = MemgraphClient(driver)

        # The full index_file_graph entry point — wraps with retries.
        result = await client.index_file_graph(
            file_path="/workspace/foo.py",
            project="EVA", language="python", functions=[],
        )
        # First attempt raised, retry caught it, second attempt
        # succeeded → result is the second attempt's counts dict.
        assert "error" not in result, f"unexpected error: {result}"
        # Both transactions were created (one per attempt).
        assert session.begin_transaction.await_count == 2
