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
def trace_test():
    """No-op override of the parent's Phoenix-wrapping fixture.

    The parent's `trace_test` flushes Phoenix + sleeps 1s per test (~4s
    total overhead). Contract tests only inspect source code / schemas —
    no agent involvement — so the trace adds nothing.
    """
    yield
