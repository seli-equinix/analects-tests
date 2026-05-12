"""Unit-test conftest: pure-Python tests that don't need the live CCA server.

Overrides the parent conftest's autouse `require_cca_healthy` fixture so
tests in this directory can run without a running cca container. Unit
tests should never hit the network — anything that does belongs under
tests/integration/, tests/coder/, etc.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def require_cca_healthy():
    """No-op override of the parent autouse fixture for unit tests."""
    yield


@pytest.fixture(autouse=True)
def trace_test():
    """No-op override of the parent's Phoenix-wrapping fixture.

    The parent's `trace_test` does `provider.force_flush()` + `time.sleep(1)`
    after every test plus the gRPC export round-trip — ~4s per test. Unit
    tests don't run agent loops and don't need traces; skipping the fixture
    cuts the unit-tests bucket from ~18min back under 1min.
    """
    yield
