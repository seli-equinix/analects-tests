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
