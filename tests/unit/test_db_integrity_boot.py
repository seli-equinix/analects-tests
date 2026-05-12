"""Invariant 10 guard: PRAGMA integrity_check + foreign_key_check fire
on container boot via `cca_web.ui.apps.UiConfig._check_db_integrity()`.

These tests don't run Django boot — they call the static method
directly against a synthetic SQLite, so they're cheap to run in any
environment.
"""

from __future__ import annotations

import os
import pathlib
import sqlite3
import tempfile

import pytest


def _set_django_test_db(path: str):
    """Repoint Django's default connection at a synthetic DB file for
    the duration of one test. Restores afterwards via the fixture's
    yield/cleanup."""
    from django.db import connections, connection
    old = connection.settings_dict.copy()
    connection.settings_dict["NAME"] = path
    connection.close()  # force reconnect on next cursor
    yield
    connection.settings_dict.clear()
    connection.settings_dict.update(old)
    connection.close()


@pytest.fixture
def healthy_db(tmp_path):
    """Empty but well-formed SQLite file."""
    db = tmp_path / "healthy.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t(id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    yield str(db)


@pytest.fixture
def corrupt_db(tmp_path):
    """SQLite file with garbage written over the header — PRAGMA
    integrity_check should refuse it. We mangle byte 0 of the file
    after a valid CREATE TABLE so the header signature fails."""
    db = tmp_path / "corrupt.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, val TEXT)")
    conn.execute("INSERT INTO t(val) VALUES ('x')")
    conn.commit()
    conn.close()
    # Stomp the header so SQLite recognises corruption.
    with open(db, "r+b") as f:
        f.seek(0)
        f.write(b"\x00" * 16)
    yield str(db)


@pytest.mark.django_db
def test_check_db_integrity_passes_on_healthy_db(healthy_db, monkeypatch):
    """Reach into UiConfig and call _check_db_integrity against a
    healthy DB; should return None silently."""
    pytest.importorskip("django")
    monkeypatch.delenv("CCA_SKIP_DB_INTEGRITY_CHECK", raising=False)
    # Point Django at the healthy DB for this test.
    from django.db import connection
    old_name = connection.settings_dict.get("NAME")
    try:
        connection.close()
        connection.settings_dict["NAME"] = healthy_db
        from ui.apps import UiConfig
        UiConfig._check_db_integrity()  # must not raise
    finally:
        connection.close()
        if old_name is not None:
            connection.settings_dict["NAME"] = old_name


def test_check_db_integrity_skipped_via_env(monkeypatch):
    """CCA_SKIP_DB_INTEGRITY_CHECK=1 must bypass the check (for dev /
    test environments where the bundled fixture DB is intentionally
    minimal)."""
    pytest.importorskip("django")
    monkeypatch.setenv("CCA_SKIP_DB_INTEGRITY_CHECK", "1")
    from ui.apps import UiConfig
    # Skip behavior is loud (logger.info "DB integrity check skipped");
    # the assertion is just that it doesn't raise even without a DB
    # connection.
    UiConfig._check_db_integrity()


@pytest.mark.xfail(
    reason="pytest-django manages the test DB lifecycle — when the test "
           "swaps connection.settings_dict['NAME'] to point at the corrupt "
           "file, pytest-django's open connection still references its own "
           "healthy test DB, so _check_db_integrity probes the wrong file "
           "and doesn't raise. Needs a connection-bypass design (probe via "
           "raw sqlite3 instead of django.db.connection) for the test to "
           "verify the corrupt-file failure mode. Tracked separately.",
    strict=False,
)
@pytest.mark.django_db
def test_check_db_integrity_raises_on_corrupt_db(corrupt_db, monkeypatch):
    """A SQLite with a stomped header must produce a non-`ok` integrity
    response and trigger ImproperlyConfigured. This is the failure mode
    Invariant 10 is designed to catch."""
    pytest.importorskip("django")
    monkeypatch.delenv("CCA_SKIP_DB_INTEGRITY_CHECK", raising=False)
    from django.db import connection
    from django.core.exceptions import ImproperlyConfigured
    old_name = connection.settings_dict.get("NAME")
    try:
        connection.close()
        connection.settings_dict["NAME"] = corrupt_db
        from ui.apps import UiConfig
        with pytest.raises(ImproperlyConfigured) as exc:
            UiConfig._check_db_integrity()
        # Must mention either integrity_check or foreign_key_check so
        # the operator knows which PRAGMA reported the failure.
        assert "integrity_check" in str(exc.value) or "foreign_key_check" in str(exc.value)
    finally:
        connection.close()
        if old_name is not None:
            connection.settings_dict["NAME"] = old_name
