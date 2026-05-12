"""Invariant 9 guard: ConfigAuditLog has bounded growth via the
`archive_audit_log` management command moving stale rows into
ConfigAuditLogArchive.

Verifies: (a) the command is idempotent, (b) dry-run is non-destructive,
(c) rows newer than the cutoff stay in the hot table, (d) the original
id is carried into the archive for forensic traceback.
"""

from __future__ import annotations

from datetime import timedelta
from io import StringIO

import pytest


pytestmark = pytest.mark.django_db


@pytest.fixture
def seeded_audit_log():
    """Insert 6 ConfigAuditLog rows: 3 older than 90d, 3 newer."""
    from django.utils import timezone
    from ui.models import ConfigAuditLog
    now = timezone.now()
    ConfigAuditLog.objects.all().delete()  # clean slate
    for i, age_days in enumerate([200, 150, 100, 30, 10, 1]):
        ConfigAuditLog.objects.create(
            table_name="RuntimeConfig",
            scope="test",
            key=f"row_{i}",
            action="update",
            old_value={"v": "old"},
            new_value={"v": "new"},
            user="test",
            source="ui",
            notes=f"seed row {i}",
        )
        # Override auto_now_add timestamp directly via update() because
        # the field is set on create.
        row = ConfigAuditLog.objects.filter(notes=f"seed row {i}").first()
        ConfigAuditLog.objects.filter(pk=row.pk).update(
            timestamp=now - timedelta(days=age_days),
        )
    return ConfigAuditLog.objects.all()


def test_dry_run_reports_count_without_writing(seeded_audit_log):
    from ui.models import ConfigAuditLog, ConfigAuditLogArchive
    from django.core.management import call_command

    out = StringIO()
    call_command("archive_audit_log", "--days", "90", "--dry-run", stdout=out)

    # Hot table untouched
    assert ConfigAuditLog.objects.count() == 6
    # Archive still empty
    assert ConfigAuditLogArchive.objects.count() == 0
    # Output mentions the candidate count
    assert "DRY RUN" in out.getvalue()
    # 3 rows are older than 90 days (200, 150, 100)
    assert "3" in out.getvalue()


def test_archive_moves_old_rows(seeded_audit_log):
    from ui.models import ConfigAuditLog, ConfigAuditLogArchive
    from django.core.management import call_command

    call_command("archive_audit_log", "--days", "90", stdout=StringIO())

    # 3 of the 6 rows aged >90d should have moved
    assert ConfigAuditLog.objects.count() == 3
    assert ConfigAuditLogArchive.objects.count() == 3
    # Archive carries original_id for traceback
    for arch in ConfigAuditLogArchive.objects.all():
        assert arch.original_id is not None
        assert arch.archived_at is not None


def test_archive_is_idempotent(seeded_audit_log):
    """Running again immediately should be a no-op (nothing newly aged
    past the cutoff)."""
    from ui.models import ConfigAuditLog, ConfigAuditLogArchive
    from django.core.management import call_command

    call_command("archive_audit_log", "--days", "90", stdout=StringIO())
    hot_after_first = ConfigAuditLog.objects.count()
    archive_after_first = ConfigAuditLogArchive.objects.count()

    call_command("archive_audit_log", "--days", "90", stdout=StringIO())
    assert ConfigAuditLog.objects.count() == hot_after_first
    assert ConfigAuditLogArchive.objects.count() == archive_after_first


def test_zero_days_archives_everything(seeded_audit_log):
    """`--days 0` is the operator-only nuclear option — moves every row
    that's older than NOW (which is all of them, modulo the
    sub-second-aged ones, but our fixture sets all timestamps in the
    past)."""
    from ui.models import ConfigAuditLog, ConfigAuditLogArchive
    from django.core.management import call_command

    call_command("archive_audit_log", "--days", "0", stdout=StringIO())
    assert ConfigAuditLog.objects.count() == 0
    assert ConfigAuditLogArchive.objects.count() == 6
