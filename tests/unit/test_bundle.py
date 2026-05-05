"""Phase 4 — bundle assembly + hashing + validation tests.

Pure-Python tests over `confucius.core.quality.bundle`. No DB access:
the smoke tests against the live ``assemble_bundle()`` happen during
the Spark1 deploy, not here.

Run:
    pytest tests/unit/test_bundle.py -v
"""
from __future__ import annotations

import json

import pytest

from confucius.core.quality import (
    BUNDLE_SCHEMA_VERSION,
    KNOWN_ROUTES,
    compute_bundle_hash,
    hash_text,
    validate_bundle,
)


# ── compute_bundle_hash determinism ─────────────────────────────────


def _sample_payload(**overrides) -> dict:
    base = {
        "schema_version": "1",
        "route": "coder",
        "prompts": [{"slug": "task.coder", "body_hash": "abc123"}],
        "tool_groups": ["FILE", "BASH"],
        "tool_descriptions": [{"slug": "tool.web_search", "body_hash": "def456"}],
        "llm_role": {"role": "coder", "model": "qwen3", "temperature": 0.0},
        "agent_flow": {"max_iterations": 20},
        "retrieval": {"embed_model": "qwen-embed", "pipeline_version": 5},
    }
    base.update(overrides)
    return base


class TestBundleHash:
    def test_hash_is_64_hex_chars(self):
        h = compute_bundle_hash(_sample_payload())
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_deterministic_across_calls(self):
        a = compute_bundle_hash(_sample_payload())
        b = compute_bundle_hash(_sample_payload())
        assert a == b

    def test_hash_changes_on_route_change(self):
        a = compute_bundle_hash(_sample_payload(route="coder"))
        b = compute_bundle_hash(_sample_payload(route="search"))
        assert a != b

    def test_hash_changes_on_prompt_body_change(self):
        a = compute_bundle_hash(_sample_payload())
        b = compute_bundle_hash(_sample_payload(
            prompts=[{"slug": "task.coder", "body_hash": "DIFFERENT"}],
        ))
        assert a != b

    def test_hash_changes_on_llm_role_change(self):
        a = compute_bundle_hash(_sample_payload())
        b = compute_bundle_hash(_sample_payload(
            llm_role={"role": "coder", "model": "qwen3", "temperature": 0.7},
        ))
        assert a != b

    def test_hash_independent_of_dict_iteration_order(self):
        # Both dicts have the same content; order shouldn't matter
        # (compute_bundle_hash uses sort_keys=True).
        a = compute_bundle_hash(_sample_payload())
        # Reverse the keys via dict construction trick:
        scrambled = {k: _sample_payload()[k] for k in reversed(list(_sample_payload().keys()))}
        b = compute_bundle_hash(scrambled)
        assert a == b


class TestHashText:
    def test_empty_uses_sentinel(self):
        # Same sentinel as Phase 0 fingerprint — empty SHA-256.
        empty_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert hash_text(None) == empty_sha
        assert hash_text("") == empty_sha

    def test_deterministic(self):
        a = hash_text("You are a coding agent.")
        b = hash_text("You are a coding agent.")
        assert a == b

    def test_changes_on_edit(self):
        a = hash_text("You are a coding agent.")
        b = hash_text("You are a coding agent. ")
        assert a != b


# ── validate_bundle ────────────────────────────────────────────────


class TestValidateBundle:
    def test_valid_payload_returns_no_errors(self):
        assert validate_bundle(_sample_payload()) == []

    def test_non_dict_rejected(self):
        errors = validate_bundle("not a dict")  # type: ignore[arg-type]
        assert len(errors) == 1
        assert "must be a dict" in errors[0]

    def test_missing_top_level_key_caught(self):
        payload = _sample_payload()
        del payload["llm_role"]
        errors = validate_bundle(payload)
        assert any("llm_role" in e for e in errors)

    def test_unknown_route_caught(self):
        errors = validate_bundle(_sample_payload(route="nonexistent_route"))
        assert any("unknown route" in e for e in errors)

    def test_known_routes_set(self):
        # Sanity — bundle scope is per-route from G-E
        assert KNOWN_ROUTES == {"coder", "search", "user", "infrastructure", "planner"}

    def test_empty_prompts_list_caught(self):
        errors = validate_bundle(_sample_payload(prompts=[]))
        assert any("empty" in e.lower() for e in errors)

    def test_malformed_prompt_entry_caught(self):
        errors = validate_bundle(_sample_payload(
            prompts=[{"slug": "task.coder"}],  # missing body_hash
        ))
        assert any("body_hash" in e for e in errors)

    def test_non_list_tool_groups_caught(self):
        errors = validate_bundle(_sample_payload(tool_groups="FILE,BASH"))
        assert any("tool_groups" in e for e in errors)

    def test_retrieval_missing_field_caught(self):
        errors = validate_bundle(_sample_payload(
            retrieval={"embed_model": "x"},  # missing pipeline_version
        ))
        assert any("pipeline_version" in e for e in errors)

    def test_validate_aggregates_all_errors(self):
        # Multiple violations in one call — validator should report all.
        broken = {
            "schema_version": "1",
            # missing 'route', 'prompts', 'tool_groups', 'tool_descriptions',
            # 'llm_role', 'agent_flow', 'retrieval'
        }
        errors = validate_bundle(broken)
        assert len(errors) >= 2  # at least the missing-keys + downstream errors


class TestSchemaVersion:
    def test_version_constant_is_string(self):
        assert isinstance(BUNDLE_SCHEMA_VERSION, str)
        assert BUNDLE_SCHEMA_VERSION == "1"


# ── Roundtrip: payload → hash → JSON → hash matches ─────────────────


class TestRoundtrip:
    def test_payload_can_serialize_then_rehash(self):
        payload = _sample_payload()
        h1 = compute_bundle_hash(payload)
        # Persist as JSON, reload, rehash — must match.
        s = json.dumps(payload, sort_keys=True)
        round_tripped = json.loads(s)
        h2 = compute_bundle_hash(round_tripped)
        assert h1 == h2


# ── Raw-sqlite version stamping (FastAPI side) ──────────────────────


class TestEnsureVersionViaSqlite:
    """Smoke tests for the parallel raw-sqlite stamping path that
    catches FastAPI saves which bypass Django's post_save signal."""

    def _make_db(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "test.sqlite3")
        conn = sqlite3.connect(db_path)
        # Minimal versions of the two tables — only the columns that
        # ``ensure_version_via_sqlite`` reads/writes.
        conn.executescript(
            """
            CREATE TABLE ui_contextbundle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                route varchar(20) NOT NULL UNIQUE,
                description TEXT NOT NULL,
                workspace_context_ref varchar(64) NOT NULL,
                schema_version varchar(10) NOT NULL,
                created_at datetime NOT NULL,
                updated_at datetime NOT NULL
            );
            CREATE TABLE ui_contextbundleversion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bundle_hash varchar(64) NOT NULL UNIQUE,
                parent_hash varchar(64) NOT NULL,
                status varchar(20) NOT NULL,
                member_refs TEXT NOT NULL,
                eval_pass_rate REAL,
                eval_summary_path varchar(300) NOT NULL,
                created_at datetime NOT NULL,
                created_by varchar(80) NOT NULL,
                activated_at datetime,
                activated_by varchar(80) NOT NULL,
                schema_version varchar(10) NOT NULL,
                bundle_id bigint NOT NULL
            );
            """
        )
        conn.commit()
        conn.close()
        return db_path

    def test_first_stamp_is_live(self, tmp_path):
        from confucius.core.quality.bundle import ensure_version_via_sqlite
        db_path = self._make_db(tmp_path)
        status = ensure_version_via_sqlite(
            "coder",
            bundle_hash="a" * 64,
            payload={"route": "coder"},
            full_bodies={"task.coder": "body"},
            db_path=db_path,
        )
        assert status == "live"

        import sqlite3
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT status, activated_by FROM ui_contextbundleversion"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "live"
        assert rows[0][1] == "auto-on-save"  # default created_by promoted
        conn.close()

    def test_second_stamp_is_candidate(self, tmp_path):
        from confucius.core.quality.bundle import ensure_version_via_sqlite
        db_path = self._make_db(tmp_path)
        ensure_version_via_sqlite(
            "coder", bundle_hash="a" * 64,
            payload={"route": "coder"}, full_bodies={}, db_path=db_path,
        )
        status = ensure_version_via_sqlite(
            "coder", bundle_hash="b" * 64,
            payload={"route": "coder", "v": 2}, full_bodies={}, db_path=db_path,
        )
        assert status == "candidate"

    def test_idempotent_for_same_hash(self, tmp_path):
        from confucius.core.quality.bundle import ensure_version_via_sqlite
        db_path = self._make_db(tmp_path)
        for _ in range(3):
            ensure_version_via_sqlite(
                "coder", bundle_hash="a" * 64,
                payload={"route": "coder"}, full_bodies={}, db_path=db_path,
            )
        import sqlite3
        conn = sqlite3.connect(db_path)
        n = conn.execute("SELECT COUNT(*) FROM ui_contextbundleversion").fetchone()[0]
        conn.close()
        assert n == 1  # second + third call short-circuited

    def test_parent_hash_chains(self, tmp_path):
        from confucius.core.quality.bundle import ensure_version_via_sqlite
        db_path = self._make_db(tmp_path)
        ensure_version_via_sqlite(
            "coder", bundle_hash="a" * 64,
            payload={"route": "coder"}, full_bodies={}, db_path=db_path,
        )
        ensure_version_via_sqlite(
            "coder", bundle_hash="b" * 64,
            payload={"route": "coder", "v": 2}, full_bodies={}, db_path=db_path,
        )
        import sqlite3
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT bundle_hash, parent_hash, status FROM ui_contextbundleversion ORDER BY id"
        ).fetchall()
        conn.close()
        assert rows[0] == ("a" * 64, "", "live")
        assert rows[1] == ("b" * 64, "a" * 64, "candidate")

    def test_missing_db_returns_none(self, tmp_path):
        from confucius.core.quality.bundle import ensure_version_via_sqlite
        bogus = str(tmp_path / "does-not-exist.sqlite3")
        result = ensure_version_via_sqlite(
            "coder", bundle_hash="a" * 64,
            payload={}, full_bodies={}, db_path=bogus,
        )
        assert result is None  # graceful no-op, not an exception

    def test_routes_get_independent_bundles(self, tmp_path):
        from confucius.core.quality.bundle import ensure_version_via_sqlite
        db_path = self._make_db(tmp_path)
        s1 = ensure_version_via_sqlite(
            "coder", bundle_hash="a" * 64,
            payload={"route": "coder"}, full_bodies={}, db_path=db_path,
        )
        s2 = ensure_version_via_sqlite(
            "search", bundle_hash="b" * 64,
            payload={"route": "search"}, full_bodies={}, db_path=db_path,
        )
        # Each route's first stamp is its own live — they don't share
        # the "has_live" check across bundles.
        assert s1 == "live"
        assert s2 == "live"
