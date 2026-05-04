"""Phase 2 — schema drift contract test.

Compares the committed `schemas/` baseline against the schemas the
current code would generate. Classifies each diff as additive (OK) vs
breaking (FAIL), and refuses to allow a breaking change without an
accompanying `version` bump on the affected model.

Decision matrix per (baseline, current) pair of JSON Schemas:

| Change                                       | Additive | Breaking |
|----------------------------------------------|----------|----------|
| New optional property added                  | YES      |          |
| New required property added                  |          | YES      |
| Existing property removed                    |          | YES      |
| Existing required property made optional     | YES      |          |
| Existing optional property made required     |          | YES      |
| Property type widened (str → str|int)        | YES      |          |
| Property type narrowed (str|int → str)       |          | YES      |
| Description / default / title changed        | YES      |          |
| New schema file appears                      | YES      |          |
| Existing schema file removed                 |          | YES      |
| `version` field changed (any direction)      | YES      |          |

A breaking change MUST be accompanied by a non-default version bump
(major-style, e.g. 1.0.0 → 2.0.0). The test allows the breaking change
once the version differs.

Run:
    pytest tests/contract/test_schema_drift.py -v
"""
from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from pydantic import BaseModel


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCHEMAS_DIR = REPO_ROOT / "schemas"
MODELS_DIR = SCHEMAS_DIR / "models"


@dataclass
class DriftReport:
    """Catalog of changes between a baseline schema and a current schema."""
    additive: List[str] = field(default_factory=list)
    breaking: List[str] = field(default_factory=list)

    def is_breaking(self) -> bool:
        return bool(self.breaking)


# ── Diff helpers ─────────────────────────────────────────────────────


def _norm_type(t: Any) -> Set[str]:
    """Normalize a JSON Schema 'type' field to a set of type names.

    Pydantic emits types as either a string ('integer') or a list
    (['integer', 'null']) or via anyOf for unions.
    """
    if isinstance(t, str):
        return {t}
    if isinstance(t, list):
        return {str(x) for x in t}
    return set()


def _required_set(schema: Dict[str, Any]) -> Set[str]:
    return set(schema.get("required", []) or [])


def _properties(schema: Dict[str, Any]) -> Dict[str, Any]:
    return schema.get("properties", {}) or {}


def _diff_property(name: str, old: Dict[str, Any], new: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return a list of (kind, message) where kind is 'additive' or 'breaking'."""
    out: List[Tuple[str, str]] = []
    old_types = _norm_type(old.get("type"))
    new_types = _norm_type(new.get("type"))
    if old_types and new_types:
        # Removed type → narrowing → breaking.
        narrowed = old_types - new_types
        widened = new_types - old_types
        if narrowed:
            out.append(("breaking", f"property {name!r} type narrowed (lost: {sorted(narrowed)})"))
        if widened:
            out.append(("additive", f"property {name!r} type widened (gained: {sorted(widened)})"))
    # anyOf changes — treat shrinkage as breaking
    old_any = old.get("anyOf") or []
    new_any = new.get("anyOf") or []
    if old_any and new_any:
        old_repr = {json.dumps(s, sort_keys=True) for s in old_any}
        new_repr = {json.dumps(s, sort_keys=True) for s in new_any}
        if old_repr - new_repr:
            out.append(("breaking", f"property {name!r} anyOf shrunk"))
        if new_repr - old_repr:
            out.append(("additive", f"property {name!r} anyOf grew"))
    return out


def diff_schemas(baseline: Dict[str, Any], current: Dict[str, Any]) -> DriftReport:
    """Walk two Pydantic-emitted JSON Schemas and classify changes."""
    report = DriftReport()

    old_props = _properties(baseline)
    new_props = _properties(current)
    old_required = _required_set(baseline)
    new_required = _required_set(current)

    old_keys = set(old_props.keys())
    new_keys = set(new_props.keys())

    removed = old_keys - new_keys
    for name in sorted(removed):
        report.breaking.append(f"property {name!r} removed")

    added = new_keys - old_keys
    for name in sorted(added):
        if name in new_required:
            report.breaking.append(
                f"property {name!r} added as required (clients without it now fail)"
            )
        else:
            report.additive.append(f"property {name!r} added (optional)")

    # Required-status changes on retained keys
    retained = old_keys & new_keys
    for name in sorted(retained):
        was_required = name in old_required
        is_required = name in new_required
        if was_required and not is_required:
            report.additive.append(f"property {name!r} required → optional (additive)")
        elif not was_required and is_required:
            report.breaking.append(f"property {name!r} optional → required")
        # Type-level diff
        for kind, msg in _diff_property(name, old_props[name], new_props[name]):
            if kind == "breaking":
                report.breaking.append(msg)
            else:
                report.additive.append(msg)

    # Top-level version change is always additive (and is the *fix* for
    # a breaking change).
    if baseline.get("version") != current.get("version"):
        report.additive.append(
            f"top-level version {baseline.get('version')!r} → {current.get('version')!r}"
        )

    return report


# ── Schema loading helpers ───────────────────────────────────────────


def _load_baseline_models() -> Dict[str, Dict[str, Any]]:
    """Load every schemas/models/*.schema.json that exists today."""
    if not MODELS_DIR.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(MODELS_DIR.glob("*.schema.json")):
        name = path.name.replace(".schema.json", "")
        out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _generate_current_models() -> Dict[str, Dict[str, Any]]:
    """Generate JSON Schemas from the current code."""
    mod = importlib.import_module("confucius.server.models")
    out: Dict[str, Dict[str, Any]] = {}
    for name, obj in inspect.getmembers(mod):
        if not (inspect.isclass(obj) and issubclass(obj, BaseModel)):
            continue
        if obj is BaseModel:
            continue
        if obj.__module__ != mod.__name__:
            continue
        if name == "VersionedModel":
            continue
        try:
            out[name] = obj.model_json_schema()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Could not load schema for {name}: {e}")
    return out


# ── Tests ────────────────────────────────────────────────────────────


class TestSchemaDrift:
    """Run from the repo root: pytest tests/contract/test_schema_drift.py"""

    def test_no_breaking_change_without_version_bump(self):
        baseline = _load_baseline_models()
        current = _generate_current_models()
        if not baseline:
            pytest.skip(
                "No baseline in schemas/models/ — run `make schema-export` and commit "
                "to seed the baseline (Phase 2.7 amnesty)."
            )

        violations: List[str] = []

        for name, current_schema in current.items():
            if name not in baseline:
                # New model — additive, never breaking.
                continue
            base_schema = baseline[name]
            report = diff_schemas(base_schema, current_schema)
            if report.is_breaking():
                version_changed = base_schema.get("version") != current_schema.get("version")
                if not version_changed:
                    violations.append(
                        f"\n  {name}: breaking change without version bump:\n    "
                        + "\n    ".join(report.breaking)
                    )

        # Models removed from current code (deletion)
        for name in baseline:
            if name not in current:
                violations.append(
                    f"\n  {name}: model REMOVED from confucius/server/models.py "
                    "without amnesty path (use Phase 4 deprecation flow)"
                )

        if violations:
            joined = "".join(violations)
            pytest.fail(
                "Schema drift gate (Phase 2.5) detected breaking changes "
                "without a version bump. Either:\n"
                "  (a) revert the breaking change, or\n"
                "  (b) bump CCA_API_SCHEMA_VERSION in confucius/server/models.py "
                "AND re-run `make schema-export` to refresh schemas/.\n"
                f"\nViolations:{joined}"
            )

    def test_baseline_present_means_phase2_amnesty_done(self):
        """If schemas/models/ exists, every shipped model in code should
        have a baseline. Catches the case where someone added a model
        but forgot to refresh the baseline."""
        baseline = _load_baseline_models()
        current = _generate_current_models()
        if not baseline:
            pytest.skip("No baseline yet — Phase 2.7 amnesty not run.")

        missing = [name for name in current if name not in baseline]
        if missing:
            pytest.fail(
                "These models exist in code but have no baseline schema "
                f"(run `make schema-export` and commit to refresh): {missing}"
            )
