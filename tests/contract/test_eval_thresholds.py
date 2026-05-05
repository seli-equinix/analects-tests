"""Phase 3.7 — eval-gate threshold contract.

Per-bundle assertion that the captured replay corpus, scored through
the Phase-3.6 runner, stays above the floor numbers in
``tests/agent_eval/thresholds.yaml``. A bundle dropping below any
threshold fails the gate; CI blocks the PR.

Run:
    pytest tests/contract/test_eval_thresholds.py -v

CI: dedicated `eval-thresholds` job (allow_failure: false) + per-test
`test-eval-thresholds` for the dashboard. Bypassable with the
existing `[skip-contract]` token in the commit message.

Threshold lookup: bundle_id-keyed match in thresholds.yaml, falling
back to ``default``. Add a new bundle to the YAML when capturing
against a new prompt-set so the gate has a baseline to score
against.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pytest

from tests.agent_eval import metrics as agent_metrics
from tests.agent_eval.loaders import load_default_corpus
from tests.agent_eval.runner import DEFAULT_REPLAY_DIR, evaluate_corpus

_THRESHOLDS_PATH = (
    Path(__file__).resolve().parent.parent
    / "agent_eval" / "thresholds.yaml"
)


@pytest.fixture(autouse=True)
def _restore_agent_registry():
    """The Phase-3.2 metric registry can be wiped by other tests'
    clear_registry_for_tests() — rebuild it before each gate run."""
    agent_metrics.ensure_registered()
    yield


def _load_thresholds() -> Dict[str, Dict[str, float]]:
    import yaml
    with _THRESHOLDS_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_threshold(
    bundle_id: Optional[str],
    table: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Pick a threshold row for the bundle. Strategy:
      1. ``no-bundle`` literal for None.
      2. Exact match by bundle_id prefix (16-char hash).
      3. Fall back to ``default``.
    """
    if bundle_id is None:
        if "no-bundle" in table:
            return table["no-bundle"]
    else:
        # Try the canonical 16-char prefix used in LIVE_SUMMARY filenames.
        short = bundle_id[:16]
        if short in table:
            return table[short]
    return table["default"]


@pytest.fixture(scope="module")
def scored_corpus():
    """Score the default replay corpus once per test session."""
    corpus = load_default_corpus()
    if not DEFAULT_REPLAY_DIR.exists() or not list(DEFAULT_REPLAY_DIR.glob("*.json")):
        pytest.skip("no captures in replay_corpus/ — run `python3 -m tests.agent_eval.capture --all` first")
    by_bundle, errors = evaluate_corpus(replay_dir=DEFAULT_REPLAY_DIR, corpus=corpus)
    if errors:
        pytest.fail(f"unexpected scoring errors:\n  - " + "\n  - ".join(f"{n}: {m}" for n, m in errors))
    return by_bundle


@pytest.fixture(scope="module")
def thresholds():
    if not _THRESHOLDS_PATH.exists():
        pytest.skip(f"thresholds file missing: {_THRESHOLDS_PATH}")
    return _load_thresholds()


# ── Gate tests ──────────────────────────────────────────────────────


class TestEvalThresholds:
    """Per-bundle floor checks."""

    def test_at_least_one_bundle_scored(self, scored_corpus):
        assert len(scored_corpus) >= 1, (
            "no bundles produced — replay corpus is empty or scoring failed"
        )

    def test_per_bundle_floors_met(self, scored_corpus, thresholds):
        violations = []
        for bundle_id, evals in scored_corpus.items():
            row = _resolve_threshold(bundle_id, thresholds)
            n = len(evals)
            r = sum(1 for e in evals if e.rubric_passed)
            c = sum(1 for e in evals if e.criterion_passed)
            both = sum(1 for e in evals if e.rubric_passed and e.criterion_passed)
            r_pct = 100.0 * r / n if n else 0.0
            c_pct = 100.0 * c / n if n else 0.0
            b_pct = 100.0 * both / n if n else 0.0

            label = bundle_id or "<no-bundle>"
            min_tasks = int(row.get("min_tasks", 1))
            if n < min_tasks:
                violations.append(
                    f"{label}: only {n} tasks scored (floor {min_tasks})"
                )
            if r_pct < float(row.get("min_rubric_pct", 0)):
                violations.append(
                    f"{label}: rubric {r_pct:.1f}% < floor {row['min_rubric_pct']}% ({r}/{n})"
                )
            if c_pct < float(row.get("min_criterion_pct", 0)):
                violations.append(
                    f"{label}: criterion {c_pct:.1f}% < floor {row['min_criterion_pct']}% ({c}/{n})"
                )
            if b_pct < float(row.get("min_both_pct", 0)):
                violations.append(
                    f"{label}: both {b_pct:.1f}% < floor {row['min_both_pct']}% ({both}/{n})"
                )

        if violations:
            pytest.fail(
                "Eval-gate threshold failures:\n  - " + "\n  - ".join(violations)
            )

    def test_every_scored_bundle_has_a_threshold_row(self, scored_corpus, thresholds):
        """If a bundle was scored but isn't in thresholds.yaml, fall
        back to `default`. That's allowed but warn-loudly so a
        reviewer notices an unknown bundle reached production."""
        unknown = []
        for bundle_id in scored_corpus:
            if bundle_id is None:
                key = "no-bundle"
            else:
                key = bundle_id[:16]
            if key not in thresholds:
                unknown.append(bundle_id or "<no-bundle>")
        if unknown:
            # Don't fail the gate — `default` row covers it. But report
            # so reviewers see the gap and add a baseline if needed.
            print(
                f"\nNOTICE: {len(unknown)} bundle(s) using `default` thresholds: "
                f"{unknown}. Add explicit rows to thresholds.yaml after "
                f"baselining."
            )
