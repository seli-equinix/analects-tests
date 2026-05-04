"""Unit tests for the Phase 3.2 eval primitives.

Covers `confucius/core/quality/{metric,cohort,rubric,live_summary}.py`.

The contracts being tested:
- Metric registration is name-keyed, idempotent, introspectable.
- Cohort YAML loads both legacy flat form and richer dict form.
- Rubric scoring is pure (same inputs → same RubricResult tuple).
- Rubric error containment: a metric that throws becomes a `passed=False`
  result with the exception in `details`; the rubric itself doesn't crash.
- LiveSummary `to_markdown_for_diff` is byte-identical across two runs of
  the same bundle (the determinism contract that Phase 3.8 will gate on).

Run:
    pytest tests/unit/test_quality_eval.py -v
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from confucius.core.quality import (
    Cohort,
    LiveSummary,
    MetricResult,
    Rubric,
    cohort_from_items,
    clear_registry_for_tests,
    get_metric,
    list_metrics,
    load_cohorts,
    register_metric,
)


# ── Metric registration ─────────────────────────────────────────────


class TestMetricRegistry:
    def setup_method(self):
        clear_registry_for_tests()

    def teardown_method(self):
        clear_registry_for_tests()

    def test_register_and_lookup(self):
        @register_metric("test.always_pass")
        def m(target, ctx=None):
            return True

        looked_up = get_metric("test.always_pass")
        assert looked_up is m
        assert "test.always_pass" in list_metrics()

    def test_bare_bool_wraps_to_metric_result(self):
        @register_metric("test.bool_pass")
        def m(target):
            return True

        r = get_metric("test.bool_pass")("anything")
        assert isinstance(r, MetricResult)
        assert r.value is True
        assert r.passed is True

    def test_bare_float_uses_threshold(self):
        @register_metric("test.score", threshold=0.8)
        def m(target):
            return 0.9

        r = get_metric("test.score")("anything")
        assert r.value == 0.9
        assert r.threshold == 0.8
        assert r.passed is True

        @register_metric("test.score_low", threshold=0.8)
        def m_low(target):
            return 0.5

        r = get_metric("test.score_low")("anything")
        assert r.passed is False

    def test_metric_returning_full_result_keeps_it(self):
        @register_metric("test.detailed")
        def m(target, ctx=None):
            return MetricResult(
                name="test.detailed",
                value="custom",
                passed=False,
                details=(("error_count", 3),),
            )

        r = get_metric("test.detailed")("x")
        assert r.passed is False
        assert dict(r.details) == {"error_count": 3}

    def test_lookup_unknown_raises_keyerror(self):
        import pytest as _pytest
        with _pytest.raises(KeyError):
            get_metric("does.not.exist")


# ── Cohort loading ───────────────────────────────────────────────────


class TestCohortLoader:
    def test_legacy_flat_form(self, tmp_path: Path):
        path = tmp_path / "cohorts.yaml"
        path.write_text(textwrap.dedent("""\
            gold:
              - microsoft/pwsh-archive
              - vmware/powercli-vm
            python_regen:
              - dagster/postgres
        """))
        cohorts = load_cohorts(path)
        assert set(cohorts.keys()) == {"gold", "python_regen"}
        assert cohorts["gold"].size == 2
        assert cohorts["gold"].items == ("microsoft/pwsh-archive", "vmware/powercli-vm")
        assert cohorts["gold"].description == ""

    def test_rich_dict_form(self, tmp_path: Path):
        path = tmp_path / "cohorts.yaml"
        path.write_text(textwrap.dedent("""\
            code_edit_simple:
              description: One-file edits in EVA workspace
              success_criterion: passes pytest + matches expected diff
              items:
                - eva-rename-helper-fn
                - eva-add-cli-flag
              priority: 1
        """))
        cohorts = load_cohorts(path)
        c = cohorts["code_edit_simple"]
        assert c.size == 2
        assert c.description == "One-file edits in EVA workspace"
        assert c.success_criterion.startswith("passes pytest")
        assert dict(c.metadata) == {"priority": 1}

    def test_missing_file_raises(self, tmp_path: Path):
        import pytest as _pytest
        with _pytest.raises(FileNotFoundError):
            load_cohorts(tmp_path / "nope.yaml")

    def test_top_level_must_be_dict(self, tmp_path: Path):
        import pytest as _pytest
        path = tmp_path / "cohorts.yaml"
        path.write_text("- just\n- a\n- list\n")
        with _pytest.raises(ValueError, match="mapping at the top level"):
            load_cohorts(path)

    def test_cohort_from_items_sets_metadata(self):
        c = cohort_from_items(
            "smoke", ["a", "b"], description="quick", priority=1, owner="seli",
        )
        assert c.size == 2
        assert c.description == "quick"
        assert dict(c.metadata) == {"owner": "seli", "priority": 1}


# ── Rubric scoring ───────────────────────────────────────────────────


class TestRubricScoring:
    def setup_method(self):
        clear_registry_for_tests()

    def teardown_method(self):
        clear_registry_for_tests()

    def test_score_target_runs_every_metric(self):
        @register_metric("test.always_pass")
        def m1(target):
            return True

        @register_metric("test.length_ge_10")
        def m2(target):
            return MetricResult(name="test.length_ge_10", value=len(target), passed=len(target) >= 10)

        r = Rubric(name="r", metrics=("test.always_pass", "test.length_ge_10")).score("hello world!")
        assert r.passed is True
        assert len(r.metric_results) == 2
        m = r.metric("test.length_ge_10")
        assert m and m.value == len("hello world!")

    def test_required_subset_decides_overall_pass(self):
        @register_metric("required.ok")
        def m1(target):
            return True

        @register_metric("optional.fails")
        def m2(target):
            return MetricResult(name="optional.fails", value=0, passed=False)

        r = Rubric(
            name="r",
            metrics=("required.ok", "optional.fails"),
            required=("required.ok",),
        ).score("anything")
        assert r.passed is True  # optional metric failure ignored

    def test_metric_exception_becomes_passed_false(self):
        @register_metric("test.raises")
        def m(target):
            raise ValueError("synthetic boom")

        @register_metric("test.also_pass")
        def m2(target):
            return True

        result = Rubric(name="r", metrics=("test.raises", "test.also_pass")).score("x")
        bad = result.metric("test.raises")
        assert bad and bad.passed is False
        assert "synthetic boom" in dict(bad.details).get("error", "")
        assert result.passed is False  # required default = all metrics

    def test_unknown_metric_in_rubric_raises_at_construction(self):
        import pytest as _pytest
        with _pytest.raises(KeyError, match="unknown metrics"):
            Rubric(name="bad", metrics=("nonexistent.metric",))


# ── LiveSummary determinism ──────────────────────────────────────────


class TestLiveSummary:
    def setup_method(self):
        clear_registry_for_tests()

        @register_metric("Y1")
        def y1(target):
            return True

        @register_metric("Y2", threshold=2)
        def y2(target):
            return len(str(target))

    def teardown_method(self):
        clear_registry_for_tests()

    def _build_summary(self, ts: str) -> LiveSummary:
        rubric = Rubric(name="r.test", metrics=("Y1", "Y2"))
        bucket_a = [rubric.score("ab", target_id="t1"), rubric.score("abcd", target_id="t2")]
        bucket_b = [rubric.score("xyz", target_id="t3")]
        return LiveSummary.from_results(
            bundle_hash="sha256:abc123",
            rubric_name="r.test",
            generated_at=ts,
            results_by_cohort={"alpha": bucket_a, "bravo": bucket_b},
            thresholds={"Y2": 2.0},
        )

    def test_to_markdown_contains_expected_structure(self):
        s = self._build_summary("2026-05-04T01:00:00Z")
        md = s.to_markdown()
        assert md.startswith("# LIVE_SUMMARY.sha256:abc123.md")
        assert "## Cohort: alpha" in md
        assert "## Cohort: bravo" in md
        assert "overall_pass_rate" in md
        # Cohort ordering is alphabetic
        assert md.index("Cohort: alpha") < md.index("Cohort: bravo")

    def test_diff_form_strips_generated_at(self):
        a = self._build_summary("2026-05-04T01:00:00Z").to_markdown_for_diff()
        b = self._build_summary("2026-05-05T03:30:00Z").to_markdown_for_diff()
        assert a == b
        assert "generated_at" not in a

    def test_full_markdown_differs_only_in_generated_at(self):
        a = self._build_summary("2026-05-04T01:00:00Z").to_markdown()
        b = self._build_summary("2026-05-05T03:30:00Z").to_markdown()
        # Strip the generated_at line from both and compare — should be equal
        a_stripped = "\n".join(l for l in a.splitlines() if not l.startswith("- **generated_at**:"))
        b_stripped = "\n".join(l for l in b.splitlines() if not l.startswith("- **generated_at**:"))
        assert a_stripped == b_stripped

    def test_overall_pass_rate(self):
        s = self._build_summary("2026-05-04T01:00:00Z")
        # 3 targets — all "ab"/"abcd"/"xyz" satisfy Y1=True and Y2 (length>=2)
        assert s.total_targets == 3
        assert s.total_passed == 3
        assert s.overall_pass_rate == 1.0
