"""Phase 3.6 — runner unit tests.

Covers:
  - `_CapturedChatResult` adapter surface (matches what tests/cca_client.py
    ChatResult exposes — content, metadata, usage, tool_names, tool_errors).
  - `evaluate_criterion()` per kind (count_match, regex_match,
    route_matches, clarification_requested, no_refusal, any_of,
    tool_named, capture-time-kind skip).
  - `evaluate_corpus()` bundle-grouping over a synthetic replay dir.
  - `emit_summary_for_bundle()` produces a LiveSummary that round-trips
    through the markdown renderer.
  - **Phase 3.8 determinism**: two consecutive --deterministic runs
    against the same captures must produce byte-identical artifacts.

Run:
    pytest tests/unit/test_runner.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.agent_eval import metrics as agent_metrics  # noqa: F401  side-effect register
from tests.agent_eval.loaders import (
    Corpus,
    SuccessCriterion,
)
from tests.agent_eval.runner import (
    _CapturedChatResult,
    emit_summary_for_bundle,
    evaluate_corpus,
    evaluate_criterion,
)


@pytest.fixture(autouse=True)
def _restore_agent_registry():
    """Phase 3.3 metric registry can be wiped by other tests'
    clear_registry_for_tests(); rebuild before each runner test."""
    agent_metrics.ensure_registered()
    yield


# ── Adapter ─────────────────────────────────────────────────────────


def _sample_record(overrides: dict | None = None) -> dict:
    base = {
        "task_name": "sample_task",
        "cohort": "code_edit_simple",
        "uid": "abc12345",
        "captured_at": "2026-05-04T01:00:00Z",
        "bundle_id": "abc123def456",
        "session_id": "agent-eval-sample-abc12345",
        "prompt_rendered": "do the thing",
        "response_content": "I did the thing.",
        "response_role": "assistant",
        "finish_reason": "stop",
        "metadata": {
            "route": "coder",
            "tool_iterations": 3,
            "tool_calls": [
                {"name": "str_replace_editor", "success": True, "iteration": 1},
                {"name": "bash", "success": False, "iteration": 2},
                {"name": "bash", "success": True, "iteration": 3},
            ],
            "stream_guard_fired": False,
            "stream_guard_fires": 0,
        },
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        "elapsed_ms": 4500.0,
        "cca_url": "https://localhost:8500",
        "completion_id": "chatcmpl-test123",
    }
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = {**base[k], **v}
            else:
                base[k] = v
    return base


class TestCapturedChatResult:
    def test_basic_fields(self):
        c = _CapturedChatResult(_sample_record())
        assert c.content == "I did the thing."
        assert c.role == "assistant"
        assert c.finish_reason == "stop"
        assert c.elapsed_ms == 4500.0
        assert c.usage["total_tokens"] == 150
        assert c.metadata["route"] == "coder"

    def test_tool_names_extracted(self):
        c = _CapturedChatResult(_sample_record())
        assert c.tool_names == ["str_replace_editor", "bash", "bash"]

    def test_tool_errors_only_unrecovered(self):
        # The middle bash call failed but a later bash succeeded.
        # eval_tool_errors does its own recovery analysis on top —
        # the adapter just exposes the raw failed entries.
        c = _CapturedChatResult(_sample_record())
        errs = c.tool_errors
        assert len(errs) == 1
        assert "bash" in errs[0]


# ── Criterion evaluator ─────────────────────────────────────────────


class TestEvaluateCriterion:
    def _captured(self, content="hello world", route="coder", tools=None):
        rec = _sample_record({
            "response_content": content,
            "metadata": {"route": route, "tool_calls": tools or []},
        })
        return _CapturedChatResult(rec)

    def test_count_match_required_present(self):
        cap = self._captured(content="hello world stripe")
        sc = SuccessCriterion(
            kind="count_match",
            data=(("response_must_contain", ["stripe"]),),
        )
        v = evaluate_criterion(sc, cap, expected_route="coder")
        assert v.passed is True

    def test_count_match_required_missing(self):
        cap = self._captured(content="hello world")
        sc = SuccessCriterion(
            kind="count_match",
            data=(("response_must_contain", ["stripe"]),),
        )
        v = evaluate_criterion(sc, cap, expected_route="coder")
        assert v.passed is False
        assert "stripe" in v.reason

    def test_count_match_any_of(self):
        cap = self._captured(content="cmd ran with healthy result")
        sc = SuccessCriterion(
            kind="count_match",
            data=(("response_must_contain_any", ["healthy", "Up"]),),
        )
        assert evaluate_criterion(sc, cap, expected_route="coder").passed

    def test_regex_match_passes(self):
        cap = self._captured(content="disk usage is 42% full")
        # `\b\d{1,3}%\b` doesn't work — the position after `%` isn't a
        # word boundary because both `%` and ` ` are non-word chars.
        # `\b\d{1,3}%` matches without requiring a trailing boundary.
        sc = SuccessCriterion(
            kind="regex_match",
            data=(("response_must_match", r"\b\d{1,3}%"),),
        )
        assert evaluate_criterion(sc, cap, expected_route="coder").passed

    def test_route_matches_pass(self):
        cap = self._captured(route="planner", content="step 1: do thing")
        sc = SuccessCriterion(
            kind="route_matches",
            data=(("route", "planner"),),
        )
        assert evaluate_criterion(sc, cap, expected_route="planner").passed

    def test_route_matches_fail(self):
        cap = self._captured(route="coder", content="x")
        sc = SuccessCriterion(
            kind="route_matches",
            data=(("route", "planner"),),
        )
        v = evaluate_criterion(sc, cap, expected_route="planner")
        assert v.passed is False
        assert "coder" in v.reason

    def test_route_matches_with_additional_forbidden_tools(self):
        cap = self._captured(
            route="planner",
            content="plan steps",
            tools=[{"name": "str_replace_editor", "success": True, "iteration": 1}],
        )
        sc = SuccessCriterion(
            kind="route_matches",
            data=(
                ("route", "planner"),
                ("additional", {"forbidden_tools": ["str_replace_editor"]}),
            ),
        )
        v = evaluate_criterion(sc, cap, expected_route="planner")
        assert v.passed is False
        assert "str_replace_editor" in v.reason

    def test_clarification_requested_with_question_mark(self):
        cap = self._captured(content="Which file did you mean?")
        sc = SuccessCriterion(
            kind="clarification_requested",
            data=(("response_must_contain_any", ["which", "what"]),),
        )
        assert evaluate_criterion(sc, cap, expected_route="clarify").passed

    def test_no_refusal_clean_response_passes(self):
        cap = self._captured(content="here you go")
        sc = SuccessCriterion(kind="no_refusal", data=())
        assert evaluate_criterion(sc, cap, expected_route="coder").passed

    def test_no_refusal_with_refusal_marker_fails(self):
        cap = self._captured(content="i don't have access to that")
        sc = SuccessCriterion(kind="no_refusal", data=())
        v = evaluate_criterion(sc, cap, expected_route="coder")
        assert v.passed is False

    def test_any_of_first_passes(self):
        cap = self._captured(content="not found")
        sc = SuccessCriterion(
            kind="any_of",
            data=(("options", [
                {"kind": "count_match", "response_must_contain_any": ["not found", "missing"]},
                {"kind": "tool_named", "tool": "str_replace_editor"},
            ]),),
        )
        v = evaluate_criterion(sc, cap, expected_route="coder")
        assert v.passed is True
        assert "any_of OK via count_match" in v.reason

    def test_tool_named_pass(self):
        cap = self._captured(
            tools=[{"name": "find_orphan_functions", "success": True, "iteration": 1}],
        )
        sc = SuccessCriterion(
            kind="tool_named",
            data=(("tool", "find_orphan_functions"),),
        )
        assert evaluate_criterion(sc, cap, expected_route="coder").passed

    def test_capture_time_kind_skips_gracefully(self):
        cap = self._captured()
        for kind in ("file_exists", "file_contains", "file_unchanged", "judge_completed"):
            sc = SuccessCriterion(kind=kind, data=())
            v = evaluate_criterion(sc, cap, expected_route="coder")
            assert v.passed is True
            assert "skipped" in v.reason


# ── Bundle grouping + summary emission ──────────────────────────────


def _make_corpus(tmp_path: Path) -> Corpus:
    """Build a tiny synthetic corpus with one cohort + 5 tasks for the
    bundle-grouping + summary tests."""
    cohorts_yaml = tmp_path / "cohorts.yaml"
    cohorts_yaml.write_text(
        "smoke:\n"
        "  description: tiny\n"
        "  items: [t1, t2, t3, t4, t5]\n"
    )
    tasks_yaml = tmp_path / "tasks.yaml"
    body = ""
    for name in ("t1", "t2", "t3", "t4", "t5"):
        body += (
            f"{name}:\n"
            f"  cohort: smoke\n"
            f"  prompt: do {name}\n"
            f"  expected_route: coder\n"
            f"  expected_tools: []\n"
            f"  success_criterion:\n"
            f"    kind: count_match\n"
            f"    response_must_contain: ['ok']\n"
            f"  min_iterations: 0\n"
            f"  max_iterations: 5\n"
            f"  mutating: false\n"
        )
    tasks_yaml.write_text(body)
    from tests.agent_eval.loaders import load_corpus
    return load_corpus(cohorts_yaml, tasks_yaml, min_per_cohort=0, min_total=0)


def _write_replay(replay_dir: Path, *, name: str, bundle_id: str | None, content: str):
    rec = _sample_record({
        "task_name": name,
        "cohort": "smoke",
        "bundle_id": bundle_id,
        "response_content": content,
    })
    (replay_dir / f"{name}.json").write_text(json.dumps(rec))


class TestBundleGrouping:
    def test_evaluate_corpus_groups_by_bundle(self, tmp_path: Path):
        corpus = _make_corpus(tmp_path)
        replay = tmp_path / "replay"
        replay.mkdir()
        _write_replay(replay, name="t1", bundle_id="bundle-A", content="ok")
        _write_replay(replay, name="t2", bundle_id="bundle-A", content="missing")
        _write_replay(replay, name="t3", bundle_id="bundle-B", content="ok")
        _write_replay(replay, name="t4", bundle_id=None, content="ok")
        # t5 absent — runner should just skip it

        by_bundle, errors = evaluate_corpus(replay_dir=replay, corpus=corpus)
        assert errors == []
        assert set(by_bundle.keys()) == {"bundle-A", "bundle-B", None}
        assert len(by_bundle["bundle-A"]) == 2
        assert len(by_bundle["bundle-B"]) == 1
        assert len(by_bundle[None]) == 1

    def test_evaluate_corpus_reports_bad_capture_filename(self, tmp_path: Path):
        corpus = _make_corpus(tmp_path)
        replay = tmp_path / "replay"
        replay.mkdir()
        # task name doesn't exist in the corpus
        rec = _sample_record({"task_name": "unknown_task"})
        (replay / "unknown.json").write_text(json.dumps(rec))

        _by, errors = evaluate_corpus(replay_dir=replay, corpus=corpus)
        assert len(errors) == 1
        assert "unknown_task" in errors[0][1]


class TestSummaryEmission:
    def test_emits_per_bundle_file(self, tmp_path: Path):
        corpus = _make_corpus(tmp_path)
        replay = tmp_path / "replay"
        replay.mkdir()
        for name in ("t1", "t2", "t3"):
            _write_replay(replay, name=name, bundle_id="abc123", content="ok")

        by_bundle, _err = evaluate_corpus(replay_dir=replay, corpus=corpus)
        path = emit_summary_for_bundle(
            "abc123", by_bundle["abc123"],
            output_dir=tmp_path,
            generated_at="2026-05-04T01:00:00Z",
        )
        assert path.exists()
        body = path.read_text()
        assert "LIVE_SUMMARY.abc123" in body
        assert "smoke" in body
        assert "criterion.task_specific" in body

    def test_deterministic_runs_byte_identical(self, tmp_path: Path):
        corpus = _make_corpus(tmp_path)
        replay = tmp_path / "replay"
        replay.mkdir()
        for name in ("t1", "t2", "t3"):
            _write_replay(replay, name=name, bundle_id="abc123", content="ok")

        by_bundle, _err = evaluate_corpus(replay_dir=replay, corpus=corpus)

        # Two emissions with different generated_at, both deterministic
        path_a = emit_summary_for_bundle(
            "abc123", by_bundle["abc123"],
            output_dir=tmp_path / "run_a",
            generated_at="2026-05-04T01:00:00Z",
            deterministic=True,
        )
        path_b = emit_summary_for_bundle(
            "abc123", by_bundle["abc123"],
            output_dir=tmp_path / "run_b",
            generated_at="2026-06-01T02:30:00Z",  # different timestamp
            deterministic=True,
        )
        assert path_a.read_bytes() == path_b.read_bytes(), (
            "deterministic mode must produce byte-identical output across runs"
        )


class TestCorpusRoundTrip:
    """Sanity-check that the runner can score the actual default corpus
    + actual default replay records without crashing — the data is
    committed in the repo so this is a real integration sanity check."""

    def test_default_corpus_loads_and_scores(self):
        from tests.agent_eval.loaders import load_default_corpus
        from tests.agent_eval.runner import DEFAULT_REPLAY_DIR
        corpus = load_default_corpus()
        if not DEFAULT_REPLAY_DIR.exists() or not list(DEFAULT_REPLAY_DIR.glob("*.json")):
            pytest.skip("no captures in replay_corpus/ — Phase 3.5 not yet run")
        by_bundle, errors = evaluate_corpus(replay_dir=DEFAULT_REPLAY_DIR, corpus=corpus)
        assert errors == [], f"unexpected scoring errors: {errors}"
        # Should see at least 1 bundle and at least 1 task
        assert len(by_bundle) >= 1
        total_tasks = sum(len(v) for v in by_bundle.values())
        assert total_tasks >= 1
