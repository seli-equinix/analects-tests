"""Phase 3.4 — corpus validation gate.

Verifies that ``tests/agent_eval/cohorts.yaml`` and ``tasks.yaml`` are
internally consistent. This is the foundation for Phase 3.5 (replay
corpus capture) and 3.6 (live_validate_agent.py); if the corpus
shape is wrong, downstream tooling silently runs against bad data.

Run:
    pytest tests/unit/test_corpus_load.py -v

CI: bucketed under the ``unit-tests`` job; also has a per-test
``test-corpus-load`` GitLab job for the dashboard Run button.
"""
from __future__ import annotations

import re

import pytest

from tests.agent_eval.loaders import (
    VALID_CRITERION_KINDS,
    VALID_ROUTES,
    Corpus,
    CorpusValidationError,
    SuccessCriterion,
    TaskSpec,
    load_corpus,
    load_default_corpus,
)


# ── Loader: default corpus passes validation ───────────────────────


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return load_default_corpus()


class TestDefaultCorpus:
    def test_loads_without_errors(self, corpus: Corpus):
        assert isinstance(corpus, Corpus)

    def test_six_cohorts(self, corpus: Corpus):
        assert set(corpus.cohorts.keys()) == {
            "code_edit_simple",
            "code_edit_multi_file",
            "infra_query",
            "search_synthesize",
            "planner_only",
            "failure_handling",
        }

    def test_thirty_tasks(self, corpus: Corpus):
        assert len(corpus.tasks) == 30

    def test_each_cohort_has_five_items(self, corpus: Corpus):
        for name, cohort in corpus.cohorts.items():
            assert cohort.size == 5, f"{name}: expected 5 items, got {cohort.size}"

    def test_every_cohort_item_has_a_task(self, corpus: Corpus):
        for cohort_name, cohort in corpus.cohorts.items():
            for item in cohort.items:
                assert item in corpus.tasks, (
                    f"cohort {cohort_name!r} references {item!r} but no spec exists"
                )

    def test_every_task_has_a_cohort_referencing_it(self, corpus: Corpus):
        cohort_items = {item for c in corpus.cohorts.values() for item in c.items}
        orphans = sorted(set(corpus.tasks.keys()) - cohort_items)
        assert orphans == [], f"orphan tasks (no cohort references them): {orphans}"


# ── Per-task field validation ──────────────────────────────────────


class TestTaskFields:
    def test_all_routes_known(self, corpus: Corpus):
        for name, task in corpus.tasks.items():
            assert task.expected_route in VALID_ROUTES, (
                f"task {name!r}: route {task.expected_route!r} unknown"
            )

    def test_all_criterion_kinds_known(self, corpus: Corpus):
        for name, task in corpus.tasks.items():
            assert task.success_criterion.kind in VALID_CRITERION_KINDS, (
                f"task {name!r}: criterion kind {task.success_criterion.kind!r} unknown"
            )

    def test_no_empty_prompts(self, corpus: Corpus):
        for name, task in corpus.tasks.items():
            assert task.prompt.strip(), f"task {name!r}: empty prompt"

    def test_iteration_bounds_sane(self, corpus: Corpus):
        for name, task in corpus.tasks.items():
            assert task.min_iterations >= 0, f"task {name!r}: negative min_iterations"
            assert task.max_iterations >= task.min_iterations, (
                f"task {name!r}: max < min ({task.max_iterations} < {task.min_iterations})"
            )

    def test_uid_placeholder_renders(self, corpus: Corpus):
        # Any task whose prompt mentions {uid} must produce a valid
        # rendered string that no longer contains the placeholder.
        for name, task in corpus.tasks.items():
            if "{uid}" not in task.prompt:
                continue
            rendered = task.render("abc12345")
            assert "{uid}" not in rendered, f"task {name!r}: render leaves {{uid}}"
            assert "abc12345" in rendered, f"task {name!r}: render didn't substitute uid"

    def test_mutating_tasks_use_unique_prefix(self, corpus: Corpus):
        # Every mutating task should reference a path with the {uid}
        # placeholder so concurrent runs don't collide.
        for name, task in corpus.tasks.items():
            if not task.mutating:
                continue
            haystack = task.prompt
            sc_data = dict(task.success_criterion.data)
            haystack += " " + str(sc_data)
            assert "{uid}" in haystack, (
                f"task {name!r}: marked mutating but no {{uid}} placeholder "
                f"in prompt or success_criterion — concurrent runs would collide"
            )


# ── Cohort-specific contracts ──────────────────────────────────────


class TestCohortContracts:
    def test_planner_tasks_route_to_planner(self, corpus: Corpus):
        for task in corpus.tasks_in_cohort("planner_only"):
            assert task.expected_route == "planner", (
                f"{task.name}: planner_only cohort but route={task.expected_route!r}"
            )

    def test_planner_tasks_have_no_expected_tools(self, corpus: Corpus):
        # planner_only tasks should NOT call tools — they're produce-a-plan,
        # not execute-a-plan.
        for task in corpus.tasks_in_cohort("planner_only"):
            assert task.expected_tools == (), (
                f"{task.name}: planner_only but expected_tools={task.expected_tools}"
            )

    def test_infra_tasks_route_to_infrastructure(self, corpus: Corpus):
        for task in corpus.tasks_in_cohort("infra_query"):
            assert task.expected_route == "infrastructure"
            assert "bash" in task.expected_tools, (
                f"{task.name}: infra_query without bash tool — how does it inspect?"
            )

    def test_failure_handling_tasks_have_recovery_criterion(self, corpus: Corpus):
        # Every failure_handling task must encode "what good looks like"
        # — the criterion should mention text that recovery would say,
        # OR check file_unchanged, OR be clarification_requested, OR any_of.
        ok_kinds = {
            "count_match",
            "file_unchanged",
            "clarification_requested",
            "any_of",
        }
        for task in corpus.tasks_in_cohort("failure_handling"):
            assert task.success_criterion.kind in ok_kinds, (
                f"{task.name}: failure_handling tasks must use one of "
                f"{sorted(ok_kinds)}; got {task.success_criterion.kind!r}"
            )


# ── Negative tests: validator catches bad input ────────────────────


class TestValidatorRejectsBadInput:
    def test_rejects_unknown_route(self, tmp_path):
        cohorts = tmp_path / "cohorts.yaml"
        cohorts.write_text(
            "smoke:\n  description: x\n  success_criterion: y\n  items: [task_a, task_b, task_c, task_d, task_e]\n",
        )
        tasks = tmp_path / "tasks.yaml"
        # 5 tasks, one with a bogus route
        spec_template = (
            "{name}:\n"
            "  cohort: smoke\n"
            "  prompt: do thing\n"
            "  expected_route: {route}\n"
            "  expected_tools: []\n"
            "  success_criterion:\n"
            "    kind: count_match\n"
            "    response_must_contain: [x]\n"
            "  min_iterations: 0\n"
            "  max_iterations: 1\n"
            "  mutating: false\n"
        )
        bodies = []
        for i, name in enumerate(["task_a", "task_b", "task_c", "task_d", "task_e"]):
            route = "made_up_route" if i == 0 else "coder"
            bodies.append(spec_template.format(name=name, route=route))
        tasks.write_text("".join(bodies))

        with pytest.raises(CorpusValidationError, match="made_up_route"):
            load_corpus(cohorts, tasks)

    def test_rejects_missing_task_spec(self, tmp_path):
        cohorts = tmp_path / "cohorts.yaml"
        cohorts.write_text(
            "smoke:\n  items: [task_a, task_b, task_c, task_d, task_e]\n",
        )
        tasks = tmp_path / "tasks.yaml"
        # Only define task_a — task_b..task_e are dangling
        tasks.write_text(
            "task_a:\n"
            "  cohort: smoke\n"
            "  prompt: do thing\n"
            "  expected_route: coder\n"
            "  expected_tools: []\n"
            "  success_criterion:\n"
            "    kind: count_match\n"
            "  min_iterations: 0\n"
            "  max_iterations: 1\n"
            "  mutating: false\n"
        )
        with pytest.raises(CorpusValidationError, match="no entry exists"):
            load_corpus(cohorts, tasks)

    def test_rejects_orphan_task(self, tmp_path):
        cohorts = tmp_path / "cohorts.yaml"
        cohorts.write_text(
            "smoke:\n  items: [task_a, task_b, task_c, task_d, task_e]\n",
        )
        tasks = tmp_path / "tasks.yaml"
        # Define 5 expected + 1 orphan
        body = ""
        for name in ["task_a", "task_b", "task_c", "task_d", "task_e", "orphan_extra"]:
            body += (
                f"{name}:\n"
                f"  cohort: smoke\n"
                f"  prompt: do thing\n"
                f"  expected_route: coder\n"
                f"  expected_tools: []\n"
                f"  success_criterion:\n"
                f"    kind: count_match\n"
                f"  min_iterations: 0\n"
                f"  max_iterations: 1\n"
                f"  mutating: false\n"
            )
        tasks.write_text(body)
        with pytest.raises(CorpusValidationError, match="orphan_extra"):
            load_corpus(cohorts, tasks)

    def test_rejects_max_lt_min(self, tmp_path):
        cohorts = tmp_path / "cohorts.yaml"
        cohorts.write_text(
            "smoke:\n  items: [task_a, task_b, task_c, task_d, task_e]\n",
        )
        tasks = tmp_path / "tasks.yaml"
        body = ""
        for i, name in enumerate(["task_a", "task_b", "task_c", "task_d", "task_e"]):
            min_v, max_v = (5, 1) if i == 0 else (0, 1)
            body += (
                f"{name}:\n"
                f"  cohort: smoke\n"
                f"  prompt: do thing\n"
                f"  expected_route: coder\n"
                f"  expected_tools: []\n"
                f"  success_criterion:\n"
                f"    kind: count_match\n"
                f"  min_iterations: {min_v}\n"
                f"  max_iterations: {max_v}\n"
                f"  mutating: false\n"
            )
        tasks.write_text(body)
        with pytest.raises(CorpusValidationError, match="max"):
            load_corpus(cohorts, tasks)
