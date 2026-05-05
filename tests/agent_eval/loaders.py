"""Phase 3.4 — corpus loader.

Reads `cohorts.yaml` + `tasks.yaml` from this directory, validates the
shape of every task spec, and returns a frozen ``Corpus`` containing
both the cohort objects (Phase 3.2 ``Cohort``) and per-task
``TaskSpec`` records.

Public surface::

    from tests.agent_eval.loaders import load_default_corpus
    corpus = load_default_corpus()
    for task in corpus.tasks_in_cohort("code_edit_simple"):
        ...

This module deliberately stays declarative — actual execution
(``run_task(task) -> ChatResult``) lives in Phase 3.6's
``live_validate_agent.py``. The loader's job is to fail loudly on a
malformed corpus before any live LLM time is spent.

Cross-validation enforced here (matches the exit criteria in
``~/.claude/plans/robust-mixing-harp.md`` Phase 3.4):

  - every cohort item has a corresponding entry in tasks.yaml
  - every task references a cohort that exists in cohorts.yaml
  - prompts are non-empty
  - expected_route is one of the 7 known routes
  - success_criterion.kind is one of the 8 known kinds
  - cohort cardinality: each cohort has ≥5 items, total ≥30
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Sequence, Tuple

from confucius.core.quality.cohort import Cohort, load_cohorts


# ── Constants ───────────────────────────────────────────────────────

VALID_ROUTES: FrozenSet[str] = frozenset({
    "coder", "search", "user", "infrastructure",
    "planner", "direct", "clarify",
})

VALID_CRITERION_KINDS: FrozenSet[str] = frozenset({
    "file_exists",
    "file_contains",
    "file_unchanged",
    "tool_named",
    "route_matches",
    "judge_completed",
    "no_refusal",
    "clarification_requested",
    "count_match",
    "regex_match",
    "any_of",
})

# Default paths relative to this file.
_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_COHORTS_PATH = _THIS_DIR / "cohorts.yaml"
DEFAULT_TASKS_PATH = _THIS_DIR / "tasks.yaml"


# ── Data classes ────────────────────────────────────────────────────


@dataclass(frozen=True)
class SuccessCriterion:
    """Frozen view over one task's success_criterion dict.

    The shape varies by ``kind`` — callers branch on ``kind`` and pull
    the relevant fields from ``data`` (which is a frozen dict-like
    tuple-of-pairs).
    """
    kind: str
    data: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def get(self, key: str, default: Any = None) -> Any:
        for k, v in self.data:
            if k == key:
                return v
        return default

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.data)


@dataclass(frozen=True)
class TaskSpec:
    """One row of the corpus.

    ``prompt`` may contain ``{uid}`` placeholders that the runner fills
    with a fresh uuid4 hex prefix per evaluation pass.
    """
    name: str
    cohort: str
    prompt: str
    expected_route: str
    expected_tools: Tuple[str, ...]
    success_criterion: SuccessCriterion
    min_iterations: int
    max_iterations: int
    mutating: bool
    notes: str = ""

    def render(self, uid: str) -> str:
        """Fill {uid} placeholders in the prompt for one run."""
        return self.prompt.replace("{uid}", uid)


@dataclass(frozen=True)
class Corpus:
    cohorts: Mapping[str, Cohort]
    tasks: Mapping[str, TaskSpec]

    def tasks_in_cohort(self, cohort_name: str) -> Tuple[TaskSpec, ...]:
        c = self.cohorts.get(cohort_name)
        if c is None:
            raise KeyError(f"unknown cohort: {cohort_name!r}")
        return tuple(self.tasks[name] for name in c.items if name in self.tasks)

    def all_task_names(self) -> Tuple[str, ...]:
        return tuple(self.tasks.keys())


# ── Loader ──────────────────────────────────────────────────────────


class CorpusValidationError(ValueError):
    """Raised when cohorts.yaml or tasks.yaml fails the cross-validation
    contract. The message includes every offence the loader found, not
    just the first — fix-once-and-be-done semantics."""


def load_corpus(
    cohorts_path: Path | str | None = None,
    tasks_path: Path | str | None = None,
    *,
    min_per_cohort: int = 5,
    min_total: int = 30,
) -> Corpus:
    """Load and validate the cohort + task definitions.

    Cardinality bounds default to the Phase 3.4 contract (≥5 per cohort,
    ≥30 total). Tests with synthetic corpora pass ``min_per_cohort=0`` /
    ``min_total=0`` to skip the size checks while keeping every other
    contract (orphan tasks, unknown routes, broken cross-references).

    Raises ``CorpusValidationError`` if any contract is violated.
    """
    cohorts_p = Path(cohorts_path) if cohorts_path else DEFAULT_COHORTS_PATH
    tasks_p = Path(tasks_path) if tasks_path else DEFAULT_TASKS_PATH

    cohorts = load_cohorts(cohorts_p)
    raw_tasks = _load_yaml_dict(tasks_p)

    tasks: Dict[str, TaskSpec] = {}
    errors: List[str] = []

    for name, body in raw_tasks.items():
        try:
            tasks[name] = _validate_task(name, body, set(cohorts.keys()))
        except CorpusValidationError as e:
            errors.append(str(e))

    # Cross-check: every cohort item must have a task spec
    for cohort_name, cohort in cohorts.items():
        for item in cohort.items:
            if item not in tasks:
                errors.append(
                    f"cohort {cohort_name!r} references task {item!r} but "
                    f"no entry exists in {tasks_p.name}"
                )

    # Cardinality
    if min_per_cohort > 0:
        for cohort_name, cohort in cohorts.items():
            if cohort.size < min_per_cohort:
                errors.append(
                    f"cohort {cohort_name!r} has {cohort.size} items "
                    f"(<{min_per_cohort}) — bump it back up before approval"
                )
    if min_total > 0:
        total_items = sum(c.size for c in cohorts.values())
        if total_items < min_total:
            errors.append(
                f"corpus total is {total_items} (<{min_total}) — Phase 3.4 contract requires ≥{min_total}"
            )

    # Orphan tasks: task spec exists but no cohort references it
    cohort_items = {item for c in cohorts.values() for item in c.items}
    orphans = sorted(set(tasks.keys()) - cohort_items)
    for orphan in orphans:
        errors.append(
            f"task {orphan!r} has a spec in {tasks_p.name} but no cohort "
            f"in {cohorts_p.name} references it"
        )

    if errors:
        raise CorpusValidationError(
            f"Corpus validation failed with {len(errors)} error(s):\n  - "
            + "\n  - ".join(errors)
        )

    return Corpus(cohorts=dict(cohorts), tasks=dict(tasks))


def load_default_corpus() -> Corpus:
    """Convenience wrapper: load from this directory's default YAML files."""
    return load_corpus()


# ── Internal helpers ────────────────────────────────────────────────


def _load_yaml_dict(path: Path) -> Dict[str, Any]:
    import yaml
    if not path.exists():
        raise FileNotFoundError(f"tasks file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise CorpusValidationError(
            f"top-level of {path.name} must be a YAML mapping, got "
            f"{type(raw).__name__}"
        )
    return raw


def _validate_task(name: str, body: Any, valid_cohorts: set) -> TaskSpec:
    if not isinstance(body, dict):
        raise CorpusValidationError(
            f"task {name!r}: body must be a mapping, got {type(body).__name__}"
        )

    required = ("cohort", "prompt", "expected_route", "expected_tools",
                "success_criterion", "min_iterations", "max_iterations",
                "mutating")
    missing = [k for k in required if k not in body]
    if missing:
        raise CorpusValidationError(
            f"task {name!r}: missing required field(s) {missing}"
        )

    cohort = body["cohort"]
    if cohort not in valid_cohorts:
        raise CorpusValidationError(
            f"task {name!r}: cohort {cohort!r} not in cohorts.yaml "
            f"(known: {sorted(valid_cohorts)})"
        )

    prompt = str(body["prompt"]).strip()
    if not prompt:
        raise CorpusValidationError(f"task {name!r}: empty prompt")

    route = body["expected_route"]
    if route not in VALID_ROUTES:
        raise CorpusValidationError(
            f"task {name!r}: expected_route {route!r} not in {sorted(VALID_ROUTES)}"
        )

    tools = body.get("expected_tools", [])
    if not isinstance(tools, list):
        raise CorpusValidationError(
            f"task {name!r}: expected_tools must be a list, got {type(tools).__name__}"
        )
    expected_tools = tuple(str(t) for t in tools)

    sc_raw = body["success_criterion"]
    if not isinstance(sc_raw, dict):
        raise CorpusValidationError(
            f"task {name!r}: success_criterion must be a mapping"
        )
    kind = sc_raw.get("kind")
    if kind not in VALID_CRITERION_KINDS:
        raise CorpusValidationError(
            f"task {name!r}: success_criterion.kind {kind!r} not in "
            f"{sorted(VALID_CRITERION_KINDS)}"
        )
    extra = tuple(sorted((k, v) for k, v in sc_raw.items() if k != "kind"))
    success_criterion = SuccessCriterion(kind=kind, data=extra)

    try:
        min_iters = int(body["min_iterations"])
        max_iters = int(body["max_iterations"])
    except (TypeError, ValueError):
        raise CorpusValidationError(
            f"task {name!r}: min/max_iterations must be integers"
        )
    if min_iters < 0 or max_iters < 0:
        raise CorpusValidationError(
            f"task {name!r}: iteration bounds must be ≥0"
        )
    if max_iters < min_iters:
        raise CorpusValidationError(
            f"task {name!r}: max_iterations ({max_iters}) < min_iterations ({min_iters})"
        )

    mutating = bool(body["mutating"])
    notes = str(body.get("notes", "") or "")

    return TaskSpec(
        name=name,
        cohort=cohort,
        prompt=prompt,
        expected_route=route,
        expected_tools=expected_tools,
        success_criterion=success_criterion,
        min_iterations=min_iters,
        max_iterations=max_iters,
        mutating=mutating,
        notes=notes,
    )


__all__ = [
    "Corpus",
    "TaskSpec",
    "SuccessCriterion",
    "CorpusValidationError",
    "VALID_ROUTES",
    "VALID_CRITERION_KINDS",
    "load_corpus",
    "load_default_corpus",
    "DEFAULT_COHORTS_PATH",
    "DEFAULT_TASKS_PATH",
]
