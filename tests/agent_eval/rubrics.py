"""Phase 3.3.c — route-specific agent rubrics.

Each rubric is a frozen ``Rubric`` (Phase 3.2) bundling the right
metrics for its route. ``required`` is the load-bearing subset: the
rubric's overall ``passed`` is True iff every metric in ``required``
passes. Non-required metrics still appear in the artifact (and feed
the LIVE_SUMMARY tables) so trends are visible even when they don't
gate.

Importing this module triggers metric registration (see
``tests/agent_eval/metrics.py``). Tests should import this module
directly when they want to score against a rubric — they don't need
to also import ``metrics``.
"""
from __future__ import annotations

from confucius.core.quality import Rubric

# Importing the metrics module registers every Y/bonus metric. Keep
# this import at module top so Rubric construction below sees them.
from . import metrics as _metrics  # noqa: F401


# ── Per-route rubrics (v1) ──────────────────────────────────────────


CODER = Rubric(
    name="agent.coder.v1",
    description=(
        "Coder route — code editing, debugging, file ops. Y2 (tool errors) "
        "and Y3 (iteration efficiency) are load-bearing alongside the "
        "judge-scored Y1/Y4 since coder runs frequently call many tools."
    ),
    metrics=(
        "Y1.task_completion",
        "Y2.tool_errors_clean",
        "Y3.iteration_efficiency",
        "Y4.response_quality",
        "Y5.no_hallucination",
        "Y6.no_stream_guard_fire",
        "Y7.latency_ok",
        "Y8.token_cost",
        "bonus.not_empty",
        "bonus.no_error",
        "bonus.no_refusal",
        "bonus.coherent",
    ),
    required=(
        "Y1.task_completion",
        "Y2.tool_errors_clean",
        "bonus.not_empty",
        "bonus.no_error",
        "bonus.coherent",
    ),
)


SEARCH = Rubric(
    name="agent.search.v1",
    description=(
        "Search route — web research and document synthesis. Y4 (response "
        "quality, judge-scored) replaces Y2 as a required gate since "
        "search runs are LLM-heavy and tool-light."
    ),
    metrics=(
        "Y1.task_completion",
        "Y4.response_quality",
        "Y5.no_hallucination",
        "Y6.no_stream_guard_fire",
        "Y7.latency_ok",
        "Y8.token_cost",
        "bonus.not_empty",
        "bonus.no_error",
        "bonus.no_refusal",
        "bonus.coherent",
    ),
    required=(
        "Y1.task_completion",
        "Y4.response_quality",
        "bonus.not_empty",
        "bonus.no_refusal",
        "bonus.coherent",
    ),
)


USER = Rubric(
    name="agent.user.v1",
    description=(
        "User route — identity / profile / fact storage. Adds the "
        "user_identified bonus metric to the required set since "
        "the route's whole purpose is identity work."
    ),
    metrics=(
        "Y1.task_completion",
        "Y4.response_quality",
        "Y6.no_stream_guard_fire",
        "Y7.latency_ok",
        "bonus.not_empty",
        "bonus.no_error",
        "bonus.user_identified",
        "bonus.coherent",
    ),
    required=(
        "Y1.task_completion",
        "bonus.user_identified",
        "bonus.not_empty",
        "bonus.coherent",
    ),
)


INFRA = Rubric(
    name="agent.infra.v1",
    description=(
        "Infrastructure route — Docker, SSH, Swarm, deployments. Tool "
        "correctness is critical (Y2 required); iteration count tends "
        "to be higher than coder so Y3 is observed but not required."
    ),
    metrics=(
        "Y1.task_completion",
        "Y2.tool_errors_clean",
        "Y3.iteration_efficiency",
        "Y6.no_stream_guard_fire",
        "Y7.latency_ok",
        "Y8.token_cost",
        "bonus.not_empty",
        "bonus.no_error",
        "bonus.no_refusal",
        "bonus.coherent",
    ),
    required=(
        "Y1.task_completion",
        "Y2.tool_errors_clean",
        "bonus.not_empty",
        "bonus.no_error",
        "bonus.coherent",
    ),
)


PLANNER = Rubric(
    name="agent.planner.v1",
    description=(
        "Planner route — multi-step planning without execution. Quality "
        "of the produced plan (Y4 judge) is the primary gate; tool "
        "metrics relaxed since planner runs typically don't call tools."
    ),
    metrics=(
        "Y1.task_completion",
        "Y4.response_quality",
        "Y6.no_stream_guard_fire",
        "Y7.latency_ok",
        "Y8.token_cost",
        "bonus.not_empty",
        "bonus.no_error",
        "bonus.coherent",
    ),
    required=(
        "Y1.task_completion",
        "Y4.response_quality",
        "bonus.not_empty",
        "bonus.coherent",
    ),
)


# Convenience lookup: route name → rubric
RUBRICS_BY_ROUTE: dict[str, Rubric] = {
    "coder": CODER,
    "search": SEARCH,
    "user": USER,
    "infrastructure": INFRA,
    "infra": INFRA,
    "planner": PLANNER,
}


def rubric_for_route(route: str) -> Rubric:
    """Return the rubric for a route name; defaults to CODER for unknown
    routes so a misclassification never crashes scoring."""
    return RUBRICS_BY_ROUTE.get(route, CODER)


__all__ = [
    "CODER",
    "SEARCH",
    "USER",
    "INFRA",
    "PLANNER",
    "RUBRICS_BY_ROUTE",
    "rubric_for_route",
]
