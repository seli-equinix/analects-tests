"""Router guard: pure planning/architecture requests → planner (P22679).

Regression guard for routing-edge-cases test_planning_flow Turn 2: the Qwen3-4B
router classified "design a CI/CD pipeline … high-level architecture and steps"
as `coder` (tech mentions Docker/K8s/Python pulled it), so the agent explored +
web-researched instead of producing a structured plan. `_guard_planner_coder`
deterministically overrides coder→planner for a PURE planning request, while
leaving real coding/infra/codebase-query requests on their route.

Pure-Python (expert_router has no langchain import). Runs on node5 AND in CI.
This battery is the deterministic half of the "full regression-validate" — the
end-to-end 4B+guard behaviour is spot-checked live via POST /route/test.
"""
from __future__ import annotations

import pytest

from confucius.server.expert_router import (
    ExpertType,
    RouteDecision,
    _guard_planner_coder,
    _PLANNING_PATTERNS,
    _PLAN_ONLY_PATTERNS,
)


def _route(expert, msg):
    d = RouteDecision(expert=expert, estimated_steps=3)
    return _guard_planner_coder(d, msg).expert


# ── coder → planner (the misroute being fixed) ───────────────────────


class TestFlipsToPlanner:
    def test_cicd_pipeline_design(self):
        msg = (
            "I need to design a CI/CD pipeline for a Python microservices "
            "project with 5 services. The pipeline should handle testing, "
            "building Docker images, and deploying to Kubernetes. What would "
            "be the high-level architecture and steps?"
        )
        assert _route(ExpertType.CODER, msg) == ExpertType.PLANNER

    def test_monitoring_stack_design(self):
        msg = (
            "Design a monitoring stack for a production Kubernetes cluster "
            "with 20 microservices. What's the architecture?"
        )
        assert _route(ExpertType.CODER, msg) == ExpertType.PLANNER

    def test_migration_steps_only(self):
        # Names a file path (pulled it to coder) + "migrate" (a coding verb),
        # but "Steps only" forces the planner flip.
        msg = (
            "How would you migrate from Pydantic v1 to v2 in "
            "confucius/server/models.py? Steps only."
        )
        assert _route(ExpertType.CODER, msg) == ExpertType.PLANNER

    def test_plan_dont_execute_refactor(self):
        # "refactor" is a coding verb, but "(don't execute)" is an explicit
        # no-code marker that overrides the execution gate.
        msg = (
            "Plan (don't execute) how to refactor a 500-line monolith into 3 "
            "modules. Walk through the steps."
        )
        assert _route(ExpertType.CODER, msg) == ExpertType.PLANNER

    def test_what_would_be_the_best_approach(self):
        msg = "What would be the best approach to structure a multi-tenant SaaS database?"
        assert _route(ExpertType.CODER, msg) == ExpertType.PLANNER

    def test_estimated_steps_bumped(self):
        d = RouteDecision(expert=ExpertType.CODER, estimated_steps=1)
        out = _guard_planner_coder(d, "Design a CI/CD pipeline architecture and steps")
        assert out.expert == ExpertType.PLANNER
        assert out.estimated_steps >= 5


# ── stays coder (no regression) ──────────────────────────────────────


class TestStaysCoder:
    def test_write_function_and_run(self):
        msg = "Write a Python function to compute fibonacci and run it with n=10"
        assert _route(ExpertType.CODER, msg) == ExpertType.CODER

    def test_build_calculator_project(self):
        msg = "Build me a Python calculator project with add and subtract functions"
        assert _route(ExpertType.CODER, msg) == ExpertType.CODER

    def test_refactor_module(self):
        msg = "Refactor the auth module to use dependency injection"
        assert _route(ExpertType.CODER, msg) == ExpertType.CODER

    def test_existing_project_architecture_query(self):
        # KEY: asking ABOUT an existing project's architecture is a codebase
        # query (coder), NOT a design request. Bare "architecture" must not flip.
        msg = "Tell me about the EVA project architecture and how the modules fit together"
        assert _route(ExpertType.CODER, msg) == ExpertType.CODER


# ── guard only acts on coder (never hijacks other routes) ────────────


class TestOnlyActsOnCoder:
    def test_infra_planning_message_stays_infra(self):
        # If the 4B routed a design-y infra task to infrastructure, the guard
        # must NOT pull it to planner.
        msg = "Design a monitoring stack for the production cluster"
        assert _route(ExpertType.INFRASTRUCTURE, msg) == ExpertType.INFRASTRUCTURE

    def test_search_stays_search(self):
        msg = "Design patterns — what would be the best approach? Search the web."
        assert _route(ExpertType.SEARCH, msg) == ExpertType.SEARCH

    def test_planner_stays_planner(self):
        assert _route(ExpertType.PLANNER, "Design a CI/CD pipeline") == ExpertType.PLANNER


# ── pattern unit checks ──────────────────────────────────────────────


class TestPatterns:
    def test_planning_matches_design_verb(self):
        assert _PLANNING_PATTERNS.search("design a CI/CD pipeline")

    def test_planning_does_not_match_bare_architecture(self):
        assert not _PLANNING_PATTERNS.search("the EVA project architecture is layered")

    def test_plan_only_matches_steps_only(self):
        assert _PLAN_ONLY_PATTERNS.search("Steps only.")
        assert _PLAN_ONLY_PATTERNS.search("(don't execute)")
        assert not _PLAN_ONLY_PATTERNS.search("write the code now")
