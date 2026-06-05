"""PLANNER must not run the dual-model + synthesis flow (Slice 3).

Regression guard for test_knowledge_pipeline migration ("Only 1/4 migration
topics covered"). A plan's deliverable IS the prose answer. PLANNER had
inherited the default enable_dual_model=True + requires_tool_use=True, so a
planning request got nudged to call write_memory (→ _had_tool_iterations) and
then forced through the CODER-oriented recovery.synthesis prompt ("keep it
concise… do not repeat… just reference it"), which collapsed a multi-section
plan into a terse action-summary and dropped requested topics. The planner's
own task prompt already says "Return the COMPLETE plan".

Fix: PLANNER mirrors SEARCH (a prose-deliverable route that works correctly) —
enable_dual_model=False + requires_tool_use=False — so the model's complete
first-pass plan is returned verbatim. This locks the code defaults; the live
ui_agentflowconfig row is updated at deploy time (the seed is create-if-missing
and won't auto-update existing rows).

Pure-stdlib import (agent_config.py has no Django/langchain deps).
"""
from __future__ import annotations

from confucius.server.agent_config import AGENT_FLOW_DEFAULTS


def _merged(expert: str) -> dict:
    """Code-default merge (default block + sparse expert block), DB-free —
    mirrors get_agent_flow_config's merge when no DB row exists."""
    m = dict(AGENT_FLOW_DEFAULTS.get("default", {}))
    m.update(AGENT_FLOW_DEFAULTS.get(expert, {}))
    return m


class TestPlannerFlowDefaults:
    def test_planner_disables_dual_model(self):
        p = _merged("planner")
        assert p["enable_dual_model"] is False, (
            "PLANNER must NOT run dual-model/synthesis — the synthesis prompt "
            "collapses a multi-section plan into a terse summary."
        )

    def test_planner_does_not_require_tool_use(self):
        p = _merged("planner")
        assert p["requires_tool_use"] is False, (
            "PLANNER must not be nudged to call a tool — a plan's deliverable "
            "is the prose answer, not a write_memory side effect."
        )

    def test_planner_mirrors_search_flow_shape(self):
        # SEARCH is the proven prose-deliverable route; PLANNER must match it.
        p, s = _merged("planner"), _merged("search")
        assert p["enable_dual_model"] is False
        assert s["enable_dual_model"] is False
        assert p["requires_tool_use"] is False
        assert s["requires_tool_use"] is False

    def test_coder_still_runs_dual_model(self):
        # The fix is scoped to PLANNER — CODER must keep the dual-model flow.
        c = _merged("coder")
        assert c["enable_dual_model"] is True
        assert c["requires_tool_use"] is True


class TestProseDeliverableExperts:
    """The REAL runtime gate. http_routed_entry sets orchestrator._tool_orch_params
    (8B handoff) and _requires_tool_use from a hardcoded expert set — NOT from the
    agent-flow config (those flags are vestigial). The synthesis branch is gated on
    _requires_tool_use (dual_model_orchestrator.py:2115), so PLANNER must be in the
    prose-deliverable exemption to avoid the synthesis collapse."""

    def test_planner_and_search_are_prose_deliverable(self):
        import pytest
        pytest.importorskip(
            "langchain_core",
            reason="http_routed_entry imports the orchestrator runtime; runs in "
                   "the cca-tests CI image.",
        )
        from confucius.server.http_routed_entry import _PROSE_DELIVERABLE_EXPERTS
        from confucius.server.expert_router import ExpertType
        assert ExpertType.PLANNER in _PROSE_DELIVERABLE_EXPERTS
        assert ExpertType.SEARCH in _PROSE_DELIVERABLE_EXPERTS

    def test_coder_infra_user_require_tools(self):
        import pytest
        pytest.importorskip("langchain_core")
        from confucius.server.http_routed_entry import _PROSE_DELIVERABLE_EXPERTS
        from confucius.server.expert_router import ExpertType
        # These routes' deliverable is a tool side effect — they keep the nudge.
        assert ExpertType.CODER not in _PROSE_DELIVERABLE_EXPERTS
        assert ExpertType.INFRASTRUCTURE not in _PROSE_DELIVERABLE_EXPERTS
        assert ExpertType.USER not in _PROSE_DELIVERABLE_EXPERTS
