"""PLANNER runs the dual-model synthesis flow with a per-route synthesis prompt.

Replaces test_planner_flow_no_synthesis.py, which asserted the REVERTED Slice-3
behavior (PLANNER exempt from the synthesis flow). That blunt exemption fixed
knowledge-pipeline's topic collapse but regressed test_planning_flow — the
planner stopped at a preamble because nothing forced a complete plan.

The correct fix (this rework): PLANNER stays in the dual-model + synthesis flow
(like CODER) but uses a dedicated synthesis prompt, recovery.synthesis_planner
("produce the COMPLETE plan"), instead of the CODER recovery.synthesis ("keep it
concise / just reference it") that collapsed multi-section plans. The coder
prompt itself is also softened to KEEP concrete results (test output, file
paths, command results) — the test_complex_multi_file 'tests_ran' fix.

Locks:
  - planner agent-flow config: requires_tool_use=True, enable_dual_model=True
    (inherited), synthesis_template='recovery.synthesis_planner';
  - the two AGENT_FLOW_DEFAULTS sources (agent_config.py + models.py) agree;
  - PLANNER is NOT in _PROSE_DELIVERABLE_EXPERTS (SEARCH still is);
  - the synthesis prompt bodies (planner = complete-plan, coder = keep-results);
  - _get_synthesis_prompt honors self._flow_config['synthesis_template'].
"""
from __future__ import annotations

import types

import pytest

from confucius.server.agent_config import AGENT_FLOW_DEFAULTS


def _merged(expert: str) -> dict:
    m = dict(AGENT_FLOW_DEFAULTS.get("default", {}))
    m.update(AGENT_FLOW_DEFAULTS.get(expert, {}))
    return m


class TestPlannerFlowConfig:
    def test_planner_runs_synthesis_flow(self):
        p = _merged("planner")
        assert p["requires_tool_use"] is True, (
            "PLANNER must require tool use so the synthesis turn runs (it forces "
            "a complete plan)."
        )
        assert p["enable_dual_model"] is True, (
            "PLANNER inherits dual-model from default — it is NOT exempted."
        )

    def test_planner_uses_dedicated_synthesis_template(self):
        assert _merged("planner")["synthesis_template"] == "recovery.synthesis_planner"

    def test_coder_uses_default_synthesis_template(self):
        assert _merged("coder")["synthesis_template"] == "recovery.synthesis"

    def test_planner_base_iterations(self):
        assert _merged("planner")["base_max_iterations"] == 10


class TestAgentFlowDefaultsTwoSourceAgreement:
    """agent_config.py is read at FastAPI runtime; cca_web/ui/models.py is the
    Django seed source. The planner block MUST be semantically identical in both
    or the deployed behavior diverges from the seeded DB."""

    def test_planner_block_matches_models_py(self):
        # Parse cca_web/ui/models.py via ast (no Django import needed) and
        # extract its AGENT_FLOW_DEFAULTS literal, so we compare the two
        # sources without standing up the Django app.
        import ast
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        models_path = os.path.normpath(
            os.path.join(here, "..", "..", "cca_web", "ui", "models.py")
        )
        tree = ast.parse(open(models_path).read())
        dj = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "AGENT_FLOW_DEFAULTS":
                        dj = ast.literal_eval(node.value)
        assert dj is not None, "AGENT_FLOW_DEFAULTS not found in models.py"
        dm = dict(dj.get("default", {})); dm.update(dj.get("planner", {}))
        for k in ("requires_tool_use", "enable_dual_model", "synthesis_template",
                  "base_max_iterations"):
            a = _merged("planner").get(k)
            assert a == dm.get(k), (
                f"planner.{k} drift: agent_config={a!r} models.py={dm.get(k)!r}"
            )


class TestProseDeliverableExperts:
    def test_planner_not_exempt_search_is(self):
        pytest.importorskip(
            "langchain_core",
            reason="http_routed_entry imports orchestrator runtime; cca-tests image.",
        )
        from confucius.server.http_routed_entry import _PROSE_DELIVERABLE_EXPERTS
        from confucius.server.expert_router import ExpertType
        assert ExpertType.PLANNER not in _PROSE_DELIVERABLE_EXPERTS, (
            "PLANNER must run the synthesis flow (regression test_planning_flow)."
        )
        assert ExpertType.SEARCH in _PROSE_DELIVERABLE_EXPERTS


class TestSynthesisPromptBodies:
    def test_planner_synthesis_body_is_complete_plan(self):
        from confucius.server.prompt_loader import get_template
        body = get_template("recovery.synthesis_planner")
        assert body, "recovery.synthesis_planner must never resolve empty"
        low = body.lower()
        assert "complete plan" in low
        assert "do not collapse" in low or "drop any requested section" in low
        # Must NOT carry the coder collapse phrasing:
        assert "just reference it" not in low
        assert "don't show it again" not in low and "do not show it again" not in low

    def test_coder_synthesis_keeps_concrete_results(self):
        from confucius.server.prompt_loader import get_template
        body = get_template("recovery.synthesis")
        low = body.lower()
        assert "keep every concrete result" in low or "without dropping evidence" in low
        # The old collapse bullets must be gone:
        assert "just reference it" not in low
        assert "do not repeat code" not in low


class TestGetSynthesisPromptHonorsFlowConfig:
    """_get_synthesis_prompt() must select the slug from self._flow_config."""

    def _call(self, flow_config: dict) -> str:
        pytest.importorskip(
            "langchain_core",
            reason="dual_model_orchestrator imports langchain; cca-tests image.",
        )
        from confucius.server.dual_model_orchestrator import DualModelOrchestrator
        stub = types.SimpleNamespace(_flow_config=flow_config)
        return DualModelOrchestrator._get_synthesis_prompt(stub)

    def test_planner_flow_config_yields_planner_prompt(self):
        out = self._call({"synthesis_template": "recovery.synthesis_planner"})
        assert "complete plan" in out.lower()

    def test_empty_flow_config_falls_back_to_default_synthesis(self):
        out = self._call({})
        # default slug recovery.synthesis (softened coder prompt)
        assert "keep every concrete result" in out.lower() or "concrete" in out.lower()
