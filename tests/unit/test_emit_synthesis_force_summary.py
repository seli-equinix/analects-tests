"""DualModelOrchestrator._emit_synthesis: force-summary vs normal synthesis.

Regression guard for complex_multi_file_project turn 3: a 'read the file back'
turn called str_replace_editor(view) but the agent's only assistant text was a
CoT preamble ("I'll use str_replace_editor to view it") — the content stayed in
the tool RESULT and was never summarized, so the returned response was the bare
reasoning (0 function references found).

_emit_synthesis now branches:
  - no real assistant answer in the buffer (get_response_text empty) → inject a
    directive 'summarize the tool results' nudge (NOT clear_response_text), set
    _synthesis_done (one-shot) → the next turn produces the actual answer.
  - a real answer exists → normal synthesis (clear working notes + synthesis
    prompt), unchanged.

Drives the method with a stub self + mock context, so no live server/runtime.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip(
    "langchain_core",
    reason="dual_model_orchestrator imports the orchestrator runtime; runs in "
           "the cca-tests CI image.",
)

from confucius.server.dual_model_orchestrator import DualModelOrchestrator  # noqa: E402


def _ctx(has_real_answer: bool):
    """Mock AnalectRunContext capturing injected messages."""
    io = MagicMock()
    io.has_real_answer = MagicMock(return_value=has_real_answer)
    io.clear_response_text = AsyncMock()
    mm = MagicMock()
    mm.added = []
    mm.add_messages = MagicMock(side_effect=lambda msgs: mm.added.extend(msgs))
    return SimpleNamespace(io=io, memory_manager=mm), io, mm


def _stub_self():
    return SimpleNamespace(
        _synthesis_done=False,
        _all_called_tools={"str_replace_editor"},
        _get_synthesis_prompt=lambda: "SYNTHESIS_PROMPT_BODY",
    )


def _run(self_obj, ctx):
    import asyncio
    asyncio.run(DualModelOrchestrator._emit_synthesis(self_obj, ctx))


class TestEmitSynthesis:
    def test_no_real_answer_forces_summary_nudge(self):
        # has_real_answer() False → only a CoT preamble (or empty) after tool work
        s = _stub_self()
        ctx, io, mm = _ctx(has_real_answer=False)
        _run(s, ctx)
        # force-summary path: NO clear (nothing real to clear), one message injected
        io.clear_response_text.assert_not_called()
        assert s._synthesis_done is True
        assert len(mm.added) == 1
        body = mm.added[0].content
        assert "have NOT yet answered" in body
        assert "viewed or read a file" in body
        assert "SYNTHESIS_PROMPT_BODY" not in body  # not the generic synthesis prompt

    def test_real_answer_uses_normal_synthesis(self):
        s = _stub_self()
        ctx, io, mm = _ctx(has_real_answer=True)
        _run(s, ctx)
        # normal synthesis: clears working notes + injects the synthesis prompt
        io.clear_response_text.assert_awaited_once()
        assert s._synthesis_done is True
        assert len(mm.added) == 1
        assert mm.added[0].content == "SYNTHESIS_PROMPT_BODY"
