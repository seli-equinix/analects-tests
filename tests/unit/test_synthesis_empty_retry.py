"""Synthesis-empty retry gate (Slice 1 — complex_multi_file P-fix).

Regression guard for test_complex_multi_file_project Turn 6: the synthesis
turn ran, the model reasoned but emitted nothing after the bare ``</think>``,
the reasoning/answer split correctly stripped it to empty, and CCA returned a
0-char response (`response_not_empty: 0 chars`). The near-empty guard didn't
catch it because ``clear_response_text`` (called at synthesis) wipes the
response buffer but NOT ``_primary_streamed_chars`` — so the stale-high counter
fooled the guard.

``_should_retry_empty_synthesis`` is the one-shot gate that, in exactly this
situation (synthesis done + tool work + empty real response + not yet retried),
forces ONE direct-answer turn so the work just done is actually reported.

Driven via a stub ``self`` — no full orchestrator instance needed (same shape
as test_hallucination_guard). importorskip-guarded for node5; runs in the
cca-tests CI image.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip(
    "langchain_core",
    reason="dual_model_orchestrator imports langchain; runs in cca-tests CI image.",
)

from confucius.server.dual_model_orchestrator import (  # noqa: E402
    DualModelOrchestrator,
)


def _gate(
    *,
    response_text: str,
    synthesis_done: bool = True,
    retried: bool = False,
    had_tools: bool = True,
):
    """Call _should_retry_empty_synthesis with a stub self + stub io/context."""
    s = types.SimpleNamespace()
    s._synthesis_done = synthesis_done
    s._synthesis_empty_retried = retried
    s._had_tool_iterations = had_tools
    io = types.SimpleNamespace(get_response_text=lambda: response_text)
    ctx = types.SimpleNamespace(io=io)
    return DualModelOrchestrator._should_retry_empty_synthesis(s, ctx)


class TestRetryFires:
    def test_empty_after_synthesis_with_tool_work_fires(self):
        # The exact P22... Turn-6 shape: synthesis ran, tools ran, the real
        # assembled response is empty → force one direct-answer retry.
        assert _gate(response_text="") is True

    def test_whitespace_only_response_fires(self):
        # The </think> split can leave just stray whitespace.
        assert _gate(response_text="   \n\n  ") is True

    def test_none_response_fires(self):
        # get_response_text() never returns None today, but the gate must
        # treat a None defensively (the `or ""`).
        s = types.SimpleNamespace(
            _synthesis_done=True,
            _synthesis_empty_retried=False,
            _had_tool_iterations=True,
        )
        ctx = types.SimpleNamespace(
            io=types.SimpleNamespace(get_response_text=lambda: None)
        )
        assert DualModelOrchestrator._should_retry_empty_synthesis(s, ctx) is True


class TestRetryDoesNotFire:
    def test_non_empty_response_does_not_fire(self):
        # The normal case: synthesis produced a real answer → no retry.
        assert _gate(response_text="All 6 tests passed (test_add, ...).") is False

    def test_already_retried_does_not_fire_again(self):
        # One-shot: never loop, even if the retry itself produced empty.
        assert _gate(response_text="", retried=True) is False

    def test_no_synthesis_does_not_fire(self):
        # Synthesis never ran (e.g., DIRECT-ish path) → not our case.
        assert _gate(response_text="", synthesis_done=False) is False

    def test_no_tool_work_does_not_fire(self):
        # A pure-conversation empty turn has no concrete results to report;
        # leave it to the near-empty guard, don't force a tool-result recap.
        assert _gate(response_text="", had_tools=False) is False
