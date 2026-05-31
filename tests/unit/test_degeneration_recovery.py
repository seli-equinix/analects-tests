"""Degeneration recovery — nudge-then-stop, no hard kill (sweep Fix A).

Regression guard for P22679_routing-edge-cases: a stream_guard
``DegenerationTerminal`` (retry budget exhausted on a legitimate structured
response) propagated uncaught and KILLED the whole agent run (output=None) —
the test only received the buffered "let me explore" preamble. The fix mirrors
the error circuit-breaker: ``_handle_degeneration`` nudges the model to produce
clean output and CONTINUES the loop, stopping gracefully only when the nudges
stop producing forward progress (same degenerate sample) or the absolute
backstop is hit. No hardcoded retry-to-death.

These tests drive ``_handle_degeneration`` / ``_degeneration_nudge`` directly
via a lightweight stub ``self`` + fake context, so no live LLM / full
orchestrator instance is needed. importorskip-guarded for node5 (dual_model
pulls the orchestrator runtime / langchain); runs in the cca-tests CI image.
"""
from __future__ import annotations

import asyncio
import types

import pytest

pytest.importorskip(
    "langchain_core",
    reason="dual_model_orchestrator imports langchain; runs in cca-tests CI image.",
)

from confucius.server.dual_model_orchestrator import (  # noqa: E402
    DualModelOrchestrator,
)


# ── Fakes ────────────────────────────────────────────────────────────


class _FakeIO:
    def __init__(self):
        self.ai_calls = []

    async def ai(self, text):
        self.ai_calls.append(text)


class _FakeMem:
    def __init__(self):
        self.added = []

    def add_messages(self, msgs):
        self.added.extend(msgs)


class _FakeCtx:
    def __init__(self):
        self.io = _FakeIO()
        self.memory_manager = _FakeMem()


class _Detection:
    def __init__(self, sample, field):
        self.buffer_sample = sample
        self.field = field


class _Terminal(Exception):
    def __init__(self, sample, field="content"):
        self.detection = _Detection(sample, field)


def _make_self(**over):
    s = types.SimpleNamespace()
    s._last_degen_sample_hash = 0
    s._consecutive_degenerations = 0
    s._degen_no_progress = 0
    s._flow_config = {}
    # Bind the real static helper so the nudge text is the production one.
    s._degeneration_nudge = DualModelOrchestrator._degeneration_nudge
    for k, v in over.items():
        setattr(s, k, v)
    return s


def _handle(s, de, ctx):
    return asyncio.run(DualModelOrchestrator._handle_degeneration(s, de, ctx))


# ── First degeneration nudges + continues (does NOT kill) ────────────


class TestNudgeAndContinue:
    def test_first_degeneration_continues(self):
        s, ctx = _make_self(), _FakeCtx()
        cont = _handle(s, _Terminal("repeat-sample"), ctx)
        assert cont is True, "first degeneration must continue, not kill the run"
        assert ctx.io.ai_calls == [], "no graceful-stop summary on a continue"
        assert len(ctx.memory_manager.added) == 1, "a corrective nudge was injected"
        assert s._consecutive_degenerations == 1
        assert s._degen_no_progress == 0  # new sample = forward progress

    def test_forward_progress_resets_no_progress(self):
        s, ctx = _make_self(), _FakeCtx()
        assert _handle(s, _Terminal("sample-A"), ctx) is True
        assert _handle(s, _Terminal("sample-B"), ctx) is True  # different → progress
        assert s._degen_no_progress == 0
        assert s._consecutive_degenerations == 2


# ── Stop ONLY when no forward progress (same sample) ─────────────────


class TestStopOnNoProgress:
    def test_same_sample_stops_at_threshold(self):
        s, ctx = _make_self(_flow_config={"degen_stop_threshold": 2}), _FakeCtx()
        assert _handle(s, _Terminal("stuck"), ctx) is True   # no_progress=0
        assert _handle(s, _Terminal("stuck"), ctx) is True   # no_progress=1
        result = _handle(s, _Terminal("stuck"), ctx)         # no_progress=2 >= 2
        assert result is False, "repeated identical degeneration must stop"
        assert len(ctx.io.ai_calls) == 1, "graceful summary emitted on stop"
        assert "trouble" in ctx.io.ai_calls[0].lower()

    def test_absolute_backstop_on_varied_degeneration(self):
        # Different sample every time → always forward progress, so the
        # no-progress gate never trips; the absolute max_nudges backstop must.
        s = _make_self(_flow_config={"degen_stop_threshold": 99, "degen_max_nudges": 4})
        ctx = _FakeCtx()
        for i in range(3):
            assert _handle(s, _Terminal(f"diff-{i}"), ctx) is True
        result = _handle(s, _Terminal("diff-final"), ctx)    # 4th → hits max_nudges
        assert result is False, "absolute backstop must stop unbounded nudging"


# ── Nudge text escalates ─────────────────────────────────────────────


class TestNudgeText:
    def test_first_nudge_is_corrective(self):
        msg = DualModelOrchestrator._degeneration_nudge(1, "content")
        assert "repeat" in msg.lower()
        assert "more than once" not in msg.lower()

    def test_repeat_nudge_demands_brevity(self):
        msg = DualModelOrchestrator._degeneration_nudge(2, "content")
        assert "more than once" in msg.lower()
        assert "brief" in msg.lower()
