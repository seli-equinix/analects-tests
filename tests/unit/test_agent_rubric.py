"""Phase 3.3.d — wrapper-math tests for the agent rubric.

We don't run the live evaluators (they need a real ChatResult, real
Phoenix, real vLLM judge). Instead we feed each registered metric a
synthetic ``FakeChatResult`` carrying just the fields the metric reads,
and assert the resulting ``MetricResult`` has the expected
``passed`` + ``value`` + ``details`` shape.

This catches:
- The wrapper passing through ``score`` / ``label`` correctly.
- The wrapper turning a ``None`` (skipped) eval into ``passed=True``.
- Threshold semantics — strict vs partial-credit.
- The new Y5/Y6/Y8 metrics' lookup logic over ``ctx`` and
  ``target.metadata`` / ``target.usage``.

Run:
    pytest tests/unit/test_agent_rubric.py -v
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

# Importing these modules registers every Y/bonus metric and constructs
# the rubrics. Side-effect imports are intentional.
from tests.agent_eval import metrics as agent_metrics  # noqa: F401
from tests.agent_eval import rubrics as agent_rubrics

from confucius.core.quality import MetricResult, get_metric


@dataclass
class FakeChatResult:
    """Minimal stand-in for `tests.cca_client.ChatResult` covering only
    the fields the metric wrappers read."""
    content: str = ""
    elapsed_ms: float = 1000.0
    raw: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"

    @property
    def tool_labels(self) -> List[str]:
        return self.raw.get("tool_labels", [])

    @property
    def tool_errors(self) -> List[str]:
        return [
            l for l in self.tool_labels
            if any(k in l.lower() for k in ("failed", "error", "invalid"))
        ]


# ── Y2 / Y3 / Y7 (existing eval_* wrappers) ──────────────────────────


class TestExistingEvalWrappers:
    def test_y2_clean_when_no_tool_errors(self):
        # eval_tool_errors returns "clean" with score=1.0 when there
        # are no errors in tool_labels — never returns None.
        r = FakeChatResult(content="hi", metadata={"tool_calls": None})
        out = get_metric("Y2.tool_errors_clean")(r)
        assert isinstance(out, MetricResult)
        assert out.passed is True
        assert dict(out.details).get("label") == "clean"

    def test_y2_clean_run(self):
        r = FakeChatResult(
            content="ok",
            metadata={
                "tool_calls": [{"name": "bash", "success": True, "iteration": 1}],
            },
        )
        out = get_metric("Y2.tool_errors_clean")(r)
        assert out.passed is True

    def test_y3_iteration_skipped_when_no_metadata(self):
        r = FakeChatResult(content="hi", metadata={})
        out = get_metric("Y3.iteration_efficiency")(r)
        # iter=0 vs steps=10 default → "efficient" → score 1.0 → passed
        assert out.passed is True

    def test_y3_iteration_looping(self):
        r = FakeChatResult(content="hi", metadata={"tool_iterations": 25, "estimated_steps": 5})
        out = get_metric("Y3.iteration_efficiency")(r)
        # 25 > 15 → "excessive" → SCORE_FAIL → 0.0 < threshold(0.5) → fails
        assert out.passed is False

    def test_y7_latency_fast(self):
        r = FakeChatResult(content="hi", elapsed_ms=5000.0)
        out = get_metric("Y7.latency_ok")(r)
        assert out.passed is True

    def test_y7_latency_timeout_fails(self):
        r = FakeChatResult(content="hi", elapsed_ms=400_000.0)  # > 5min
        out = get_metric("Y7.latency_ok")(r)
        assert out.passed is False


# ── Y1 / Y4 / Y5 — judge metrics skip cleanly without judge ─────────


class TestJudgeMetricSkip:
    def test_y1_skipped_without_judge(self):
        r = FakeChatResult(content="hi")
        out = get_metric("Y1.task_completion")(r, ctx={})
        assert out.passed is True
        assert dict(out.details).get("skipped") is True

    def test_y4_skipped_without_judge(self):
        r = FakeChatResult(content="hi")
        out = get_metric("Y4.response_quality")(r, ctx={})
        assert out.passed is True
        assert dict(out.details).get("skipped") is True

    def test_y5_skipped_without_judge(self):
        r = FakeChatResult(content="hi")
        out = get_metric("Y5.no_hallucination")(r, ctx={})
        assert out.passed is True
        assert dict(out.details).get("skipped") is True


# ── Y6 stream_guard ──────────────────────────────────────────────────


class TestY6StreamGuard:
    def test_y6_no_field_skips(self):
        # Pre-3.3.b runs don't have the field — metric must skip, not fail.
        r = FakeChatResult(content="hi", metadata={})
        out = get_metric("Y6.no_stream_guard_fire")(r)
        assert out.passed is True
        assert dict(out.details).get("skipped") is True

    def test_y6_no_fires_passes(self):
        r = FakeChatResult(
            content="hi",
            metadata={"stream_guard_fired": False, "stream_guard_fires": 0},
        )
        out = get_metric("Y6.no_stream_guard_fire")(r)
        assert out.passed is True
        assert dict(out.details) == {"fired": False, "count": 0}

    def test_y6_fired_fails(self):
        r = FakeChatResult(
            content="hi",
            metadata={"stream_guard_fired": True, "stream_guard_fires": 2},
        )
        out = get_metric("Y6.no_stream_guard_fire")(r)
        assert out.passed is False
        assert dict(out.details) == {"fired": True, "count": 2}


# ── Y8 token cost ────────────────────────────────────────────────────


class TestY8TokenCost:
    def test_y8_under_budget_passes(self):
        r = FakeChatResult(
            content="hi",
            usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        )
        out = get_metric("Y8.token_cost")(r, ctx={"route": "coder"})
        assert out.value == 1500
        assert out.passed is True

    def test_y8_over_budget_fails(self):
        r = FakeChatResult(
            content="hi",
            usage={"prompt_tokens": 6000, "completion_tokens": 5000, "total_tokens": 11000},
        )
        out = get_metric("Y8.token_cost")(r, ctx={"route": "coder"})
        assert out.value == 11000
        assert out.passed is False  # default budget = 8000

    def test_y8_falls_back_to_sum_when_total_missing(self):
        r = FakeChatResult(
            content="hi",
            usage={"prompt_tokens": 200, "completion_tokens": 300},  # no total_tokens
        )
        out = get_metric("Y8.token_cost")(r, ctx={"route": "search"})
        assert out.value == 500


# ── Bonus metrics ────────────────────────────────────────────────────


class TestBonusMetrics:
    def test_bonus_not_empty(self):
        assert get_metric("bonus.not_empty")(FakeChatResult(content="x")).passed is True
        assert get_metric("bonus.not_empty")(FakeChatResult(content="")).passed is False

    def test_bonus_no_error(self):
        assert get_metric("bonus.no_error")(FakeChatResult(content="ok", raw={})).passed is True
        assert get_metric("bonus.no_error")(
            FakeChatResult(content="bad", raw={"error": "boom"})
        ).passed is False

    def test_bonus_no_refusal(self):
        assert get_metric("bonus.no_refusal")(
            FakeChatResult(content="here you go")
        ).passed is True
        # Use a phrase that actually appears in _REFUSAL_PATTERNS:
        # "i don't have access to your" is one of the literal patterns.
        assert get_metric("bonus.no_refusal")(
            FakeChatResult(content="I don't have access to your filesystem")
        ).passed is False

    def test_bonus_coherent(self):
        assert get_metric("bonus.coherent")(
            FakeChatResult(content="A clean coherent response.")
        ).passed is True


# ── Rubric composition ──────────────────────────────────────────────


class TestRubricComposition:
    def test_coder_rubric_passes_clean_run(self):
        r = FakeChatResult(
            content="Done.",
            elapsed_ms=2000.0,
            metadata={
                "tool_iterations": 3,
                "estimated_steps": 5,
                "tool_calls": [{"name": "bash", "success": True, "iteration": 1}],
                "stream_guard_fired": False,
                "stream_guard_fires": 0,
            },
            usage={"total_tokens": 1500},
            raw={},
        )
        out = agent_rubrics.CODER.score(r, target_id="t1", ctx={"route": "coder"})
        # required = (Y1.task_completion[skipped, judge off], Y2, bonus.not_empty,
        # bonus.no_error, bonus.coherent) — all should pass
        assert out.passed is True
        # Sanity: every metric in the rubric appears in the result
        result_names = {m.name for m in out.metric_results}
        assert set(agent_rubrics.CODER.metrics) == result_names

    def test_search_rubric_fails_on_refusal(self):
        # Search rubric requires bonus.no_refusal — synthesise a refusal
        # using a literal _REFUSAL_PATTERNS entry.
        r = FakeChatResult(
            content="I don't have access to your data store, so I cannot answer.",
            elapsed_ms=2000.0,
            metadata={"stream_guard_fired": False},
            usage={"total_tokens": 200},
        )
        out = agent_rubrics.SEARCH.score(r, target_id="r1", ctx={"route": "search"})
        assert out.passed is False
        refusal = out.metric("bonus.no_refusal")
        assert refusal is not None and refusal.passed is False

    def test_rubric_for_route_unknown_falls_back_to_coder(self):
        assert agent_rubrics.rubric_for_route("nonexistent") is agent_rubrics.CODER


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
