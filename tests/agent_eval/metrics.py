"""Phase 3.3 — adapter layer between ``tests/evaluators.py`` and the
Phase-3.2 ``confucius.core.quality`` framework primitives.

For every existing evaluator function in ``tests/evaluators.py``, we
register a ``Metric`` (via ``@register_metric``) that wraps the
evaluator's dict output into a ``MetricResult``. The wrapped function
is named with the canonical ``Y<n>.<short>`` slug from the master
roadmap so rubrics can reference it cleanly.

Existing test paths keep calling ``evaluate_response()`` (queue/Phoenix
flow) unchanged. The new path is for ``live_validate_agent.py``
(Phase 3.6) which iterates the same evaluators against a cohort and
builds a ``LiveSummary`` artifact.

Score contract for adapter wrappers:

    raw eval dict → MetricResult
    {"score": 1.0, "label": "pass", ...} → MetricResult(passed=score >= threshold)

    None (evaluator skipped) → MetricResult(value=None, passed=True,
                                            details={"skipped": True})
    A skipped metric does NOT count as a failure — the rubric's
    `required` set decides what's load-bearing.

Y5/Y6/Y8 are NEW metrics that didn't exist in ``tests/evaluators.py``.
They live here so 3.3 has the full Y1–Y8 surface in one place; if a
metric grows beyond ~30 lines or needs its own state it'll graduate
into ``tests/evaluators.py`` alongside the others.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from confucius.core.quality import MetricResult, register_metric


# Threshold above which a numeric score counts as "passed". The existing
# evaluators use 1.0 = full pass, 0.5 = partial, 0.0 = fail. We keep
# Y1/Y2/etc. strict (>= 1.0) by default; rubric authors can override
# at construction time if they want partial-credit thresholds.
_PASS_THRESHOLD = 1.0


def _wrap_eval_dict(
    metric_name: str,
    raw: Optional[Dict[str, Any]],
    threshold: float = _PASS_THRESHOLD,
) -> MetricResult:
    """Convert one ``eval_*`` return value into a ``MetricResult``."""
    if raw is None:
        return MetricResult(
            name=metric_name,
            value=None,
            threshold=threshold,
            passed=True,
            details=(("skipped", True),),
        )
    score = raw.get("score")
    label = raw.get("label", "")
    explanation = raw.get("explanation", "")
    details: list[tuple[str, Any]] = []
    if label:
        details.append(("label", label))
    if explanation:
        details.append(("explanation", explanation[:300]))
    if isinstance(score, (int, float)):
        passed = float(score) >= float(threshold)
    elif isinstance(score, bool):
        passed = score
    else:
        passed = True
    return MetricResult(
        name=metric_name,
        value=score,
        threshold=threshold,
        passed=passed,
        details=tuple(details),
    )


# ── Y1: task completion (LLM judge — opt-in) ────────────────────────


@register_metric("Y1.task_completion")
def y1_task_completion(target, ctx=None):  # noqa: D401
    """Wrap ``eval_task_completion``. Skipped if no judge is wired."""
    from tests.evaluators import eval_task_completion

    judge = (ctx or {}).get("judge_model")
    if judge is None:
        return MetricResult(
            name="Y1.task_completion",
            value=None,
            passed=True,
            details=(("skipped", True), ("reason", "no judge configured")),
        )
    user_message = (ctx or {}).get("user_message", "")
    raw = eval_task_completion(target, user_message, judge)
    return _wrap_eval_dict("Y1.task_completion", raw)


# ── Y2: tool usage correctness ──────────────────────────────────────


@register_metric("Y2.tool_errors_clean")
def y2_tool_errors(target, ctx=None):
    """Wrap ``eval_tool_errors``. Returns "pass" when no unrecovered
    tool errors were observed in this run."""
    from tests.evaluators import eval_tool_errors

    raw = eval_tool_errors(target)
    return _wrap_eval_dict("Y2.tool_errors_clean", raw)


# ── Y3: iteration efficiency ────────────────────────────────────────


@register_metric("Y3.iteration_efficiency")
def y3_iteration_efficiency(target, ctx=None):
    """Wrap ``eval_iteration_efficiency``. Allow PARTIAL scores
    (0.5) to count as passing — the strict bar is "did NOT loop"."""
    from tests.evaluators import eval_iteration_efficiency

    raw = eval_iteration_efficiency(target)
    return _wrap_eval_dict("Y3.iteration_efficiency", raw, threshold=0.5)


# ── Y4: response quality (LLM judge — opt-in) ───────────────────────


@register_metric("Y4.response_quality")
def y4_response_quality(target, ctx=None):
    """Wrap ``eval_response_quality``. Skipped if no judge is wired."""
    from tests.evaluators import eval_response_quality

    judge = (ctx or {}).get("judge_model")
    if judge is None:
        return MetricResult(
            name="Y4.response_quality",
            value=None,
            passed=True,
            details=(("skipped", True), ("reason", "no judge configured")),
        )
    user_message = (ctx or {}).get("user_message", "")
    raw = eval_response_quality(target, user_message, judge)
    return _wrap_eval_dict("Y4.response_quality", raw, threshold=0.5)


# ── Y5: hallucination ───────────────────────────────────────────────


_Y5_PROMPT = """\
You are scoring whether the assistant's final response invents information
that is NOT supported by the user's prompt or by the assistant's own tool
output trail. Anything explicitly returned by a tool (file content, search
result, command output, knowledge-base lookup) counts as supported, even
if the user couldn't have known it themselves.

Output exactly one of these labels on its own line, with no other text:
- supported    — every factual claim is sourced from the prompt or a tool result
- partial      — most claims are sourced; one or two are unsupported
- hallucinated — major claims are unsourced or contradict tool output

Conversation transcript follows. Score it.

USER MESSAGE:
{user_message}

ASSISTANT RESPONSE:
{response}

TOOL CALL TRAIL (compact):
{tool_trail}
"""


@register_metric("Y5.no_hallucination")
def y5_no_hallucination(target, ctx=None):
    """LLM-judged hallucination check. Skipped if no judge is wired
    (consistent with Y1/Y4 — the judge metrics are opt-in via
    ``--with-judge`` / ``CCA_RUN_JUDGE=1``)."""
    judge = (ctx or {}).get("judge_model")
    if judge is None:
        return MetricResult(
            name="Y5.no_hallucination",
            value=None,
            passed=True,
            details=(("skipped", True), ("reason", "no judge configured")),
        )
    from tests.evaluators import _run_llm_classify  # type: ignore[attr-defined]

    user_message = (ctx or {}).get("user_message", "") or ""
    response = getattr(target, "content", "") or ""
    tool_calls = (
        getattr(target, "metadata", {}) or {}
    ).get("tool_calls") or []
    tool_trail = ", ".join(
        f"{tc.get('name', '?')}({'ok' if tc.get('success') else 'err'})"
        for tc in tool_calls
    ) or "(no tools called)"
    prompt = _Y5_PROMPT.format(
        user_message=user_message[:2000],
        response=response[:4000],
        tool_trail=tool_trail[:1000],
    )
    rails = ["supported", "partial", "hallucinated"]
    label = _run_llm_classify(prompt, rails, judge)
    score_map = {"supported": 1.0, "partial": 0.5, "hallucinated": 0.0}
    score = score_map.get(label, 0.0)
    return MetricResult(
        name="Y5.no_hallucination",
        value=score,
        threshold=0.5,
        passed=score >= 0.5,
        details=(("label", label or "unknown"),),
    )


# ── Y6: stream_guard fire rate ──────────────────────────────────────


@register_metric("Y6.no_stream_guard_fire")
def y6_no_stream_guard_fire(target, ctx=None):
    """Read ``metadata.stream_guard_fired`` (orchestrator-populated) or
    fall back to checking the raw ChatCompletion response for guard
    fingerprints. ``passed=True`` iff the run had zero stream_guard
    fires.

    The metadata field is added by Phase 3.3.b; pre-3.3.b runs return
    a skipped MetricResult instead of failing.
    """
    metadata = getattr(target, "metadata", {}) or {}
    if "stream_guard_fired" not in metadata and "stream_guard_fires" not in metadata:
        return MetricResult(
            name="Y6.no_stream_guard_fire",
            value=None,
            passed=True,
            details=(("skipped", True), ("reason", "metadata field absent (pre-3.3.b run)")),
        )
    fired = bool(metadata.get("stream_guard_fired", False))
    fires = int(metadata.get("stream_guard_fires", 0) or 0)
    return MetricResult(
        name="Y6.no_stream_guard_fire",
        value=fires,
        threshold=0,
        passed=(not fired and fires == 0),
        details=(("fired", fired), ("count", fires)),
    )


# ── Y7: latency ─────────────────────────────────────────────────────


@register_metric("Y7.latency_ok")
def y7_latency(target, ctx=None):
    """Wrap ``eval_latency``. Allow PARTIAL ("slow") to pass — only
    ``timeout`` is a hard fail. Aggregate p50/p95 is computed by
    ``LiveSummary`` later from the per-target values."""
    from tests.evaluators import eval_latency

    raw = eval_latency(target)
    return _wrap_eval_dict("Y7.latency_ok", raw, threshold=0.5)


# ── Y8: cost (tokens) ───────────────────────────────────────────────


@register_metric("Y8.token_cost")
def y8_token_cost(target, ctx=None):
    """Score against a per-route token budget read from runtime_config
    (`agent_eval.token_budget.<route>`). Default 8000 tokens.

    The metric value is the total tokens consumed; ``passed`` is
    ``total <= budget``."""
    usage = getattr(target, "usage", {}) or {}
    total = int(
        usage.get("total_tokens")
        or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
    )
    route = (ctx or {}).get("route", "coder")
    budget = _token_budget_for(route)
    return MetricResult(
        name="Y8.token_cost",
        value=total,
        threshold=budget,
        passed=(total <= budget),
        details=(("route", route), ("budget", budget)),
    )


def _token_budget_for(route: str) -> int:
    """Lookup the per-route token budget from runtime_config; fall
    back to a generous default (8000) so a missing knob never fails
    a run."""
    try:
        from confucius.server.runtime_config import get as rc_get
        return int(rc_get("agent_eval", f"token_budget.{route}", default=8000))
    except Exception:  # noqa: BLE001
        return 8000


# ── Bonus metrics (already in tests/evaluators.py — exposed for use
# in rubrics that want a richer-than-Y1..Y8 contract) ───────────────


@register_metric("bonus.not_empty")
def bonus_not_empty(target, ctx=None):
    from tests.evaluators import eval_not_empty
    return _wrap_eval_dict("bonus.not_empty", eval_not_empty(target))


@register_metric("bonus.no_error")
def bonus_no_error(target, ctx=None):
    from tests.evaluators import eval_no_error
    return _wrap_eval_dict("bonus.no_error", eval_no_error(target))


@register_metric("bonus.no_refusal")
def bonus_no_refusal(target, ctx=None):
    from tests.evaluators import eval_no_refusal
    return _wrap_eval_dict("bonus.no_refusal", eval_no_refusal(target))


@register_metric("bonus.coherent")
def bonus_coherent(target, ctx=None):
    from tests.evaluators import eval_coherence
    return _wrap_eval_dict("bonus.coherent", eval_coherence(target))


@register_metric("bonus.user_identified")
def bonus_user_identified(target, ctx=None):
    from tests.evaluators import eval_user_identified
    return _wrap_eval_dict("bonus.user_identified", eval_user_identified(target))


@register_metric("bonus.code_present")
def bonus_code_present(target, ctx=None):
    from tests.evaluators import eval_code_present
    return _wrap_eval_dict("bonus.code_present", eval_code_present(target))


@register_metric("bonus.no_self_repetition")
def bonus_no_self_repetition(target, ctx=None):
    from tests.evaluators import eval_response_duplication
    # Allow partial (0.5) to pass; only the worst tier (0.0) fails.
    return _wrap_eval_dict(
        "bonus.no_self_repetition",
        eval_response_duplication(target),
        threshold=0.5,
    )


# Public list of all registered names — useful for diagnostics + tests.
ALL_METRIC_NAMES: tuple[str, ...] = (
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
    "bonus.user_identified",
    "bonus.code_present",
    "bonus.no_self_repetition",
)
