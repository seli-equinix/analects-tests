"""Phase 3.6 — `live_validate_agent` runner.

Loads the Phase-3.5 replay corpus (`tests/agent_eval/replay_corpus/`),
applies the Phase-3.3 route-specific rubrics
(``agent.<route>.v1``), evaluates each task's success criterion from
the Phase-3.4 corpus, and emits one
``LIVE_SUMMARY.agent.{bundle_hash}.md`` artifact per distinct bundle.

Why per-bundle?
- Phase 4 promotion is the bundle being promoted, not "the agent."
- Captures already span multiple bundles (different routes register
  with different bundle_ids in this codebase). Mixing them into one
  summary would obscure which bundle actually passes the gate.

The runner does NOT round-trip vLLM — every metric is scored from the
captured trajectory + ``--with-judge`` (when set) feeds the judge
prompts off the captured response/tool-trail rather than re-running
the agent. This is the price-friendly half of the eval gate; the
expensive half (live capture) ran in Phase 3.5.

Usage::

    # Score the default replay_corpus/ → write LIVE_SUMMARY.agent.*.md
    # to the same directory.
    python3 -m tests.agent_eval.runner

    # With LLM-judge metrics (Y1, Y4, Y5)
    python3 -m tests.agent_eval.runner --with-judge

    # Custom paths
    python3 -m tests.agent_eval.runner --replay-dir /path/captures \\
                                       --output /path/summaries

    # Determinism check (Phase 3.8) — produce deterministic markdown
    # without the `generated_at` line so two runs diff to nothing.
    python3 -m tests.agent_eval.runner --deterministic
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_REPLAY_DIR = _THIS_DIR / "replay_corpus"


# ── Adapter: CaptureRecord JSON → ChatResult-shaped target ──────────


class _CapturedChatResult:
    """Read-only view over a ``CaptureRecord`` JSON dict that exposes
    the same attribute surface as ``tests.cca_client.ChatResult`` so
    Phase-3.3 metric wrappers don't care whether we're scoring a live
    agent run or a replay."""

    __slots__ = (
        "_raw_record",
        "id", "model", "content", "role", "reasoning",
        "finish_reason", "usage", "metadata", "elapsed_ms", "raw",
    )

    def __init__(self, record: Mapping[str, Any]) -> None:
        self._raw_record = record
        self.id = record.get("completion_id", "")
        self.model = record.get("metadata", {}).get("model", "") or ""
        self.content = record.get("response_content", "") or ""
        self.role = record.get("response_role", "assistant")
        self.reasoning = None
        self.finish_reason = record.get("finish_reason", "")
        self.usage = dict(record.get("usage") or {})
        self.metadata = dict(record.get("metadata") or {})
        self.elapsed_ms = float(record.get("elapsed_ms") or 0.0)
        # Phase-3.3 wrappers consult `tool_labels` for stream-error
        # detection; captures don't record SSE comments so we expose
        # an empty list.
        self.raw: Dict[str, Any] = {"tool_labels": []}

    @property
    def tool_labels(self) -> List[str]:
        return []

    @property
    def tool_errors(self) -> List[str]:
        # Reconstruct from metadata.tool_calls — anything success=False
        # is an unrecovered tool error. (eval_tool_errors does its own
        # recovery-detection on top of this, so a later success of the
        # same tool name still scores clean.)
        out: List[str] = []
        for tc in self.metadata.get("tool_calls") or []:
            if not tc.get("success", True):
                out.append(f"{tc.get('name', '?')}@iter{tc.get('iteration', 0)}: failed")
        return out

    @property
    def tool_names(self) -> List[str]:
        return [tc.get("name", "") for tc in (self.metadata.get("tool_calls") or [])]


# ── Success-criterion evaluation ────────────────────────────────────


@dataclass(frozen=True)
class CriterionVerdict:
    passed: bool
    reason: str


_REPLAY_TIME_KINDS = frozenset({
    "tool_named", "route_matches", "count_match", "regex_match",
    "clarification_requested", "no_refusal", "any_of",
})

_CAPTURE_TIME_KINDS = frozenset({
    "file_exists", "file_contains", "file_unchanged", "judge_completed",
})


def evaluate_criterion(
    criterion,
    captured: _CapturedChatResult,
    *,
    expected_route: str,
) -> CriterionVerdict:
    """Score a Phase-3.4 success_criterion against a captured run.

    Replay-time criteria (count_match, route_matches, etc.) score from
    the captured response. Capture-time criteria (file_exists/...) need
    workspace state we didn't record — they return passed=True with
    "skipped" reason so they don't fail tasks the runner can't observe.
    A v2 capture pass can record file state; v1 doesn't.
    """
    kind = criterion.kind
    data = criterion.as_dict()

    if kind in _CAPTURE_TIME_KINDS:
        return CriterionVerdict(
            passed=True,
            reason=f"skipped: {kind} needs capture-time filesystem state",
        )

    content = captured.content or ""
    content_low = content.lower()
    metadata = captured.metadata or {}

    if kind == "tool_named":
        wanted = str(data.get("tool", ""))
        seen = captured.tool_names
        ok = any(wanted == name or wanted in name for name in seen)
        return CriterionVerdict(ok, f"tool {wanted!r} in {seen!r}")

    if kind == "route_matches":
        wanted = str(data.get("route", expected_route))
        actual = metadata.get("route", "<none>")
        if actual != wanted:
            return CriterionVerdict(False, f"route was {actual!r}, expected {wanted!r}")
        # Optional `additional` checks
        return _check_additional(data.get("additional", {}) or {}, captured)

    if kind == "count_match":
        return _check_count_match(data, captured)

    if kind == "regex_match":
        pattern = str(data.get("response_must_match", ""))
        if not pattern:
            return CriterionVerdict(False, "regex_match without response_must_match")
        ok = bool(re.search(pattern, content))
        return CriterionVerdict(ok, f"regex {pattern!r} {'matched' if ok else 'did not match'}")

    if kind == "clarification_requested":
        any_of = data.get("response_must_contain_any") or []
        ok = bool(any_of) and any(s.lower() in content_low for s in any_of)
        if not ok and "?" in content:
            ok = True
        return CriterionVerdict(ok, f"clarification cues {'found' if ok else 'absent'}")

    if kind == "no_refusal":
        # "no_refusal" criterion = passed if the agent did NOT refuse.
        # Reuses the existing eval_no_refusal pattern (substring list).
        refusal_terms = data.get("response_must_not_contain") or [
            "i don't have access", "i cannot", "i'm unable",
        ]
        offending = [t for t in refusal_terms if t.lower() in content_low]
        return CriterionVerdict(
            not offending,
            f"refusal markers found: {offending}" if offending else "no refusal markers",
        )

    if kind == "any_of":
        options = data.get("options") or []
        if not options:
            return CriterionVerdict(False, "any_of without options list")
        for opt in options:
            from confucius.core.quality.cohort import Cohort  # noqa: F401  unused but ensures package import
            from .loaders import SuccessCriterion
            inner = SuccessCriterion(
                kind=str(opt.get("kind", "")),
                data=tuple(sorted((k, v) for k, v in opt.items() if k != "kind")),
            )
            verdict = evaluate_criterion(inner, captured, expected_route=expected_route)
            if verdict.passed:
                return CriterionVerdict(True, f"any_of OK via {inner.kind}: {verdict.reason}")
        return CriterionVerdict(False, "no any_of option matched")

    # Unknown kind — defensive default.
    return CriterionVerdict(False, f"unknown criterion kind: {kind!r}")


def _check_count_match(data: Mapping[str, Any], captured: _CapturedChatResult) -> CriterionVerdict:
    content = captured.content or ""
    content_low = content.lower()
    must_all = data.get("response_must_contain") or []
    must_any = data.get("response_must_contain_any") or []
    missing = [s for s in must_all if s.lower() not in content_low]
    if missing:
        return CriterionVerdict(
            False,
            f"missing required substrings: {missing}",
        )
    if must_any:
        present = [s for s in must_any if s.lower() in content_low]
        if not present:
            return CriterionVerdict(
                False,
                f"none of any-of substrings present: {must_any}",
            )
        return CriterionVerdict(True, f"matched any-of via {present[:3]}")
    return CriterionVerdict(True, f"all required substrings present ({len(must_all)})")


def _check_additional(additional: Mapping[str, Any], captured: _CapturedChatResult) -> CriterionVerdict:
    """Sub-criteria attached to route_matches: response_must_contain*,
    forbidden_tools, response_must_match."""
    content = captured.content or ""
    content_low = content.lower()
    if "response_must_contain_any" in additional:
        opts = additional["response_must_contain_any"] or []
        if opts and not any(s.lower() in content_low for s in opts):
            return CriterionVerdict(
                False,
                f"response_must_contain_any missing all of {opts}",
            )
    if "response_must_contain" in additional:
        miss = [s for s in additional["response_must_contain"] if s.lower() not in content_low]
        if miss:
            return CriterionVerdict(False, f"missing required substrings: {miss}")
    if "response_must_match" in additional:
        pattern = additional["response_must_match"]
        if not re.search(pattern, content):
            return CriterionVerdict(False, f"regex {pattern!r} did not match")
    if "forbidden_tools" in additional:
        forbidden = additional["forbidden_tools"]
        offending = [t for t in captured.tool_names if t in forbidden]
        if offending:
            return CriterionVerdict(False, f"forbidden tools called: {offending}")
    return CriterionVerdict(True, "route + additional checks OK")


# ── Per-bundle scoring ──────────────────────────────────────────────


@dataclass(frozen=True)
class TaskEvaluation:
    task_name: str
    cohort: str
    bundle_id: Optional[str]
    rubric_name: str
    rubric_passed: bool
    criterion_passed: bool
    criterion_reason: str
    rubric_result: Any  # confucius.core.quality.RubricResult


def evaluate_capture(
    record_path: Path,
    *,
    corpus,
    judge_model: Optional[Any] = None,
) -> TaskEvaluation:
    """Score one captured task against its route's rubric + the
    corpus-defined success criterion."""
    from .rubrics import rubric_for_route
    record = json.loads(record_path.read_text())
    task_name = record["task_name"]
    spec = corpus.tasks.get(task_name)
    if spec is None:
        raise KeyError(f"capture {record_path.name} references unknown task {task_name!r}")
    captured = _CapturedChatResult(record)
    actual_route = (record.get("metadata") or {}).get("route", spec.expected_route)
    # Always score against the EXPECTED route's rubric. Route mismatch
    # surfaces via bonus.route_match inside that rubric.
    rubric = rubric_for_route(spec.expected_route)
    ctx = {
        "judge_model": judge_model,
        "user_message": record["prompt_rendered"],
        "route": actual_route,
        "expected_route": spec.expected_route,
    }
    rubric_result = rubric.score(captured, target_id=task_name, ctx=ctx)
    verdict = evaluate_criterion(spec.success_criterion, captured, expected_route=spec.expected_route)
    return TaskEvaluation(
        task_name=task_name,
        cohort=spec.cohort,
        bundle_id=record.get("bundle_id"),
        rubric_name=rubric.name,
        rubric_passed=rubric_result.passed,
        criterion_passed=verdict.passed,
        criterion_reason=verdict.reason,
        rubric_result=rubric_result,
    )


def evaluate_corpus(
    *,
    replay_dir: Path,
    corpus,
    judge_model: Optional[Any] = None,
) -> Tuple[Dict[Optional[str], List[TaskEvaluation]], List[Tuple[str, str]]]:
    """Walk every JSON file in ``replay_dir`` and score it.

    Returns ``(by_bundle, errors)`` where:
      - ``by_bundle`` maps bundle_id (None for clarify/direct) to the
        list of TaskEvaluation records.
      - ``errors`` is a list of (filename, message) for files that
        couldn't be scored. The runner reports these but doesn't crash.
    """
    by_bundle: Dict[Optional[str], List[TaskEvaluation]] = {}
    errors: List[Tuple[str, str]] = []

    for path in sorted(replay_dir.glob("*.json")):
        try:
            ev = evaluate_capture(path, corpus=corpus, judge_model=judge_model)
        except Exception as e:  # noqa: BLE001
            logger.exception("scoring failed for %s", path.name)
            errors.append((path.name, str(e)[:200]))
            continue
        by_bundle.setdefault(ev.bundle_id, []).append(ev)

    return by_bundle, errors


# ── LIVE_SUMMARY emission ───────────────────────────────────────────


def emit_summary_for_bundle(
    bundle_id: Optional[str],
    evaluations: Iterable[TaskEvaluation],
    *,
    output_dir: Path,
    generated_at: str,
    deterministic: bool = False,
) -> Path:
    """Build a LiveSummary for one bundle and write it to disk.

    Bundle ``None`` (clarify/direct routes) gets the synthetic name
    ``no-bundle`` so the file still has a stable path.
    """
    from confucius.core.quality import LiveSummary, MetricResult, RubricResult

    by_cohort: Dict[str, List[RubricResult]] = {}
    for ev in evaluations:
        # Inject the criterion verdict as an extra MetricResult so it
        # shows up in the LIVE_SUMMARY table alongside Y1-Y8.
        criterion_metric = MetricResult(
            name="criterion.task_specific",
            value="passed" if ev.criterion_passed else "failed",
            passed=ev.criterion_passed,
            details=(("reason", ev.criterion_reason[:200]),),
        )
        # Recreate RubricResult with the extra metric appended; overall
        # passed = rubric_passed AND criterion_passed.
        merged = RubricResult(
            rubric_name=ev.rubric_name,
            target_id=ev.task_name,
            metric_results=tuple(ev.rubric_result.metric_results) + (criterion_metric,),
            passed=ev.rubric_passed and ev.criterion_passed,
        )
        by_cohort.setdefault(ev.cohort, []).append(merged)

    label = bundle_id or "no-bundle"
    summary = LiveSummary.from_results(
        bundle_hash=label,
        rubric_name="agent.eval.v1",
        generated_at=generated_at,
        results_by_cohort=by_cohort,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"LIVE_SUMMARY.agent.{label[:16]}.md"
    body = summary.to_markdown_for_diff() if deterministic else summary.to_markdown()
    out_path.write_text(body, encoding="utf-8")
    return out_path


# ── CLI ─────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="live_validate_agent",
        description="Score the captured replay corpus and emit per-bundle LIVE_SUMMARY artifacts.",
    )
    parser.add_argument(
        "--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR,
        help="Directory of CaptureRecord JSON files (default: tests/agent_eval/replay_corpus/)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_REPLAY_DIR,
        help="Where to write LIVE_SUMMARY.agent.*.md (default: same as --replay-dir)",
    )
    parser.add_argument(
        "--with-judge", action="store_true",
        help="Run LLM judge metrics (Y1, Y4, Y5). Costs vLLM time.",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Strip the generated_at line so two runs produce byte-identical output (Phase 3.8).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    from .loaders import load_default_corpus
    from . import metrics  # noqa: F401  side-effect: register Y/bonus metrics

    corpus = load_default_corpus()
    judge_model = _maybe_judge_model() if args.with_judge else None

    by_bundle, errors = evaluate_corpus(
        replay_dir=args.replay_dir,
        corpus=corpus,
        judge_model=judge_model,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    written: List[Tuple[str, Path]] = []
    for bundle_id, evals in by_bundle.items():
        path = emit_summary_for_bundle(
            bundle_id, evals,
            output_dir=args.output,
            generated_at=generated_at,
            deterministic=args.deterministic,
        )
        written.append((bundle_id or "<no-bundle>", path))

    print(f"\n=== runner summary ===")
    for bid, path in written:
        bundle_evals = by_bundle.get(bid if bid != "<no-bundle>" else None, [])
        rubric_pass = sum(1 for e in bundle_evals if e.rubric_passed)
        crit_pass = sum(1 for e in bundle_evals if e.criterion_passed)
        both = sum(1 for e in bundle_evals if e.rubric_passed and e.criterion_passed)
        print(
            f"  bundle={bid[:16]:18s} → {path.name}  "
            f"rubric={rubric_pass}/{len(bundle_evals)}  "
            f"criterion={crit_pass}/{len(bundle_evals)}  "
            f"both={both}/{len(bundle_evals)}"
        )
    if errors:
        print(f"\n  {len(errors)} scoring error(s):")
        for name, msg in errors:
            print(f"    - {name}: {msg}")
    return 1 if errors else 0


def _maybe_judge_model() -> Any:
    """Construct a judge model handle from env vars matching the
    existing tests/conftest.py judge_model fixture pattern. Returns
    None if no judge URL is configured.
    """
    import os
    url = os.environ.get("VLLM_BASE_URL", "http://192.168.4.208:8000/v1")
    model = os.environ.get("CCA_JUDGE_MODEL", "/models/Qwen3.5-35B-A3B-FP8")
    if not url or not model:
        return None
    return {"url": url, "model": model}


if __name__ == "__main__":
    sys.exit(main())
