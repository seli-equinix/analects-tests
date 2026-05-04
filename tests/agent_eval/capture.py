"""Phase 3.5 — replay-corpus capture.

For each task in the Phase-3.4 corpus:
  1. Generate a fresh uid (8-char uuid4 hex).
  2. Pre-seed workspace state if the task requires it.
  3. Send the rendered prompt to the live cca instance with seed=42
     so the run is deterministic per the Phase 3.1 spike.
  4. Record the trajectory (final response + context_metadata + usage
     + bundle_id + elapsed_ms) to ``replay_corpus/<task_name>.json``.
  5. Clean up workspace files used by mutating tasks.

Captured records become the input to Phase 3.6's
``live_validate_agent.py``, which scores them against the
``agent.<route>.v1`` rubrics without re-running the live agent.

Usage::

    # All 30 tasks
    python3 -m tests.agent_eval.capture --all

    # One task
    python3 -m tests.agent_eval.capture --task simple_create_python

    # Re-capture a cohort
    python3 -m tests.agent_eval.capture --cohort code_edit_simple

    # Custom output dir
    python3 -m tests.agent_eval.capture --all --output /tmp/captures

The script never writes to the cca container DB — it only reads through
``/v1/chat/completions`` and writes JSON to the local repo. Re-running
overwrites the previous capture for that task.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_REPLAY_DIR = _THIS_DIR / "replay_corpus"


# ── Capture record ──────────────────────────────────────────────────


@dataclass(frozen=True)
class CaptureRecord:
    """One captured run of a task. Serialised to JSON via to_dict()."""
    task_name: str
    cohort: str
    uid: str
    captured_at: str  # ISO-8601 UTC
    bundle_id: Optional[str]
    session_id: str
    prompt_rendered: str
    response_content: str
    response_role: str
    finish_reason: str
    metadata: Dict[str, Any]
    usage: Dict[str, int]
    elapsed_ms: float
    cca_url: str
    completion_id: str

    def to_dict(self) -> Dict[str, Any]:
        # asdict() loses the explicit ordering — render manually so the
        # JSON output is stable.
        return {
            "task_name": self.task_name,
            "cohort": self.cohort,
            "uid": self.uid,
            "captured_at": self.captured_at,
            "bundle_id": self.bundle_id,
            "session_id": self.session_id,
            "prompt_rendered": self.prompt_rendered,
            "response_content": self.response_content,
            "response_role": self.response_role,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata,
            "usage": self.usage,
            "elapsed_ms": self.elapsed_ms,
            "cca_url": self.cca_url,
            "completion_id": self.completion_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)


# ── Capture flow ────────────────────────────────────────────────────


def capture_task(
    task,  # TaskSpec
    cca,  # CCAClient
    *,
    output_dir: Path,
    uid: Optional[str] = None,
    keep_workspace: bool = False,
) -> CaptureRecord:
    """Capture one task's trajectory to JSON. Returns the record."""
    from .preseed import cleanup_uid, run_preseed

    if uid is None:
        uid = uuid.uuid4().hex[:8]
    rendered = task.render(uid)
    session_id = f"agent-eval-{task.name}-{uid}"
    captured_at = datetime.now(timezone.utc).isoformat()

    logger.info("capturing task=%s uid=%s session=%s", task.name, uid, session_id)

    # 1. Pre-seed workspace state if required.
    run_preseed(task.name, uid)

    # 2. Send the prompt.
    t0 = time.time()
    result = cca.chat(rendered, session_id=session_id)
    elapsed_ms = (time.time() - t0) * 1000.0

    # 3. Build the record.
    raw = result.raw or {}
    record = CaptureRecord(
        task_name=task.name,
        cohort=task.cohort,
        uid=uid,
        captured_at=captured_at,
        bundle_id=(result.metadata or {}).get("bundle_id"),
        session_id=session_id,
        prompt_rendered=rendered,
        response_content=result.content,
        response_role=result.role,
        finish_reason=result.finish_reason,
        metadata=dict(result.metadata or {}),
        usage=dict(result.usage or {}),
        elapsed_ms=elapsed_ms,
        cca_url=cca.base_url,
        completion_id=str(raw.get("id", "")),
    )

    # 4. Save.
    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task.name}.json"
    out_path.write_text(record.to_json() + "\n", encoding="utf-8")
    logger.info("wrote %s (bundle=%s, %.0fms, %d tokens)",
                out_path.name,
                (record.bundle_id or "<none>")[:16],
                record.elapsed_ms,
                record.usage.get("total_tokens", 0))

    # 5. Cleanup mutating tasks.
    if task.mutating and not keep_workspace:
        cleanup_uid(uid)

    return record


# ── CLI ─────────────────────────────────────────────────────────────


def _filter_tasks(corpus, args) -> List:
    if args.all:
        return list(corpus.tasks.values())
    if args.task:
        if args.task not in corpus.tasks:
            raise SystemExit(f"unknown task: {args.task!r}")
        return [corpus.tasks[args.task]]
    if args.cohort:
        if args.cohort not in corpus.cohorts:
            raise SystemExit(f"unknown cohort: {args.cohort!r}")
        return list(corpus.tasks_in_cohort(args.cohort))
    raise SystemExit("must pass --all, --task <name>, or --cohort <name>")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="capture",
        description="Capture replay corpus for the Phase 3.4 agent eval tasks.",
    )
    parser.add_argument("--all", action="store_true", help="capture every task")
    parser.add_argument("--task", type=str, help="capture one task by name")
    parser.add_argument("--cohort", type=str, help="capture all tasks in a cohort")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_REPLAY_DIR,
        help="output directory for replay JSON (default: tests/agent_eval/replay_corpus/)",
    )
    parser.add_argument(
        "--keep-workspace", action="store_true",
        help="don't clean up workspace files after mutating tasks (debug only)",
    )
    parser.add_argument(
        "--cca-url", type=str, default=None,
        help="override CCA base URL (default: $CCA_BASE_URL or https://192.168.4.205:8500)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    # Lazy imports — keep module import-time light
    from tests.agent_eval.loaders import load_default_corpus
    from tests.cca_client import CCAClient

    corpus = load_default_corpus()
    tasks = _filter_tasks(corpus, args)

    cca_url = args.cca_url or None
    cca = CCAClient(base_url=cca_url) if cca_url else CCAClient()
    health = cca.health()
    if health.get("status") != "healthy":
        raise SystemExit(f"cca server unreachable: {health}")

    successes: List[str] = []
    failures: List[Tuple[str, str]] = []
    for task in tasks:
        try:
            capture_task(task, cca, output_dir=args.output)
            successes.append(task.name)
        except Exception as e:  # noqa: BLE001
            logger.exception("capture failed for %s", task.name)
            failures.append((task.name, str(e)[:200]))

    print(f"\n=== capture summary ===")
    print(f"  succeeded: {len(successes)}/{len(tasks)}")
    if failures:
        print(f"  failed:    {len(failures)}")
        for name, err in failures:
            print(f"    - {name}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
