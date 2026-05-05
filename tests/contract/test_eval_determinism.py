"""Phase 3.8 — byte-identical determinism contract.

Runs the agent eval runner twice in ``--deterministic`` mode against
the same captured replay corpus and asserts every emitted
``LIVE_SUMMARY.agent.{bundle_hash}.md`` is byte-identical across
runs.

Why: re-scoring is supposed to be a pure function of (corpus, captures,
metric registry, thresholds). Drift between runs would mean a metric
calculation, a JSON sort order, or a dict-iteration is sneaking
non-determinism into the artifact — which would break the Phase-4
promotion contract (a bundle is identified by the artifact hash;
a flapping artifact has no stable identity).

Run:
    pytest tests/contract/test_eval_determinism.py -v

CI: ``eval-determinism`` job (allow_failure: false). Bypassable with
the existing ``[skip-contract]`` token.

Failure mode: when this fails, look for:
  - A metric returning a non-deterministic value (e.g., timestamp,
    random sample, hash of object id).
  - JSON serialization without ``sort_keys=True``.
  - Iteration over a dict whose insertion order varies.
  - Reading a file whose mtime / content changes between runs.
  - A judge metric that wasn't skipped (judge calls add randomness;
    the runner must skip judges in --deterministic mode unless
    --with-judge is also set).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.agent_eval import metrics as agent_metrics
from tests.agent_eval.runner import DEFAULT_REPLAY_DIR


@pytest.fixture(autouse=True)
def _restore_agent_registry():
    """Rebuild the Phase-3.2 metric registry — quality_eval tests
    that ran earlier in the session may have wiped it."""
    agent_metrics.ensure_registered()
    yield


@pytest.fixture(scope="module")
def replay_dir() -> Path:
    if not DEFAULT_REPLAY_DIR.exists() or not list(DEFAULT_REPLAY_DIR.glob("*.json")):
        pytest.skip("no captures in replay_corpus/ — run `python3 -m tests.agent_eval.capture --all` first")
    return DEFAULT_REPLAY_DIR


def _run_runner(output_dir: Path, replay_dir: Path) -> int:
    """Invoke the runner CLI in-process with --deterministic.
    Returns exit code; raises on unexpected failure."""
    from tests.agent_eval.runner import main
    return main([
        "--replay-dir", str(replay_dir),
        "--output", str(output_dir),
        "--deterministic",
    ])


class TestEvalDeterminism:

    def test_two_consecutive_runs_byte_identical(self, tmp_path: Path, replay_dir: Path):
        run_a = tmp_path / "run_a"
        run_b = tmp_path / "run_b"

        rc_a = _run_runner(run_a, replay_dir)
        rc_b = _run_runner(run_b, replay_dir)
        assert rc_a == 0, "first run failed"
        assert rc_b == 0, "second run failed"

        a_files = sorted(run_a.glob("LIVE_SUMMARY.agent.*.md"))
        b_files = sorted(run_b.glob("LIVE_SUMMARY.agent.*.md"))
        assert len(a_files) == len(b_files), (
            f"different number of summaries: a={len(a_files)} b={len(b_files)}"
        )
        assert len(a_files) >= 1, "no summaries emitted — corpus must be non-empty"

        diffs: list[str] = []
        for fa, fb in zip(a_files, b_files):
            assert fa.name == fb.name, (
                f"summary filenames differ: {fa.name} vs {fb.name}"
            )
            if fa.read_bytes() != fb.read_bytes():
                diffs.append(fa.name)

        if diffs:
            # Show the first byte-diff hint so a failing run is
            # debuggable from the CI log without re-running locally.
            first = diffs[0]
            head_a = (run_a / first).read_text()[:1500]
            head_b = (run_b / first).read_text()[:1500]
            pytest.fail(
                f"non-deterministic output across {len(diffs)} summary file(s):\n"
                f"  {diffs}\n\n"
                f"first diff (head 1500 chars of {first}):\n"
                f"--- run_a ---\n{head_a}\n"
                f"--- run_b ---\n{head_b}\n"
            )

    def test_default_replay_dir_artifacts_match_a_fresh_render(
        self, tmp_path: Path, replay_dir: Path
    ):
        """Sanity: render to a tmp dir and confirm the committed
        LIVE_SUMMARY.* in the repo are themselves deterministic re-renders.

        If this fails, the committed artifacts have drifted from what
        the current code would emit — re-run the runner and commit
        the refreshed artifacts.
        """
        fresh = tmp_path / "fresh"
        rc = _run_runner(fresh, replay_dir)
        assert rc == 0

        committed = sorted(replay_dir.glob("LIVE_SUMMARY.agent.*.md"))
        if not committed:
            pytest.skip("no committed summaries to compare against")

        diffs: list[str] = []
        for cf in committed:
            ff = fresh / cf.name
            if not ff.exists():
                diffs.append(f"{cf.name} missing in fresh render")
                continue
            # Compare with --deterministic stripping done — committed
            # files have generated_at; fresh deterministic ones don't.
            # So we strip the generated_at line from both before comparing.
            cf_lines = [
                l for l in cf.read_text().splitlines()
                if not l.startswith("- **generated_at**:")
            ]
            ff_lines = [
                l for l in ff.read_text().splitlines()
                if not l.startswith("- **generated_at**:")
            ]
            if cf_lines != ff_lines:
                diffs.append(cf.name)

        if diffs:
            pytest.fail(
                "Committed LIVE_SUMMARY artifacts diverge from a fresh "
                f"render. Re-run `python3 -m tests.agent_eval.runner` and "
                f"commit. Files affected: {diffs}"
            )
