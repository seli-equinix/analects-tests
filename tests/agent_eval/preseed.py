"""Phase 3.5 — pre-seed hooks for the replay-corpus capture.

A handful of tasks need workspace state in place before the agent
prompt arrives (e.g., simple_fix_typo needs a file with a typo). This
module owns those pre-seeds so the corpus capture script can call them
deterministically.

Pre-seeds run via `docker exec cca bash -c 'cat > <path> <<EOF ...'`
over SSH — fast, deterministic, and they leave no Phoenix span trail
(the agent's first action is reading the pre-seeded file, not creating
it). That keeps captured trajectories focused on the actual eval-task
behavior.

Each hook takes (uid: str) and writes whatever files the task needs
under /workspace inside the cca container on Spark1. The capture
runner cleans them up afterward via cleanup_uid().
"""
from __future__ import annotations

import shlex
import subprocess
from typing import Callable, Dict, List, Optional


# The cca container is on Spark1 at 192.168.4.205. We invoke
# `docker exec cca bash -c '...'` over SSH to write files. Plain
# password auth works here (sshpass is installed on the dev host); a
# more locked-down setup would use a key but that's out of scope for
# the eval capture loop.

SPARK1 = "192.168.4.205"
SSH_USER = "seli"
SSH_PASS = "Loveme-sex64"  # noqa: S105 — local-lan only


def _docker_exec(cmd: str) -> subprocess.CompletedProcess:
    """Run `docker exec cca bash -c <cmd>` via sshpass+ssh on Spark1."""
    full = [
        "sshpass", "-p", SSH_PASS,
        "ssh", "-o", "StrictHostKeyChecking=no",
        f"{SSH_USER}@{SPARK1}",
        f"docker exec cca bash -c {shlex.quote(cmd)}",
    ]
    return subprocess.run(full, check=False, capture_output=True, text=True, timeout=30)


def _write_file(path: str, content: str) -> None:
    """Write content to <path> inside the cca container. Creates
    parent dirs if needed."""
    parent = path.rsplit("/", 1)[0]
    cmd = f"mkdir -p {shlex.quote(parent)} && cat > {shlex.quote(path)} <<'CCA_EVAL_EOF'\n{content}\nCCA_EVAL_EOF"
    res = _docker_exec(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"preseed write_file({path}) failed: {res.stderr or res.stdout}")


def _rm(path: str) -> None:
    """Remove a path inside the cca container. Tolerates missing files."""
    cmd = f"rm -rf {shlex.quote(path)}"
    _docker_exec(cmd)  # ignore failures — best-effort cleanup


# ── Per-task pre-seed hooks ─────────────────────────────────────────


def _preseed_simple_fix_typo(uid: str) -> None:
    """Write a file with `def gretting()` so the agent has something
    to fix."""
    path = f"/workspace/cca-typo-{uid}.py"
    content = (
        "def gretting():\n"
        '    return "hello"\n'
        "\n"
        'if __name__ == "__main__":\n'
        "    print(gretting())\n"
    )
    _write_file(path, content)


def _preseed_multi_extract_helper(uid: str) -> None:
    """Two files (a.py + b.py) with the same `format_date` block."""
    base = f"/workspace/cca-extract-{uid}"
    duplicate_block = (
        "from datetime import datetime\n"
        "\n"
        "def format_date(d):\n"
        '    """Format a date as YYYY-MM-DD."""\n'
        "    return d.strftime('%Y-%m-%d')\n"
    )
    _write_file(f"{base}/a.py", duplicate_block + "\n\nprint(format_date(datetime.now()))\n")
    _write_file(f"{base}/b.py", duplicate_block + "\n\nprint('B says: ' + format_date(datetime.now()))\n")


def _preseed_multi_split_module(uid: str) -> None:
    """Single long.py with two unrelated function families + a main.py
    that imports from it."""
    base = f"/workspace/cca-long-{uid}"
    long_py = (
        "import math\n"
        "\n"
        "# ─── Geometry helpers ───\n"
        "def area_circle(r):\n"
        "    return math.pi * r * r\n"
        "\n"
        "def area_rectangle(w, h):\n"
        "    return w * h\n"
        "\n"
        "def perimeter_rectangle(w, h):\n"
        "    return 2 * (w + h)\n"
        "\n"
        "# ─── Statistics helpers ───\n"
        "def mean(nums):\n"
        "    return sum(nums) / len(nums) if nums else 0\n"
        "\n"
        "def median(nums):\n"
        "    s = sorted(nums)\n"
        "    n = len(s)\n"
        "    if n == 0:\n"
        "        return 0\n"
        "    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2\n"
        "\n"
        "def stddev(nums):\n"
        "    m = mean(nums)\n"
        "    return math.sqrt(sum((x - m) ** 2 for x in nums) / len(nums)) if nums else 0\n"
    )
    main_py = (
        "from long import area_circle, area_rectangle, perimeter_rectangle\n"
        "from long import mean, median, stddev\n"
        "\n"
        "print('circle area:', area_circle(3))\n"
        "print('rect peri:', perimeter_rectangle(2, 5))\n"
        "print('mean:', mean([1, 2, 3, 4, 5]))\n"
        "print('stddev:', stddev([1, 2, 3, 4, 5]))\n"
    )
    _write_file(f"{base}/long.py", long_py)
    _write_file(f"{base}/main.py", main_py)


def _preseed_multi_pair(uid: str) -> None:
    """Pre-seed the cca-pair-{uid}/ directory with lib.py (def square)
    and main.py importing it. Used as the starting state for
    multi_rename_function and multi_add_param."""
    base = f"/workspace/cca-pair-{uid}"
    lib_py = (
        "def square(x):\n"
        "    return x * x\n"
    )
    main_py = (
        "from lib import square\n"
        "\n"
        "print(square(5))\n"
    )
    _write_file(f"{base}/lib.py", lib_py)
    _write_file(f"{base}/main.py", main_py)


# Tasks that take a previously-created uid (chain) — the runner
# passes the same uid to the chain so the second/third task in the
# chain finds the file the first task created.
PRESEED_HOOKS: Dict[str, Callable[[str], None]] = {
    "simple_fix_typo": _preseed_simple_fix_typo,
    "multi_extract_helper": _preseed_multi_extract_helper,
    "multi_split_module": _preseed_multi_split_module,
    # multi_rename_function + multi_add_param + failure_invalid_diff
    # all expect /workspace/cca-pair-{uid}/{lib,main}.py to exist.
    # The capture runner pre-seeds them via this helper.
    "multi_rename_function": _preseed_multi_pair,
    "multi_add_param": _preseed_multi_pair,
    "failure_invalid_diff": _preseed_multi_pair,
}


# ── Cleanup ─────────────────────────────────────────────────────────


# Path templates that may have been written for a given uid. Cleanup
# walks all of them at end-of-task; safe to run even when the task
# wasn't mutating (rm -rf tolerates missing paths).
_UID_CLEANUP_PATHS: List[str] = [
    "/workspace/cca-eval-{uid}.py",
    "/workspace/cca-typo-{uid}.py",
    "/workspace/cca-pair-{uid}",
    "/workspace/cca-extract-{uid}",
    "/workspace/cca-long-{uid}",
    "/workspace/this-file-does-not-exist-{uid}.py",
]


def cleanup_uid(uid: str) -> None:
    """Remove every workspace path that could have been created by a
    task using this uid. Best-effort — silently swallows failures."""
    for tmpl in _UID_CLEANUP_PATHS:
        _rm(tmpl.format(uid=uid))


def needs_preseed(task_name: str) -> bool:
    return task_name in PRESEED_HOOKS


def run_preseed(task_name: str, uid: str) -> None:
    """Run the pre-seed hook for a task if registered. No-op otherwise."""
    hook = PRESEED_HOOKS.get(task_name)
    if hook is None:
        return
    hook(uid)


__all__ = [
    "PRESEED_HOOKS",
    "needs_preseed",
    "run_preseed",
    "cleanup_uid",
]
