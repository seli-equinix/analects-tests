"""End-to-end crown-jewel test for file-as-process semantics.

Stages 1+2+4a together produce this user-visible win: a Python script
that orchestrates other scripts via subprocess.run becomes a Process
node whose STEP_IN_PROCESS chain spans every invoked file.

Without these stages, the orchestrator's chain stops at named function
calls within itself — the cross-file invocations are completely invisible.

Test flow:
  1. SSH-stage a 3-file Python project under /data/cca-workspace/<uuid>/
     (visible to cca as /workspace/<uuid>/).
  2. Trigger /workspace/reindex with paths=[the staged dir].
  3. Poll until completed.
  4. Three assertions:
       a) Each file has a __module__ Function node (Stage 1).
       b) orchestrator.py's __module__ has 2 outgoing CALLS edges with
          method='subprocess', targeting step_a.py and step_b.py's
          __module__ entries (Stage 2).
       c) At least one Process node spans all 3 module entries via
          STEP_IN_PROCESS edges (Stage 4a + the existing PROCESS
          BFS phase).
  5. Teardown: SSH rm -rf the staged dir + DETACH DELETE the project's
     Memgraph nodes.

Requires CCA_SSH_PASSWORD env var (the seli user's SSH password on
Spark1, also used by the rest of the CCA docs/CLAUDE.md infra paths).
If unset, the test is SKIPPED — no point failing CI on missing infra.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import uuid
from typing import Iterator

import pytest

from tests.integration._admin import (
    cypher, cypher_rows, trigger_reindex, wait_for_reindex,
)


pytestmark = [pytest.mark.integration, pytest.mark.slow]


SSH_HOST = os.environ.get("CCA_SSH_HOST", "192.168.4.205")
SSH_USER = os.environ.get("CCA_SSH_USER", "seli")
SSH_PASSWORD = os.environ.get("CCA_SSH_PASSWORD", "")
# The host path that the cca container bind-mounts at /workspace.
HOST_WORKSPACE = os.environ.get(
    "CCA_HOST_WORKSPACE", "/data/cca-workspace",
)


def _ssh(cmd: str, *, timeout_s: int = 30) -> str:
    """Run a command on Spark1 via sshpass; return stdout. Raises on non-zero exit."""
    full = [
        "sshpass", "-p", SSH_PASSWORD,
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        f"{SSH_USER}@{SSH_HOST}",
        cmd,
    ]
    res = subprocess.run(full, capture_output=True, text=True, timeout=timeout_s)
    if res.returncode != 0:
        raise RuntimeError(
            f"SSH command failed (exit {res.returncode}): {cmd}\n"
            f"stderr: {res.stderr[:500]}"
        )
    return res.stdout


def _ssh_write_file(remote_path: str, content: str) -> None:
    """Write `content` to a remote file via SSH + cat heredoc."""
    # base64-encode content to safely handle special characters / quotes.
    import base64
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    _ssh(
        f"mkdir -p {shlex.quote(os.path.dirname(remote_path))} && "
        f"echo {shlex.quote(b64)} | base64 -d > {shlex.quote(remote_path)}"
    )


@pytest.fixture(scope="module")
def orchestrator_project() -> Iterator[dict]:
    """Stage a 3-file Python orchestration on Spark1; yield project metadata.

    Module-scoped so all three assertions reuse the same indexing pass
    — Memgraph state survives between tests within the module.
    """
    if not SSH_PASSWORD:
        pytest.skip(
            "CCA_SSH_PASSWORD not set — cannot stage synthetic project on Spark1"
        )

    # Unique project name = directory basename. Stays out of the way of
    # other projects and lets teardown be a precise DETACH DELETE.
    proj_name = f"test-orch-{uuid.uuid4().hex[:8]}"
    host_dir = f"{HOST_WORKSPACE}/{proj_name}"
    container_dir = f"/workspace/{proj_name}"

    # Chained orchestration: orchestrator → step_a → step_b. A 3-step
    # path through module entries, which clears PROCESS phase's
    # default min_steps=3 threshold (see
    # confucius/server/code_intelligence/pipeline/phases/processes.py:44).
    # A flat orchestrator-runs-both pattern would produce two 2-step
    # traces, each filtered out by min_steps and never producing a
    # Process node.
    orchestrator_py = (
        "import subprocess\n"
        "subprocess.run(['python', 'step_a.py'])\n"
    )
    step_a_py = (
        "import subprocess\n"
        "def process_a():\n"
        "    return 1\n"
        "\n"
        "process_a()\n"
        "subprocess.run(['python', 'step_b.py'])\n"
    )
    step_b_py = (
        "def process_b():\n"
        "    return 2\n"
        "\n"
        "process_b()\n"
    )

    try:
        # Stage the 3 files via SSH
        _ssh_write_file(f"{host_dir}/orchestrator.py", orchestrator_py)
        _ssh_write_file(f"{host_dir}/step_a.py", step_a_py)
        _ssh_write_file(f"{host_dir}/step_b.py", step_b_py)

        # Trigger indexing (paths in container-path namespace, not host)
        job_id = trigger_reindex(paths=[container_dir], force=True)
        result = wait_for_reindex(job_id, timeout_s=300, poll_s=5)
        if result.get("status") != "completed":
            pytest.fail(
                f"reindex of {container_dir} did not complete cleanly: "
                f"{result.get('status')!r} — {result}"
            )
        yield {
            "name": proj_name,
            "container_dir": container_dir,
            "host_dir": host_dir,
            "orchestrator_qname": f"{container_dir}/orchestrator.py::__module__",
            "step_a_qname": f"{container_dir}/step_a.py::__module__",
            "step_b_qname": f"{container_dir}/step_b.py::__module__",
        }
    finally:
        # Teardown — clean Memgraph then disk. Tolerant of partial state.
        try:
            # Project name in the graph is derived from the indexer's
            # project map. Since this dir isn't in any configured project,
            # the indexer falls back to 'workspace' as the project name.
            # We delete by file_path prefix instead.
            cypher(
                "MATCH (n) WHERE n.file_path STARTS WITH $prefix DETACH DELETE n",
                prefix=container_dir,
            )
            cypher(
                "MATCH (f:File) WHERE f.path STARTS WITH $prefix DETACH DELETE f",
                prefix=container_dir,
            )
        except Exception:
            pass
        try:
            _ssh(f"rm -rf {shlex.quote(host_dir)}")
        except Exception:
            pass


# ── Assertion 1: each file has a __module__ entry (Stage 1) ──────────────


class TestModuleEntriesCreated:
    def test_orchestrator_has_module(self, orchestrator_project: dict) -> None:
        rows = cypher_rows(
            "MATCH (f:Function {is_module: true, qualified_name: $q}) "
            "RETURN f.qualified_name AS qname",
            q=orchestrator_project["orchestrator_qname"],
        )
        assert rows, (
            f"no __module__ entry for orchestrator.py — Stage 1 broken? "
            f"Expected: {orchestrator_project['orchestrator_qname']}"
        )

    def test_step_a_has_module(self, orchestrator_project: dict) -> None:
        rows = cypher_rows(
            "MATCH (f:Function {is_module: true, qualified_name: $q}) "
            "RETURN f.qualified_name AS qname",
            q=orchestrator_project["step_a_qname"],
        )
        assert rows, f"no __module__ entry for step_a.py"

    def test_step_b_has_module(self, orchestrator_project: dict) -> None:
        rows = cypher_rows(
            "MATCH (f:Function {is_module: true, qualified_name: $q}) "
            "RETURN f.qualified_name AS qname",
            q=orchestrator_project["step_b_qname"],
        )
        assert rows, f"no __module__ entry for step_b.py"


# ── Assertion 2: cross-file CALLS edges with method='subprocess' (Stage 2) ──


class TestCrossFileCallsEdges:
    def test_orchestrator_calls_step_a(
        self, orchestrator_project: dict,
    ) -> None:
        """orchestrator.py subprocess-runs step_a.py. CALLS edge must exist
        from orchestrator's __module__ → step_a's __module__ with
        method='subprocess'.
        """
        rows = cypher_rows(
            """
            MATCH (a:Function {qualified_name: $orch})-[r:CALLS]->(b:Function)
            WHERE r.method = 'subprocess'
            RETURN b.qualified_name AS target, r.method AS method, r.line AS line
            """,
            orch=orchestrator_project["orchestrator_qname"],
        )
        targets = {r["target"] for r in rows}
        assert orchestrator_project["step_a_qname"] in targets, (
            f"orchestrator.py is missing its subprocess CALLS edge to step_a. "
            f"Got: {targets}. "
            f"Stage 2 (invoke_detector → resolve_cross_file_invocations) broken?"
        )

    def test_step_a_calls_step_b(
        self, orchestrator_project: dict,
    ) -> None:
        """step_a.py subprocess-runs step_b.py — the second hop of the chain.
        Without this edge, the 3-step Process chain wouldn't form.
        """
        rows = cypher_rows(
            """
            MATCH (a:Function {qualified_name: $step_a})-[r:CALLS]->(b:Function)
            WHERE r.method = 'subprocess'
            RETURN b.qualified_name AS target, r.method AS method
            """,
            step_a=orchestrator_project["step_a_qname"],
        )
        targets = {r["target"] for r in rows}
        assert orchestrator_project["step_b_qname"] in targets, (
            f"step_a.py is missing its subprocess CALLS edge to step_b. "
            f"Got: {targets}"
        )

    def test_method_property_is_subprocess(
        self, orchestrator_project: dict,
    ) -> None:
        """Both cross-file edges in the chain must carry method='subprocess',
        not 'direct' — confirming Stage 2c's property write went through."""
        rows = cypher_rows(
            """
            MATCH (a:Function)-[r:CALLS]->(b:Function)
            WHERE a.qualified_name IN [$orch, $step_a]
              AND b.qualified_name IN [$step_a, $step_b]
              AND a.qualified_name <> b.qualified_name
            RETURN r.method AS method
            """,
            orch=orchestrator_project["orchestrator_qname"],
            step_a=orchestrator_project["step_a_qname"],
            step_b=orchestrator_project["step_b_qname"],
        )
        assert rows, "no CALLS edges between orchestrator/step_a/step_b"
        methods = {r["method"] for r in rows}
        # Both edges should carry the 'subprocess' method label, not
        # 'direct' (which would mean the cross-file resolver didn't fire).
        assert methods == {"subprocess"}, (
            f"expected method='subprocess' on the chain edges, got {methods}"
        )


# ── Assertion 3: a Process node spans all 3 module entries (Stage 4a) ──────


class TestProcessChainSpansAllModules:
    def test_process_chain_includes_orchestrator_and_steps(
        self, orchestrator_project: dict,
    ) -> None:
        # PROCESS phase runs BFS from entry-point seeds; with the Stage
        # 4a boost, orchestrator's __module__ ranks high and the BFS
        # follows its CALLS edges into step_a + step_b.
        rows = cypher_rows(
            """
            MATCH (p:Process)<-[:STEP_IN_PROCESS]-(sym:Function {is_module: true})
            WHERE sym.qualified_name IN [$orch, $a, $b]
            WITH p, collect(DISTINCT sym.qualified_name) AS modules
            WHERE size(modules) >= 3
            RETURN p.id AS id, p.label AS label, modules
            ORDER BY size(modules) DESC
            LIMIT 5
            """,
            orch=orchestrator_project["orchestrator_qname"],
            a=orchestrator_project["step_a_qname"],
            b=orchestrator_project["step_b_qname"],
        )
        assert rows, (
            "no Process node spans all 3 __module__ entries — Stage 4a "
            "(PROCESS phase module-aware scoring) or the cross-file CALLS "
            "edges aren't reachable via BFS. Check that "
            "resolve_cross_file_invocations() ran and emitted edges with "
            "method='subprocess' BEFORE the processes phase fired."
        )
        # The first row is the largest-span process — its module set must
        # be exactly the 3 we staged.
        modules = set(rows[0]["modules"])
        expected = {
            orchestrator_project["orchestrator_qname"],
            orchestrator_project["step_a_qname"],
            orchestrator_project["step_b_qname"],
        }
        assert expected.issubset(modules), (
            f"Process spans some modules but not all 3: got {modules}"
        )
