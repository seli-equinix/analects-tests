"""Flow test: Multi-turn infrastructure inspection — read-only Docker checkup.

Journey: check running containers → drill into a specific service →
check a remote node → ask for a health summary across services.

This simulates a real user doing a morning check on their Docker setup.
All operations are read-only — no containers are started, stopped,
or modified. The agent should use bash (docker ps, curl, ssh) to
gather real system state.

Exercises: bash_tool (docker, curl, ssh), INFRASTRUCTURE route,
multi-turn session context.
"""

import subprocess
import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _ssh_reachable(host: str = "192.168.4.205", timeout: int = 5) -> bool:
    """Check if SSH to a host is reachable via sshpass."""
    try:
        result = subprocess.run(
            [
                "sshpass", "-p", "Loveme-sex64", "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=5",
                f"seli@{host}", "echo", "ok",
            ],
            capture_output=True, timeout=timeout + 5, text=True,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


class TestInfraInspectionFlow:
    """Multi-turn INFRA route: inspect a running Docker setup without changes."""

    @pytest.fixture(autouse=True)
    def require_ssh(self):
        """Skip infra inspection tests if SSH to Spark1 is unreachable."""
        if not _ssh_reachable():
            pytest.skip("SSH to Spark1 unreachable — skipping infra test")

    def test_docker_inspection_flow(self, test_run, trace_test, judge_model):
        """4-turn read-only inspection: containers → service detail → remote → summary."""
        sid = f"test-infra-inspect-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        # ── Turn 1: What's running? ──
        # A natural starting question — should list containers
        msg1 = (
            "Hey, can you check what Docker containers are running "
            "on Spark1? I want to see their names and whether they're healthy."
        )
        r1 = test_run.chat(msg1, session_id=sid)
        evaluate_response(r1, msg1, trace_test, judge_model, "integration")

        trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
        assert r1.content, "Turn 1 returned empty"

        iters1 = r1.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t1_iters", iters1)
        assert iters1 >= 1, (
            f"Agent didn't use tools to check containers (iters={iters1})"
        )

        # Should mention actual container names we know exist
        content1 = r1.content.lower()
        known_containers = ["redis", "qdrant", "embedding", "cca"]
        found_containers = [c for c in known_containers if c in content1]
        trace_test.set_attribute(
            "cca.test.t1_containers_found", ",".join(found_containers)
        )
        assert len(found_containers) >= 2, (
            f"Expected at least 2 known containers in response, "
            f"found {found_containers}: {r1.content[:400]}"
        )

        # ── Turn 2: Drill into a specific service ──
        # Follow-up in same session — agent should have context
        msg2 = (
            "How about Redis specifically — is it responding? "
            "Can you ping it and tell me how much memory it's using? "
            "The container is called redis-memory and the password is Loveme-sex64."
        )
        r2 = test_run.chat(msg2, session_id=sid)
        evaluate_response(r2, msg2, trace_test, judge_model, "integration")

        trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
        assert r2.content, "Turn 2 returned empty"

        iters2 = r2.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t2_iters", iters2)
        assert iters2 >= 1, (
            f"Agent didn't use tools to check Redis (iters={iters2})"
        )

        # Should mention PONG or connected or memory stats
        content2 = r2.content.lower()
        has_redis_response = any(w in content2 for w in [
            "pong", "connected", "memory", "used_memory",
            "redis", "mb", "gb", "bytes",
        ])
        trace_test.set_attribute("cca.test.t2_redis_info", has_redis_response)
        assert has_redis_response, (
            f"Response doesn't contain Redis status info: {r2.content[:400]}"
        )

        # ── Turn 3: Check a remote node ──
        # This tests SSH capability — target node1 (always-on swarm manager)
        # instead of Spark2 (frequently down). Read-only check only.
        msg3 = (
            "Can you SSH to node1 at 192.168.4.200 and run "
            "'docker node ls' to check the swarm status? "
            "Use sshpass with user seli and password Loveme-sex64."
        )
        r3 = test_run.chat(msg3, session_id=sid)
        evaluate_response(r3, msg3, trace_test, judge_model, "integration")

        trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
        assert r3.content, "Turn 3 returned empty"

        iters3 = r3.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t3_iters", iters3)
        assert iters3 >= 1, (
            f"Agent didn't use tools to check vLLM (iters={iters3})"
        )

        # Should mention health status — either up or down is fine,
        # we're testing that it actually checked, not that vLLM is running.
        # Spark2 frequently goes down, so the agent may report failure —
        # that's still a valid check.
        content3 = r3.content.lower()
        has_node_check = any(w in content3 for w in [
            "node", "ready", "active", "leader", "manager", "worker",
            "swarm", "192.168.4.200", "seli", "docker",
            "not responding", "unreachable", "connection refused",
            "timed out", "down", "error", "ssh", "failed",
        ])
        trace_test.set_attribute("cca.test.t3_node_checked", has_node_check)
        assert has_node_check, (
            f"Response doesn't indicate node1 was checked: {r3.content[:400]}"
        )

        # ── Turn 4: Ask for a summary ──
        # Tests that the agent can synthesize across previous turns
        msg4 = (
            "Give me a quick summary — which services are healthy "
            "and which ones need attention?"
        )
        r4 = test_run.chat(msg4, session_id=sid)
        evaluate_response(r4, msg4, trace_test, judge_model, "integration")

        trace_test.set_attribute("cca.test.t4_response", r4.content[:500])
        assert r4.content, "Turn 4 returned empty"

        # Summary should reference multiple services
        content4 = r4.content.lower()
        services_mentioned = sum(1 for s in [
            "redis", "qdrant", "vllm", "embedding", "searxng", "cca",
        ] if s in content4)
        trace_test.set_attribute(
            "cca.test.t4_services_mentioned", services_mentioned
        )
        assert services_mentioned >= 2, (
            f"Summary only mentions {services_mentioned} services, "
            f"expected at least 2: {r4.content[:400]}"
        )

    def test_infra_container_logs(self, test_run, trace_test, judge_model):
        """Ask to check container logs — should use docker logs, not modify anything."""
        sid = f"test-infra-logs-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        # ── Turn 1: Ask for logs ──
        msg1 = (
            "Can you show me the last few log lines from the "
            "embedding server container? I want to make sure "
            "it's not throwing errors."
        )
        r1 = test_run.chat(msg1, session_id=sid)
        evaluate_response(r1, msg1, trace_test, judge_model, "integration")

        trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
        assert r1.content, "Turn 1 returned empty"

        iters1 = r1.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t1_iters", iters1)
        assert iters1 >= 1, (
            f"Agent didn't use tools to check logs (iters={iters1})"
        )

        # Should show actual log content or mention the container
        content1 = r1.content.lower()
        has_log_info = any(w in content1 for w in [
            "log", "embedding", "qwen", "error", "info",
            "request", "startup", "loaded", "model",
            "no errors", "healthy", "running",
        ])
        trace_test.set_attribute("cca.test.t1_has_logs", has_log_info)
        assert has_log_info, (
            f"Response doesn't contain log info: {r1.content[:400]}"
        )

        # ── Turn 2: Follow up on something in the logs ──
        msg2 = (
            "Are there any warning or error messages in those logs? "
            "What about the Qdrant container — any issues there?"
        )
        r2 = test_run.chat(msg2, session_id=sid)
        evaluate_response(r2, msg2, trace_test, judge_model, "integration")

        trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
        assert r2.content, "Turn 2 returned empty"

        iters2 = r2.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t2_iters", iters2)
        assert iters2 >= 1, (
            f"Agent didn't check Qdrant logs (iters={iters2})"
        )

        # Should mention both services
        content2 = r2.content.lower()
        has_qdrant = "qdrant" in content2
        trace_test.set_attribute("cca.test.t2_has_qdrant", has_qdrant)
        assert has_qdrant, (
            f"Response doesn't mention Qdrant: {r2.content[:400]}"
        )

    def test_infra_disk_and_resources(self, test_run, trace_test, judge_model):
        """Ask about disk usage and system resources — common ops check."""
        sid = f"test-infra-resources-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        # ── Turn 1: Disk usage ──
        msg1 = (
            "How's the disk usage on this machine? "
            "Are any partitions getting close to full?"
        )
        r1 = test_run.chat(msg1, session_id=sid)
        evaluate_response(r1, msg1, trace_test, judge_model, "integration")

        trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
        assert r1.content, "Turn 1 returned empty"

        iters1 = r1.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t1_iters", iters1)
        assert iters1 >= 1, (
            f"Agent didn't use tools to check disk (iters={iters1})"
        )

        # Should contain disk/filesystem data
        content1 = r1.content.lower()
        has_disk_info = any(w in content1 for w in [
            "disk", "filesystem", "/dev/", "use%", "mounted",
            "gb", "tb", "available", "capacity", "%",
        ])
        trace_test.set_attribute("cca.test.t1_has_disk", has_disk_info)
        assert has_disk_info, (
            f"Response doesn't contain disk info: {r1.content[:400]}"
        )

        # ── Turn 2: Docker resource usage ──
        msg2 = (
            "What about Docker — how much disk space are images "
            "and containers using? Is there anything to clean up?"
        )
        r2 = test_run.chat(msg2, session_id=sid)
        evaluate_response(r2, msg2, trace_test, judge_model, "integration")

        trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
        assert r2.content, "Turn 2 returned empty"

        iters2 = r2.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t2_iters", iters2)
        assert iters2 >= 1, (
            f"Agent didn't check Docker disk usage (iters={iters2})"
        )

        # Should mention images, volumes, or docker system
        content2 = r2.content.lower()
        has_docker_disk = any(w in content2 for w in [
            "image", "volume", "docker system", "reclaimable",
            "build cache", "container", "gb", "mb",
        ])
        trace_test.set_attribute("cca.test.t2_has_docker_disk", has_docker_disk)
        assert has_docker_disk, (
            f"Response doesn't contain Docker disk info: {r2.content[:400]}"
        )
