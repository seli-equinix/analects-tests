"""Flow test: Bash commands → write a bash script → run and verify.

Journey: introduce as user → run system commands → write a bash script
that gathers the same info → run the script and verify output → confirm
the agent still knows who we are.

CODER bash_tool allowlist includes: cat, df, ls, grep, python3, etc.
Commands like uname, whoami, hostname are NOT allowed.

Exercises: bash_tool (cat, df, script execution), str_replace_editor
(create script), user identification, CODER route.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.coder, pytest.mark.slow]


class TestBashExecution:
    """CODER route: run commands, write a bash script, run it, verify."""

    def test_bash_execution(self, cca, trace_test, judge_model):
        """System commands → bash script → run → verify → user recall."""
        tracker = cca.tracker()
        user_name = f"BashTest_{uuid.uuid4().hex[:6]}"
        script_name = f"sysinfo_{uuid.uuid4().hex[:6]}.sh"
        sid = f"test-bash-{uuid.uuid4().hex[:8]}"
        tracker.track_user(user_name)
        tracker.track_session(sid)
        tracker.track_workspace_prefix(script_name.replace(".sh", ""))

        try:
            # ── Turn 1: Introduce user + run two commands ──
            # Combine commands in one turn to avoid routing issues
            # (follow-up "run X" messages get routed to direct-answer).
            msg1 = (
                f"Hi, I'm {user_name}. I'm a sysadmin checking this "
                f"server. Execute these two commands and show me the "
                f"raw output:\n"
                f"1. `cat /etc/os-release`\n"
                f"2. `df -h /`"
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
            assert r1.content, "Turn 1 returned empty"

            iters1 = r1.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t1_iters", iters1)
            assert iters1 >= 1, (
                f"Agent didn't use tools to run commands (iters={iters1})"
            )

            # Verify bash tool was actually called
            assert "bash" in r1.tool_names, (
                f"bash tool not called. Tools used: {r1.tool_names}"
            )

            # Should contain OS info and disk info from actual execution
            content_lower1 = r1.content.lower()
            has_os = any(w in content_lower1 for w in [
                "ubuntu", "debian", "linux", "name=", "version",
            ])
            has_disk = any(w in content_lower1 for w in [
                "%", "gb", "tb", "filesystem", "mounted",
            ]) or any(c.isdigit() for c in r1.content)
            trace_test.set_attribute("cca.test.has_os_info", has_os)
            trace_test.set_attribute("cca.test.has_disk_info", has_disk)
            assert has_os, (
                f"No OS info in response: {r1.content[:300]}"
            )

            # ── Turn 2: Write a bash script that does the same ──
            msg2 = (
                f"Now create a bash script at /workspace/{script_name} "
                f"that does the same two things: prints the os-release "
                f"file and the df output for root. Add "
                f"'echo SYSINFO_COMPLETE' as the last line."
            )
            r2 = cca.chat(msg2, session_id=sid)
            evaluate_response(r2, msg2, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t2_response", r2.content[:300])
            assert r2.content, "Turn 2 returned empty"

            iters2 = r2.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t2_iters", iters2)
            assert iters2 >= 1, (
                f"Agent didn't create script (iters={iters2})"
            )

            # Verify script exists via REST
            files = cca.list_workspace_files()
            file_list = files.get("files", [])
            file_names = [
                f.get("name", "") if isinstance(f, dict) else str(f)
                for f in file_list
            ]
            has_script = any(script_name in name for name in file_names)
            trace_test.set_attribute("cca.test.script_created", has_script)
            assert has_script, (
                f"Script '{script_name}' not found. "
                f"Files: {file_names[:10]}"
            )

            # ── Turn 3: Run the script and verify output ──
            msg3 = (
                f"Execute /workspace/{script_name} with bash and "
                f"show me the complete output."
            )
            r3 = cca.chat(msg3, session_id=sid)
            evaluate_response(r3, msg3, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
            assert r3.content, "Turn 3 returned empty"

            iters3 = r3.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t3_iters", iters3)
            assert iters3 >= 1, (
                f"Agent didn't run the script (iters={iters3})"
            )

            # SYSINFO_COMPLETE marker proves actual execution
            content_lower3 = r3.content.lower()
            has_done = "sysinfo_complete" in content_lower3
            trace_test.set_attribute("cca.test.has_done_marker", has_done)
            assert has_done, (
                f"SYSINFO_COMPLETE marker not found — script may not "
                f"have been executed: {r3.content[:500]}"
            )

            # ── Turn 4: Verify user recall ──
            msg4 = "Hey, what's my name and what do I do?"
            r4 = cca.chat(msg4, session_id=sid)
            # Skip judge on recall — non-deterministic user memory
            evaluate_response(r4, msg4, trace_test, None, "integration")

            trace_test.set_attribute("cca.test.t4_response", r4.content[:300])
            assert r4.content, "Turn 4 returned empty"

            content_lower4 = r4.content.lower()
            has_name = user_name.lower() in content_lower4
            has_role = any(w in content_lower4 for w in [
                "sysadmin", "admin", "system",
            ])
            trace_test.set_attribute("cca.test.user_recalled", has_name)
            trace_test.set_attribute("cca.test.role_recalled", has_role)
            assert has_name or has_role, (
                f"Agent didn't recall user ({user_name}/sysadmin): "
                f"{r4.content[:300]}"
            )

        finally:
            tracker.cleanup()
