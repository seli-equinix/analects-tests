"""Flow test: Write code, edit it, then run it to verify it works.

Journey: introduce as a user → create a Python file → edit it to add
a function → run the file with python3 → verify ACTUAL execution output.

The code includes a computation (sum of range) that produces a specific
value — checking for this value in the response proves the code was
actually executed, not just described by the agent.

Exercises: str_replace_editor (create, str_replace), bash_tool (python3),
CODER route, user identification.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.coder, pytest.mark.slow]


class TestCodeEditFlow:
    """CODER route: write code, edit it, run it, verify output."""

    def test_code_edit_flow(self, cca, trace_test, judge_model):
        """Full dev workflow: create → edit → run → verify actual execution."""
        tracker = cca.tracker()
        filename = f"test_edit_{uuid.uuid4().hex[:6]}.py"
        user_name = f"EditFlow_{uuid.uuid4().hex[:6]}"
        sid = f"test-edit-{uuid.uuid4().hex[:8]}"
        tracker.track_user(user_name)
        tracker.track_session(sid)
        tracker.track_workspace_prefix(filename.replace(".py", ""))

        try:
            # ── Turn 1: Introduce user + create a Python file ──
            msg1 = (
                f"Hi, I'm {user_name}. Create a Python file called "
                f"{filename} in /workspace with a function called "
                f"greet(name) that returns 'Hello, {{name}}!'. "
                f"Add a main block that prints greet('World')."
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:300])
            assert r1.content, "Turn 1 returned empty"

            iters = r1.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t1_iters", iters)
            assert iters >= 1, (
                f"Agent didn't use tools to create file (iters={iters})"
            )

            # Verify str_replace_editor was used for file creation
            assert "str_replace_editor" in r1.tool_names, (
                f"str_replace_editor not called for file creation. "
                f"Tools: {r1.tool_names}"
            )

            # Verify file exists via REST — ground truth, not LLM response
            files = cca.list_workspace_files()
            file_list = files.get("files", [])
            file_names = [
                f.get("name", "") if isinstance(f, dict) else str(f)
                for f in file_list
            ]
            has_file = any(filename in name for name in file_names)
            trace_test.set_attribute("cca.test.file_created", has_file)
            assert has_file, (
                f"File '{filename}' not found in workspace. "
                f"Files: {file_names[:10]}"
            )

            # ── Turn 2: Edit — add farewell + execution checksum ──
            # Natural follow-up — no name repetition.
            # The checksum (sum of 1..50 = 1275) proves actual execution
            # when we verify it in Turn 3.
            msg2 = (
                "Now add a farewell(name) function to that file that "
                "returns 'Goodbye, {name}!'. Update the main block to "
                "print greet('World'), then farewell('World'), then "
                "print('checksum:', sum(range(1, 51)))"
            )
            r2 = cca.chat(msg2, session_id=sid)
            evaluate_response(r2, msg2, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t2_response", r2.content[:300])
            assert r2.content, "Turn 2 returned empty"

            iters2 = r2.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t2_iters", iters2)
            assert iters2 >= 1, (
                f"Agent didn't use tools to edit file (iters={iters2})"
            )

            # ── Turn 3: Run the file and verify ACTUAL output ──
            # Must include filename — the router classifies each message
            # independently and "that file" gets routed to clarify.
            msg3 = (
                f"Run /workspace/{filename} with python3 and show me "
                f"the output."
            )
            r3 = cca.chat(msg3, session_id=sid)
            evaluate_response(r3, msg3, trace_test, judge_model, "integration")

            trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
            assert r3.content, "Turn 3 returned empty"

            iters3 = r3.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t3_iters", iters3)
            assert iters3 >= 1, (
                f"Agent didn't run the file (iters={iters3})"
            )

            # Verify ACTUAL execution — not just described output.
            # The agent tends to show "Output: Hello, World!" even when
            # it only created/edited the file without running it.
            # Checking for "1275" (sum of 1..50) proves the code ran.
            content_lower = r3.content.lower()
            has_hello = "hello" in content_lower
            has_goodbye = "goodbye" in content_lower
            has_checksum = "1275" in r3.content
            trace_test.set_attribute("cca.test.has_hello", has_hello)
            trace_test.set_attribute("cca.test.has_goodbye", has_goodbye)
            trace_test.set_attribute("cca.test.has_checksum", has_checksum)

            assert has_hello, (
                f"Output doesn't contain 'Hello': {r3.content[:300]}"
            )
            assert has_goodbye, (
                f"Output doesn't contain 'Goodbye': {r3.content[:300]}"
            )
            assert has_checksum, (
                f"Output doesn't contain checksum '1275' — code may not "
                f"have been actually executed (agent might be describing "
                f"expected output without running it): {r3.content[:500]}"
            )

        finally:
            tracker.cleanup()
