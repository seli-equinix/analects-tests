"""Flow test: Code trace tools — trace execution and assemble code.

Journey: trace a known function → assemble traced code → ask about
impact of changes. A developer exploring a codebase before making
modifications.

Exercises: trace_execution, assemble_traced_code (TRACE group),
CODER route.

Requires: Indexed codebase in Qdrant + Memgraph with at least one
project that has function call relationships.
"""

import uuid

import pytest

from tests.evaluators import assert_tools_called, evaluate_response

pytestmark = [pytest.mark.coder, pytest.mark.slow]


class TestCodeTrace:
    """CODER route: trace execution paths and assemble code."""

    def test_trace_and_assemble(self, cca, trace_test, judge_model):
        """3-turn flow: trace a function → assemble code → ask about impact.

        Turn 1: Trace execution from a known entry point
        Turn 2: Assemble the traced code into a single view
        Turn 3: Ask which files need changes (uses trace context)
        """
        tracker = cca.tracker()
        sid = f"test-trace-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)

        try:
            # ── Turn 1: Trace from a known function ──
            # Use a general query — the indexed codebase has Python files
            # from the MCP server or CCA projects. Ask about something
            # that should exist in ANY indexed codebase.
            msg1 = (
                "I need to understand how the health check works in the "
                "codebase. Trace the execution path starting from the "
                "health endpoint function — what functions does it call, "
                "and what files are involved?"
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
            assert r1.content, "Turn 1 returned empty"

            iters1 = r1.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t1_iters", iters1)
            assert iters1 >= 1, (
                f"Agent didn't use trace tools (iters={iters1}). "
                f"Response: {r1.content[:200]}"
            )

            # Should have used trace_execution, search_codebase, or query_call_graph
            tool_names_1 = r1.tool_names
            trace_test.set_attribute("cca.test.t1_tools", str(tool_names_1))
            used_code_tools = any(
                t in name for name in tool_names_1
                for t in [
                    "trace_execution", "search_codebase",
                    "query_call_graph", "search_knowledge",
                ]
            )
            assert used_code_tools, (
                f"Agent didn't use code intelligence tools. Called: {tool_names_1}"
            )

            # Response should mention function names or file paths
            content1 = r1.content.lower()
            has_code_refs = any(w in content1 for w in [
                ".py", "def ", "health", "function", "endpoint",
                "calls", "trace",
            ])
            trace_test.set_attribute("cca.test.t1_has_code_refs", has_code_refs)
            assert has_code_refs, (
                f"Response doesn't reference code: {r1.content[:300]}"
            )

            # ── Turn 2: Assemble the traced code ──
            msg2 = (
                "Now assemble the traced code into a single view so I can "
                "review all the functions involved. Show me the complete "
                "function bodies, not just signatures."
            )
            r2 = cca.chat(msg2, session_id=sid)
            evaluate_response(r2, msg2, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
            assert r2.content, "Turn 2 returned empty"

            iters2 = r2.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t2_iters", iters2)
            assert iters2 >= 1, (
                f"Agent didn't use assemble tools (iters={iters2}). "
                f"Response: {r2.content[:200]}"
            )

            # Must have used assemble_traced_code or str_replace_editor
            tool_names_2 = r2.tool_names
            trace_test.set_attribute("cca.test.t2_tools", str(tool_names_2))
            assert any(
                t in name for name in tool_names_2
                for t in ["assemble_traced_code", "str_replace_editor", "search_codebase"]
            ), (
                f"No assembly/search tool called in Turn 2. "
                f"Tools: {tool_names_2}"
            )

            # Response should contain actual code
            content2 = r2.content
            has_code = "```" in content2 or "def " in content2 or "function" in content2.lower()
            trace_test.set_attribute("cca.test.t2_has_code", has_code)
            assert has_code, (
                f"Response doesn't include function bodies: {r2.content[:300]}"
            )

            # ── Turn 3: Impact analysis ──
            msg3 = (
                "Based on what you traced, which files would I need to "
                "modify if I wanted to change how the health check works? "
                "Are there any downstream callers I should be aware of?"
            )
            r3 = cca.chat(msg3, session_id=sid)
            evaluate_response(r3, msg3, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
            assert r3.content, "Turn 3 returned empty"

            # Should reference specific files or functions from the trace
            content3 = r3.content.lower()
            has_impact_refs = any(w in content3 for w in [
                ".py", "file", "modify", "change", "impact",
                "depend", "caller", "import",
            ])
            trace_test.set_attribute("cca.test.t3_has_impact", has_impact_refs)
            assert has_impact_refs, (
                f"Response doesn't discuss impact: {r3.content[:300]}"
            )

        finally:
            tracker.cleanup()
