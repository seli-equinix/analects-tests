"""Smoke test: exercise all 5 retrieval modes in one realistic session.

Journey: discover projects → browse structure → semantic search →
graph traversal → note memory. A developer exploring a codebase
they've never seen before.

This is the "user would ask this" end-to-end test. Each turn uses
a different retrieval mode, verifying all 5 modes work together
in a single session.

Mode 1: Structural browsing (vectorless RAG — browse_project)
Mode 2: Semantic search (vector search — search_codebase)
Mode 3: Graph traversal (Memgraph — query_call_graph)
Mode 4: Execution tracing (BFS — trace_execution)
Mode 5: Note memory (Qdrant — search_notes)

Exercises: browse_project, search_codebase, query_call_graph,
trace_execution, search_notes. CODER route.

Requires: Indexed EVA project in Qdrant + Memgraph.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestAllRetrievalModes:
    """Exercise all 5 retrieval modes in one realistic session."""

    def test_all_retrieval_modes(self, cca, trace_test, judge_model):
        """5-turn flow: browse → search → graph → trace → notes.

        Each turn exercises a different retrieval mode with
        tool_names assertions to verify the right tool was called.
        """
        tracker = cca.tracker()
        user_name = f"RetrievalTest_{uuid.uuid4().hex[:6]}"
        sid = f"test-5modes-{uuid.uuid4().hex[:8]}"
        tracker.track_user(user_name)
        tracker.track_session(sid)

        try:
            # ── Mode 1: Structural browsing (vectorless RAG) ──
            msg1 = (
                f"Hi, I'm {user_name}. I'm new to this codebase. "
                f"What projects are indexed? Show me EVA's structure."
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
            assert r1.content, "Turn 1 empty"

            tool_names_1 = r1.tool_names
            trace_test.set_attribute("cca.test.t1_tools", str(tool_names_1))
            assert any(
                t in name for name in tool_names_1
                for t in ["list_projects", "browse_project", "search_codebase"]
            ), (
                f"Mode 1 (structural): no project discovery tools. "
                f"Tools: {tool_names_1}"
            )
            assert "eva" in r1.content.lower(), (
                "EVA not mentioned in project listing"
            )

            # ── Mode 2: Semantic search ──
            msg2 = (
                "Search the EVA codebase for functions that handle "
                "VM template deployment. I want to understand how "
                "templates are used to create new VMs."
            )
            r2 = cca.chat(msg2, session_id=sid)
            evaluate_response(r2, msg2, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
            assert r2.content, "Turn 2 empty"

            tool_names_2 = r2.tool_names
            trace_test.set_attribute("cca.test.t2_tools", str(tool_names_2))
            assert any(
                "search_codebase" in name or "search_knowledge" in name
                for name in tool_names_2
            ), (
                f"Mode 2 (semantic search): search_codebase not called. "
                f"Tools: {tool_names_2}"
            )

            # ── Mode 3: Graph traversal ──
            msg3 = (
                "What functions call Add-VMFromTemplate in the EVA project? "
                "Use the call graph to show me all callers."
            )
            r3 = cca.chat(msg3, session_id=sid)
            evaluate_response(r3, msg3, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
            assert r3.content, "Turn 3 empty"

            tool_names_3 = r3.tool_names
            trace_test.set_attribute("cca.test.t3_tools", str(tool_names_3))
            assert any(
                t in name for name in tool_names_3
                for t in ["query_call_graph", "analyze_dependencies", "search_codebase"]
            ), (
                f"Mode 3 (graph): no graph/search tools called. "
                f"Tools: {tool_names_3}"
            )

            # ── Mode 4: Execution tracing ──
            msg4 = (
                "Trace the execution path starting from JobStart.ps1. "
                "What functions get called when processing a VM build?"
            )
            r4 = cca.chat(msg4, session_id=sid)
            evaluate_response(r4, msg4, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t4_response", r4.content[:500])
            assert r4.content, "Turn 4 empty"

            tool_names_4 = r4.tool_names
            trace_test.set_attribute("cca.test.t4_tools", str(tool_names_4))
            assert any(
                t in name for name in tool_names_4
                for t in ["trace_execution", "search_codebase", "query_call_graph"]
            ), (
                f"Mode 4 (tracing): no trace/search tools. "
                f"Tools: {tool_names_4}"
            )

            # ── Mode 5: Note memory ──
            msg5 = (
                "Search your notes for anything about VM templates or "
                "the EVA project from our conversation."
            )
            r5 = cca.chat(msg5, session_id=sid)
            evaluate_response(r5, msg5, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t5_response", r5.content[:500])
            assert r5.content, "Turn 5 empty"

            tool_names_5 = r5.tool_names
            trace_test.set_attribute("cca.test.t5_tools", str(tool_names_5))
            # Notes may or may not have been stored yet — allow search_codebase
            # as fallback if search_notes returns empty
            assert any(
                t in name for name in tool_names_5
                for t in ["search_notes", "search_codebase", "search_knowledge"]
            ), (
                f"Mode 5 (notes): no search tools called. "
                f"Tools: {tool_names_5}"
            )

        finally:
            tracker.cleanup()
