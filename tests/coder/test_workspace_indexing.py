"""Flow test: Workspace indexing and multi-tier knowledge search.

Journey 1: create a file -> index workspace -> search for it.
A developer adding new code and verifying it's searchable.

Journey 2: upload doc -> promote to knowledge -> search knowledge tiers.
A developer storing reference material for long-term use.

Exercises: str_replace_editor (FILE), index_workspace, search_codebase
(CODE_SEARCH), upload_document, promote_doc_to_knowledge (DOCUMENT),
search_knowledge (CODE_SEARCH), CODER route.
"""

import uuid

import pytest

from tests.evaluators import assert_tools_called, evaluate_response

pytestmark = [pytest.mark.coder, pytest.mark.slow]


class TestWorkspaceIndexing:
    """CODER route: index workspace and search indexed content."""

    def test_index_and_search(self, test_run, trace_test, judge_model):
        """3-turn flow: create file -> index workspace -> search for content.

        Turn 1: Create a Python file with distinctive function names
        Turn 2: Index the workspace to pick up the new file
        Turn 3: Search the codebase for the distinctive function
        """
        sid = f"test-index-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        # Unique marker so search won't match pre-existing content
        unique_id = uuid.uuid4().hex[:8]
        func_name = f"calculate_zephyr_flux_{unique_id}"
        file_prefix = f"test_indexing_{unique_id}"
        test_run.track_workspace_prefix(file_prefix)

        # -- Turn 1: Create a file with distinctive content --
        msg1 = (
            f"Create a Python file at /workspace/{file_prefix}.py "
            f"with a function called {func_name} that takes two "
            f"parameters (voltage, resistance) and returns their "
            f"quotient. Add a docstring explaining it calculates "
            f"current using Ohm's law."
        )
        r1 = test_run.chat(msg1, session_id=sid)
        evaluate_response(r1, msg1, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t1_response", r1.content[:300])
        assert r1.content, "Turn 1 returned empty"

        assert_tools_called(
            r1.metadata, ["str_replace_editor"], "Turn 1: create file",
        )

        # -- Turn 2: Index the workspace --
        msg2 = (
            "Index the workspace so I can search for code later. "
            "Show me the indexing stats when done."
        )
        r2 = test_run.chat(msg2, session_id=sid)
        evaluate_response(r2, msg2, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
        assert r2.content, "Turn 2 returned empty"

        iters2 = r2.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t2_iters", iters2)
        assert iters2 >= 1, (
            f"Agent didn't use index tool (iters={iters2}). "
            f"Response: {r2.content[:200]}"
        )
        assert_tools_called(
            r2.metadata, ["index_workspace"], "Turn 2: index",
        )

        # Response should mention indexing results
        content2 = r2.content.lower()
        has_stats = any(w in content2 for w in [
            "index", "file", "scanned", "processed", "function",
            "indexed", "complete", "workspace",
        ])
        trace_test.set_attribute("cca.test.t2_has_stats", has_stats)
        assert has_stats, (
            f"Response doesn't mention indexing: {r2.content[:300]}"
        )

        # -- Turn 3: Search for the newly indexed function --
        msg3 = (
            f"Search the codebase for a function that calculates "
            f"current using Ohm's law. I think it was called something "
            f"like 'zephyr flux'."
        )
        r3 = test_run.chat(msg3, session_id=sid)
        evaluate_response(r3, msg3, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
        assert r3.content, "Turn 3 returned empty"

        assert_tools_called(
            r3.metadata, ["search_codebase"], "Turn 3: search",
        )

        # Should find our distinctive function
        content3 = r3.content.lower()
        found_func = any(w in content3 for w in [
            func_name.lower(), "zephyr_flux", "ohm", unique_id,
        ])
        trace_test.set_attribute("cca.test.t3_found_func", found_func)
        assert found_func, (
            f"Search didn't find the indexed function: {r3.content[:300]}"
        )

    def test_knowledge_search_tiers(self, test_run, trace_test, judge_model):
        """3-turn flow: upload doc -> promote -> search knowledge tiers.

        Turn 1: Upload a document with unique content
        Turn 2: Promote to permanent knowledge
        Turn 3: Search all knowledge sources for the unique term
        """
        sid = f"test-knowledge-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        unique_id = uuid.uuid4().hex[:8]
        unique_term = f"Heliostat_{unique_id}"

        # -- Turn 1: Upload a document --
        doc_content = (
            f"Project {unique_term} Technical Specification:\n"
            f"The {unique_term} system uses dual-axis solar tracking "
            f"with a PID controller running at 100Hz. The mirror array "
            f"consists of 47 hexagonal segments with individual actuators. "
            f"Calibration uses a quad-cell photodetector with 0.1 degree "
            f"accuracy."
        )
        msg1 = (
            f"Please store these technical notes for me, I'll need "
            f"them later:\n\n{doc_content}"
        )
        r1 = test_run.chat(msg1, session_id=sid)
        evaluate_response(r1, msg1, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t1_response", r1.content[:300])
        assert r1.content, "Turn 1 returned empty"

        assert_tools_called(
            r1.metadata, ["upload_document"], "Turn 1: upload",
        )

        # -- Turn 2: Promote to permanent knowledge --
        msg2 = (
            f"Promote the {unique_term} document to permanent project "
            f"knowledge so I can find it in future sessions."
        )
        r2 = test_run.chat(msg2, session_id=sid)
        evaluate_response(r2, msg2, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t2_response", r2.content[:300])
        assert r2.content, "Turn 2 returned empty"

        iters2 = r2.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t2_iters", iters2)
        assert iters2 >= 1, (
            f"Agent didn't use promotion tools (iters={iters2})"
        )

        # -- Turn 3: Search all knowledge tiers --
        msg3 = (
            f"Search all your knowledge sources for information about "
            f"the {unique_term} solar tracking system. What's the "
            f"calibration accuracy?"
        )
        r3 = test_run.chat(msg3, session_id=sid)
        evaluate_response(r3, msg3, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
        assert r3.content, "Turn 3 returned empty"

        iters3 = r3.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t3_iters", iters3)
        assert iters3 >= 1, (
            f"Agent didn't search knowledge (iters={iters3})"
        )

        # Should recall the specific detail
        content3 = r3.content.lower()
        has_recall = any(w in content3 for w in [
            "0.1 degree", "quad-cell", "photodetector",
            unique_term.lower(), "solar tracking", "pid",
        ])
        trace_test.set_attribute("cca.test.t3_has_recall", has_recall)
        assert has_recall, (
            f"Agent didn't recall document content: {r3.content[:300]}"
        )
