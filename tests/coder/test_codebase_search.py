"""Flow test: Codebase search via the CODER route.

Journey: developer explores the EVA project codebase -- searches for
VM snapshot functions, VCenter session management, then analyzes
cross-file dependencies.

Uses real EVA project functions that exist in the indexed workspace.
Validates that the LLM uses search_codebase (not bash grep) and
returns structured results from the AST-indexed knowledge graph.

Exercises: search_codebase, search_knowledge (CODE_SEARCH group),
analyze_dependencies (GRAPH group), CODER route.

Requires: Indexed EVA project in Qdrant + Memgraph.
"""

import uuid

import pytest

from tests.evaluators import assert_tools_called, evaluate_response

pytestmark = [pytest.mark.coder, pytest.mark.slow]


class TestCodebaseSearch:
    """CODER route: search indexed codebase for functions and patterns."""

    def test_codebase_search(self, test_run, trace_test, judge_model):
        """3-turn flow: search functions -> search patterns -> analyze deps.

        Turn 1: Search for VM snapshot functions in EVA (real indexed functions)
        Turn 2: Search for VCenter connection management
        Turn 3: Analyze cross-file dependencies of the main module
        """
        sid = f"test-codesearch-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        # -- Turn 1: Search for real EVA functions --
        # The EVA project has VM snapshot functions (JobAddVMSnapShotVM etc.)
        # indexed in Qdrant.  This validates search_codebase finds them.
        msg1 = (
            "I need to understand the VM snapshot operations in the EVA "
            "project. Search the codebase for functions related to "
            "snapshots -- how are VM snapshots created and managed?"
        )
        r1 = test_run.chat(msg1, session_id=sid)
        evaluate_response(r1, msg1, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
        assert r1.content, "Turn 1 returned empty"

        iters1 = r1.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t1_iters", iters1)
        assert iters1 >= 1, (
            f"Agent didn't use tools (iters={iters1}). "
            f"Response: {r1.content[:200]}"
        )

        # Must use search_codebase (NOT bash grep)
        tool_names_1 = r1.tool_names
        trace_test.set_attribute("cca.test.t1_tools", str(tool_names_1))
        assert any("search_codebase" in name for name in tool_names_1), (
            f"search_codebase not called for snapshot search. "
            f"Tools: {tool_names_1}"
        )

        # Response should mention snapshot-related content
        content1 = r1.content.lower()
        has_snapshot = any(w in content1 for w in [
            "snapshot", "snap", ".ps1", "function",
            "jobaddvmsnapshot", "vcenter", "powershell",
        ])
        trace_test.set_attribute("cca.test.t1_has_snapshot", has_snapshot)
        assert has_snapshot, (
            f"Response doesn't mention snapshot functions: "
            f"{r1.content[:300]}"
        )

        # -- Turn 2: Search for VCenter session management --
        msg2 = (
            "Now search for how VCenter connections are managed in the "
            "codebase. Which files handle VCenter session management? "
            "I'm looking for the connection setup functions."
        )
        r2 = test_run.chat(msg2, session_id=sid)
        evaluate_response(r2, msg2, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
        assert r2.content, "Turn 2 returned empty"

        iters2 = r2.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t2_iters", iters2)
        assert iters2 >= 1, (
            f"Agent didn't use tools for VCenter query (iters={iters2})"
        )

        # Should mention VCenter/connection content
        content2 = r2.content.lower()
        has_vcenter = any(w in content2 for w in [
            "vcenter", "connect", "session", ".psm1", ".ps1",
            "powershell", "function",
        ])
        trace_test.set_attribute("cca.test.t2_has_vcenter", has_vcenter)
        assert has_vcenter, (
            f"Response doesn't mention VCenter: {r2.content[:300]}"
        )

        # -- Turn 3: Analyze dependencies of main module --
        msg3 = (
            "For the main VCenter module equinix.automation.vcenter.psm1 "
            "-- how many other files depend on it? What would be impacted "
            "if I changed its main functions?"
        )
        r3 = test_run.chat(msg3, session_id=sid)
        evaluate_response(r3, msg3, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
        assert r3.content, "Turn 3 returned empty"

        iters3 = r3.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t3_iters", iters3)
        assert iters3 >= 1, (
            f"Agent didn't use tools for deps query (iters={iters3})"
        )

        # Should discuss dependencies/impact
        content3 = r3.content.lower()
        has_deps = any(w in content3 for w in [
            "depend", "import", "impact", "caller", "called",
            ".ps1", ".psm1", "file", "function", "vcenter",
        ])
        trace_test.set_attribute("cca.test.t3_has_deps", has_deps)
        assert has_deps, (
            f"Response doesn't discuss dependencies: "
            f"{r3.content[:300]}"
        )
