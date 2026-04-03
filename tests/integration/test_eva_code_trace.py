"""Flow test: EVA project code trace — full pipeline validation.

Journey: identify as user → search EVA PowerShell files →
trace execution path → assemble standalone code. A developer
preparing to understand and extract VM automation code.

Exercises the ENTIRE code intelligence pipeline end-to-end:
  Nextcloud → GitLab → workspace-sync → tree-sitter AST parse →
  Qdrant vectors → Memgraph call graph → LLM tools → user output.

Exercises: search_codebase (CODE_SEARCH), trace_execution,
assemble_traced_code (TRACE), str_replace_editor (FILE), CODER route.

Requires: Indexed EVA project in Qdrant + Memgraph (PowerShell files
from /workspace/EVA/code/).
"""

import uuid

import pytest

from tests.evaluators import assert_tools_called, evaluate_response

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestEvaCodeTrace:
    """Real-world EVA scenario: trace PowerShell VM automation code."""

    def test_eva_powershell_trace_and_assemble(self, test_run, trace_test, judge_model):
        """3-turn flow: find EVA files → trace execution → assemble code.

        Validates that the full chain works: workspace files indexed →
        tree-sitter parsed PowerShell → Qdrant has vectors → Memgraph
        has call graph → LLM uses trace/search/assemble tools.

        Turn 1: Identify + ask about EVA project files
        Turn 2: Trace execution path for a Linux VM build
        Turn 3: Assemble all functions into a standalone file
        """
        sid = f"test-eva-trace-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)
        test_run.track_user("Sean")

        # Pre-check: Sean must exist in the user database for memory tools
        user = test_run.client.find_user_by_name("Sean")
        if not user:
            pytest.skip("User 'Sean' not found in Qdrant — seed required")

        # ── Turn 1: Identify + find EVA project files ──
        # Natural request: developer wants to understand what's indexed
        msg1 = (
            "Hello this is Sean. In your knowledge on the EVA Project, "
            "find me the files equinix.automation.vcenter.psm1 and "
            "jobstart.ps1. Tell me what functions are in each file."
        )
        r1 = test_run.chat(msg1, session_id=sid)
        evaluate_response(r1, msg1, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
        assert r1.content, "Turn 1 returned empty"

        # Should have been identified
        trace_test.set_attribute("cca.test.t1_user_identified", r1.user_identified)
        assert r1.user_identified, (
            "Sean should be identified as a known user"
        )

        # Should have used search tools to find EVA files
        iters1 = r1.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t1_iters", iters1)
        assert iters1 >= 1, (
            f"Agent didn't use search tools (iters={iters1}). "
            f"Response: {r1.content[:200]}"
        )

        # Check that agent used code intelligence tools (search, browse, or analysis)
        tool_names_1 = r1.tool_names
        trace_test.set_attribute("cca.test.t1_tools", str(tool_names_1))
        used_search = any(
            t in name for name in tool_names_1
            for t in ["search_codebase", "search_knowledge", "browse_project",
                       "analyze_dependencies", "query_call_graph"]
        )
        assert used_search, (
            f"Agent didn't use code intelligence tools. Called: {tool_names_1}"
        )

        # Response should mention EVA-specific content
        content1 = r1.content.lower()
        has_eva_content = any(w in content1 for w in [
            "jobstart", "vcenter", "psm1", "powershell",
            "searchvm", "addvmfromtemplate", "eva",
        ])
        trace_test.set_attribute("cca.test.t1_has_eva_content", has_eva_content)
        assert has_eva_content, (
            f"Response doesn't mention EVA files or functions: "
            f"{r1.content[:300]}"
        )

        # ── Turn 2: Trace execution for Linux VM build ──
        # Developer wants to understand the call chain
        msg2 = (
            "Trace the execution path when JobStart.ps1 processes a "
            "Linux VM build request. Show me which functions get called "
            "and in what order — SearchVM, AddVMFromTemplate, "
            "InvokeIPAddressUpdate, etc."
        )
        r2 = test_run.chat(msg2, session_id=sid)
        evaluate_response(r2, msg2, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
        assert r2.content, "Turn 2 returned empty"

        iters2 = r2.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t2_iters", iters2)
        assert iters2 >= 1, (
            f"Agent didn't use trace tools (iters={iters2}). "
            f"Response: {r2.content[:200]}"
        )

        # Should have used trace_execution or query_call_graph
        tool_names_2 = r2.tool_names
        trace_test.set_attribute("cca.test.t2_tools", str(tool_names_2))
        used_trace = any(
            t in name for name in tool_names_2
            for t in ["trace_execution", "query_call_graph"]
        )
        assert used_trace, (
            f"Agent didn't use trace/graph tools. Called: {tool_names_2}"
        )

        # Response should show execution chain with function names
        content2 = r2.content.lower()
        has_trace_data = any(w in content2 for w in [
            "searchvm", "addvmfromtemplate", "jobstart",
            "function", "calls", "execution", "trace",
        ])
        trace_test.set_attribute("cca.test.t2_has_trace_data", has_trace_data)
        assert has_trace_data, (
            f"Response doesn't show execution trace: {r2.content[:300]}"
        )

        # ── Turn 3: Assemble into standalone file ──
        # Developer wants a single .ps1 with all needed functions
        msg3 = (
            "Now give me a single standalone .ps1 file that contains "
            "ALL the functions needed to run both a Linux and Windows "
            "VM build from JobStart.ps1. Do NOT change or refactor any "
            "code — I need the original functions exactly as they are. "
            "This code is in production."
        )
        r3 = test_run.chat(msg3, session_id=sid)
        evaluate_response(r3, msg3, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
        assert r3.content, "Turn 3 returned empty"

        iters3 = r3.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t3_iters", iters3)
        assert iters3 >= 1, (
            f"Agent didn't use assemble/edit tools (iters={iters3}). "
            f"Response: {r3.content[:200]}"
        )

        # Should have used assemble_traced_code or str_replace_editor
        tool_names_3 = r3.tool_names
        trace_test.set_attribute("cca.test.t3_tools", str(tool_names_3))
        used_assemble_or_edit = any(
            t in name for name in tool_names_3
            for t in ["assemble_traced_code", "str_replace_editor"]
        )
        assert used_assemble_or_edit, (
            f"Agent didn't use assemble or file tools. Called: {tool_names_3}"
        )

        # Response should contain PowerShell function definitions
        content3 = r3.content.lower()
        has_ps_code = any(w in content3 for w in [
            "function ", "param(", ".ps1", "powershell",
            "searchvm", "addvmfromtemplate",
        ])
        trace_test.set_attribute("cca.test.t3_has_ps_code", has_ps_code)
        assert has_ps_code, (
            f"Response doesn't contain PowerShell code: "
            f"{r3.content[:300]}"
        )

        # Should NOT mention refactoring (user said don't change code)
        refactor_words = ["refactor", "improved", "optimized", "cleaned up"]
        mentions_refactor = any(w in content3 for w in refactor_words)
        trace_test.set_attribute(
            "cca.test.t3_mentions_refactor", mentions_refactor,
        )
        # Advisory — don't fail on this, just track it
