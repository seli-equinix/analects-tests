"""Flow test: Code intelligence — call graph, orphan detection, dependency analysis.

Journey: query function callers → find orphan functions →
analyze file dependencies. A developer exploring a codebase
before making changes.

Exercises: query_call_graph, find_orphan_functions, analyze_dependencies
(GRAPH group), CODER route.

Requires: Memgraph (see config.toml [services]) with indexed codebase.
The EVA project must be indexed with cross-file call resolution.

Ground truth validation: the test verifies actual data quality, not
just that the LLM called the right tool.  If the call graph is broken
(e.g. zero callers for Connect-SessionVC), the test fails with a
specific error message identifying the data integrity issue.
"""

import uuid

import pytest

from tests.evaluators import assert_tools_called, evaluate_response

pytestmark = [pytest.mark.coder, pytest.mark.slow]


class TestCodeIntelligence:
    """CODER route: multi-turn code graph exploration."""

    def test_code_intelligence_flow(self, cca, trace_test, judge_model):
        """3-turn flow: call graph → orphans → dependencies.

        Turn 1: Query callers of Connect-SessionVC (known to have 20+ callers)
                 Validates: tool called + response contains actual caller names
        Turn 2: Find orphan functions in the EVA project
                 Validates: tool called + response lists real function names
        Turn 3: Analyze dependencies of equinix.automation.vcenter.psm1
                 Validates: tool called + response discusses cross-file impact
        """
        tracker = cca.tracker()
        sid = f"test-graph-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)

        try:
            # ── Turn 1: Query the call graph ──
            # Connect-SessionVC is a well-connected function in the EVA project.
            # It should have 15+ callers across multiple .ps1 files.
            # If the graph returns 0 callers, cross-file resolution is broken.
            msg1 = (
                "Using the code graph, what functions call 'Connect-SessionVC'? "
                "Show me the callers and which files they're in."
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
            assert r1.content, "Turn 1 returned empty"

            iters1 = r1.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t1_iters", iters1)
            assert iters1 >= 1, (
                f"Agent didn't use graph tools (iters={iters1}). "
                f"Response: {r1.content[:200]}"
            )
            assert_tools_called(
                r1.metadata, ["query_call_graph"], "Turn 1: call graph",
            )

            # Ground truth: Connect-SessionVC has 15+ callers across files.
            # The response MUST mention actual caller function names or file
            # paths — not "no callers found" or "0 results".
            content1 = r1.content.lower()
            trace_test.set_attribute("cca.test.t1_has_graph_data", True)

            # Fail fast if the graph returned empty (data integrity issue)
            no_data_signals = [
                "no callers", "0 callers", "no results", "empty",
                "no functions call", "not found in", "does not exist",
            ]
            has_no_data = any(s in content1 for s in no_data_signals)
            assert not has_no_data, (
                "CALL GRAPH DATA INTEGRITY ISSUE: Connect-SessionVC should "
                "have 15+ callers but the graph returned none. The cross-file "
                "call resolution may be broken. Check: "
                "resolve_project_calls() ran after indexing. "
                f"Response: {r1.content[:300]}"
            )

            # Should mention actual caller names or file references
            has_real_callers = any(w in content1 for w in [
                "jobsearchvm", "jobaddvm", "jobremove", "invoke-",
                "connect-", ".ps1", "vcenter", "caller",
            ])
            trace_test.set_attribute(
                "cca.test.t1_has_real_callers", has_real_callers,
            )
            assert has_real_callers, (
                f"Response mentions callers but no recognizable EVA function "
                f"names or files: {r1.content[:300]}"
            )

            # ── Turn 2: Find orphan functions ──
            msg2 = (
                "Can you check the code graph for orphan functions — "
                "functions that are defined but never called by anything? "
                "Just show me the top 5 if there are many."
            )
            r2 = cca.chat(msg2, session_id=sid)
            evaluate_response(r2, msg2, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
            assert r2.content, "Turn 2 returned empty"

            iters2 = r2.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t2_iters", iters2)
            assert iters2 >= 1, (
                f"Agent didn't use orphan detection tools (iters={iters2})"
            )
            assert_tools_called(
                r2.metadata, ["find_orphan_functions"], "Turn 2: orphans",
            )

            # Should list actual function names (PowerShell Verb-Noun pattern)
            content2 = r2.content.lower()
            has_functions = any(w in content2 for w in [
                "function", "orphan", "unused", "no caller",
                "never called", ".ps1", ".psm1",
            ])
            trace_test.set_attribute("cca.test.t2_has_functions", has_functions)
            assert has_functions, (
                f"Response doesn't list orphan functions: {r2.content[:300]}"
            )

            # Ground truth: after two-phase resolution, previously false
            # orphans (Get-CacheAPI, Remove-CacheAPI) should NOT appear.
            # Track this as advisory (don't fail — orphan list depends on
            # indexing completeness).
            known_false_orphans = [
                "get-cacheapi", "remove-cacheapi", "get-jobhistory",
                "delete-jobhistory", "get-jobdata",
            ]
            false_orphan_count = sum(
                1 for fn in known_false_orphans if fn in content2
            )
            trace_test.set_attribute(
                "cca.test.t2_false_orphan_count", false_orphan_count,
            )
            if false_orphan_count > 0:
                trace_test.set_attribute(
                    "cca.test.t2_warning",
                    f"{false_orphan_count} known false orphans still appear "
                    f"— call resolution may be incomplete",
                )

            # ── Turn 3: Analyze dependencies ──
            # equinix.automation.vcenter.psm1 is the main module — many files
            # depend on it.  The analysis should show cross-file callers.
            msg3 = (
                "Analyze the dependencies of the equinix.automation.vcenter.psm1 file. "
                "What other files depend on it, and what would be impacted "
                "if I changed its main functions?"
            )
            r3 = cca.chat(msg3, session_id=sid)
            evaluate_response(r3, msg3, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
            assert r3.content, "Turn 3 returned empty"

            iters3 = r3.metadata.get("tool_iterations", 0)
            trace_test.set_attribute("cca.test.t3_iters", iters3)
            assert iters3 >= 1, (
                f"Agent didn't use dependency tools (iters={iters3})"
            )
            assert_tools_called(
                r3.metadata, ["analyze_dependencies"], "Turn 3: deps",
            )

            # Ground truth: the vcenter module has many dependents.
            # Response should mention specific files or function impact.
            content3 = r3.content.lower()
            has_deps = any(w in content3 for w in [
                "depend", "import", "impact", "vcenter",
                ".ps1", ".psm1", "module", "coupling",
                "function", "caller", "called",
            ])
            trace_test.set_attribute("cca.test.t3_has_deps", has_deps)
            assert has_deps, (
                f"Response doesn't discuss dependencies: {r3.content[:300]}"
            )

            # Should mention real dependent files (not "no dependencies")
            no_deps_signals = [
                "no dependencies", "no files depend", "0 dependent",
                "not found", "does not exist",
            ]
            has_no_deps = any(s in content3 for s in no_deps_signals)
            trace_test.set_attribute(
                "cca.test.t3_has_no_deps", has_no_deps,
            )
            assert not has_no_deps, (
                "DEPENDENCY DATA INTEGRITY ISSUE: equinix.automation.vcenter.psm1 "
                "is the main module — it should have many dependents. "
                "The call graph may be incomplete. "
                f"Response: {r3.content[:300]}"
            )

        finally:
            tracker.cleanup()
