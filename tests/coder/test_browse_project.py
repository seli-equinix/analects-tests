"""Flow test: Browse project structure via vectorless RAG tools.

Journey: list indexed projects → browse EVA project overview →
drill into a directory → browse a specific file.

Exercises: browse_project, list_projects (PROJECT_TREE group),
CODER route.

Requires: Indexed EVA project in Qdrant + Memgraph.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.coder, pytest.mark.slow]


class TestBrowseProject:
    """CODER route: browse project tree top-down without vector search."""

    def test_browse_project(self, cca, trace_test, judge_model):
        """4-turn flow: list projects → overview → directory → file.

        Turn 1: List all indexed projects
        Turn 2: Browse EVA project overview (directory tree)
        Turn 3: Drill into a specific directory
        Turn 4: Browse a specific file (function detail)
        """
        tracker = cca.tracker()
        sid = f"test-browse-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)

        try:
            # ── Turn 1: List indexed projects ──
            msg1 = (
                "What projects are indexed in the workspace? "
                "List them with their file and function counts."
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
            assert r1.content, "Turn 1 returned empty"

            # Must use project tree or search tools
            tool_names_1 = r1.tool_names
            trace_test.set_attribute("cca.test.t1_tools", str(tool_names_1))
            assert any(
                t in name for name in tool_names_1
                for t in ["list_projects", "browse_project", "search_codebase"]
            ), (
                f"No project discovery tools called. Tools: {tool_names_1}"
            )

            # EVA must appear in the response
            assert "eva" in r1.content.lower(), (
                f"EVA project not found in response: {r1.content[:300]}"
            )

            # ── Turn 2: Browse EVA project overview ──
            msg2 = (
                "Show me the directory structure of the EVA project. "
                "What modules and directories does it have?"
            )
            r2 = cca.chat(msg2, session_id=sid)
            evaluate_response(r2, msg2, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
            assert r2.content, "Turn 2 returned empty"

            # Must use browse_project
            tool_names_2 = r2.tool_names
            trace_test.set_attribute("cca.test.t2_tools", str(tool_names_2))
            assert "browse_project" in tool_names_2, (
                f"browse_project not called. Tools: {tool_names_2}"
            )

            # Response should describe project structure
            content2 = r2.content.lower()
            has_structure = any(w in content2 for w in [
                "scripts", "code", "directory", "files", "functions",
                "powershell", "python",
            ])
            assert has_structure, (
                f"Response doesn't describe project structure: "
                f"{r2.content[:300]}"
            )

            # ── Turn 3: Drill into scripts directory ──
            msg3 = (
                "Browse the scripts directory in EVA. "
                "What functions are in there?"
            )
            r3 = cca.chat(msg3, session_id=sid)
            evaluate_response(r3, msg3, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
            assert r3.content, "Turn 3 returned empty"

            # Should mention functions or file names
            content3 = r3.content.lower()
            has_functions = any(w in content3 for w in [
                "function", ".ps1", ".psm1", "script",
                "invoke", "get-", "set-", "add-",
            ])
            assert has_functions, (
                f"Response doesn't show functions: {r3.content[:300]}"
            )

            # ── Turn 4: Browse a specific file ──
            msg4 = (
                "Show me the details of JobStart.ps1 in the EVA project — "
                "what functions does it contain?"
            )
            r4 = cca.chat(msg4, session_id=sid)
            evaluate_response(r4, msg4, trace_test, judge_model, "coder")

            trace_test.set_attribute("cca.test.t4_response", r4.content[:500])
            assert r4.content, "Turn 4 returned empty"

            # Should mention JobStart-specific content
            content4 = r4.content.lower()
            has_jobstart = any(w in content4 for w in [
                "jobstart", "param", "function", "json",
            ])
            assert has_jobstart, (
                f"Response doesn't describe JobStart.ps1: "
                f"{r4.content[:300]}"
            )

        finally:
            tracker.cleanup()
