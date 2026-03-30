"""Negative tests: agent should fail gracefully, not hallucinate.

These tests ask the agent to do things that should NOT succeed —
nonexistent functions, nonexistent projects, invalid operations.
The agent should report the error clearly rather than making up results.

Exercises: browse_project, trace_execution, search_codebase error handling.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.integration]


class TestNegativeCases:
    """Verify graceful failure instead of hallucination."""

    def test_nonexistent_function_trace(self, test_run, trace_test, judge_model):
        """Agent should report 'not found' for a nonexistent function."""
        sid = f"test-neg-func-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Trace the execution path of the function "
            "FakeNonExistent_XYZ123 in the EVA project. "
            "Show me its callers and callees."
        )
        r1 = test_run.chat(msg, session_id=sid)
        evaluate_response(r1, msg, trace_test, judge_model, "coder")

        assert r1.content, "Response empty"

        # Agent should acknowledge the function doesn't exist
        content_lower = r1.content.lower()
        acknowledges_missing = any(w in content_lower for w in [
            "not found", "doesn't exist", "does not exist",
            "no function", "could not find", "couldn't find",
            "no results", "no match", "unable to find",
        ])
        trace_test.set_attribute(
            "cca.test.acknowledges_missing", acknowledges_missing,
        )
        assert acknowledges_missing, (
            f"Agent didn't report function as missing — may be "
            f"hallucinating: {r1.content[:500]}"
        )

    def test_nonexistent_project_browse(self, test_run, trace_test, judge_model):
        """Agent should report error for a nonexistent project."""
        sid = f"test-neg-proj-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Browse the project structure of FakeProject_ABC999. "
            "Show me its directories and files."
        )
        r1 = test_run.chat(msg, session_id=sid)
        evaluate_response(r1, msg, trace_test, judge_model, "coder")

        assert r1.content, "Response empty"

        # Agent should report the project doesn't exist
        content_lower = r1.content.lower()
        acknowledges_missing = any(w in content_lower for w in [
            "not found", "doesn't exist", "does not exist",
            "no project", "not indexed", "couldn't find",
            "no files", "unable to find",
        ])
        trace_test.set_attribute(
            "cca.test.acknowledges_missing", acknowledges_missing,
        )
        assert acknowledges_missing, (
            f"Agent didn't report project as missing — may be "
            f"hallucinating: {r1.content[:500]}"
        )
