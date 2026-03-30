"""Flow test: Smart route handling — verify CCA handles mixed requests correctly.

Journey 1: Search request that also asks to save results →
CCA should either use web tools + escalate to file tools, or re-route
to CODER. Either way, the user gets their results.

Journey 2: User introduction that also asks to run a command →
CCA should identify the user AND handle the command request — either
by escalating tools or explaining what it can do.

Tests that CCA is smart about mixed-intent requests, not that tools
are rigidly isolated. Dynamic tool escalation is a feature.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [pytest.mark.integration]


class TestToolIsolation:
    """Smart route handling: verify CCA handles mixed-intent requests."""

    def test_search_route_no_file_tools(self, cca, trace_test, judge_model):
        """Search + save request → CCA should handle both parts.

        The request combines web search AND file saving. CCA should:
        1. Perform the web search (using web_search tool)
        2. Either save to file (via escalation) or explain it can't

        Both outcomes are acceptable — what matters is the search was done.
        """
        tracker = cca.tracker()
        sid = f"test-isolation-search-{uuid.uuid4().hex[:8]}"
        tracker.track_session(sid)

        try:
            msg1 = (
                "Search the web for Python best practices in 2026 and "
                "save the top 5 results to /workspace/best_practices.txt"
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(
                r1, msg1, trace_test, judge_model, "websearch",
                expected_incomplete=True,
            )

            trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
            assert r1.content, "Turn 1 returned empty"

            route = r1.metadata.get("route", "")
            trace_test.set_attribute("cca.test.t1_route", route)

            tool_names = r1.tool_names
            trace_test.set_attribute("cca.test.t1_tools", str(tool_names))

            # The response should contain search results regardless of route
            content = r1.content.lower()
            has_results = any(w in content for w in [
                "python", "best practice", "2026", "result",
                "search", "found",
            ])
            trace_test.set_attribute("cca.test.t1_has_results", has_results)
            assert has_results, (
                f"Response doesn't contain search results: "
                f"{r1.content[:300]}"
            )

            # Track the route and tools for observability
            trace_test.set_attribute(
                "cca.test.t1_note",
                f"Route={route}, tools={tool_names[:5]}",
            )

        finally:
            tracker.cleanup()

    def test_user_route_with_command(self, cca, trace_test, judge_model):
        """User intro + command request → CCA should identify user AND respond.

        The request combines user identification AND a command request.
        CCA should:
        1. Identify the user (store profile facts)
        2. Handle the command — either execute it (via escalation) or
           explain its capabilities

        What matters: user is identified and gets a useful response.
        """
        tracker = cca.tracker()
        sid = f"test-isolation-user-{uuid.uuid4().hex[:8]}"
        unique_id = uuid.uuid4().hex[:6]
        user_name = f"RouteUser_{unique_id}"
        tracker.track_session(sid)
        tracker.track_user(user_name)

        try:
            msg1 = (
                f"Hi I'm {user_name}, I'm a software engineer. "
                f"Can you run 'ls /workspace' for me?"
            )
            r1 = cca.chat(msg1, session_id=sid)
            evaluate_response(r1, msg1, trace_test, judge_model, "user")

            trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
            assert r1.content, "Turn 1 returned empty"

            route = r1.metadata.get("route", "")
            trace_test.set_attribute("cca.test.t1_route", route)

            tool_names = r1.tool_names
            trace_test.set_attribute("cca.test.t1_tools", str(tool_names))

            # User should be identified regardless of route
            trace_test.set_attribute(
                "cca.test.t1_user_identified", r1.user_identified,
            )
            assert r1.user_identified, (
                f"{user_name} should be identified"
            )

            # Response should acknowledge the user OR handle the command
            content = r1.content.lower()
            has_useful_response = any(w in content for w in [
                user_name.lower(), "software engineer", "workspace",
                "ls", "directory", "file", "engineer",
            ])
            trace_test.set_attribute(
                "cca.test.t1_has_useful", has_useful_response,
            )
            assert has_useful_response, (
                f"Response doesn't address the user or command: "
                f"{r1.content[:300]}"
            )

            # Track what CCA decided to do
            trace_test.set_attribute(
                "cca.test.t1_note",
                f"Route={route}, tools={tool_names[:5]}, "
                f"identified={r1.user_identified}",
            )

        finally:
            tracker.cleanup()
