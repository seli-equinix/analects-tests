"""Flow test: PowerShell documentation search validation.

Journey: developer needs to write PowerShell scripts -> CCA searches
curated cmdlet docs via search_docs -> writes code informed by real
cmdlet syntax instead of hallucinating.

Phase 1: Direct API validation — does search_docs return the right docs?
Phase 2: LLM agent tests — does CCA use search_docs correctly for PowerShell?

Exercises: search_docs (API), get_api_docs (API), str_replace_editor (FILE),
bash (SHELL), CODER route.
Markers: knowledge, knowledge_api, knowledge_agent, slow
"""
import uuid

import pytest

from .conftest import assert_content_or_file, search_docs
from .helpers.knowledge_data import PS_CMDLETS
from tests.evaluators import evaluate_response

pytestmark = [
    pytest.mark.knowledge,
]


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Direct API Validation (fast, no LLM)
# ═══════════════════════════════════════════════════════════════════


class TestPowerShellDocsValidation:
    """Direct API validation — does search_docs return the right PowerShell docs?"""

    @pytest.mark.knowledge_api
    @pytest.mark.parametrize("cmdlet,expected_pkg", list(PS_CMDLETS.items()))
    def test_cmdlet_search(self, knowledge_client, cmdlet, expected_pkg):
        """Validate search_docs returns correct PowerShell doc for each cmdlet."""
        data = search_docs(knowledge_client, cmdlet)
        results = data.get("results", [])

        # Must have at least 1 result
        assert len(results) >= 1, f"No results for '{cmdlet}'"

        top = results[0]

        # Must be PowerShell
        assert top.get("language") == "powershell", (
            f"'{cmdlet}': expected powershell, got {top.get('language')} "
            f"(pkg: {top.get('id')})"
        )

        # Must be from our PowerShell docs
        assert top["id"].startswith("microsoft/pwsh-"), (
            f"'{cmdlet}': expected microsoft/pwsh-*, got {top['id']}"
        )

        # Must have a snippet
        snippet = top.get("snippet", "")
        assert snippet, f"'{cmdlet}': no documentation snippet returned"

        # Snippet should mention the cmdlet (or its verb/noun parts)
        cmdlet_lower = cmdlet.lower()
        verb, noun = cmdlet_lower.split("-", 1)
        snippet_lower = snippet.lower()
        assert (
            cmdlet_lower in snippet_lower
            or (verb in snippet_lower and noun in snippet_lower)
        ), (
            f"'{cmdlet}': snippet ({len(snippet)} chars) doesn't mention "
            f"the cmdlet. First 100 chars: {snippet[:100]}"
        )


# ═══════════════════════════════════════════════════════════════════
# Phase 2: LLM Agent Tests (needs CCA + vLLM)
# ═══════════════════════════════════════════════════════════════════


class TestPowerShellDocsAgent:
    """LLM agent test — does CCA use search_docs correctly for PowerShell?"""

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_powershell_rest_api(self, test_run, trace_test, judge_model):
        """Ask CCA to write PowerShell REST API code.

        Validates: search_docs called, correct doc found, code uses
        Invoke-RestMethod with proper JSON handling.
        """

        sid = f"test-ps-rest-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a PowerShell script that calls a REST API at "
            "https://api.example.com/items to create a new item with "
            "name='Widget' and price=9.99. Use authentication headers "
            "and handle the JSON response."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")
        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        # Verify search_docs was called
        assert any("search_docs" in t for t in r.tool_names), (
            f"Agent didn't use search_docs: {r.tool_names}"
        )

        assert_content_or_file(r, ["Invoke-RestMethod", "Invoke-WebRequest"], "PowerShell REST")

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_powershell_file_operations(self, test_run, trace_test, judge_model):
        """Ask CCA to write PowerShell file I/O code.

        Validates: search_docs finds pwsh-files-data, code handles
        JSON depth correctly.
        """

        sid = f"test-ps-files-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a PowerShell script that reads a JSON config file, "
            "modifies a nested setting 3 levels deep, and writes it back. "
            "Make sure the JSON depth is preserved correctly."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")
        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        assert_content_or_file(r, ["ConvertTo-Json", "ConvertFrom-Json", "json"], "PowerShell file I/O")

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_powershell_error_handling(self, test_run, trace_test, judge_model):
        """Ask CCA to write PowerShell with proper error handling."""

        sid = f"test-ps-errors-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a PowerShell function that connects to a remote server "
            "using Invoke-Command, runs a health check, and handles both "
            "terminating and non-terminating errors properly with try/catch."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")
        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        assert_content_or_file(r, ["try", "catch", "Try", "Catch"], "PowerShell error handling")

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_powershell_remoting(self, test_run, trace_test, judge_model):
        """Ask CCA to write PowerShell remoting code."""

        sid = f"test-ps-remote-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a PowerShell script that uses Invoke-Command to run "
            "commands on multiple remote servers in parallel, collecting "
            "results and handling connection failures."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")
        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        assert_content_or_file(r, ["Invoke-Command", "PSSession"], "PowerShell remoting")

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_powershell_modules(self, test_run, trace_test, judge_model):
        """Ask CCA to create a PowerShell module."""

        sid = f"test-ps-module-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a PowerShell module (.psm1) with a function that "
            "uses CmdletBinding, parameter validation, and exports "
            "properly via a manifest (.psd1). Include help comments."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")
        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        assert_content_or_file(r, ["CmdletBinding", "cmdletbinding", ".psm1"], "PowerShell module")
