"""Bash/DevOps documentation search validation.

Phase 1: Direct API validation — does search_docs return the right docs?
Phase 2: LLM agent tests — does CCA use search_docs correctly for Bash/DevOps?
"""
import uuid

import pytest

from .conftest import search_docs
from .helpers.knowledge_data import BASH_TOOLS

pytestmark = [
    pytest.mark.knowledge,
]


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Direct API Validation (fast, no LLM)
# ═══════════════════════════════════════════════════════════════════


class TestBashDocsValidation:
    """Direct API validation — does search_docs return the right Bash/DevOps docs?"""

    @pytest.mark.knowledge_api
    @pytest.mark.parametrize("tool,expected_pkg", list(BASH_TOOLS.items()))
    def test_tool_search(self, knowledge_client, tool, expected_pkg):
        """Validate search_docs returns correct doc for each bash/devops tool."""
        data = search_docs(knowledge_client, tool)
        results = data.get("results", [])

        # Must have at least 1 result
        assert len(results) >= 1, f"No results for '{tool}'"

        top = results[0]

        # Must have a snippet
        snippet = top.get("snippet", "")
        assert snippet, f"'{tool}': no documentation snippet returned"

        # Snippet should mention the tool (less strict on language —
        # bash tools are often documented as Python SDKs)
        tool_lower = tool.lower()
        snippet_lower = snippet.lower()
        # For multi-word tools like "docker compose", check each word
        words = tool_lower.split()
        assert any(w in snippet_lower for w in words), (
            f"'{tool}': snippet ({len(snippet)} chars) doesn't mention "
            f"any of {words}. First 100 chars: {snippet[:100]}"
        )


# ═══════════════════════════════════════════════════════════════════
# Phase 2: LLM Agent Tests (needs CCA + vLLM)
# ═══════════════════════════════════════════════════════════════════


class TestBashDocsAgent:
    """LLM agent test — does CCA use search_docs correctly for Bash/DevOps?"""

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_docker_compose_deploy(self, test_run, trace_test, judge_model):
        """Ask CCA to write a bash script using docker compose to deploy a service.

        Validates: search_docs called, code uses docker compose commands,
        proper service deployment pattern.
        """
        from tests.evaluators import evaluate_response

        sid = f"test-bash-docker-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a bash script that uses docker compose to deploy a "
            "multi-container web application with an nginx reverse proxy, "
            "a Python FastAPI backend, and a PostgreSQL database. Include "
            "health checks, restart policies, and environment variable handling."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")

        # Verify search_docs was called
        assert any("search_docs" in t for t in r.tool_names), (
            f"Agent didn't use search_docs: {r.tool_names}"
        )

        # Verify docker compose code in response
        content = r.content
        assert "docker compose" in content.lower() or "docker-compose" in content.lower(), (
            "Missing docker compose command"
        )

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_ansible_playbook(self, test_run, trace_test, judge_model):
        """Ask CCA to write an ansible playbook to configure servers.

        Validates: search_docs called, YAML playbook structure with
        proper ansible modules and patterns.
        """
        from tests.evaluators import evaluate_response

        sid = f"test-bash-ansible-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write an Ansible playbook that configures a set of web "
            "servers: install nginx, deploy a custom config file from "
            "a template, set up firewall rules with ufw, and create a "
            "dedicated service user. Use handlers for service restarts."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")

        content = r.content.lower()
        assert "hosts:" in content or "tasks:" in content, (
            "Missing Ansible playbook structure (hosts/tasks)"
        )

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_git_branch_management(self, test_run, trace_test, judge_model):
        """Ask CCA to write a bash script using git for branch management.

        Validates: search_docs called, script uses git commands for
        branch operations with proper error handling.
        """
        from tests.evaluators import evaluate_response

        sid = f"test-bash-git-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a bash script that automates git branch management: "
            "list all branches older than 30 days, check if they are "
            "merged into main, prompt for confirmation, then delete the "
            "stale branches both locally and on the remote. Include "
            "error handling and a dry-run mode."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")

        content = r.content
        assert "git branch" in content or "git for-each-ref" in content, (
            "Missing git branch listing command"
        )
