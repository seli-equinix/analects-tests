"""Flow test: Multi-language knowledge validation.

Journey: developer works across technology stacks -> CCA searches
curated docs from multiple domains (Python + PowerShell, Bash + Nutanix,
Ansible + FastAPI + Docker) -> synthesizes cross-domain solutions.

Phase 2 ONLY: LLM agent tests that require docs from multiple languages.

Exercises: search_docs (API), get_api_docs (API), str_replace_editor (FILE),
bash (SHELL), CODER route.
Markers: knowledge, knowledge_agent, slow
"""
import uuid

import pytest

from tests.evaluators import evaluate_response

from .conftest import assert_content_or_file

pytestmark = [
    pytest.mark.knowledge,
]


# ═══════════════════════════════════════════════════════════════════
# Phase 2: LLM Agent Tests (needs CCA + vLLM)
# ═══════════════════════════════════════════════════════════════════


class TestMultiLanguageAgent:
    """LLM agent test — does CCA combine knowledge from multiple doc sets?"""

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_python_to_powershell_conversion(self, test_run, trace_test, judge_model):
        """Ask CCA to convert a Python FastAPI endpoint to a PowerShell client.

        Needs both Python and PowerShell docs — validates CCA can pull
        from multiple knowledge domains in one response.
        """
        sid = f"test-multi-py-ps-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Convert this Python FastAPI endpoint to a PowerShell "
            "Invoke-RestMethod client that calls it:\n\n"
            "@app.post('/api/items')\n"
            "async def create_item(name: str, price: float, "
            "tags: list[str] = []):\n"
            "    return {'id': 1, 'name': name, 'price': price, "
            "'tags': tags}\n\n"
            "The PowerShell client should handle authentication with "
            "an API key header, send JSON, and parse the response."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")
        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        # Verify search_docs was called (needs docs from multiple domains)
        assert any("search_docs" in t for t in r.tool_names), (
            f"Agent didn't use search_docs: {r.tool_names}"
        )

        assert_content_or_file(
            r,
            ["Invoke-RestMethod", "Invoke-WebRequest", "PowerShell"],
            "PowerShell client",
        )

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_bash_nutanix_api_curl(self, test_run, trace_test, judge_model):
        """Ask CCA to write a bash script calling the Nutanix API with curl.

        Needs bash + Nutanix docs — validates CCA combines shell scripting
        knowledge with Nutanix API documentation.
        """
        sid = f"test-multi-bash-ntnx-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a bash script that calls the Nutanix Prism Central "
            "REST API using curl to list all VMs, then parses the JSON "
            "response with jq to extract each VM's name, UUID, power "
            "state, and IP addresses. Use basic auth and handle "
            "pagination if the response has more than 20 VMs."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")
        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        assert_content_or_file(r, ["curl", "jq", "json", "nutanix"], "Bash + Nutanix curl")

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_ansible_fastapi_docker_deploy(self, test_run, trace_test, judge_model):
        """Ask CCA to create a deployment script using ansible, Python, and docker.

        Needs all three doc domains — validates CCA can synthesize
        ansible playbook patterns, Docker compose configuration, and
        Python FastAPI application knowledge together.
        """
        sid = f"test-multi-all-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Create a deployment script that uses Ansible to deploy a "
            "Python FastAPI application with docker compose to a set of "
            "remote servers. The Ansible playbook should: clone the git "
            "repo, build the Docker image, create a docker-compose.yml "
            "with the FastAPI app and a Redis cache, and start the "
            "services. Include health check verification after deploy."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")
        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        assert_content_or_file(
            r,
            ["hosts:", "tasks:", "playbook", "ansible", "Ansible",
             "docker compose", "docker-compose", "fastapi", "uvicorn", "FastAPI"],
            "Ansible + FastAPI + Docker",
        )
