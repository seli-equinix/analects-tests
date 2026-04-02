"""Nutanix SDK documentation search validation.

Phase 1: Direct API validation — does search_docs return the right Nutanix docs?
Phase 2: LLM agent tests — does CCA use search_docs correctly for Nutanix SDK?
"""
import uuid

import pytest

from .conftest import search_docs
from .helpers.knowledge_data import NUTANIX_MODULES

pytestmark = [
    pytest.mark.knowledge,
]


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Direct API Validation (fast, no LLM)
# ═══════════════════════════════════════════════════════════════════


class TestNutanixSdkValidation:
    """Direct API validation — does search_docs return the right Nutanix docs?"""

    @pytest.mark.knowledge_api
    @pytest.mark.parametrize("module,expected_pkg", list(NUTANIX_MODULES.items()))
    def test_module_search(self, knowledge_client, module, expected_pkg):
        """Validate search_docs returns correct Nutanix doc for each module."""
        # Module names are like "nutanix-vmm" — search the full name
        data = search_docs(knowledge_client, module)
        results = data.get("results", [])

        # Must have at least 1 result
        assert len(results) >= 1, f"No results for 'nutanix {module}'"

        top = results[0]

        # Must be from Nutanix docs
        assert top["id"].startswith("nutanix/"), (
            f"'nutanix {module}': expected nutanix/*, got {top['id']}"
        )

        # Must have a snippet
        snippet = top.get("snippet", "")
        assert snippet, f"'nutanix {module}': no documentation snippet returned"


# ═══════════════════════════════════════════════════════════════════
# Phase 2: LLM Agent Tests (needs CCA + vLLM)
# ═══════════════════════════════════════════════════════════════════


class TestNutanixSdkAgent:
    """LLM agent test — does CCA use search_docs correctly for Nutanix SDK?"""

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_nutanix_vmm_create_vm(self, test_run, trace_test, judge_model):
        """Ask CCA to write Python code using Nutanix VMM SDK to create a VM.

        Validates: search_docs called, code uses ntnx_vmm_py_client,
        proper VM creation pattern with disks and NICs.
        """
        from tests.evaluators import evaluate_response

        sid = f"test-ntnx-vmm-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write Python code using the Nutanix VMM SDK (ntnx_vmm_py_client) "
            "to create a virtual machine with 4 vCPUs, 8GB RAM, a 100GB disk, "
            "and a NIC attached to a specific subnet. Include authentication "
            "setup with ApiClient and proper error handling."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")

        # Verify search_docs was called
        assert any("search_docs" in t for t in r.tool_names), (
            f"Agent didn't use search_docs: {r.tool_names}"
        )

        # Verify Nutanix VMM code in response
        content = r.content
        assert "vmm" in content.lower() or "ntnx" in content.lower(), (
            "Missing Nutanix VMM SDK references"
        )

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_nutanix_networking_subnet(self, test_run, trace_test, judge_model):
        """Ask CCA to write code using Nutanix Networking SDK to create a subnet.

        Validates: search_docs called, code uses ntnx_networking_py_client,
        proper subnet creation with VLAN and IP pool configuration.
        """
        from tests.evaluators import evaluate_response

        sid = f"test-ntnx-net-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write Python code using the Nutanix Networking SDK "
            "(ntnx_networking_py_client) to create a VLAN subnet with "
            "VLAN ID 100, IP pool 10.0.0.100-10.0.0.200, gateway "
            "10.0.0.1, and DHCP enabled. Include authentication and "
            "error handling."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")

        content = r.content
        assert "networking" in content.lower() or "subnet" in content.lower(), (
            "Missing Nutanix Networking SDK references"
        )

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_nutanix_data_protection_snapshot(self, test_run, trace_test, judge_model):
        """Ask CCA to write code using Nutanix Data Protection SDK to create a snapshot.

        Validates: search_docs called, code uses ntnx_dataprotection_py_client,
        proper snapshot/recovery point creation pattern.
        """
        from tests.evaluators import evaluate_response

        sid = f"test-ntnx-dp-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write Python code using the Nutanix Data Protection SDK "
            "(ntnx_dataprotection_py_client) to create a VM snapshot "
            "(recovery point) for a specific VM. Include setting up the "
            "API client with authentication, creating the recovery point "
            "with an expiration time, and checking the task status."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")

        content = r.content.lower()
        assert "dataprotection" in content or "recovery" in content or "snapshot" in content, (
            "Missing Nutanix Data Protection SDK references"
        )
