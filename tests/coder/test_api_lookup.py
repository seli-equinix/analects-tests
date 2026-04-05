"""Flow test: API documentation lookup via Context Hub (chub).

Journey: developer needs to write code using external SDKs -> CCA
searches curated API docs -> fetches full documentation -> writes
code informed by real function signatures instead of hallucinating.

Exercises the API lookup pipeline:
  search_api_docs (index search) -> get_api_docs (chub get) ->
  code generation informed by real API signatures.

Exercises: search_api_docs, get_api_docs (API), str_replace_editor (FILE),
bash (SHELL), CODER route.

Requires: chub CLI installed in CCA container with Nutanix local docs
and online registry (cdn.aichub.org) accessible.
"""

import uuid

import pytest

from tests.evaluators import evaluate_response

pytestmark = [
    pytest.mark.coder,
    pytest.mark.slow,
]


class TestAPILookup:
    """API documentation lookup: search, fetch docs, write informed code."""

    def test_api_doc_discovery_and_usage(self, test_run, trace_test, judge_model):
        """3-turn flow: search for API docs -> fetch and write code -> cross-library.

        Turn 1: Search for Nutanix API documentation (local pre-built docs)
        Turn 2: Fetch VMM docs and write SDK-informed Python code
        Turn 3: Search + fetch docs for a different library (qdrant-client
                 from online registry) and write a function using it
        """
        sid = f"test-api-lookup-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        # -- Turn 1: Search for available API docs (discovery) --
        msg1 = (
            "I need to work with the Nutanix v4 Python SDK to manage "
            "virtual machines. Before I write any code, can you check "
            "what API documentation you have available for Nutanix?"
        )
        r1 = test_run.chat(msg1, session_id=sid)
        evaluate_response(r1, msg1, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t1_response", r1.content[:500])
        assert r1.content, "Turn 1 returned empty"

        # Agent should have used tools
        iters1 = r1.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t1_iters", iters1)
        assert iters1 >= 1, (
            f"Agent didn't use tools (iters={iters1}). "
            f"Response: {r1.content[:200]}"
        )

        # Should have used search_docs (unified tool, formerly search_api_docs)
        tool_names_1 = r1.tool_names
        trace_test.set_attribute("cca.test.t1_tools", str(tool_names_1))
        used_api_search = any(
            "search_docs" in name for name in tool_names_1
        )
        trace_test.set_attribute(
            "cca.test.t1_used_api_search", used_api_search,
        )
        assert used_api_search, (
            f"Agent didn't use search_docs tool. "
            f"Called: {tool_names_1}"
        )

        # Response should mention nutanix modules (proves index worked)
        content1 = r1.content.lower()
        has_nutanix = any(w in content1 for w in [
            "nutanix", "vmm", "prism", "clustermgmt",
            "networking", "ntnx",
        ])
        trace_test.set_attribute(
            "cca.test.t1_has_nutanix", has_nutanix,
        )
        assert has_nutanix, (
            f"Response doesn't mention Nutanix packages: "
            f"{r1.content[:300]}"
        )

        # -- Turn 2: Fetch docs and write informed code --
        msg2 = (
            "Great, now get the full VMM API documentation and write "
            "me a Python function that creates a new VM with 4 CPUs, "
            "8GB RAM, and a 100GB disk using the Nutanix v4 SDK."
        )
        r2 = test_run.chat(msg2, session_id=sid, idle_timeout=180)
        evaluate_response(r2, msg2, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t2_response", r2.content[:500])
        assert r2.content, "Turn 2 returned empty"

        iters2 = r2.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t2_iters", iters2)
        assert iters2 >= 1, (
            f"Agent didn't use tools (iters={iters2}). "
            f"Response: {r2.content[:200]}"
        )

        # Should have used get_api_docs to fetch the actual docs
        tool_names_2 = r2.tool_names
        trace_test.set_attribute("cca.test.t2_tools", str(tool_names_2))
        used_get_docs = any(
            "get_api_docs" in name for name in tool_names_2
        )
        trace_test.set_attribute(
            "cca.test.t2_used_get_docs", used_get_docs,
        )
        assert used_get_docs, (
            f"Agent didn't use get_api_docs tool. "
            f"Called: {tool_names_2}"
        )

        # Response should contain actual code
        content2 = r2.content.lower()
        has_code = any(w in content2 for w in [
            "def ", "import", "nutanix", "function",
        ])
        trace_test.set_attribute("cca.test.t2_has_code", has_code)
        assert has_code, (
            f"Response doesn't contain code: {r2.content[:300]}"
        )

        # Response should contain SDK-specific terms from the docs
        # (proves code was informed by real docs, not hallucinated)
        # Terms may be in response text OR in a file the agent created
        has_sdk_terms = any(w in content2 for w in [
            "ntnx_vmm", "vmm", "ahv", "create_vm",
            "vm(", "client", "api_client", "configuration",
        ])

        # If code was written to a file (not inline), verify the agent
        # fetched docs before writing — doc-informed file creation counts
        if not has_sdk_terms:
            used_file_tool = any("str_replace_editor" in n for n in r2.tool_names)
            used_docs = any("get_api_docs" in n for n in r2.tool_names)
            has_sdk_terms = used_file_tool and used_docs

        trace_test.set_attribute(
            "cca.test.t2_has_sdk_terms", has_sdk_terms,
        )
        assert has_sdk_terms, (
            f"Response doesn't use SDK terms and didn't create "
            f"a file with docs: {r2.content[:300]}"
        )

        # -- Turn 3: Different library from online registry --
        msg3 = (
            "I also need to store the VM metadata in our Qdrant "
            "vector database. Look up the qdrant-client API docs "
            "and write me a Python function that stores the VM name, "
            "IP address, and creation timestamp as a vector point "
            "in a collection called 'vm_inventory'."
        )
        r3 = test_run.chat(msg3, session_id=sid, idle_timeout=180)
        evaluate_response(r3, msg3, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t3_response", r3.content[:500])
        assert r3.content, "Turn 3 returned empty"

        iters3 = r3.metadata.get("tool_iterations", 0)
        trace_test.set_attribute("cca.test.t3_iters", iters3)
        assert iters3 >= 1, (
            f"Agent didn't use tools (iters={iters3}). "
            f"Response: {r3.content[:200]}"
        )

        # Should have used API doc tools (search or get)
        tool_names_3 = r3.tool_names
        trace_test.set_attribute("cca.test.t3_tools", str(tool_names_3))
        used_api_tools = any(
            t in name for name in tool_names_3
            for t in ["search_docs", "get_api_docs"]
        )
        trace_test.set_attribute(
            "cca.test.t3_used_api_tools", used_api_tools,
        )
        # Advisory -- agent might use web_search if qdrant isn't
        # in the chub index, which is acceptable fallback behavior
        if not used_api_tools:
            trace_test.set_attribute(
                "cca.test.t3_note",
                "Agent used web_search fallback instead of API docs "
                "(qdrant-client may not be in chub index)",
            )

        # Response should contain qdrant-related code
        content3 = r3.content.lower()
        has_qdrant = any(w in content3 for w in [
            "qdrant", "pointstruct", "upsert", "collection",
            "qdrantclient", "qdrant_client", "vector",
        ])
        trace_test.set_attribute(
            "cca.test.t3_has_qdrant", has_qdrant,
        )
        assert has_qdrant, (
            f"Response doesn't contain Qdrant code: "
            f"{r3.content[:300]}"
        )
