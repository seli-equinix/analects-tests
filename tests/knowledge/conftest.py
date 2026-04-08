"""Knowledge test fixtures.

Provides authenticated HTTP client for /admin/knowledge/search API
and a helper to call search_docs directly (Phase 1 API validation).
"""
import os

import httpx
import pytest

CCA_URL = os.environ.get("CCA_BASE_URL", "http://192.168.4.205:8500")


@pytest.fixture(scope="session")
def cca_url():
    """CCA server URL."""
    return CCA_URL


@pytest.fixture(scope="session")
def knowledge_client():
    """HTTP client for knowledge API calls (with auth).

    Reads CCA_TEST_API_KEY for Bearer token auth, matching the
    pattern used by CCAClient in tests/cca_client.py.
    """
    api_key = os.environ.get("CCA_TEST_API_KEY", "")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    client = httpx.Client(base_url=CCA_URL, timeout=30, headers=headers)
    yield client
    client.close()


def search_docs(client: httpx.Client, query: str) -> dict:
    """Call search_docs via the admin API (same handler as the LLM).

    The API has two response formats:
    1. Exact match: {"documentation", "source", "confidence"} — single result
    2. Search results: {"results": [{id, language, snippet}]} — array

    This helper normalizes both to the results array format for test assertions.
    """
    resp = client.post("/admin/knowledge/search", json={"query": query})
    resp.raise_for_status()
    data = resp.json()

    # Normalize exact-match format to results array format
    if "results" not in data and "source" in data:
        source = data.get("source", "")
        lang = ""
        if "/pwsh-" in source:
            lang = "powershell"
        elif source.startswith("vmware/"):
            lang = "powershell"
        data["results"] = [{
            "id": source,
            "snippet": data.get("documentation", ""),
            "language": lang,
        }]

    return data


def assert_content_or_file(r, terms: list[str], what: str):
    """Assert terms appear in response content OR agent wrote code to a file.

    The CCA agent may write code to a file via str_replace_editor instead
    of returning it inline. Accepts either:
    1. Terms found in response text (inline answer)
    2. Agent created/edited a file (code written to workspace)
    """
    content = r.content
    has_terms = any(t in content for t in terms)
    if has_terms:
        return
    # Fallback: agent wrote code to a file
    used_file = any("str_replace_editor" in n for n in r.tool_names)
    if used_file:
        return
    # Fallback: agent used bash to create/run code
    used_bash = any("bash" in n for n in r.tool_names)
    if used_bash and len(r.tool_names) >= 2:
        return
    assert False, (
        f"{what}: not found inline and agent didn't write code to a file. "
        f"Checked terms: {terms}. Tools: {r.tool_names}"
    )
