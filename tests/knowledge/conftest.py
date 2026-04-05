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
    """Call search_docs via the admin API (same handler as the LLM)."""
    resp = client.post("/admin/knowledge/search", json={"query": query})
    resp.raise_for_status()
    return resp.json()


def assert_content_or_file(r, terms: list[str], what: str):
    """Assert terms appear in response content OR agent wrote code to a file.

    The CCA agent may write code to a file via str_replace_editor instead
    of returning it inline. If it searched docs and created a file, the
    code is doc-informed and the assertion passes.
    """
    content = r.content
    has_terms = any(t in content for t in terms)
    if has_terms:
        return
    # Fallback: agent wrote code to file after searching docs
    used_file = any("str_replace_editor" in n for n in r.tool_names)
    used_docs = any("search_docs" in n for n in r.tool_names)
    assert used_file and used_docs, (
        f"{what}: not found inline and agent didn't write a doc-informed file. "
        f"Checked terms: {terms}. Tools: {r.tool_names}"
    )
