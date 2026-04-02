"""Knowledge test fixtures."""
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
    """HTTP client for knowledge API calls."""
    client = httpx.Client(base_url=CCA_URL, timeout=30)
    yield client
    client.close()


def search_docs(client: httpx.Client, query: str) -> dict:
    """Call search_docs via the admin API (same handler as the LLM)."""
    resp = client.post("/admin/knowledge/search", json={"query": query})
    resp.raise_for_status()
    return resp.json()
