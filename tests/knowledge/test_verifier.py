"""Unit tests for confucius.server.knowledge.verifier.

Mocks the note-taker httpx.AsyncClient + get_llm_params so the test
runs without a live cca server or vLLM. Covers:
  - Trusted top-1 paths skip the verifier (caller decides this; the
    verifier itself is dumb — but we still assert ANSWER parsing).
  - Low-confidence paths invoke the model and the model's choice wins.
  - 'ANSWER: none' returns None (caller falls back to web_search).
  - Unknown id returned by model is ignored (returns None).
  - LRU cache: second identical call doesn't re-invoke the LLM.
  - Timeout / network error returns None silently (graceful degradation).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def candidates():
    return [
        {"id": "vmware/powercli-core", "languages": ["powershell"], "score": 60,
         "snippet": "PowerCLI cmdlet docs ..."},
        {"id": "microsoft/pwsh-windows-admin", "languages": ["powershell"], "score": 40,
         "snippet": "Windows admin pwsh module ..."},
        {"id": "microsoft/pwsh-functions", "languages": ["powershell"], "score": 35,
         "snippet": "Functions module ..."},
    ]


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear the verifier cache between tests."""
    from confucius.server.knowledge.verifier import clear_verifier_cache
    clear_verifier_cache()
    yield
    clear_verifier_cache()


def _fake_llm_params(base_url: str = "http://localhost:8400/v1", model: str = "note-taker"):
    m = MagicMock()
    m.model = model
    m.additional_kwargs = {"base_url": base_url}
    return m


def _fake_response(content: str) -> MagicMock:
    """Build an httpx Response stand-in whose .json() returns vLLM-shaped output."""
    resp = MagicMock()
    resp.json = MagicMock(return_value={
        "choices": [{"message": {"content": content}}],
    })
    return resp


class FakeAsyncClient:
    """Async context-manager that returns ``response`` from .post().

    Replaces httpx.AsyncClient so no network IO happens during tests.
    """
    def __init__(self, response: MagicMock):
        self._response = response
        self.post = AsyncMock(return_value=response)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None


@pytest.mark.asyncio
async def test_verifier_picks_correct_candidate(candidates):
    from confucius.server.knowledge import verifier

    fake_resp = _fake_response(
        "The query is for a Microsoft cmdlet, not VMware PowerCLI.\n"
        "ANSWER: microsoft/pwsh-windows-admin"
    )
    with patch.object(verifier, "httpx") as mock_httpx, \
         patch("confucius.core.config.get_llm_params", return_value=_fake_llm_params()):
        mock_httpx.AsyncClient = lambda *a, **kw: FakeAsyncClient(fake_resp)
        chosen = await verifier.verify_top_candidates(
            "Get-NetFirewallProfile", "powershell", candidates,
        )
    assert chosen == "microsoft/pwsh-windows-admin"


@pytest.mark.asyncio
async def test_verifier_none_returns_none(candidates):
    from confucius.server.knowledge import verifier

    fake_resp = _fake_response("None of these match.\nANSWER: none")
    with patch.object(verifier, "httpx") as mock_httpx, \
         patch("confucius.core.config.get_llm_params", return_value=_fake_llm_params()):
        mock_httpx.AsyncClient = lambda *a, **kw: FakeAsyncClient(fake_resp)
        chosen = await verifier.verify_top_candidates(
            "totally unrelated", None, candidates,
        )
    assert chosen is None


@pytest.mark.asyncio
async def test_verifier_unknown_id_ignored(candidates):
    """Model returns an id that wasn't in the candidate list → treat as None."""
    from confucius.server.knowledge import verifier

    fake_resp = _fake_response("ANSWER: completely/fabricated-package")
    with patch.object(verifier, "httpx") as mock_httpx, \
         patch("confucius.core.config.get_llm_params", return_value=_fake_llm_params()):
        mock_httpx.AsyncClient = lambda *a, **kw: FakeAsyncClient(fake_resp)
        chosen = await verifier.verify_top_candidates(
            "Get-Whatever", "powershell", candidates,
        )
    assert chosen is None


@pytest.mark.asyncio
async def test_verifier_lru_cache_hits_skip_llm(candidates):
    """Two identical calls → LLM is invoked exactly once."""
    from confucius.server.knowledge import verifier

    fake_resp = _fake_response("ANSWER: microsoft/pwsh-functions")
    fake_client = FakeAsyncClient(fake_resp)
    with patch.object(verifier, "httpx") as mock_httpx, \
         patch("confucius.core.config.get_llm_params", return_value=_fake_llm_params()):
        mock_httpx.AsyncClient = lambda *a, **kw: fake_client
        chosen1 = await verifier.verify_top_candidates("Get-Foo", "powershell", candidates)
        chosen2 = await verifier.verify_top_candidates("Get-Foo", "powershell", candidates)
    assert chosen1 == chosen2 == "microsoft/pwsh-functions"
    assert fake_client.post.await_count == 1


@pytest.mark.asyncio
async def test_verifier_network_error_returns_none(candidates):
    """httpx.AsyncClient.post raises → verifier returns None (graceful degrade)."""
    from confucius.server.knowledge import verifier

    class ErrorClient:
        post = AsyncMock(side_effect=RuntimeError("connection refused"))
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return None

    with patch.object(verifier, "httpx") as mock_httpx, \
         patch("confucius.core.config.get_llm_params", return_value=_fake_llm_params()):
        mock_httpx.AsyncClient = lambda *a, **kw: ErrorClient()
        chosen = await verifier.verify_top_candidates(
            "any query", None, candidates,
        )
    assert chosen is None


@pytest.mark.asyncio
async def test_verifier_no_llm_config_skips(candidates):
    """Missing base_url / model → return None, no LLM call attempted."""
    from confucius.server.knowledge import verifier

    bad_params = MagicMock()
    bad_params.model = ""
    bad_params.additional_kwargs = {}
    with patch("confucius.core.config.get_llm_params", return_value=bad_params):
        chosen = await verifier.verify_top_candidates(
            "any query", None, candidates,
        )
    assert chosen is None


@pytest.mark.asyncio
async def test_verifier_empty_candidates_returns_none():
    from confucius.server.knowledge.verifier import verify_top_candidates
    chosen = await verify_top_candidates("anything", None, [])
    assert chosen is None


@pytest.mark.asyncio
async def test_verifier_strips_think_blocks_before_parsing(candidates):
    """Qwen3-8B emits <think>...</think> blocks before the answer; the parser
    must drop them so the final ANSWER: line is found correctly."""
    from confucius.server.knowledge import verifier

    fake_resp = _fake_response(
        "<think>\nGet-Msg looks like a Microsoft Outlook cmdlet. The "
        "powercli-core package doesn't have any Outlook cmdlets, so the "
        "windows-admin or pwsh-* package is more likely correct.\n</think>\n"
        "Microsoft's pwsh-functions package is the best match.\n"
        "ANSWER: microsoft/pwsh-functions"
    )
    with patch.object(verifier, "httpx") as mock_httpx, \
         patch("confucius.core.config.get_llm_params", return_value=_fake_llm_params()):
        mock_httpx.AsyncClient = lambda *a, **kw: FakeAsyncClient(fake_resp)
        chosen = await verifier.verify_top_candidates(
            "Get-Msg", "powershell", candidates,
        )
    assert chosen == "microsoft/pwsh-functions"
