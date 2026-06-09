"""Round-2 flaky-baseline hardening (deep-dive findings).

Locks three additive robustness fixes:

RANK 1 — transient vLLM/network errors are retryable. A mid-stream
httpx.ReadTimeout / openai.APITimeoutError / APIConnectionError /
InternalServerError previously killed the agent run (the "fails then passes on
re-run" CI flake class). They must now be in RETRYABLE_EXCEPTIONS (BadRequestError
must stay excluded — a 400 is never transient).

RANK 2 — degenerate file_text is REJECTED with a directive, not silently blanked.
Silently blanking wrote an empty file, the agent saw a false "File created"
success, and the write-fail loop-break never incremented (api-lookup P24924).

RANK 3 — browse_project vs search_docs descriptions disambiguate workspace-
structure browsing from external-library docs (python-docs tool-choice +
code-intelligence wrong-copy).

importorskip-guarded for node5; runs in the cca-tests CI image.
"""
from __future__ import annotations

import pytest

pytest.importorskip(
    "langchain_core",
    reason="imports the chat-model + orchestrator runtime; runs in cca-tests image.",
)


class TestTransientErrorsRetryable:
    def test_openai_adapter_retries_transient_llm_errors(self):
        import httpx
        from openai import (
            APIConnectionError, APITimeoutError, BadRequestError, InternalServerError,
        )
        from confucius.core.chat_models.openai.openai import RETRYABLE_EXCEPTIONS
        for exc in (APITimeoutError, APIConnectionError, InternalServerError,
                    httpx.TimeoutException):
            assert exc in RETRYABLE_EXCEPTIONS, f"{exc.__name__} must be retryable"
        assert BadRequestError not in RETRYABLE_EXCEPTIONS, (
            "BadRequestError (400) is never transient — must fail fast"
        )

    def test_azure_adapter_mirrors_transient_retries(self):
        import httpx
        from openai import (
            APIConnectionError, APITimeoutError, BadRequestError, InternalServerError,
        )
        from confucius.core.chat_models.azure.openai import RETRYABLE_EXCEPTIONS
        for exc in (APITimeoutError, APIConnectionError, InternalServerError,
                    httpx.TimeoutException):
            assert exc in RETRYABLE_EXCEPTIONS
        assert BadRequestError not in RETRYABLE_EXCEPTIONS


class TestDegenerateFileTextRejected:
    def test_degenerate_file_text_raises_directive(self):
        from confucius.orchestrator.extensions.file.edit import _normalize_editor_input
        raw = {"command": "create", "path": "/workspace/x.py",
               "file_text": "ab" * 10000}  # 100%-coverage degenerate
        with pytest.raises(ValueError) as excinfo:
            _normalize_editor_input(raw)
        msg = str(excinfo.value).lower()
        assert "degenerate" in msg
        assert "do not resend" in msg or "smaller pieces" in msg

    def test_normal_file_text_not_rejected(self):
        from confucius.orchestrator.extensions.file.edit import _normalize_editor_input
        raw = {"command": "create", "path": "/workspace/x.py",
               "file_text": "def add(a, b):\n    return a + b\n"}
        out = _normalize_editor_input(raw)  # must NOT raise
        assert out.get("file_text") == "def add(a, b):\n    return a + b\n"


class TestToolChoiceDisambiguation:
    def test_browse_project_disclaims_external_docs(self):
        from confucius.server.prompt_loader import get_template
        body = get_template("tool.browse_project").lower()
        assert "search_docs" in body, "browse_project must point at search_docs"
        assert "external library" in body or "do not use it for external" in body

    def test_search_docs_disclaims_workspace_browsing(self):
        from confucius.server.prompt_loader import get_template
        body = get_template("tool.search_docs_description").lower()
        assert "browse_project" in body or "search_codebase" in body
        assert "external" in body and "not for browsing" in body
