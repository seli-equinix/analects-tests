"""Regression guard: module-level `logger` in chat_completions adapter.

P22065_5-retrieval-modes failed because the "Dynamic max_tokens cap"
blocks at lines 695 (`_invoke_api`) and 753 (`_invoke_api_streamed`)
both call `logger.warning(...)` but no module-level `logger` was
defined. The bug was dormant for short-context requests (the condition
gating the call evaluates False); it fired deterministically for the
multi-turn 5-retrieval-modes test by Turn 3, raising NameError that
bubbled up through the planner extension and surfaced as a misleading
"Planning failed" tool error.

These invariants catch the same drift from sneaking back via future
removal of the module-level `logger`.

Pure-Python import + attribute inspection — no Memgraph, no langchain
runtime needed (the file's own imports do pull in openai + langchain
but those are present everywhere CCA runs).
"""
from __future__ import annotations

import inspect
import logging

import pytest

# chat_completions transitively imports langchain_core which isn't on
# node5's venv. Skip there; CI runs in cca-tests image.
pytest.importorskip(
    "langchain_core",
    reason="chat_completions's runtime deps are only installed inside "
           "the cca container.",
)

from confucius.core.chat_models.azure.adapters import (  # noqa: E402
    chat_completions as _cc,
)


class TestModuleLevelLoggerExists:
    """The module must expose a module-level `logger` that is a
    `logging.Logger` instance. Without this, the dynamic max_tokens cap
    blocks raise NameError at attribute lookup time."""

    def test_logger_attribute_present(self):
        assert hasattr(_cc, "logger"), (
            "chat_completions module must define a module-level `logger` "
            "attribute. The dynamic max_tokens cap blocks (lines ~695, "
            "~753) call `logger.warning(...)`; without this attribute, "
            "they raise NameError under high-context requests. See "
            "P22065_5-retrieval-modes for the reproducer."
        )

    def test_logger_is_a_logging_logger(self):
        assert isinstance(_cc.logger, logging.Logger), (
            f"chat_completions.logger must be a logging.Logger; got "
            f"{type(_cc.logger).__name__}"
        )

    def test_logger_named_after_module(self):
        # Convention check — the module-level logger should use
        # __name__ so log lines route via the module's logger config.
        assert _cc.logger.name == _cc.__name__, (
            f"chat_completions.logger should be named after the module; "
            f"got {_cc.logger.name!r} vs module name {_cc.__name__!r}"
        )


class TestDynamicCapCallsitesUseBoundName:
    """The two `logger.warning(...)` callsites must reference a name
    that the module's globals resolve. Without this guard, someone
    could rename `logger` → `_logger` and pass the existence check
    above while breaking the callsites."""

    def test_invoke_api_source_contains_logger_warning(self):
        """The blocking-path `_invoke_api` must have a `logger.warning`
        call (the dynamic-cap warning line)."""
        src = inspect.getsource(_cc.ChatCompletionsAdapter._invoke_api)
        assert "logger.warning(" in src, (
            "_invoke_api should contain `logger.warning(...)` for the "
            "max_tokens-cap warning. If you renamed `logger`, update "
            "this test AND audit other callsites in the file."
        )

    def test_invoke_api_streamed_source_contains_logger_warning(self):
        """Same for the streaming-path `_invoke_api_streamed`."""
        src = inspect.getsource(
            _cc.ChatCompletionsAdapter._invoke_api_streamed
        )
        assert "logger.warning(" in src, (
            "_invoke_api_streamed should contain `logger.warning(...)` "
            "for the max_tokens-cap warning (streaming twin of the "
            "blocking-path block)."
        )

    def test_logger_callable_resolves_in_module_globals(self):
        """Belt-and-suspenders: confirm that evaluating the literal
        `logger.warning` expression against the module's globals
        actually finds the name without raising NameError. This is the
        exact failure mode P22065 hit."""
        compiled = compile(
            "logger.warning",
            "<test_logger_resolution>",
            mode="eval",
        )
        # Evaluating against the module's globals must NOT raise
        # NameError. The expression result is a bound method — we
        # don't call it, just resolve the lookup.
        result = eval(compiled, vars(_cc))  # noqa: PGH001 (eval intentional)
        assert callable(result), (
            "logger.warning should resolve to a callable; the resolution "
            "itself must not raise NameError."
        )
