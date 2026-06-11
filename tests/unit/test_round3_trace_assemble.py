"""Round-3 Fix A — trace_execution closes the loop to assemble_traced_code.

P25334 eva-code-trace: the agent traced (trace_execution x2) then wandered to
write_memory / search_codebase, never calling assemble_traced_code or a file
tool. Root cause: nothing told the model to assemble next. Fix:
  - the trace_execution RESULT carries an explicit ``next_step`` nudge.
  - the TRACE group prompt directs "IMMEDIATELY call assemble_traced_code".

The prompt-seed assertions are pure-Python (seeds/prompts.py only imports
``__future__``) so they run on node5. The result-side nudge lives in
trace_extension.py (orchestrator runtime imports) → importorskip-guarded;
runs in the cca-tests image.
"""
from __future__ import annotations

import pytest


class TestTraceGroupPrompt:
    """The seed prompt body the model sees must direct assemble-next."""

    def test_trace_group_body_directs_immediate_assemble(self):
        from confucius.server.seeds.prompts import _TRACE_GROUP_BODY
        body = _TRACE_GROUP_BODY.lower()
        assert "assemble_traced_code" in body
        assert "immediately" in body, (
            "the prompt must explicitly direct an IMMEDIATE assemble step"
        )
        assert "do not" in body and (
            "search_codebase" in body or "write_memory" in body
        ), "the prompt must steer away from search/memory between the 2 steps"

    def test_trace_group_seeded_into_defaults(self):
        from confucius.server.seeds.prompts import (
            _TRACE_GROUP_BODY,
            get_new_defaults,
        )
        defaults = get_new_defaults()
        assert defaults.get("tool.trace_group") == _TRACE_GROUP_BODY, (
            "tool.trace_group must surface _TRACE_GROUP_BODY to the model"
        )

    def test_seed_version_bumped(self):
        from confucius.server.seeds.prompts import SEED_VERSION
        assert SEED_VERSION >= 20, (
            "SEED_VERSION must bump so the new TRACE prompt reseeds on start"
        )


class TestTraceNextStep:
    """The trace_execution result carries an actionable assemble nudge."""

    def test_next_step_mentions_assemble_and_path(self):
        pytest.importorskip(
            "langchain_core",
            reason="trace_extension imports the orchestrator runtime; "
                   "runs in the cca-tests image.",
        )
        from confucius.server.code_intelligence.trace_extension import (
            _format_trace_next_step,
        )
        msg = _format_trace_next_step("/workspace/EVA/JobStart.ps1", "abc123", 42)
        assert "assemble_traced_code" in msg
        assert "abc123" in msg, "the trace_id must be threaded through"
        assert "42" in msg, "the needed-function count must appear"
        assert ".ps1" in msg, "output path keeps the entry-file extension"
        assert "JobStart" in msg, "output base derived from the entry file"
        low = msg.lower()
        assert "do not" in low and "search_codebase" in low and "write_memory" in low, (
            "the nudge must steer away from the P25334 wander targets"
        )

    def test_next_step_handles_no_extension(self):
        pytest.importorskip("langchain_core")
        from confucius.server.code_intelligence.trace_extension import (
            _format_trace_next_step,
        )
        msg = _format_trace_next_step("/workspace/script", "t1", 3)
        assert "assemble_traced_code" in msg and "t1" in msg
        assert ".txt" in msg, "no-extension entry falls back to .txt output"
