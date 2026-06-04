"""Response-assembly hardening for HttpIOInterface (sweep regression 2→6).

Root cause (2026-06-02): the synthesis turn calls io.clear_response_text()
which DELETED every assistant chunk (the good intermediate answer holding the
recalled facts / output-file mention / "N tests passed" / FTP-rejection), then
injected a "produce a final consolidated response" prompt. The model often
replied by REASONING about that instruction ("The user wants me to produce a
final consolidated response…"). That reasoning reached io.ai() as the FIRST
assistant chunk, and the CoT filter's "never empty" guardrail KEPT it (because
clear_response_text() had emptied the buffer) — so the leaked reasoning became
the entire returned response. Compounded by a per-chunk think-strip that
couldn't remove a <think>…</think> block split across streamed chunks.

These tests pin the three io_adapter fixes:
  1. clear_response_text() stashes the pre-synthesis answer; get_response_text()
     restores it when the synthesis turn yields nothing usable.
  2. a pure-CoT chunk is dropped when a stash exists (synthesis path); the
     "keep first CoT chunk" fallback is preserved when there's no stash.
  3. stateful think-strip removes a <think>…</think> block that spans chunks.

Exercises io.ai() (which lazily imports the degeneration policy + the dual-model
dedup helper), so this runs in the cca-tests CI image — not on bare node5.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip(
    "langchain_core",
    reason="io_adapter.ai() lazily imports the orchestrator dedup helper + "
           "quality policy; runs in the cca-tests CI image.",
)

from confucius.server.io_adapter import HttpIOInterface  # noqa: E402


def _new() -> HttpIOInterface:
    return HttpIOInterface()


def _feed(io: HttpIOInterface, *chunks: str) -> None:
    for c in chunks:
        asyncio.run(io.ai(c))


# ── Fix 3: stateful think-strip (pure method, no async/heavy deps) ──────


class TestStatefulThinkStrip:
    def test_split_block_drops_reasoning_keeps_answer(self):
        io = _new()
        # <think> opens here, no close → everything from the open is dropped
        # and we enter the think state.
        assert io._strip_think_streaming("Intro <think>The user wants X") == "Intro "
        assert io._in_think is True
        # Close arrives in a later chunk → drop up to </think>, keep the rest.
        assert io._strip_think_streaming("more reasoning</think>The real answer.") \
            == "The real answer."
        assert io._in_think is False

    def test_mid_block_chunk_fully_dropped(self):
        io = _new()
        io._strip_think_streaming("<think>opening")  # enter think state
        assert io._in_think is True
        # A chunk with no close while inside the block is entirely reasoning.
        assert io._strip_think_streaming("still thinking, no close yet") == ""
        assert io._in_think is True

    def test_complete_pair_in_one_chunk(self):
        io = _new()
        out = io._strip_think_streaming("Answer A <think>reasoning</think> Answer B")
        assert "reasoning" not in out
        assert "Answer A" in out and "Answer B" in out
        assert io._in_think is False

    def test_bare_close_preserves_real_content(self):
        # Qwen3.5 re-enters thinking after tool results without an explicit
        # <think>; a bare </think> with no open → keep content before it.
        io = _new()
        assert io._strip_think_streaming("real content</think>tail") == "real contenttail"
        assert io._in_think is False


# ── Fix 1 + 2: synthesis stash / restore + CoT-drop-when-stash ──────────


class TestSynthesisStashAndCoT:
    def test_cot_synthesis_falls_back_to_stash(self):
        io = _new()
        _feed(io, "Here is the real answer with the facts: postgresql, pgvector.")
        asyncio.run(io.clear_response_text())          # synthesis empties buffer
        # synthesis turn emits pure reasoning → CoT filter drops it (stash exists)
        _feed(io, "The user wants me to produce a final consolidated response "
                  "without repeating code that already appeared.")
        result = io.get_response_text()
        assert "postgresql" in result and "pgvector" in result, (
            "must restore the stashed pre-synthesis answer, not return reasoning; "
            f"got: {result!r}"
        )
        assert "consolidated response" not in result

    def test_empty_synthesis_falls_back_to_stash(self):
        io = _new()
        _feed(io, "The deployment completed; output at /workspace/out.ps1.")
        asyncio.run(io.clear_response_text())
        # synthesis produced nothing at all
        result = io.get_response_text()
        assert result == "The deployment completed; output at /workspace/out.ps1."

    def test_clean_synthesis_wins_stash_unused(self):
        io = _new()
        _feed(io, "draft answer with details")
        asyncio.run(io.clear_response_text())          # stash = draft
        _feed(io, "Final: the deployment is complete and 9 tests passed.")
        result = io.get_response_text()
        assert result == "Final: the deployment is complete and 9 tests passed.", (
            "a clean synthesis answer must be used verbatim; stash only fills "
            f"in when synthesis is empty/reasoning-only; got: {result!r}"
        )

    def test_no_stash_keeps_first_cot_chunk(self):
        # Non-synthesis routes: no clear_response_text() ⇒ no stash ⇒ the
        # "never return empty" guardrail still keeps a lone CoT chunk.
        io = _new()
        _feed(io, "The user wants me to do something specific here.")
        assert io.get_response_text() == "The user wants me to do something specific here."

    def test_clear_resets_in_think_and_stash(self):
        io = _new()
        _feed(io, "real answer")
        asyncio.run(io.clear_response_text())           # set stash
        assert io.get_stashed_response() == "real answer"
        io._in_think = True                              # simulate a dangling open block
        io.clear()
        assert io.get_stashed_response() == ""
        assert io._in_think is False
        # a fresh normal answer is unaffected by the stale think-state
        _feed(io, "brand new answer")
        assert io.get_response_text() == "brand new answer"


class TestStripLeadingCoTPreamble:
    """A chunk that leads with a CoT preamble but carries the real answer after
    a blank line must keep the answer (complex_multi_file turn 3)."""

    def test_preamble_then_answer_keeps_answer(self):
        io = _new()
        _feed(io,
              "The user wants to review the ops.py file, so I'll use "
              "str_replace_editor to view it.\n\n\n"
              "Here's the contents of /workspace/calc_ops.py:\n\n"
              "```python\ndef add(a, b): return a + b\ndef subtract(a, b): return a - b\n```")
        out = io.get_response_text()
        assert "def add" in out and "def subtract" in out, out
        # the leading 'I'll use str_replace_editor to view it' preamble is gone
        assert "I'll use str_replace_editor to view it" not in out

    def test_pure_preamble_single_paragraph_unchanged_path(self):
        # No blank line + pure CoT → strip helper returns it unchanged, caller
        # keeps it as the first chunk (never-empty fallback) — behavior preserved.
        io = _new()
        _feed(io, "The user wants me to do something specific here and now.")
        assert io.get_response_text() == "The user wants me to do something specific here and now."

    def test_strip_helper_direct(self):
        from confucius.server.io_adapter import _strip_leading_cot_paragraphs as strip
        assert strip("I'll do X.\n\nHere is the answer: 42") == "Here is the answer: 42"
        assert strip("The answer is 42.\n\nmore detail") == "The answer is 42.\n\nmore detail"  # first para not CoT
        assert strip("I'll do X. The answer is 42.") == "I'll do X. The answer is 42."  # single para → unchanged
        assert strip("Let me check.\n\nThe user wants Y.\n\nResult: ok") == "Result: ok"  # two CoT paras


class TestHasRealAnswer:
    def test_empty_buffer_has_no_real_answer(self):
        assert _new().has_real_answer() is False

    def test_cot_only_buffer_has_no_real_answer(self):
        # The complex_multi_file turn-3 case: the only assistant chunk is a
        # kept CoT preamble → not a real answer → force-summary should fire.
        io = _new()
        _feed(io, "The user wants to see the contents of ops.py, so I'll use "
                  "str_replace_editor to view the file.")
        assert io.get_response_text()  # the preamble IS buffered (kept first chunk)
        assert io.has_real_answer() is False

    def test_substantive_answer_is_real(self):
        io = _new()
        _feed(io, "The ops.py file defines add(a, b), subtract(a, b), "
                  "multiply(a, b), and divide(a, b).")
        assert io.has_real_answer() is True
