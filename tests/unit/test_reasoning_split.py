"""The </think> reasoning/answer split (root-cause fix for the reasoning-leak).

vLLM 0.19 separates reasoning↔answer in NON-streaming (reasoning→message.reasoning,
clean answer→content) but NOT in streaming: for Qwen3.5 prefix-format thinking
(<think> is the prompt prefix → generated output is "reasoning</think>answer",
bare close only), the whole thing floods delta.content. CCA HAD the split in
io_adapter (rfind("</think>")+slice) but commit 3b35cbfb changed it to
replace()/keep-before, which leaked the reasoning once the model moved to prefix
format. This locks the restored split:

  - ChunkAccumulator.finalize() splits content on </think> (answer after,
    reasoning→reasoning_parts), mirroring vLLM's non-streaming extract_reasoning.
  - io_adapter.ai() / get_all_text() restore the rfind+slice (defense-in-depth).

Both are DYNAMIC: no-op when no </think> (or no <think>) is present.

Pure-Python; importorskip-guarded for node5 (io_adapter pulls the orchestrator
runtime via its lazy imports inside ai()).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "langchain_core",
    reason="io_adapter.ai() lazily imports orchestrator runtime; runs in the "
           "cca-tests CI image.",
)

from confucius.core.chat_models.stream_guard.accumulator import ChunkAccumulator  # noqa: E402
from confucius.server.io_adapter import HttpIOInterface  # noqa: E402


def _split(content: str):
    """Run ChunkAccumulator._separate_reasoning with a stub choice builder."""
    cb = SimpleNamespace(reasoning_parts=[])
    answer = ChunkAccumulator._separate_reasoning(content, cb)
    return answer, "".join(cb.reasoning_parts)


class TestAccumulatorSeparateReasoning:
    def test_prefix_format_bare_close(self):
        # The exact failing shape: "reasoning</think>answer"
        answer, reasoning = _split(
            "The user wants to see the projects, I'll use list_projects.\n</think>\n\n"
            "Here are the projects: EVA, cache, fastapi_user_app."
        )
        assert answer == "Here are the projects: EVA, cache, fastapi_user_app."
        assert "list_projects" in reasoning  # reasoning captured, not lost

    def test_clean_answer_no_tags_unchanged(self):
        answer, reasoning = _split("Here are the projects: EVA, cache.")
        assert answer == "Here are the projects: EVA, cache."
        assert reasoning == ""

    def test_full_pair_stripped(self):
        answer, reasoning = _split("<think>let me think</think>The answer is 42.")
        assert answer == "The answer is 42."

    def test_tool_only_turn_yields_none(self):
        # Reasoning preamble + bare close, no answer after (a tool-call turn).
        answer, reasoning = _split("I'll call the tool now.\n</think>\n\n")
        assert answer is None
        assert "call the tool" in reasoning

    def test_reentry_pairs_then_bare(self):
        # Re-entered thinking (complete pair) + the prefix-format bare close.
        answer, reasoning = _split(
            "initial reasoning</think>partial<think>more reasoning</think> final answer"
        )
        # complete <think>more reasoning</think> stripped; bare </think> splits
        assert "final answer" in answer
        assert "more reasoning" not in answer and "initial reasoning" not in answer


class TestIoAdapterThinkSlice:
    def _resp(self, *chunks):
        io = HttpIOInterface()
        for c in chunks:
            asyncio.run(io.ai(c))
        return io.get_response_text()

    def test_bare_close_keeps_answer_only(self):
        out = self._resp("The user wants X, so I'll do Y.\n</think>\n\nThe real answer.")
        assert out == "The real answer."

    def test_clean_answer_unchanged(self):
        out = self._resp("The real answer with no tags.")
        assert out == "The real answer with no tags."

    def test_full_pair_stripped(self):
        out = self._resp("<think>reasoning</think>Final answer here.")
        assert out == "Final answer here."
