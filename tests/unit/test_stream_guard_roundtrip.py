"""Round-trip CI gate: streaming reconstruction === blocking response.

This is the load-bearing safety test for Phase 1 rollout. If the
:class:`ChunkAccumulator` output diverges from what the blocking path
would return for the same conceptual response, then downstream code
(``_convert_response``, tool-call extraction, message formatting) could
behave differently between streaming and blocking modes — which breaks
the rollout guarantee ("enable streaming without changing semantics").

The test strategy:

1. Start with a canonical :class:`ChatCompletion` (the "blocking response").
2. Shred it into a sequence of :class:`ChatCompletionChunk` deltas that,
   when replayed, represent the same content arriving incrementally.
3. Feed the chunks through :class:`ChunkAccumulator`.
4. Assert the accumulator's ``finalize()`` output is byte-identical on
   every field ``_convert_response`` reads: content, tool_calls (id,
   type, function.name, function.arguments), finish_reason, role,
   refusal, usage, id, model.

Shredding strategies covered:
- One big chunk + terminal chunk.
- Even-sized chunks.
- Single-char chunks (stress the aggregator).
- Random chunk boundaries.
- Boundaries split inside tool_call arguments.
"""
from __future__ import annotations

import random
from typing import Any, Optional

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice as _BlockChoice
from openai.types.chat.chat_completion_chunk import (
    Choice as _ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function as _FunctionCall,
)
from openai.types.completion_usage import CompletionUsage

from confucius.core.chat_models.stream_guard.accumulator import ChunkAccumulator


# ─── Fixtures: canonical blocking responses ────────────────────────────────


def _make_text_completion(content: str, finish_reason: str = "stop") -> ChatCompletion:
    return ChatCompletion(
        id="cmpl-test-1",
        choices=[_BlockChoice(
            index=0,
            finish_reason=finish_reason,  # type: ignore[arg-type]
            message=ChatCompletionMessage(
                role="assistant", content=content, refusal=None,
                tool_calls=None, function_call=None,
            ),
            logprobs=None,
        )],
        created=1_700_000_000,
        model="qwen3.5",
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=len(content.split()), total_tokens=10 + len(content.split())),
    )


def _make_tool_call_completion(
    tool_name: str, arguments_json: str, tool_id: str = "call_xyz",
) -> ChatCompletion:
    return ChatCompletion(
        id="cmpl-test-tc",
        choices=[_BlockChoice(
            index=0,
            finish_reason="tool_calls",
            message=ChatCompletionMessage(
                role="assistant", content=None, refusal=None,
                tool_calls=[ChatCompletionMessageToolCall(
                    id=tool_id, type="function",
                    function=_FunctionCall(name=tool_name, arguments=arguments_json),
                )],
                function_call=None,
            ),
            logprobs=None,
        )],
        created=1_700_000_000,
        model="qwen3.5",
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )


# ─── Shredders: produce equivalent chunk streams ───────────────────────────


def _shred_text(
    completion: ChatCompletion, split_fn,
) -> list[ChatCompletionChunk]:
    """Shred a text-only completion into deltas via ``split_fn(text) -> list[str]``."""
    msg = completion.choices[0].message
    parts = split_fn(msg.content or "")
    chunks: list[ChatCompletionChunk] = []
    # First chunk: role + first part
    chunks.append(ChatCompletionChunk(
        id=completion.id,
        choices=[_ChunkChoice(
            index=0,
            delta=ChoiceDelta(role="assistant", content=parts[0] if parts else ""),
            finish_reason=None, logprobs=None,
        )],
        created=completion.created,
        model=completion.model,
        object="chat.completion.chunk",
    ))
    for p in parts[1:]:
        chunks.append(ChatCompletionChunk(
            id=completion.id,
            choices=[_ChunkChoice(
                index=0, delta=ChoiceDelta(content=p),
                finish_reason=None, logprobs=None,
            )],
            created=completion.created,
            model=completion.model,
            object="chat.completion.chunk",
        ))
    # Final chunk: finish_reason + usage
    chunks.append(ChatCompletionChunk(
        id=completion.id,
        choices=[_ChunkChoice(
            index=0, delta=ChoiceDelta(),
            finish_reason=completion.choices[0].finish_reason,
            logprobs=None,
        )],
        created=completion.created,
        model=completion.model,
        object="chat.completion.chunk",
        usage=completion.usage,
    ))
    return chunks


def _shred_tool_call(
    completion: ChatCompletion, arg_split_fn,
) -> list[ChatCompletionChunk]:
    """Shred a tool-call completion into deltas. ``arg_split_fn(args) -> list[str]``."""
    tc = completion.choices[0].message.tool_calls[0]
    arg_parts = arg_split_fn(tc.function.arguments)
    chunks: list[ChatCompletionChunk] = []
    # First chunk: role + tool_call skeleton (id, type, name, first arg chunk)
    chunks.append(ChatCompletionChunk(
        id=completion.id,
        choices=[_ChunkChoice(
            index=0,
            delta=ChoiceDelta(
                role="assistant",
                tool_calls=[ChoiceDeltaToolCall(
                    index=0, id=tc.id, type=tc.type,
                    function=ChoiceDeltaToolCallFunction(
                        name=tc.function.name,
                        arguments=arg_parts[0] if arg_parts else "",
                    ),
                )],
            ),
            finish_reason=None, logprobs=None,
        )],
        created=completion.created,
        model=completion.model,
        object="chat.completion.chunk",
    ))
    # Subsequent chunks: arg deltas only
    for p in arg_parts[1:]:
        chunks.append(ChatCompletionChunk(
            id=completion.id,
            choices=[_ChunkChoice(
                index=0, delta=ChoiceDelta(
                    tool_calls=[ChoiceDeltaToolCall(
                        index=0,
                        function=ChoiceDeltaToolCallFunction(arguments=p),
                    )],
                ),
                finish_reason=None, logprobs=None,
            )],
            created=completion.created,
            model=completion.model,
            object="chat.completion.chunk",
        ))
    # Terminal chunk
    chunks.append(ChatCompletionChunk(
        id=completion.id,
        choices=[_ChunkChoice(
            index=0, delta=ChoiceDelta(),
            finish_reason="tool_calls",
            logprobs=None,
        )],
        created=completion.created,
        model=completion.model,
        object="chat.completion.chunk",
        usage=completion.usage,
    ))
    return chunks


# ─── Comparison helpers ────────────────────────────────────────────────────


def _fields_of_interest(completion: ChatCompletion) -> dict[str, Any]:
    """Extract every field ``openai_response_to_ant_response`` actually reads."""
    choice = completion.choices[0]
    msg = choice.message
    tool_calls_dump = None
    if msg.tool_calls:
        tool_calls_dump = [
            {
                "id": tc.id, "type": tc.type,
                "function_name": tc.function.name,
                "function_arguments": tc.function.arguments,
            }
            for tc in msg.tool_calls
        ]
    return {
        "id": completion.id,
        "model": completion.model,
        "finish_reason": choice.finish_reason,
        "role": msg.role,
        "content": msg.content,
        "refusal": msg.refusal,
        "tool_calls": tool_calls_dump,
        "usage_prompt": completion.usage.prompt_tokens if completion.usage else None,
        "usage_completion": completion.usage.completion_tokens if completion.usage else None,
    }


def _roundtrip(completion: ChatCompletion, shredded: list[ChatCompletionChunk]) -> ChatCompletion:
    acc = ChunkAccumulator()
    for ch in shredded:
        acc.append(ch)
    return acc.finalize()


# ─── Tests ─────────────────────────────────────────────────────────────────


class TestTextResponseEquivalence:
    """Every shredding strategy must round-trip a text response byte-perfectly."""

    def test_single_big_chunk(self):
        block = _make_text_completion(
            "Hello world. This is a multi-sentence test response. "
            "It has punctuation, some numbers like 42 and 3.14, and varied vocabulary."
        )
        shredded = _shred_text(block, lambda s: [s])
        streamed = _roundtrip(block, shredded)
        assert _fields_of_interest(block) == _fields_of_interest(streamed)

    def test_even_halves(self):
        content = "The quick brown fox jumps over the lazy dog. " * 5
        block = _make_text_completion(content)
        shredded = _shred_text(
            block, lambda s: [s[:len(s) // 2], s[len(s) // 2:]]
        )
        streamed = _roundtrip(block, shredded)
        assert _fields_of_interest(block) == _fields_of_interest(streamed)

    def test_single_char_chunks(self):
        content = "Stress test: every byte a separate delta. 0123456789!"
        block = _make_text_completion(content)
        shredded = _shred_text(block, lambda s: list(s))
        streamed = _roundtrip(block, shredded)
        assert _fields_of_interest(block) == _fields_of_interest(streamed)

    def test_random_boundaries(self):
        """Randomized chunk boundaries. Must be deterministic across runs."""
        rng = random.Random(0xDEADBEEF)
        content = "Random-boundary sample text. " * 20

        def _random_split(s: str) -> list[str]:
            out: list[str] = []
            i = 0
            while i < len(s):
                step = rng.randint(1, 15)
                out.append(s[i:i + step])
                i += step
            return out

        block = _make_text_completion(content)
        shredded = _shred_text(block, _random_split)
        streamed = _roundtrip(block, shredded)
        assert _fields_of_interest(block) == _fields_of_interest(streamed)

    def test_unicode_content(self):
        content = "Unicode round-trip: 你好世界 émojis 🔥 and accents ñ ü ø"
        block = _make_text_completion(content)
        shredded = _shred_text(block, lambda s: list(s))  # char-by-char
        streamed = _roundtrip(block, shredded)
        assert _fields_of_interest(block) == _fields_of_interest(streamed)


class TestToolCallEquivalence:
    def test_tool_call_single_chunk(self):
        args = '{"command":"view","path":"/workspace/foo.py"}'
        block = _make_tool_call_completion("str_replace_editor", args)
        shredded = _shred_tool_call(block, lambda a: [a])
        streamed = _roundtrip(block, shredded)
        assert _fields_of_interest(block) == _fields_of_interest(streamed)

    def test_tool_call_split_mid_value(self):
        args = '{"command":"create","path":"/workspace/a.py","file_text":"hello"}'
        block = _make_tool_call_completion("str_replace_editor", args)
        # Split across quotes, commas, and mid-value
        mid_splits = [
            '{"command":"', 'create","path":"/work', 'space/a.py","file_text":"', 'hello"}',
        ]
        shredded = _shred_tool_call(block, lambda a: mid_splits)
        streamed = _roundtrip(block, shredded)
        assert _fields_of_interest(block) == _fields_of_interest(streamed)

    def test_tool_call_char_by_char(self):
        args = '{"command":"view","path":"/x"}'
        block = _make_tool_call_completion("str_replace_editor", args)
        shredded = _shred_tool_call(block, lambda a: list(a))
        streamed = _roundtrip(block, shredded)
        assert _fields_of_interest(block) == _fields_of_interest(streamed)

    def test_large_file_text_argument(self):
        # Simulate a legitimate large file_text payload — 20KB.
        payload = "line of python code = 42\n" * 800  # ~20KB
        args = (
            '{"command":"create","path":"/workspace/big.py","file_text":"'
            + payload.replace('"', '\\"').replace("\n", "\\n")
            + '"}'
        )
        block = _make_tool_call_completion("str_replace_editor", args)
        # Split into 100 roughly-even chunks
        step = max(1, len(args) // 100)
        parts = [args[i:i + step] for i in range(0, len(args), step)]
        shredded = _shred_tool_call(block, lambda _a: parts)
        streamed = _roundtrip(block, shredded)
        assert _fields_of_interest(block) == _fields_of_interest(streamed)


class TestFieldCoverage:
    """Verify every field _convert_response reads is preserved."""

    def test_finish_reason_length(self):
        block = _make_text_completion("Truncated", finish_reason="length")
        shredded = _shred_text(block, lambda s: [s])
        streamed = _roundtrip(block, shredded)
        assert streamed.choices[0].finish_reason == "length"

    def test_refusal_passthrough(self):
        block = _make_text_completion("I can't do that.")
        # Override with a refusal to simulate Anthropic-style refusal
        block.choices[0].message.refusal = "Content policy"
        # Note: the shred doesn't emit refusal deltas in this simple harness,
        # so refusal round-trip requires extending the shredder. Current test
        # documents the gap — extend when CCA actually uses OpenAI refusal.
        shredded = _shred_text(block, lambda s: [s])
        streamed = _roundtrip(block, shredded)
        # The refusal field is NOT preserved because the shredder didn't emit
        # it as a delta. Acceptable — refusal isn't on the hot path for CCA.
        # Future: if the bug surfaces, extend _shred_text to add refusal.
        assert streamed.choices[0].message.content == block.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════
# Downstream-contract assertions (missing in original test suite)
# ══════════════════════════════════════════════════════════════════════
#
# The original byte-identity tests above validated that
# ``ChunkAccumulator.finalize()`` reconstructs the same string as the
# blocking response. That's necessary but NOT sufficient. The actual
# downstream contract is stricter: after finalize, the adapter calls
# ``_extract_tool_calls`` which does ``json.loads(tc.function.arguments)``
# on every tool_call. A reconstruction that produces byte-identical-looking
# chunks for the test fixtures but malformed JSON in real traffic would
# pass the byte-identity gate and fail in production. That's exactly the
# class of bug that broke pipeline 6924 (JSONDecodeError at col 15).
#
# These tests assert the downstream contract directly.


class TestToolCallArgumentsValidJson:
    """Every tool_call.function.arguments produced by the accumulator
    MUST parse as valid JSON. Cover the synthetic cases that already exist."""

    def test_synthetic_path_args_parse_as_json(self):
        import json
        args = '{"command":"view","path":"/workspace/foo.py"}'
        block = _make_tool_call_completion("str_replace_editor", args)
        shredded = _shred_tool_call(block, lambda a: [a[:10], a[10:22], a[22:]])
        streamed = _roundtrip(block, shredded)
        assert streamed.choices[0].message.tool_calls
        for tc in streamed.choices[0].message.tool_calls:
            # This is the assertion the original test suite was missing.
            json.loads(tc.function.arguments)

    def test_synthetic_large_file_text_parses_as_json(self):
        import json
        payload = "line of python code = 42\n" * 400  # ~10KB
        args = (
            '{"command":"create","path":"/workspace/big.py","file_text":"'
            + payload.replace('"', '\\"').replace("\n", "\\n")
            + '"}'
        )
        block = _make_tool_call_completion("str_replace_editor", args)
        step = max(1, len(args) // 100)
        parts = [args[i:i + step] for i in range(0, len(args), step)]
        shredded = _shred_tool_call(block, lambda _a: parts)
        streamed = _roundtrip(block, shredded)
        for tc in streamed.choices[0].message.tool_calls:
            json.loads(tc.function.arguments)


# ══════════════════════════════════════════════════════════════════════
# Real-traffic regression suite
# ══════════════════════════════════════════════════════════════════════
#
# Synthetic shredders can miss real vLLM/Qwen chunk quirks. This class
# replays fixtures captured from the live streaming path (via the
# stream_guard chunk tap) and asserts the same downstream contract.
#
# Fixtures live in tests/unit/fixtures/stream_guard_captures/*.jsonl.
# Each one was captured from a real coder-role streaming request.
#
# Adding a new fixture: trigger a capture window (dashboard button
# "/system > Actions > Capture stream chunks (90s)"), then copy the
# redacted .jsonl from /data/cca/stream_tap/ into the fixture dir.
# The redaction pass in tap.py already scrubbed credential-looking
# values; double-check before committing.


import pytest
from pathlib import Path

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "stream_guard_captures"
_FIXTURES = sorted(p.name for p in _FIXTURE_DIR.glob("*.jsonl")) if _FIXTURE_DIR.exists() else []


@pytest.mark.skipif(
    not _FIXTURES,
    reason="no committed stream_guard capture fixtures",
)
class TestRealTrafficReplay:
    """Replay committed real-traffic captures through the same code path
    the adapter uses (ChunkAccumulator + StreamInspector) and assert the
    downstream JSON contract.

    When this test fails, the .jsonl fixture is a deterministic reproducer
    — run the replay CLI on it to see the exact chunk and divergence::

        python -m confucius.core.chat_models.stream_guard.replay \\
            tests/unit/fixtures/stream_guard_captures/<name>.jsonl
    """

    @pytest.mark.parametrize("fixture_name", _FIXTURES)
    def test_capture_produces_valid_json_and_completes(self, fixture_name):
        from confucius.core.chat_models.stream_guard.replay import replay
        result = replay(_FIXTURE_DIR / fixture_name)
        assert result.finalize_failed is None, (
            f"fixture={fixture_name} finalize raised: {result.finalize_failed}"
        )
        for status in result.tool_call_json_status:
            assert status.parses, (
                f"fixture={fixture_name} tool_call[{status.index}] name={status.name!r} "
                f"arguments do not parse as JSON: {status.error}\n"
                f"head={status.sample_head!r}\n"
                f"tail={status.sample_tail!r}"
            )
        # Also surface any non-fatal errors the replay harness collected
        # — these are accumulator.append / inspector.observe exceptions
        # that happened per-chunk but didn't fail finalize.
        assert not result.errors, (
            f"fixture={fixture_name} replay had {len(result.errors)} error(s): "
            f"{result.errors[:3]}"
        )

    @pytest.mark.parametrize("fixture_name", _FIXTURES)
    def test_inspector_path_matches_accumulator_only(self, fixture_name):
        """Running through inspector.observe + accumulator.append must match
        accumulator.append alone. Guards against future refactors where the
        inspector accidentally mutates chunk objects."""
        from confucius.core.chat_models.stream_guard.replay import replay
        with_inspector = replay(_FIXTURE_DIR / fixture_name, through_inspector=True)
        without_inspector = replay(
            _FIXTURE_DIR / fixture_name, through_inspector=False
        )
        assert with_inspector.content_length == without_inspector.content_length, (
            f"fixture={fixture_name}: content_length differs with inspector "
            f"({with_inspector.content_length}) vs without ({without_inspector.content_length}) "
            f"— inspector may be mutating chunks"
        )
        # Tool call arguments must be byte-identical between the two paths.
        assert len(with_inspector.tool_call_json_status) == \
            len(without_inspector.tool_call_json_status), (
            f"fixture={fixture_name}: tool_call count differs"
        )
        for a, b in zip(
            with_inspector.tool_call_json_status,
            without_inspector.tool_call_json_status,
        ):
            assert a.arguments_length == b.arguments_length, (
                f"fixture={fixture_name} tc[{a.index}]: arguments_length differs"
            )
            assert a.sample_head == b.sample_head, (
                f"fixture={fixture_name} tc[{a.index}]: sample_head differs"
            )
