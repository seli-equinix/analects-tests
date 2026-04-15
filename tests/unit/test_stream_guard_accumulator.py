"""Unit tests for ChunkAccumulator.

The accumulator is the load-bearing component of Phase 1 safe rollout:
if it produces a ChatCompletion byte-identical to what the blocking
`stream=False` path would produce, then everything downstream of the
adapter (``_convert_response``, message extraction, tool-call parsing)
behaves identically whether the inner LLM call streamed or blocked.

These tests build synthetic chunk streams from known ChatCompletion
fixtures and verify round-trip equivalence on every field the
downstream adapter inspects.
"""
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice as _ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from confucius.core.chat_models.stream_guard.accumulator import ChunkAccumulator


def _mk_chunk(
    *,
    cid: str = "chatcmpl-test",
    model: str = "test-model",
    created: int = 1_700_000_000,
    choices: list[_ChunkChoice],
    usage=None,
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=cid,
        choices=choices,
        created=created,
        model=model,
        object="chat.completion.chunk",
        usage=usage,
    )


def _text_chunks(text: str, *, chunks: int = 3) -> list[ChatCompletionChunk]:
    """Split ``text`` across N deltas for a pure-content stream."""
    step = max(1, len(text) // chunks)
    parts = [text[i:i + step] for i in range(0, len(text), step)]
    out: list[ChatCompletionChunk] = []
    # First chunk carries role.
    out.append(_mk_chunk(choices=[
        _ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content=parts[0]), finish_reason=None)
    ]))
    for p in parts[1:]:
        out.append(_mk_chunk(choices=[
            _ChunkChoice(index=0, delta=ChoiceDelta(content=p), finish_reason=None)
        ]))
    out.append(_mk_chunk(choices=[
        _ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")
    ]))
    return out


class TestPureContent:
    def test_accumulates_text_across_chunks(self):
        acc = ChunkAccumulator()
        for ch in _text_chunks("Hello world, this is a test response from the model."):
            acc.append(ch)
        completion = acc.finalize()
        assert isinstance(completion, ChatCompletion)
        assert completion.id == "chatcmpl-test"
        assert completion.model == "test-model"
        assert completion.choices[0].message.role == "assistant"
        assert completion.choices[0].message.content == (
            "Hello world, this is a test response from the model."
        )
        assert completion.choices[0].finish_reason == "stop"
        assert completion.choices[0].message.tool_calls is None

    def test_byte_boundaries_preserved(self):
        """Split a response at every byte; reassembly must be exact."""
        original = "Exact-text-preservation test 0123456789!@#$%^&*()"
        chunks = _text_chunks(original, chunks=len(original))
        acc = ChunkAccumulator()
        for ch in chunks:
            acc.append(ch)
        assert acc.finalize().choices[0].message.content == original


class TestToolCalls:
    def test_single_tool_call_aggregated(self):
        acc = ChunkAccumulator()
        # First chunk: role + tool_call skeleton (id, type, name)
        acc.append(_mk_chunk(choices=[
            _ChunkChoice(index=0, delta=ChoiceDelta(
                role="assistant",
                tool_calls=[ChoiceDeltaToolCall(
                    index=0, id="call_abc", type="function",
                    function=ChoiceDeltaToolCallFunction(name="str_replace_editor", arguments=""),
                )],
            ), finish_reason=None)
        ]))
        # Subsequent chunks: arguments delta only
        for arg in ['{"command":"', 'create","path":"', '/workspace/a.py","file_text":"hello"', "}"]:
            acc.append(_mk_chunk(choices=[
                _ChunkChoice(index=0, delta=ChoiceDelta(
                    tool_calls=[ChoiceDeltaToolCall(
                        index=0, function=ChoiceDeltaToolCallFunction(arguments=arg),
                    )],
                ), finish_reason=None)
            ]))
        acc.append(_mk_chunk(choices=[
            _ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="tool_calls")
        ]))

        completion = acc.finalize()
        tc_list = completion.choices[0].message.tool_calls
        assert tc_list is not None and len(tc_list) == 1
        tc = tc_list[0]
        assert tc.id == "call_abc"
        assert tc.type == "function"
        assert tc.function.name == "str_replace_editor"
        # Arguments must concatenate in exact order.
        assert tc.function.arguments == (
            '{"command":"create","path":"/workspace/a.py","file_text":"hello"}'
        )
        assert completion.choices[0].finish_reason == "tool_calls"

    def test_multiple_tool_calls_distinct(self):
        acc = ChunkAccumulator()
        # Two tool calls at indices 0 and 1, interleaved args.
        acc.append(_mk_chunk(choices=[
            _ChunkChoice(index=0, delta=ChoiceDelta(
                role="assistant",
                tool_calls=[
                    ChoiceDeltaToolCall(index=0, id="c0", type="function",
                        function=ChoiceDeltaToolCallFunction(name="bash", arguments='{"cmd":"')),
                    ChoiceDeltaToolCall(index=1, id="c1", type="function",
                        function=ChoiceDeltaToolCallFunction(name="view", arguments='{"path":"')),
                ],
            ), finish_reason=None)
        ]))
        acc.append(_mk_chunk(choices=[
            _ChunkChoice(index=0, delta=ChoiceDelta(tool_calls=[
                ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments='ls"}')),
                ChoiceDeltaToolCall(index=1, function=ChoiceDeltaToolCallFunction(arguments='/x"}')),
            ]), finish_reason=None)
        ]))
        acc.append(_mk_chunk(choices=[
            _ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="tool_calls")
        ]))
        tc_list = acc.finalize().choices[0].message.tool_calls
        assert tc_list is not None and len(tc_list) == 2
        assert tc_list[0].id == "c0"
        assert tc_list[0].function.name == "bash"
        assert tc_list[0].function.arguments == '{"cmd":"ls"}'
        assert tc_list[1].id == "c1"
        assert tc_list[1].function.arguments == '{"path":"/x"}'


class TestMetadataPreservation:
    def test_id_and_model_from_first_chunk(self):
        acc = ChunkAccumulator()
        acc.append(_mk_chunk(cid="abc123", model="qwen3.5", choices=[
            _ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="x"), finish_reason=None)
        ]))
        assert acc.finalize().id == "abc123"
        assert acc.finalize().model == "qwen3.5"

    def test_usage_taken_from_last_chunk(self):
        from openai.types.completion_usage import CompletionUsage
        acc = ChunkAccumulator()
        acc.append(_mk_chunk(choices=[
            _ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="hi"), finish_reason=None)
        ]))
        acc.append(_mk_chunk(
            choices=[_ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ))
        completion = acc.finalize()
        assert completion.usage is not None
        assert completion.usage.prompt_tokens == 10
        assert completion.usage.completion_tokens == 5


class TestReasoningContentSideChannel:
    def test_reasoning_deltas_accessible(self):
        # vLLM emits reasoning_content deltas for <think> blocks.
        # Not part of ChatCompletion schema but preserved in side channel.
        class _DeltaWithReasoning:
            """Minimal delta mock that carries reasoning_content."""
            def __init__(self, reasoning: str, content: str = "") -> None:
                self.role = None
                self.content = content or None
                self.refusal = None
                self.tool_calls = None
                self.reasoning_content = reasoning

        class _ChoiceMock:
            def __init__(self, delta, finish_reason=None):
                self.delta = delta
                self.finish_reason = finish_reason
                self.index = 0
                self.logprobs = None

        class _ChunkMock:
            def __init__(self, choices, usage=None):
                self.id = "r"
                self.choices = choices
                self.created = 0
                self.model = "m"
                self.object = "chat.completion.chunk"
                self.service_tier = None
                self.system_fingerprint = None
                self.usage = usage

        acc = ChunkAccumulator()
        acc.append(_ChunkMock(choices=[_ChoiceMock(_DeltaWithReasoning("Let me think. "))]))
        acc.append(_ChunkMock(choices=[_ChoiceMock(_DeltaWithReasoning("First I'll check X."))]))
        acc.append(_ChunkMock(choices=[_ChoiceMock(
            _DeltaWithReasoning("", content="Here is the answer."),
            finish_reason="stop"
        )]))
        # Reasoning is accessible via side-channel even though ChatCompletion
        # itself doesn't carry it.
        assert acc.reasoning_content() == "Let me think. First I'll check X."
        # Main content still aggregates correctly.
        completion = acc.finalize()
        assert completion.choices[0].message.content == "Here is the answer."


class TestEmptyStreams:
    def test_empty_accumulator_produces_valid_completion(self):
        acc = ChunkAccumulator()
        assert acc.is_empty()
        completion = acc.finalize()
        # Graceful empty result — zero choices is valid structurally.
        assert isinstance(completion, ChatCompletion)
        assert len(completion.choices) == 0
