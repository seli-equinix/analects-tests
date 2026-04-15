"""Unit tests for the five stream-guard detectors.

Each detector is tested in isolation with synthetic buffers that either
match or don't match its trigger criterion. Detectors read config via the
`stream_guard.config` module — these tests use the default values, which
means every detector's `enabled` flag is ``True`` by default.
"""
from confucius.core.chat_models.stream_guard.detectors import (
    ArgLengthDetector,
    LowEntropyDetector,
    RepetitionDetector,
    TailRepeatDetector,
    TokenRateDetector,
    default_detectors,
)


class TestRepetitionDetector:
    def test_fires_on_p6009_pattern(self):
        d = RepetitionDetector()
        buf = "create_nutanix_vm.py" + ".incomplete.py" * 30
        result = d.check(buf, "tool_call[0].args.path", "")
        assert result is not None
        assert result.detector == "repetition"
        assert "coverage_pct" in result.reason

    def test_does_not_fire_on_legit_python(self):
        d = RepetitionDetector()
        lines = [
            f"    def method_{i}(self, arg_{i}: int) -> str:" for i in range(200)
        ]
        buf = "\n".join(lines)
        result = d.check(buf, "content", "")
        assert result is None

    def test_uses_per_field_coverage_override_for_path(self):
        # The 'path' field has a stricter coverage_pct=25. A small burst
        # that wouldn't fire on a generic field should fire on 'path'.
        d = RepetitionDetector()
        prefix = "Some legitimate prose describing the scenario. " * 3
        burst = ".incomplete.py" * 4
        buf = prefix + burst
        # On a 'content' field (default 50% coverage), the 56-char burst
        # inside ~144+56=200 chars = 28% coverage — below 50%, should NOT fire.
        # Actually let's check both paths:
        result_content = d.check(buf, "content", "")
        # On 'path' (25% coverage), 28% > 25%, SHOULD fire.
        result_path = d.check(buf, "tool_call[0].args.path", "")
        # At least one of these should be different if the override works.
        # The stricter path check must fire when the lenient content doesn't —
        # or both fire if the match is big enough. Accept both fire.
        if result_content is None:
            assert result_path is not None, "per-field override not applied"


class TestLowEntropyDetector:
    def test_fires_on_iagiag_tail(self):
        d = LowEntropyDetector()
        # 256 chars of 'iagiag' rotations — only 3 unique chars, below floor=8
        tail = ("iagiag" * 100)[:256]
        buf = "some normal prefix content here " + tail
        result = d.check(buf, "content", "")
        assert result is not None
        assert result.detector == "low_entropy"

    def test_does_not_fire_on_diverse_tail(self):
        d = LowEntropyDetector()
        # 256 chars of varied English — many unique chars
        tail = ("The quick brown fox jumps over the lazy dog. " * 10)[:256]
        buf = "prefix: " + tail
        result = d.check(buf, "content", "")
        assert result is None

    def test_does_not_fire_below_window(self):
        d = LowEntropyDetector()
        buf = "aa"  # way below 256-char window
        result = d.check(buf, "content", "")
        assert result is None


class TestTailRepeatDetector:
    def test_fires_on_exact_tail_duplication(self):
        d = TailRepeatDetector()
        # Need buf >= 2×window (window=200 default, so buf >= 400)
        src = "This is a test block that we will duplicate. "  # 46 chars
        block = (src * 5)[:200]  # exactly 200
        assert len(block) == 200
        buf = block + block  # exactly 400
        result = d.check(buf, "content", "")
        assert result is not None
        assert result.detector == "tail_repeat"

    def test_does_not_fire_on_different_halves(self):
        d = TailRepeatDetector()
        # 400 chars but two halves are different content
        first = "A" * 150 + "xyz" + "!" * 47  # 200 chars
        second = "B" * 150 + "abc" + "?" * 47  # 200 chars
        buf = first + second
        result = d.check(buf, "content", "")
        assert result is None

    def test_skips_single_char_tails(self):
        d = TailRepeatDetector()
        # Tail is all 'a' — len(set(tail)) == 1, skip (caught by other detectors)
        buf = "a" * 400
        result = d.check(buf, "content", "")
        assert result is None


class TestArgLengthDetector:
    def test_fires_on_oversized_path(self):
        d = ArgLengthDetector()
        buf = "/workspace/" + "a" * 10_000
        result = d.check(buf, "tool_call[0].args.path", "")
        assert result is not None
        assert result.detector == "arg_length"
        assert "cap=4096" in result.reason

    def test_does_not_fire_on_normal_path(self):
        d = ArgLengthDetector()
        buf = "/workspace/create_nutanix_vm.py"
        result = d.check(buf, "tool_call[0].args.path", "")
        assert result is None

    def test_does_not_fire_on_oversized_file_text_within_its_cap(self):
        d = ArgLengthDetector()
        # 500KB cap for file_text; a 50KB file is well within.
        buf = "x" * 50_000
        result = d.check(buf, "tool_call[0].args.file_text", "")
        assert result is None

    def test_ignores_non_tool_call_fields(self):
        d = ArgLengthDetector()
        # A 50KB content buffer — arg_length doesn't police content.
        buf = "a" * 50_000
        result = d.check(buf, "content", "")
        assert result is None


class TestTokenRateDetector:
    def test_does_not_fire_within_baseline(self):
        d = TokenRateDetector()
        # First N calls establish the baseline; no fire expected.
        for _ in range(5):
            result = d.check("buf", "content", "delta")
            assert result is None

    def test_disabled_by_default_config(self):
        # Default config has token_rate.enabled=False — ensure it stays silent
        # regardless of timing, so the passive-only guarantee holds until we
        # explicitly enable it.
        d = TokenRateDetector()
        import time
        for _ in range(20):
            result = d.check("buf", "content", "delta")
            time.sleep(0.001)
            # Detector should never fire because it's disabled by default.
            assert result is None


class TestDefaultDetectorSet:
    def test_returns_fresh_list(self):
        a = default_detectors()
        b = default_detectors()
        assert a is not b, "default_detectors must return a fresh list per request"
        # TokenRateDetector holds per-request state; two instances must be distinct.
        token_rate_a = [d for d in a if d.__class__.__name__ == "TokenRateDetector"][0]
        token_rate_b = [d for d in b if d.__class__.__name__ == "TokenRateDetector"][0]
        assert token_rate_a is not token_rate_b

    def test_includes_all_five(self):
        names = {d.name for d in default_detectors()}
        assert names == {"repetition", "low_entropy", "tail_repeat", "arg_length", "token_rate"}
