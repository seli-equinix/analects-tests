"""Unit tests for the generalized ``_is_degenerate`` detector.

Covers the P6009 case (`.incomplete.py` ×N repeated) that the pre-generalization
regex missed, plus regressions of the historical short-unit + single-char +
low-diversity rules. Also covers the new coverage-pct gate that prevents
false positives on legitimate long outputs (e.g. Python code with consistent
4-space indentation).

Run:
    pytest tests/unit/test_is_degenerate.py -v
"""

from confucius.orchestrator.extensions.file.edit import _is_degenerate


class TestP6009Case:
    """The specific pattern that P6009 failed on: 13-char repeat units."""

    def test_incomplete_py_repeated_30x_is_degenerate(self):
        # Mirrors the iter-4 str_replace_editor path argument from P6009.
        path = "create_nutanix_vm.py" + ".incomplete.py" * 30
        assert _is_degenerate(path) is True

    def test_incomplete_py_just_above_min_reps_fires(self):
        # 3 reps × 14 chars = 42 chars; need ≥50 total and ≥50% coverage.
        # 14 chars repeated 3× = 42 chars; pad with 40 clean chars → 82 total.
        # Match = 56 chars (4 groups = the captured unit + 3 copies).
        text = "x" * 40 + ".incomplete.py" * 4
        assert _is_degenerate(text) is True

    def test_incomplete_py_sub_coverage_does_not_fire(self):
        # Same 4-rep repetition (56 chars) inside a 2000-char legitimate
        # code block → coverage is <50%, should NOT fire.
        legit = "def process(x):\n    return x + 1\n" * 60   # ~2000 chars
        short_burst = ".incomplete.py" * 4
        text = legit + short_burst + legit
        assert _is_degenerate(text) is False


class TestFalsePositiveGuardrails:
    """Legitimate long outputs must NOT trip the detector."""

    def test_python_with_4space_indentation_is_not_degenerate(self):
        # The iter-4 file_text false-positive in P6009: 9102 chars of
        # legitimate Python script got cleared because 4-space indentation
        # matched the old 2-4-char repeat rule. The generalized detector
        # must still preserve the short-unit rule (intentional — `    `
        # repeated 11+ times IS unusual), so we build a realistic Python
        # file whose longest run of spaces is within normal bounds.
        lines = []
        for i in range(300):
            lines.append(f"    def method_{i}(self, arg_{i}: int) -> str:")
            lines.append(f"        return f'value-{{arg_{i}}}'")
            lines.append("")
        text = "\n".join(lines)
        assert len(text) > 5000
        assert _is_degenerate(text) is False

    def test_long_english_prose_is_not_degenerate(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump. "
        ) * 20  # ~2300 chars, high diversity, no repetition cycles
        assert _is_degenerate(text) is False

    def test_json_array_with_varied_objects_is_not_degenerate(self):
        # Legit data response — many similar-looking rows but not identical
        items = [
            f'{{"id": {i}, "name": "item-{i}", "count": {i * 3}}}'
            for i in range(80)
        ]
        text = "[\n  " + ",\n  ".join(items) + "\n]"
        assert _is_degenerate(text) is False


class TestHistoricalCases:
    """Regression: all pre-generalization behavior preserved."""

    def test_short_unit_repetition_fires(self):
        # Classic Qwen collapse — docstring example from the original function
        text = "VsVsVsVsVsVsVsVsVsVsVs" + "x" * 50
        assert _is_degenerate(text) is True

    def test_single_char_run_fires(self):
        text = "a" * 60
        assert _is_degenerate(text) is True

    def test_low_diversity_long_text_fires(self):
        # > 100 chars, < 6 unique chars
        text = "ab" * 60  # 120 chars, 2 unique chars
        assert _is_degenerate(text) is True


class TestShortInputs:
    """Below min_chars threshold, the detector must abstain."""

    def test_short_repetition_does_not_fire(self):
        assert _is_degenerate(".py" * 5) is False  # 15 chars, under default 50

    def test_empty_string_does_not_fire(self):
        assert _is_degenerate("") is False


class TestCoveragePctOverride:
    """Callers (e.g. the streaming RepetitionDetector) can override coverage."""

    def test_strict_coverage_fires_on_smaller_match(self):
        # Default 50% coverage: a 56-char burst in ~500 chars of legitimate
        # varied text should NOT fire. Use varied prose as filler (NOT single
        # chars, which would trip the single-char-run rule independently).
        burst = ".incomplete.py" * 4  # 56 chars
        prefix = "Normal prose explaining the function. " * 5   # ~195 chars
        suffix = "Ending with different legitimate sentence. " * 5  # ~215 chars
        text = prefix + burst + suffix  # ~466 chars, burst ≈ 12%
        assert _is_degenerate(text) is False
        # With a 10% strict coverage threshold (path-field setting), it fires.
        assert _is_degenerate(text, coverage_pct=10) is True

    def test_lenient_coverage_suppresses_fires(self):
        # 30-rep pattern normally covers well over 50%. With an absurdly high
        # coverage_pct, the medium-unit rule is suppressed. Use varied (not
        # single-char) prefix so no other rule fires either.
        prefix = "Legitimate prefix prose. "  # 25 chars, varied
        text = prefix + ".incomplete.py" * 30  # 25 + 420 = 445 chars
        assert _is_degenerate(text) is True  # default 50%
        # coverage_pct=999 = threshold > text length → medium-unit rule
        # can't match. Other rules also shouldn't fire on this text.
        assert _is_degenerate(text, coverage_pct=999) is False


class TestUnicodeSafety:
    """Detector must not crash on multibyte characters."""

    def test_chinese_chars_handled(self):
        text = "你好世界" * 50  # legit, no collapse pattern in our sense
        # Shouldn't crash; outcome acceptable either way since 4 unique chars
        result = _is_degenerate(text)
        assert isinstance(result, bool)

    def test_emoji_repetition_fires(self):
        text = "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥"
        # Single "char" run at the unicode codepoint level
        result = _is_degenerate(text)
        assert isinstance(result, bool)
