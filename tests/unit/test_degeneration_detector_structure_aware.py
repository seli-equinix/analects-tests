"""Structure-aware content degeneration detection (sweep Fix B).

Regression guard for P22679_routing-edge-cases: the stream_guard repetition
detector (R1) false-fired on a LEGITIMATE structured CI/CD planning response —
numbered lists + repeated ``**headers**`` read as a repeating unit — and the
run was killed (DegenerationTerminal). The fix gates the content/reasoning
repetition check with a trigram-uniqueness confirmation
(:func:`confucius.core.quality.is_degenerate_content`): a buffer is only
flagged when it ALSO has genuinely low trigram diversity (real spew). Tool-call
argument fields keep the strict R1 check so a degenerate ``path``/``new_str``
(P22686's ``…amen.amen…endofscript`` filename) still fires.

Pure-Python — only the quality detectors, no langchain/bs4. Runs on node5 AND
in the cca-tests CI image.
"""
from __future__ import annotations

from confucius.core.quality import is_degenerate, is_degenerate_content
from confucius.core.quality._detectors import (
    CONTENT_DEGEN_TRIGRAM_FLOOR,
    detect_unit_repetition,
    is_degenerate_unit_repetition,
    _trigram_uniqueness,
)


# ── Legitimate structured prose is NOT flagged ───────────────────────


class TestStructuredProseNotFlagged:
    def test_cicd_plan_with_lists_and_headers(self):
        plan = (
            "## CI/CD Pipeline Architecture\n\n"
            "1. **Testing Stage**: run pytest across all 5 microservices.\n"
            "2. **Build Stage**: build Docker images with multi-stage caching.\n"
            "3. **Deploy Stage**: roll out to Kubernetes via Helm canary.\n\n"
            "### Testing Stage\n- Lint with ruff\n- Unit tests\n- Integration\n\n"
            "### Build Stage\n- docker build per service\n- push to registry\n\n"
            "### Deploy Stage\n- helm upgrade\n- kubectl rollout status\n"
        ) * 3
        assert not is_degenerate_content(plan), (
            "a structured CI/CD plan must NOT be flagged as degenerate"
        )

    def test_high_trigram_buffer_suppressed_even_if_r1_fires(self):
        # Construct a buffer where R1 fires (a repeated unit covers >= coverage)
        # but the overall trigram diversity is high (varied prose around it).
        # The structure-aware check must suppress it.
        varied = (
            "Deploy the canary to the staging cluster and watch rollout "
            "metrics for error-rate regressions before promoting to live. "
        )
        repeated_unit = "abcdefghij" * 30  # single 10-char unit, R1-positive
        buf = varied + repeated_unit + varied
        r1, _ = detect_unit_repetition(buf, coverage_pct=50)
        if r1 and _trigram_uniqueness(buf) >= CONTENT_DEGEN_TRIGRAM_FLOOR:
            assert not is_degenerate_content(buf, coverage_pct=50), (
                "R1-positive but high-trigram buffer must be suppressed for content"
            )


# ── Genuine degeneration IS still flagged ────────────────────────────


class TestGenuineDegenerationStillFlagged:
    def test_token_spew(self):
        assert is_degenerate_content("etcetera etcetera " * 200)

    def test_single_char_run(self):
        assert is_degenerate_content("a" * 500)

    def test_low_diversity_long_text(self):
        assert is_degenerate_content(("ab " * 400))

    def test_degenerate_has_low_trigram(self):
        spew = "etcetera etcetera " * 200
        assert _trigram_uniqueness(spew) < CONTENT_DEGEN_TRIGRAM_FLOOR


# ── Tool-arg fields keep the STRICT check (unchanged) ────────────────


class TestToolArgStrictness:
    def test_degenerate_filename_still_strict(self):
        # P22686: the model emitted a giant repetitive filename. The strict
        # tool-arg path (is_degenerate / R1) must still catch it.
        fname = (
            "Modify-NestedJsonConfig.ps1." + ".".join(["amen"] * 60)
            + "." + "x." * 40 + "endofscript"
        )
        assert is_degenerate_unit_repetition(fname)
        assert is_degenerate(fname)

    def test_strict_check_unaffected_by_trigram_gate(self):
        # is_degenerate (strict) must equal raw R1 — no trigram gating.
        spew = "etcetera etcetera " * 200
        assert is_degenerate(spew) == is_degenerate_unit_repetition(spew)


# ── RepetitionDetector field routing ─────────────────────────────────


class TestRepetitionDetectorRouting:
    def test_prose_fields_constant(self):
        from confucius.core.chat_models.stream_guard.detectors import (
            RepetitionDetector,
        )
        assert RepetitionDetector._PROSE_FIELDS == frozenset(
            {"content", "reasoning"}
        )
