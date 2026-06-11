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
    find_content_runaway,
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


# ── Round-3: content runaway pre-gate (knowledge-pipeline) ───────────
# A SHORT degenerate run inside an otherwise-structured answer keeps
# whole-buffer trigram high, so the R1+trigram gate above misses it and the
# degenerate response leaks to the coherence evaluator
# (P-knowledge-pipeline `degenerate_repetition`). find_content_runaway is a
# coverage- and trigram-INDEPENDENT pre-gate mirroring that evaluator's
# signal (same word 5+ times, or a short exact-repeat unit). It must fire on
# genuine runaway regardless of trigram, and NEVER on legit structured prose.


class TestContentRunawayFires:
    def test_word_runaway_fires(self):
        # Identical to tests/evaluators.py::eval_coherence — same word 5+ times.
        assert find_content_runaway("service " * 6) is not None
        assert is_degenerate_content("service " * 6)

    def test_short_unit_punct_glued_fires(self):
        # The live-repro shape: "settings.py/settings.py/..." (no whitespace).
        kind_unit = find_content_runaway("settings.py/" * 6)
        assert kind_unit is not None and kind_unit[0] == "short_unit"
        assert is_degenerate_content("settings.py/" * 6)

    def test_pregate_catches_runaway_in_structured_plan(self):
        # The exact failure class: a short degenerate run buried in a long,
        # otherwise-structured migration plan. Whole-buffer trigram stays at/
        # above the floor, so the OLD R1+trigram path would NOT flag it — only
        # the new pre-gate does. Assert both: it fires, AND trigram >= floor
        # (proving the pre-gate, not the trigram path, did the work).
        plan = (
            "## Migration Plan\n\n"
            "1. **Extract the Django models** into a shared schema package.\n"
            "2. **Stand up the auth microservice** behind the gateway.\n"
            "3. **Carve out orders** with its own datastore and events.\n"
            "### Testing\n" + ("testing " * 6) + "\n"
            "### Rollout\n- canary 5%\n- watch error rates\n- promote\n"
        ) * 2
        assert is_degenerate_content(plan), "runaway-in-plan must be flagged"
        assert _trigram_uniqueness(plan) >= CONTENT_DEGEN_TRIGRAM_FLOOR, (
            "if trigram were below the floor the OLD path would have caught "
            "it — this test only proves the pre-gate when trigram is high"
        )


class TestContentRunawayNoFalsePositives:
    # Legit structured prose must NOT trip the pre-gate (exact 5x-repeat guard).
    def test_markdown_headers(self):
        assert find_content_runaway(
            "## Alpha\n## Bravo\n## Charlie\n## Delta\n## Echo\n"
        ) is None

    def test_list_markers(self):
        assert find_content_runaway(
            "- one\n- two\n- three\n- four\n- five\n- six\n"
        ) is None

    def test_markdown_table_rows(self):
        assert find_content_runaway(
            "| a | b |\n| c | d |\n| e | f |\n| g | h |\n"
        ) is None

    def test_repeated_but_varying_lines(self):
        assert find_content_runaway(
            "user_service -> auth_db\norder_service -> order_db\n"
            "cart_service -> cart_db\npay_service -> pay_db\n"
        ) is None

    def test_bold_step_markers(self):
        assert find_content_runaway(
            "**Step 1** do\n**Step 2** do\n**Step 3** do\n"
            "**Step 4** do\n**Step 5** do\n"
        ) is None

    def test_word_repeated_only_four_times_boundary(self):
        # 4 consecutive repeats = 5-total threshold NOT met → must NOT fire
        # (matches the evaluator's `(?:\s+\1){4,}` = 5 occurrences).
        assert find_content_runaway("gateway gateway gateway gateway done") is None

    def test_horizontal_rule_not_flagged(self):
        # All-punctuation run — find_degenerate_run's alnum guard skips it.
        assert find_content_runaway("intro\n" + "-" * 20 + "\nbody\n") is None

    def test_numbered_list_not_flagged(self):
        assert find_content_runaway(
            "1. a\n2. b\n3. c\n4. d\n5. e\n6. f\n"
        ) is None

    def test_cicd_plan_still_clean(self):
        # The original structure-aware regression fixture must stay green
        # through the new pre-gate too (no 5x exact word/unit run in it).
        plan = (
            "## CI/CD Pipeline Architecture\n\n"
            "1. **Testing Stage**: run pytest across all 5 microservices.\n"
            "2. **Build Stage**: build Docker images with multi-stage caching.\n"
            "3. **Deploy Stage**: roll out to Kubernetes via Helm canary.\n"
        ) * 3
        assert find_content_runaway(plan) is None
        assert not is_degenerate_content(plan)


class TestContentRunawayDetectorIntegration:
    def test_content_field_check_fires_via_detector(self, monkeypatch):
        # End-to-end: the runaway buffer routed through RepetitionDetector as a
        # content field yields a Detection (so stream_guard would cancel+retry).
        # Force the detector enabled so the assertion doesn't depend on DB
        # runtime config (absent in unit context). Signature is
        # check(buffer, field, delta).
        from confucius.core.chat_models.stream_guard import detectors as _d
        monkeypatch.setattr(_d._cfg, "is_detector_enabled", lambda name: True)
        monkeypatch.setattr(
            _d._cfg, "field_repetition_coverage_pct", lambda field, default_pct=50: 50
        )
        det = _d.RepetitionDetector()
        result = det.check("service " * 6, "content", "")
        assert result is not None, (
            "RepetitionDetector must flag content-field word-runaway"
        )
        assert result.field == "content"

    def test_tool_arg_field_unaffected_by_content_pregate(self):
        # The content pre-gate must NOT leak into the strict tool-arg path
        # (is_degenerate / R1). Build a buffer where the runaway is a small
        # minority (below R1 coverage, no single-char run) so R1 alone does
        # NOT fire — only the content pre-gate does. The two paths must DIFFER.
        varied = (
            "Deploy the canary to staging and watch rollout metrics for "
            "error-rate regressions before promoting the build to production. "
        )
        buf = varied + ("service " * 6) + varied
        assert is_degenerate_content(buf), "content pre-gate fires on the runaway"
        assert not is_degenerate(buf), (
            "strict tool-arg path (R1) must NOT inherit the content pre-gate"
        )
        assert is_degenerate(buf) == is_degenerate_unit_repetition(buf)
