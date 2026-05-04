"""Unit tests for the Phase 0 context fingerprint helper.

Exercises the pure functions in confucius/core/context_fingerprint.py and
the OTel context propagation in confucius/core/tracing.py:

- compute_fingerprint() returns deterministic, byte-stable output.
- All 5 fields are present on the result.
- as_span_attrs() emits the exact 5 keys we want in Phoenix.
- Whitespace / unicode / None inputs are handled cleanly.
- using_context_fingerprint() + ContextFingerprintSpanProcessor stamp the
  attrs onto child spans created inside the block.

Run:
    pytest tests/unit/test_context_fingerprint.py -v
"""
from __future__ import annotations

from confucius.core.context_fingerprint import (
    ContextFingerprint,
    compute_fingerprint,
    hash_prompt,
    hash_tool_schemas,
    safe_compute_fingerprint,
)
from confucius.core.tracing import (
    CONTEXT_BUNDLE_ID,
    CONTEXT_PROMPT_HASH,
    CONTEXT_TOOL_SCHEMA_HASH,
    EMBED_MODEL_VERSION,
    PIPELINE_VERSION_ATTR,
)


# ── Pure helper tests ──


class TestHashHelpers:
    def test_hash_prompt_empty_uses_sentinel(self):
        # Empty SHA-256 — easily searchable in Phoenix as "missing prompt"
        empty_sha = (
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855"
        )
        assert hash_prompt(None) == empty_sha
        assert hash_prompt("") == empty_sha

    def test_hash_prompt_deterministic(self):
        body = "You are a coding agent.\n\nFollow these rules..."
        assert hash_prompt(body) == hash_prompt(body)

    def test_hash_prompt_changes_on_edit(self):
        body_a = "You are a coding agent."
        body_b = "You are a coding agent. "  # trailing space
        assert hash_prompt(body_a) != hash_prompt(body_b)

    def test_hash_tool_schemas_order_independent(self):
        a = [{"name": "edit_file"}, {"name": "run_bash"}]
        b = [{"name": "run_bash"}, {"name": "edit_file"}]
        assert hash_tool_schemas(a) == hash_tool_schemas(b)

    def test_hash_tool_schemas_empty(self):
        assert hash_tool_schemas(None) == hash_tool_schemas([])

    def test_hash_tool_schemas_changes_on_added_tool(self):
        a = [{"name": "edit_file"}]
        b = [{"name": "edit_file"}, {"name": "search_notes"}]
        assert hash_tool_schemas(a) != hash_tool_schemas(b)


# ── compute_fingerprint() ──


class TestComputeFingerprint:
    @staticmethod
    def _sample_inputs():
        return dict(
            route_name="coder",
            prompt_body="You are a coding agent.",
            tool_schemas=[{"name": "edit_file"}, {"name": "run_bash"}],
            embed_model="Qwen/Qwen3-Embedding-8B",
            pipeline_version=5,
        )

    def test_all_five_fields_populated(self):
        fp = compute_fingerprint(**self._sample_inputs())
        assert isinstance(fp, ContextFingerprint)
        assert len(fp.bundle_id) == 64  # sha256 hex
        assert len(fp.prompt_hash) == 64
        assert len(fp.tool_schema_hash) == 64
        assert fp.embed_model_version == "Qwen/Qwen3-Embedding-8B"
        assert fp.pipeline_version == "5"

    def test_deterministic_across_calls(self):
        fp1 = compute_fingerprint(**self._sample_inputs())
        fp2 = compute_fingerprint(**self._sample_inputs())
        assert fp1 == fp2

    def test_bundle_id_changes_on_route_change(self):
        a = compute_fingerprint(**self._sample_inputs())
        inputs = self._sample_inputs()
        inputs["route_name"] = "search"
        b = compute_fingerprint(**inputs)
        assert a.bundle_id != b.bundle_id

    def test_bundle_id_changes_on_prompt_edit(self):
        a = compute_fingerprint(**self._sample_inputs())
        inputs = self._sample_inputs()
        inputs["prompt_body"] = "You are a different coding agent."
        b = compute_fingerprint(**inputs)
        assert a.bundle_id != b.bundle_id
        assert a.prompt_hash != b.prompt_hash
        assert a.tool_schema_hash == b.tool_schema_hash

    def test_bundle_id_changes_on_pipeline_bump(self):
        a = compute_fingerprint(**self._sample_inputs())
        inputs = self._sample_inputs()
        inputs["pipeline_version"] = 6
        b = compute_fingerprint(**inputs)
        assert a.bundle_id != b.bundle_id
        assert a.prompt_hash == b.prompt_hash
        assert b.pipeline_version == "6"
        assert a.pipeline_version == "5"

    def test_unknown_embed_model_falls_back(self):
        inputs = self._sample_inputs()
        inputs["embed_model"] = None
        fp = compute_fingerprint(**inputs)
        assert fp.embed_model_version == "unknown"

    def test_unknown_pipeline_version_falls_back(self):
        inputs = self._sample_inputs()
        inputs["pipeline_version"] = None
        fp = compute_fingerprint(**inputs)
        assert fp.pipeline_version == "unknown"


class TestSpanAttrs:
    def test_as_span_attrs_has_5_keys(self):
        fp = compute_fingerprint(
            route_name="coder",
            prompt_body="x",
            tool_schemas=[{"name": "y"}],
            embed_model="m",
            pipeline_version=1,
        )
        attrs = fp.as_span_attrs()
        assert set(attrs.keys()) == {
            CONTEXT_BUNDLE_ID,
            CONTEXT_PROMPT_HASH,
            CONTEXT_TOOL_SCHEMA_HASH,
            EMBED_MODEL_VERSION,
            PIPELINE_VERSION_ATTR,
        }


class TestSafeWrapper:
    def test_returns_none_on_bad_input(self):
        # Pass a non-string prompt_body that triggers the encode path failure.
        # compute_fingerprint expects str|None; an object without encode raises
        # at hash_prompt → safe wrapper swallows.
        class NotAString:
            def __bool__(self):
                return True

        result = safe_compute_fingerprint(
            route_name="coder",
            prompt_body=NotAString(),  # type: ignore[arg-type]
            tool_schemas=[],
            embed_model="m",
            pipeline_version=1,
        )
        assert result is None

    def test_returns_fingerprint_on_good_input(self):
        result = safe_compute_fingerprint(
            route_name="coder",
            prompt_body="x",
            tool_schemas=[],
            embed_model="m",
            pipeline_version=1,
        )
        assert result is not None
        assert isinstance(result, ContextFingerprint)


# ── Span propagation ──


class TestSpanPropagation:
    """Verify that using_context_fingerprint() actually attaches the
    5 attrs to a span created inside the block. This is the integration
    seam between the helper and the SpanProcessor."""

    def test_child_span_gets_5_attrs(self):
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            SimpleSpanProcessor,
        )
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from confucius.core.tracing import (
            ContextFingerprintSpanProcessor,
            using_context_fingerprint,
        )

        # Isolated provider so we don't pollute the global one
        provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
        provider.add_span_processor(ContextFingerprintSpanProcessor())
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        fp = compute_fingerprint(
            route_name="coder",
            prompt_body="prompt body",
            tool_schemas=[{"name": "edit_file"}],
            embed_model="Qwen/Qwen3-Embedding-8B",
            pipeline_version=5,
        )

        with using_context_fingerprint(fp):
            with tracer.start_as_current_span("test.child"):
                pass

        provider.force_flush()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert attrs[CONTEXT_BUNDLE_ID] == fp.bundle_id
        assert attrs[CONTEXT_PROMPT_HASH] == fp.prompt_hash
        assert attrs[CONTEXT_TOOL_SCHEMA_HASH] == fp.tool_schema_hash
        assert attrs[EMBED_MODEL_VERSION] == fp.embed_model_version
        assert attrs[PIPELINE_VERSION_ATTR] == fp.pipeline_version

    def test_no_attrs_when_fingerprint_is_none(self):
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from confucius.core.tracing import (
            ContextFingerprintSpanProcessor,
            using_context_fingerprint,
        )

        provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
        provider.add_span_processor(ContextFingerprintSpanProcessor())
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("test")

        with using_context_fingerprint(None):
            with tracer.start_as_current_span("test.child"):
                pass

        provider.force_flush()
        spans = exporter.get_finished_spans()
        attrs = dict(spans[0].attributes or {})
        assert CONTEXT_BUNDLE_ID not in attrs
        assert CONTEXT_PROMPT_HASH not in attrs
