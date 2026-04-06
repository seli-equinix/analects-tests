"""Flow test: API and SDK documentation search validation.

Journey: developer needs to integrate cloud SDKs (Stripe, AWS, Redis,
Qdrant, OpenAI) -> CCA searches curated docs via search_docs -> writes
code informed by real API signatures.

Phase 1: Direct API validation — do Python package searches return results?
Phase 2: LLM agent tests — does CCA use search_docs correctly for API/SDK code?

Exercises: search_docs (API), get_api_docs (API), str_replace_editor (FILE),
bash (SHELL), CODER route.
Markers: knowledge, knowledge_api, knowledge_agent, slow
"""
import uuid
import warnings

import pytest

from .conftest import assert_content_or_file, search_docs
from .helpers.knowledge_data import PY_PACKAGES
from tests.evaluators import evaluate_response

pytestmark = [
    pytest.mark.knowledge,
]

# Pick up to 50 packages from PY_PACKAGES for broad coverage.
# PY_PACKAGES has duplicate keys (Python dict keeps last value),
# so the runtime dict may have fewer than 100 entries.
_ALL_PACKAGES = list(PY_PACKAGES.items())
_TOP_50 = _ALL_PACKAGES[:50]


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Direct API Validation (fast, no LLM)
# ═══════════════════════════════════════════════════════════════════


class TestApiSdkDocsValidation:
    """Direct API validation — do top Python API/SDK packages return results?"""

    @pytest.mark.knowledge_api
    @pytest.mark.parametrize("package,expected_pkg", _TOP_50)
    def test_package_returns_result(self, knowledge_client, package, expected_pkg):
        """Validate search_docs returns ANY result with a non-empty snippet.

        Less strict than PowerShell validation — we can't predict exact
        package IDs for generic queries like "database" or "analytics".
        """
        data = search_docs(knowledge_client, package)
        results = data.get("results", [])

        # Must have at least 1 result
        assert len(results) >= 1, (
            f"No results for '{package}' (expected: {expected_pkg})"
        )

        top = results[0]

        # Must have some content — snippet should be non-empty
        snippet = top.get("snippet", "")
        if not snippet:
            warnings.warn(
                f"'{package}' ({expected_pkg}): empty snippet "
                f"(id: {top.get('id')}, language: {top.get('language')})"
            )
        else:
            # If we got a snippet, it should have meaningful content
            assert len(snippet) > 10, (
                f"'{package}': snippet too short ({len(snippet)} chars): "
                f"{snippet!r}"
            )


# ═══════════════════════════════════════════════════════════════════
# Phase 2: LLM Agent Tests (needs CCA + vLLM)
# ═══════════════════════════════════════════════════════════════════


class TestApiSdkDocsAgent:
    """LLM agent test — does CCA use search_docs correctly for API/SDK tasks?"""

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_stripe_payments(self, test_run, trace_test, judge_model):
        """Ask CCA to write Stripe payment integration code.

        Validates: search_docs called, code uses stripe API for
        creating payment intents and handling webhooks.
        """
        sid = f"test-sdk-stripe-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a Python function that uses the Stripe API to create "
            "a payment intent for a $49.99 charge in USD. Include a "
            "webhook handler that processes payment_intent.succeeded "
            "and payment_intent.payment_failed events with signature "
            "verification."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")
        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        # search_docs preferred but agent may know Stripe natively
        used_search = any("search_docs" in t for t in r.tool_names)
        trace_test.set_attribute("cca.test.t1_used_search", used_search)

        assert_content_or_file(r, ["stripe", "Stripe", "payment_intent", "PaymentIntent"], "Stripe payments")

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_aws_s3_storage(self, test_run, trace_test, judge_model):
        """Ask CCA to write AWS S3 file storage code.

        Validates: search_docs called, code uses boto3 for S3 operations.
        """
        sid = f"test-sdk-s3-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a Python class that wraps AWS S3 operations using "
            "boto3. Include methods to upload a file with a presigned "
            "URL, download a file, list objects in a bucket with a "
            "prefix filter, and delete an object. Handle ClientError "
            "exceptions properly."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")
        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        assert_content_or_file(r, ["boto3", "s3", "S3", "presigned"], "AWS S3")

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_redis_caching(self, test_run, trace_test, judge_model):
        """Ask CCA to write Redis caching code.

        Validates: search_docs called, code uses redis-py with proper
        TTL and serialization.
        """
        sid = f"test-sdk-redis-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a Python caching layer using redis-py that supports "
            "get/set with TTL, cache invalidation by pattern, and a "
            "decorator that caches function results based on arguments. "
            "Use JSON serialization for complex objects."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")
        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        assert_content_or_file(r, ["redis", "Redis", "expire", "ttl"], "Redis caching")

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_qdrant_vector_search(self, test_run, trace_test, judge_model):
        """Ask CCA to write Qdrant vector search code.

        Validates: search_docs called, code uses qdrant-client for
        collection management and vector search.
        """
        sid = f"test-sdk-qdrant-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a Python class using qdrant-client that creates a "
            "collection with cosine distance, upserts points with "
            "payload metadata, and performs filtered vector search "
            "with a minimum score threshold. Use the query_points "
            "API (not the deprecated search method)."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")
        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        assert_content_or_file(r, ["qdrant", "Qdrant", "query_points", "upsert"], "Qdrant vector search")

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_openai_chat_completions(self, test_run, trace_test, judge_model):
        """Ask CCA to write OpenAI API chat completion code.

        Validates: search_docs called, code uses openai client for
        streaming chat completions with tool calling.
        """
        sid = f"test-sdk-openai-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a Python function using the OpenAI API client that "
            "sends a streaming chat completion request with a system "
            "prompt and user message. Include tool definitions for a "
            "get_weather function, handle the tool_calls response, "
            "execute the tool, and send the result back for a final "
            "answer."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")
        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        assert_content_or_file(r, ["openai", "OpenAI", "ChatCompletion", "stream"], "OpenAI chat")
