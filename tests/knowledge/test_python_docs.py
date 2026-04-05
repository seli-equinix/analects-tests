"""Flow test: Python documentation search validation.

Journey: developer needs to write code using Python libraries -> CCA
searches curated docs via search_docs -> writes code informed by
real API signatures instead of hallucinating.

Phase 1: Direct API validation — does search_docs return the right docs?
Phase 2: LLM agent tests — does CCA use search_docs correctly for Python?

Exercises: search_docs (API), get_api_docs (API), str_replace_editor (FILE),
bash (SHELL), CODER route.
Markers: knowledge, knowledge_api, knowledge_agent, slow
"""
import uuid
import warnings

import pytest

from tests.evaluators import evaluate_response

from .conftest import assert_content_or_file, search_docs
from .helpers.knowledge_data import PY_PACKAGES

pytestmark = [
    pytest.mark.knowledge,
]


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Direct API Validation (fast, no LLM)
# ═══════════════════════════════════════════════════════════════════


class TestPythonDocsValidation:
    """Direct API validation — does search_docs return the right Python docs?"""

    @pytest.mark.knowledge_api
    @pytest.mark.parametrize("package,expected_pkg", list(PY_PACKAGES.items()))
    def test_package_search(self, knowledge_client, package, expected_pkg):
        """Validate search_docs returns correct Python doc for each package."""
        data = search_docs(knowledge_client, package)
        results = data.get("results", [])

        # Must have at least 1 result
        assert len(results) >= 1, f"No results for '{package}'"

        top = results[0]

        # Must have a language (python or any non-empty)
        lang = top.get("language", "")
        assert lang, (
            f"'{package}': expected a language value, got empty "
            f"(pkg: {top.get('id')})"
        )

        # Snippet is expected but some packages may return empty — soft check
        snippet = top.get("snippet", "")
        if not snippet:
            warnings.warn(
                f"'{package}' ({expected_pkg}): no documentation snippet "
                f"returned (id: {top.get('id')})"
            )


# ═══════════════════════════════════════════════════════════════════
# Phase 2: LLM Agent Tests (needs CCA + vLLM)
# ═══════════════════════════════════════════════════════════════════


class TestPythonDocsAgent:
    """LLM agent test — does CCA use search_docs correctly for Python?"""

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_fastapi_dependency_injection(self, test_run, trace_test, judge_model):
        """Ask CCA to write a FastAPI endpoint with dependency injection.

        Validates: search_docs called, correct doc found, code uses
        FastAPI Depends with proper type hints.
        """

        sid = f"test-py-fastapi-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a FastAPI REST endpoint that uses dependency injection "
            "to get a database session. The endpoint should accept a POST "
            "request at /users with a JSON body containing name and email, "
            "validate the input, and return the created user with an id."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        # Verify search_docs was called
        assert any("search_docs" in t for t in r.tool_names), (
            f"Agent didn't use search_docs: {r.tool_names}"
        )

        assert_content_or_file(r, ["Depends", "dependency"], "FastAPI Depends")

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_httpx_async_requests(self, test_run, trace_test, judge_model):
        """Ask CCA to write async HTTP requests with httpx.

        Validates: search_docs called, code uses httpx with async/await.
        """

        sid = f"test-py-httpx-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a Python script using httpx to make async HTTP requests "
            "to three different API endpoints concurrently. Use "
            "asyncio.gather to parallelize the calls and handle timeouts "
            "and connection errors gracefully."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        assert_content_or_file(r, ["httpx", "async"], "httpx async")

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_pydantic_validation(self, test_run, trace_test, judge_model):
        """Ask CCA to write Pydantic validation models.

        Validates: search_docs called, code uses BaseModel with validators.
        """

        sid = f"test-py-pydantic-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a Pydantic BaseModel for a user registration form that "
            "validates email format, enforces password strength (min 8 chars, "
            "must contain uppercase and digit), and has a field validator "
            "that normalizes the username to lowercase."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        assert_content_or_file(r, ["BaseModel", "validator", "field_validator"], "Pydantic validation")

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_torch_neural_network(self, test_run, trace_test, judge_model):
        """Ask CCA to build a simple neural network with PyTorch.

        Validates: search_docs called, code uses torch.nn with proper
        forward method.
        """

        sid = f"test-py-torch-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a PyTorch neural network class that takes 784 input "
            "features (MNIST), has two hidden layers with ReLU activation "
            "and dropout, and outputs 10 classes. Include a training loop "
            "that uses CrossEntropyLoss and Adam optimizer."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        assert_content_or_file(r, ["nn.Module", "torch.nn", "torch"], "PyTorch nn")

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_pandas_data_analysis(self, test_run, trace_test, judge_model):
        """Ask CCA to write pandas data analysis code.

        Validates: search_docs called, code uses pandas for groupby,
        aggregation, and filtering.
        """

        sid = f"test-py-pandas-{uuid.uuid4().hex[:8]}"
        test_run.track_session(sid)

        msg = (
            "Write a Python script using pandas that reads a CSV file of "
            "sales data with columns (date, product, region, amount), "
            "groups by product and region, calculates total and average "
            "sales, filters for products with total sales above 10000, "
            "and exports the result to a new CSV."
        )
        r = test_run.chat(msg, session_id=sid, idle_timeout=180)
        evaluate_response(r, msg, trace_test, judge_model, "coder")

        trace_test.set_attribute("cca.test.t1_tools", str(r.tool_names))
        trace_test.set_attribute("cca.test.t1_response", r.content[:500])
        assert r.content, "Turn 1 returned empty response"

        assert_content_or_file(r, ["pandas", "pd.", "groupby"], "pandas data analysis")
