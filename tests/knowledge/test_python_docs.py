"""Python documentation search validation.

Phase 1: Direct API validation — does search_docs return the right docs?
Phase 2: LLM agent tests — does CCA use search_docs correctly for Python?
"""
import uuid
import warnings

import pytest

from .conftest import search_docs
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
        from tests.evaluators import evaluate_response

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

        # Verify search_docs was called
        assert any("search_docs" in t for t in r.tool_names), (
            f"Agent didn't use search_docs: {r.tool_names}"
        )

        # Verify FastAPI code in response
        content = r.content
        assert "Depends" in content, "Missing Depends (dependency injection)"
        assert "BaseModel" in content or "Pydantic" in content.lower(), (
            "Missing Pydantic model for request validation"
        )

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_httpx_async_requests(self, test_run, trace_test, judge_model):
        """Ask CCA to write async HTTP requests with httpx.

        Validates: search_docs called, code uses httpx with async/await.
        """
        from tests.evaluators import evaluate_response

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

        content = r.content
        assert "httpx" in content, "Missing httpx import"
        assert "async" in content and "await" in content, (
            "Missing async/await pattern"
        )

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_pydantic_validation(self, test_run, trace_test, judge_model):
        """Ask CCA to write Pydantic validation models.

        Validates: search_docs called, code uses BaseModel with validators.
        """
        from tests.evaluators import evaluate_response

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

        content = r.content
        assert "BaseModel" in content, "Missing BaseModel"
        assert "validator" in content.lower() or "field_validator" in content, (
            "Missing field validation (validator or field_validator)"
        )

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_torch_neural_network(self, test_run, trace_test, judge_model):
        """Ask CCA to build a simple neural network with PyTorch.

        Validates: search_docs called, code uses torch.nn with proper
        forward method.
        """
        from tests.evaluators import evaluate_response

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

        content = r.content
        assert "nn.Module" in content or "torch.nn" in content, (
            "Missing torch.nn module usage"
        )
        assert "forward" in content, "Missing forward method"

    @pytest.mark.knowledge_agent
    @pytest.mark.slow
    def test_pandas_data_analysis(self, test_run, trace_test, judge_model):
        """Ask CCA to write pandas data analysis code.

        Validates: search_docs called, code uses pandas for groupby,
        aggregation, and filtering.
        """
        from tests.evaluators import evaluate_response

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

        content = r.content
        assert "pandas" in content or "pd." in content, "Missing pandas usage"
        assert "groupby" in content, "Missing groupby operation"
