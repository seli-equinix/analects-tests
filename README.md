# Analects Test Suite

Integration and end-to-end tests for [Analects](https://github.com/seli-equinix/analects) — an Agent-as-a-Model AI coding agent built on the Confucius framework.

## What This Repo Contains

This is the **public test suite** for Analects. It is automatically synced from the private Analects codebase via GitHub Actions whenever tests change.

```
tests/
  user/           # User identification, onboarding, memory recall
  coder/          # Code editing, bash execution, workspace indexing
  integration/    # Cross-feature flows, routing, security, tool isolation
  websearch/      # Web search integration
  helpers/        # Shared test utilities (polling, client)
  conftest.py     # pytest configuration, fixtures, Phoenix tracing
  cca_client.py   # HTTP client for CCA's OpenAI-compatible API
  evaluators.py   # Response quality evaluators (duplication, coherence, code)
  report_generator.py  # Markdown test report generation
```

## Prerequisites

- A running Analects (CCA) deployment accessible via HTTP
- A GitLab instance with a CI runner (for automated test execution)
- Python 3.12+

## Quick Start

### Run tests locally

```bash
pip install -r requirements-test.txt

# Point at your CCA deployment
export CCA_BASE_URL="http://your-cca-host:8500"

# Run all tests
python -m pytest tests/ -v --tb=short

# Run with LLM judge evaluation
python -m pytest tests/ -v --with-judge
```

### Run via GitLab CI

1. Push this repo to your GitLab instance
2. Set CI variables: `CCA_BASE_URL`, `CCA_TEST_API_KEY`, `PHOENIX_COLLECTOR_ENDPOINT`
3. Trigger a pipeline with `RUN_TEST=<test-name>` (e.g., `new-user-onboarding`)

### Run via the Analects Dashboard

The Analects web dashboard (`:8443/config` -> Test Setup tab) can:
- Push these test files to your GitLab project automatically
- Pull the latest tests from this repo
- Configure CI variables
- Run tests and view reports

## Test Categories

| Category | Tests | What They Validate |
|----------|-------|--------------------|
| **user** | 3 | New user onboarding, profile CRUD, returning user memory |
| **coder** | 11 | File editing, bash execution, code intelligence, workspace indexing |
| **integration** | 11 | Cross-session recall, routing edge cases, security, tool isolation |
| **websearch** | 1 | Web search end-to-end flow |

## Configuration

Tests connect to CCA via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CCA_BASE_URL` | `http://localhost:8500` | CCA server URL |
| `CCA_TEST_API_KEY` | *(none)* | API key for authenticated endpoints |
| `PHOENIX_COLLECTOR_ENDPOINT` | *(none)* | OpenTelemetry collector for trace export |

## License

MIT
