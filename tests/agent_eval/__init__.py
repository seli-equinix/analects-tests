"""Agent evaluation suite (Phase 3.3+ of the Context-as-Code roadmap).

Public surface:

- ``metrics`` — adapters that register the existing ``tests/evaluators``
  functions as Phase-3.2 ``Metric`` objects so they compose into
  ``Rubric``s. Drop-in: existing tests keep calling
  ``evaluate_response()`` unchanged; the new path lives alongside.

- ``rubrics`` — route-specific bundles
  (``agent.coder.v1``, ``agent.search.v1``, ``agent.user.v1``,
  ``agent.infra.v1``, ``agent.planner.v1``).

- (Phase 3.5+) ``replay_corpus/`` — captured vLLM completions per task.
- (Phase 3.6) ``live_validate_agent.py`` — runner that emits
  ``LIVE_SUMMARY.agent.{bundle_hash}.md``.
"""
