# CCA Failing Tests — 2026-05-16 audit

**Source**: latest CI run (`/mnt/cca-reports/P199{00-21}_*`).
**Scope**: every test report from the run; pass/fail truth-check vs the
user's "tests pass but they didn't really" + "previously-passing now failing"
concerns.

## Pass/fail accounting (current design — UNCHANGED per user decision)

| Decision layer | Files | Gates? |
|---|---|---|
| pytest assertions in test body | `tests/**/test_*.py` | **YES** — raises → pytest fails |
| Code evaluators (gating ones: `no_error`, `no_refusal`, `response_not_empty`, `coherence`, `tool_errors`) | `tests/evaluators.py:1019-1023` | **YES** — score 0.0 → AssertionError → pytest fails |
| Code evaluators (advisory: `latency`, `iteration_efficiency`, `response_duplication`, `code_present`, `user_identified`) | same file | **NO** — logged only |
| LLM judge (`response_quality`, `task_completion`) | `tests/evaluators.py` + `tests/conftest.py:494-531` | **NO (by design)** — informational, posted to Phoenix, never raises |

User decision (2026-05-16): keep judge advisory; fix the loud regressions only.

## Real regressions in this run

### 1. `tests/knowledge/test_verifier.py` — 6/8 fail, AttributeError (P19921)

```
AttributeError: <module 'confucius.server.knowledge.verifier' …> does
not have the attribute 'httpx'
```

**Cause**: the test does `with patch.object(verifier, "httpx") …` (line 82,
96, 111, 127, 148, 192) but `verifier.py` no longer imports `httpx` at the
module level. Drift between test mock target and refactored production code,
almost certainly from a "sync from auto-discovery" commit that updated one
side and not the other.

**Fix shape**: patch the actual HTTP client wherever verifier.py uses it
today (likely an injected client passed in via constructor or a function-local
import). Read `verifier.py` for the real call site; update tests to patch
that.

### 2. Knowledge / docs tests — 4 files, ~114 failures total (P19915, P19918, P19919, P19920)

Same shape across all four:
- P19915_api-sdk-docs: 29 fail — `No results for 'providers-atlassian-jira'`
- P19918_nutanix-sdk: 13 fail — `No results for 'nutanix nutanix-aiops'`
- P19919_powershell-docs: 18 fail — `No results for 'Connect-Widget'`
- P19920_python-docs: 54 fail — `No results for 'providers-atlassian-jira'`

**Cause hypothesis**: depends on the verifier-pipeline being healthy. If
the verifier's HTTP client isn't being mocked/injected correctly in tests
(see #1) OR if the BM25 index lost the relevant docs entries during a
recent re-seed, every "lookup package X" query returns []. The high
failure count (~114) plus uniform pattern strongly suggests a single root
cause — likely the verifier breakage cascading to the dependent tests.

**Fix shape**: fix verifier first (#1). If knowledge tests still fail after,
inspect `confucius/server/knowledge/` BM25 index state on Spark1.

### 3. `tests/coder/test_bash_execution.py` — 1 fail (P19916)

```
AssertionError: 'docker compose': no documentation snippet returned
assert ''
```

**Cause**: `search_docs("docker compose")` returned empty. Either the
docker-compose docs entry isn't in the knowledge corpus, or the search
pipeline returned `""` because of the verifier regression. Probably
linked to #1+#2.

### 4. `tests/integration/test_eva_code_trace.py` — 1 fail (P19903)

```
AssertionError: Code evaluator 'tool_errors' FAILED: label=1_errors,
explanation=? failed cmd='view'
```

**Cause**: real product issue. A `view` cmd in the EVA trace flow
failed and wasn't retried. The `tool_errors` evaluator was promoted
from advisory to gating in Phase 3.3 — that's correct behavior; it's
exposing a real CCA bug, not a false positive.

**Fix shape**: chase the `view` cmd failure in the actual trajectory.
Open `/mnt/cca-reports/P19903_eva-code-trace/report.md` (or whatever
the artifact path is) and trace the tool call that failed. Either
fix the underlying file-view tool (likely candidate: `str_replace_editor`
view action) OR tighten retry policy if the cmd is transient.

### 5. `tests/integration/test_routing_edge_cases.py` — 1 fail (P19912)

```
AssertionError: No evidence that tests ran: I need to create a unit
test file for the calculator operations using Python's unittest
module, testing ad…
```

**Cause**: test-body assertion that grep'd the response for "I ran the
tests" or similar markers and didn't find them. The agent wrote test
code but never executed it. Either the agent loop terminated before
running tests, or the assertion is too strict about the wording.

**Fix shape**: inspect the report to see if the agent should have used
`bash -c "pytest …"` and didn't. If yes, a prompt nudge. If no, relax
the assertion to accept "tests written but execution deferred" as a
soft pass.

### 6. `tests/integration/test_gitnexus_parity.py` — 0 fail but suspicious WARN (P19906)

Test passed (`fail=0 err=0`) but emitted: `too few CALLS edges (16) for
process detection`. This is the "passed but maybe shouldn't" pattern
the user noticed. The judge or an evaluator emitted a warning rather
than raising.

**Cause**: a threshold check that warns instead of asserts.

**Fix shape**: either lower the threshold (16 CALLS edges for a tiny
project is fine, don't warn) or convert the warn → assert if the
threshold reflects a real quality bar. Read the eval to decide.

## All clean (run again for stability)

P19917_multi-language, P19900_workspace-indexing, P19901_5-retrieval-modes,
P19902_cross-session-recall, P19904_eva-full-trace, P19905_file-scope-tabs,
P19907_infra-inspection-flow, P19908_knowledge-pipeline, P19909_negative-cases,
P19910_orchestrator-process-chain, P19911_project-scoped-notes,
P19913_security-edge-cases, P19914_tool-isolation, P19898_document-workflow,
P19899_rule-lifecycle.

## Recommended fix order

1. **`test_verifier.py`** — patch the right target. Smallest blast radius;
   probably unblocks half the knowledge failures cascade.
2. **Knowledge / docs tests** — re-run after #1; expect significant
   recovery. Anything still failing is a real BM25 index gap.
3. **`test_eva_code_trace`** — investigate the `view` cmd failure. Real
   product bug to chase.
4. **`test_routing_edge_cases`** — read the trajectory; decide whether the
   "tests not run" assertion is correct.
5. **`test_gitnexus_parity` warning** — threshold tune or upgrade to gating.

## Notes on test infrastructure

- Per-test Phoenix project routing (`PHOENIX_PROJECT_NAME=test/{name}`) is
  working end-to-end — every report's `metadata.json` shows the correct
  project. The trace-tree shape issue (Slice B) is separate from this audit.
- Several auto-discovery sync commits in May overwrote test files
  (`tests/coder/*`, `tests/knowledge/*`). The verifier mismatch in #1 is
  consistent with that pattern. Diff each sync commit against its previous
  state before re-running affected tests.
