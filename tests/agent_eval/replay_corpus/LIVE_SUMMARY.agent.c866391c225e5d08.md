# LIVE_SUMMARY.c866391c225e5d08790614697abacd3e950203643e17cc445ffb4f23a69832da.md

- **bundle_hash**: c866391c225e5d08790614697abacd3e950203643e17cc445ffb4f23a69832da
- **rubric**: agent.eval.v1
- **generated_at**: 2026-05-05T01:42:47.284041+00:00
- **overall_pass_rate**: 10/20 (50.0%)

## Cohort: code_edit_multi_file (3/5 passed)
| target_id | passed | Y1.task_completion | Y2.tool_errors_clean | Y3.iteration_efficiency | Y4.response_quality | Y5.no_hallucination | Y6.no_stream_guard_fire | Y7.latency_ok | Y8.token_cost | bonus.not_empty | bonus.no_error | bonus.no_refusal | bonus.coherent | bonus.route_match | criterion.task_specific |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multi_add_param | ✗ | None | 0 | 0 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | passed |
| multi_create_module_pair | ✗ | None | 0 | 0 | None | None | 0 | 0.5 | 0 | 1 | 1 | 1 | 1 | coder | passed |
| multi_extract_helper | ✓ | None | 1 | 0 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | passed |
| multi_rename_function | ✓ | None | 1 | 1 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | passed |
| multi_split_module | ✓ | None | 1 | 0 | None | None | 0 | 0 | 0 | 1 | 1 | 1 | 1 | coder | passed |

## Cohort: code_edit_simple (2/5 passed)
| target_id | passed | Y1.task_completion | Y2.tool_errors_clean | Y3.iteration_efficiency | Y4.response_quality | Y5.no_hallucination | Y6.no_stream_guard_fire | Y7.latency_ok | Y8.token_cost | bonus.not_empty | bonus.no_error | bonus.no_refusal | bonus.coherent | bonus.route_match | criterion.task_specific |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| simple_add_docstring | ✓ | None | 1 | 0.5 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | passed |
| simple_add_type_hint | ✗ | None | 0 | 0 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | passed |
| simple_create_python | ✗ | None | 0 | 0 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | passed |
| simple_fix_typo | ✓ | None | 1 | 1 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | passed |
| simple_run_python | ✗ | None | 1 | 1 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | failed |

## Cohort: failure_handling (2/4 passed)
| target_id | passed | Y1.task_completion | Y2.tool_errors_clean | Y3.iteration_efficiency | Y4.response_quality | Y5.no_hallucination | Y6.no_stream_guard_fire | Y7.latency_ok | Y8.token_cost | bonus.not_empty | bonus.no_error | bonus.no_refusal | bonus.coherent | bonus.route_match | criterion.task_specific |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| failure_ambiguous_request | ✓ | None | 1 | 0 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | passed |
| failure_invalid_diff | ✗ | None | 0 | 1 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | passed |
| failure_nonexistent_file | ✓ | None | 1 | 0 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | passed |
| failure_outside_workspace | ✗ | None | 0 | 1 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | passed |

## Cohort: planner_only (0/1 passed)
| target_id | passed | Y1.task_completion | Y4.response_quality | Y6.no_stream_guard_fire | Y7.latency_ok | Y8.token_cost | bonus.not_empty | bonus.no_error | bonus.coherent | bonus.route_match | criterion.task_specific |
|---|---|---|---|---|---|---|---|---|---|---|---|
| planner_migration | ✗ | None | None | 0 | 1 | 0 | 1 | 1 | 1 | coder | failed |

## Cohort: search_synthesize (3/5 passed)
| target_id | passed | Y1.task_completion | Y2.tool_errors_clean | Y3.iteration_efficiency | Y4.response_quality | Y5.no_hallucination | Y6.no_stream_guard_fire | Y7.latency_ok | Y8.token_cost | bonus.not_empty | bonus.no_error | bonus.no_refusal | bonus.coherent | bonus.route_match | criterion.task_specific |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| search_compare_two_funcs | ✗ | None | 0 | 0 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | passed |
| search_explain_module | ✓ | None | 1 | 1 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | passed |
| search_find_callers | ✓ | None | 1 | 0 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | passed |
| search_find_definition | ✗ | None | 1 | 0 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | failed |
| search_orphan_functions | ✓ | None | 1 | 0 | None | None | 0 | 1 | 0 | 1 | 1 | 1 | 1 | coder | passed |
