# ASR MTP graph verification (2026-11-09)

## Results

- Related tests: 88 passed; actual NPU capture/replay test skipped on this CPU host.
- Full current research suite: 255 tests, 234 passed, 19 errors, 1 failure, 1 NPU skip.
- Unmodified HEAD b63fbdaaf exported into an isolated baseline: 208 tests,
  188 passed, the same 19 errors and 1 failure. Failure/error identities match
  exactly; there are no additional failing identities in the current suite.
- Windows sandbox could not initialize even asyncio.run(asyncio.sleep(0)).
  Async and full-suite checks were rerun outside that sandbox and completed.
- Fresh and existing-MTP upgrade artifacts applied in isolated Git fixtures;
  installed modules equal source, compile, and repeat export is empty.
- Installed upgrade graph integration tests: 12 CPU passed, 1 NPU skipped.
- New graph source/tests pass Ruff and format checks; changed Python sources
  pass syntax/import checks; whitespace checked separately.
- Independent review found graph-pool KV accounting omission; fixed by preparing
  before eager profile and reserving retained memory at startup. The follow-up
  identified NPUWorker310's separate memory path; full graphs are explicitly
  rejected there. 910B review found no remaining P1/P2 in the reviewed path.

## Existing full-suite failures/errors (also present in HEAD)

```text
ERROR: test_first_verify_reads_bookkept_prefix_and_stops_after_failed_aligned_retry (test_external_draft_first_verify_patch.EnginePatchTests.test_first_verify_reads_bookkept_prefix_and_stops_after_failed_aligned_retry)
ERROR: test_prespec_can_retry_same_anchor_in_new_target_context (test_external_draft_first_verify_patch.EnginePatchTests.test_prespec_can_retry_same_anchor_in_new_target_context)
ERROR: test_prespec_complete_aligned_suffix_has_no_more_candidate (test_external_draft_first_verify_patch.EnginePatchTests.test_prespec_complete_aligned_suffix_has_no_more_candidate)
ERROR: test_prespec_partial_aligned_acceptance_switches_to_ngram (test_external_draft_first_verify_patch.EnginePatchTests.test_prespec_partial_aligned_acceptance_switches_to_ngram)
ERROR: test_prespec_rechecks_ngram_after_ar_and_later_anchor (test_external_draft_first_verify_patch.EnginePatchTests.test_prespec_rechecks_ngram_after_ar_and_later_anchor)
ERROR: test_prespec_target_only_matching_token_keeps_alignment (test_external_draft_first_verify_patch.EnginePatchTests.test_prespec_target_only_matching_token_keeps_alignment)
ERROR: test_prespec_truncated_full_acceptance_keeps_aligned_continuation (test_external_draft_first_verify_patch.EnginePatchTests.test_prespec_truncated_full_acceptance_keeps_aligned_continuation)
ERROR: test_prespec_uses_ngram_after_aligned_second_token_rejected (test_external_draft_first_verify_patch.EnginePatchTests.test_prespec_uses_ngram_after_aligned_second_token_rejected)
ERROR: test_deadline_aborts_shadow_without_waiting_for_formal_submit (test_streaming_prespec_helpers.ModelHelperTests.test_deadline_aborts_shadow_without_waiting_for_formal_submit)
ERROR: test_error_or_truncated_stream_does_not_publish_partial_candidate (test_streaming_prespec_helpers.ModelHelperTests.test_error_or_truncated_stream_does_not_publish_partial_candidate) (mode='error')
ERROR: test_error_or_truncated_stream_does_not_publish_partial_candidate (test_streaming_prespec_helpers.ModelHelperTests.test_error_or_truncated_stream_does_not_publish_partial_candidate) (mode='truncated')
ERROR: test_first_chunk_without_old_draft_can_launch (test_streaming_prespec_helpers.ModelHelperTests.test_first_chunk_without_old_draft_can_launch)
ERROR: test_formal_submit_cancels_shadow_and_aborts_only_its_request (test_streaming_prespec_helpers.ModelHelperTests.test_formal_submit_cancels_shadow_and_aborts_only_its_request)
ERROR: test_full_audio_snapshot_and_corrected_continuation (test_streaming_prespec_helpers.ModelHelperTests.test_full_audio_snapshot_and_corrected_continuation)
ERROR: test_full_kmax_old_draft_is_verified_but_oversized_result_not_consumed (test_streaming_prespec_helpers.ModelHelperTests.test_full_kmax_old_draft_is_verified_but_oversized_result_not_consumed)
ERROR: test_old_draft_is_sent_whole_with_separate_output_budget (test_streaming_prespec_helpers.ModelHelperTests.test_old_draft_is_sent_whole_with_separate_output_budget)
ERROR: test_output_budget_covers_first_token_old_draft_and_extension (test_streaming_prespec_helpers.ModelHelperTests.test_output_budget_covers_first_token_old_draft_and_extension)
ERROR: test_oversized_old_draft_is_skipped_before_engine_submission (test_streaming_prespec_helpers.ModelHelperTests.test_oversized_old_draft_is_skipped_before_engine_submission)
ERROR: test_skip_without_submitting_or_advancing_formal_cutoff (test_streaming_prespec_helpers.ModelHelperTests.test_skip_without_submitting_or_advancing_formal_cutoff)
FAIL: test_default_policy_runs_twenty_connection_shadows_concurrently (test_streaming_prespec.PreSpecCoordinatorTests.test_default_policy_runs_twenty_connection_shadows_concurrently)
```

## Evidence and boundaries

Raw logs are under `.tmp/mtp-test-env/`: `mtp-graph-full-suite-final.log`,
`mtp-graph-head-baseline.log`, and `mtp-graph-installed-tests.log`.
The full run includes unrelated pre-existing dirty research work, which was left
untouched. The comparison above reports matching failure identities rather than
claiming that every unrelated change is validated.

No NPU capture, checkpoint numerical acceptance, or online latency result is
claimed. The device test must run without skipping; then compare graph off/full
using the same checkpoint, head count, service settings, and workload. Graph
memory reservation is conservative and may double-count some static buffers.
Explicit manual KV budgets remain the operator's responsibility.

## Delivery

Development sources and the full installer remain in the ASR research tree.
The engine upgrade patch, documentation, and mirrored unit/integration tests
are prepared in the existing `310P-graph` isolated checkout at
`.tmp/mtp-test-env/engine-310P-graph-mixed-full/examples/910b/`.
No commits were pushed or services restarted during implementation verification.
The subsequent user request authorizes publication of these changes to the two
existing branches; it does not change the pending device-validation status.
