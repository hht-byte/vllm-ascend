# Cold/warm MTP depth implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline with TDD. User authorized this replacement; do not push or restart services.

**Goal:** Replace the uncommitted conditional EOS graph path with cold=5, warm=3 auxiliary execution.

**Architecture:** Track whether each engine request has actually offered an MTP proposal, independently of optional metrics. Capture bounded ordinary graphs per execution depth and existing token/BQ buckets. Group mixed phases by captured depth before execution and retain one batch D2H. Preserve target verifier, EOS truncation and cache validity rules.

**Tech Stack:** Python, Torch CPU parity tests; TorchNPU 2.10 compatible ordinary graph capture and existing FIA updates.

**Spec:** User-approved cold MTP5 / warm MTP3 scheme in this conversation.

## Constraints and decisions

- Preserve all unrelated dirty files. Audited EOS-only edits have been backed up under `.tmp/mtp-test-env/eos-rollback-backup` and reverted.
- `num_heads` remains the loaded maximum. Default `cold_heads=loaded_heads`, `warm_heads=min(3, loaded_heads)`; explicit values must be positive integers not exceeding loaded heads. Setting both equal restores fixed-depth behavior.
- Prefill-only, legacy rollback, disabled and finished/EOS requests do not consume the cold phase. Finish removes phase state. Rejection/recomputation within a request does not reset phase.
- No device EOS conditions, new per-head D2H, scheduler changes or target graph changes.
- Smaller remaining budgets select a containing captured depth; committed caches and returned output respect actual limits. Graph count validation includes all depth variants.
- Cold and warm metrics retain their existing request-proposal definitions. Keep the independent ignore_eos correction found during EOS work.

## Review focus

- Mixed cold/warm batches preserve ordering and persistent outputs before graph reuse.
- Heads outside the selected depth never execute or advance validity.
- Multi-round rejection and truncated budgets preserve numerical parity.
- Request ID reuse starts cold after cleanup; optional acceptance tracking does not control phase.
- All patch upgrade routes remove EOS requirements, deploy matching source and retain unrelated changes.

## Tasks

- [x] Audit and roll back conditional EOS artifacts only.
- [x] Write failing phase/lifecycle, graph depth, cache parity and configuration tests.
- [x] Implement worker phase selection and depth-aware ordinary graph capture/dispatch for token/BQ paths.
- [x] Verify eager/graph numerical parity, output transfer count and cleanup; preserve EOS stop handling.
- [x] Update documentation, patch exports and engine test mirrors; verify installation routes.
- [x] Run relevant and full research checks, compare pre-existing failures, lint and review final diff.

## Execution ledger

- Rollback: preserved an archive of each reverted file; no changes to unrelated handoff work.
- Ruling: use the existing delivery checkout and root working tree because the requested rollback concerns these exact uncommitted edits; do not move or discard other work.

## Verification and limits

- RED: new phase suite initially failed five cases: warm remained five, short phase budget fell back, mixed phases, and the existing ignore_eos bug.
- Related `test_asr_mtp_*.py`: 165 tests, 150 pass / 15 real-NPU skips.
- Installed patch test mirror: 143 tests, 128 pass / 15 NPU skips.
- Full research: 385 tests, 19 errors + 1 failure, exactly the same failure identities as `.tmp/mtp-test-env/mtp-eos-full.log`. Current log: `.tmp/mtp-test-env/mtp-phase-full.log`.
- Full suite initially blocked in Windows asyncio's socketpair initialization inside the sandbox; faulthandler located socket.accept, and the same suite completed outside that restriction. No production code change for this environment issue.
- Seven patch routes apply/check/reverse and match all 13 deployed modules; compilation and empty re-export checks pass. Twelve mirrored test files match AST. Verification script: `.tmp/mtp-test-env/verify_mtp_phase_release.py`.
- Actual NPU test added for ordinary paged 5/3 graphs with changing boundaries and rejection. This host has no NPU; no compatibility/performance claim is made for the server.
- Review: target/scheduler APIs unchanged; only actual proposal advances phase, cleanup resets request state; shallow programs instantiate only their FIA task count, preserve the full arena stride and only commit executed budgets. One batch tolist remains.

### Existing full-suite failure identities

- ERROR: test_deadline_aborts_shadow_without_waiting_for_formal_submit (test_streaming_prespec_helpers.ModelHelperTests.test_deadline_aborts_shadow_without_waiting_for_formal_submit)
- ERROR: test_error_or_truncated_stream_does_not_publish_partial_candidate (test_streaming_prespec_helpers.ModelHelperTests.test_error_or_truncated_stream_does_not_publish_partial_candidate) (mode='error')
- ERROR: test_error_or_truncated_stream_does_not_publish_partial_candidate (test_streaming_prespec_helpers.ModelHelperTests.test_error_or_truncated_stream_does_not_publish_partial_candidate) (mode='truncated')
- ERROR: test_first_chunk_without_old_draft_can_launch (test_streaming_prespec_helpers.ModelHelperTests.test_first_chunk_without_old_draft_can_launch)
- ERROR: test_first_verify_reads_bookkept_prefix_and_stops_after_failed_aligned_retry (test_external_draft_first_verify_patch.EnginePatchTests.test_first_verify_reads_bookkept_prefix_and_stops_after_failed_aligned_retry)
- ERROR: test_formal_submit_cancels_shadow_and_aborts_only_its_request (test_streaming_prespec_helpers.ModelHelperTests.test_formal_submit_cancels_shadow_and_aborts_only_its_request)
- ERROR: test_full_audio_snapshot_and_corrected_continuation (test_streaming_prespec_helpers.ModelHelperTests.test_full_audio_snapshot_and_corrected_continuation)
- ERROR: test_full_kmax_old_draft_is_verified_but_oversized_result_not_consumed (test_streaming_prespec_helpers.ModelHelperTests.test_full_kmax_old_draft_is_verified_but_oversized_result_not_consumed)
- ERROR: test_old_draft_is_sent_whole_with_separate_output_budget (test_streaming_prespec_helpers.ModelHelperTests.test_old_draft_is_sent_whole_with_separate_output_budget)
- ERROR: test_output_budget_covers_first_token_old_draft_and_extension (test_streaming_prespec_helpers.ModelHelperTests.test_output_budget_covers_first_token_old_draft_and_extension)
- ERROR: test_oversized_old_draft_is_skipped_before_engine_submission (test_streaming_prespec_helpers.ModelHelperTests.test_oversized_old_draft_is_skipped_before_engine_submission)
- ERROR: test_prespec_can_retry_same_anchor_in_new_target_context (test_external_draft_first_verify_patch.EnginePatchTests.test_prespec_can_retry_same_anchor_in_new_target_context)
- ERROR: test_prespec_complete_aligned_suffix_has_no_more_candidate (test_external_draft_first_verify_patch.EnginePatchTests.test_prespec_complete_aligned_suffix_has_no_more_candidate)
- ERROR: test_prespec_partial_aligned_acceptance_switches_to_ngram (test_external_draft_first_verify_patch.EnginePatchTests.test_prespec_partial_aligned_acceptance_switches_to_ngram)
- ERROR: test_prespec_rechecks_ngram_after_ar_and_later_anchor (test_external_draft_first_verify_patch.EnginePatchTests.test_prespec_rechecks_ngram_after_ar_and_later_anchor)
- ERROR: test_prespec_target_only_matching_token_keeps_alignment (test_external_draft_first_verify_patch.EnginePatchTests.test_prespec_target_only_matching_token_keeps_alignment)
- ERROR: test_prespec_truncated_full_acceptance_keeps_aligned_continuation (test_external_draft_first_verify_patch.EnginePatchTests.test_prespec_truncated_full_acceptance_keeps_aligned_continuation)
- ERROR: test_prespec_uses_ngram_after_aligned_second_token_rejected (test_external_draft_first_verify_patch.EnginePatchTests.test_prespec_uses_ngram_after_aligned_second_token_rejected)
- ERROR: test_skip_without_submitting_or_advancing_formal_cutoff (test_streaming_prespec_helpers.ModelHelperTests.test_skip_without_submitting_or_advancing_formal_cutoff)
- FAIL: test_default_policy_runs_twenty_connection_shadows_concurrently (test_streaming_prespec.PreSpecCoordinatorTests.test_default_policy_runs_twenty_connection_shadows_concurrently)
