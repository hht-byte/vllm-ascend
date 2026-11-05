# Prespec rollback verification -> MTP handoff

> Use executing-plans / test-driven-development, preserving prior unrelated local reference edits.

**Goal:** Allow the background prespec request to verify its legacy rollback suffix once and then accelerate remaining candidate decode with the existing worker MTP module.

**Design:** Shadow continues target-first sampling followed by C[1:] verification. Explicit ascend_mtp enables the worker for shadow; without a first-verify marker, a nonempty external draft must be returned once before any MTP proposal. Capture confirmed prefill hidden during that first step, reuse it after rollback validation, and immediately switch to MTP after partial/all acceptance. Draftless shadow can use MTP after the first sample. No scheduler, target model, extra target forward or new checkpoint load. Existing candidate budget, deadline, cumulative partial publishing and abort/finally remain in effect.

**Control:** Updated prespec helper uses model.prespec_mtp_enabled (default True); False explicitly opts shadow out. It clears inherited formal first-verify/rejoin metadata on the request clone and preserves actual legacy draft token IDs. Global engine asr_mtp remains required. Formal request MTP settings are independent.

**Tests / tasks:**

- [x] RED->GREEN worker legacy stage: old draft precedes MTP, partial/zero/all acceptance and corrected context, no-draft direct MTP, cleanup and capacity fallback.
- [x] RED->GREEN shadow request builder: default/opt-out markers, no inherited formal scheduling, unchanged C[1:], budgets and original SamplingParams; deadline cancellation retains partial candidate and aborts the engine request.
- [x] Refresh exported engine patch and docs, verify scoped suite and artifact freshness; independent review of phase/lifecycle semantics.

**Review focus:** Never skip or re-offer the old rollback draft, never first-verify C[1:], don't accidentally enable formal metadata on shadow, no extra target forward or per-head D2H, no leaked per-request handoff markers after abort/reused IDs. Device contention/performance still needs 910B A/B.

## Verification ledger

- Worker tests first reproduced immediate MTP (skipping legacy) and missing legacy on auxiliary fallback. Fixed two-stage proposal and kept submitted phase through fallback. Helper tests first reproduced inherited formal flags / MTP opt-out, then passed with request-clone cleanup and independent shadow switch.
- Final online-related suite: 46 tests pass. Real tiny Qwen3 target loop now includes shadow target-first/legacy verification with 0/partial/all initial acceptance and subsequent MTP, matching greedy output. Chunked prefill does not advance the legacy phase before a real sampled drive token.
- Async tests with MTP enabled preserve eligible verified partial candidates at formal cancellation and deadline timeout, and verify abort/aclose. The old coordinator condition that a partial must extend beyond the rollback range is preserved.
- Fresh-install and incremental-upgrade engine diffs both passed git apply --check, applied into local fixtures, and second export is empty. Module artifact freshness, compileall and git diff --check pass.
- Broader suite: 211 tests at the full-run checkpoint had the same pre-existing 1 assertion failure + 19 errors, no new failing IDs. The subsequently added chunked-prefill regression passes in the final 46-test scoped run.
- Independent review found no concrete issue. It also verified chunked prefill -> legacy -> zero accepted -> MTP -> finish -> reused ID. No additional target forward or target/scheduler source changes were introduced.
- Not deployed, committed or pushed in this task. Full 910B checkpoint/kernel execution, deadline readiness and competition with formal P95 remain unmeasured. Earlier unrelated local reference edits are preserved.
