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

## Service switch follow-up (2026-11-05)

- Found the reported gap: the shadow builder read prespec_mtp_enabled with a default, but there was no executable startup initializer or CLI registration in this repository. The real server factory / parser source is not present here.
- Added streaming_mtp_config.py: --mtp / --no-mtp master defaults OFF; --prespec-mtp / --no-prespec-mtp controls shadow under that master; checkpoint override is service-only. OFF removes asr_mtp before engine creation while preserving other settings. ON validates the existing external-draft backend and checkpoint, and sets compatible scheduling/cache defaults without overriding explicit incompatible values.
- Moved enable_prespec initialization into the copyable mixin. It saves formal and effective shadow settings even with prespec disabled, preserves an explicitly supplied logged producer, and logs the settings. The request builder also enforces master OFF over a stale shadow ON setting.
- Added complete server integration instructions (parser -> explicit AsyncLLM arguments -> engine config before construction -> model initialization -> both formal helper call sites). Removed the stale duplicate helper code block from the general guide to avoid copy/paste drift.
- RED reproduced missing startup methods and a stale-shadow override; final scoped service / engine / launch suite passes 63 tests. compileall and diff checks pass. Independent review and a separate check of the final defensive master override found no concrete issue.
- Server source is not present; no server changes, deployment or service restart performed. User subsequently requested a scoped commit/push of the service-switch changes; earlier independent reference-script edits remain local.

## Startup circular-import correction (2026-11-05)

- User's real stack showed model_runner_v1 importing patch.worker before NPUModelRunner was defined; the old registry then imported the MTP module, which eagerly read that incomplete class. Earlier hook tests supplied an already-defined mock runner and missed this startup order.
- Added a temporary package imported through real importlib machinery. Before the fix it reproduced the exact partially initialized NPUModelRunner ImportError. After the fix it checks registration on 0.23, skip on other versions, and enabled / disabled drafter selection.
- Moved only the registration point: remove old eager MTP import from patch/worker/__init__.py, add the version-gated MTP import to worker/worker.py after its original runner import. Runner methods, decoding logic and first-verify transport are unchanged. Exporter handles both fresh installation and upgrade, and validates the worker anchor/version import.
- Refreshed the release patch and deployment instructions. Fresh and upgrade fixtures pass git apply --check and installation, with empty second exports. Blank unified-diff context lines are normalized to avoid trailing-space warnings in the committed patch artifact.
- Related CPU regression suite passes 65 tests; independent review found no concrete issue and confirmed the XliteWorker -> NPUWorker -> runner registration order. Real server startup/NPU validation is still pending; no server changes, restart or push performed in this correction.
