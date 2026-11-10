# Native auxiliary fusion and graph handoff implementation plan

**Goal:** Replace unfused MTP auxiliary operations with native NPU kernels and packed projections, and reduce avoidable work between MTP replay and the next target step.

**Approved scope:** The user requested native fused auxiliary heads and graph-mode optimization for the observed handoff bubble. Keep target weights, scheduling, head count, token buckets and request-slot limits unchanged. Do not push or restart services in this implementation turn.

## Design

- Strict-load the original checkpoint, place/freeze the auxiliary model, then prepare packed QKV and gate/up weights once before profiling/capture.
- Preserve original checkpoint keys and use shared-storage parameter views into nonpersistent packed buffers rather than duplicate permanent weights.
- Replace auxiliary RMSNorm locally; use native RMSNorm, residual Add+RMSNorm and SwiGLU on NPU, with CPU math references and `native_fusion=False` for controlled comparison.
- Share projection/residual helpers across eager, rectangular and packed token paths. Do not patch target modules or global PyTorch operators.
- If all eligible requests form one complete group, consume the graph output directly and skip the intermediate zero tensor and advanced-index aggregation. Multi-group execution still copies outputs before a reused graph can overwrite them and performs one final D2H.
- Reuse unchanged FIA task bindings. When cumulative Q/KV lengths are unchanged, skip rebinding but still record the external event required by the captured wait/reset. Changed lengths follow the existing update path.
- `profile_stages=False` adds no clocks or device synchronization. Optional host timing separates preparation/enqueue, FIA updates and D2H waiting; these values are not device execution timings.

## Tasks

- [x] Add RED tests for packed weights, native operation dispatch, checkpoint compatibility and shared storage.
- [x] Implement the native helper module and integrate all three MTP execution paths after strict loading.
- [x] Add RED tests for single-group output handoff, safe multi-group reuse and timing-off behavior.
- [x] Implement handoff fast path, repeated-length FIA binding reuse and opt-in stage timings.
- [x] Update export dependencies, fresh/cumulative/focused engine patches, mirrored tests and installation/configuration docs.
- [x] Verify CPU multi-round parity, patch application/source parity/idempotence, related/full-suite baseline, lint and independent review.

## Review focus

- QKV bias and GQA output sizes; gate/up ordering; FP16/BF16 numerical differences.
- Original state dictionary names and strict loading; idempotent preparation and absence of duplicated permanent weight storage.
- Correct residual rounding/dataflow and cache contents in all execution modes.
- Persistent graph outputs must be consumed before another replay overwrites them.
- Skipping FIA parameter updates must retain external event recording and graph execution on changed data.
- Default-off timing must not call clocks, create NPU events, transfer device metadata or synchronize streams.
- Native fusion and graph handoff do not establish that host/scheduler idle gaps are resolved; real 910B traces and commit latency are still required.

## Baseline

- Source HEAD `46d2537e1`; engine delivery HEAD `efd6f94ee`.
- Preserve existing unrelated research changes and reuse the existing isolated engine delivery checkout.

## User-requested pause: 2026-11-10 16:38 Asia/Shanghai

Resume once in three hours, then remove/disable the one-shot automation if necessary.
Do not push, restart services, or overwrite unrelated dirty research files.

Implemented but uncommitted:

- New `research/asr_mtp_fused.py` and `test_asr_mtp_fused.py`: four helpers,
  startup-only packed QKV/gate-up shared storage preserving original checkpoint
  keys, native RMSNorm/AddRMSNorm/SwiGLU, CPU reference and native API tests.
- All three paths (`asr_mtp_heads.py`, `asr_mtp_graph.py`,
  `asr_mtp_token_graph.py`) use helpers. Worker strict-loads, places/freezes,
  then prepares; `native_fusion=True` default, explicit False restores math.
- `GraphOptions.profile_stages=False` and worker config validated and passed;
  no clocks/sync when off. New host logs distinguish task update/enqueue and
  D2H wait (not pure transfer or independent device duration).
- Single complete group consumes persistent output directly; multiple groups
  copy before another replay overwrites output, then one D2H.
- FIA binding signature lengths+Q/K/V pointers+scale skips unchanged updates,
  but always records external events. Strict physical TND KV coverage preserved.
- Exporter includes new module (seven total). `test_build_asr_mtp_patch.py`
  expectations updated, but RELEASE PATCHES NOT REGENERATED YET, so release
  parity test will currently fail until next steps.
- `test_asr_mtp_fused_paths.py` proves eager/BQ/token matmuls 27 to 18 for
  three heads and rejection-cache parity. Engine load test covers switches.
- Agent added native-fused variants in ActualNPUReplayTests and
  ActualPackedNPUReplayTests; original unfused cases retained.
- `asr_mtp_engine.md` only partly updated (module list and task-reuse sentence).
  Still add native-fusion/diagnostics/configuration/install details there and in
  engine examples/910b/asr_mtp_graph.md. Existing 3-argument service factory stays.

Evidence so far:

- `.tmp/mtp-test-env/mtp-native-root-green.log`: 98 tests, 93 passed,
  five NPU skips, before the last two native-fused NPU tests were added.
- Independent reviewer `/root/mtp_fusion_review`: no blocking issue;
  270 FP32/FP16/BF16 multi-request rejection rounds, identical draft tokens,
  max FP32 KV difference 9.54e-7. No local NPU acceptance/performance evidence.
- Ruff check passed changed production files; new fused-path test import fixed;
  graph/token/attention/new tests formatted. Avoid whole-file reformat of old
  heads/worker/test_engine style. Agent formatted NPU test additions.

Remaining concrete steps:

1. Review current diff; finish docs, config examples and profiling commands.
2. Generate fresh root `research/asr_mtp_v023.patch`, cumulative engine
   `examples/910b/asr_mtp_graph_v023.patch`, row-cache-to-token upgrade
   `asr_mtp_token_graph_v023.patch`, and NEW focused
   `asr_mtp_native_fusion_v023.patch` against installed strict-TND HEAD46d.
   Existing historical focused patches must not be stacked with newer full ones.
3. Task-owned prepared fixtures (do not alter older fixture dirs):
   `.tmp/mtp-test-env/engine-mtp-native-fresh`, `engine-mtp-native-upgrade`,
   `engine-mtp-native-token-upgrade`, `engine-mtp-native-focused`.
   First is pre-MTP first-verify; second pre-graph b63; third row-cache c5b;
   fourth current installed46d. Exporter reads fixture and creates diff only.
   Verify apply --check/apply/compile/source parity/empty re-export/reverse check.
4. Engine delivery checkout remains
   `.tmp/mtp-test-env/engine-310P-graph-mixed-full`, branch310P-graph at efd6f94ee.
   It packages patches/docs/tests, not installed source modules. Mirror tests
   under examples/910b/tests, format with engine config, compare AST to source.
5. Python `.tmp/mtp-test-env/Scripts/python.exe`; run related suites including
   test_asr_mtp_fused, test_asr_mtp_fused_paths, test_asr_mtp_runtime,
   graph/token_attention/token_graph/engine/patch/build and service suites.
   Full research known baseline 19 errors+1failure in unchanged HEAD; compare
   failure identities with mtp-tnd-full-recheck.log, not raw count. One 10ms
   monitor SummaryAlignment test can flake; unchanged failure recheck is needed.
   Async/full suites may hang in sandbox; use require_escalated and log to file.
6. Test patch routes, installed tests, Ruff, markdownlint, git diff --check;
   update plan/evidence. Finish with concise outcome and explicit NPU gap.

Engine git needs command-scoped safe.directory and sometimes escalation;
never change global safe.directory. No commits/push needed until asked.
Source test Python has transformers4.57.6 and `.tmp/asr-mtp-reference`.
Ruff C:/Python314/Scripts/ruff.exe; cached markdownlint D:/nodejs/npx.cmd.

## Completion after scheduled resume: 2026-11-10

The one-shot `mtp` automation was deleted when work resumed; no further run is
scheduled. No commit, push, service restart or unrelated cleanup was performed.
The pause notes above record the earlier checkpoint, not outstanding work.

- Both configuration/installation documents are complete. Existing factory
  arguments stay unchanged. Fusion defaults on for MTP auxiliary heads;
  diagnostic host timing defaults off and adds no device synchronization.
- All four release routes were generated and applied to their prepared bases.
  Each passed apply checking, seven-module compilation and source parity,
  empty re-export, and reverse-apply checking. Evidence helper:
  `.tmp/mtp-test-env/verify_mtp_native_release.py`.
- Six engine-side test files mirror source ASTs, with engine formatting.
- `.tmp/mtp-test-env/mtp-native-related.log`: 153 tests, 146 passed,
  seven real NPU tests skipped on this CPU host.
- `.tmp/mtp-test-env/mtp-native-installed.log`: 77 tests against installed
  modules, 70 passed, the same seven device tests skipped.
- `.tmp/mtp-test-env/mtp-native-full.log`: 319 tests; 19 errors and one failure
  have exactly the same identities as `mtp-tnd-full-recheck.log`. No new
  full-suite failure identity. These old research issues remain unresolved.
- Ruff passes changed production files, exporter and new/graph tests;
  formatted source and all six installed tests pass their relevant checks.
  `test_asr_mtp_engine.py` retains five pre-existing I001/RUF059 diagnostics,
  confirmed against HEAD; no unrelated style rewrite was performed.
- Markdownlint actually inspected both edited documents with the engine rules:
  two files, zero issues. Both repositories pass `git diff --check`.
- Independent final review found no blocking issue, independently verified
  mirrored ASTs and reverse application of all four routes, and ran 25 new
  tests (23 passed, two NPU skips). Earlier random parity covered 270
  FP32/FP16/BF16 multi-request rejection rounds with identical draft tokens.

Real 910B native arithmetic, full BQ/token FIA capture/replay, acceptance and
commit latency still need device validation. Reduced matrix call count and
host work do not establish a measured speedup or removal of scheduler gaps.

## Authorized push follow-up

The user subsequently requested pushing this change. Refreshing both origins
confirmed each local branch matched its remote before committing. The related
suite was rerun (153 tests: 146 passed, seven NPU skips), and independent
installed-module verification reran 77 tests (70 passed, seven NPU skips).
Source changes belong on `codex/streaming-prespec-monitor`; the nine engine
delivery files belong on `310P-graph`. Other dirty research files are excluded.
