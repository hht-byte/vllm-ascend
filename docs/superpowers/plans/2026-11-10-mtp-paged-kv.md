# Auxiliary paged KV read/write implementation plan

**Goal:** Eliminate packed MTP history K/V gathers and combine new K/V writes,
while preserving rejection cropping and non-invasive engine integration.

**Spec:** The user selected independent fixed blocks per auxiliary request slot,
direct FIA/PA cache reads, and lazy overwrite after rejection. No target block
allocator, target attention, scheduler or service restart is in scope.

**Architecture:** Add opt-in `kv_cache_backend="paged"` for token FULL graphs.
Keep `"tnd"` default. A hidden-only arena plus fixed paged KV provides coherent
strided history views to eager fallback; it never duplicates live KV. Each graph
uses independent fixed-size block-table buffers, with per-head compact segments.
FIA query remains TND; PA KV lengths are per sequence, not cumulative. The
existing positive dummy query is backed by separate scratch pages.

**Writer:** `kv_write_backend="auto"` uses ND ScatterPa when its schema supports
`cache_mode="Norm"`, otherwise one combined K/V ScatterNd update. Explicit
`scatter_pa` must fail early if unsupported; `scatter_nd` enables comparison.
Production block sizes are 128/256/512. No probing, host tensor inspection or
workspace allocation occurs during replay.

## Constraints and review

- Cache budget includes block rounding, all live slots and dummy pages.
- Slot release resets valid lengths, not device memory; stale data is never read.
- Graph/eager transitions see identical K/V through shared storage views.
- Padded writes use unique scratch addresses, never live cache addresses.
- Block-table segment count matches Q lengths, including inactive-head budgets.
- Each head's CPU lengths and current table view are rebound before replay;
  unchanged bindings still signal external events.
- Hidden gathers/writes and token scatter remain; no claim of removing all gathers.
- CPU parity is not device graph support or a measured latency improvement.
- Preserve existing dirty research files. Do not push or restart services.

## Steps

- [x] RED: cache allocation/budget/layout/slot reuse and eager shared views.
- [x] Implement fixed paged arena and combined/dedicated writer.
- [x] RED: PA argument/length/table contract, CPU attention reference and updates.
- [x] Implement PA backend using existing graph capture/task update lifecycle.
- [x] RED: multi-round packed/eager parity, mixed budgets/rejection, padding safety,
  and dispatch proving no historical KV gather in the device tensor program.
- [x] Integrate optional backend/configuration, preserving TND defaults.
- [x] Export fresh/cumulative/focused patches and mirrored device tests/docs.
- [x] Verify related suites, baseline failure identities, patch routes, lint,
  resource contract, and real NPU conditional tests; self-review the diff.

## Primary evidence

- Local engine `attention_v1.py` FIA full-graph path passes `block_table`,
  `block_size`, cumulative Q lengths and per-sequence KV lengths.
- Ascend op-plugin FIA and ScatterPa API documentation; ND ScatterPa needs the
  newer `cache_mode` argument, absent from the referenced 7.2.0 schema.

## Completed verification

- `test_asr_mtp_paged`: 13 real CPU tests passed, covering storage aliasing,
  page rounding/budget, combined and dedicated-write contracts, context-edge
  padding, mixed budgets, rejection, native fusion and worker graph/eager mix.
- Resource contract: three-head packed program performs three hidden reads and
  three joint KV writes; no program-side history K/V read/materialization.
- Added two real NPU graph cases to `ActualPackedNPUReplayTests`: automatic and
  forced combined writers, native fusion, budget/batch/order changes and crop.
  This host lacks NPU; these remain device acceptance, not passed hardware tests.
- `.tmp/mtp-test-env/mtp-paged-related.log`: 168 tests, 159 passed, nine NPU skips.
- `.tmp/mtp-test-env/mtp-paged-installed.log`: 92 tests, 83 passed, nine NPU skips.
- `.tmp/mtp-test-env/mtp-paged-full.log`: 334 tests, 19 errors and one failure;
  failure identities exactly match `mtp-native-full.log`. Old research issues
  remain, with no new failure identity.
- Four release routes apply, compile all nine modules, match source, re-export
  empty, and reverse-check. Seven mirrored test files have AST parity.
  Task-local verifier: `.tmp/mtp-test-env/verify_mtp_paged_release.py`.
- Ruff passes changed production/export/new tests and engine mirror tests;
  source format checks and both `git diff --check` pass. Existing engine-head
  test style diagnostics are outside this change, as in the previous baseline.
- Markdownlint inspected three documents with engine rules, zero issues.
- Self-review checked PA per-sequence lengths, table batch shapes, unique scratch
  addresses, no duplicate live KV, fallback coherence, capability-gated ND writes,
  cache budget underflow and no added per-head host/device synchronization.

Source remains on `codex/streaming-prespec-monitor`; engine deliverables remain
on `310P-graph`. No commit, push or service restart was performed. New focused
upgrade: `examples/910b/asr_mtp_paged_kv_v023.patch` for native-fusion source a5e.
Real 910B numerical acceptance, capture/replay and performance are unmeasured.
