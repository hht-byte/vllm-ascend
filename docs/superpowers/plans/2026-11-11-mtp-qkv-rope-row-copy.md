# Auxiliary QKV/RoPE and sparse row-write optimization

**Goal:** Fuse auxiliary rotary/normalization work and remove generic ScatterND
from graph arena writes without changing target decoding or cache ownership.

**User request:** Continue existing MTP optimization, use NPU-friendly RoPE and
split-QKV RMSNorm, and address a 457us ScatterNdUpdate reported by profiling.

## Evidence and selected approach

The supplied configuration confirms heads=5, context=512, slots=22, paged KV
with ScatterPa, and largest token bucket1024. Destination rows match exactly:
`22*512*6*2048/128 + (1024+1)*2048/128 = 1097744`. Update rows2320 correspond
to145 token positions times16 hidden vector rows. This is the remaining hidden
arena writer, with roughly268MiB destination and580KiB updates, not the PA KV
writer. These dimensions identify the path; they do not establish the device
implementation's internal memory traffic.

- Keep the flat arena and unique live/scratch addresses. A sparse Triton writer
  directly loads row IDs/updates and stores only updated rows, never copying or
  scanning the destination. Retain ScatterND and CPU index_copy controls.
- Precompute ordinary half-RoPE coefficients once, indexed by clamped position
  plus head depth. Padding can exceed512 due to1024-token buckets, so its
  anchors must be clamped without changing valid-token positions.
- Reuse native `npu_rotary_mul` for RoPE-only. Explicit joint fusion reuses the
  installed `triton_split_qkv_rmsnorm_mrope` operator with identical three axes.
  Do not use the older split-QKV kernel with hard-coded BF16 on FP16 service.
- Packed projection includes its bias exactly once. Joint RMSNorm affine bias
  stays None. Preserve original model state dictionary and projection storage.

## Configuration and boundaries

- `qkv_rope_backend`: auto / torch / native / triton. Auto selects native
  RoPE-only when available; explicit triton enables joint split+norm+RoPE,
  whose FP32 intermediates can change low-precision rounding.
- `row_write_backend`: auto / triton / scatter_nd. Auto selects Triton on NPU
  if importable, otherwise original ScatterND. Explicit unavailable selection
  fails during startup. CPU uses reference index_copy. Runtime failures must
  not be swallowed. No environment variables or target runner rewrites.
- `native_fusion=False` retains original QKV/RoPE path for auto/torch; reject
  explicit native/triton with the master auxiliary fusion disabled.
- Applies to eager, rectangular and packed graphs as appropriate. Row writes
  concern auxiliary flat-arena updates; dedicated paged KV writer is unchanged.
- Keep existing three-argument service factory, startup allocation/profiling,
  one batch D2H, unique padding sinks and lazy validity cropping.
- No push/restart. Preserve unrelated dirty research files.

## Tasks

- [x] RED and implement sparse row writer, CPU work-count reference and NPU test.
- [x] RED and implement coefficient cache/native/joint QKV-RoPE helper.
- [x] Integrate helpers into three paths and startup configuration with tests.
- [x] Refresh fresh/cumulative/token/focused upgrades and mirrored tests/docs.
- [x] Verify parity, padding/caps, graph/eager lifecycle, patch application,
  lint, installed tests and existing full-suite failure identities.
- [x] Independent review. Document real NPU capture/numerical/performance gap.

## Verification evidence

- RED: startup test originally lacked prepared QKV helper and row option;
  three path tests reported zero helper calls; native coefficient contiguity
  test failed before changing the table to `[2, positions, D]`.
- Related tests: 183 run, 169 passed and 14 NPU skips, plus eight passing export
  tests. Logs: `.tmp/mtp-test-env/mtp-native-ops-related.log` and
  `.tmp/mtp-test-env/mtp-native-ops-export.log`.
- Installed modules: 115 run, 101 passed and 14 NPU skips. Log:
  `.tmp/mtp-test-env/mtp-native-ops-installed.log`.
- Full research: 357 run, 19 errors and one failure; identities match the
  previous paged baseline exactly. Logs:
  `.tmp/mtp-test-env/mtp-native-ops-full.log` and `mtp-paged-full.log`.
- Four release routes apply/check, compile and exactly match all 12 source
  modules; re-export is empty and reverse-check succeeds. Nine mirrored test
  files have AST parity after engine formatting. Task-local verifier:
  `.tmp/mtp-test-env/verify_mtp_native_ops_release.py`.
- Independent read-only review found no blocking issue. The new helper tests
  were independently run: 17 passed and three NPU skips.
- No local NPU: Ascend Triton compilation, true capture/replay, numerical
  tolerance, output/acceptance and latency remain device acceptance work.
  Native fused RoPE and joint FP32 norm/rotation need not be bitwise identical
  to the old separate low-precision operations. Target verification is retained.

## Interpretation boundaries

The original in-place ScatterNd op-plugin adapter has no explicit full tensor
clone; the shape alone does not establish full-arena traffic or workspace size.
`GRAPH_READY reserved_mib` also counts persistent FIA workspace and allocator
reservation, not just graph-private temporary memory. Compare the sparse writer
against `scatter_nd` with the same graph buckets before assigning memory savings.

No service restart, commit or push was performed in this implementation turn.
