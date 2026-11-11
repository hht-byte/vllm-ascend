# Online ASR auxiliary MTP graphs

This optional upgrade captures the sequential MTP blocks, shared LM head,
argmax and auxiliary cache writes. It does not change target forward, target
verification, audio encoding or scheduling. Target decoder FULL graphs and
auxiliary MTP graphs have separate controls and statistics.

Prerequisite: the existing online ASR MTP / external-draft / first-verify
extension is already installed (the MTP source corresponding to
`codex/streaming-prespec-monitor` at `b63fbdaaf`). The engine graph deliverable
is kept on `310P-graph`. The patch is explicit opt-in; pulling this branch alone
does not overwrite a server's locally installed ASR extension.

This auxiliary graph implementation targets the 910B `NPUWorker` memory path.
`310P-graph` is the delivery branch name, not a claim of 310P auxiliary-graph
support. `NPUWorker310` overrides memory profiling and is explicitly rejected for
auxiliary `graph_mode=full`; its existing eager MTP path is unchanged.

## Install

```bash
git apply --check examples/910b/asr_mtp_graph_v023.patch
git apply examples/910b/asr_mtp_graph_v023.patch
```

If the check fails because the server's MTP modules differ, regenerate against
the actual installed checkout using the updated ASR source's
`research/build_asr_mtp_patch.py`. Do not force an incompatible diff. The exporter
checks the v0.23 runner APIs and adds a late worker memory registration after
the worker class is defined, avoiding the old circular import.

## Enable and compare

Keep the existing checkpoint, head count, service MTP flags, Prespec settings,
target graph mode and load identical. Set in the existing engine dictionary:

```python
base_config["additional_config"]["asr_mtp"]["graph_mode"] = "full"  # "off" for baseline
```

Optional capacities (defaults shown):

```python
base_config["additional_config"]["asr_mtp"].update({
    "graph_max_batch_size": 8,
    "graph_query_sizes": [8, 64],
    "graph_context_tokens": 128,
})
```

Exact batch sizes 1..8 avoid padding the expensive LM head. Two query buckets
keep incremental head work separate from the initial hidden history. True
lengths, cache prefixes, slots and remaining per-request budgets are device
inputs. This is normally 16 pre-captured graphs, bounded by worker/cache slots;
at most 64 entries are permitted. Uncovered batch/query/context shapes and a
shortened maximum head budget use the original eager path. There is no online
capture or deliberate request batching wait. Increasing capacities adds startup
time, scratch/static memory, graph-pool memory and possibly padded computation.

Keep the existing configuration function and factory interface unchanged. Set
the dictionary fields above before calling the original function:

```python
base_config = mtp_engine_config(
    base_config,
    mtp_enabled=mtp_enabled,
    mtp_checkpoint=mtp_checkpoint,
)
```

This existing function preserves the graph settings while applying the master
MTP switch and checkpoint override. No new graph argument is needed in the
factory or this call. If using the optional `--mtp-graph-mode full/off` parser
flag, map its value to `base_config["additional_config"]["asr_mtp"]["graph_mode"]`
at the entry point first. Setting only the parser flag does not enable graphs.

Graphs are prepared during memory profiling before eager fallback profiling and
KV sizing. A startup-only memory hook conservatively subtracts the retained
graph-pool/static-buffer reserved-memory delta from the automatic KV budget.
Explicit `kv_cache_memory_bytes` is left unchanged; reserve auxiliary graph
space yourself in that manual budget. Graph memory is additional to `cache_mib`.
Capture errors fail startup; disable the auxiliary graph option to recover.

Check these INFO logs after the normal service restart:

```bash
rg -n 'ASR_MTP_CONFIG|ASR_MTP_GRAPH_READY|ASR_MTP_GRAPH_MEMORY|ASR_MTP_GRAPH_REPLAY' "$LOG"
```

`ASR_MTP_GRAPH_READY` reports captured entries and retained MiB;
`ASR_MTP_GRAPH_REPLAY` appears at the first actual replay. DEBUG
`ASR_MTP_STEP` reports cumulative graph/eager batches and shape/budget misses.
Auxiliary replays are not included in the target decoder `CUDAGraph Stats`.

## Collect context miss lengths

For a service that already installed the auxiliary graph version, the focused
diagnostic upgrade changes only `vllm_ascend/spec_decode/asr_mtp_graph.py`:

```bash
git apply --check examples/910b/asr_mtp_context_diagnostics_v023.patch
git apply examples/910b/asr_mtp_context_diagnostics_v023.patch
```

The cumulative `asr_mtp_graph_v023.patch` above already includes this diagnostic;
do not apply both patches consecutively. If a patch check fails, regenerate
against the actual installed source using the exporter. Restart workers normally
after updating; no new service parameter is needed.

Each context fallback emits an INFO record:

```text
ASR_MTP_CONTEXT_MISS batch_size=2 max_context=180 context_limit=128 max_fresh_query=79 query_limit=128 miss_count=57
```

`max_context` is the largest hidden-history length in this auxiliary batch,
including prompt positions. `max_fresh_query` is the largest newly required span
over active heads; it may come from a different row. `batch_size` counts requests
in the batch, and `miss_count` matches this worker's cumulative context-miss
counter. The limits report the current graph capacities.

```bash
rg -n 'ASR_MTP_GRAPH_READY|ASR_MTP_CONTEXT_MISS' "$LOG"
```

Only existing CPU integer metadata is read, without tensor-to-host copies or
device synchronization. No graph selection, cache state or fallback precedence
changes. If INFO is disabled, the extra diagnostic query scan is skipped.

## Row cache transfer upgrade

For an installed auxiliary graph implementation that already includes the
context diagnostics, apply only this focused upgrade:

```bash
git apply --check examples/910b/asr_mtp_cache_rows_v023.patch
git apply examples/910b/asr_mtp_cache_rows_v023.patch
```

The cumulative graph patch already includes this upgrade. Do not apply both
consecutively, or apply the older diagnostics-only patch afterward. For a
different installed version, regenerate against the actual engine with the
exporter. Worker restart is required; no new service flag is introduced.

The graph program preserves the arena layout, graph keys and KV semantics. It
addresses rows of `gcd(hidden_size, head_dim)` elements instead of scalar
addresses. Reads use 2D row `index_select`; NPU writes explicitly use
`torch_npu.npu_scatter_nd_update_`, avoiding the legacy per-index host fallback
of generic `index_copy_`. An unavailable device update API fails graph
initialization. CPU parity tests use row `index_copy_`.

The configured final head no longer stores its unused hidden output or updates
the local token sequence. Its KV and valid lengths remain available for later
rounds. A five-head graph has 14 row cache updates and 4 token scatters, replacing
15 scalar cache scatters and 5 token scatters. The 15 cache reads keep their
count but use compact row indices. KV update/attention-input fusion and the
eager fallback implementation are unchanged.

Keep graph parameters and load fixed when comparing this upgrade. Profile
`aclnnScatterNdUpdate` and the row gathers, and check complete target outputs,
acceptance, commit P95/P99 and memory. Reduced index counts are not a measured
latency gain. The device replay test below must run without being skipped.

## Packed token graphs

To upgrade an installed row-cache graph implementation, apply only:

```bash
git apply --check examples/910b/asr_mtp_token_graph_v023.patch
git apply examples/910b/asr_mtp_token_graph_v023.patch
```

The cumulative graph patch already includes this upgrade. Other installed
versions should use the exporter against their actual checkout. Restart workers
normally after applying; pulling this branch alone does not install the modules.

Set `graph_token_sizes` in the existing settings before calling the unchanged
three-argument `mtp_engine_config` service factory:

```python
additional = dict(base_config.get("additional_config") or {})
mtp_settings = dict(additional.get("asr_mtp") or {})
mtp_settings.update({
    "graph_mode": "full",
    "graph_token_sizes": [8, 32, 64, 128, 256, 512],
    "graph_max_batch_size": 8,
    "graph_context_tokens": 128,
})
additional["asr_mtp"] = mtp_settings
base_config["additional_config"] = additional
base_config = mtp_engine_config(
    base_config,
    mtp_enabled=mtp_enabled,
    mtp_checkpoint=mtp_checkpoint,
)
```

Absent `graph_token_sizes`, the rectangular `(B,Q)` path remains available. When
both settings are present, token buckets take precedence. Each token bucket
captures one graph with fixed request and history capacities. For active heads,
let `q[i,d] = length[i] - valid[i,d]`; inactive heads contribute zero. Select the
smallest bucket covering `max_d sum_i q[i,d]`, not a sum over all heads. A
6/80/45-fresh-token batch therefore fits one 256-token graph.

Requests from the same target step are packed together regardless of per-request
Q. Only the configured request/token limits split a packed batch; context and
single-request token overflows retain eager fallbacks. Token capacity is
independent of per-request context capacity and of the target FULL graph size.

Packed projections and MLP use `[T,H]`. One segmented FIA TND attention per head
isolates requests using staged cumulative Q/KV lengths, with graph task updates
before replay. Fixed request-slot logits gather only each request's last hidden
row; unused request slots still incur LM-head computation. Every logical token
bucket reserves one extra physical query row. A positive-query dummy segment
owns all remaining zero-filled KV rows, with dummy KV length at least dummy Q
length. Arena layout, row updates, rejection cropping and target verification
are unchanged.

The token NPU backend requires FP16/BF16, supported FIA head dimensions
(currently 128/192), FIA out/max-workspace APIs and graph task updates. Workspace
allocation is startup-only. Unsupported geometry may continue using the old
rectangular graph or eager path. Compare acceptance, complete target outputs,
step latency, commit P95/P99, queueing and reserved memory under equal load.
Device capture/replay and performance have not been established on this host.
Only heads whose cumulative lengths, Q/K/V pointers or scale changed need a
host task update. Unchanged bindings are reused, while every replay still
records all external events required by the captured wait/reset. Rebinding
cost and fixed request-slot logits must be included in the performance comparison.

## Strict non-PA TND KV coverage fix

Some deployed CANN versions require the final cumulative KV length to equal
physical K/V T exactly. The original token implementation left an unused KV
tail and can fail startup with `keyT/valueT must be equal to ...
actualSeqenceLengthKV`. For an installed initial token version, apply only:

```bash
git apply --check examples/910b/asr_mtp_tnd_kv_coverage_v023.patch
git apply examples/910b/asr_mtp_tnd_kv_coverage_v023.patch
```

The fresh, cumulative graph and row-cache-to-token patches already contain this
fix. Do not stack them with the focused fix. Other source versions should use
the exporter against the actual checkout. Restart workers normally; service
configuration is unchanged.

A logical bucket of 512 still accommodates 512 real fresh positions; its
physical query capacity is 513. The always-positive dummy query segment absorbs
the entire remaining physical KV capacity, including at the exact logical
bucket boundary. This avoids zero-query segments and runtime tensor shape
changes. The backend checks exact cumulative Q/KV coverage before FIA.
`ASR_MTP_GRAPH_READY` logs logical `token_sizes` and physical
`query_capacities`; the first replay also logs `bucket`, `query_capacity` and
`kv_capacity`. The extra row's performance impact still needs device measurement.

## Native auxiliary fusion and graph handoff

For the installed strict-TND version (ASR source `46d2537e1`), apply only:

```bash
git apply --check examples/910b/asr_mtp_native_fusion_v023.patch
git apply examples/910b/asr_mtp_native_fusion_v023.patch
```

The fresh ASR `research/asr_mtp_v023.patch`, cumulative graph patch and
row-cache-to-token patch already include this upgrade. Do not stack them with
the focused patch. Other installed versions should regenerate against their
actual source. Updating the files requires the normal worker restart; this
document does not restart services.

The new default `native_fusion=True` uses native RMSNorm, residual Add+RMSNorm
and SwiGLU only inside auxiliary heads. QKV and gate/up projections are packed
once after strict checkpoint loading, final device/dtype placement and freezing,
before profiling/capture. Original Parameter names remain in the checkpoint;
nonpersistent packed buffers share their storage rather than keeping a second
resident weight copy. Eager, rectangular and token paths use the same helpers.
No target model or global Torch operator is patched. Missing native APIs fail
startup; `native_fusion=False` restores the original auxiliary math.

Merge the startup settings before the unchanged three-argument service factory:

```python
additional = dict(base_config.get("additional_config") or {})
mtp_settings = dict(additional.get("asr_mtp") or {})
mtp_settings.update({
    "native_fusion": True,    # default; False for a fusion-only comparison
    "profile_stages": False,  # opt-in host diagnostics; leave off for timings
})
additional["asr_mtp"] = mtp_settings
base_config["additional_config"] = additional
base_config = mtp_engine_config(
    base_config,
    mtp_enabled=mtp_enabled,
    mtp_checkpoint=mtp_checkpoint,
)
```

This also works when the base dictionary has no `asr_mtp`; an enabled master
switch still requires a checkpoint. The factory preserves existing checkpoint
and graph fields. These switches are not hot reconfiguration of a live engine.
Five heads now use 30 linear matrix calls instead of 45. This verified call-count
reduction is not a measured device latency gain. Sequential LM heads and unused
fixed request-slot logits remain part of the cost.

A single complete proposal group consumes the persistent graph output directly,
removing the intermediate zeros tensor and advanced-index aggregation. Multiple
groups still copy each result before another replay can overwrite its persistent
buffer, then perform one final `tolist()` boundary. FIA bindings are reused as
described above without dropping any event notification. No additional target
forward, deliberate batching wait or per-head device synchronization is added.

With `profile_stages=True`, INFO `ASR_MTP_HOST_STAGE` separates worker prepare,
proposal, CPU-transfer wait and postprocessing. Packed graphs additionally emit
`ASR_MTP_GRAPH_HOST_STAGE` with real requests/tokens/bucket, staging, task-update,
replay-enqueue and commit host durations, plus `task_updates` and
`task_update_reuses`. Eager and rectangular graphs retain worker-level timings.
Default-off diagnostics read no clocks, create no profiling events and add no
synchronization.

```bash
rg -n 'ASR_MTP_CONFIG|ASR_MTP_HOST_STAGE|ASR_MTP_GRAPH_HOST_STAGE' "$LOG"
```

These are host wall-clock intervals. `to_cpu_wait_ms` includes preceding queued
device work, not just transfer, and must not be added again to device execution
time. Graph timings are nested inside the worker proposal interval. Logging also
has overhead, so disable diagnostics for the final comparison. Input preparation,
IPC and scheduling gaps after the MTP handoff remain outside these intervals;
this upgrade does not establish that every observed bubble is removed.

## Optional auxiliary paged KV

For an installed native-fusion version (source `a5e591b9c`), apply only:

```bash
git apply --check examples/910b/asr_mtp_paged_kv_v023.patch
git apply examples/910b/asr_mtp_paged_kv_v023.patch
```

Fresh, cumulative graph and row-cache-to-token upgrades already include it.
Do not stack upgrades; regenerate against other installed versions. This patch
does not restart workers or modify target attention/scheduling.

Merge into the existing settings before the unchanged three-argument factory:

```python
additional = dict(base_config.get("additional_config") or {})
settings = dict(additional.get("asr_mtp") or {})
settings.update({
    "graph_mode": "full",
    "graph_token_sizes": [8, 32, 64, 128, 256, 512],
    "kv_cache_backend": "paged",  # default tnd preserves the old path
    "kv_block_size": 128,          # 128, 256 or 512
    "kv_write_backend": "auto",   # auto, scatter_pa or scatter_nd
})
additional["asr_mtp"] = settings
base_config["additional_config"] = additional
base_config = mtp_engine_config(
    base_config, mtp_enabled=mtp_enabled, mtp_checkpoint=mtp_checkpoint,
)
```

Each auxiliary slot owns fixed pages, independent of the target block allocator.
Hidden and paged KV have separate storage; eager fallback uses shared strided
views of the same KV. Rejection still crops valid prefixes, with lazy tail
overwrite. No full-history copy is introduced when switching execution paths.
Padded writes use independent dummy pages and unique addresses.

The graph combines new K/V writes once per head and gives persistent pages
directly to FIA through a block table. For five heads, explicit history K/V
gathers fall from ten to zero and K/V write calls from ten to five. Hidden
reads/writes, token updates and final-hidden extraction remain. These are
program call counts, not measured kernel counts or latency speedups.

`auto` checks ND `cache_mode="Norm"` support at startup. If available, it uses
ScatterPa; older NZ-only APIs use one joint K/V ScatterNd update instead.
Explicit `scatter_pa` fails early if unsupported; `scatter_nd` enables write
ablation under the same paged layout. Both graph write paths avoid historical
KV gathering; eager fallback retains its original dense attention preparation.
FIA receives cumulative TND Q lengths and per-sequence PA KV lengths. All head
tables upload together; identical tables skip upload. Length changes retain
task updates/event notifications. No device metadata is read on the host.

The cache budget includes rounded live pages, hidden storage and dummy pages;
weights, graph metadata and workspaces remain additional. Slot capacity may
decrease at equal MiB. Check actual `cache_slots`, `kv_cache_backend` and
`kv_write_backend` in `ASR_MTP_CONFIG`. Capture remains token-bucket based.

The installed suite adds `test_asr_mtp_paged.py` and real NPU paged cases in
`ActualPackedNPUReplayTests`, covering request counts/order/budgets, rejection,
native fusion and both write selections. CPU checks do not establish 910B FIA
capture/replay or speed. Compare complete target outputs, acceptance, commit
P95/P99, prefill/decode, host gaps and memory at equal load. PA may lose on very
short contexts. Hold all other MTP/graph settings fixed during comparison.

## QKV/RoPE and sparse hidden-row writes

For the previous native-ops version (ASR source `637b6f878`, engine deliverable
`5d757257c`), apply only the native-row correction:

```bash
git apply --check examples/910b/asr_mtp_native_rows_v023.patch
git apply examples/910b/asr_mtp_native_rows_v023.patch
```

For an earlier paged version corresponding to source `f9bf28b42`, use the updated
native-ops upgrade, which already includes that correction:

```bash
git apply --check examples/910b/asr_mtp_native_ops_v023.patch
git apply examples/910b/asr_mtp_native_ops_v023.patch
```

Choose one focused upgrade for the installed baseline; do not stack them. Fresh
ASR installation, cumulative graph and row-cache-to-token upgrades already
include this correction. Do not stack them with a focused patch. Regenerate against
other installed versions. The deployable module count increases from nine to
twelve: include `asr_mtp_qkv_rope.py`, `asr_mtp_row_copy.py` and
`asr_mtp_row_copy_kernel.py` under `vllm_ascend/spec_decode/`.

The three-argument service factory remains unchanged. Merge settings before it:

```python
additional = dict(base_config.get("additional_config") or {})
settings = dict(additional.get("asr_mtp") or {})
settings.update({
    "qkv_rope_backend": "auto",   # auto / torch / native / triton
    "row_write_backend": "scatter_nd",  # recommended here; auto also uses ScatterND
})
additional["asr_mtp"] = settings
base_config["additional_config"] = additional
base_config = mtp_engine_config(
    base_config, mtp_enabled=mtp_enabled, mtp_checkpoint=mtp_checkpoint,
)
```

| QKV/RoPE selection | Execution and comparison boundary |
| --- | --- |
| `auto` (default) | Native half-style RoPE when `npu_rotary_mul` is available, otherwise Torch. Separate Q/K RMSNorm remains; joint fusion is never selected automatically. |
| `torch` | Original separate Q/K norm and multiply/add rotation rounding order. |
| `native` | `npu_rotary_mul(rotary_mode="half")` after Q/K norm; unavailable API fails startup. |
| `triton` | Explicit joint split QKV, Q/K RMSNorm and RoPE through the registered `triton_split_qkv_rmsnorm_mrope`; unsupported API/geometry fails startup. |

The joint kernel retains FP32 normalization/rotation until its final cast, so it
may change low-precision rounding. Validate complete target output and acceptance
on the NPU before adopting it. The older split-QKV kernel with hard-coded BF16
output is unsuitable for this FP16 service. Packed Linear applies projection
bias once; the joint kernel's post-norm Q/K affine biases remain `None`.

All three MTP paths (eager, rectangular graph and packed token graph) share the
prepared helper. Position tables are budgeted and computed at startup on the final
device/dtype, with capacity `context_cap + number_of_heads + 1`. Runtime performs
table selection without trigonometry, device scalar reads or CPU length syncs.
Clamp padding anchors to the arena context cap before adding head depth: the
1024-token bucket may exceed the 512-token history cap. With
`native_fusion=False`, `auto/torch` keep the original math without installing the
helper; explicit `native/triton` is rejected.

The row option accepts `auto/triton/scatter_nd` and affects flat-arena `_write`,
chiefly hidden writes in paged mode. NPU `auto` now directly selects the native
ScatterNd API without importing Triton; CPU retains `index_copy_`. Explicit
`triton` remains experimental: unavailable imports fail startup and runtime/JIT
errors propagate without a silent retry. The user's 910B profile reported about
1.377 ms for one `row_copy_kernel`, slower than the earlier native writer. Start
with `scatter_nd`, retaining the existing native helper, graph settings, head
count and budgets. The reported recomputation loads were only similar, so this
is not a strictly matched-shape stable slowdown ratio or proof of inefficient IR.
The dedicated paged K/V `scatter_pa` writer is unchanged.

The experimental copy kernel reads update indices/values and stores only those destination
rows, without reading/scanning/cloning/zeroing the entire destination. The caller
provides unique live and padding-sink addresses. The reported FP16 destination
`(1097744,128)` maps to hidden storage at the user's current settings:

```text
22 * 512 * 6 * 2048 / 128 + (1024 + 1) * 2048 / 128 = 1,097,744 rows
2320 update rows / (2048 / 128) = 145 positions = 144-token bucket + one dummy
```

This is about 268 MiB of destination and 580 KiB of updates. It identifies the
hidden write, but does not prove that the previous ScatterNd kernel internally
scanned or copied the full destination. The paged token graph now expresses each
complete hidden vector as one row: `(1097744,128)` becomes `(68609,2048)`, and
`(2320,128)` updates become `(145,2048)`. Hidden gathers likewise use one row per
token. Byte counts, scratch isolation and cache validity are unchanged; the mixed
hidden/KV TND and rectangular layouts retain their gcd row width. Official
[VectorCore guidance](https://github.com/Ascend/triton-ascend/blob/main/docs/en/programming_guide/vector_operator.md)
suggests future optimization candidates, without proving the
current root cause or a latency gain. No target model, scheduler, head count or
draft budget changes are introduced.

`ASR_MTP_CONFIG qkv_rope_backend` reports the selected QKV/RoPE backend;
`row_write_backend` there is the requested option. The actual graph row writer is
reported by `ASR_MTP_GRAPH_READY row_write_backend`; `row_width=2048` identifies
the whole-token hidden rows in the paged token graph. Check with:

```bash
rg -n 'ASR_MTP_CONFIG|ASR_MTP_GRAPH_READY' "$LOG"
```

Helper tests and conditional full NPU replay tests cover the new operations.
The authoring host has no NPU; CPU tests cannot establish 910B numerical,
capture/replay or latency acceptance. Hold checkpoint, heads, token buckets,
budgets and load fixed; change one backend at a time and compare complete outputs,
acceptance, prefill/decode, commit P95/P99 and the graph handoff timeline.

## Device test

After applying the patch, run the unit/integration tests against the installed
modules (no checkpoint download or standalone HF replay):

```bash
PYTHONPATH="$PWD/vllm_ascend/spec_decode${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s examples/910b/tests -p test_asr_mtp_graph.py -v

PYTHONPATH="$PWD/vllm_ascend/spec_decode${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s examples/910b/tests -p 'test_asr_mtp_token_*.py' -v

PYTHONPATH="$PWD/vllm_ascend/spec_decode${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s examples/910b/tests -p 'test_asr_mtp_fused*.py' -v

PYTHONPATH="$PWD/vllm_ascend/spec_decode${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s examples/910b/tests -p test_asr_mtp_runtime.py -v

PYTHONPATH="$PWD/vllm_ascend/spec_decode${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s examples/910b/tests -p test_asr_mtp_paged.py -v

PYTHONPATH="$PWD/vllm_ascend/spec_decode${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s examples/910b/tests -p test_asr_mtp_qkv_rope.py -v

PYTHONPATH="$PWD/vllm_ascend/spec_decode${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s examples/910b/tests -p test_asr_mtp_row_copy.py -v
```

The actual NPU tests retain unfused cases and additionally combine native fused
heads with rectangular graphs and packed TND FIA. They use NPUGraph capture/replay
with changed inputs, rejection, request reordering and multiple buckets. They
must not be skipped on the device.
CPU tests cover the same fixed tensor program, masking, cache isolation, caps and
worker drive-token boundaries; they do not establish NPU compatibility or speedup.
No 910B device acceptance or latency gain has been measured on the authoring host.

Compare full target outputs, acceptance, prefill/decode, commit P95/P99, queue
wait and memory under the same load. Graphing removes host launch work; LM-head
matrix computation and padded attention computation remain.
