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
Each replay performs one host task update per auxiliary head; this rebinding
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

## Device test

After applying the patch, run the unit/integration tests against the installed
modules (no checkpoint download or standalone HF replay):

```bash
PYTHONPATH="$PWD/vllm_ascend/spec_decode${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s examples/910b/tests -p test_asr_mtp_graph.py -v

PYTHONPATH="$PWD/vllm_ascend/spec_decode${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s examples/910b/tests -p 'test_asr_mtp_token_*.py' -v
```

The actual NPU test uses NPUGraph capture/replay with changed inputs, rejection,
request reordering and multiple buckets. It must not be skipped on the device.
CPU tests cover the same fixed tensor program, masking, cache isolation, caps and
worker drive-token boundaries; they do not establish NPU compatibility or speedup.
No 910B device acceptance or latency gain has been measured on the authoring host.

Compare full target outputs, acceptance, prefill/decode, commit P95/P99, queue
wait and memory under the same load. Graphing removes host launch work; LM-head
matrix computation and padded attention computation remain.
