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

## Device test

After applying the patch, run the unit/integration tests against the installed
modules (no checkpoint download or standalone HF replay):

```bash
PYTHONPATH="$PWD/vllm_ascend/spec_decode${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s examples/910b/tests -p test_asr_mtp_graph.py -v
```

The actual NPU test uses NPUGraph capture/replay with changed inputs, rejection,
request reordering and multiple buckets. It must not be skipped on the device.
CPU tests cover the same fixed tensor program, masking, cache isolation, caps and
worker drive-token boundaries; they do not establish NPU compatibility or speedup.
No 910B device acceptance or latency gain has been measured on the authoring host.

Compare full target outputs, acceptance, prefill/decode, commit P95/P99, queue
wait and memory under the same load. Graphing removes host launch work; LM-head
matrix computation and padded attention computation remain.
