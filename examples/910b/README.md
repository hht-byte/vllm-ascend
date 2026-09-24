# 910B PA token graph feasibility probe

This is an isolated device acceptance tool, not a production model-runner switch.
It uses ND `[blocks, block_size, kv_heads, head_size]` caches and the native
`_npu_paged_attention` operator. Query rows, int32 context lengths and
block tables retain their addresses. Each scheduled query token gets its own
PA row, with visible context `C-Q+j+1`; padding reads dummy block zero with
context length one. KV is pre-populated, including the current query tokens.

The default is now `--context-device cpu --update-mode task_update`, based on
910B4 measurements: device lengths failed eager Setup; host lengths passed
eager/capture but changed host lengths failed inplace replay. Each bucket owns
a task-group handle and external event. Before replay, workspace is obtained
for the current parameters and PA is reissued inside graph_task_update on a
dedicated stream. No runtime recapture is used. Every bucket is captured before
dynamic testing. Replay outputs are poisoned, synchronized and checked against
an independent request-level CPU causal GQA reference **before** running eager
again. Output poisoning happens after task update as well. This avoids eager
setup or execution during update accidentally satisfying the replay comparison.
`--update-mode inplace` remains available as an explicit negative control.

Start in a fresh process for each command:

```bash
python examples/910b/probe_token_graph.py --buckets 20 --eager-only --output pa-eager.json
python examples/910b/probe_token_graph.py --buckets 20 --context-device cpu --update-mode task_update --control contexts --output pa-host-update.json
python examples/910b/probe_token_graph.py --buckets 20 --control inputs --output pa-inputs.json
python examples/910b/probe_token_graph.py --buckets 20 --control contexts --output pa-contexts.json
python examples/910b/probe_token_graph.py --buckets 20 --control blocks --output pa-blocks.json
python examples/910b/probe_token_graph.py --buckets 20 --control qlens --output pa-qlens.json
python examples/910b/probe_token_graph.py --buckets 20 80 192 --output pa-all.json
python examples/910b/probe_token_graph.py --buckets 20 80 192 --graph-pool shared --output pa-shared.json
```

Defaults: 8 query heads, 2 KV heads, head size 64, block size 128, FP16, max
context 512. Set `--heads`, `--kv-heads`, `--head-size`, `--block-size`,
`--max-context`, `--dtype` and `--device` for the target model. Control `inputs`
changes Q/K/V; `contexts` changes only lengths; `blocks` changes only block
mapping; `qlens` keeps request count and final contexts fixed while changing
query partitions. `all` combines changes and tests different request counts and
padding. Eager-only checks every case without capturing any graph.

The JSON report contains environment versions, last execution stage, inputs,
capture count, graph identities, metadata addresses and errors. A numeric
mismatch also saves `.failure.pt` with actual/expected values. Captures that
fail can leave the NPU runtime unhealthy: stop and use a fresh process; do not
retry within a failed capture. Do not enable synchronous launch blocking during
normal graph acceptance. Record CANN/ATB and driver versions alongside the report.

Passing this probe establishes only PA attention replay for the tested shapes
and software stack. It does not validate cache-write kernels, FULL-model replay,
rollback custom_class proposals/acceptance, distributed execution, quantized KV,
or performance. Timing is intentionally not reported: CPU reference and explicit
synchronization make this a correctness probe. Default production routing remains unchanged.

## Experimental full-model acceptance

The 910B4 CPU-context/task-update probe passed 3 graphs and 28 cases on the
reported torch 2.10.0 / torch_npu 2.10.0.post4 stack. Full-model integration
still requires device acceptance, including real KV writes and speculative rollback.

Enable the opt-in runner with `additional_config={"token_graph_910b":{"enabled":true}}`,
`async_scheduling=false`, and `compilation_config.cudagraph_mode="FULL"`.
Startup captures one PA graph per total-token bucket. All phases currently use
the token layout; 310P phase routing is separate. Requests beyond the configured
buckets use the normal eager path. Metadata is expanded only for PA and cache
write padding; scheduler request boundaries and speculative token counts remain intact.
Each layer's PA task is updated before replay using host context lengths. Cache
writes remain in the graph before attention. Explicit synchronization and per-layer
workspace refresh prioritize correctness; benchmark before drawing performance conclusions.
Layers with matching PA tensor layouts/attributes share one workspace per bucket.
Each update refreshes one workspace per matching layout, shared by its serial tasks.
Capture logs report the allocated bytes per layout. Capture allocations stay alive
for graph lifetime; previous update allocations remain alive until replacement tasks
finish updating. This prevents workspace retention from growing with layer count.
If capture still runs out of memory, lower `gpu_memory_utilization` in the model
configuration to reserve more memory outside the KV cache budget, then restart the
process. This is separate from the layer-sharing fix; full-model peak memory must
still be measured on the device.

Use the shared acceptance harness (it selects the backend for both child processes):

```bash
python examples/310p/accept_token_graph.py --backend 910b \
  --config model-config.json --cases model-cases.json \
  --buckets 20 80 192 --out model-910b-acceptance
```

The existing model-config/cases JSON format is unchanged. Keep `custom_class`
speculative settings in model-config when testing that path; the harness does not
replace the custom class or draft count. First run a non-speculative baseline,
then repeat with the actual custom class in a separate output directory. Offline
generation does not reproduce every streaming rollback: the application's streaming
finish/rejection/rollback workloads require an additional eager-versus-graph check.

Acceptance checks exact generated tokens, unchanged graph identities, request-independent
keys, PA task updates, and completion of all captures before workload warmup.
`missing_coverage` must be empty; set `--max-slowdown` and `--max-peak-gib` to supply
performance and memory acceptance bounds. Retain eager.log, graph.log and summary.json.

Initial scope: V1, one device, homogeneous causal full attention, floating-point KV,
no sliding windows, sinks, ALiBi, LoRA, KV transfer or ENPU. Speculation is limited to
ngram/custom_class (no draft model graph). Distributed and other attention backends
are not validated by this implementation.

### Compare against native FULL graphs

Native FULL already buckets tokens and updates attention parameters. Its uniform
decode descriptors and operator selection differ from the token PA path; native
FULL is not an eager baseline. To measure the incremental benefit directly:

```bash
python examples/310p/accept_token_graph.py --backend 910b --baseline native_full \
  --config model-config.json --cases model-cases.json \
  --buckets 20 80 192 --out native-full-vs-token
```

This runs native_full and token graph in separate processes with identical inputs,
seed, scheduling settings and requested buckets. Native bucket resolution is kept
unchanged (including speculative rounding); resolved graph keys are reported.
The ratio `native_full_over_token_graph` is native median latency divided by token
graph median latency: above 1 favors token graph, below 1 favors native FULL.
Check `native_full_only`, dispatch counts, graph counts and memory alongside ratios.
A baseline with fallback is marked explicitly; it is not a pure FULL replay comparison.
The prefill/decode labels denote generation workloads, not isolated attention kernel
timings. Repeat runs in fresh output directories to assess timing variability.

## Isolating eager Setup failures

`PagedAttentionOperation setup failed` before capture is not evidence of a graph
replay failure. The first eager call now prints all tensor descriptors, including
NPU format, storage offset, strides, length device and block-index range.
The op-plugin 910B PA tests use host context lengths. Compare that contract
explicitly without silently switching the device-inplace probe:

```bash
python examples/910b/probe_token_graph.py --buckets 20 --eager-only --context-device cpu --output pa-host-eager.json
python examples/910b/probe_token_graph.py --buckets 20 --eager-only --context-device npu --output pa-device-eager.json
```

If both fail, repeat the CPU case with `--stream default` to isolate side-stream
setup. Default-stream mode is restricted to eager-only. Preserve the first ATB
error from `/root/ascend/log/atb` (or the log directory printed by ATB) together
with the JSON. The generic Python Setup exception does not identify the rejected
parameter. Synchronous launch can be used for a separate eager-only diagnosis,
but must be removed before graph testing. A successful host-length eager test
does not establish host-inplace replay: op-plugin may clone host input tensors.
