# 910B PA token graph feasibility probe

This is an isolated device acceptance tool, not a production model-runner switch.
It uses ND `[blocks, block_size, kv_heads, head_size]` caches and the native
`_npu_paged_attention` operator. Query rows, device int32 context lengths and
block tables retain their addresses. Each scheduled query token gets its own
PA row, with visible context `C-Q+j+1`; padding reads dummy block zero with
context length one. KV is pre-populated, including the current query tokens.

No task update or runtime recapture is used. Every bucket is captured before
dynamic testing. Replay outputs are poisoned, synchronized and checked against
an independent request-level CPU causal GQA reference **before** running eager
again. This avoids eager setup accidentally refreshing state before replay.

Start in a fresh process for each command:

```bash
python examples/910b/probe_token_graph.py --buckets 20 --eager-only --output pa-eager.json
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
synchronization make this a correctness probe. Production routing remains unchanged.
