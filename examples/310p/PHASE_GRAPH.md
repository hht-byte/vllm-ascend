# 310P phase graphs: startup capture, in-place replay

This experimental extension is opt-in. The previous token layout has device
validation; the new native FA/PA graphs require the probes below on the target
310P software stack. CPU tests do not establish native kernel graph support.

## Routing

| Scheduler batch | Graph family | Operator |
| --- | --- | --- |
| All scheduled tokens are prompt tokens, no computed prefix | prefill | `_npu_flash_attention` (normal mask) |
| All prompts are complete and every request schedules one token | decode | `_npu_paged_attention` |
| Cached/chunked prefill, mixed batches, multi-token verification | token | `_npu_paged_attention_splitfuse_v2` |

A one-token prompt is prefill. A step crossing the prompt boundary is token.
Each family is keyed by total padded tokens, without request counts or qLens
partitions. For buckets `[20, 80, 192]`, startup captures nine FULL model graphs.
Existing startup capture guards remain enabled: runtime cannot lazily capture,
replace, or recapture missing graphs. No `graph_task_update` is used by this mode.

When token graphs are enabled, graph-mode resolution temporarily uses a capture
query length of 1. Upstream speculative decoding would otherwise round every
bucket to a multiple of `num_speculative_tokens + 1` and deduplicate the list.
The real runner query length is restored after resolution, including on errors;
the speculative token budget is unchanged. For example, the 24 buckets
`[1,2,3,4,5,6,7,8,10,12,14,16,18,20,32,40,64,80,128,256,384,512,768,1024]`
remain 24 buckets (72 graphs with phase routing) even with 15 speculative tokens.
Other limits, including the model context and scheduler token cap, still apply.
This protection does not add support for new speculative methods: the stock
configuration validator currently allows only ngram, and externally integrated
rollback implementations still require their own device acceptance.

## Fresh prefill representation

All Q/K/V tokens in bucket T are packed into one virtual sequence with constant
`seq_len=[T]`. FA receives a separate immutable CPU int32 length tensor for
each bucket, matching the op-plugin SelfAttention host-length contract. It is not
the device context-length buffer used by PA, and cannot be shared or overwritten
when another bucket runs. A persistent device additive mask permits attention exactly when
query and key belong to the same request and key position is not after query
position. Padding belongs to a separate virtual request; its KV write slots are
`-1`. Real positions/RoPE and scheduler request boundaries are unchanged.

Thus an 8-request partition and a 10-request partition reuse one FA graph at
the same T, without changing the operator's lengths or tensor shapes. The mask
is updated in place before replay. The original FA v3 implementation failed the
192-token eager probe on device: its compressed triangular mask cannot represent
the masked lower-triangle regions between requests. The packed path now explicitly
uses `_npu_flash_attention` with `MASK_TYPE_NORM`; the existing non-graph FA v3 path
is unchanged. See the [compressed mask constraints](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/acce/ascendtb/ascendtb_01_0358.html).
The probe compares eager and replay to an independent per-request CPU causal
reference. Normal-mask execution and replay still need target-device validation.
If either fails, do not enable phase
routing. No fallback to a different operator or runtime capture is performed.

Each bucket owns an FP16 normal mask with both dimensions rounded up to 16,
in NZ format. Its address and shape stay fixed across replay and bucket switches.
Logical mask storage is `2 * ceil(T/16)^2 * 16^2` bytes per bucket; changed partitions
rebuild/upload that bucket's mask, while identical layouts reuse it.
This adds preparation cost, and packing may compute masked cross-request
scores. Measure end-to-end latency and memory; do not assume a speedup over the
previous token implementation. Only fresh prefill uses this representation.

Decode uses T fixed single-query rows with persistent context lengths, block
tables and slot mapping. Cached prefill and verification keep the previously
validated per-token causal-context representation.

## Device probes

Run in separate processes from the repository root:

```bash
python examples/310p/probe_phase_graph.py --family prefill --buckets 20 80 192 --output prefill-phase.json
python examples/310p/probe_phase_graph.py --family decode --buckets 20 80 192 --output decode-phase.json
python examples/310p/probe_token_graph.py --buckets 20 80 192 --layout token --update-mode inplace --output token-phase.json
```

The native probe executes production attention methods and metadata preparation,
captures every bucket before replay workloads, revisits buckets sharing metadata,
and checks changed request counts, partitions, padding, contexts and block tables.
It poisons graph outputs before replay and snapshots them before eager reference
execution. The report is saved on failure; capture failures require a new process.

If failure occurs at `[EAGER]`, isolate native setup before investigating graphs:

```bash
python examples/310p/probe_phase_graph.py --family prefill --buckets 192 --eager-only --output prefill-eager.json
```

`[INPUTS]` records shapes, devices, dtypes, addresses, strides, NPU formats and
host length values. The report separates cache-write, attention and capture/replay
stages. An eager-only pass does not validate graph capture or dynamic mask replay.

## Full-model configuration and acceptance

After the probes pass, enable:

```json
{
  "additional_config": {
    "token_graph_310p": {
      "enabled": true,
      "phase_routing": true,
      "request_layout": "token",
      "update_mode": "inplace"
    }
  }
}
```

Use the existing model/cases JSON files with:

```bash
python examples/310p/accept_token_graph.py --config model.json --cases cases.json --buckets 20 80 192 --phase-routing --out model-phase-acceptance
```

The audit requires stable identities, one graph per family/bucket, zero runtime
captures, matching eager/graph outputs and actual replay in all three families.
Missing family coverage remains an incomplete acceptance result. Include long
prompts spanning scheduler chunks to exercise cached prefill through token graphs.
User-supplied latency and memory bounds remain necessary for full acceptance.

Omitting `phase_routing` preserves the device-validated all-token behavior.
