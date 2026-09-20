# 310P phase graphs: startup capture, in-place replay

This experimental extension is opt-in. The previous token layout has device
validation; the new native FA/PA graphs require the probes below on the target
310P software stack. CPU tests do not establish native kernel graph support.

## Routing

| Scheduler batch | Graph family | Operator |
| --- | --- | --- |
| All scheduled tokens are prompt tokens, no computed prefix | prefill | `_npu_flash_attention_v3` |
| All prompts are complete and every request schedules one token | decode | `_npu_paged_attention` |
| Cached/chunked prefill, mixed batches, multi-token verification | token | `_npu_paged_attention_splitfuse_v2` |

A one-token prompt is prefill. A step crossing the prompt boundary is token.
Each family is keyed by total padded tokens, without request counts or qLens
partitions. For buckets `[20, 80, 192]`, startup captures nine FULL model graphs.
Existing startup capture guards remain enabled: runtime cannot lazily capture,
replace, or recapture missing graphs. No `graph_task_update` is used by this mode.

## Fresh prefill representation

All Q/K/V tokens in bucket T are packed into one virtual sequence with constant
`seq_len=[T]`. A persistent device additive mask permits attention exactly when
query and key belong to the same request and key position is not after query
position. Padding belongs to a separate virtual request; its KV write slots are
`-1`. Real positions/RoPE and scheduler request boundaries are unchanged.

Thus an 8-request partition and a 10-request partition reuse one FA graph at
the same T, without changing the operator's lengths or tensor shapes. The mask
is updated in place before replay. This requires the compressed FA v3 mask path
to honor the block-diagonal mask; the probe compares eager and replay to an
independent per-request CPU causal reference. If it fails, do not enable phase
routing. No fallback to a different operator or runtime capture is performed.

The first version rebuilds/uploads the 2048x2048 FP16 compressed-mask backing
when the partition changes (8 MiB logical data). Identical consecutive layouts
reuse it. This adds preparation cost, and packing may compute masked cross-request
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
