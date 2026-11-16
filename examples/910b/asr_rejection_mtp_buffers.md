# Rejection to MTP buffer reuse

This update targets the **target rejection sampler to MTP input staging**
interval. It does not change MTP head count, graph buckets, acceptance, EOS or
the later MTP-result `tolist()` readback. No measured latency improvement is
claimed until a matching 910B run verifies the change.

## Upgrade an existing installation

The focused patch assumes engine `310P-graph` at `55d67eb75` and installed MTP
sources matching ASR `codex/streaming-prespec-monitor` at `6bf62166a`.

```bash
git apply --check examples/910b/asr_rejection_mtp_buffers_v023.patch
git apply examples/910b/asr_rejection_mtp_buffers_v023.patch
python tests/ut/worker/test_asr_rejection_output_standalone.py
PYTHONPATH=vllm_ascend/spec_decode python -m unittest discover \
  -s examples/910b/tests -p 'test_asr_mtp_staging.py'
```

If using the updated branch's engine source already, apply only its MTP hunks:

```bash
git apply --check --include='vllm_ascend/spec_decode/*' \
  examples/910b/asr_rejection_mtp_buffers_v023.patch
git apply --include='vllm_ascend/spec_decode/*' \
  examples/910b/asr_rejection_mtp_buffers_v023.patch
```

Do not apply the focused patch on top of the updated cumulative patches.
The fresh `research/asr_mtp_v023.patch` and the cumulative
`examples/910b/asr_mtp_graph_v023.patch` contain the upload helper.
The cumulative `asr_input_prepare_graph_v023.patch` contains the rejection
helper and runner hooks. Historical incremental patches keep their old bases.
For other installed MTP versions, regenerate using `build_asr_mtp_patch.py`.

Deploy all changed files together. The new modules are
`vllm_ascend/worker/asr_rejection_output.py` and
`vllm_ascend/spec_decode/asr_mtp_staging.py`.

## Enable and ablate

Preserve `mtp_engine_config(...)` and the existing configuration dictionary:

```python
additional = dict(base_config.get("additional_config") or {})
prepare = dict(additional.get("asr_input_prepare_graph") or {})
prepare.update(enabled=True, reuse_rejection_buffer=True)
additional["asr_input_prepare_graph"] = prepare
base_config["additional_config"] = additional
```

This inherits the preparation feature's synchronous, single-device, dense ASR
and FULL-graph requirements. `reuse_rejection_buffer` defaults to `True` only
inside that opt-in feature. Set it to `False` for the original rejection D2H
path; other preparation optimizations remain enabled. No new service CLI flag.

The buffer covers `max_num_seqs * (num_spec_tokens + 1)` int32 IDs. Each eligible
batch copies into a contiguous prefix with `non_blocking=True`, records one
event and waits before CPU access. The original vLLM parser handles the CPU
view, preserving all filtering semantics. Logprobs, unsupported dtype/shape,
single-token and asynchronous paths retain their original behavior.

## MTP metadata and PA tables

Both BQ and token graph programs preallocate pinned host metadata. Token PA
tables share their program's upload owner. Each staging call records one event
after changed buffers have been copied; the next reuse queries it and waits
only if incomplete. Unchanged tables still skip upload. CPU data populates
NumPy views of pinned storage, avoiding per-group `torch.tensor`/`pin_memory`
construction. Capture/replay and FIA task-update dependencies are unchanged.

This is automatic for the updated graph programs; eager MTP remains unchanged.
It needs no extra device buffers or graph variants. It does **not** remove the
required host visibility of accepted target IDs or claim that every CPU event
can be captured inside a graph.

## Verify on 910B

```bash
rg -n 'ASR_INPUT_PREPARE_GRAPH replays=|ASR_MTP_GRAPH_HOST_STAGE|ASR_MTP_HOST_STAGE' "$LOG"
```

Preparation logs now include cumulative `rejection_copies` and
`rejection_fallbacks`. They appear at the existing rate limit, and may lag the
current sampling step. A copy count above zero confirms the reusable D2H path.
Use existing `asr_mtp.profile_stages=True` only for a diagnostic run; disable it
for timing comparisons. No new per-request logs or timer events were added.

Compare the same load/configuration, actual query sizes and head counts. Measure
rejection-end to the matching MTP graph's first device kernel, separately from
MTP-end to the next target replay. Inspect CPU self time and pinned allocation
counts; do not sum nested `copy_`, `to` and synchronization spans. CPU tests and
patch checks establish contracts, not NPU replay correctness or latency gains.
