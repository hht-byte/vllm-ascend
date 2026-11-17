# Synchronous ASR CPU turnaround

This update reduces CPU preparation and redundant API calls before the existing
preparation replay. It changes no graph layout, tensor address, model forward,
sampling policy or request-level MTP switch.

## Changes

- The opt-in preparation owner already rejects async, hybrid/GDN and CP modes.
  On this dense vLLM 0.23 path, the accepted-token event is allocated but its
  only producer is in hybrid state postprocessing. Skip that unproduced event
  wait while retaining the real CPU accepted counts, remapping and padding.
- Dense attention does not consume the hybrid decode-draft flags. Omit their
  CPU construction and unused GPU upload on this validated path. Other modes
  keep their existing flags, transfer and event synchronization.
- A successful device-rejection handoff returns after its same-stream readback
  event has completed. Acknowledge the preparation upload generation that
  preceded it, so the next host-buffer reuse need not query that event again.
  A stale generation or an open upload batch cannot release the fence.

Upload event records remain in place. Handoff fallback, exceptions and direct
helper replay keep the original query/wait protection. There are no additional
D2Hs, device synchronizations, per-step timings/logs or public startup options.
The existing `asr_input_prepare_graph.enabled` and MTP configuration are enough.

## Deployment

Update all three runtime files: `worker/model_runner_v1.py`,
`worker/asr_input_prepare_graph.py` and
`patch/worker/patch_external_draft_mtp.py` under `vllm_ascend`.

For an engine still at `159589efb` with the previous online MTP extension already
installed, the focused delta updates all three plus the focused tests:

```bash
git apply --check examples/910b/asr_cpu_prepare_v023.patch
git apply examples/910b/asr_cpu_prepare_v023.patch
```

If the branch has already been updated to this commit, the two tracked runner
files are current. An older MTP hook installed from the extension patch needs
only the hook upgrade:

```bash
git apply --check examples/910b/asr_mtp_upload_completion_v023.patch
git apply examples/910b/asr_mtp_upload_completion_v023.patch
```

Fresh installation through the updated cumulative `asr_mtp_v023.patch` already
contains the new hook; do not apply the hook upgrade a second time. Its existing
external-draft/first-verify prerequisites remain unchanged. Restart through the
normal service process to load changed Python modules. This work did not restart
the production service.

The completion API is optional for cross-version compatibility: an older helper
or hook retains the original upload fence behavior rather than clearing an
unproven copy. Deploying the matching files is necessary for the full benefit.

## 910B evidence

Baseline is `159589efb`. One isolated native ASR engine on 910B3 device 1 uses
FP16 Qwen3-ASR-1.7B, TorchNPU 2.10.0.post4, CANN 9.1.0 and vLLM 0.23.0. Both
variants share the same weights, input cases, preparation owner, graphs and
GPU storage; fingerprints are checked at every CPU-method switch. This is a
public-input simulation, not a production WebSocket replay.

Order is baseline/candidate/candidate/baseline. Each phase warms all twelve
fixed audio cases per client, measures C4 with three 72-completion windows, and
then separately profiles 32 requests with `record_shapes=False`, stack/memory
off and a Python-only shape ring. API-count proxies are removed before timing;
control-plane synchronization is outside active traffic. `profile_stages=False`.

All 864 measured and 128 profiled responses match reference text/token IDs.
Each trace has 56 warm decode steps across B4T16/B4T14/B2T8/B2T7; prefill/mixed
shapes are excluded by shape, with no latency clipping. Medians, milliseconds:

| Interval | Baseline 1 -> candidate 2 | Baseline 4 -> candidate 3 | Pooled |
| --- | ---: | ---: | ---: |
| CPU `_prepare_inputs` | 1.36181 -> 1.11169 | 1.36105 -> 1.11548 | -18.30% |
| MTP tail -> preparation replay | 3.29957 -> 2.99474 | 3.26312 -> 3.02342 | -8.62% |
| MTP tail -> target verify | 4.12754 -> 3.81387 | 4.06440 -> 3.84570 | -6.74% |

Every matched shape improves CPU preparation, the first gap and total gap in
both order-balanced pairs. The boundary starts at MTP's final ScatterNdUpdate,
passes preparation MODEL_EXECUTE and ends at the target decoder's first RmsNorm.
The prior generic slot kernel is absent and is not used as a boundary.

Diagnostic API counts confirm the mechanism: baseline has one accepted-event
synchronize, upload-event query and unused draft-count H2D per step; the combined
candidate has zero of these calls. Event record remains once per upload batch,
with a matching successful completion receipt. These are API counts; they do
not imply that the unproduced event was blocking device work.

The complete-request mean is 251.38952 -> 250.52125ms, only 0.35%, with mixed
pair directions. The evidence establishes a local turnaround improvement,
not a general end-to-end latency/throughput gain or elimination of all CPU time.
A NumPy sequence-length variant had no independent incremental gain in the
balanced screening and was reverted. Lazy event recording was not introduced.
Aggregates and source hashes are in `asr_cpu_prepare_results.json`.

## Verification

The selected version passes 55 worker/preparation tests and 50 MTP/handoff tests
in the real 910B environment. Coverage includes preserved CPU counts, legacy
hybrid/async behavior, cold/mixed prefill, shrinking/reordered batches, stale
upload generations, failed handoff and standalone helper reuse. A fresh review
checked producer/consumer contracts and pinned-buffer lifetime. The focused
delta, hook-only upgrade and cumulative hook creation pass installation,
exact-source comparison and reverse-application checks.
