# Device rejection / prefill → MTP

`additional_config.asr_mtp.device_rejection` is an **opt-in startup switch**,
default `False`. It requires the existing synchronous MTP worker with
`graph_mode="full"`, token buckets (`graph_token_sizes`) and
`kv_cache_backend="paged"`. Keep all other model/graph/head settings unchanged
for the first comparison.

After the existing `mtp_engine_config(...)` call has enabled MTP and populated
`additional_config.asr_mtp`, add:

```python
engine_config["additional_config"]["asr_mtp"]["device_rejection"] = True
```

Set it to `False` or omit the key to use the existing CPU rejection handoff.
This changes startup graph capture and requires the usual service restart; it
does not hot-reconfigure a running worker.

## Execution and coverage

Previously: rejection → target samples D2H/wait → CPU bookkeeping → MTP →
drafts D2H/wait.

Enabled, eligible batch: rejection → small device metadata graph → existing
FIA MTP graph → **one combined samples/drafts D2H and wait** → original CPU
bookkeeping → consume already-generated drafts. The original target sampler
tensor, scheduler, rejection parser, and target model execution remain intact.
The hook wraps `_bookkeeping_sync`; no target runner source change is needed.

Cold and warm greedy MTP requests can share a batch, including:

- Complete prefill with a single sampled token, or first-prefill verification
  of an external draft. Cold auxiliary caches initialize from target hidden
  already computed in this forward; there is no extra target forward.
- Intermediate chunked prefill: save the executed hidden prefix, discard the
  unused sampled row and run no heads for that request. Its final prefill chunk
  starts cold MTP using the saved prefix.
- Warm speculative verification or ordinary single-token continuation, with
  cold/warm head budgets selected separately for each request.

A pending legacy rollback still goes through the target before MTP. That row
only saves hidden during the first handoff, then late routing submits the
original rollback. Inactive rows do not create fictitious proposal rounds. A
whole cache-only batch runs the metadata/hidden-write prelude without a head
graph. Missing cold slots are leased only after complete CPU capacity preflight.

Every row must have MTP enabled, supported greedy sampling, no invalid
speculative tokens or logprobs, and contiguous target hidden history. The batch
must fit one token bucket and the graph's context/request bounds. Unsupported
rows, missing prefixes, preemption/recomputation or capacity misses retain the
existing CPU-bookkeeping → MTP path for the entire batch; eligible requests
still use their normal MTP graphs there. EOS, minimum/maximum tokens and cleanup
keep their original semantics. This does not remove final CPU bookkeeping.

Non-speculative prefill returns a padded CPU sample block only to the existing
bookkeeping parser, avoiding its single-column device-copy/event branch. Other
consumers retain the original sampler tensor.

FIA keeps its CPU-known Q/KV lengths and task update. They describe the maximum
possible confirmed span of this scheduled verification. The device derives the
actual accepted length, drive token and last-row indices. Causal attention
prevents computations beyond that last row from changing the useful output.
CPU cache validity is committed only through the actual confirmed end.

This spends extra tail computations to remove a pre-MTP synchronization. It is
not conditional EOS graph execution: terminal rows may still compute unused
heads. `ASR_MTP_DEVICE_REJECTION optimistic_positions` counts the upper-bound
work, including rows whose drafts are not ultimately consumed.
The worker's execution position/batch counters also include that work. Acceptance
statistics attribute all planned head work to a surviving proposal; a terminal
row offers no new proposal, so its wasted work is visible in execution counters
rather than in a fictitious acceptance round.

Only a small metadata prelude is added per existing graph; the expensive MTP
head graph is reused. Startup reservation includes the prelude graphs and
persistent buffers. Metadata is uploaded through one pinned block, raw samples
and target hidden use device-to-device copies, and there are no per-head host
readbacks. CPU mirror/scheduler work still occurs after MTP.

## Install and verify

For an engine already carrying the 2026-11-16 buffer reuse version (canonical
e55795326 / delivery 92c64fb2d), apply **only** the focused upgrade:

```bash
git apply --check examples/910b/asr_mtp_device_rejection_v023.patch
git apply examples/910b/asr_mtp_device_rejection_v023.patch
```

Fresh/cumulative installations use the updated `research/asr_mtp_v023.patch`
or the existing graph upgrade workflow with
`examples/910b/asr_mtp_graph_v023.patch`. Do not stack these alternative MTP
delivery patches. The exporter `research/build_asr_mtp_patch.py` can generate
an upgrade against the actual installed source, checking runner contracts.

```bash
rg -n 'ASR_MTP_CONFIG|ASR_MTP_GRAPH_READY|ASR_MTP_DEVICE_REJECTION' "$LOG"
```

Expect `ASR_MTP_DEVICE_REJECTION_HOOK installed=True`, `device_rejection=True`,
a positive `prelude_graphs`, and an initial
`ASR_MTP_DEVICE_REJECTION batches=1` line, then every 100 eligible batches.
The cumulative `cold_requests`, `warm_requests` and `cache_only_requests`
separate head execution from hidden-only prefill/rollback rows.
Fallbacks also emit `ASR_MTP_DEVICE_REJECTION_FALLBACK` independently: once per
reason on its first occurrence and then every 100 total fallback batches.
Thus a completely falling-back workload still reports `attempts`, successful
`batches`, `fallback_batches`, reason counters and CPU gate values. Batch
diagnostics include request count versus `max_batch`, sampler shape/dtype and
unsupported flags; geometry includes confirmed/executed/cached offsets;
context diagnostics include the actual optimistic end versus its limit.
No token payload or extra device read is used for these diagnostics.
Attempts cover synchronous bookkeeping calls, including prefill; successful
batches can include cache-only prefill and are not a count of rejection rounds.
`readbacks` is combined handoff copies, not all readbacks in the engine.

If capture succeeds but there is no handoff activity, check the installed
hook, not only the `spec_decode` files:

```bash
rg -n -C 3 'def bookkeeping|handoff.prepare|cls\._bookkeeping_sync = bookkeeping' \
  vllm_ascend/patch/worker/patch_external_draft_mtp.py
```

No match means the runtime entry was omitted from the installed patch: graph
capture alone cannot activate the handoff. If the other files already contain
the device rejection upgrade and the hook still matches the documented old
baseline, repair **only** that file with the same focused patch:

```bash
git apply --check --include='vllm_ascend/patch/worker/patch_external_draft_mtp.py' \
  examples/910b/asr_mtp_device_rejection_v023.patch
git apply --include='vllm_ascend/patch/worker/patch_external_draft_mtp.py' \
  examples/910b/asr_mtp_device_rejection_v023.patch
```

If the check fails, regenerate the upgrade against the actual installed
checkout with `research/build_asr_mtp_patch.py`; do not force or stack a full
patch onto a partially updated installation. Restart the engine after repair.
The current loader checks a capability marker on the bound bookkeeping method
before loading weights when `device_rejection=True`; a missing/old/overridden
hook now fails startup explicitly. Deploy the hook and loader together. The
default OFF path does not require the marker.

Compare the same workload ON/OFF. Check text/token parity first, then the
rejection-end→MTP-start span, total step time, throughput, commit P95 and memory.
Do not infer a speedup from draft acceptance alone. `ASR_MTP_HOST_STAGE` measures
the late proposal call; in this mode it excludes the work already performed by
the early handoff, so use the profiler for complete timing.

## Post-handoff CPU work and optional stage timing

The pending proposal path validates only the newly confirmed token suffix. It
does not materialize the full token history again or recompute per-head cost
already recorded by early execution. The combined readback still uses one
buffer/event: submit the copy, compute rejection-independent CPU accounting,
then wait once before consuming results. This overlaps small CPU work with
queued MTP/D2H; it does not remove the required synchronous engine boundary.

On runners with `_rejection_output_buffer`, the bookkeeping hook temporarily
uses a CPU-only parser for the exact completed handoff tensor. This avoids
NumPy masks and per-row indexed copies. Placeholder/vocabulary/discard rules
match vLLM 0.23, including logprob and different-tensor fallback. The original
buffer is restored in `finally`; other samplers and requests are unchanged.

Set the existing `additional_config.asr_mtp.profile_stages` to `true` for a
short diagnostic run. `ASR_MTP_HANDOFF_STAGE` records:

| Field | Host interval |
| --- | --- |
| `prepare_host_ms` | Eligibility, optimistic metadata, staging |
| `replay_submit_ms` | Prelude/MTP replay submission |
| `readback_submit_ms` | Combined D2H enqueue and event record |
| `overlap_cpu_ms` | Independent accounting before waiting |
| `readback_wait_ms` | Completion wait, including queued device work |
| `result_host_ms` | CPU result materialization and counters |
| `bookkeeping_host_ms` | Original runner bookkeeping with CPU mirror |
| `late_propose_host_ms` | Pending proposal consumption |
| `turnaround_host_ms` | Late proposal return to next `_update_states` entry |
| `cpu_parser_used` | Whether this runner exposes the CPU parser seam |

The log is emitted on the next `_update_states`; the final batch with no next
engine step has no turnaround sample. Turnaround includes output delivery,
scheduling and possibly waiting for new work; it is not scheduler CPU time or
device idle time. These fields are host spans, not NPU event durations, and
must not be added to overlapping device durations. With the default `false`,
there are no timing-clock reads, new timing events or per-step timing logs.

The independent NPU simulation and its limitations are recorded in
`asr_mtp_post_handoff_validation.md`. Benchmark without profiling enabled;
collect a separate short trace when attribution is needed.

## Validation limits

CPU tests exercise zero/partial/full acceptance, cold/prefill/warm mixing,
multi-chunk prefill, legacy rollback priority, persistent-cache parity,
reordering, EOS/minimum/length bounds, stale padded samples, no duplicate
proposal, transactional cold-slot allocation, fallback and lifetime guards.
The device tests were also executed on an idle Ascend 910B3 with Python 3.12.13,
Torch 2.10.0+cpu, TorchNPU 2.10.0.post4, CANN 9.1.0 and vLLM 0.23.0+empty
(vLLM-Ascend 0.23.0.post1). Both actual prelude +
PA/FIA graph replay tests passed against eager MTP: warm partial rejection, and
partial then completed prefill mixed with warm verification. The installed
vLLM CPU rejection parser also passed the padded single-sample/discarded-row
contract test. These are small synthetic models, not the trained ASR checkpoint.

Full-service token/text parity, startup memory coexistence and end-to-end
latency/throughput remain to be measured on the actual ASR workload. The tests
do not establish a production speedup; no running ASR service was restarted.
