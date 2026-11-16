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

Expect `device_rejection=True`, a positive `prelude_graphs`, and an initial
`ASR_MTP_DEVICE_REJECTION batches=1` line, then every 100 eligible batches.
The cumulative `cold_requests`, `warm_requests` and `cache_only_requests`
separate head execution from hidden-only prefill/rollback rows.
Fallback reasons are printed in that line. A lack of eligible batches is not
evidence that the prelude ran. `readbacks` is combined handoff copies, not all
readbacks in the engine.

Compare the same workload ON/OFF. Check text/token parity first, then the
rejection-end→MTP-start span, total step time, throughput, commit P95 and memory.
Do not infer a speedup from draft acceptance alone. `ASR_MTP_HOST_STAGE` measures
the late proposal call; in this mode it excludes the work already performed by
the early handoff, so use the profiler for complete timing.

## Validation limits

The subsequent [blue host-staging validation](blue_mtp_host_staging.md) reduces
metadata preparation and late bookkeeping costs, adds attribution for this early
path, and provides a focused `asr_mtp_host_staging_v023.patch` for an engine already
at this document's installed baseline. [Real-weight validation](blue_mtp_trained_validation.md)
records the trained five-head graph check and the separate public ASR smoke test.
Those component results do not establish end-to-end MTP service gains.

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
