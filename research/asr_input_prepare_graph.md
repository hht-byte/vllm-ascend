# ASR input preparation graph

This optional 910B optimization targets host dispatch gaps between MTP and the
next target decoder verification. It captures a separate preparation graph;
it does not change the decoder FULL graph or the MTP graph.

## Enable

Add this to the engine kwargs before constructing the ASR engine. Preserve the
existing `asr_mtp` dictionary and the `mtp_engine_config(...)` startup helper.
These preparation options are siblings of `asr_mtp`, not fields inside it.

```python
additional = dict(base_config.get("additional_config") or {})
additional["asr_sync_input_packing"] = False
additional["asr_input_prepare_graph"] = {
    "enabled": True,
    "token_sizes": [16, 64, 256, 1024],
}
base_config["additional_config"] = additional
# Continue using the existing mtp_engine_config(...) and engine construction.
```

Keep the existing FULL graph configuration and MTP enabled. This startup option
requires synchronous scheduling/speculation, TP/PP/DP/PCP/DCP all equal to 1,
a dense model and the standard NPU runner input preparation method. Unsupported
settings fail startup instead of silently pretending to enable the graph.
The separate `asr_sync_input_packing` optimization must be disabled here.
No new CLI argument is introduced.

Explicit token sizes must fit the runner's token capacity. If omitted, defaults
are 16, 64, 256 and 1024 clipped to that capacity. They need not equal decoder or
MTP graph buckets. Each preparation graph has a fixed maximum request grid and
variable device query starts; request count and draft length do not add keys.
For example, 6 requests containing 73 scheduled tokens select bucket 256.
Batches above the largest bucket use the original path without runtime capture.

Disable with `additional["asr_input_prepare_graph"] = {"enabled": False}` or
remove that key before restarting the engine. To compare against previous input
packing, enable `asr_sync_input_packing` only in the graph-disabled run.

## What is captured

The CPU scheduler already computes authoritative positions and sequence lengths.
One pinned upload carries them along with computed counts, query starts and
discard indices/mask and synchronized accepted-token counts. One graph replay
writes the fixed runner buffers, fills padding, and invokes the existing slot
mapping kernel.
Device index/cast/add operations for recomputing positions are unnecessary.

The private query starts used for slot mapping repeat the actual token count
for inactive request slots. Attention query starts retain their original -1
padding and separate storage, including the FULL-mode extra dummy-request entry.
This preserves FIA's later query-start rewrite. Slots between the actual token
count and the preparation bucket are explicitly invalidated on every replay.
Reordering, rejection, mixed prefill/decode and shrinking batches do not retain
old slot addresses. Block table contents can change without changing pointers.

Capture happens once after final KV tables and decoder capture, with one shared
preparation pool. Only serial runner-stream execution is supported. The uploaded
buffer and final outputs live outside the graph pool. Host reuse waits only for
the previous metadata copy when necessary. Extra reserved capture bytes are
reported through the runner's existing graph-memory return value. This is not
a second deduction from the automatic KV budget.

CPU scheduler work, the earlier MTP handoff gap, input IDs/embedding, spec-decode
sampler metadata (now packed separately), FIA's later padding update and audio
encoder stay outside this
graph. It therefore does not promise to eliminate every gap or every H2D in the
reported 6.4 ms interval.

## Verify on the server

```bash
rg -n 'ASR_INPUT_PREPARE_GRAPH_READY|ASR_INPUT_PREPARE_GRAPH replays=' "$LOG"
python tests/ut/worker/test_asr_input_prepare_graph_standalone.py
```

`READY` reports captured buckets and additional reserved bytes. Runtime records
appear on first traffic and then at most once per five seconds while traffic
continues, including cumulative replay/miss counts, the current actual/bucket
token count and `spec_metadata_uploads`. Older versions log only replay 1 and
multiples of 1000; one log line does not mean only one replay occurred. A READY
line alone proves capture, not that requests reached this path. Miss counts cover
oversize/invalid token counts after capture; startup profiling is not counted.

Use identical workload, MTP settings and decoder graph settings in OFF/ON runs.
In profiling, compare wall time from input preparation to the next decoder replay
as well as host launches, instead of expecting faster slot-mapping arithmetic.
The covered copies/fills and slot mapping should appear in the additional graph
replay. Remaining outside-graph operators are described above. Separately track
the earlier approximately 2 ms handoff gap and end-to-end commit latency.

Local validation uses Torch CPU; the real NPU graph/slot-kernel test is skipped
without Ascend hardware. No hardware speedup is claimed from CPU tests.

## Patch delivery

`asr_input_prepare_graph_v023.patch` targets `310P-graph` at `2b3cb3c97`, which
already contains the previous synchronous input packing support. The focused
patch adds the helper and runner hooks only; tests and documentation accompany
the branch. Apply it to a clean matching checkout, then restart through the
normal service process to load the changed Python modules.

```bash
git apply --check examples/910b/asr_input_prepare_graph_v023.patch
git apply examples/910b/asr_input_prepare_graph_v023.patch
```

Do not apply the focused patch again if the branch already contains this change.

## Startup compatibility fix

vLLM 0.23 creates `num_accepted_tokens_event` whenever speculative tokens are
configured, including synchronous dense ASR. Its presence is not an unsupported
mode. The initial implementation incorrectly rejected this valid allocation.
The fix removes that check; real asynchronous, hybrid and unsupported graph
modes remain rejected, with each actual conflicting setting in the error.

The existing event wait and CPU accepted-count synchronization remain intact.
Only the accepted-count GPU transfer moves into the packed preparation replay;
actual synchronized counts are copied, with inactive slots padded to one.
Graph misses and graph-disabled runs retain the original transfer path.

For a checkout already containing `0343e34b8`, use the incremental startup fix:

```bash
git apply --check examples/910b/asr_input_prepare_graph_startup_fix_v023.patch
git apply examples/910b/asr_input_prepare_graph_startup_fix_v023.patch
```

For a checkout still at `2b3cb3c97`, the updated full preparation patch includes
this fix. Use either the full patch or the incremental fix for its matching
base. Updating to a branch commit that includes the fix needs no extra patch.

## Synchronous verifier metadata upload

The original verifier metadata builder uploads five separately pinned tensors,
then gathers input IDs twice on device (with an index addition). This happens
outside the preparation graph. The synchronous ASR fast path builds the same
indices and draft IDs from the already prepared CPU input IDs, then uploads one
persistent packed buffer. It adds no graph, device index kernels or D2H readback.

This path is enabled with the existing preparation graph setting. To isolate
this change while keeping the same preparation graphs, compare:

```python
additional["asr_input_prepare_graph"] = {
    "enabled": True,
    "token_sizes": [16, 64, 256, 1024],
    "pack_spec_metadata": True,  # Set False for the original verifier builder.
}
```

`spec_metadata_uploads` counts packed verifier calls, not graph replays. Batches
without drafts do not increment it. If previous input IDs are repaired on GPU,
the original builder is retained. CPU packing does not require a graph bucket
hit, so a preparation bucket miss can still use the packed verifier metadata.
The buffer is reused only in synchronous execution; a copy event protects pinned
host reuse and the runner stream orders device consumers before the next upload.

In a comparable speculative batch, expect five small metadata H2Ds to become
one and the builder's two Index/IndexCheck pairs and Add to disappear. Aggregate
MEMCPY counts include device-to-device copies inside the preparation graph, so
they are not a host-launch count. Other indices may have unrelated call sites.
Compare the same MTP-to-decoder wall interval with identical concurrency and
batch/token distributions. Embedding and the scheduler handoff remain outside
this optimization. CPU equivalence tests do not establish NPU latency gains.

```bash
python tests/ut/worker/test_asr_spec_decode_metadata_standalone.py
rg -n 'ASR_INPUT_PREPARE_GRAPH_READY|ASR_INPUT_PREPARE_GRAPH replays=' "$LOG"
```

The cumulative preparation patch includes this change. For a checkout already
at `fd89726b5`, use the focused incremental patch instead:

```bash
git apply --check examples/910b/asr_spec_decode_metadata_v023.patch
git apply examples/910b/asr_spec_decode_metadata_v023.patch
```

The incremental patch adds `asr_spec_decode_metadata.py` as well as updating the
runner and preparation helper. Copy all three files if deploying manually.
