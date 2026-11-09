# Online ASR MTP graph execution

**Goal:** Capture the sequential auxiliary blocks, shared LM head, greedy argmax,
and auxiliary cache writes in one replay, without changing target verification.

**Scope:** User requested implementation after selecting MTP graph execution in
the profiling discussion. Preserve existing unrelated work. Engine deployment
belongs to `310P-graph`; do not push or restart a service in this task.

**Architecture:** Keep the existing eager implementation as the reference and
fallback. Add an independent fixed-shape tensor program and NPU graph controller.
Use the existing arena with a bounded scratch suffix for masked scatter writes.
Slots, lengths, per-head valid prefixes, budgets, and tokens are device inputs;
no per-request Python slicing or per-head device-to-host transfer inside replay.
Capture only during startup memory profiling, before target KV sizing. Replay
uses the current stream and outputs outside the shared temporary graph pool.

**Configuration:** `asr_mtp.graph_mode=off/full`, default off. Exact batch sizes
1..8 (bounded by slots/runner capacity), query buckets 8/64, context capacity 128
(clamped to arena capacity), configurable with a bounded graph count. Exact
batch sizes avoid padding the expensive LM head. Uncovered query/context/batch
shapes and shortened maximum head budgets use eager, without online capture.
Explicit full-mode capture failures fail startup; a partially executed graph is
never retried eagerly against potentially mutated caches.

**Invariants:** Confirmed hidden rows end before the new drive token; cropped
per-head prefixes are authoritative. No target/audio forward, target KV change,
new scheduler delay, request-ID graph key, or graph per exact sequence length.
Only one batch D2H remains in the worker. Padded query writes go to unique scratch
locations, never live cache rows. All graph temporaries are internal; persistent
inputs, outputs, weights, and caches stay outside the shared graph pool.

## Tasks

- [x] Write RED CPU tensor-program parity tests against eager, covering mixed
  lengths/budgets, rejection, recomputation, reordering, reused slots and NaN
  padding. Add actual NPU capture/replay tests guarded by device availability.
- [x] Implement the fixed tensor program and optional arena scratch storage.
- [x] Write RED controller/startup/export tests; implement bounded pre-capture,
  replay selection, counters, explicit off mode and eager fallbacks.
- [x] Wire worker profile/proposal and service configuration, update the patch
  exporter and deployment documentation, generate a fresh install artifact.
- [x] Run scoped Torch tests, export/apply/upgrade checks, lint and diff checks;
  get an independent review. Device execution/performance remains unverified
  until the user runs the NPU tests and matched service A/B.

## Review focus

Graph-pool lifetime and stream ordering; invalid scatter never touching live
rows; no stale slot or rejected suffix reuse; varying starts at different head
depths; shortened budgets; metadata upload lifetime; first-profile capture and
no recapture during target dummy runs; exporter includes every new dependency.

Verification: see `2026-11-09-asr-mtp-graph-validation.md`; device acceptance remains pending.
