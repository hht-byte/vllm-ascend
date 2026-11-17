# Reduce synchronous ASR CPU preparation and redundant waits

**Goal:** Reduce the remaining approximately 3.5ms host turnaround before the
preparation replay, with matched 910B evidence before committing.

**Architecture:** Keep the synchronous engine and existing captured graphs.
Use startup-validated dense ASR conditions to avoid an event wait with no
producer and unused decode-draft metadata. Explicitly acknowledge an upload
generation only after device-rejection readback has completed, preserving
the upload fence for unsupported requests, exceptions and standalone replay.
If needed, replace redundant CPU Torch wrappers with equivalent NumPy work.

**Baseline:** engine 159589efb; source 30bc37ed0. Reuse the previous public-input
native ASR simulation, fixed weights and graph settings. Do not touch existing
services or push. User explicitly requested continued scoped implementation.

**Validation:** Same engine/owner/memory layout, switch baseline/candidate CPU
methods only between drained phases. ABBA, full-case warmup, exact output and
cache agreement, main timing logs off. Separate record_shapes=False traces and
Python-only shape ring establish the device boundary. No concurrent NPU tests
during performance measurement. Commit only repeatable improvements.

- [x] RED/GREEN: skip unproduced accepted event only on validated dense path;
  preserve CPU counts, and leave unsupported/legacy waits and metadata intact.
- [x] RED/GREEN: acknowledge upload only after completed readback; stale upload
  generations, fallback, exception and standalone replay retain protection.
- [x] Same-process single-variable performance checks; remove losing variants.
- [x] Full ASR/NPU correctness and low-overhead matched boundary comparison.
- [x] Review, focused export/installation check, results and gated commit.

## Review focus

Keep hybrid/async/CP consumers intact; retain actual accepted counts; do not
trust an old completion to clear a new upload; preserve the shared verifier
uploader's protection; cover cold, mixed prefill, EOS, bucket miss and fallback.

## Execution ledger

- Same-engine API-count smoke proves accepted synchronize/upload query/unused
  draft H2D removed; original upload record remains once per generation.
- Selected safe v1 combined variant. NumPy sequence-length experiment had no
  independent incremental improvement and was reverted. No lazy-record change.
- Main ABBA: both order-balanced pairs improve every matched shape's CPU prepare,
  first gap and total gap. Pooled CPU prepare -18.30%, first gap -8.62%, total
  gap -6.74%; complete-request means differ only 0.35% with mixed pair directions.
- 864 main +128 profiled requests match reference text/token IDs. Same engine,
  owner, graphs and GPU pointers throughout. Record-shapes off; no concurrent
  NPU correctness runs during timing. Service exits cleanly before unit tests.
- Final real-NPU tests: worker 55/55, MTP handoff 50/50. Independent review checks
  actual upstream producer and dense metadata consumers; no blocking findings.
- Focused full delta, hook-only upgrade and cumulative hook creation each pass
  fresh installation, exact-source comparison and reverse application.
- Engine delivery committed on `310P-graph` as `22137cfa0` after the 910B gate.
  Source hook, exports and evidence are mirrored in the source branch. No push
  or production restart was performed.
