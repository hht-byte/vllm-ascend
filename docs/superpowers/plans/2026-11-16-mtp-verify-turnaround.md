# Post-MTP to verify turnaround

**Goal:** Reduce exposed CPU dispatch and preparation time between MTP completion
and the next target verification on 910B, with reproducible device evidence.

**Architecture:** Extend the existing opt-in synchronous preparation owner. The
CPU already owns positions and block tables: pack their resulting slots and
table snapshots into its persistent upload, and replay copies instead of a
general slot kernel and per-step clone/pin allocations. Reuse the device query
starts in target attention metadata once the preparation owner completed them.
Keep all unsupported/capture/fallback paths intact.

**Constraints:** Current user authorization covers implementation and isolated
server validation. Keep live services untouched. Preserve EOS, rejected tokens,
reordering, chunked prefill and graph padding. Timings/logs off for performance
runs. Commit only after repeatable gain on 910B; no push requested.

**Alternatives:** Retain GPU slot mapping (comparison baseline); CPU slot packing
with one upload (chosen for CPU-authoritative synchronous inputs); a new fused
device kernel (deferred because it retains a launch and needs another backend).

- [x] RED/GREEN tests for slots, new block allocation, reorder, shrinking batch,
  multiple groups, dummy padding and option-off legacy behavior.
- [x] Extend preparation packing and runner deferral; eliminate duplicate query
  starts upload on the completed target preparation path.
- [x] Local targeted tests, installed runner correctness, isolated 910B FULL
  engine simulation with matched requests and no stage logging.
- [x] Alternating baseline/candidate runs; report preparation and complete step
  timings, output equality, sampling and batch differences.
- [x] Review, export focused patch and commit only if the performance gate passes.

## Evidence ledger

- v14 stage logs reveal bookkeeping/parser typically 0.07–0.10ms, and only
  0.02–0.03ms of overlapped accounting. Neither alone explains the 3.5–4ms gap.
- Source confirms block-table clone/pin every step and a second query-start H2D
  in attention metadata construction after preparation graph replay.
- CPU slot packing passed real 910B graph tests. Four same-process ABBA cases
  (1/4/8 request decode and mixed prefill) reduce preparation wall approximately
  0.11–0.12ms. This is local preparation evidence, not the full bubble result.
- Runtime ASR embedding was excluded by mRoPE. The model config explicitly has
  mrope_section [24,20,20]. Pure token embedding is position independent; return
  the runner's original three-axis positions while capturing only embedding.
  CPU and NPU tests cover this and original audio fallback.
- Full ASR ABBA runs use physical 910B 1 after another experiment acquired 0.
  Prior 0-card runs are correctness/smoke only and excluded from performance.
- v2 full native ASR ABBA: exact reference parity. Matched profiled warm steps
  (42/variant) reduce total MTP-tail-to-verify median 5.04175 -> 4.600625ms.
  Preparation-to-verify improves 1.458875 -> 0.99175ms, but the first CPU
  turnaround does not improve (3.59475 -> 3.62475ms). Main request latency is
  noisy, including isolated candidate long tails; no general E2E gain claimed.
- Continue rather than committing v2 immediately: v3 also packs original CPU
  mRoPE values into the same upload and replays into the existing GPU position
  tensor. Fixed-capacity H2D bytes are retained; only the standalone upload is
  removed. 48 real NPU tests pass; independent correctness review found no issue.
- Final performance gate includes v3 vs frozen v2, longer full-case warmup and
  GC attribution. Short traces with record_shapes enabled identify location;
  also validate direction with record_shapes disabled before committing.
- Gate complete: record_shapes=False, 56 matched warm steps per variant,
  total MTP-tail-to-verify median 4.81275 -> 4.14825ms (13.81%). Every group
  improves; repeated prior ABBA supports the later 0.4–0.5ms improvement.
- Reverted v3 mRoPE upload packing: incremental gain was small and mixed. Keep
  v2 core code only. Final profiling/main-load instrumentation remains solely
  in isolated experiment tools, outside delivered code.
- Long-span diagnostics locate comparable seconds-long sampling spans in both
  baseline/candidate warmup, outside measured windows. No production E2E gain
  or resolution of the entire first CPU turnaround is claimed.
- Independent review verified cache/padding/pointer/lifetime semantics; found
  pruning recomputation guard, covered by a RED->GREEN fallback test and suite.
- Delivery committed on 310P-graph as 159589efb. Final selected source passes
  48 NPU tests and four attention-builder tests; focused patch fresh/reverse
  checks pass. Export and report are mirrored to the source branch. No push.
- User explicitly requests continued execution and measured benefit before
  committing. No further design approval is needed for these scoped changes.
