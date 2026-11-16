# MTP post-handoff P0 implementation and validation

> **For agentic workers:** Use superpowers:executing-plans. Do not commit until
> server comparison establishes a repeatable improvement, per the user's request.

**Goal:** Reduce CPU work exposed after device-rejection MTP while preserving
exact outputs and cache state; add optional host stage attribution.

**Architecture:** Keep the existing monkey patch and synchronous engine API.
Late proposal consumes pending device results without rebuilding full token
histories or redundant fresh-position sums. Split combined readback into submit
and finish; execute acceptance-independent accounting between them. Keep one
serial buffer and one completion event. Reuse `profile_stages=False` for timing.

**Tech stack:** Python, Torch 2.10 / TorchNPU 2.10, vLLM 0.23, Ascend 910B.
**Spec:** User request in this chat: both P0 optimizations, stage timing, minimal
intrusion/overhead, server verification before committing; continue if no gain.

## Constraints and verification

- Preserve existing uncommitted activation diagnostics and all unrelated edits.
- CPU sample mirrors, EOS/min/max tokens, legacy rollback priority, chunked
  prefill and optimistic-cache truncation must remain unchanged.
- No extra D2H, device synchronization, per-head event or default-on timings.
- Benchmark in isolated server files/device; do not restart existing services.
- Source branch codex/streaming-prespec-monitor; delivery branch 310P-graph.
- No commit/push before measured benefit. A synthetic kernel test proves
  correctness or local cost only, not real ASR service latency.

## Tasks

- [ ] Snapshot current code as common comparison baseline; inspect available
  checkpoint, ASR service and workload on 9.15.147.2.
- [ ] RED: prevent full-history conversion on pending consumption, preserve
  tail verification and reference output/cache equality.
- [ ] Implement pending-specific CPU reductions in research/asr_mtp_worker.py.
- [ ] RED: enforce copy/event submission -> independent CPU accounting -> wait
  -> CPU result consumption; keep buffer lifetime and exception handling safe.
- [ ] Implement split readback in research/asr_mtp_device_rejection.py.
- [ ] Add opt-in host stage spans in existing worker/hook; tests verify default
  path does not call clocks, and timing fields cover distinct host phases.
- [ ] Run related tests/lint; export and validate installation patches.
- [ ] NPU correctness plus alternating baseline/optimized same-workload runs;
  record outputs, model/load configuration, repetitions, latency and variance.
- [ ] If gain is unclear, locate next exposed CPU cost and repeat. Request
  missing actual workload/weights if absent; never claim synthetic service gain.
- [ ] Fresh review; commit only when the user's performance gate is satisfied.

## Execution ledger

- Initial state: prior activation-check/log changes remain uncommitted.
- Server reachable, eight 910B3 cards currently show no running NPU processes.
- User authorized an independent blue-zone simulation because no service load
  command is available. Use isolated files/NPU, and label this as handoff-level
  evidence, not full-service latency.
- P0 pending suffix and split readback tests went RED then GREEN. Optional hook
  timing also went RED then GREEN; default clock-disabled behavior is tested.
- Initial cross-process A/B showed only tiny local savings within process drift.
  Ruling: extend CPU bookkeeping reduction to an exact-tensor, scoped CPU parser
  at the existing buffer seam, preserving upstream filtering and restoring the
  buffer on exceptions. Reason: user requires continuing until measured benefit.
  Cost if wrong: token filtering differs; installed-vLLM parity tests cover it.
- Switched validation to in-process ABBA on the same captured graphs; prelude
  readback compatibility uses the baseline readback method for baseline only.
  Inputs and reset work are identical and resets are outside timed regions.
- NPU eager/graph parity, mixed prefill/warm and installed parser tests passed.
  Initial same-process 3-head cases show approximately 2–3% handoff wall savings;
  real-size/checkpoint and delivery-bookkeeping repeats are in progress.
- Final: actual checkpoint, native RoPE and target embedding/LM-head weights
  passed all paired output checks. Final delivery-bookkeeping medians improve
  0.054–0.140 ms (0.7–2.2%) across all ten head/phase/batch cases. Earlier repeats
  show the same direction. This satisfies the independent simulation gate,
  not a claim that the entire production 3.24 ms gap was fixed.
- Final verification: local210 tests173pass37skip; focused30 tests29pass1skip;
  NPU4tests4pass; five separate short CPU+NPU traces produced. No live service
  restarted. Patch fresh/upgrade/reverse and hook-only repair checks passed.
- Fresh review: post_handoff_review found no actionable correctness or
  architecture defect; independently ran48 targeted tests (4 dependency skips).
- Performance gate satisfied for this scoped handoff improvement. Commit source
  and310P delivery artifacts; do not push without a new push request.
