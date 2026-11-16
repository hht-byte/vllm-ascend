# Cold and prefill device MTP handoff

Continue the uncommitted device-rejection implementation; preserve all existing
changes. User explicitly requests cold and prefill support, including their
coexistence with warm requests. Use the existing branches/worktree; no push or
service restart requested.

## Design

Separate CPU-known target execution `[start,start+count)` from the sampled-token
append point C. Completed prefill/decode satisfies start+count=C+draft_count;
true MTP end is C+number_of_samples-1. Intermediate chunked prefill ends before C,
has a discarded sample row, and only stores target hidden. Last prefill can then
initialize the auxiliary heads without copying samples to the host first.

Allocate missing request slots transactionally after all CPU capacity checks.
Keep zero auxiliary validity for new requests; select cold/warm head policy per
request. Permit zero-limit rows in the device-only graph staging API for partial
prefill and a pending legacy rollback. These rows do not execute meaningful MTP
queries and do not acquire fake acceptance rounds. Mixed rows keep original order.

Single-token non-spec prefill must also enter the early hook. Return the full
fixed-width CPU sample block padded with -1 so original bookkeeping uses its CPU
rejection parser instead of its device-only `_to_list` event path; leave the real
sampler output unchanged. No new target runner monkey patch is needed.

Preserve EOS/min/max limits, legacy external draft priority, partial hidden
continuity, normal fallback, graph memory accounting and a single combined D2H.
Use the same graph pool, token buckets and opt-in flag. Unsupported geometry,
preemption, capacity, sampling/logprobs continue to fall back.

## Tasks / validation

1. Write failing CPU parity tests for full cold prefill, first-prefill external
   verification, cold+warm batches, partial then final prefill, legacy rollback,
   transaction rollback and single-column bookkeeping dispatch. Implement generic
   execution offsets, inactive staging and cold slot initialization.
2. Add NPU-gated prefill/mixed replay coverage; run all related CPU/hook tests.
   Update documentation, export fresh/focused/historical patches, verify installs.
3. Fresh architectural review, one regression-test fix pass, final verification.

## Ledger

- Starting point: warm handoff from the prior task is verified and uncommitted.
  Root e55795326, engine delivery 92c64fb2d remain the patch distribution bases.
- Plan interfaces: prelude uses independent target offsets and sample capacities;
  late worker commits actual end and respects per-row cached-only limits. CPU
  parser receives padded sample columns, while sampler consumer retains original.
- No further user approval needed: this implements the explicit cold/prefill
  extension to the previously authorized device-side handoff.
- Task 1 complete: five new cold/prefill parity tests failed on the warm-only
  guards, then passed with independent target offsets and sample capacities.
  Non-spec hook test separately failed before removing its spec-only gate.
  Expanded coverage includes mixed inactive prefill, distinct cold/warm head
  budgets, EOS/output limits, cache/capture misses and transactional allocation.
- Task 2: exports validated for fresh/current/historical upgrades and installed
  source equality. User identified an accessible 910B server; existing SSH
  identity was used to run isolated tests on idle physical device 6, in /tmp,
  without modifying service source/configuration or restarting it.
- Hardware evidence: 910B3, TorchNPU 2.10.0.post4, Torch 2.10.0+cpu, Python
  3.12.13, CANN 9.1.0, vLLM 0.23.0+empty / vLLM-Ascend 0.23.0.post1.
  Two actual prelude + PA/FIA graph tests and the installed
  vLLM CPU-parser contract test passed (3 tests, no skips). Local result log:
  `.tmp/device-prefill-npu.log`. Small synthetic heads, not full ASR weights;
  full-engine/real-load output and latency remain unverified.
- Task 3 review: cold_prefill_handoff_review found no concrete correctness or
  architectural findings; it also checked 30 randomized cold-to-warm sequences
  for proposal/hidden/valid-key parity. Its lack-of-NPU note was superseded by
  the remote hardware execution above. No production edits followed review.
- Final verification: 201 MTP tests (165 passed, 36 environment-dependent
  skips), 21 hook/export tests (20 passed, 1 skip). Ruff, py_compile and both
  worktree diff checks passed. Fresh/current/historical exports passed apply,
  reverse-check and canonical source equality. Final installed-fixture suite:
  75 tests, 63 passed, 12 skips. Remote 3 tests passed without skips.
- Work retained uncommitted in the existing two worktrees; no push and no
  service restart. Remote test process exited; physical card 6 has no running
  process afterward. Full real-ASR workload benchmarking is not claimed.
- User subsequently requested push. Pre-push checks confirmed unchanged remote
  branch bases e55795326 (codex/streaming-prespec-monitor) and 92c64fb2d
  (310P-graph), and reran the 201 MTP / 21 hook-export tests plus Ruff and diff
  checks successfully. Deliver source and engine patch assets on their existing
  branches; keep unrelated .tmp contents outside both commits.
