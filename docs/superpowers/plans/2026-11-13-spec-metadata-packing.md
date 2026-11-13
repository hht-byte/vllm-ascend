# Synchronous verifier metadata packing implementation plan

> Use superpowers:executing-plans and test-driven-development for implementation.

**Goal:** Remove the remaining verifier metadata host launches after ASR input
preparation replay, and expose replay counts without waiting for 1000 calls.

**Spec:** User requests continued optimization of slow operator dispatch between
MTP and decoder verification; exclude Conv2D/audio encoder for this change.

**Architecture:** Build the already CPU-known verifier indices and draft IDs on
CPU, using one persistent pinned buffer and one H2D. Keep the decoder, graph
buckets and sampler contract unchanged. Preserve original path when disabled or
when device-only previous sampled IDs are present. Add five-second traffic-driven
stats to the existing preparation graph counter.

**Tech stack:** vLLM 0.23, vLLM-Ascend 310P-graph, NumPy, Torch/NPU.

## Evidence and boundaries

Base engine fd89726b5, root a34486a41. The user confirms READY and replay=1.
Source logs only replay 1 and multiples of 1000; silence is not proof of misses.
The two Index/IndexCheck pairs match _calc_spec_decode_metadata's two device
indexing operations after five separately pinned uploads. Their elimination is
verifiable structurally. The profile table cannot establish the remaining wall
gap's duration. Do not claim measured speedup without NPU retesting.

Ruling: improve ownership of synchronous verifier metadata instead of adding
another graph/notification boundary. A second small replay is not justified.
The existing graph remains opt-in; pack_spec_metadata defaults true inside it
and can be disabled for an isolated A/B. No asynchronous/PCP support is added.
No target forward, new model graph capture, or device readback is introduced.

## Review focus

- Mixed prefill/decode, zero drafts and changing lengths preserve sampler IDs.
- Reordered/shrunk batches use current CPU input IDs, not cached request order.
- Existing GPU-only input repair must use the original implementation.
- Pinned host reuse waits for its copy event; metadata consumers are synchronous.
- Disabled and oversize preparation graphs retain correctness; packing is
  independent of the selected graph bucket.

## Steps

- [x] RED tests against metadata oracle and actual runner method; launch counts,
      dtype/shape, boundary/fallback, host reuse, interval logging, optional NPU.
- [x] Implement standalone metadata packer and minimal guarded runner hook;
      retain input preparation graph topology and existing service settings.
- [x] Related tests/lint, cumulative and incremental patch application checks,
      mirror docs/artifacts and final architectural review.

## Ledger

Existing isolated engine worktree reused; no new worktree, push or service restart.
Pre-flight: packer consumes authoritative input_ids.cpu after _prepare_input_ids;
consumer is SpecDecodeMetadata through synchronous sampling on the runner stream.
The fast path must refuse prev_sampled_token_ids, even with sync flags configured.

RED observed: five missing-packer failures, one missing timed-stats failure and
one unsupported-switch error. GREEN: five new CPU metadata tests pass (including
80 randomized batches and the actual runner fallback), one device test skips;
preparation suite has 16 CPU passes and one device skip. The two unchanged
Index calls are retained only in the original/GPU-repaired fallback. One packed
H2D is asserted at runtime by the CPU contract test. No NPU wall-time assertion.

Final verification: 87 CPU tests pass across worker standalone, 910B FULL graph
and token graph suites; three NPU tests skip. Scoped Ruff (including the runner),
Markdown, spelling and source/patch whitespace checks pass. Full Bash pre-commit
is unavailable on this Windows host; no claim of a full hardware CI run.
Cumulative patch from 2b3cb3c97 and incremental patch from fd89726b5 both apply,
match all three source files and reverse-check. Engine/root delivery mirrors
match. Fresh architectural review found no critical/important issue and
independently repeated the two changed standalone suites. NPU latency and
sampler-output A/B remain unverified. No push or service restart performed.

## Authorized delivery

User requested pushing this completed optimization. Engine commit `7e1be27cb`
targets `origin/310P-graph`; this documentation and both deployment artifacts
will be committed to `origin/codex/streaming-prespec-monitor`. Both remotes are
`https://github.com/hht-byte/vllm-ascend.git`. Remote branch heads matched the
implementation bases before delivery. Fresh validation again passed 87 CPU
tests with three NPU skips, scoped lint, mirror equality and patch checks.
No service restart is included.
