# ASR input preparation graph

User confirmed slow host dispatch between MTP and decoder replay and approved
implementation. Execute inline with TDD and a final architectural review.

## Scope and design

Opt-in `additional_config.asr_input_prepare_graph` with enabled and token_sizes.
Keep the existing decoder FULL graph and CPU scheduler unchanged. Require sync,
single-device dense ASR MTP and the standard runner preparation method. Reject
simultaneous asr_sync_input_packing to avoid conflicting buffer ownership.

One pinned metadata upload feeds a separately captured preparation graph. Replay
copies positions, computed/sequence lengths, query starts and discard metadata,
fills accepted counts and padding, then computes slot mappings. CPU already has
authoritative positions/lengths; do not recalculate them with device indexing.
Capture by token capacity only. Private slot query starts repeat actual token
count for inactive requests, while attention query starts retain -1 padding.
Always clear slot padding, including actual_tokens:bucket, after shrink/reorder.

Capture once after final KV tables exist, in capture_model; return extra reserved
bytes with decoder graph memory accounting. Fixed shared buffers and one shared
pool, sequential default-stream replay only. No runtime capture or synchronization
for device metadata. Protect pinned host reuse with a copy event. Oversize batches
and pre-capture profiling use the original path. Unsupported configurations fail
early. Audio/embedding, spec sampler metadata and the earlier 2 ms handoff stay
outside this first preparation graph.

## Steps

- [x] RED: dynamic lengths, reorder/shrink, padding and slot mapping, stable
      buffers, config guards, capture lifecycle and actual runner hook tests.
- [x] Implement isolated helper and minimal runner hooks, device replay test.
- [x] Verify related tests, lint, patch apply/export and usage documentation.
- [x] Fresh review, fix any blocking findings and report device validation gap.

## Decisions and validation ledger

Base engine: 2b3cb3c97, branch 310P-graph, existing isolated delivery checkout.
Base root: 2bd577d26, branch codex/streaming-prespec-monitor.
Ruling: use already-computed CPU positions and sequence lengths rather than redo
index/add in the preparation graph; fewer kernels and same sync contract.
No NPU on this host. CPU equivalence is necessary, not proof of NPU speedup.
No push or service restart in this implementation task.

RED observed: five missing-helper failures, then three actual-runner wiring
failures. GREEN: all eight CPU tests pass; old packing ten pass, one device skip.
Graph includes discard metadata to consolidate two additional small uploads.
No intermediate kernel results escape shared graph pool; final buffers remain
runner-owned and the private slot qsl is separate from FIA's mutable qsl output.

## Final verification

- Final related suites: 76 CPU tests pass, 2 device-only tests skipped.
- New preparation suite executes actual runner metadata/copy/capture hooks.
- Added production-layout coverage: attention qsl is R+2, slot mapping int32
  includes untouched extra draft capacity, buckets use scheduler token limit
  rather than padded positions allocation. Regression tests failed before fixes.
- Ruff, Markdown, spelling and git diff checks pass. Focused patch applies to
  engine base HEAD, matches both source files and reverse-checks successfully.
- Fresh architectural review: no remaining critical or important findings.
- Full pytest invocation unavailable: local environment has no pytest module;
  also no installed vLLM/NPU runtime. Only listed standalone suites ran.
- Real NPU graph capture, memory and wall-time gains remain unverified. No claim
  that all 6.4 ms bubbles or the separate 2 ms handoff are eliminated.
- Implementation completed before the user requested pushing both branches.
  No service restart performed.

## Authorized delivery

User requested push after reviewing the implementation summary. Engine target:
`origin/310P-graph`; documentation/patch target:
`origin/codex/streaming-prespec-monitor`. Both use the previously authorized
`https://github.com/hht-byte/vllm-ascend.git` remote.

Pre-commit verification repeated: 76 CPU passes and 2 device skips; focused Ruff
and Markdown checks pass. The repository-wide `bash format.sh ci` command is
unavailable because this Windows environment has no Bash on PATH. Unified patch
context contains mandatory leading spaces on blank context lines; check source
whitespace normally, and validate patch whitespace with blank-at-eol disabled
alongside forward/reverse application checks. No source changes made at delivery.

Engine delivery commit: `0343e34b8`, pushed to `310P-graph`.
