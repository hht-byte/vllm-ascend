# Synchronous ASR input metadata packing

User approved implementation and push. Execute inline with TDD.

## Design

Opt-in `additional_config.asr_sync_input_packing=True` (sibling of `asr_mtp`).
Pack CPU computed counts, sequence lengths and positions into one pinned buffer
and transfer into stable contiguous device views. Replace four H2D submissions
and redundant device index/cast/add/fill. Leave slot mapping, encoder, embedding,
FULL graph and the MTP-to-scheduler round trip unchanged. No extra graph required.

Require ASR MTP, synchronous scheduling/speculation, TP/PP/DP/PCP/DCP=1,
no hybrid/GDN/sparse/compressed model. Default False; unsupported ON fails early.
Allocate before capture. Protect pinned host reuse with a copy-completion event;
no device-wide synchronization. Refresh sequence padding and copy only active
positions plus fixed request fields. Keep fallback behavior unchanged.

## Tasks

- [x] RED: mixed lengths, rejection/reorder/shrink, padding, stable pointers,
      host-buffer ownership, config gates, real runner fast/fallback wiring.
- [x] Helper and minimal runner hooks in 310P-graph.
- [x] NPU graph equivalence test and A/B microbenchmark.
- [x] CPU verification, lint, architectural review; NPU results unavailable locally.
- [x] Document switch, signed commits and push engine and root documentation.

Scope: the profile's 21 copies are not all attributed to this block. Do not claim
all bubbles, especially the separate 2 ms handoff gap, are eliminated.

## Validation and review

- New suite: 10 CPU tests pass, 1 NPU replay test skipped (no Ascend device).
- Existing 910B graph suites: 18 pass; token-graph suites: 38 pass.
- Tests execute the actual changed runner preparation region and constructor
  installation call from AST, not a duplicate implementation.
- Fixed override guard after review: 310P's independent `_prepare_inputs` must
  fail enabled startup instead of falsely reporting this optimization active.
- Focused patch applies to prior engine HEAD, matches both source files and
  reverse-checks. Initial fixture check exposed CRLF patch text; exported patch
  now uses LF explicitly.
- Standalone benchmark CPU smoke passes; its CPU timings are not NPU evidence.
- Ruff with the engine repository config passes on all changed Python files;
  Markdown and spelling checks pass.
- No online restart or device profiling performed. Full vLLM/NPU suite unavailable
  on this Windows CPU host. Graph compatibility/device latency remains to test.
- Architectural review found no remaining blocking issue. Default remains off.
