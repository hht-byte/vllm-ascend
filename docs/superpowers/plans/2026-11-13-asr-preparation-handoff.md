# ASR preparation handoff implementation plan

> Use superpowers:executing-plans and test-driven-development.

**Goal:** Reduce CPU dispatch between MTP and target verification without adding
another replay boundary.

**Spec:** User approved shared upload protection, final-padding preparation and
pure decode embedding coverage. Keep audio/prefill behavior, switch-off fallback,
minimal runner hooks, and no automatic push or server restart.

**Architecture:** Stage CPU metadata during _prepare_inputs, then replay once
when final FULL padding is known, before attention metadata construction. CPU
non-spec logits indices are also staged so no early GPU qsl consumer exists.
Preparation and verifier uploads share one event owner. Capture a text-embedding
variant in the same preparation graph pool and select it only for eligible pure
decode batches; retain the original multimodal preprocess for all others.

**Tech stack:** vLLM 0.23, vLLM-Ascend 310P-graph, Torch/NPU, NumPy.

## Constraints and evidence

Base engine 7e1be27cb; root bbabfd24c. Reuse the isolated engine checkout. The
profile shows CPU Event query/record costs and two fills after preparation.
Source confirms independent copy fences, later FIA qsl upload and padding fills,
and graph-external multimodal embedding. These are verified launch sources;
NPU wall-time gain remains unmeasured locally.

Keep accepted-token synchronization unchanged. Do not remove host lifetime
protection or assume a later sampler barrier covers every path. One shared event
is recorded after both uploads, queried once before any shared host buffer reuse.
No extra D2H, runtime capture, decoder change or target forward.

## Review focus

- Stage before graph replay must not expose stale qsl-derived logits indices.
- Decoder bucket may exceed the preparation bucket; preserve valid fallbacks.
- Graph fills must preserve live block rows and clear only actual dummy rows.
- Mixed prefill/audio, prompt embeds, GPU-repaired IDs, positional variants,
  LoRA and encoder connectors retain original preprocessing.
- Inactive slots, zero drafts, batch shrink/reorder and oversize fallback must
  not reuse a prior step's embedding/preparation completion marker.

## Tasks

- [x] RED: shared event ownership, deferred outputs, final padding, eligible
      embedding replay, fallback guards and actual runner hook contracts.
- [x] Implement helper lifecycle and guarded runner hooks; config switches
      fuse_handoff and capture_decode_embedding default on under opt-in graph.
- [x] Verify related suites and lint; export cumulative/incremental patches and
      mirrored docs; fresh architectural review.

## Ledger

Pre-flight: logits metadata consumes CPU input IDs; attention consumes final
qsl/slots only after replay. Pure decode requires all computed positions at or
past each prompt end, no encoder work, no prompt embeddings, standard positions,
no GPU input repair and no encoder connector. No fusion of actual audio work.

Completed implementation and local verification:

- Preparation and verifier metadata share one upload fence. Legacy immediate
  replay remains for the LM-head early logits consumer.
- Final FIA query padding and dummy block-row fills execute in preparation.
  Pure decode embedding selects a captured variant of the same replay; original
  audio/mixed/unsupported preprocessing remains available.
- Both switches default on only within the existing opt-in preparation graph.
  Disabling embedding omits its extra input storage/upload capacity.
- RED tests reproduced the missing stage/fence/embedding contracts, positional
  signature regression, early LM-head logits dependency, and unused embedding
  allocation. All passed after implementation.
- Final suites: worker standalone 46 tests (42 pass, 4 NPU skips); 910B graph
  tests 18 pass; token graph tests 38 pass. Total 98 CPU passes, 4 NPU skips.
- The optional NPU handoff test alternates 8/16/32 buckets and plain/embedding
  variants, checking IDs, embeddings, query padding and slot tails.
- Fresh architectural review found no critical or important issues. Device
  equivalence, shared graph-pool behavior and wall-time improvement still need
  910B validation. No local NPU or server restart was available/performed.
- Scoped Ruff passes excluding two unchanged UP040 aliases reproduced from
  engine HEAD 7e1be27cb. Markdown validation passes. Codespell is unavailable in
  this test environment; the full Bash CI script is also unavailable locally.
- Cumulative patch against 2b3cb3c97 and incremental handoff patch against
  7e1be27cb each pass clean-fixture apply/reverse checks and exact three-source
  equality. Root research patches/docs mirror engine examples.
- No commit or push requested for this implementation turn; changes are left
  reviewable in the existing engine worktree and root research delivery files.

Push follow-up: user explicitly requested pushing both delivery branches.
Fresh related-suite verification again passed 98 CPU tests with 4 NPU skips.
Fetched both remote branches and confirmed their pre-commit heads are unchanged.
