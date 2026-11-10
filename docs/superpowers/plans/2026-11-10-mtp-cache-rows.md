# MTP row cache updates implementation plan

> **For agentic workers:** Use superpowers:executing-plans for implementation and a fresh code review before completion.

**Goal:** Reduce MTP graph cache indexing overhead while preserving proposed tokens, live KV state, rejection handling, and request isolation.

**Approved scope:** The user explicitly approved items 1 and 2 of the preceding analysis for direct implementation. Item 3 (fusing KV updates with attention input construction) is deferred.

**Architecture:** Preserve the existing arena layout and graph keys. Address aligned rows of `gcd(hidden_size, head_dim)` elements, using 2D row `index_select` for reads. On NPU use the explicit in-place ScatterNdUpdate device API for row writes; use CPU `index_copy_` for parity tests. Keep padding writes in distinct scratch rows. Omit only the configured final head's persistent hidden write and final local sequence update.

**Tech stack:** Python, PyTorch, torch-npu, existing vLLM-Ascend v0.23 optional patch exporter.

## Constraints and review focus

- No changes to target forward, scheduling, service flags, graph shapes, head count, or KV attention assembly.
- No per-index device-to-host transfers, runtime captures, or cache allocation in replay.
- Preserve mixed lengths, changing budgets, rejection crops, noncontiguous/reordered slots, and slot reuse.
- Row alignment must support nonstandard geometry and a scratch suffix with unused unaligned trailing elements.
- NPU row updates must not silently use the legacy per-index `index_copy_` fallback.
- CPU parity is not proof of NPU graph compatibility or performance.
- Preserve unrelated dirty files; do not push or restart services.

## Task 1: Row cache addressing and dead writes

- [x] Add regression tests for real tensor indexing shapes and unused final-head writes; observe RED.
- [x] Replace scalar cache addressing with aligned row indices and explicit NPU row updates.
- [x] Remove final-head hidden and local sequence writes; keep all KV and valid bookkeeping.
- [x] Verify multi-round parity, graph/eager transitions, padding, budget growth, and unusual alignment.

## Task 2: Deliverable patches and verification

- [x] Generate a focused upgrade for an installed diagnostics-era MTP graph implementation.
- [x] Regenerate fresh and cumulative graph patches and mirror graph tests in the existing engine worktree.
- [x] Document installation choices, expected operator changes, and device validation limits.
- [x] Verify patch application, compilation, source parity, and idempotent re-export.
- [x] Run related tests, research-suite baseline comparison, formatting, and an independent code review.

## Execution evidence

- Baseline: root HEAD `3b7a680ed`; existing unrelated research changes are excluded.
- The existing engine checkout is reused for delivery files; no new worktree is created.
- RED: 18 graph tests, two expected failures (scalar cache addressing and final-head hidden writes), one NPU skip.
- GREEN: 94 related tests, 93 passed and one actual NPU replay skipped. Log: `.tmp/mtp-test-env/mtp-cache-rows-related.log`.
- Full research discovery: 260 tests, 19 errors, one failure and one NPU skip. Failure identities exactly match `.tmp/mtp-test-env/mtp-context-diagnostics-full.log`; no new failure. Current log: `.tmp/mtp-test-env/mtp-cache-rows-full.log`.
- Focused, fresh and cumulative graph patches each passed check/apply, compile, installed-source parity, empty re-export and reverse-check.
- Independent review found no actionable issue and passed an additional 16 CPU geometry cases. NPU capture/replay and latency remain unmeasured on this host.
- No scope rulings or deferred review findings; KV/attention fusion remains outside the approved scope.
