# Packed MTP token graph implementation plan

> **For agentic workers:** Execute with TDD and independent code review. The user approved this design and requested implementation directly.

**Goal:** Combine mixed fresh-query lengths from one target step into MTP graphs selected by total token capacity, with buckets 8, 32, 64, 128, 256 and 512.

**Architecture:** Add an opt-in `graph_token_sizes` configuration alongside the existing rectangular graph path. Each auxiliary head packs its active requests into `[T,H]`; request boundaries, anchor positions, cache slots and per-head fresh spans are metadata. Use a single segmented TND attention kernel per head with graph task updates for changing cumulative lengths. Keep the existing arena and row cache updates. The graph key is token capacity alone; request count and historical context have fixed configured upper bounds.

**Approved boundary:** LM head uses fixed request slots, never all T token rows. Padding those request slots adds compute and must be measured. Target decoder, scheduler and service factory signatures are unchanged. No KV-update/attention-input fusion is included.

**Tech stack:** Python, PyTorch, torch-npu FIA TND/task-update, existing optional vLLM-Ascend v0.23 patch exporter.

## Constraints and review focus

- MTP total fresh tokens are independent of the target FULL graph's padded token capacity and of historical context length.
- Per-head required tokens are summed over active requests; choose capacity covering the maximum of these sums, not the sum over all heads.
- No cross-request attention, invalid KV/hidden reuse, per-index host synchronization, or runtime graph capture.
- Deployment correction: each logical bucket reserves one extra physical Q row; a positive dummy Q owns the complete remaining KV tail. Final cumulative KV must equal physical KV T. See `2026-11-10-mtp-tnd-kv-coverage.md`.
- Preserve EOS/budget fallback, rejection crops, row reordering, slot reuse, row alignment and scratch isolation.
- Fixed request-slot LM output may add padding; mask does not eliminate matrix computation.
- Keep existing `(B,Q)` settings operational when token buckets are absent. Token buckets take precedence when both settings are present.
- No push or service restart in this implementation turn. CPU tests do not establish 910B acceptance.

## Task 1: Packed attention backend

Files: new `research/asr_mtp_token_attention.py` and its tests.

- [x] Add RED contract tests for staged cumulative lengths, packed attention isolation and task-update lifecycle.
- [x] Implement CPU segmented reference and NPU FIA out API with startup workspace allocation, capture task handles and replay-time updates.
- [x] Verify the backend contracts and retain a real NPU capture/replay test.

## Task 2: Packed tensor program and graph manager

Files: new `research/asr_mtp_token_graph.py` and its tests.

- [x] Add RED packed parity, variable request count, capacity, rollback and padding tests.
- [x] Implement per-head packed metadata, row cache reads/writes, fixed request-slot logits and final-head dead-write omission.
- [x] Capture one program per token bucket; stage/update/replay and commit cache lengths without runtime capture.
- [x] Verify multi-round outputs and live cache parity against the existing eager heads.

## Task 3: Integration and delivery

Files: `research/asr_mtp_graph.py`, `asr_mtp_worker.py`, exporter, related tests, engine packaging and docs.

- [x] Add RED token-option and same-step mixed-query batching tests.
- [x] Validate token buckets and choose packed graphs; split only at total-token/request limits, preserving eager fallbacks.
- [x] Update module export, fresh/cumulative patches, focused token-graph upgrade, mirrored tests and configuration documentation.
- [x] Verify patch application/source parity/idempotence; run related and research tests, lint and independent review.

## Evidence and decisions

- Baseline source commit: `c5b02c9d8`; engine package commit: `139d3e42a`.
- Existing unrelated research changes are preserved; use the existing engine delivery checkout.
- Token mode is enabled by setting `graph_token_sizes` with `graph_mode=full`; existing deployments keep their previous graph layout until configured.
- RED evidence: backend/program initially failed because the packed modules were missing; token option/packing assertions and checkpoint factory wiring failed before integration. Logs: `.tmp/mtp-test-env/mtp-token-program-red.log`, `mtp-token-integration-red.log`, `mtp-token-export-red.log`.
- Related suite: 122 tests, 119 passed and three real NPU tests skipped. Log: `.tmp/mtp-test-env/mtp-token-related.log`.
- Installed focused-patch suite: 46 tests, 43 passed and three real NPU tests skipped. Mirrored tests have identical ASTs. Log: `.tmp/mtp-test-env/mtp-token-installed.log`.
- Full research discovery: 288 tests, the same 19 errors and one failure as the row-cache baseline, three NPU skips, no new failures. Log: `.tmp/mtp-test-env/mtp-token-full.log`.
- Fresh, cumulative graph and focused token upgrades passed check/apply, compilation, six-module source parity, empty re-export and reverse-check.
- Ruff, Markdownlint and scoped whitespace checks passed. Independent review found no actionable issue and additionally passed 300 randomized CPU multi-round parity cases.
- Ruling: token mode uses a fixed request-slot LM head and per-head FIA task updates, accepting bounded logits padding and host rebinding costs to retain a single token-keyed replay. These costs require device measurement; no dynamic GEMM, per-request NPU attention loop or target runner rewrite was introduced.
- Actual 910B capture/replay, numeric acceptance and performance remain device-validation work; no service was restarted or branch pushed in this turn.
