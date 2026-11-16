# Rejection to MTP buffer reuse

User approved the two buffer optimizations on 2026-11-16. Implement locally;
do not push or restart services. Preserve the existing isolated 310P-graph
delivery checkout and unrelated work.

## Contract

- Reuse a bounded pinned CPU buffer for synchronous speculative sampler output.
  One nonblocking D2H and one event wait precede CPU parsing. Delegate filtering
  to the original vLLM parser; preserve discard, placeholder and vocabulary
  semantics. Unsupported shapes/dtypes and logprobs use the original path.
- Enable this under the existing opt-in ASR preparation feature, with an
  independent `reuse_rejection_buffer` boolean for ablation.
- Reuse fixed pinned upload buffers for BQ/token MTP metadata and PA tables.
  A token program shares one upload fence across metadata and tables; unchanged
  tables still skip copying. Wait before overwriting an in-flight host buffer.
- Do not remove the required acceptance readback, alter EOS/rejection/graphs,
  or change the post-MTP draft readback. No runtime global stream synchronize.

## Steps and validation ledger

1. RED tests for result parsing, buffer identity, copy/event/read ordering,
   fallback, upload reuse, grouped copies, and unchanged PA tables.
2. Implement small helpers and minimal runner/graph hooks; run CPU parity and
   optional NPU tests. Use existing worktrees, no new implementation agents.
3. Export fresh and focused deployment patches, verify applying and source
   equivalence, update deployment docs and mirrored tests.
4. One fresh final review as required by executing-plans; fix important findings
   with RED/GREEN coverage. Report NPU timing as unverified.

No timing claim follows from CPU tests. Original profiling ranges contain
nested CPU events and must not be added as independent costs.

## Execution ledger

- RED: four upload tests failed on missing helper; three readback tests failed
  on missing implementation. Install-gate test then failed on missing install.
- GREEN: buffer/event tests pass; actual runner bookkeeping comparison preserves
  output/discard/CPU token state. Worker suite: 42 tests, 38 pass, 4 NPU skips.
- Graph/BQ/paged/export run: 76 tests, 65 pass, 10 NPU skips, one expected stale
  release-patch failure before regenerating artifacts.
- Fresh MTP, b63 graph cumulative, 55d+6bf focused upgrade and 2b3 preparation
  patch application verified in isolated minimal fixtures. Canonical source
  equality and focused reverse check pass. Prerequisite external-draft registry
  imports are supplied in fixtures; these checks do not start an NPU engine.
- Final regression: research `test_asr_mtp*.py` 176 tests, 143 pass and 33
  device/reference-dependency skips; exporter/hook suite 18 tests, 17 pass and
  one skip; worker suite 42 tests, 38 pass and four device skips. No failures.
- Installed fresh-patch graph/staging/paged suite: 71 tests, 61 pass and ten
  device skips. Ruff, compilation and whitespace checks pass. Mirror test ASTs
  equal canonical tests while retaining existing engine formatting.
- Final independent architectural review: no Critical/Important/Minor findings.
  Final: Ruling: post-MTP `tolist()` remains unchanged because this approved
  patch targets rejection-to-MTP; the later readback cost remains measurable.
  Final: Ruling: concurrent/cross-stream graph execution remains unsupported,
  consistent with existing serial runner-stream contract; enabling it without
  a separate dependency design could corrupt metadata/cache state.
- No commits, push or service restart. User then proposed device-resident
  rejection-to-MTP handoff; investigate separately without discarding this work.
- Subsequent user instruction authorized pushing the completed buffer reuse
  change to the two existing branches. Pre-push rerun: 89 graph/staging/export/
  hook tests (78 pass, 11 skips), 42 worker tests (38 pass, four device skips).
  Remote tips still match 6bf62166a and 55d67eb75. Device-resident handoff is
  not part of this delivery; service restart remains outside this operation.
