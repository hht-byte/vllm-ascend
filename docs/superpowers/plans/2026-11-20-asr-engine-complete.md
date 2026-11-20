# Complete ASR Engine Delivery Implementation Plan

> **For agentic workers:** Use subagent-driven development for the independently
> owned components and a final whole-change review. Keep tests before changes.

**Goal:** Make 310P-graph contain the complete ASR external-draft/MTP runtime,
with its upstream vLLM scheduler prerequisite explicitly delivered.

**Architecture:** Preserve the current graph/preparation implementation. Restore
the validated base proposer and transport, overlay first-verify/rejoin semantics,
and register base/first-verify/MTP hooks after NPUModelRunner is fully imported.
Keep the shared worker's default path unchanged when ASR is not configured.

**Tech Stack:** Python, vLLM 0.23.0, vLLM-Ascend, TorchNPU 2.10/CANN 9.1.
The device checks ran on Atlas A3, SoC `Ascend910_9362`, CANN family
`Ascend910_93`; they do not establish 910B performance.

**Spec:** ../specs/2026-11-20-asr-engine-complete-design.md

## Global constraints

- Work in the existing isolated 310P-graph worktree; no streaming service edits.
- No new inference policy or graph math; retain the already validated payload.
- No production restart; preserve unrelated and uncommitted changes.
- The user authorized pushing the completed change after validation. Finish
  validation and final review before the coordinating agent commits and pushes.
- Business acceptance is Chinese-only and excludes evaluation with real MTP
  weights. Retain all MTP runtime code, hooks and tests.
- Version gates, defaults, import order and lifecycle remain explicit.

## Tasks

- [x] Recover the external-draft contracts and official upstream scheduler.
- [x] RED/GREEN: restore base transport/proposer, first-verify metadata/rejoin,
  strict input validation and per-request cleanup with idempotent installers.
- [x] RED/GREEN: integrate the fourteen MTP modules and worker hook; preserve
  disabled/default behavior and late worker registration.
- [x] RED/GREEN: first-verify ordinary/recompute scheduling with complete prompt
  and draft, budget/encoder/chunk fallback, and an upstream-only vLLM patch.
- [x] RED/GREEN: clean-tree import/registration contracts, package import closure,
  repeated installation and existing preparation/FULL behavior.
- [x] Run focused/broader relevant CPU tests, lint and fresh/reverse upstream
  patch checks.
- [x] Run isolated Atlas A3 standalone contracts and tiny real-NPU graph checks.
- [x] Complete the Chinese actual-engine business acceptance and record its
  observed results in `examples/910b/asr_engine_validation.md`.
- [x] Update engine-only installation documentation and patch compatibility;
  obtain an independent whole-change review.
- [x] Finish final review and prepare the validated delivery for the authorized push.

## Execution ledger

- The fourteen MTP modules and worker hook retain the validated payload's
  semantics. Ruff reformatted fourteen imported runtime files; all 27 staged
  Python files have identical ASTs before and after formatting, with hashes and
  AST evidence retained. Parity is AST equivalence rather than byte identity.
  Base transport was reconstructed from the documented contract
  and reference proposer transform; a custom constructor avoids upstream ngram
  JIT warmup before the runner has an input batch.
- RED/GREEN fixes include legacy post-bookkeeping context accounting, exact
  boolean marker validation, ordinary recompute behavior without the optional
  upstream helper, and preserving unmarked consumer metadata.
- Actual engine startup exposed the upstream custom-proposer factory's initial
  config-only construction before the runner-bound drafter is installed. A new
  regression failed before the constructor fix; the external-draft file then
  passed all 18 methods, including its normal command in the final local batch.
- Same CachedRequestState owns consumed/continuation state across temporary
  batch removal. Resident maps and owner references clear; a new object reusing
  the ID starts fresh. Independent review reproduced then verified these fixes.
- Final formatted-tree CPU runs: worker/preparation 62 (57 pass, 5 skipped),
  external drafts 18 pass, recompute scheduler 15 pass, existing MTP suite 219
  (199 pass, 20 skipped). Total 314 tests, 289 passed, 25 skipped; all 29 per-file
  processes exited successfully. This includes the added constructor regression.
- Device coverage contains 317 distinct methods: worker/preparation 62,
  attention query-start reuse 4, external drafts 17, recompute scheduler 15 and
  the MTP suite 219. All passed with zero skips after two standalone joint-Triton
  fixtures were corrected to perform the worker's device-property initialization.
  The affected files passed all 31 methods through their normal commands; the
  other 286 methods were unchanged. Initial logs retain three failure events
  across those two methods, alongside the passing rerun evidence.
- The upstream-only patch applies and reverses against official vLLM v0.23.0
  scheduler SHA256 9ae1491f57a6294e32416b93ef2ea68107fa0d65c3a4d7f745f1261012f735e9;
  restored bytes match exactly. Nine actual patched upstream source contracts
  also pass. Diff whitespace checks pass.
- Full repository pytest cannot run in the available local environment:
  `No module named pytest`; real vLLM/TorchNPU tests are not inferred from CPU.
- Final Ruff check/format passes for all 27 staged Python files. Forbidden-import,
  package-init, logger, long-function, boolean-context and five-file Markdownlint
  checks pass. The aggregate `format.sh ci` check could not complete because
  `pre-commit` is unavailable. Report that limit separately from the completed
  scoped checks.
- Chinese actual-engine correctness acceptance passed: 25 requests across
  ordinary, eager external-draft and native FULL modes had identical final
  greedy tokens. FULL executed 37 target replays from nine captured entries,
  including variable decode queries and mixed fresh prefill/decode. Exact
  internal-ID lifecycle checks and normal shutdown passed. Trained MTP weights
  were not loaded. These are correctness results, not a performance benchmark.
  No production service was changed.
- See `examples/910b/asr_engine_validation.md` for the sanitized validation record
  and the observed Chinese acceptance result.

## Review focus

- Default-off and unrelated custom classes retain the original drafter.
- Package import cannot read a partially initialized NPUModelRunner.
- Request transport rejects malformed drafts before mutating persistent state.
- First-verify uses complete candidates and cannot verify a partial prefill as
  a complete candidate; fallback preserves normal generation.
- Finished, removed, reordered and preempted requests do not leak draft state;
  rejoin and MTP consume only confirmed tokens and keep the drive out of drafts.
- Upstream patch is checked against a real v0.23 scheduler, not a synthetic
  fixture; fresh branch imports require no files from the service repository.
