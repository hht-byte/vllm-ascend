# Device rejection to MTP handoff

User request: continue implementation, minimize CPU readback and waiting. Preserve
FIA FULL replay; do not attribute the pre-MTP gap to attention execution.

## Specification and constraints

Add an opt-in `additional_config.asr_mtp.device_rejection` path for synchronous,
greedy warm token-graph batches. Launch MTP before reading target rejection output
on the host. Read target samples and MTP proposals together once afterwards. Keep
CPU scheduler/bookkeeping semantics, original cold/unsupported fallback, EOS and
length handling, acceptance attribution, and request-lifetime cleanup.

FIA Q/KV lengths remain CPU metadata. Use the scheduled upper bound Emax and
conservative per-head cache start S[d] <= C-d (C is confirmed length before this
sample). Device rejection produces true end E=C+n-1, sequence and final-row
indices. Computing the extra tail is safe only because causal attention prevents
future positions from affecting earlier outputs. Commit CPU cache validity to E,
never Emax. All speculative tail data must be finite. No runtime graph capture,
no graph-per-acceptance variant, no target/audio re-forward, no default behavior
change. Bound memory; include startup capture allocations in graph accounting.

Initial scope: a whole eligible warm batch fitting one token graph and one captured
head depth. Mixed cold/disabled/preempted/unsupported/oversize batches use the
existing path. This coverage limit and extra optimistic work are deliberate costs
to be measured on 910B, not assumed speedups.

## Tasks

1. RED: CPU parity tests for zero/partial/full acceptance, reorder, cached heads,
   rejected tail poison, EOS/limits; implement stable device metadata prelude and
   token-graph selection/capture reuse. No host reads in prelude.
2. RED: worker/hook tests for ordering, one readback, pending result consumption,
   opt-out/unsupported fallback, cleanup and CPU mirror parity. Implement opt-in
   early bookkeeping wrapper and reuse existing finalization semantics.
3. Configuration, documentation, cumulative/focused patch exports and installation
   verification; run related tests, lint and one fresh architectural review.

## Validation

Python: `.tmp/input-prep-env/Scripts/python.exe`; `PYTHONPATH=research`.
Run focused RED/GREEN tests, then research `test_asr_mtp*.py`, patch/export tests,
and installed-fixture imports/CPU tests. Add NPU capture/replay parity test gated
on torch_npu. NPU profiling unavailable locally, explicitly report that limit.

## Ledger

- Base: canonical e55795326, engine 92c64fb2d; clean tracked state.
- Existing engine worktree on 310P-graph retained; canonical branch retained.
- No push/restart for this task. Preserve unrelated files and scratch checkouts.
- Preflight: prelude writes stable program metadata; attention update must precede
  prelude replay; CPU validity update must happen after combined readback. Original
  late proposal hook consumes pending result without launching a second proposal.
- Task 1: RED missing prelude module; GREEN zero/partial/full CPU parity and
  EOS/budget handling. Causal prefix values match the original exact-length path.
- Task 2: RED missing worker keyword and missing early hook; GREEN 9 CPU cases,
  including reorder, cached KV, terminal rows, min_tokens, stale padded columns,
  fallback and lifetime guards; 12 hook cases pass. Small prelude reuses the
  existing head graph. Original bookkeeping receives a shallow sampler copy with
  CPU samples after the combined fence. No runner source edit.
- Additional hot-path test RED on eager fill/zero, GREEN with scheduled-length
  masks. Prelude CPU metadata/addresses share one pinned H2D block. Completion
  acknowledges both upload fences, avoiding their next redundant event query.
- Related research suite: 185 tests, 151 pass / 34 environment skips before the
  final stale-column case. NPU-gated actual capture test added; no NPU available.
- Task 3: fresh, current-version focused upgrade, and historical graph upgrade
  patches apply/reverse-check; installed module text equals canonical source.
  Installed-fixture tests: 50 pass / 10 environment skips before final counter fix.
- Final architectural review: one P2, execution counters omitted budget-cut heads
  and terminal rows. Both reproduced RED (45 executed vs 34 or 0 reported), fixed
  by counting planned work once at early execution, independently of cache commit.
  Acceptance attribution uses planned per-request work. Both regressions GREEN.
- Review found no further supported-path token/cache correctness defect. All -1
  sampler rows were considered a defensive case outside the supported greedy
  sampler contract (normal discard/invalid routes are excluded), not a demonstrated
  runtime blocker. NPU capture/FIA numerical parity, pool behavior, memory and
  latency remain hardware validation items, explicitly documented.
- Final verification complete: research MTP suite 187 tests (153 pass, 34 skips),
  hook/export suite 20 tests (19 pass, 1 skip), installed fresh fixture 61 tests
  (51 pass, 10 skips). Skips require NPU or reference dependencies. Three export
  routes apply and source-equality/reverse checks pass. Ruff, py_compile, and
  both checkout diff whitespace checks pass. No server restart or push performed.
- Final review fix pass complete; no unresolved supported-path findings or
  deferred minor changes. Hardware verification limitations remain as documented.
