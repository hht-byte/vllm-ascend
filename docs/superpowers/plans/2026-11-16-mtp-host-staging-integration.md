# Integrate and measure f714fecb host staging changes

Spec: user requests review of f714fecb7338a5f3a97ad55794f4b4ee324481bc and
integration with the current change if beneficial. Retain the earlier requirement
for NPU verification before committing. Independent blue simulation is authorized.

Baseline source561f5914a, delivery86be22365. Canonical upstream source6f39716af
contains the implementation carried by f714fecb7. Preserve current CPU parser,
split readback, detailed fallback diagnostics and startup capability marker.

1. Read source and measurement conditions; identify overlapping changes.
2. Run upstream direct-staging/strided-write tests RED against our current source.
3. Integrate direct pinned staging and strided-write fix, retain current pending
   suffix implementation. Attribute active-row readback before deciding whether
   to adapt it; retain the existing detailed optional stage record.
4. Compare current baseline and composed implementation on idle910B with actual
   checkpoint dimensions/context160. Benchmark staging dispatch must switch
   old/new `program.stage` as well as worker methods; shared unchanged device
   computation must not accidentally hide the optimization under test.
5. Attribute active-row readback separately: omit it if CPU slice cost outweighs
   reduced DMA for small batches. Do not assume the context4096 headline applies.
6. Correctness/tests, patch exports/install parity, fresh architecture review.
7. Commit only validated beneficial composition, keep prior commits/history,
   no push unless requested. Record exact scope and remaining limitations.

Ledger:
- Compared commit and canonical source. Direct staging is new; pending suffix,
  fallback reporting and timing partly overlap. External benchmark is component
  only with context4096; our realistic context160 must be measured independently.
- RED 12 tests: six fail on561. GREEN after composition:12 pass.
- Three NPU ABBA runs (3/5/3 heads): all15 output digests equal, median savings
  0.0348–0.0793ms. Long stalls affect means on both sides; do not claim tail or
  end-to-end gain. Extended5-head repetition added to check robustness.
- Active-row readback isolated ablation regresses by~0.014ms for batch1/4/8;
  omitted. Keep one combined copy, split submit/wait and all existing timings.
- Local suite222:185pass/37skip; NPU suite222:205pass/17skip. Fresh architecture
  review found no actionable defect. Full exports and focused install verified.
