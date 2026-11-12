# Request-specific MTP depth plan

User approved testing chunk cold/warm 3/3 and finish 5/3. Execute inline using
the existing TDD workflow; do not push or restart services in this task.

## Design

- Add optional flat settings `chunk_cold_heads`, `chunk_warm_heads`,
  `finish_cold_heads`, `finish_warm_heads`. Each omitted setting inherits the
  corresponding global `cold_heads` / `warm_heads`; existing defaults stay intact.
- Recommended configuration: load `num_heads=5`, set global cold/warm 3/3,
  and set `finish_cold_heads=5`. This also leaves shadow/other at global 3/3.
- Reuse the existing request-type prefix classifier, including engine suffixes.
  No service API or target/scheduler modification is required.
- Validate every configured depth against the actually loaded maximum. Build
  the union of global and request-specific depths before graph_options(), so
  only distinct depths are captured and graph-count limits remain effective.
- Reuse current depth-based batching, output order and single batch D2H.
  Lifecycle tracking remains per engine request and independent of metrics.
- Log resolved type policies. Add no per-head sync or hot reconfiguration.

## Tasks and checks

- [x] RED mixed chunk/finish/shadow graph/eager policy and cache-boundary tests.
- [x] Implement config resolution, typed phase selection and deduplicated capture depths.
- [x] Check inheritance, invalid depths, loaded-head cap and service dictionary forwarding.
- [x] Update docs, patch routes and mirrored tests; keep memory-accounting fix intact.
- [x] Run focused/installed/full regression and lint; preserve unrelated dirty files.

## Validation boundary

This adds a policy for a controlled comparison, not a claim that 3/3 or 5/3 is
faster on 910B. Two depths still require two graph families, not one per request
type. Real NPU memory and latency acceptance remain necessary.

## Verification evidence

- RED: the request_heads constructor argument was missing and config resolution failed its explicit assertion.
- Mixed chunk/finish/shadow cold policies return 3/5/3, then 3/3/3 warm; graph and eager outputs match a separate-arena reference and graph outputs use one batch tolist.
- 194 focused tests: 179 passed, 15 NPU skips (`.tmp/mtp-test-env/mtp-request-focused.log`).
- 146 installed mirror tests: 131 passed, 15 NPU skips (`.tmp/mtp-test-env/mtp-request-installed.log`).
- Full research: 390 tests, 19 errors and one failure, all 20 failure identities unchanged from the memory-fix baseline (`.tmp/mtp-test-env/mtp-request-full.log`).
- Nine independent installation routes: apply/check/reverse, 13-module source parity and compile, empty re-export; 12 mirrored test AST checks pass. Verifier `.tmp/mtp-test-env/verify_mtp_request_release.py`.
- Ruff and whitespace checks pass. Existing ordinary graph and memory-accounting tests remain green. No NPU acceptance or performance result is claimed.
- Review: resolved global and per-type policies are validated before device model allocation; all required depths are captured exactly once per bucket. Shadow/other fallback and existing default behavior remain intact. No new production module or service parser argument.
- No commit, push or service restart performed.
