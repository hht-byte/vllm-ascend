# MTP non-PA TND KV coverage fix

**Observed failure:** The 910B FIA V3 tiler rejects physical K/V T=2304 with final cumulative KV length 1280. The prior assumption that non-PA TND can ignore unused physical KV rows is invalid for this deployed CANN.

**Fix:** Preserve each logical token bucket, allocate one additional physical query row, and always append a positive-query dummy segment whose KV segment consumes the entire remaining KV capacity. Keep all real request boundaries unchanged. Validate exact cumulative Q/KV coverage before calling FIA.

**Decision:** Avoid zero-query dummy segments and runtime changes to tensor shape. A logical 512-token graph has physical query capacity 513, with fixed KV capacity and one graph key 512. This adds one padded query row per head; device performance must be measured.

## Work

- [x] Reproduce the incomplete KV coverage and exact-full-token boundary in RED tests.
- [x] Fix packed metadata, physical query/scratch sizing, backend validation and capacity logging.
- [x] Update strict fake FIA contracts and real NPU capture/replay tests.
- [x] Regenerate fresh/cumulative/focused patches and update installation guidance.
- [x] Verify related tests, full-suite baseline, patch application/parity, lint and independent review.

Existing unrelated research work is preserved. No service restart or remote push is authorized by this implementation turn.

## Verification

- Four contract regressions went RED to GREEN; fake FIA workspace and execution calls require exact physical Q/KV coverage. Logs: `.tmp/mtp-test-env/mtp-kv-contract-red.log` and `mtp-kv-contract-green.log`.
- Related suite: 126 tests, 123 passed and three real NPU tests skipped (`mtp-tnd-related.log`). Installed focused-patch suite: 50 tests, 47 passed and three NPU skips (`mtp-tnd-installed.log`).
- Full research recheck: 292 tests, the same 19 errors and one failure as the token baseline, three NPU skips, no new failures (`mtp-tnd-full-recheck.log`). The initial run also hit unchanged `test_sample_records_separate_source_completion_times`, whose 10ms threaded deadline missed the metrics sample; the isolated full recheck passed that test. Its production and test source are unchanged.
- Fresh, cumulative graph, row-cache-to-token and focused TND-fix patches passed apply checks, compilation, six-module parity, idempotent re-export and reverse checks.
- Ruff, Markdownlint and scoped whitespace checks passed; mirrored test ASTs match the source tests.
- Independent review found no actionable issue and passed 648 additional strict-coverage CPU parity rounds across exact buckets, small capacities, request reordering, budgets and rejection crops.
- Actual 910B startup, capture/replay and performance remain to be retested. No zero-query support or runtime tensor-shape update is assumed by the fix.
