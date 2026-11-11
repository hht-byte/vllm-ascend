# MTP hidden-row writer regression follow-up

## Evidence and decision

The user measured one `row_copy_kernel` execution at 1.37704 ms after the
native-ops release. The earlier ScatterND sample was 0.45704 ms. Workloads were
reported as approximately similar, not proven identical, so this is evidence
against making the unvalidated Triton kernel automatic, not a controlled ratio.

Restore NPU `row_write_backend=auto` to native ScatterND without a Triton import.
Keep explicit `triton` for device experiments; do not change QKV/RoPE fusion.

For paged token graphs, the flat arena contains only complete hidden vectors.
Use full-H rows for hidden reads and writes, instead of gcd(H,D) fragments:

- Destination `(1097744,128)` becomes a view `(68609,2048)`.
- Updates `(2320,128)` become `(145,2048)` with 145 row indices.
- Total data bytes, cache ownership and independent padding sinks are unchanged.
- Mixed hidden/KV TND and rectangular graphs keep gcd-aligned fragments.
- No new backend/API, allocation pool, device index reads or extra forward.

The old Triton kernel uses a two-dimensional indirect store and small tiles.
Ascend's vector guide recommends physical VectorCore scheduling and contiguous
row stores. These are investigation leads; compiled IR and matched-device
measurements are required before attributing the observed delay to one cause.

References: [vector guide](https://github.com/Ascend/triton-ascend/blob/main/docs/en/programming_guide/vector_operator.md)
and [scatter example](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/004-gather_scatter.py).

## Verification and delivery

- [x] RED default-selection tests, then restore auto ScatterND.
- [x] RED full-token transfer test (4 vs 12), then change paged row width.
- [x] Test actual row/update shapes, unique addresses, zero-copy cache views,
  full cache parity across rejection, budgets and request reordering.
- [x] Independent read-only review of paged offsets, strides and scratch bounds.
- [x] Regenerate fresh, cumulative, token and both focused upgrade routes.
- [x] Update installed tests/docs, run related and full baseline comparison,
  lint and patch apply/parity/reverse checks.

The focused native-rows upgrade starts at research `637b6f878` / engine
`5d757257c`. The existing native-ops upgrade retains its older `f9bf28b42` base.
No push or service restart is part of this follow-up. Real NPU timing and graph
compatibility remain device acceptance work; fewer indices do not prove speedup.

## Results

- Related: 193 tests, 179 passed and 14 NPU skips;
  `.tmp/mtp-test-env/mtp-native-rows-related.log`.
- Installed: 117 tests, 103 passed and 14 NPU skips;
  `.tmp/mtp-test-env/mtp-native-rows-installed.log`.
- Five release routes apply/check, compile and match 12 source modules;
  re-export is empty, reverse-check succeeds, nine mirrored tests have AST
  parity. Verifier: `.tmp/mtp-test-env/verify_mtp_native_rows_release.py`.
- Full research serial run: 359 tests, 19 errors and one failure, identical
  failure identities to the previous baseline. First concurrent run also had
  a monitoring sample test fail to collect sources within its 10 ms window;
  that unchanged monitoring class passed all four isolated checks and then the
  serial full run. Both original and retry logs are preserved as
  `mtp-native-rows-full.log` / `mtp-native-rows-full-serial.log`.
- Independent review found no blocking row-offset, stride or scratch issue;
  independent focused run: 24 passed and one NPU skip.
- Ruff, Markdownlint and both repository diff checks pass. No NPU measurement
  or server restart, commit or push was performed in this follow-up.
