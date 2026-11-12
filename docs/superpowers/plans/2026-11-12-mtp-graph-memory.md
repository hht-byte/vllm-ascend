# MTP graph memory accounting correction

User evidence: cold/warm 5/3 captures 62 token graphs, reserved=15658 MiB;
original profiling reports 4.73 GiB available KV and the extension subtracts the
entire graph reservation again. No service restart or push is part of this fix.

## Evidence and design

`MTPWorker.profile()` captures graphs inside `NPUWorker.determine_available_memory()`'s
profiling interval. Its allocated-byte peak already includes live graph tensors.
`reserved_bytes` measures the entire allocator reservation increase, including
both those live tensors and unallocated retained private-pool memory. Subtracting
all reserved bytes duplicates the live allocation portion. The actual remaining
server capacity still needs the new allocator counters to be measured.

Capture graphs before calling the original memory profiler, so its baseline
includes their live tensors and target activation profiling measures coexistence
with resident graphs. Subtract the full retained pool once after original
profiling. Record both reserved and allocated deltas for diagnostics. Keep
explicit manual KV behavior unchanged and retain the insufficient-budget check.
Log every term, including base and adjusted budgets.

## Tasks

- [x] Trace original profiler, drafter profile timing and graph-pool accounting.
- [x] RED test proves graph capture must precede original profiling and repeated
  profiling must not recapture or accumulate reservation subtraction.
- [x] Implement allocated delta tracking and reorder capture before profiling.
- [x] Verify negative budget, manual override, repeated profiling and old objects.
- [x] Update docs, export/mirror patches and run relevant/full tests and lint.

## Limits

This does not reduce the real 15.29 GiB graph footprint. Fewer token buckets or
smaller request capacities may still be needed after corrected accounting. No
automatic graph disabling, manual KV bypass or device speedup claim.

Ruling: do not just subtract `reserved - allocated` after an in-profile capture.
The earlier target activation peak may precede graph allocation, so that formula
can underestimate simultaneous graph plus target memory. Reordering capture
makes the original activation delta valid with the graphs resident throughout.

## Verification

- Focused: 183 tests, 168 passed / 15 device skips; `.tmp/mtp-test-env/mtp-memory-focused.log`. Includes repeated startup accounting, explicit manual KV and real shortage rejection.
- Installed: 143 tests, 128 passed / 15 device skips; `.tmp/mtp-test-env/mtp-memory-installed.log`.
- Full repeat: 387 tests, 19 errors + 1 failure matching the 20 identities in the prior phase baseline; `.tmp/mtp-test-env/mtp-memory-full-repeat.log`.
- First full run additionally hit `test_sample_records_separate_source_completion_times`, a 10 ms sampler test; independent rerun and complete repeat passed it. Its source and monitor implementation are unchanged.
- Eight deployment routes: apply/check/reverse, 13-module source parity/compile and empty re-export passed; 12 mirrored tests have AST parity. Script `.tmp/mtp-test-env/verify_mtp_memory_release.py`.
- Ruff and whitespace checks pass on changed code. Actual NPU tests now assert retained and allocated graph counters are populated, but remain unexecuted on this CPU host.
- No service restart, commit or push performed. No claim that the specific server now fits in memory without measuring its adjusted budget.

## Publication

The user subsequently requested push. Publish this verified memory-accounting fix
to `codex/streaming-prespec-monitor` and the matching patches to `310P-graph`.
Workspace sharing and other graph-footprint optimizations remain unimplemented.
