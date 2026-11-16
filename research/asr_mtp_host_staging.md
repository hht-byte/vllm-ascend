# Host staging integration after the device handoff P0 changes

Reviewed delivery commit: [f714fecb7](https://github.com/hht-byte/vllm-ascend/commit/f714fecb7338a5f3a97ad55794f4b4ee324481bc).
Its canonical implementation is source commit `6f39716af`. This integration is
incremental to source `561f5914a` / delivery `86be22365`; it retains the CPU parser,
split readback, hook capability check, fallback diagnostics and optional host
stage timings from those commits. It does not revert to the earlier handoff.

## Selected changes

- Fill reusable pinned metadata through NumPy views. Convert/validate input rows
  before touching reusable storage; fill padding in the array rather than
  constructing and flattening full Python lists. Keep one metadata H2D and the
  existing upload lifetime fence, including changed PA block tables.
- Materialize contiguous rows only when generic cache writers receive a strided
  fused-QKV value view. Already contiguous values retain their storage. This is
  a correctness fix for B/Q and non-paged token paths, not a measured speedup of
  the paged KV benchmark.
- Retain the existing confirmed-suffix validation, combined readback and
  `ASR_MTP_HANDOFF_STAGE` timing. Upstream versions of those changes overlap and
  must not overwrite this branch's parser, diagnostics or split submission/wait.
- Omit active-row-only readback. In a separate isolated 910B copy/event/wait
  microbenchmark, slicing the tiny `(8, 21)` int64 buffer increased median host
  time from 0.0846–0.0867 ms to 0.0981–0.1016 ms (batch 1/4/8). These are copy-only
  timings, not whole-handoff or service savings.

No new switch, additional readback, timing event or per-step log is introduced.
`profile_stages` remains optional and defaults to false.

## Incremental benefit on the current context size

The upstream report used context 4096. Its multi-millisecond staging reduction
does not establish a comparable improvement with context 160. We compared the
current P0 baseline against the composed implementation on an idle Ascend 910B3,
using actual Qwen3-ASR dimensions and the available MTP checkpoint.

Initial run medians, with unchanged device computation shared between A and B:

| Heads | Phase / requests | Current baseline ms | Integrated ms | Saving ms |
| --- | --- | ---: | ---: | ---: |
| 3 | warm / 1 | 4.5559 | 4.4843 | 0.0715 |
| 3 | warm / 4 | 4.8768 | 4.8186 | 0.0583 |
| 3 | warm / 8 | 5.8065 | 5.7665 | 0.0400 |
| 3 | cold / 1 | 5.1223 | 5.0432 | 0.0791 |
| 3 | cold / 4 | 6.4515 | 6.3904 | 0.0612 |
| 5 | warm / 1 | 6.8319 | 6.7526 | 0.0793 |
| 5 | warm / 4 | 8.2470 | 8.1842 | 0.0628 |
| 5 | warm / 8 | 8.6342 | 8.5955 | 0.0387 |
| 5 | cold / 1 | 7.8490 | 7.7744 | 0.0746 |
| 5 | cold / 4 | 9.8279 | 9.7647 | 0.0632 |

A separate repeated 3-head run retained the direction in all five cases. Across
these first three runs, median savings were 0.0348–0.0793 ms (0.45%–1.60%). Output
digests match in every case. Occasional long host stalls occurred on both sides;
some arithmetic means regress and their mean confidence intervals cross zero.
The supported conclusion is a small, reproducible typical handoff improvement,
**not a proven mean/tail improvement or elimination of the production gap**.
Raw-result hashes, all means/medians and paired-block bootstrap intervals are
recorded in `asr_mtp_host_staging_results.json`; no outliers are removed.

An extended 5-head repetition with 180 measured ABBA blocks (360 observations
per variant/case) also improved all five medians by 0.0354–0.0778 ms
(0.41%–1.10%). All five means improve in that repetition, but some mean
intervals still cross zero. Across all four runs, paired-block median bootstrap
intervals are positive in all 20 cases. Those intervals describe this repeated
synthetic fixture, not uncertainty across production workloads.

## Measurement scope and reproduction

- Python 3.12.13, Torch 2.10.0+cpu, TorchNPU 2.10.0.post4, CANN 9.1.0;
  vLLM 0.23.0+empty / vLLM-Ascend 0.23.0.post1, physical NPU 6.
- Available `mtp-inference.pt`, original Qwen3-ASR-1.7B embedding and LM-head
  weights, hidden 2048 / vocab 151936. Checkpoint identity with production is
  not asserted. Synthetic target hidden states and deterministic rejection rows.
- Native fusion/RoPE, paged KV, graph token buckets 32/128/512, max batch 8,
  context 160. Warm history 96; cold prompt 59. Both 3 and 5 heads.
- Same-process ABBA: 15 warmup blocks, 60 measured blocks, 120 observations per
  variant/case for the initial three runs. Reset and pre-measure synchronization
  excluded; the real readback wait remains inside measurement.
- The benchmark switches the old/new `PackedMTPProgram.stage` method outside
  timing. Sharing the graph without this switch would accidentally measure
  the candidate staging on both sides. Graph computation is identical for the
  measured paged path. The strided-value fix is validated separately.
- Uses the delivery runner's extracted `_bookkeeping_sync`, CPU input mirrors
  and output lists, with prompt logprobs disabled. Includes early handoff,
  bookkeeping and late proposal; excludes target decoder, audio, scheduler and
  the complete hook-wrapper turnaround. Profiling/stage logging is disabled.

Run `research/benchmark_asr_mtp_handoff.py` as described in
`asr_mtp_post_handoff_validation.md`, with an unmodified `561f5914a` source
directory for `--baseline` and the integrated source on `PYTHONPATH`. Repeat
with `--heads 3` and `--heads 5`. Production commit latency and throughput still
need the actual streaming workload; this independent simulation is not a
replacement for that measurement.

Artifacts are retained under `/home/codex-mtp-staging-integration-20261116/` on
the authorized server: source snapshots, benchmark, raw JSON runs and test logs.
The server clock reports November 17; the local task date is November 16.
No installed service files were overwritten and no live service was restarted.

## Validation and installation

- New regression tests: 12 total; six failures reproduced on the prior source,
  all pass after integration. Covers exact padded metadata bytes, shrink/reorder,
  mixed inactive heads, aliased input, H2D lifetime and failed validation, plus
  strided/dense row writes in both layouts.
- Local MTP suite: 222 tests, 185 passed / 37 dependency-device skips.
- NPU MTP suite: 222 tests, 205 passed / 17 skips, including actual paged and
  non-paged fused graph replays, rejection, cold/prefill and depth changes.
- Fresh architecture review found no actionable issue. Paged performance runs
  alone do not validate non-paged behavior; that is covered by regression tests.

For an engine already installed from source `561f5914a` / delivery `86be22365`,
the focused patch changes only the three staging/writer modules:

```bash
git apply --check /path/to/asr_mtp_host_staging_v023.patch
git apply /path/to/asr_mtp_host_staging_v023.patch
```

The current cumulative `asr_mtp_graph_v023.patch` and
`asr_mtp_device_rejection_v023.patch` also include this integration; select the
patch matching the installed baseline, do not stack all three. If already
installed from `f714fecb7`, or manually mixed versions, generate an upgrade from
the canonical source against that exact checkout with `build_asr_mtp_patch.py`.
Do not force an incompatible patch. Restart the engine after deploying all
required files together; configuration settings are unchanged.
