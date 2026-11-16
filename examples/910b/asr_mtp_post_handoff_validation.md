# Post-MTP handoff P0 validation

This is an independent blue-zone NPU simulation, authorized because the actual
streaming service and load command were unavailable. It measures the MTP
handoff, CPU bookkeeping and late proposal, **not complete target-decoder steps,
ASR commit latency or scheduler turnaround**. The observed improvement does not
establish that the production 3.24 ms post-MTP gap has been eliminated.

## Changes and outcome

- Pending proposals validate the confirmed suffix instead of rematerializing
  full histories and repeating cost calculations.
- Independent CPU accounting runs after combined D2H submission and before its
  completion wait. No additional readback or event is introduced.
- After the first two changes alone showed no stable total improvement across
  separate processes, CPU result parsing was reduced at the existing runner
  buffer seam. Its placeholder/vocab/discard rules match the installed parser;
  the original buffer is restored even when bookkeeping raises.
- Optional host stage timing uses `profile_stages`; it is off in measurements.

Same-process paired runs show a small reproducible improvement. Representative
medians with the actual delivery `_bookkeeping_sync` method:

| Heads | Phase / requests | Baseline handoff ms | Optimized ms | Reduction |
| --- | --- | ---: | ---: | ---: |
| 3 | warm / 1 | 4.660 | 4.599 | 1.3% |
| 3 | warm / 4 | 5.002 | 4.904 | 2.0% |
| 3 | warm / 8 | 5.958 | 5.827 | 2.2% |
| 3 | cold / 1 | 5.210 | 5.149 | 1.2% |
| 3 | cold / 4 | 6.534 | 6.440 | 1.4% |
| 5 | warm / 1 | 6.989 | 6.927 | 0.9% |
| 5 | warm / 4 | 8.362 | 8.262 | 1.2% |
| 5 | warm / 8 | 8.796 | 8.656 | 1.6% |
| 5 | cold / 1 | 7.877 | 7.823 | 0.7% |
| 5 | cold / 4 | 10.019 | 9.930 | 0.9% |

All paired output digests matched. CPU-tail paired mean savings were about
0.04–0.13 ms. Detailed medians, paired mean savings, block-bootstrap intervals,
environment, source hashes and raw-result hashes are in
`asr_mtp_post_handoff_results.json`. These intervals describe repeated samples
on this isolated machine, not variation across production workloads.

## Method and limits

- Ascend 910B3, physical device 6, with no other running NPU processes before
  testing; Python 3.12.13, Torch 2.10.0+cpu, TorchNPU 2.10.0.post4, CANN 9.1.0,
  installed vLLM 0.23.0+empty and vLLM-Ascend 0.23.0.post1.
- Existing server `mtp-inference.pt`, Qwen3-ASR-1.7B config and original embedding
  and LM-head weights; hidden size 2048 and vocabulary 151936. Native fusion,
  native RoPE, paged KV, graph token buckets 32/128/512, max batch 8, context 160.
  This checkpoint is not asserted to be identical to the user's production file.
- Synthetic hidden states and deterministic rejection results, partial/full
  acceptance across rows; warm history 96 and cold prompt 59. Actual MTP
  prelude/FIA graphs execute. This is not audio recognition or accuracy testing.
- Baseline is the pre-P0 source snapshot, including the common activation
  diagnostics. Both versions share unchanged captured graph computation.
  Baseline calls its original readback method; optimized calls split readback.
  Source modules are isolated; there is no double graph-memory advantage.
- 15 warmup ABBA blocks, then 60 measured ABBA blocks (120 observations per
  variant/case). State reset and the pre-measurement synchronization are outside
  timed regions. CPU threads=1. Timed path retains its real completion wait.
- Five cases per run: warm batch 1/4/8 and cold batch 1/4. Two additional 3-head
  runs and a 5-head run with parser/append simulation showed the same direction.
  The final runs use the extracted delivery bookkeeping method, with prompt
  logprobs disabled and real token mirrors/output lists. They do not run the
  entire service, runner forward, scheduler, or hook-wrapper turnaround.
- A separate short CPU+NPU profiling run produced five `trace_view.json` files;
  its samples are excluded from performance results. No live service was
  restarted and no installed engine file was overwritten.

Server artifacts remain under `/home/codex-mtp-post-handoff-20261116/`:
`baseline/`, `optimized/`, `reference_model_runner.py`, `final3.json`,
`final5.json`, `real3a.json`, `real3b.json`, `real5.json`, test logs and `trace/`.
The server clock reports 2026-11-17; the local task date was 2026-11-16.

## Reproduce

Keep an unmodified baseline source directory. Use the candidate module directory
as `PYTHONPATH` and a separate output file. Use trusted local checkpoint/source
paths; the benchmark reads the checkpoint and extracts the named bookkeeping
method from that source without replacing the live runner.

```bash
ASCEND_RT_VISIBLE_DEVICES=6 PYTHONPATH=/path/to/candidate \
python benchmark_asr_mtp_handoff.py \
  --baseline /path/to/baseline \
  --checkpoint /path/to/mtp-inference.pt \
  --model /path/to/Qwen3-ASR-1.7B \
  --runner-source /path/to/reference_model_runner.py \
  --heads 3 --repeats 60 --out final3.json
```

Repeat with `--heads 5`. For attribution only, run separately with
`--repeats 1 --profile-dir trace --out profile-only.json`; do not compare profiled
latencies against unprofiled runs. Profiling follows the CPU+NPU short-window
approach from the supplied [Ascend profiling skill](https://github.com/ascend-ai-coding/awesome-ascend-skills/blob/main/skills/profiling/pytorch-profiling-collection/SKILL.md).

## Correctness and delivery checks

- Local MTP suite: 210 tests, 173 passed and 37 dependency/device skips.
- Worker-hook and pending-path suite: 22 passed, including scoped parser restore
  on success/error, token-tail contract, timing spans and default clock disabled.
- NPU: four tests passed without skips: actual prelude/FIA vs eager after partial
  rejection; chunked prefill completed together with warm decode; installed
  vLLM parser contract; exact fast-parser parity across widths/discard patterns.
- Fresh install, focused upgrade, historical graph upgrade and hook-only repair
  checked for applicability, reverse application and canonical source equality.
- Fresh architectural review found no actionable defect.

The next production measurement should use `ASR_MTP_HANDOFF_STAGE` together with
a device timeline to separate remaining bookkeeping, late proposal and engine
turnaround. This change does not claim millisecond-scale scheduler savings.
