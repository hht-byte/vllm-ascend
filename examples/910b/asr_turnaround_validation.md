# Post-MTP to verify: 910B validation

The selected change packs CPU-owned slot/block-table metadata in the existing
preparation upload, reuses prepared device query starts, and captures pure
decode token embedding for mRoPE ASR. Original three-axis positions, audio
prefill, unsupported-input fallbacks and synchronous MTP readback are retained.
No new runtime D2H, synchronization, per-step timing or logging is introduced.

## Device interval evidence

Independent native-vLLM ASR simulation on physical 910B3 device 1, FP16
Qwen3-ASR-1.7B, Torch 2.10.0 / TorchNPU 2.10.0.post4, CANN 9.1.0 and vLLM 0.23.
The baseline delivery modules are `f9e8710d3`; the ASR simulation uses a private
service/overlay and twelve fixed public Portuguese audio cases. This is not a
replay of the production service or its load.

Both variants use synchronous scheduling, prefix cache off, FULL target graphs,
20 maximum requests, 2048 batched-token capacity and a manual 2GiB target KV
budget. MTP cache context is 512, graph context 160, maximum graph batch 8,
token buckets 16/64/256/1024, chunk heads 3/3 and finish heads 5/3. Preparation
buckets are 16/64/256/1024. Main measurements disable `profile_stages`; short
traces are collected separately afterwards.

Four initial runs use baseline/candidate/candidate/baseline order, including
C4/C20 full requests and short traces. Matching actual batch/token groups
(4,16), (4,14), (2,8) and (2,7) gives 42 warm steps per variant. Their profiled
MTP-tail-to-verify median improves 5.0418 -> 4.6006ms. Preparation-to-verify
improves 1.4589 -> 0.9918ms; the first CPU turnaround does not improve in that
set (3.5948 -> 3.6248ms).

A final lower-overhead comparison disables `record_shapes`, stack and memory
profiling. A Python-only shape ring aligns all 72 preparation markers with
actual requests/tokens, yielding 56 matched warm steps per variant:

| Device interval, median | Baseline | Candidate |
| --- | ---: | ---: |
| MTP last compute kernel -> preparation replay | 3.4746ms | 3.3003ms |
| Preparation replay -> target verification | 1.3283ms | 0.8256ms |
| MTP last compute kernel -> target verification | 4.8127ms | 4.1482ms |

The last interval improves 0.6645ms (13.81%); all four matched groups improve.
The repeated evidence supports approximately 0.4–0.5ms reduction after the
preparation replay. The first CPU turnaround's smaller improvement is not
consistent across all runs. Medians of separate spans should not be added.

The start boundary is the end of MTP's last compute kernel, ScatterNdUpdate in
these traces. The middle boundary is device MODEL_EXECUTE for preparation, as
the candidate eliminates the slot kernel. The end is the first RmsNorm of the
28-layer target graph. This start is later than the preceding SWhere used in
the original production report. Numeric aggregates are in
`asr_turnaround_results.json`.

## Correctness and limits

Full simulated requests match reference text/token IDs in both variants.
Forty-eight relevant tests pass on real NPU, including graph replay, changed
block rows, batch reorder/shrink, multiple cache groups, mRoPE positions and
fallbacks. Four additional attention-builder tests verify query-start reuse and
the legacy snapshot behavior. The focused patch passes fresh apply, exact-file
comparison and reverse application against the delivery baseline.

The local preparation-only same-process ABBA test also reduces stage/upload/
replay/completion time by 0.11–0.12ms in four cases, approximately 20%. Two
repetitions show the same direction. This is a component result.

Complete-request latency does not establish a general improvement: C4/C20
means overlap or regress in some runs. Older short-warmup runs also contain
multi-second outliers. The final instrumented runs locate comparable 3.75–3.88s
spans in baseline and candidate `sample_tokens`, all in warmup and outside the
measured request windows. Bookkeeping, handoff, late proposal and model-execute
spans do not contain them; measured requests have no multi-second outliers.
This does not identify the specific sampling operator or prove an explanation
for every earlier outlier. It is not evidence of a new post-MTP stall.

An additional experiment merged the mRoPE position upload into preparation.
Its independent improvement was only tens of microseconds with mixed group
directions, so it was reverted. Only the selected, repeatably beneficial core
change is delivered. The approximately 3.5ms CPU turnaround remains a future
optimization target; it is not claimed to be eliminated.
