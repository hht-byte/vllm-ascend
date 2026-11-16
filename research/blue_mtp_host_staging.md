# Blue MTP host staging validation

The recurring packed metadata staging work is a measured optimization opportunity in the current implementation. In the final paired 910B3 component run, staging fell from **2.43–2.57 ms to 0.24–0.67 ms** across the six tested cases (**73.9–90.0%** lower). This measures metadata preparation and transfer completion; it does not measure ASR service latency or a rejection-to-MTP gap in the product.

Baseline source: `41a2dde544690c430355f00f502d0f9a3b0de8c9`. Candidate implementation: `6f39716af53e75a19be8055d71a3e37646fa9cfc`. Deployment baseline: `310P-graph@271fa79db29e6064430e84c77673b4bf7d26208e`. Exact configuration, source hashes and all round means are in [the aggregate metrics](blue_mtp_host_staging_metrics.json).

| Active batch | History tokens | Baseline ms | Candidate ms | Reduction |
| --- | --- | --- | --- | --- |
| 1 | 64 | 2.428 | 0.243 | 90.0% |
| 4 | 64 | 2.520 | 0.326 | 87.1% |
| 1 | 512 | 2.431 | 0.264 | 89.1% |
| 4 | 512 | 2.519 | 0.404 | 84.0% |
| 1 | 2048 | 2.462 | 0.327 | 86.7% |
| 4 | 2048 | 2.566 | 0.669 | 73.9% |

The source audit found three bounded recurring costs. `PackedMTPProgram.stage()` constructed padded Python sequence lists, flattened them and built another complete payload before copying into an existing pinned array. It now fills typed views of that array after the lifetime fence. Pending late bookkeeping still converted complete confirmed histories although it used only the sampled suffix; it now reads that suffix, while ordinary fallback proposals keep their complete input histories. Combined readback now copies active rows. It retains the existing single completion fence, per-head FIA event signals and cache-validity rules.

An isolated host conversion check with eight 2048-token rows measured about 0.530 ms for complete histories and 0.0068 ms for four-token suffixes. That comparison isolates Python/NumPy conversion and does not establish a worker-call or service speedup. Active-row readback has correctness coverage; its independent latency benefit is not claimed.

The old flamegraph predates the eligible device-handoff path and cannot attribute its current gap. `profile_stages=True` now adds `ASR_MTP_DEVICE_HANDOFF_HOST_STAGE` spans for preflight, program/prelude staging, replay enqueue, combined readback and CPU conversion. Replay-provided task-update/enqueue spans are included when available. `combined_readback_host_ms` includes transfer and waiting for preceding device work; it is not pure D2H time. Cache-only prefill does not reuse an earlier replay's timing. Disabled diagnostics read no clocks and create no extra device events. Fallback-only traffic emits a first/periodic reason summary.

Separate torch_npu profiler collection was exercised using the existing collection workflow. The raw trace stays in blue storage. Profiled durations and overlapping CPU/device events are not used as benchmark samples or summed into a service latency.

## Reproduce the component experiment

Use matching baseline and candidate checkouts, the same idle card and the existing runtime. Run the **same candidate benchmark script** with each checkout's `research` directory first on Python's search path:

```sh
ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/baseline/research \
  python /path/to/candidate/research/benchmark_asr_mtp_host.py > baseline.json
ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/candidate/research \
  python /path/to/candidate/research/benchmark_asr_mtp_host.py > candidate.json
```

The fixed fixture uses capacity 8, context 4096, token bucket 64, three small random heads, hidden size 256, FP16 and paged KV. It performs ten warmups followed by seven rounds of 100 staging calls, with device synchronization at round boundaries. The table shows the median of round mean wall times. No target forward runs. Two earlier paired runs showed the same direction; the table uses the final code snapshot. For a separate profile run, pass `--profile-dir /path/to/blue-local-trace`; preserve its output outside Git.

## Correctness and delivery

The final full MTP suite ran on the blue 910B3: **222 tests, 17 skips, no failures or errors**. Both actual prelude/FIA parity tests passed. The project CPU suite also passed. The full NPU suite needs the installed engine's usual Triton device-properties initialization before test discovery:

```python
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
init_device_properties_triton()
```

Full-suite testing also reproduced two older failures on the pinned baseline: fused QKV produced strided V views, which the contiguous graph row writer rejected. Both callers now make those updates contiguous; already dense rows preserve their pointers. The strict writer contract remains. Regression tests and both real NPU tests passed after the correction.

The wider `research/test*.py` collection is not wholly green. The complete pinned baseline itself reproduces one admission/concurrency failure and eleven errors in unchanged prespec helper budget fixtures. A deadline/shadow case also failed during a broad candidate run, then passed when run separately on both baseline and candidate. These broader fixture issues remain recorded; the passing MTP suite and project CPU suite above must not be described as a clean result for every research test.

For an engine with the documented baseline MTP modules already installed, apply `asr_mtp_host_staging_v023.patch`. Fresh installations use the updated cumulative MTP patch and its prerequisites instead. Apply-check, installation byte equality for all 15 canonical modules, empty re-export and reverse application were checked. Do not stack alternative cumulative and focused patches. Pulling the delivery branch alone does not install optional engine modules.

The public ASR simulation and trained-weight validation have separate evidence and limits. Before a yellow validation request, freeze effective service/benchmark configuration and the exact source/deployment commits, establish token/text parity on the approved workload, then compare end-to-end latency, throughput and memory on 910B4. No yellow raw data is needed in the blue report.
