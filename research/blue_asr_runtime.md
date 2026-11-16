# Public ASR baseline on an isolated blue runtime

`blue_asr_service.py` is a bounded **cumulative-audio re-decode simulation**. At
each chunk boundary it sends all audio received so far through the real public
Qwen3-ASR model. It saves the resulting text, token IDs and observed wall time.
It does not simulate audio arriving in real time. Every row identifies the
simulation and explicitly records `mtp_enabled: false`.

The private `qwen3_asr_async.py` service is absent from this repository; see
[the existing service boundary](streaming_mtp_service.md). This baseline does
not reproduce that service's rollback, prespec consumption, concurrent
scheduling, encoder or paged-KV reuse, target verification, MTP execution, or
network latency. A baseline run alone makes no optimization claim. MTP
checkpoint loading and correctness comparisons require the separate worker
integration and the matching trained checkpoint.

## Frozen public inputs

- Public base: `Qwen/Qwen3-ASR-1.7B`, revision
  `7278e1e70fe206f11671096ffdd38061171dd6e5`.
- Public example audio: the [Qwen English example](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav).
  The original is mono 48 kHz, 722459 samples, SHA256
  `f9b4440ac8393e47c14a6240e9739dea09b645bb1592b8f2dd48feb9666cea7f`.
  The runtime explicitly prepares mono 16 kHz PCM16 with
  `scipy.signal.resample_poly` (SciPy1.13.1), yielding 240820 samples and SHA256
  `f9a00b0bb612a74b996b81c71631a989bc4b2d14140ca5ea7c4ccedaf10dad2c`.
- Official source reference: [QwenLM/Qwen3-ASR at
  7c6daf77a2421100f5fb066495372c00129d39ff](https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff).
- Optional Torch fallback package: `qwen-asr==0.0.6`, wheel SHA256
  `b9c55a38413298f3a990a4475467399daec6e8f4172363053fc42e2166c2dfd3`.
- Isolated compatibility packages: `transformers==4.57.6` and
  `huggingface-hub==0.36.2`; installed Torch/TorchNPU remain in use.

The default backend is the installed vLLM 0.23 native `qwen3_asr` model and
processor with the existing Transformers 5.5.4 environment. No custom model
registration or shared-runtime change is needed. The older public
`qwen-asr` vLLM backend cannot import against this engine because it references
the removed `vllm.inputs.data` module; do not register it.

For the optional `--backend transformers` fallback, the official package's
numerical model/config/processor code is unchanged. In
the isolated scratch copy only, `qwen_asr/__init__.py` defers imports of the
optional inference wrapper and multilingual forced aligner. The baseline
imports public model/processor classes directly, reads mono 16 kHz audio using
SoundFile, and performs deterministic greedy generation with eager attention.
This avoids installing unused alignment dependencies or changing the shared
vLLM/Transformers environment. Full upstream package behavior is not implied.

## Running the baseline

Download assets and prepare the isolated dependency directory outside source
Git. On the blue test host, the runtime uses a dedicated directory on an
existing large bind mount, with identical host/container paths. An earlier
`/tmp` attempt hit a writable-layer limit despite the
underlying volume reporting free space; keep large assets on the bind mount.

After checking that the assigned card is idle, run a bounded owned process:

```sh
ASR_ROOT=/path/to/asr-runtime
ASCEND_RT_VISIBLE_DEVICES=7 timeout --kill-after=10s 600s python \
  "$ASR_ROOT/blue_asr_service.py" \
  --backend vllm \
  --model "$ASR_ROOT/models/Qwen3-ASR-1.7B" \
  --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --audio "$ASR_ROOT/asr_en_16k.wav" \
  --output-dir "$ASR_ROOT/baseline-attempt-001" \
  --device npu:0 --chunk-seconds 2 --max-audio-seconds 4 \
  --max-new-tokens 64 --language English
```

Each attempt needs a new output directory. `manifest.json` records the source
hash, declared model revision, audio hash, effective limits and package
versions. Native engine options are also recorded: one sequence, TP1, BF16,
seed0, a 2048-token model/batch limit, bounded audio profiling input, 25% device
memory utilization, eager execution, and disabled prefix caching, chunked
prefill and async scheduling. There is no speculative configuration.
The native Ascend backend requires `block_size=128`; the installed engine's
default 16 fails KV initialization because the backend supports only 128.
This baseline preserves its explicit chunked-prefill setting; the installed
engine warns that disabling it is not officially supported for this model.
Treat this as a startup/inference smoke test rather than service validation.
`chunks.jsonl` contains raw public-example outputs and observations;
keep it outside tracked files. Preserve the asset download manifest, model
file hashes, isolated-wheel hashes and attempt logs alongside it. The script
does not download weights, load pickle checkpoints, expose model text in CLI
errors, or start a listener. The adapter shuts down its owned engine core after
the last request; the deployment launcher must bound and reap its process
group and verify the card is clear afterward. The optional Torch fallback
needs `PYTHONPATH="$ASR_ROOT/site"`; do not set that path for the native backend.

The standard-library orchestration checks run with:

```sh
python -m unittest discover -s research -p test_blue_asr_service.py -v
```

These tests validate boundaries and cumulative inputs, not ASR quality or NPU
performance. Runtime success must be reported separately from those tests.

## Actual MTP engine boundary

The native ASR baseline completed two public-audio requests on the tested
910B3 runtime. **Actual native-engine MTP was not executed**, and no end-to-end
MTP speedup is established. The separate trained-head checks use the root
agent's normalized inference-only checkpoint with `weights_only=True`;
neither those weights nor the original training pickle belong in source Git.

The installed clean runtime differs from the deployment fork assumed by the
ASR extension: vLLM commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`,
vLLM-Ascend commit `1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`.
Read-only checks found no external-draft proposer, transport patch,
first-verify hook, or ASR-MTP hook. The Ascend drafter dispatcher lacks
`custom_class` routing even though the vLLM configuration accepts that method.
The existing first-verify patch does not apply cleanly: scheduler and
recompute-scheduler anchors differ, the worker registry anchor is absent,
and the external-draft proposer file is missing. The MTP exporter's runner
signature, post-bookkeeping CPU proposal order and late-worker-entry checks
pass, but its first-verify registry prerequisite fails.

The delivered graph upgrades explicitly assume that base online-MTP /
external-draft / first-verify extension is already installed; they do not
bootstrap this clean engine. A future validation needs either the matching
approved deployment fork or a deliberate scheduler/transport port in a
dedicated full-package overlay, selected only for its bounded process with
`PYTHONPATH`. Preserve compiled extension files and verify child import
origins before starting an engine. Keep restricted checkpoint loading and
the baseline's explicit block size. No such port or global installation was
performed here. The window-cache `prepare_vllm_config()` helper requires
prefix caching and is unsuitable for the current MTP path, which requires
prefix caching disabled to retain complete target hidden-state history.
