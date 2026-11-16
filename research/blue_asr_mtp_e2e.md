# Isolated blue ASR/MTP comparison

These tools integrate the existing MTP research implementation with pinned
vLLM 0.23 and vLLM-Ascend sources. The common transport and scheduler port is
applied equally to both compared MTP implementations. The generic controller
adapter lives in `hht-byte/LLM_Optimize`; model details stay outside its core.

## Frozen sources and data

- vLLM: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.
- Installed vLLM-Ascend: `1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`.
- Baseline MTP: `41a2dde544690c430355f00f502d0f9a3b0de8c9`.
- Optimized MTP: `6f39716af53e75a19be8055d71a3e37646fa9cfc`.
- Qwen3-ASR-1.7B and the trained five-head inference checkpoint retain the
  file hashes recorded in `blue_mtp_trained_validation.md`.
- Public audio: [google/fleurs](https://huggingface.co/datasets/google/fleurs),
  `pt_br/test`, revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, CC BY 4.0.
- VAD: [Silero](https://github.com/snakers4/silero-vad), v6.2.1,
  commit `7e30209a3e901f9842f81b225f3e93d8199902b1`, MIT.

`blue_prepare_audio.py` checks the archive, VAD code and model hashes. It selects
the first four sorted qualifying mono 16 kHz recordings, crops each to 15 seconds,
and prefers VAD speech ends near four seconds while enforcing 2–6 second chunks.
Silence is retained; a forced duration cut is labelled separately. Preserve the
generated manifest and WAV hashes outside Git. This preparation produced 12
segments; it simulates the requested product workload and does not establish
quality or performance on private yellow data.

## Execution boundaries

`build_blue_mtp_overlay.py` copies clean registered engine packages into a new
directory, validates supported source contracts, applies bounded integration
changes, and hashes every copied package file. It never patches installed
packages. The inference checkpoint uses the restricted Torch loader.

`blue_isolated_run.py` requires an administrator-provided Linux environment
with namespace capabilities. It creates private mount, network, PID and IPC
namespaces, mounts code/model/audio/weights read-only, exposes one physical NPU,
drops capabilities and sets no-new-privileges. Only runtime output and private
temporary storage are writable. Writable output must be disjoint from inputs.
The launcher kills its namespace child on parent death. The outer trusted
Worker still owns process groups, cleanup confirmation and quarantine.

Inside the namespace the single exposed physical NPU is logical device zero.
Register vendor library/Python paths explicitly in the outer execution profile;
credentials and project Python paths are excluded from the child environment.
Verify the bounded worker probe's `/overlay/` origins and hashes before use.
Raw model outputs, audio, profiling and logs remain local.

An independently pinned [batch-invariant configuration](blue_batch_invariant.md)
is available when exact greedy output is affected by batch shape. It requires
its own complete correctness gates and experiment ID before timing.

## Correctness before timing

Use `blue_mtp_preflight.py ar` against the native autoregressive service to
create references. Run `parity --variant baseline` and `parity --variant candidate`
with the same manifest and references. Both run C1 and C20, require exact token
IDs and text, reject truncated output, and bind live code/config/data identities
and the loaded checkpoint. An increase in actual MTP head work is required.

`diagnostic` separately exercises correct and incorrect initial drafts. It is
ineligible for the parity gate and performance measurements. The normal
transcription path supplies no reference-derived draft. Each VAD segment has a
fresh request ID; native resumable first-verification sessions are rejected by
this port until a session lifetime protocol is implemented.

`gate` assembles both verified reports and the AR reference report into the
restricted global correctness gate. References, gate and local registration
stay outside candidate mounts. The controller checks this gate before even the
first baseline benchmark. All attempts use fresh paths; failed evidence is
preserved instead of overwritten.

## Load and measurement

`blue_mtp_load.py` ramps one closed-loop client per second to 20 clients. Each
client completes two requests started after the ramp, then three contiguous
windows each record 240 completions. Actual per-case counts are recorded.
Ramp, warmup and drain samples are excluded from performance metrics, while
every response must still match the frozen references.

The controller pools 720 measured latency samples, uses linear interpolation
for percentiles, and divides completion/token counts by window duration for
throughput. NPU allocator peaks cover the post-warmup reset through the
post-window worker observation, which may overlap in-flight drain. Device-wide
used memory is a snapshot, not a peak. Profiling runs are separate experiments.

CPU verification uses `python -m unittest discover -s research -p 'test_blue*.py' -v`.
The Linux suite includes actual Unix HTTP and namespace child-death boundaries;
no NPU or external API is used by these tests. Service measurements require
separate real-hardware evidence and do not follow from these tests.
