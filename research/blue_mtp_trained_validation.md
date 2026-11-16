# Real weights and public ASR validation

Two independent checks completed on a blue Ascend 910B3. They establish real inference and trained auxiliary graph correctness for the stated fixtures. They do not establish end-to-end MTP service gains.

## Public native ASR smoke check

The installed vLLM 0.23 native `Qwen3ASRForConditionalGeneration` decoded cumulative 2-second and 4-second prefixes of the official public English sample with real Qwen3-ASR-1.7B weights. Both requests completed with nonempty text and EOS-terminated sequences. Speculation was disabled. The sample's original 48 kHz waveform was explicitly resampled to mono 16 kHz with SciPy 1.13.1 `resample_poly`; prepared PCM16 SHA-256 is `f9a00b0bb612a74b996b81c71631a989bc4b2d14140ca5ea7c4ccedaf10dad2c`.

The first attempt rejected the original audio format; the second loaded weights but failed because the native default KV block size 16 is unsupported by this Ascend backend. The successful attempt used supported block size 128. Failed attempts and raw outputs remain in blue storage. Owned processes exited and card cleanup was confirmed. Configuration and reproduction instructions are in [the ASR runner guide](blue_asr_runtime.md).

This runner re-decodes all cumulative audio. It does not reproduce private streaming rollback, prespec consumption, concurrent scheduling or KV/encoder reuse. Two cold requests are a smoke check, not a benchmark or ASR quality evaluation.

## Trained MTP head graph correctness

Training checkpoint: `sss1233234213124/qwen3-asr-pt-mtp5@236ded78a6a573e681dbe2403e060e4edcaca153`, file `best.pt`, 3,524,756,751 bytes, SHA-256 `ef78c076bf9d565f2006966f0711942964fb414adabf4eef0105bec1d76790bd`. It contains five Portuguese-trained sequential MTP heads and optimizer/RNG state; it does not contain the ASR backbone.

Matching public backbone: `Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5`. Config and both safetensor shard content hashes matched the checkpoint's base-content manifest. The original checkpoint was SHA-256 checked again after blue transfer, inspected without executing pickle globals, and read with `torch.load(..., weights_only=True, mmap=True)`. A local inference-only copy retains metadata and MTP tensors; its SHA-256 is `c71bb621fccb2a609598a870986b4ece4274d244aed0e6bc9cfb788c35b616b1`. Neither weights nor training state is committed.

The verifier strictly loaded all five heads (75 tensors, 293,653,760 parameters), plus the actual backbone embedding and output projection from safetensors. On real NPU BF16 it passed:

- Initial paged/FIA graph proposals equal eager trained-head proposals.
- A partial-rejection prelude followed by graph replay equals eager recomputation of the confirmed sequence.
- Confirmed hidden/K/V prefixes agree within the documented BF16 tolerance.

Draft IDs are checked exactly. Cache comparisons use `atol=0.03, rtol=0.03`; the last auxiliary head has no downstream hidden-cache requirement. The fixture uses seed 123, synthetic hidden states and token histories, batch 1, context 32 and token bucket 32. Native fusion is disabled. It validates trained-weight component math, not Portuguese ASR quality, production acceptance rates or service latency. The restricted result JSON is [included here](blue_mtp_trained_validation.json).

To reproduce using local inputs in the matching runtime:

```sh
ASCEND_RT_VISIBLE_DEVICES=0 python research/verify_asr_mtp_weights.py \
  --checkpoint /path/to/mtp-inference.pt --model /path/to/Qwen3-ASR-1.7B \
  --output /path/to/new-attempt.json \
  --source-commit 6f39716af53e75a19be8055d71a3e37646fa9cfc \
  --checkpoint-commit 236ded78a6a573e681dbe2403e060e4edcaca153 \
  --training-checkpoint-sha256 ef78c076bf9d565f2006966f0711942964fb414adabf4eef0105bec1d76790bd \
  --model-commit 7278e1e70fe206f11671096ffdd38061171dd6e5
```

The verifier computes the input checkpoint hash. Provenance arguments are explicitly declared references; they do not replace independent model-file content verification. Existing reports are preserved: each attempt needs a new output file.

## Full-service boundary

Actual native MTP service inference was not run. The clean blue vLLM/vLLM-Ascend stack lacks the project's external-draft/first-verify transport and custom-class dispatch. Read-only prerequisite patch checks fail on scheduler/registry hunks, despite the current runner signatures passing the MTP exporter's contract checks. The delivery branch's graph patch assumes those prerequisites already exist.

Use an engine carrying the matching product-fork prerequisites, or deliberately port them in an isolated full-package overlay before an MTP-on/off service comparison. No shared engine source or packages were changed in this validation. The private yellow service remains the authority for final correctness, latency, throughput and memory validation on 910B4.
