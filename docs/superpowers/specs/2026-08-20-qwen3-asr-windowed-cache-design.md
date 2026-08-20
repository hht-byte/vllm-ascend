# Qwen3-ASR Windowed Cache Reuse Design

## Objective

Add a non-invasive streaming helper for Qwen3-ASR services on vLLM and
vllm-ascend 0.23.0 that submit one independent request per audio update through
a long-lived `AsyncLLMEngine` (`AsyncLLM`'s compatibility alias in 0.23.0).
The helper must preserve the official cumulative-audio prompt layout while
allowing vLLM's existing multimodal encoder cache and automatic prefix cache
to reuse audio outside a configurable recomputation window.

## Scope

The implementation is an example-side library. It does not modify vLLM,
vllm-ascend, the Qwen3-ASR model, the scheduler, or the KV block manager.
It provides:

- per-stream PCM buffering without repeatedly concatenating the full stream;
- deterministic freezing of old audio at encoder-aligned boundaries;
- immutable multimodal IDs for frozen audio and changing IDs for the active
  recomputation window;
- construction of the cumulative Qwen3-ASR prompt as one logical audio span;
- an adapter for repeated `AsyncLLMEngine.generate` calls;
- an official-style unstable-token rollback state for text prefix prompting;
- unit tests that do not require NPU hardware or model weights.

The helper does not implement an HTTP or WebSocket server and does not claim
numerical equivalence between split and cumulative audio encoding. A real
weight accuracy gate is required before production rollout.

## Constraints

- Keep the external business flow based on independent asynchronous generate
  calls; do not use vLLM's realtime endpoint.
- Use one long-lived engine instance per cache domain.
- Keep each engine request ID unique.
- Preserve one audio BOS/EOS pair. Multiple cached audio items are represented
  by adjacent occurrences of the inner `<|audio_pad|>` item placeholder with
  no separator. Do not repeat the full placeholder returned by
  `Qwen3ASRForConditionalGeneration.get_placeholder_str`.
- Never reuse an ID for mutable audio data.
- Do not add environment variables or dependencies.
- Do not modify upstream model or scheduler code.
- Keep user-facing documentation in Chinese.

## Architecture

### `AudioChunkBuffer`

Stores normalized mono float32 PCM chunks on a global sample timeline. It can
read a bounded range and discard samples that have been promoted to immutable
segments. This avoids the official implementation's repeated concatenation of
all historical PCM.

### `WindowedAudioState`

Owns one stream's timing and segmentation state.

- `step_samples` is the inference trigger, normally two seconds.
- `freeze_unit_samples` is the encoder-aligned promotion unit, initially four
  seconds for Qwen3-ASR.
- `recompute_window_samples` is the minimum active suffix, initially eight
  seconds for an accuracy-first deployment.

At an emission boundary `T`, the target frozen position is:

```text
align_down(max(0, T - recompute_window_samples), freeze_unit_samples)
```

Audio before that position is promoted once into immutable segments. Audio
from the frozen position through `T` forms one mutable active item. Because
promotion happens in whole units, the actual recomputation window is in
`[W, W + freeze_unit)` except during stream startup.

Each item ID is a SHA-256 digest over a caller-provided cache namespace, sample
rate, absolute sample range, and canonical float32 bytes. Its ID changes
whenever its length or content changes. If an earlier active item later becomes
an identical frozen segment, it deliberately retains the same ID and can reuse
the already computed encoder output.

### `Qwen3ASRPromptBuilder`

Consumes the official raw prompt containing exactly one inner
`<|audio_pad|>` token inside one `<|audio_start|>`/`<|audio_end|>` pair. For
`N` audio items it replaces that token with `N` adjacent `<|audio_pad|>` tokens
and appends the committed ASR text. It returns a plain vLLM prompt dictionary:

```python
{
    "prompt": expanded_prompt,
    "multi_modal_data": {"audio": [audio_0, ..., active_audio]},
    "multi_modal_uuids": {"audio": [id_0, ..., active_id]},
}
```

The helper supplies raw PCM items so it remains compatible with the existing
Qwen3-ASR multimodal processor. The processor and model remain unchanged.

### `WindowedAsyncLLMEngineAdapter`

Serializes inference for one stream, calls the injected engine's asynchronous
`generate` method with a fresh request ID, and returns the last engine output
for each ready snapshot. Engine construction remains the responsibility of the
business service and must enable automatic prefix caching.

The adapter accepts a callback that converts the final engine output into the
next stable text prefix. The bundled rollback state provides the official
default policy, while Qwen-specific output parsing remains in the business
layer where the displayed transcript is managed.

### `Qwen3ASRRollbackState`

Implements the official realtime warmup and tokenizer-token rollback policy as
an optional callback for the adapter. It stores the complete raw decoded text
for `parse_asr_output`, while returning only the rolled-back stable prefix for
the next request. It depends on a tokenizer protocol and an injected function
that extracts generated text from the engine output.

## Cache Semantics

Frozen item IDs and prompt offsets remain stable across later requests. vLLM
can therefore reuse their encoder outputs and all full KV blocks whose chained
hashes end before the mutable item. The active item ID changes, so encoder and
KV reuse stops there automatically. Audio EOS, committed text, and generated
tokens occur after the mutable item and are recomputed on every request.

Encoder cache entries are engine-local and evictable. Correctness must not
depend on a hit: eviction causes recomputation, not wrong output. Streams should
use sticky routing to improve hit rate.

## Error Handling

- Reject non-positive sample rates and timing values.
- Require `recompute_window_samples >= freeze_unit_samples`.
- Require step, freeze, and recomputation durations to resolve to integral
  sample counts.
- Reject non-mono arrays instead of silently mixing channels.
- Normalize int16 PCM to float32 in `[-1, 1]`; accept other numeric input as
  float32 without clipping.
- Reject a prompt that does not contain exactly one audio item placeholder.
- Do not emit an empty active audio item.
- Do not allow concurrent inference calls within one stream adapter.

## Verification

Unit tests cover PCM normalization, range reads, promotion boundaries, stable
and mutable IDs, prompt structure, text rollback, flush behavior, unique
request IDs, and sequential async generation.

Production qualification additionally requires real Qwen3-ASR weights on
Ascend:

1. Compare cumulative and split Audio Encoder outputs at candidate boundaries.
2. Compare streaming WER and committed-prefix revision rate against the
   official cumulative implementation for 4, 8, and 12 second windows.
3. Measure encoder-cache and prefix-cache hit rates.
4. Measure per-step latency and HBM use under the target concurrency.
5. Retain the smallest window whose accuracy remains within the business
   acceptance threshold.
