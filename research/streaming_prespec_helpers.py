"""Copyable model helpers; the service supplies its existing request packer.

No vLLM import is needed to run mock tests. The engine contract is the
AsyncLLMEngine generate/abort API, with cumulative RequestOutput token IDs.
See streaming_prespec.md section 4 for the one service-specific extraction.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

if __package__:
    from .streaming_prespec import LaunchDecision, ShadowRequest
else:
    from streaming_prespec import LaunchDecision, ShadowRequest


@dataclass(frozen=True, kw_only=True)
class ModelShadowRequest(ShadowRequest):
    # Private copies of small request settings, never the live ASR state/engine.
    model_context: dict[str, Any]
    sampling_params: Any


class PreSpecModelHelpers:
    """Mixin for Qwen3ASRModelAsync, or copy these three methods into that class.

    Set ``prespec_engine`` to the existing shared engine before enabling.
    Supply ``_pack_stream_request`` as documented; it must not mutate live state.
    ``prespec`` is the existing PreSpecCoordinator, or None when disabled.
    """

    def try_start_prespec(
        self, *, ws_id, seg_id, prefix_version, full_audio_pcm_f32le,
        current_audio_end_samples, last_formal_audio_end_samples,
        chunk_size_samples, stable_prefix_token_ids, rollback_draft_token_ids,
        model_context, sampling_params, committed, audio_queue_empty,
        load=None, predicted_draft_ms=0.0,
    ):
        coordinator = getattr(self, "prespec", None)
        if coordinator is None:
            return LaunchDecision(False, "disabled")
        if committed:
            return LaunchDecision(False, "committed")
        if not audio_queue_empty:
            return LaunchDecision(False, "audio_backlog")
        new_samples = current_audio_end_samples - last_formal_audio_end_samples
        if chunk_size_samples <= 0 or new_samples < 0:
            raise ValueError("invalid chunk size or audio cutoffs")
        if new_samples < coordinator.policy.midpoint_samples:
            return LaunchDecision(False, "before_midpoint")
        if new_samples >= chunk_size_samples:
            return LaunchDecision(False, "formal_chunk_due")
        if not rollback_draft_token_ids:
            return LaunchDecision(False, "no_rollback_draft")
        # Take the time before copying so snapshot overhead consumes the budget.
        deadline_ns = coordinator.clock() + int(
            (chunk_size_samples - new_samples) / 16000 * 1e9)
        audio_bytes = bytes(full_audio_pcm_f32le)
        if not audio_bytes or len(audio_bytes) % 4:
            raise ValueError("audio must be complete mono float32 LE PCM")
        request = ModelShadowRequest(
            session_id=ws_id, audio_epoch=seg_id,
            base_prefix_version=prefix_version,
            visible_audio_end_samples=current_audio_end_samples,
            window_audio_samples=new_samples,
            audio_pcm_f32le=audio_bytes,
            stable_prefix_tokens=tuple(stable_prefix_token_ids),
            rollback_tokens=tuple(rollback_draft_token_ids),
            deadline_ns=deadline_ns,
            model_context=deepcopy(model_context),
            sampling_params=deepcopy(sampling_params),
        )
        return coordinator.maybe_start(request, load, predicted_draft_ms)

    def _build_stream_request(self, request, budget):
        params = deepcopy(request.sampling_params)
        params.max_tokens = budget
        params.min_tokens = 0
        params.n = 1
        if getattr(params, "best_of", None) is not None:
            params.best_of = 1
        # Use the service's installed vLLM enum, without a vLLM import here.
        params.output_kind = type(params.output_kind).CUMULATIVE
        kwargs = self._pack_stream_request(
            full_audio_pcm_f32le=request.audio_pcm_f32le,
            stable_prefix_token_ids=list(request.stable_prefix_tokens),
            draft_token_ids=list(request.rollback_tokens[:budget]),
            model_context=deepcopy(request.model_context),
            sampling_params=params,
        )
        if "prompt" not in kwargs or kwargs.get("sampling_params") is not params:
            raise ValueError("packer must return prompt and supplied sampling_params")
        kwargs = dict(kwargs)
        kwargs["request_id"] = request.request_id
        return kwargs

    async def _run_prespec(self, request, budget):
        kwargs = self._build_stream_request(request, budget)
        # Let an already-ready formal/commit callback cancel before submission.
        await asyncio.sleep(0)
        remaining_seconds = (request.deadline_ns - self.prespec.clock()) / 1e9
        if remaining_seconds <= 0:
            raise TimeoutError("prespec deadline passed during request preparation")
        stream = self.prespec_engine.generate(**kwargs)
        finished = False
        candidate_token_ids = ()
        try:
            async with asyncio.timeout(remaining_seconds):
                async for output in stream:
                    if output.outputs:
                        if len(output.outputs) != 1:
                            raise ValueError("prespec requires one output sequence")
                        completion = output.outputs[0]
                        # Cumulative IDs replace the previous snapshot.
                        candidate_token_ids = tuple(completion.token_ids)
                        if completion.finish_reason == "abort":
                            raise RuntimeError("prespec engine request aborted")
                    finished = output.finished
            if not finished or not candidate_token_ids:
                raise RuntimeError("prespec stream ended without a completed candidate")
            if len(candidate_token_ids) > budget:
                raise ValueError("engine output exceeds the candidate token budget")
            return candidate_token_ids
        finally:
            try:
                if not finished:
                    await self.prespec_engine.abort(request.request_id)
            finally:
                await stream.aclose()
