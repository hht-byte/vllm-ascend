"""Copyable model helpers; the service supplies its existing request packer.

No vLLM import is needed to run mock tests. The engine contract is the
AsyncLLMEngine generate/abort API, with cumulative RequestOutput token IDs.
See streaming_prespec.md section 4 for the one service-specific extraction.
"""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

if __package__:
    from .streaming_prespec import (
        AdmissionPolicy, LaunchDecision, PreSpecCoordinator, ShadowRequest,
        launch_samples_for_window,
    )
else:
    from streaming_prespec import (
        AdmissionPolicy, LaunchDecision, PreSpecCoordinator, ShadowRequest,
        launch_samples_for_window,
    )


@dataclass(frozen=True, kw_only=True)
class ModelShadowRequest(ShadowRequest):
    # Private copies of small request settings, never the live ASR state/engine.
    prefix_text: str
    model_context: dict[str, Any]
    sampling_params: Any


def snapshot_stream_audio(audio_accum, buffer) -> tuple[bytes, int, int]:
    """Snapshot all consumed full chunks plus the unconsumed audio tail.

    Returns complete 16 kHz float32 LE PCM, its sample count, and the last
    formal chunk cutoff. Both counts use segment-relative sample coordinates.
    """
    consumed = np.asarray(audio_accum, dtype=np.float32).reshape(-1)
    pending = np.asarray(buffer, dtype=np.float32).reshape(-1)
    complete = np.concatenate((consumed, pending)).astype("<f4", copy=False)
    return complete.tobytes(), int(complete.size), int(consumed.size)


class PreSpecModelHelpers:
    """Mixin for Qwen3ASRModelAsync, including startup configuration.

    Pass the existing shared engine to ``enable_prespec`` before accepting connections.
    Supply ``_pack_stream_request`` as documented; it must not mutate live state.
    ``prespec`` is the existing PreSpecCoordinator, or None when disabled.
    The service MTP master defaults off. ``prespec_mtp_enabled`` takes effect
    only with that master on and the engine MTP extension loaded.
    """

    def enable_prespec(
        self, *, enabled, kmax, engine=None, chunk_size_sec=2.0,
        launch_sec=1.6, candidate_tokens=None, on_probe=None,
        admission_mode="force", shadow_only=True, max_active_shadows=0,
        rejoin_policy="ngram_1", mtp_enabled=False, prespec_mtp_enabled=True,
        producer=None,
    ):
        """Configure once before accepting connections, using the shared engine.

        AsyncLLM must explicitly consume prespec_mtp_enabled, then pass it here.
        A service with a logged producer can pass that bound method as producer.
        """
        if type(mtp_enabled) is not bool:
            raise TypeError("mtp_enabled must be a bool")
        if type(prespec_mtp_enabled) is not bool:
            raise TypeError("prespec_mtp_enabled must be a bool")
        if rejoin_policy not in ("aligned", "ngram_1", "ngram_2", "ngram_3"):
            raise ValueError("invalid prespec rejoin policy")
        self.mtp_enabled = mtp_enabled
        self.prespec_mtp_enabled = mtp_enabled and prespec_mtp_enabled
        self.prespec_rejoin_policy = rejoin_policy
        self.prespec = None
        if enabled:
            if kmax <= 0 or chunk_size_sec <= 0:
                raise ValueError("prespec requires positive kmax and chunk size")
            if candidate_tokens is None:
                candidate_tokens = kmax
            if candidate_tokens < 3 or candidate_tokens > kmax:
                raise ValueError("candidate budget must be between 3 and engine kmax")
            if engine is None or not callable(getattr(self, "_pack_stream_request", None)):
                raise ValueError("prespec requires engine and the existing request packer")
            self.prespec_engine = engine
            self.prespec = PreSpecCoordinator(
                producer=self._run_prespec if producer is None else producer,
                policy=AdmissionPolicy(
                    admission_mode=admission_mode,
                    midpoint_samples=launch_samples_for_window(launch_sec, chunk_size_sec),
                    max_rollback_draft_tokens=kmax,
                    max_candidate_tokens=candidate_tokens,
                    max_active_shadows=max_active_shadows,
                ),
                shadow_only=shadow_only, on_probe=on_probe,
            )
        logger.info(
            "PRESPEC_MTP_CONFIG prespec_enabled=%s mtp_enabled=%s prespec_mtp_enabled=%s",
            enabled, self.mtp_enabled, self.prespec_mtp_enabled,
        )

    def try_start_prespec(
        self, *, ws_id, seg_id, window_id, prefix_version, full_audio_pcm_f32le,
        current_audio_end_samples, last_formal_audio_end_samples,
        chunk_size_samples, prefix_text, stable_prefix_token_ids,
        rollback_draft_token_ids,
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
        draft_len = len(rollback_draft_token_ids)
        if draft_len > coordinator.policy.max_rollback_draft_tokens:
            if coordinator.should_log_budget_skip(
                    ws_id, seg_id, window_id, "rollback_draft_exceeds_kmax"):
                logger.info(
                    "PRESPEC_SKIP ws_id=%s seg=%s window=%s reason=rollback_draft_exceeds_kmax "
                    "draft_tokens=%s kmax=%s",
                    ws_id, seg_id, window_id, draft_len,
                    coordinator.policy.max_rollback_draft_tokens,
                )
            return LaunchDecision(False, "rollback_draft_exceeds_kmax")
        if draft_len + 2 > coordinator.policy.max_candidate_tokens:
            if coordinator.should_log_budget_skip(
                    ws_id, seg_id, window_id, "candidate_budget_insufficient"):
                logger.info(
                    "PRESPEC_SKIP ws_id=%s seg=%s window=%s reason=candidate_budget_insufficient "
                    "draft_tokens=%s candidate_budget=%s required=%s",
                    ws_id, seg_id, window_id, draft_len,
                    coordinator.policy.max_candidate_tokens, draft_len + 2,
                )
            return LaunchDecision(False, "candidate_budget_insufficient")
        # Take the time before copying so snapshot overhead consumes the budget.
        deadline_ns = coordinator.clock() + int(
            (chunk_size_samples - new_samples) / 16000 * 1e9)
        audio_bytes = bytes(full_audio_pcm_f32le)
        if not audio_bytes or len(audio_bytes) % 4:
            raise ValueError("audio must be complete mono float32 LE PCM")
        if len(audio_bytes) // 4 != current_audio_end_samples:
            raise ValueError("audio bytes and sample cutoff disagree")
        request = ModelShadowRequest(
            session_id=ws_id, audio_epoch=seg_id, window_id=window_id,
            base_prefix_version=prefix_version,
            visible_audio_end_samples=current_audio_end_samples,
            window_audio_samples=new_samples,
            audio_pcm_f32le=audio_bytes,
            stable_prefix_tokens=tuple(stable_prefix_token_ids),
            rollback_tokens=tuple(rollback_draft_token_ids),
            deadline_ns=deadline_ns,
            prefix_text=prefix_text,
            model_context=deepcopy(model_context),
            sampling_params=deepcopy(sampling_params),
        )
        return coordinator.maybe_start(request, load, predicted_draft_ms)

    def _build_stream_request(self, request, budget):
        shadow_mtp = getattr(self, "prespec_mtp_enabled", True)
        # Preserve older copied helpers that only had the shadow flag, while
        # ensuring an explicitly disabled service master always wins.
        master_mtp = getattr(self, "mtp_enabled", True)
        if type(master_mtp) is not bool:
            raise TypeError("mtp_enabled must be a bool")
        if type(shadow_mtp) is not bool:
            raise TypeError("prespec_mtp_enabled must be a bool")
        shadow_mtp = master_mtp and shadow_mtp
        if len(request.rollback_tokens) > self.prespec.policy.max_rollback_draft_tokens:
            raise ValueError("rollback draft exceeds engine kmax")
        if len(request.rollback_tokens) + 2 > budget:
            raise ValueError("candidate budget cannot cover first token, old draft and extension")
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
            prefix_text=request.prefix_text,
            stable_prefix_token_ids=list(request.stable_prefix_tokens),
            draft_token_ids=list(request.rollback_tokens),
            model_context=deepcopy(request.model_context),
            sampling_params=params,
        )
        if "prompt" not in kwargs or kwargs.get("sampling_params") is not params:
            raise ValueError("packer must return prompt and supplied sampling_params")
        # This shadow still verifies the original legacy rollback after its
        # target drive token. Inherited formal flags would first-verify C[1:]
        # at the wrong position, or enable formal-only rejoin behavior.
        extra = dict(getattr(params, "extra_args", None) or {})
        for key in ("ascend_external_draft_first_verify", "ascend_external_draft_rejoin",
                    "ascend_external_draft_ngram_after_first_mismatch"):
            extra.pop(key, None)
        extra["ascend_mtp"] = shadow_mtp
        params.extra_args = extra
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
                        # Publish verified cumulative output before a later
                        # timeout or formal boundary cancels this task.
                        self.prespec.publish_partial(request, candidate_token_ids)
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
