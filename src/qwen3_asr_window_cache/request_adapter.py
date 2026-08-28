"""Stateful assembly of session-safe vLLM audio requests."""

import hashlib
from dataclasses import dataclass
from threading import RLock

import cbor2
import numpy as np

from .config import WindowCacheConfig
from .errors import (
    AudioLengthRegressed,
    AudioTooLong,
    InvalidWindowSize,
    SessionAlreadyFinished,
    TooManyAudioWindows,
    WindowConfigChanged,
)
from .identity import (
    build_session_namespace,
    build_window_id,
    canonical_pcm_digest,
    validate_session_scope,
)
from .prompt_builder import build_windowed_prompt
from .windowing import split_audio_windows


@dataclass(frozen=True, slots=True)
class _SessionState:
    window_sec: int
    model_fingerprint: str
    last_sample_count: int
    last_audio_digest: bytes
    finished: bool
    final_request_digest: bytes | None


class WindowedRequestAdapter:
    """Build PromptType-shaped requests while retaining CPU metadata only.

    Audio windows borrow views of ``accumulated_audio``. The caller must not
    mutate that array in place until vLLM has consumed the request input.
    Streaming services should replace their accumulated buffer rather than
    mutate a buffer that has already been submitted.
    """

    def __init__(self, config: WindowCacheConfig) -> None:
        self._config = config
        self._states: dict[tuple[str, int], _SessionState] = {}
        self._lock = RLock()

    def build_request(
        self,
        *,
        session_id: str,
        utterance_epoch: int,
        accumulated_audio: np.ndarray,
        sample_rate: int,
        window_sec: int,
        is_final: bool,
        prompt: str,
    ) -> dict[str, object]:
        """Validate and assemble one request, then atomically commit metadata."""
        session_id, utterance_epoch = validate_session_scope(
            session_id=session_id,
            utterance_epoch=utterance_epoch,
        )
        key = (session_id, utterance_epoch)
        with self._lock:
            if (
                type(window_sec) is not int
                or window_sec not in self._config.supported_window_seconds
            ):
                raise InvalidWindowSize(
                    "window_sec must be enabled by supported_window_seconds"
                )

            windows = split_audio_windows(
                accumulated_audio,
                window_sec=window_sec,
                sample_rate=sample_rate,
                is_final=is_final,
            )
            if accumulated_audio.size > self._config.max_audio_seconds * sample_rate:
                raise AudioTooLong(
                    "audio must not exceed the configured maximum duration"
                )
            if len(windows) > self._config.max_audio_windows:
                raise TooManyAudioWindows(
                    "audio must not exceed the configured maximum window count"
                )

            windowed_prompt = build_windowed_prompt(prompt, window_count=len(windows))
            namespace = build_session_namespace(
                session_id=session_id,
                utterance_epoch=utterance_epoch,
                model_fingerprint=self._config.model_fingerprint,
            )
            window_ids = [
                build_window_id(
                    namespace=namespace,
                    window=window,
                    window_sec=window_sec,
                    feature_extractor_fingerprint=(
                        self._config.feature_extractor_fingerprint
                    ),
                    audio_encoder_fingerprint=(self._config.audio_encoder_fingerprint),
                    adapter_schema_version=self._config.adapter_schema_version,
                )
                for window in windows
            ]
            audio_digest = canonical_pcm_digest(accumulated_audio)
            request_digest = self._request_digest(
                session_id=session_id,
                utterance_epoch=utterance_epoch,
                sample_rate=sample_rate,
                window_sec=window_sec,
                is_final=is_final,
                audio_digest=audio_digest,
                prompt=prompt,
            )
            result: dict[str, object] = {
                "prompt": windowed_prompt,
                "multi_modal_data": {"audio": [window.samples for window in windows]},
                "multi_modal_uuids": {"audio": window_ids},
                "cache_salt": namespace,
            }

            previous = self._states.get(key)
            if previous is not None:
                if previous.finished:
                    if request_digest != previous.final_request_digest:
                        raise SessionAlreadyFinished(
                            "finished session accepts only an identical final retry"
                        )
                    return result
                if (
                    previous.window_sec != window_sec
                    or previous.model_fingerprint != self._config.model_fingerprint
                ):
                    raise WindowConfigChanged(
                        "window or model configuration changed within an epoch"
                    )
                if accumulated_audio.size < previous.last_sample_count:
                    raise AudioLengthRegressed(
                        "accumulated audio became shorter within an epoch"
                    )

            self._states[key] = _SessionState(
                window_sec=window_sec,
                model_fingerprint=self._config.model_fingerprint,
                last_sample_count=accumulated_audio.size,
                last_audio_digest=audio_digest,
                finished=is_final,
                final_request_digest=request_digest if is_final else None,
            )
            return result

    def release_session(self, session_id: str, utterance_epoch: int) -> None:
        """Idempotently discard CPU metadata for one utterance."""
        session_id, utterance_epoch = validate_session_scope(
            session_id=session_id,
            utterance_epoch=utterance_epoch,
        )
        with self._lock:
            self._states.pop((session_id, utterance_epoch), None)

    def _request_digest(
        self,
        *,
        session_id: str,
        utterance_epoch: int,
        sample_rate: int,
        window_sec: int,
        is_final: bool,
        audio_digest: bytes,
        prompt: str,
    ) -> bytes:
        payload = (
            "qwen3-asr-final-request-v1",
            session_id,
            utterance_epoch,
            sample_rate,
            window_sec,
            is_final,
            audio_digest,
            prompt,
            self._config.model_fingerprint,
            self._config.feature_extractor_fingerprint,
            self._config.audio_encoder_fingerprint,
            self._config.adapter_schema_version,
        )
        return hashlib.sha256(cbor2.dumps(payload, canonical=True)).digest()
