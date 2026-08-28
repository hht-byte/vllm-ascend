"""Content-safe identities for PCM audio windows and cache namespaces."""

import hashlib

import cbor2
import numpy as np

from .errors import InvalidSessionId, InvalidUtteranceEpoch
from .windowing import AudioWindow


def canonical_pcm_digest(samples: np.ndarray) -> bytes:
    """Return SHA-256 of the exact canonical little-endian float32 bytes.

    The input is intentionally not copied, converted, normalized, or traversed
    as Python numbers.  Consequently signed zero and NaN payload bits remain
    part of the content identity.
    """
    if (
        not isinstance(samples, np.ndarray)
        or samples.ndim != 1
        or samples.dtype != np.dtype("<f4")
        or not samples.flags.c_contiguous
        or samples.size == 0
    ):
        raise ValueError(
            "samples must be a non-empty, one-dimensional, C-contiguous "
            "little-endian float32 numpy array"
        )

    byte_view = memoryview(samples).cast("B").toreadonly()
    return hashlib.sha256(byte_view).digest()


def build_session_namespace(
    *, session_id: str, utterance_epoch: int, model_fingerprint: str
) -> str:
    """Build a stable, session- and epoch-scoped cache namespace."""
    session_id, utterance_epoch = validate_session_scope(
        session_id=session_id,
        utterance_epoch=utterance_epoch,
    )
    return _sha256_cbor(
        ("qwen3-asr-session-v1", session_id, utterance_epoch, model_fingerprint)
    )


def validate_session_scope(
    *, session_id: object, utterance_epoch: object
) -> tuple[str, int]:
    """Validate one exact, non-normalized Session/epoch namespace key.

    Whitespace-only identifiers are rejected. Other whitespace is preserved so
    callers cannot accidentally collapse two distinct upstream identifiers.
    """
    if type(session_id) is not str or not session_id.strip():
        raise InvalidSessionId(
            "session_id must be an exact non-empty, non-whitespace string"
        )
    if type(utterance_epoch) is not int or utterance_epoch < 0:
        raise InvalidUtteranceEpoch(
            "utterance_epoch must be an exact non-negative integer"
        )
    return session_id, utterance_epoch


def build_window_id(
    *,
    namespace: str,
    window: AudioWindow,
    window_sec: int,
    feature_extractor_fingerprint: str,
    audio_encoder_fingerprint: str,
    adapter_schema_version: str,
) -> str:
    """Build a stable content and configuration identity for one window."""
    payload = (
        "qwen3-asr-window-v1",
        namespace,
        window.index,
        window_sec,
        window.start_sample,
        window.end_sample,
        canonical_pcm_digest(window.samples),
        feature_extractor_fingerprint,
        audio_encoder_fingerprint,
        adapter_schema_version,
    )
    return _sha256_cbor(payload)


def _sha256_cbor(payload: object) -> str:
    return hashlib.sha256(cbor2.dumps(payload, canonical=True)).hexdigest()
