"""Content-safe identities for PCM audio windows and cache namespaces."""

import hashlib

import cbor2
import numpy as np

from .errors import InvalidSessionId
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


def build_session_namespace(*, session_id: str) -> str:
    """Build a stable cache namespace for one serialized business Session."""
    session_id = validate_session_id(session_id)
    return _sha256_cbor(("qwen3-asr-session-v2", session_id))


def validate_session_id(session_id: object) -> str:
    """Validate one exact, non-normalized Session namespace key.

    Whitespace-only identifiers are rejected. Other whitespace is preserved so
    callers cannot accidentally collapse two distinct upstream identifiers.
    """
    if type(session_id) is not str or not session_id.strip():
        raise InvalidSessionId(
            "session_id must be an exact non-empty, non-whitespace string"
        )
    return session_id


def build_window_id(
    *,
    namespace: str,
    window: AudioWindow,
) -> str:
    """Build a stable per-item identity from Session, occurrence and PCM."""
    payload = (
        "qwen3-asr-window-v2",
        namespace,
        window.index,
        canonical_pcm_digest(window.samples),
    )
    return _sha256_cbor(payload)


def _sha256_cbor(payload: object) -> str:
    return hashlib.sha256(cbor2.dumps(payload, canonical=True)).hexdigest()
