import hashlib
from dataclasses import replace

import cbor2
import numpy as np
import pytest

from qwen3_asr_window_cache.errors import InvalidSessionId
from qwen3_asr_window_cache.identity import (
    build_session_namespace,
    build_window_id,
    canonical_pcm_digest,
)
from qwen3_asr_window_cache.windowing import AudioWindow, split_audio_windows


def pcm(seconds: int) -> np.ndarray:
    return np.arange(seconds * 16_000, dtype="<f4")


def ids_for(
    audio: np.ndarray,
    session_id: str = "u1",
    window_sec: int = 4,
) -> tuple[str, ...]:
    namespace = build_session_namespace(session_id=session_id)
    windows = split_audio_windows(
        audio, window_sec=window_sec, sample_rate=16_000, is_final=False
    )
    return tuple(
        build_window_id(namespace=namespace, window=window) for window in windows
    )


@pytest.mark.parametrize("session_id", ["", "   ", 1, True, None])
def test_session_namespace_rejects_invalid_exact_session_ids(session_id: object) -> None:
    with pytest.raises(InvalidSessionId):
        build_session_namespace(session_id=session_id)  # type: ignore[arg-type]


def test_session_namespace_preserves_nonblank_whitespace_without_normalizing() -> None:
    assert build_session_namespace(
        session_id=" session-a "
    ) != build_session_namespace(session_id="session-a")


def test_stable_windows_reuse_ids_while_open_tail_changes() -> None:
    first = ids_for(pcm(6))
    second = ids_for(pcm(8))
    assert first[0] == second[0]
    assert first[1] != second[1]

    changed = pcm(8)
    changed[3] += 1
    assert ids_for(changed)[0] != second[0]
    assert ids_for(pcm(8), session_id="u2")[0] != second[0]


def test_window_identity_depends_on_item_index_but_not_location_metadata() -> None:
    window = split_audio_windows(
        pcm(4), window_sec=4, sample_rate=16_000, is_final=True
    )[0]
    namespace = build_session_namespace(session_id="u1")
    baseline = build_window_id(namespace=namespace, window=window)

    assert build_window_id(
        namespace=namespace,
        window=replace(window, start_sample=123, end_sample=456, sealed=False),
    ) == baseline
    assert build_window_id(
        namespace=namespace,
        window=replace(window, index=1),
    ) != baseline


def test_session_namespace_is_sha256_of_versioned_canonical_tuple() -> None:
    expected = hashlib.sha256(
        cbor2.dumps(("qwen3-asr-session-v2", "u1"), canonical=True)
    ).hexdigest()
    actual = build_session_namespace(session_id="u1")
    assert actual == expected
    assert len(actual) == 64


def test_window_id_is_sha256_of_minimal_versioned_tuple() -> None:
    window = split_audio_windows(
        pcm(4), window_sec=4, sample_rate=16_000, is_final=True
    )[0]
    namespace = build_session_namespace(session_id="u1")
    pcm_digest = hashlib.sha256(memoryview(window.samples).cast("B")).digest()
    expected = hashlib.sha256(
        cbor2.dumps(
            ("qwen3-asr-window-v2", namespace, window.index, pcm_digest),
            canonical=True,
        )
    ).hexdigest()
    assert build_window_id(namespace=namespace, window=window) == expected


def test_canonical_pcm_digest_hashes_little_endian_float32_bytes() -> None:
    samples = np.array([1.0, -2.5, 0.0], dtype="<f4")
    expected = hashlib.sha256(memoryview(samples).cast("B")).digest()
    assert canonical_pcm_digest(samples) == expected


@pytest.mark.parametrize(
    "samples",
    [
        np.array([1.0], dtype="<f8"),
        np.array([[1.0]], dtype="<f4"),
        np.arange(8, dtype="<f4")[::2],
        np.array([1.0], dtype=">f4"),
        [1.0],
    ],
)
def test_canonical_pcm_digest_rejects_noncanonical_pcm_layouts(
    samples: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        canonical_pcm_digest(samples)  # type: ignore[arg-type]


def test_canonical_pcm_digest_preserves_exact_float_bits() -> None:
    assert canonical_pcm_digest(
        np.array([-0.0], dtype="<f4")
    ) != canonical_pcm_digest(np.array([0.0], dtype="<f4"))
    first = np.array([0x7FC00001], dtype="<u4").view("<f4")
    second = np.array([0x7FC00002], dtype="<u4").view("<f4")
    assert canonical_pcm_digest(first) != canonical_pcm_digest(second)


def test_audio_window_identity_accepts_the_existing_window_type() -> None:
    window = AudioWindow(0, 0, 2, np.zeros(2, dtype="<f4"), True)
    result = build_window_id(
        namespace=build_session_namespace(session_id="u1"),
        window=window,
    )
    assert isinstance(result, str)
