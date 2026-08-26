import hashlib
from dataclasses import replace

import cbor2
import numpy as np
import pytest

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
    epoch: int = 7,
    window_sec: int = 4,
    *,
    model_fingerprint: str = "model-a",
    feature_extractor_fingerprint: str = "feature-a",
    audio_encoder_fingerprint: str = "encoder-a",
    adapter_schema_version: str = "schema-a",
) -> tuple[str, ...]:
    namespace = build_session_namespace(
        session_id=session_id,
        utterance_epoch=epoch,
        model_fingerprint=model_fingerprint,
    )
    windows = split_audio_windows(
        audio, window_sec=window_sec, sample_rate=16_000, is_final=False
    )
    return tuple(
        build_window_id(
            namespace=namespace,
            window=window,
            window_sec=window_sec,
            feature_extractor_fingerprint=feature_extractor_fingerprint,
            audio_encoder_fingerprint=audio_encoder_fingerprint,
            adapter_schema_version=adapter_schema_version,
        )
        for window in windows
    )


def test_stable_windows_reuse_ids_while_open_tail_changes() -> None:
    first = ids_for(pcm(6))
    second = ids_for(pcm(8))

    assert first[0] == second[0]
    assert first[1] != second[1]

    changed = pcm(8)
    changed[3] += 1
    assert ids_for(changed)[0] != second[0]
    assert ids_for(pcm(8), session_id="u2")[0] != second[0]
    assert ids_for(pcm(8), epoch=8)[0] != second[0]


@pytest.mark.parametrize(
    "dimension",
    [
        "window_index",
        "window_sec",
        "start_sample",
        "end_sample",
        "feature_extractor_fingerprint",
        "audio_encoder_fingerprint",
        "adapter_schema_version",
    ],
)
def test_each_window_identity_dimension_invalidates(
    dimension: str,
) -> None:
    audio = pcm(8)
    namespace = build_session_namespace(
        session_id="u1", utterance_epoch=7, model_fingerprint="model-a"
    )
    window = split_audio_windows(
        audio, window_sec=4, sample_rate=16_000, is_final=False
    )[0]
    kwargs = {
        "namespace": namespace,
        "window": window,
        "window_sec": 4,
        "feature_extractor_fingerprint": "feature-a",
        "audio_encoder_fingerprint": "encoder-a",
        "adapter_schema_version": "schema-a",
    }
    baseline = build_window_id(**kwargs)

    if dimension == "window_index":
        kwargs["window"] = replace(window, index=1)
    elif dimension == "window_sec":
        kwargs["window_sec"] = 2
    elif dimension == "start_sample":
        kwargs["window"] = replace(window, start_sample=1)
    elif dimension == "end_sample":
        kwargs["window"] = replace(window, end_sample=window.end_sample - 1)
    elif dimension == "feature_extractor_fingerprint":
        kwargs[dimension] = "feature-b"
    elif dimension == "audio_encoder_fingerprint":
        kwargs[dimension] = "encoder-b"
    else:
        kwargs[dimension] = "schema-b"

    assert build_window_id(**kwargs) != baseline


def test_window_id_changes_when_namespace_changes() -> None:
    audio = pcm(4)
    window = split_audio_windows(
        audio, window_sec=4, sample_rate=16_000, is_final=False
    )[0]
    namespace = build_session_namespace(
        session_id="u1", utterance_epoch=7, model_fingerprint="model-a"
    )
    other_namespace = build_session_namespace(
        session_id="u1", utterance_epoch=7, model_fingerprint="model-b"
    )
    kwargs = {
        "window": window,
        "window_sec": 4,
        "feature_extractor_fingerprint": "feature-a",
        "audio_encoder_fingerprint": "encoder-a",
        "adapter_schema_version": "schema-a",
    }

    assert build_window_id(namespace=namespace, **kwargs) != build_window_id(
        namespace=other_namespace, **kwargs
    )


def test_session_namespace_is_sha256_of_versioned_canonical_tuple() -> None:
    expected = hashlib.sha256(
        cbor2.dumps(("qwen3-asr-session-v1", "u1", 7, "model-a"), canonical=True)
    ).hexdigest()

    actual = build_session_namespace(
        session_id="u1", utterance_epoch=7, model_fingerprint="model-a"
    )

    assert actual == expected
    assert len(actual) == 64
    assert actual == actual.lower()


def test_window_id_is_sha256_of_versioned_canonical_tuple() -> None:
    audio = pcm(4)
    window = split_audio_windows(
        audio, window_sec=4, sample_rate=16_000, is_final=True
    )[0]
    namespace = build_session_namespace(
        session_id="u1", utterance_epoch=7, model_fingerprint="model-a"
    )
    pcm_digest = hashlib.sha256(memoryview(window.samples).cast("B")).digest()
    payload = (
        "qwen3-asr-window-v1",
        namespace,
        window.index,
        4,
        window.start_sample,
        window.end_sample,
        pcm_digest,
        "feature-a",
        "encoder-a",
        "schema-a",
    )
    expected = hashlib.sha256(cbor2.dumps(payload, canonical=True)).hexdigest()

    actual = build_window_id(
        namespace=namespace,
        window=window,
        window_sec=4,
        feature_extractor_fingerprint="feature-a",
        audio_encoder_fingerprint="encoder-a",
        adapter_schema_version="schema-a",
    )

    assert actual == expected
    assert len(actual) == 64
    assert actual == actual.lower()


def test_canonical_pcm_digest_hashes_little_endian_float32_bytes() -> None:
    samples = np.array([1.0, -2.5, 0.0], dtype="<f4")
    expected = hashlib.sha256(memoryview(samples).cast("B")).digest()

    assert canonical_pcm_digest(samples) == expected
    assert isinstance(canonical_pcm_digest(samples), bytes)


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


def test_canonical_pcm_digest_does_not_normalize_signed_zero() -> None:
    negative_zero = np.array([-0.0], dtype="<f4")
    positive_zero = np.array([0.0], dtype="<f4")

    assert canonical_pcm_digest(negative_zero) != canonical_pcm_digest(positive_zero)


def test_canonical_pcm_digest_preserves_nan_payload_bytes() -> None:
    first = np.array([0x7FC00001], dtype="<u4").view("<f4")
    second = np.array([0x7FC00002], dtype="<u4").view("<f4")

    assert np.isnan(first[0]) and np.isnan(second[0])
    assert canonical_pcm_digest(first) != canonical_pcm_digest(second)


def test_audio_window_identity_accepts_the_existing_window_type() -> None:
    window = AudioWindow(0, 0, 2, np.zeros(2, dtype="<f4"), True)
    namespace = build_session_namespace(
        session_id="u1", utterance_epoch=7, model_fingerprint="model-a"
    )

    result = build_window_id(
        namespace=namespace,
        window=window,
        window_sec=2,
        feature_extractor_fingerprint="feature-a",
        audio_encoder_fingerprint="encoder-a",
        adapter_schema_version="schema-a",
    )

    assert isinstance(result, str)
