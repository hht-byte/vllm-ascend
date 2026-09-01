from __future__ import annotations

import numpy as np

from qwen3_asr_window_cache import (
    AUDIO_PLACEHOLDER,
    AudioWindow,
    WindowCacheConfig,
    WindowedRequestAdapter,
    build_session_namespace,
    build_window_id,
)


def test_config_needs_no_runtime_fingerprints() -> None:
    assert WindowCacheConfig().sample_rate == 16_000


def test_session_namespace_only_requires_session_id() -> None:
    assert build_session_namespace(session_id="session-a") == build_session_namespace(
        session_id="session-a"
    )


def test_window_identity_uses_index_and_pcm_not_location_metadata() -> None:
    samples = np.arange(32, dtype=np.float32)
    first = AudioWindow(0, 0, 32, samples, True)
    relocated = AudioWindow(0, 1_000, 1_032, samples, False)

    namespace = build_session_namespace(session_id="session-a")
    assert build_window_id(namespace=namespace, window=first) == build_window_id(
        namespace=namespace,
        window=relocated,
    )
    assert build_window_id(
        namespace=namespace,
        window=AudioWindow(1, 0, 32, samples, True),
    ) != build_window_id(namespace=namespace, window=first)


def test_adapter_lifecycle_only_requires_session_id() -> None:
    adapter = WindowedRequestAdapter(WindowCacheConfig())
    audio = np.zeros(32_000, dtype=np.float32)

    request = adapter.build_request(
        session_id="session-a",
        accumulated_audio=audio,
        sample_rate=16_000,
        window_sec=2,
        is_final=True,
        prompt=AUDIO_PLACEHOLDER,
    )
    adapter.release_session("session-a")

    assert request["cache_salt"] == build_session_namespace(session_id="session-a")
