import numpy as np
import pytest

from qwen3_asr_window_cache import (
    AUDIO_PLACEHOLDER,
    WindowCacheConfig,
    WindowedRequestAdapter,
)
from tests.fakes import FakeEmbedding, FakeLLM

SAMPLE_RATE = 16_000
PROMPT = "transcribe:" + AUDIO_PLACEHOLDER


def config() -> WindowCacheConfig:
    return WindowCacheConfig(
        model_fingerprint="model-v1",
        feature_extractor_fingerprint="extractor-v1",
        audio_encoder_fingerprint="encoder-v1",
    )


def pcm(seconds: int = 10) -> np.ndarray:
    return np.repeat(
        np.arange(seconds, dtype=np.float32),
        SAMPLE_RATE,
    )


def build(
    adapter: WindowedRequestAdapter,
    samples: np.ndarray,
    *,
    window_sec: int,
    session_id: str = "session-a",
    utterance_epoch: int = 1,
    is_final: bool = False,
) -> dict[str, object]:
    return adapter.build_request(
        session_id=session_id,
        utterance_epoch=utterance_epoch,
        accumulated_audio=samples,
        sample_rate=SAMPLE_RATE,
        window_sec=window_sec,
        is_final=is_final,
        prompt=PROMPT,
    )


def audio_ids(request: dict[str, object]) -> list[str]:
    groups = request["multi_modal_uuids"]
    assert isinstance(groups, dict)
    identifiers = groups["audio"]
    assert isinstance(identifiers, list)
    assert all(isinstance(identifier, str) for identifier in identifiers)
    return identifiers


def test_four_second_stream_has_exact_reuse_trace_and_output_equivalence() -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm()
    cached = FakeLLM(hash_block_size=4)
    traces = []

    for seconds in (6, 8, 10):
        request = build(
            adapter,
            audio[: seconds * SAMPLE_RATE],
            window_sec=4,
            is_final=seconds == 10,
        )
        trace = cached.run(request)
        recomputed = FakeLLM(hash_block_size=4).run(
            request,
            force_full_recompute=True,
        )
        assert trace.embeddings == recomputed.embeddings
        assert trace.token_ids == recomputed.token_ids
        traces.append(trace)

    assert traces[0].encoder == ["miss", "miss"]
    assert traces[1].encoder == ["hit", "miss"]
    assert traces[2].encoder == ["hit", "hit", "miss"]
    assert traces[0].prefix == ["miss", "miss", "miss"]
    assert traces[1].prefix == ["hit", "hit", "miss"]
    assert traces[2].prefix == ["hit", "hit", "hit", "miss"]
    assert traces[0].embeddings == [
        FakeEmbedding(64_000, 96_000.0, 224_000.0),
        FakeEmbedding(32_000, 144_000.0, 656_000.0),
    ]
    assert traces[1].embeddings == [
        FakeEmbedding(64_000, 96_000.0, 224_000.0),
        FakeEmbedding(64_000, 352_000.0, 2_016_000.0),
    ]
    assert traces[2].embeddings == [
        FakeEmbedding(64_000, 96_000.0, 224_000.0),
        FakeEmbedding(64_000, 352_000.0, 2_016_000.0),
        FakeEmbedding(32_000, 272_000.0, 2_320_000.0),
    ]
    assert traces[0].token_ids == [155, 502]
    assert traces[1].token_ids == [155, 317]
    assert traces[2].token_ids == [155, 317, 893]


@pytest.mark.parametrize(
    ("window_sec", "expected_encoder", "expected_prefix"),
    [
        (
            2,
            [
                ["miss", "miss", "miss"],
                ["hit", "hit", "hit", "miss"],
                ["hit", "hit", "hit", "hit", "miss"],
            ],
            [
                ["miss", "miss", "miss", "miss"],
                ["hit", "hit", "hit", "hit", "miss"],
                ["hit", "hit", "hit", "hit", "hit", "miss"],
            ],
        ),
        (
            4,
            [
                ["miss", "miss"],
                ["hit", "miss"],
                ["hit", "hit", "miss"],
            ],
            [
                ["miss", "miss", "miss"],
                ["hit", "hit", "miss"],
                ["hit", "hit", "hit", "miss"],
            ],
        ),
        (
            8,
            [
                ["miss"],
                ["miss"],
                ["hit", "miss"],
            ],
            [
                ["miss", "miss"],
                ["hit", "miss"],
                ["hit", "hit", "miss"],
            ],
        ),
    ],
)
def test_window_and_duration_matrix_matches_hand_derived_reuse(
    window_sec: int,
    expected_encoder: list[list[str]],
    expected_prefix: list[list[str]],
) -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm()
    cached = FakeLLM(hash_block_size=4)
    encoder_traces = []
    prefix_traces = []

    for seconds in (6, 8, 10):
        request = build(
            adapter,
            audio[: seconds * SAMPLE_RATE],
            window_sec=window_sec,
            is_final=seconds == 10,
        )
        trace = cached.run(request)
        full = FakeLLM(hash_block_size=4).run(
            request,
            force_full_recompute=True,
        )
        assert trace.embeddings == full.embeddings
        assert trace.token_ids == full.token_ids
        encoder_traces.append(trace.encoder)
        prefix_traces.append(trace.prefix)

    assert encoder_traces == expected_encoder
    assert prefix_traces == expected_prefix


@pytest.mark.parametrize(
    ("encoder_capacity", "expected_retry"),
    [
        (1, ["miss", "miss"]),
        (5, ["hit", "hit"]),
    ],
)
def test_identical_final_retry_exposes_encoder_lru_eviction(
    encoder_capacity: int,
    expected_retry: list[str],
) -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(8)
    cached = FakeLLM(encoder_capacity=encoder_capacity, hash_block_size=4)
    request = build(adapter, audio, window_sec=4, is_final=True)

    first = cached.run(request)
    retry_request = build(adapter, audio, window_sec=4, is_final=True)
    retry = cached.run(retry_request)
    full = FakeLLM(hash_block_size=4).run(
        retry_request,
        force_full_recompute=True,
    )

    assert first.encoder == ["miss", "miss"]
    assert retry.encoder == expected_retry
    assert retry.prefix == ["hit", "hit", "hit"]
    assert retry.embeddings == full.embeddings
    assert retry.token_ids == full.token_ids


def test_session_and_epoch_identity_do_not_share_encoder_or_prefix_entries() -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(8)
    cached = FakeLLM(encoder_capacity=8, hash_block_size=4)
    requests = [
        build(adapter, audio, window_sec=4, session_id="session-a", is_final=True),
        build(adapter, audio, window_sec=4, session_id="session-b", is_final=True),
        build(
            adapter,
            audio,
            window_sec=4,
            session_id="session-a",
            utterance_epoch=2,
            is_final=True,
        ),
    ]

    traces = [cached.run(request) for request in requests]

    assert all(trace.encoder == ["miss", "miss"] for trace in traces)
    assert all(trace.prefix == ["miss", "miss", "miss"] for trace in traces)
    assert len({str(request["cache_salt"]) for request in requests}) == 3
    assert len({identifier for request in requests for identifier in audio_ids(request)}) == 6
    assert traces[0].embeddings == traces[1].embeddings == traces[2].embeddings
    assert traces[0].token_ids == traces[1].token_ids == traces[2].token_ids


def test_historical_sealed_pcm_change_invalidates_affected_encoder_and_kv_tail() -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(8)
    cached = FakeLLM(encoder_capacity=8, hash_block_size=4)
    first_request = build(adapter, audio[: 6 * SAMPLE_RATE], window_sec=2)
    first = cached.run(first_request)
    changed = audio.copy()
    changed[2 * SAMPLE_RATE + 5] += np.float32(0.5)
    changed_request = build(adapter, changed, window_sec=2, is_final=True)

    reused = cached.run(changed_request)
    full = FakeLLM(hash_block_size=4).run(
        changed_request,
        force_full_recompute=True,
    )

    assert first.encoder == ["miss", "miss", "miss"]
    assert reused.encoder == ["hit", "miss", "hit", "miss"]
    assert reused.prefix == ["hit", "hit", "miss", "miss", "miss"]
    assert audio_ids(first_request)[0] == audio_ids(changed_request)[0]
    assert audio_ids(first_request)[1] != audio_ids(changed_request)[1]
    assert audio_ids(first_request)[2] == audio_ids(changed_request)[2]
    assert reused.embeddings[0] == first.embeddings[0]
    assert reused.embeddings[1] != first.embeddings[1]
    assert reused.embeddings[2] == first.embeddings[2]
    assert reused.embeddings == full.embeddings
    assert reused.token_ids == full.token_ids


def test_real_cache_salts_isolate_identical_uuid_and_offset_prefix_blocks() -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(8)
    session_a = build(
        adapter,
        audio,
        window_sec=4,
        session_id="session-a",
        is_final=True,
    )
    session_b = build(
        adapter,
        audio,
        window_sec=4,
        session_id="session-b",
        is_final=True,
    )
    salt_only_request: dict[str, object] = dict(session_a)
    salt_only_request["cache_salt"] = session_b["cache_salt"]
    cached = FakeLLM(hash_block_size=4)

    first = cached.run(session_a)
    isolated = cached.run(salt_only_request)

    assert audio_ids(salt_only_request) == audio_ids(session_a)
    assert salt_only_request["cache_salt"] != session_a["cache_salt"]
    assert first.prefix == ["miss", "miss", "miss"]
    assert isolated.encoder == ["hit", "hit"]
    assert isolated.prefix == ["miss", "miss", "miss"]
    assert isolated.embeddings == first.embeddings
    assert isolated.token_ids == first.token_ids


def test_prefix_cache_hashes_only_complete_configured_blocks() -> None:
    adapter = WindowedRequestAdapter(config())
    request = build(adapter, pcm(8), window_sec=4, is_final=True)
    cached = FakeLLM(hash_block_size=8)

    first = cached.run(request)
    retry = cached.run(request)

    assert first.prefix == ["miss"]
    assert retry.prefix == ["hit"]


def test_fake_llm_rejects_boolean_cache_sizes() -> None:
    with pytest.raises(ValueError):
        FakeLLM(encoder_capacity=True)

    with pytest.raises(ValueError):
        FakeLLM(hash_block_size=False)
