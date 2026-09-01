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
WINDOW_0_EMBEDDING = FakeEmbedding(
    (
        754_877_562,
        856_564_845,
        2_331_986_289,
        2_980_104_974,
        3_092_089_090,
        840_695_093,
        2_636_195_932,
        4_258_176_761,
    )
)
WINDOW_1_OPEN_EMBEDDING = FakeEmbedding(
    (
        449_590_485,
        2_931_416_663,
        3_041_717_679,
        1_001_368_175,
        412_235_671,
        735_791_612,
        3_683_413_941,
        1_579_246_826,
    )
)
WINDOW_1_SEALED_EMBEDDING = FakeEmbedding(
    (
        240_725_864,
        2_985_508_870,
        1_260_481_676,
        3_127_733_307,
        3_101_973_397,
        636_470_997,
        2_122_704_144,
        3_028_504_037,
    )
)
WINDOW_2_OPEN_EMBEDDING = FakeEmbedding(
    (
        471_895_186,
        412_046_352,
        337_618_696,
        3_797_408_179,
        2_987_980_674,
        3_742_471_037,
        1_052_270_972,
        1_782_960_690,
    )
)


def config() -> WindowCacheConfig:
    return WindowCacheConfig()


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
    is_final: bool = False,
) -> dict[str, object]:
    return adapter.build_request(
        session_id=session_id,
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


def test_equal_moment_permutation_exposes_deliberately_stale_encoder_reuse() -> None:
    ordered = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
    ordered[:4] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    permuted = ordered.copy()
    permuted[:4] = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    assert ordered.size == permuted.size
    assert np.sum(ordered, dtype=np.float64) == np.sum(
        permuted,
        dtype=np.float64,
    )
    assert np.dot(ordered.astype(np.float64), ordered.astype(np.float64)) == np.dot(
        permuted.astype(np.float64),
        permuted.astype(np.float64),
    )

    ordered_request = build(
        WindowedRequestAdapter(config()),
        ordered,
        window_sec=2,
        is_final=True,
    )
    permuted_request = build(
        WindowedRequestAdapter(config()),
        permuted,
        window_sec=2,
        is_final=True,
    )
    assert audio_ids(ordered_request) != audio_ids(permuted_request)

    cached = FakeLLM(hash_block_size=4)
    ordered_trace = cached.run(ordered_request)
    normal = cached.run(permuted_request)
    normal_full = FakeLLM(hash_block_size=4).run(
        permuted_request,
        force_full_recompute=True,
    )
    assert normal.encoder == ["miss"]
    assert normal.embeddings == normal_full.embeddings
    assert normal.token_ids == normal_full.token_ids

    stale_key_request = dict(permuted_request)
    stale_key_request["multi_modal_uuids"] = {
        "audio": [audio_ids(ordered_request)[0]],
    }
    stale = cached.run(stale_key_request)
    stale_full = FakeLLM(hash_block_size=4).run(
        stale_key_request,
        force_full_recompute=True,
    )
    assert stale.encoder == ["hit"]
    assert stale.embeddings == ordered_trace.embeddings
    assert stale.embeddings != stale_full.embeddings
    assert stale.token_ids != stale_full.token_ids


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
        WINDOW_0_EMBEDDING,
        WINDOW_1_OPEN_EMBEDDING,
    ]
    assert traces[1].embeddings == [
        WINDOW_0_EMBEDDING,
        WINDOW_1_SEALED_EMBEDDING,
    ]
    assert traces[2].embeddings == [
        WINDOW_0_EMBEDDING,
        WINDOW_1_SEALED_EMBEDDING,
        WINDOW_2_OPEN_EMBEDDING,
    ]
    assert traces[0].token_ids == [
        754_877_562,
        856_564_845,
        2_331_986_289,
        2_980_104_974,
        3_092_089_090,
        840_695_093,
        2_636_195_932,
        4_258_176_761,
        449_590_485,
        2_931_416_663,
        3_041_717_679,
        1_001_368_175,
        412_235_671,
        735_791_612,
        3_683_413_941,
        1_579_246_826,
    ]
    assert traces[1].token_ids == [
        754_877_562,
        856_564_845,
        2_331_986_289,
        2_980_104_974,
        3_092_089_090,
        840_695_093,
        2_636_195_932,
        4_258_176_761,
        240_725_864,
        2_985_508_870,
        1_260_481_676,
        3_127_733_307,
        3_101_973_397,
        636_470_997,
        2_122_704_144,
        3_028_504_037,
    ]
    assert traces[2].token_ids == [
        754_877_562,
        856_564_845,
        2_331_986_289,
        2_980_104_974,
        3_092_089_090,
        840_695_093,
        2_636_195_932,
        4_258_176_761,
        240_725_864,
        2_985_508_870,
        1_260_481_676,
        3_127_733_307,
        3_101_973_397,
        636_470_997,
        2_122_704_144,
        3_028_504_037,
        471_895_186,
        412_046_352,
        337_618_696,
        3_797_408_179,
        2_987_980_674,
        3_742_471_037,
        1_052_270_972,
        1_782_960_690,
    ]


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


def test_distinct_session_identity_does_not_share_encoder_or_prefix_entries() -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(8)
    cached = FakeLLM(encoder_capacity=8, hash_block_size=4)
    requests = [
        build(adapter, audio, window_sec=4, session_id="session-a", is_final=True),
        build(adapter, audio, window_sec=4, session_id="session-b", is_final=True),
    ]

    traces = [cached.run(request) for request in requests]

    assert all(trace.encoder == ["miss", "miss"] for trace in traces)
    assert all(trace.prefix == ["miss", "miss", "miss"] for trace in traces)
    assert len({str(request["cache_salt"]) for request in requests}) == 2
    assert len({identifier for request in requests for identifier in audio_ids(request)}) == 4
    assert traces[0].embeddings == traces[1].embeddings
    assert traces[0].token_ids == traces[1].token_ids


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


def test_incomplete_tail_uuid_change_preserves_complete_block_hit_and_output() -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(8)
    first_request = build(adapter, audio, window_sec=4, is_final=True)
    changed_audio = audio.copy()
    changed_audio[4 * SAMPLE_RATE + 1] += np.float32(0.5)
    changed_request = build(
        WindowedRequestAdapter(config()),
        changed_audio,
        window_sec=4,
        is_final=True,
    )
    assert audio_ids(first_request)[0] == audio_ids(changed_request)[0]
    assert audio_ids(first_request)[1] != audio_ids(changed_request)[1]
    assert first_request["cache_salt"] == changed_request["cache_salt"]
    cached = FakeLLM(hash_block_size=8)

    first = cached.run(first_request)
    changed = cached.run(changed_request)
    recomputed = FakeLLM(hash_block_size=8).run(
        changed_request,
        force_full_recompute=True,
    )

    assert first.prefix == ["miss"]
    assert changed.prefix == ["hit"]
    assert changed.embeddings == recomputed.embeddings
    assert changed.token_ids == recomputed.token_ids


def test_fake_llm_rejects_boolean_cache_sizes() -> None:
    with pytest.raises(ValueError):
        FakeLLM(encoder_capacity=True)

    with pytest.raises(ValueError):
        FakeLLM(hash_block_size=False)
