from itertools import pairwise

import numpy as np
import pytest

from qwen3_asr_window_cache.errors import (
    AudioTooLong,
    InvalidAudioFormat,
    InvalidSampleRate,
    InvalidWindowSize,
    TooManyAudioWindows,
)
from qwen3_asr_window_cache.windowing import split_audio_windows


def pcm(seconds: int) -> np.ndarray:
    return np.arange(seconds * 16_000, dtype=np.float32)


def test_four_second_windows_for_six_seconds() -> None:
    pcm_buffer = pcm(6)
    windows = split_audio_windows(
        pcm_buffer, window_sec=4, sample_rate=16_000, is_final=False
    )

    assert [(w.start_sample, w.end_sample, w.sealed) for w in windows] == [
        (0, 64_000, True),
        (64_000, 96_000, False),
    ]
    assert np.shares_memory(windows[0].samples, pcm_buffer)


@pytest.mark.parametrize(
    ("window_sec", "expected_bounds"),
    [
        (2, [(0, 32_000), (32_000, 64_000), (64_000, 96_000), (96_000, 128_000)]),
        (4, [(0, 64_000), (64_000, 128_000)]),
        (8, [(0, 128_000)]),
    ],
)
def test_supported_window_sizes_partition_eight_seconds_without_a_tail(
    window_sec: int,
    expected_bounds: list[tuple[int, int]],
) -> None:
    windows = split_audio_windows(
        pcm(8), window_sec=window_sec, sample_rate=16_000, is_final=False
    )

    assert [(window.start_sample, window.end_sample) for window in windows] == expected_bounds
    assert all(window.sealed for window in windows)
    assert all(window.samples.size > 0 for window in windows)


def test_final_tail_is_sealed() -> None:
    windows = split_audio_windows(
        pcm(10), window_sec=4, sample_rate=16_000, is_final=True
    )

    assert [window.sealed for window in windows] == [True, True, True]


def test_open_tail_is_not_sealed_and_windows_have_no_gaps_or_overlap() -> None:
    pcm_buffer = pcm(10)
    windows = split_audio_windows(
        pcm_buffer, window_sec=4, sample_rate=16_000, is_final=False
    )

    assert [window.index for window in windows] == [0, 1, 2]
    assert windows[-1].sealed is False
    assert np.array_equal(np.concatenate([window.samples for window in windows]), pcm_buffer)
    assert all(
        left.end_sample == right.start_sample for left, right in pairwise(windows)
    )
    assert all(np.shares_memory(window.samples, pcm_buffer) for window in windows)


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros((2, 16_000), dtype=np.float32),
        np.zeros(16_000, dtype=np.float64),
        np.zeros(32_000, dtype=np.float32)[::2],
    ],
)
def test_invalid_pcm_layouts_are_rejected(audio: np.ndarray) -> None:
    with pytest.raises(InvalidAudioFormat):
        split_audio_windows(audio, window_sec=2, sample_rate=16_000, is_final=False)


def test_non_target_sample_rate_is_rejected() -> None:
    with pytest.raises(InvalidSampleRate):
        split_audio_windows(pcm(2), window_sec=2, sample_rate=8_000, is_final=False)


def test_empty_pcm_is_rejected() -> None:
    with pytest.raises(InvalidAudioFormat):
        split_audio_windows(
            np.empty(0, dtype=np.float32),
            window_sec=2,
            sample_rate=16_000,
            is_final=False,
        )


@pytest.mark.parametrize("window_sec", [0, -1, 1, 3, 5, 16])
def test_unsupported_window_size_is_rejected(window_sec: int) -> None:
    with pytest.raises(InvalidWindowSize):
        split_audio_windows(pcm(2), window_sec=window_sec, sample_rate=16_000, is_final=False)


def test_audio_longer_than_ten_seconds_is_rejected() -> None:
    audio = np.zeros(160_001, dtype=np.float32)

    with pytest.raises(AudioTooLong):
        split_audio_windows(audio, window_sec=4, sample_rate=16_000, is_final=False)


def test_more_than_five_windows_is_rejected() -> None:
    audio = np.zeros(160_001, dtype=np.float32)

    with pytest.raises(TooManyAudioWindows):
        split_audio_windows(audio, window_sec=2, sample_rate=16_000, is_final=False)
