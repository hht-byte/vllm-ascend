"""PCM validation and stable, view-based audio window partitioning."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .errors import (
    AudioTooLong,
    InvalidAudioFormat,
    InvalidSampleRate,
    InvalidWindowSize,
    TooManyAudioWindows,
)

_SAMPLE_RATE = 16_000
_SUPPORTED_WINDOW_SECONDS = frozenset((2, 4, 8))
_MAX_AUDIO_SECONDS = 10
_MAX_AUDIO_WINDOWS = 5


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """One stable PCM segment whose samples are a view of caller-owned audio."""

    index: int
    start_sample: int
    end_sample: int
    samples: NDArray[np.float32]
    sealed: bool


def split_audio_windows(
    audio: np.ndarray,
    *,
    window_sec: int,
    sample_rate: int,
    is_final: bool,
) -> tuple[AudioWindow, ...]:
    """Validate caller-owned PCM and split it into contiguous non-copying views."""
    _validate_pcm(audio)
    if sample_rate != _SAMPLE_RATE:
        raise InvalidSampleRate(f"sample_rate must be {_SAMPLE_RATE}")
    if window_sec not in _SUPPORTED_WINDOW_SECONDS:
        raise InvalidWindowSize("window_sec must be one of 2, 4, or 8")

    window_samples = window_sec * sample_rate
    full_count, remainder = divmod(audio.size, window_samples)
    count = full_count + int(remainder > 0)
    if count > _MAX_AUDIO_WINDOWS:
        raise TooManyAudioWindows("audio must produce at most 5 windows")
    if audio.size > _MAX_AUDIO_SECONDS * sample_rate:
        raise AudioTooLong("audio must not exceed 10 seconds")

    windows: list[AudioWindow] = []
    for index in range(count):
        start = index * window_samples
        end = min(start + window_samples, audio.size)
        sealed = end - start == window_samples or is_final
        windows.append(AudioWindow(index, start, end, audio[start:end], sealed))
    return tuple(windows)


def _validate_pcm(audio: np.ndarray) -> None:
    if (
        not isinstance(audio, np.ndarray)
        or audio.ndim != 1
        or audio.dtype != np.float32
        or not audio.flags.c_contiguous
        or audio.size == 0
    ):
        raise InvalidAudioFormat(
            "audio must be a non-empty mono C-contiguous float32 numpy array"
        )
