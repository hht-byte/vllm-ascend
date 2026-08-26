"""Stable public domain API for Qwen3-ASR window caching."""

from .config import WindowCacheConfig
from .errors import (
    AudioLengthRegressed,
    AudioTooLong,
    InvalidAudioFormat,
    InvalidPromptPlaceholder,
    InvalidSampleRate,
    InvalidWindowSize,
    SessionAlreadyFinished,
    TooManyAudioWindows,
    WindowCacheError,
    WindowConfigChanged,
)
from .windowing import AudioWindow, split_audio_windows

__all__ = [
    "AudioLengthRegressed",
    "AudioTooLong",
    "AudioWindow",
    "InvalidAudioFormat",
    "InvalidPromptPlaceholder",
    "InvalidSampleRate",
    "InvalidWindowSize",
    "SessionAlreadyFinished",
    "TooManyAudioWindows",
    "WindowCacheConfig",
    "WindowCacheError",
    "WindowConfigChanged",
    "split_audio_windows",
]
