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
from .identity import build_session_namespace, build_window_id, canonical_pcm_digest
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
    "build_session_namespace",
    "build_window_id",
    "canonical_pcm_digest",
    "split_audio_windows",
]
