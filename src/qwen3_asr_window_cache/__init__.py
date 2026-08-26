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
from .prompt_builder import (
    AUDIO_END,
    AUDIO_PAD,
    AUDIO_PLACEHOLDER,
    AUDIO_START,
    build_windowed_prompt,
)
from .request_adapter import WindowedRequestAdapter
from .windowing import AudioWindow, split_audio_windows

__all__ = [
    "AUDIO_END",
    "AUDIO_PAD",
    "AUDIO_PLACEHOLDER",
    "AUDIO_START",
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
    "WindowedRequestAdapter",
    "build_session_namespace",
    "build_window_id",
    "build_windowed_prompt",
    "canonical_pcm_digest",
    "split_audio_windows",
]
