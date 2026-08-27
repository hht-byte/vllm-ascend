"""Stable public domain API for Qwen3-ASR window caching."""

from .compatibility import validate_runtime_versions
from .config import WindowCacheConfig
from .engine_config import prepare_vllm_config
from .errors import (
    AudioLengthRegressed,
    AudioTooLong,
    InvalidAudioFormat,
    InvalidEngineConfiguration,
    InvalidPromptPlaceholder,
    InvalidSampleRate,
    InvalidWindowSize,
    SessionAlreadyFinished,
    TooManyAudioWindows,
    UnsupportedRuntimeVersion,
    WindowCacheError,
    WindowConfigChanged,
)
from .identity import build_session_namespace, build_window_id, canonical_pcm_digest
from .metrics import ReuseExpectation, floor_reusable_prefix_tokens
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
    "InvalidEngineConfiguration",
    "InvalidPromptPlaceholder",
    "InvalidSampleRate",
    "InvalidWindowSize",
    "ReuseExpectation",
    "SessionAlreadyFinished",
    "TooManyAudioWindows",
    "UnsupportedRuntimeVersion",
    "WindowCacheConfig",
    "WindowCacheError",
    "WindowConfigChanged",
    "WindowedRequestAdapter",
    "build_session_namespace",
    "build_window_id",
    "build_windowed_prompt",
    "canonical_pcm_digest",
    "floor_reusable_prefix_tokens",
    "prepare_vllm_config",
    "split_audio_windows",
    "validate_runtime_versions",
]
