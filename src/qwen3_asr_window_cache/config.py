"""Immutable configuration for the stable Qwen3-ASR window domain."""

from dataclasses import dataclass

_TARGET_SAMPLE_RATE = 16_000
_SUPPORTED_WINDOW_SECONDS = frozenset((2, 4, 8))
_MAX_AUDIO_SECONDS = 10
_MAX_AUDIO_WINDOWS = 5


@dataclass(frozen=True, slots=True)
class WindowCacheConfig:
    """Validated CPU-only audio window limits."""

    supported_window_seconds: tuple[int, ...] = (2, 4, 8)
    sample_rate: int = _TARGET_SAMPLE_RATE
    max_audio_seconds: int = _MAX_AUDIO_SECONDS
    max_audio_windows: int = _MAX_AUDIO_WINDOWS

    def __post_init__(self) -> None:
        self._validate_limits()

    def _validate_limits(self) -> None:
        if type(self.sample_rate) is not int or self.sample_rate != _TARGET_SAMPLE_RATE:
            raise ValueError(f"sample_rate must be {_TARGET_SAMPLE_RATE}")
        if (
            not isinstance(self.supported_window_seconds, tuple)
            or not self.supported_window_seconds
            or any(type(window) is not int for window in self.supported_window_seconds)
        ):
            raise ValueError("supported_window_seconds must be a non-empty tuple of integers")
        if (
            len(set(self.supported_window_seconds)) != len(self.supported_window_seconds)
            or any(window not in _SUPPORTED_WINDOW_SECONDS for window in self.supported_window_seconds)
        ):
            raise ValueError("supported_window_seconds must be unique values from (2, 4, 8)")
        if (
            type(self.max_audio_seconds) is not int
            or not 0 < self.max_audio_seconds <= _MAX_AUDIO_SECONDS
        ):
            raise ValueError("max_audio_seconds must be between 1 and 10")
        if (
            type(self.max_audio_windows) is not int
            or not 0 < self.max_audio_windows <= _MAX_AUDIO_WINDOWS
        ):
            raise ValueError("max_audio_windows must be between 1 and 5")
        if min(self.supported_window_seconds) * self.max_audio_windows < self.max_audio_seconds:
            raise ValueError("max_audio_windows must cover max_audio_seconds")
