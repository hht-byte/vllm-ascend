"""Stable domain errors raised by the window-cache adapter."""


class WindowCacheError(Exception):
    """Base class for recoverable window-cache domain failures."""


class InvalidAudioFormat(WindowCacheError):
    """Raised when PCM is not a non-empty mono C-contiguous float32 array."""


class InvalidSampleRate(WindowCacheError):
    """Raised when PCM is not sampled at the supported 16 kHz rate."""


class InvalidWindowSize(WindowCacheError):
    """Raised when a window duration is not one of the stable target sizes."""


class AudioTooLong(WindowCacheError):
    """Raised when accumulated PCM exceeds the ten-second safety bound."""


class TooManyAudioWindows(WindowCacheError):
    """Raised when the request would contain more than five audio items."""


class AudioLengthRegressed(WindowCacheError):
    """Raised when accumulated PCM becomes shorter within one utterance epoch."""


class WindowConfigChanged(WindowCacheError):
    """Raised when stable window or model identity changes within an epoch."""


class InvalidPromptPlaceholder(WindowCacheError):
    """Raised when the native audio placeholder cannot be replaced safely."""


class SessionAlreadyFinished(WindowCacheError):
    """Raised when audio is appended after a final utterance request."""
