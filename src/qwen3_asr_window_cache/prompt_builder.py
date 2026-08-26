"""Build native Qwen3-ASR prompts with one anchor per audio window."""

from .errors import InvalidPromptPlaceholder

AUDIO_START = "<|audio_start|>"
AUDIO_PAD = "<|audio_pad|>"
AUDIO_END = "<|audio_end|>"
AUDIO_PLACEHOLDER = AUDIO_START + AUDIO_PAD + AUDIO_END


def build_windowed_prompt(prompt: str, *, window_count: int) -> str:
    """Replace the sole native audio placeholder with window anchors."""
    if window_count <= 0:
        raise InvalidPromptPlaceholder("window_count must be positive")

    if prompt.count(AUDIO_PLACEHOLDER) != 1:
        raise InvalidPromptPlaceholder(
            "prompt must contain exactly one native audio placeholder"
        )

    if (
        prompt.count(AUDIO_START) != 1
        or prompt.count(AUDIO_PAD) != 1
        or prompt.count(AUDIO_END) != 1
    ):
        raise InvalidPromptPlaceholder(
            "prompt must contain exactly one start, pad, and end token"
        )

    return prompt.replace(
        AUDIO_PLACEHOLDER, AUDIO_START + AUDIO_PAD * window_count + AUDIO_END
    )
