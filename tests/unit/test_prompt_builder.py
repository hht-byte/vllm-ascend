import pytest

from qwen3_asr_window_cache import (
    AUDIO_END,
    AUDIO_PAD,
    AUDIO_PLACEHOLDER,
    AUDIO_START,
    InvalidPromptPlaceholder,
    build_windowed_prompt,
)


def test_replaces_one_placeholder_with_three_anchors() -> None:
    original = "<|im_start|>user\n" + AUDIO_PLACEHOLDER + "<|im_end|>"

    result = build_windowed_prompt(original, window_count=3)

    assert result.count(AUDIO_START) == 1
    assert result.count(AUDIO_END) == 1
    assert result.count(AUDIO_PAD) == 3
    assert result == original.replace(
        AUDIO_PLACEHOLDER, AUDIO_START + AUDIO_PAD * 3 + AUDIO_END
    )


@pytest.mark.parametrize("window_count", [0, -1])
def test_rejects_non_positive_window_count(window_count: int) -> None:
    prompt = "<|im_start|>user\n" + AUDIO_PLACEHOLDER + "<|im_end|>"

    with pytest.raises(InvalidPromptPlaceholder):
        build_windowed_prompt(prompt, window_count=window_count)


@pytest.mark.parametrize(
    "prompt",
    [
        "<|im_start|>user\nhello<|im_end|>",
        "<|im_start|>user\n"
        + AUDIO_PLACEHOLDER
        + "middle"
        + AUDIO_PLACEHOLDER
        + "<|im_end|>",
        "<|im_start|>user\n"
        + AUDIO_START
        + "text"
        + AUDIO_PAD
        + AUDIO_END
        + "<|im_end|>",
        "<|im_start|>user\n"
        + AUDIO_START
        + AUDIO_START
        + AUDIO_PAD
        + AUDIO_END
        + AUDIO_END
        + "<|im_end|>",
        "<|im_start|>user\n"
        + AUDIO_PLACEHOLDER
        + AUDIO_PAD
        + "<|im_end|>",
        "<|im_start|>user\n"
        + AUDIO_START
        + AUDIO_PAD
        + AUDIO_END
        + " stray"
        + AUDIO_END
        + "<|im_end|>",
    ],
)
def test_rejects_missing_repeated_split_nested_or_stray_audio_tokens(
    prompt: str,
) -> None:
    with pytest.raises(InvalidPromptPlaceholder):
        build_windowed_prompt(prompt, window_count=2)


def test_preserves_assistant_text_and_existing_rollback_content() -> None:
    original = (
        "<|im_start|>user\n"
        + AUDIO_PLACEHOLDER
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + "answer with rollback: <rollback>keep byte-for-byte</rollback>"
        + "<|im_end|>"
    )

    result = build_windowed_prompt(original, window_count=2)

    expected = original.replace(
        AUDIO_PLACEHOLDER, AUDIO_START + AUDIO_PAD * 2 + AUDIO_END
    )
    assert result == expected
