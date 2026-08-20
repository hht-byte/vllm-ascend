from __future__ import annotations

import numpy as np
import pytest

from examples.qwen3_asr_windowed_streaming.prompt import Qwen3ASRPromptBuilder
from examples.qwen3_asr_windowed_streaming.window import WindowedAudioState


def snapshot_with_stable_prefix():
    state = WindowedAudioState(
        sample_rate=1,
        step_seconds=2,
        freeze_unit_seconds=4,
        recompute_window_seconds=8,
        cache_namespace="prompt-test",
    )
    return state.push(np.arange(12, dtype=np.float32))[-1]


def test_build_expands_one_placeholder_without_adding_separators() -> None:
    snapshot = snapshot_with_stable_prefix()
    builder = Qwen3ASRPromptBuilder(
        prompt_template=("before<|audio_start|><|audio_pad|><|audio_end|>after"),
        audio_item_placeholder="<|audio_pad|>",
    )

    prompt = builder.build(snapshot, committed_text="committed")

    assert prompt["prompt"] == ("before<|audio_start|><|audio_pad|><|audio_pad|><|audio_end|>aftercommitted")
    assert prompt["prompt"].count("<|audio_start|>") == 1
    assert prompt["prompt"].count("<|audio_end|>") == 1


def test_build_keeps_audio_and_uuid_order_identical() -> None:
    snapshot = snapshot_with_stable_prefix()
    builder = Qwen3ASRPromptBuilder("p<audio>s", "<audio>")

    prompt = builder.build(snapshot)

    audio_items = prompt["multi_modal_data"]["audio"]
    uuid_items = prompt["multi_modal_uuids"]["audio"]
    assert uuid_items == [snapshot.stable[0].cache_id, snapshot.active.cache_id]
    np.testing.assert_array_equal(audio_items[0], snapshot.stable[0].audio)
    np.testing.assert_array_equal(audio_items[1], snapshot.active.audio)


@pytest.mark.parametrize(
    "template",
    ["no placeholder", "<audio>middle<audio>"],
)
def test_builder_rejects_templates_without_exactly_one_placeholder(template: str) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Qwen3ASRPromptBuilder(template, "<audio>")


def test_builder_rejects_empty_placeholder() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Qwen3ASRPromptBuilder("prompt", "")


def test_builder_rejects_full_qwen_audio_placeholder() -> None:
    full_placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"

    with pytest.raises(ValueError, match="inner.*audio_pad"):
        Qwen3ASRPromptBuilder(full_placeholder, full_placeholder)
