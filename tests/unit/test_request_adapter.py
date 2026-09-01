from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import pytest

from qwen3_asr_window_cache import (
    AUDIO_END,
    AUDIO_PAD,
    AUDIO_PLACEHOLDER,
    AUDIO_START,
    AudioLengthRegressed,
    AudioTooLong,
    InvalidPromptPlaceholder,
    InvalidSampleRate,
    InvalidSessionId,
    InvalidWindowSize,
    SessionAlreadyFinished,
    WindowCacheConfig,
    WindowConfigChanged,
    WindowedRequestAdapter,
)
from qwen3_asr_window_cache import request_adapter as request_adapter_module


def config(**overrides: Any) -> WindowCacheConfig:
    settings: dict[str, Any] = {}
    settings.update(overrides)
    return WindowCacheConfig(**settings)


def pcm(seconds: int) -> np.ndarray:
    return np.arange(seconds * 16_000, dtype=np.float32)


def build(
    adapter: WindowedRequestAdapter,
    samples: np.ndarray,
    *,
    session_id: str = "session-a",
    sample_rate: int = 16_000,
    window_sec: int = 4,
    is_final: bool = False,
    prompt: str = AUDIO_PLACEHOLDER,
) -> dict[str, object]:
    return adapter.build_request(
        session_id=session_id,
        accumulated_audio=samples,
        sample_rate=sample_rate,
        window_sec=window_sec,
        is_final=is_final,
        prompt=prompt,
    )


def audio_items(result: dict[str, object]) -> list[np.ndarray]:
    data = result["multi_modal_data"]
    assert isinstance(data, dict)
    items = data["audio"]
    assert isinstance(items, list)
    return items


def audio_ids(result: dict[str, object]) -> list[str]:
    uuids = result["multi_modal_uuids"]
    assert isinstance(uuids, dict)
    identifiers = uuids["audio"]
    assert isinstance(identifiers, list)
    return identifiers


def test_builds_direct_vllm_prompt_across_growing_windows() -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(10)
    original_prompt = "prefix" + AUDIO_PLACEHOLDER + "suffix"

    six = build(adapter, audio[:96_000], prompt=original_prompt)
    eight = build(adapter, audio[:128_000], prompt=original_prompt)
    ten = build(adapter, audio, is_final=True, prompt=original_prompt)

    assert set(ten) == {
        "prompt",
        "multi_modal_data",
        "multi_modal_uuids",
        "cache_salt",
    }
    assert ten["prompt"] == (
        "prefix" + AUDIO_START + AUDIO_PAD * 3 + AUDIO_END + "suffix"
    )
    assert len(audio_items(six)) == len(audio_ids(six)) == 2
    assert len(audio_items(eight)) == len(audio_ids(eight)) == 2
    assert len(audio_items(ten)) == len(audio_ids(ten)) == 3
    assert six["cache_salt"] == eight["cache_salt"] == ten["cache_salt"]
    assert audio_ids(six)[0] == audio_ids(eight)[0] == audio_ids(ten)[0]
    assert audio_ids(six)[1] != audio_ids(eight)[1]
    assert audio_ids(eight)[:2] == audio_ids(ten)[:2]
    assert all(isinstance(item, np.ndarray) for item in audio_items(ten))
    assert all(item.ndim == 1 for item in audio_items(ten))
    assert all(np.shares_memory(item, audio) for item in audio_items(ten))


def test_rejects_accumulated_length_regression_without_advancing_state() -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(8)
    build(adapter, audio[:96_000])

    with pytest.raises(AudioLengthRegressed):
        build(adapter, audio[:64_000])

    with pytest.raises(AudioLengthRegressed):
        build(adapter, audio[:80_000])

    result = build(adapter, audio[:112_000])
    assert len(audio_items(result)) == 2


def test_rejects_window_change_without_advancing_state() -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(8)
    build(adapter, audio[:64_000], window_sec=4)

    with pytest.raises(WindowConfigChanged):
        build(adapter, audio[:96_000], window_sec=2)

    result = build(adapter, audio[:80_000], window_sec=4)
    assert len(audio_items(result)) == 2


def test_identical_final_request_is_an_idempotent_retry() -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(6)

    first = build(adapter, audio, is_final=True)
    retry = build(adapter, audio, is_final=True)

    assert retry["prompt"] == first["prompt"]
    assert retry["cache_salt"] == first["cache_salt"]
    assert audio_ids(retry) == audio_ids(first)
    assert all(
        np.array_equal(retry_item, first_item)
        for retry_item, first_item in zip(
            audio_items(retry), audio_items(first), strict=True
        )
    )


@pytest.mark.parametrize("changed_field", ["audio", "prompt", "final_flag"])
def test_finished_session_rejects_every_nonidentical_retry(changed_field: str) -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(6)
    build(adapter, audio, is_final=True)

    retry_audio = audio
    retry_prompt = AUDIO_PLACEHOLDER
    retry_is_final = True
    if changed_field == "audio":
        retry_audio = audio.copy()
        retry_audio[0] += 1
    elif changed_field == "prompt":
        retry_prompt = "changed" + AUDIO_PLACEHOLDER
    else:
        retry_is_final = False

    with pytest.raises(SessionAlreadyFinished):
        build(
            adapter,
            retry_audio,
            is_final=retry_is_final,
            prompt=retry_prompt,
        )


def test_finished_session_rejects_appended_audio() -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(8)
    build(adapter, audio[:96_000], is_final=True)

    with pytest.raises(SessionAlreadyFinished):
        build(adapter, audio, is_final=True)


@pytest.mark.parametrize(
    ("changed", "error"),
    [
        ({"sample_rate": 8_000}, InvalidSampleRate),
        ({"window_sec": 2}, SessionAlreadyFinished),
    ],
)
def test_finished_session_rejects_nonidentical_request_configuration(
    changed: dict[str, object], error: type[Exception]
) -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(4)
    build(adapter, audio, is_final=True)

    with pytest.raises(error):
        build(adapter, audio, is_final=True, **changed)  # type: ignore[arg-type]


def test_release_is_idempotent_and_allows_the_same_key_to_start_fresh() -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(6)
    build(adapter, audio, is_final=True)

    adapter.release_session("session-a")
    adapter.release_session("session-a")

    result = build(adapter, audio[:32_000], window_sec=2)
    assert len(audio_items(result)) == 1


@pytest.mark.parametrize("invalid_session_id", ["", "  ", 1, True, None])
def test_invalid_session_id_cannot_create_or_release_session_state(
    invalid_session_id: object,
) -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(8)
    final = build(adapter, audio[:96_000], is_final=True)

    with pytest.raises(InvalidSessionId):
        build(
            adapter,
            audio[:96_000],
            session_id=invalid_session_id,  # type: ignore[arg-type]
            is_final=True,
        )
    with pytest.raises(InvalidSessionId):
        adapter.release_session(
            invalid_session_id,  # type: ignore[arg-type]
        )

    assert audio_ids(build(adapter, audio[:96_000], is_final=True)) == audio_ids(final)
    with pytest.raises(SessionAlreadyFinished):
        build(adapter, audio, is_final=True)


def test_session_state_isolated_and_release_starts_next_utterance() -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(8)
    build(adapter, audio[:96_000], is_final=True)

    adapter.release_session("session-a")
    next_utterance = build(adapter, audio[:32_000], window_sec=2)
    other_session = build(
        adapter,
        audio[:32_000],
        session_id="session-b",
        window_sec=2,
    )

    assert len(audio_items(next_utterance)) == 1
    assert len(audio_items(other_session)) == 1
    assert next_utterance["cache_salt"] != other_session["cache_salt"]


def test_changed_historical_pcm_invalidates_only_affected_window_on_growth() -> None:
    adapter = WindowedRequestAdapter(config())
    initial = pcm(8)
    first = build(adapter, initial)
    grown = pcm(10)
    grown[3] += 1

    second = build(adapter, grown)

    assert audio_ids(second)[0] != audio_ids(first)[0]
    assert audio_ids(second)[1] == audio_ids(first)[1]


def test_same_length_changed_pcm_invalidates_only_affected_window() -> None:
    adapter = WindowedRequestAdapter(config())
    initial = pcm(8)
    first = build(adapter, initial)
    changed = initial.copy()
    changed[64_003] += 1

    second = build(adapter, changed)

    assert audio_ids(second)[0] == audio_ids(first)[0]
    assert audio_ids(second)[1] != audio_ids(first)[1]


def test_invalid_prompt_on_new_session_does_not_create_partial_state() -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(8)

    with pytest.raises(InvalidPromptPlaceholder):
        build(adapter, audio, prompt="no audio here")

    result = build(adapter, audio[:32_000], window_sec=2)
    assert len(audio_items(result)) == 1


def test_invalid_prompt_on_existing_session_does_not_advance_state() -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(8)
    build(adapter, audio[:64_000])

    with pytest.raises(InvalidPromptPlaceholder):
        build(adapter, audio[:96_000], prompt="no audio here")

    result = build(adapter, audio[:80_000])
    assert len(audio_items(result)) == 2


def test_identity_failure_does_not_create_partial_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = WindowedRequestAdapter(config())
    audio = pcm(8)

    def fail_identity(**_: object) -> str:
        raise RuntimeError("identity failed")

    monkeypatch.setattr(request_adapter_module, "build_window_id", fail_identity)
    with pytest.raises(RuntimeError, match="identity failed"):
        build(adapter, audio)

    monkeypatch.undo()
    result = build(adapter, audio[:32_000], window_sec=2)
    assert len(audio_items(result)) == 1


def test_adapter_honors_restricted_configuration_limits() -> None:
    adapter = WindowedRequestAdapter(
        config(
            supported_window_seconds=(4, 8),
            max_audio_seconds=8,
            max_audio_windows=2,
        )
    )

    with pytest.raises(InvalidWindowSize):
        build(adapter, pcm(2), window_sec=2)
    with pytest.raises(AudioTooLong):
        build(adapter, pcm(10), window_sec=4)

    result = build(adapter, pcm(8), window_sec=4)
    assert len(audio_items(result)) == 2


def test_adapter_state_keeps_only_small_cpu_metadata() -> None:
    adapter = WindowedRequestAdapter(config())
    build(adapter, pcm(6))

    def nested_values(value: object) -> list[object]:
        values = [value]
        if isinstance(value, dict):
            for key, item in value.items():
                values.extend(nested_values(key))
                values.extend(nested_values(item))
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                values.extend(nested_values(item))
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                values.extend(nested_values(getattr(value, field.name)))
        return values

    owned_values = nested_values(adapter.__dict__)
    assert not any(isinstance(value, np.ndarray) for value in owned_values)
