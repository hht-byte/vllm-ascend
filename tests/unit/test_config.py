from typing import Any

import pytest

from qwen3_asr_window_cache.config import WindowCacheConfig


def config(**overrides: Any) -> WindowCacheConfig:
    settings: dict[str, Any] = {
        "model_fingerprint": "model-v1",
        "feature_extractor_fingerprint": "extractor-v1",
        "audio_encoder_fingerprint": "encoder-v1",
    }
    settings.update(overrides)
    return WindowCacheConfig(**settings)


def test_default_configuration_preserves_target_limits() -> None:
    result = config()

    assert result.supported_window_seconds == (2, 4, 8)
    assert result.sample_rate == 16_000
    assert result.max_audio_seconds == 10
    assert result.max_audio_windows == 5


@pytest.mark.parametrize(
    "field",
    [
        "model_fingerprint",
        "feature_extractor_fingerprint",
        "audio_encoder_fingerprint",
    ],
)
def test_blank_fingerprints_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        config(**{field: "  "})


@pytest.mark.parametrize("sample_rate", [0, -1, 8_000, 44_100])
def test_non_target_sample_rates_are_rejected(sample_rate: int) -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        config(sample_rate=sample_rate)


@pytest.mark.parametrize(
    "supported_window_seconds",
    [(), (2, 3), (2, 2), (0, 2), (2, 4, 8, 16)],
)
def test_invalid_supported_window_sets_are_rejected(
    supported_window_seconds: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="supported_window_seconds"):
        config(supported_window_seconds=supported_window_seconds)


@pytest.mark.parametrize(
    ("max_audio_seconds", "max_audio_windows"),
    [(0, 5), (11, 5), (10, 0), (10, 6)],
)
def test_target_limit_bounds_are_enforced(
    max_audio_seconds: int,
    max_audio_windows: int,
) -> None:
    with pytest.raises(ValueError, match="max_audio"):
        config(
            max_audio_seconds=max_audio_seconds,
            max_audio_windows=max_audio_windows,
        )


def test_maximum_window_count_must_cover_configured_audio_length() -> None:
    with pytest.raises(ValueError, match="cover"):
        config(
            supported_window_seconds=(4, 8),
            max_audio_seconds=10,
            max_audio_windows=2,
        )


def test_restricted_supported_windows_can_cover_a_smaller_limit() -> None:
    result = config(
        supported_window_seconds=(4, 8),
        max_audio_seconds=8,
        max_audio_windows=2,
    )

    assert result.supported_window_seconds == (4, 8)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_rate", 16_000.0),
        ("max_audio_seconds", 2.5),
        ("max_audio_windows", 2.5),
        ("supported_window_seconds", (2.0,)),
        ("supported_window_seconds", [2, 4]),
    ],
)
def test_configuration_values_must_use_the_declared_integer_and_tuple_types(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        config(**{field: value})
