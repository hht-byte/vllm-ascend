import math

import pytest

from qwen3_asr_window_cache import ReuseExpectation, floor_reusable_prefix_tokens


@pytest.mark.parametrize(
    ("dirty", "block", "expected"),
    [
        (0, 32, 0),
        (31, 32, 0),
        (32, 32, 32),
        (127, 32, 96),
        (128, 32, 128),
        (133, 32, 128),
        (133, 128, 128),
    ],
)
def test_floor_reusable_prefix_tokens(
    dirty: int, block: int, expected: int
) -> None:
    assert (
        floor_reusable_prefix_tokens(
            dirty_token=dirty,
            hash_block_size=block,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("dirty", "block"),
    [
        (-1, 32),
        (0, 0),
        (0, -1),
        (True, 32),
        (0, False),
        (1.0, 32),
        (0, 32.0),
    ],
)
def test_floor_reusable_prefix_tokens_rejects_invalid_inputs(
    dirty: object, block: object
) -> None:
    with pytest.raises(ValueError):
        floor_reusable_prefix_tokens(
            dirty_token=dirty,  # type: ignore[arg-type]
            hash_block_size=block,  # type: ignore[arg-type]
        )


def test_reuse_expectation_preserves_valid_measurements() -> None:
    expectation = ReuseExpectation(
        sealed_window_count=2,
        open_window_duration_seconds=1.5,
        reusable_prefix_tokens=128,
    )

    assert expectation.sealed_window_count == 2
    assert expectation.open_window_duration_seconds == 1.5
    assert expectation.reusable_prefix_tokens == 128


@pytest.mark.parametrize(
    "values",
    [
        {"sealed_window_count": -1},
        {"sealed_window_count": True},
        {"sealed_window_count": 1.0},
        {"open_window_duration_seconds": -0.1},
        {"open_window_duration_seconds": True},
        {"open_window_duration_seconds": "1.0"},
        {"open_window_duration_seconds": math.inf},
        {"open_window_duration_seconds": math.nan},
        {"reusable_prefix_tokens": -1},
        {"reusable_prefix_tokens": False},
        {"reusable_prefix_tokens": 32.0},
    ],
)
def test_reuse_expectation_rejects_invalid_measurements(
    values: dict[str, object],
) -> None:
    valid: dict[str, object] = {
        "sealed_window_count": 2,
        "open_window_duration_seconds": 1.5,
        "reusable_prefix_tokens": 128,
    }
    valid.update(values)

    with pytest.raises(ValueError):
        ReuseExpectation(**valid)  # type: ignore[arg-type]
