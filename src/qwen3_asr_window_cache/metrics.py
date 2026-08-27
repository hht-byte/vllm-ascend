"""Pure measurements for reasoning about window and prefix reuse."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReuseExpectation:
    """Expected reusable work for one accumulated-audio request."""

    sealed_window_count: int
    open_window_duration_seconds: float
    reusable_prefix_tokens: int

    def __post_init__(self) -> None:
        if (
            type(self.sealed_window_count) is not int
            or self.sealed_window_count < 0
        ):
            raise ValueError("sealed_window_count must be a non-negative integer")
        if (
            isinstance(self.open_window_duration_seconds, bool)
            or not isinstance(self.open_window_duration_seconds, (int, float))
            or not math.isfinite(self.open_window_duration_seconds)
            or self.open_window_duration_seconds < 0
        ):
            raise ValueError(
                "open_window_duration_seconds must be a finite non-negative number"
            )
        if (
            type(self.reusable_prefix_tokens) is not int
            or self.reusable_prefix_tokens < 0
        ):
            raise ValueError("reusable_prefix_tokens must be a non-negative integer")


def floor_reusable_prefix_tokens(*, dirty_token: int, hash_block_size: int) -> int:
    """Floor a dirty-token boundary to complete reusable hash blocks."""
    if type(dirty_token) is not int or dirty_token < 0:
        raise ValueError("dirty_token must be a non-negative integer")
    if type(hash_block_size) is not int or hash_block_size <= 0:
        raise ValueError("hash_block_size must be a positive integer")
    return dirty_token // hash_block_size * hash_block_size
