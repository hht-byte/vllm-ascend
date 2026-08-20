from __future__ import annotations

import numpy as np
import pytest

from examples.qwen3_asr_windowed_streaming.window import WindowedAudioState


def make_state(*, sample_rate: int = 1) -> WindowedAudioState:
    return WindowedAudioState(
        sample_rate=sample_rate,
        step_seconds=2,
        freeze_unit_seconds=4,
        recompute_window_seconds=8,
        cache_namespace="qwen3-asr-test",
    )


def test_push_normalizes_int16_and_emits_at_step_boundary() -> None:
    state = make_state(sample_rate=2)

    snapshots = state.push(np.array([-32768, 0, 16384, 32767], dtype=np.int16))

    assert len(snapshots) == 1
    active = snapshots[0].active.audio
    assert active.dtype == np.float32
    np.testing.assert_allclose(
        active,
        np.array([-1.0, 0.0, 0.5, 32767 / 32768], dtype=np.float32),
    )


def test_push_emits_each_elapsed_step_and_promotes_only_aligned_history() -> None:
    state = make_state()

    snapshots = state.push(np.arange(12, dtype=np.float32))

    assert [snapshot.end_sample for snapshot in snapshots] == [2, 4, 6, 8, 10, 12]
    assert snapshots[-2].stable == ()
    assert [(item.start_sample, item.end_sample) for item in snapshots[-1].stable] == [(0, 4)]
    assert (snapshots[-1].active.start_sample, snapshots[-1].active.end_sample) == (
        4,
        12,
    )
    np.testing.assert_array_equal(snapshots[-1].active.audio, np.arange(4, 12))


def test_frozen_ids_stay_stable_while_active_id_changes() -> None:
    state = make_state()
    first = state.push(np.arange(12, dtype=np.float32))[-1]

    second = state.push(np.arange(12, 14, dtype=np.float32))[-1]

    assert first.stable[0].cache_id == second.stable[0].cache_id
    assert first.active.cache_id != second.active.cache_id


def test_promotion_reuses_identical_earlier_active_item_id() -> None:
    state = make_state()
    active_at_four_seconds = state.push(np.arange(4, dtype=np.float32))[-1].active

    promoted_at_twelve_seconds = state.push(np.arange(4, 12, dtype=np.float32))[-1].stable[0]

    assert promoted_at_twelve_seconds.cache_id == active_at_four_seconds.cache_id


def test_later_promotions_keep_prior_segments_and_bound_active_window() -> None:
    state = make_state()

    snapshot = state.push(np.arange(16, dtype=np.float32))[-1]

    assert [(item.start_sample, item.end_sample) for item in snapshot.stable] == [
        (0, 4),
        (4, 8),
    ]
    assert (snapshot.active.start_sample, snapshot.active.end_sample) == (8, 16)


def test_flush_emits_one_short_tail_then_becomes_idempotent() -> None:
    state = make_state()
    assert state.push(np.array([0.25], dtype=np.float32)) == []

    snapshot = state.flush()

    assert snapshot is not None
    assert snapshot.end_sample == 1
    np.testing.assert_array_equal(snapshot.active.audio, np.array([0.25], dtype=np.float32))
    assert state.flush() is None


def test_push_rejects_multi_channel_audio() -> None:
    state = make_state()

    with pytest.raises(ValueError, match="mono"):
        state.push(np.zeros((2, 4), dtype=np.float32))


def test_push_owns_unemitted_input_audio() -> None:
    state = make_state()
    caller_buffer = np.array([0.25], dtype=np.float32)

    assert state.push(caller_buffer) == []
    caller_buffer[0] = 99.0
    snapshot = state.push(np.array([0.5], dtype=np.float32))[0]

    np.testing.assert_array_equal(
        snapshot.active.audio,
        np.array([0.25, 0.5], dtype=np.float32),
    )
