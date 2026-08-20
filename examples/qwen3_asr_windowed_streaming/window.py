"""Window and cache identity management for cumulative Qwen3-ASR audio."""

from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatAudio = NDArray[np.float32]


@dataclass(frozen=True)
class AudioSegment:
    """An immutable audio item and its engine-wide cache identity."""

    start_sample: int
    end_sample: int
    audio: FloatAudio
    cache_id: str


@dataclass(frozen=True)
class WindowedAudioSnapshot:
    """The stable prefix and mutable suffix to submit in one engine request."""

    stable: tuple[AudioSegment, ...]
    active: AudioSegment
    end_sample: int


class _AudioChunkBuffer:
    """Store PCM chunks on a global sample timeline."""

    def __init__(self) -> None:
        self._chunks: deque[tuple[int, FloatAudio]] = deque()
        self._start_sample = 0
        self._end_sample = 0

    @property
    def end_sample(self) -> int:
        return self._end_sample

    def append(self, audio: FloatAudio) -> None:
        if audio.size == 0:
            return
        self._chunks.append((self._end_sample, audio))
        self._end_sample += int(audio.size)

    def read(self, start_sample: int, end_sample: int) -> FloatAudio:
        if not self._start_sample <= start_sample < end_sample <= self._end_sample:
            raise ValueError(
                f"Audio range [{start_sample}, {end_sample}) is outside [{self._start_sample}, {self._end_sample})."
            )

        parts: list[FloatAudio] = []
        for chunk_start, chunk in self._chunks:
            chunk_end = chunk_start + int(chunk.size)
            if chunk_end <= start_sample:
                continue
            if chunk_start >= end_sample:
                break
            local_start = max(start_sample, chunk_start) - chunk_start
            local_end = min(end_sample, chunk_end) - chunk_start
            parts.append(chunk[local_start:local_end])

        if not parts:
            raise RuntimeError("Audio buffer lost data required by a snapshot.")
        result = parts[0].copy() if len(parts) == 1 else np.concatenate(parts)
        result.flags.writeable = False
        return result

    def discard_before(self, sample: int) -> None:
        if not self._start_sample <= sample <= self._end_sample:
            raise ValueError(f"Cannot discard through sample {sample}.")

        while self._chunks:
            chunk_start, chunk = self._chunks[0]
            chunk_end = chunk_start + int(chunk.size)
            if chunk_end <= sample:
                self._chunks.popleft()
                continue
            if chunk_start < sample:
                self._chunks[0] = (sample, chunk[sample - chunk_start :])
            break
        self._start_sample = sample


class WindowedAudioState:
    """Build cache-stable snapshots from arbitrary PCM updates."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        step_seconds: float = 2.0,
        freeze_unit_seconds: float = 4.0,
        recompute_window_seconds: float = 8.0,
        cache_namespace: str,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if not cache_namespace:
            raise ValueError("cache_namespace must not be empty.")

        self.sample_rate = sample_rate
        self.step_samples = self._duration_to_samples(step_seconds, "step_seconds")
        self.freeze_unit_samples = self._duration_to_samples(freeze_unit_seconds, "freeze_unit_seconds")
        self.recompute_window_samples = self._duration_to_samples(recompute_window_seconds, "recompute_window_seconds")
        if self.recompute_window_samples < self.freeze_unit_samples:
            raise ValueError("recompute_window_seconds must be at least freeze_unit_seconds.")

        self._cache_namespace = cache_namespace
        self._buffer = _AudioChunkBuffer()
        self._stable: list[AudioSegment] = []
        self._stable_end = 0
        self._next_emit_sample = self.step_samples
        self._last_emitted_sample = 0

    def _duration_to_samples(self, seconds: float, name: str) -> int:
        if seconds <= 0:
            raise ValueError(f"{name} must be positive.")
        exact = seconds * self.sample_rate
        samples = round(exact)
        if not math.isclose(exact, samples, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{name} must resolve to an integral sample count.")
        return samples

    @property
    def total_samples(self) -> int:
        return self._buffer.end_sample

    def push(self, audio: np.ndarray) -> list[WindowedAudioSnapshot]:
        normalized = self._normalize_audio(audio)
        self._buffer.append(normalized)

        snapshots: list[WindowedAudioSnapshot] = []
        while self._next_emit_sample <= self.total_samples:
            snapshots.append(self._make_snapshot(self._next_emit_sample))
            self._last_emitted_sample = self._next_emit_sample
            self._next_emit_sample += self.step_samples
        return snapshots

    def flush(self) -> WindowedAudioSnapshot | None:
        if self.total_samples == self._last_emitted_sample:
            return None
        snapshot = self._make_snapshot(self.total_samples)
        self._last_emitted_sample = self.total_samples
        return snapshot

    def _make_snapshot(self, end_sample: int) -> WindowedAudioSnapshot:
        target_stable_end = (
            max(0, end_sample - self.recompute_window_samples) // self.freeze_unit_samples * self.freeze_unit_samples
        )
        while self._stable_end < target_stable_end:
            segment_end = self._stable_end + self.freeze_unit_samples
            audio = self._buffer.read(self._stable_end, segment_end)
            self._stable.append(
                self._make_segment(
                    self._stable_end,
                    segment_end,
                    audio,
                )
            )
            self._stable_end = segment_end

        self._buffer.discard_before(self._stable_end)
        active_audio = self._buffer.read(self._stable_end, end_sample)
        active = self._make_segment(
            self._stable_end,
            end_sample,
            active_audio,
        )
        return WindowedAudioSnapshot(tuple(self._stable), active, end_sample)

    def _make_segment(
        self,
        start_sample: int,
        end_sample: int,
        audio: FloatAudio,
    ) -> AudioSegment:
        digest = hashlib.sha256()
        digest.update(self._cache_namespace.encode("utf-8"))
        digest.update(b"\0")
        digest.update(self.sample_rate.to_bytes(8, "big", signed=False))
        digest.update(start_sample.to_bytes(8, "big", signed=False))
        digest.update(end_sample.to_bytes(8, "big", signed=False))
        canonical_audio = np.asarray(audio, dtype="<f4")
        digest.update(canonical_audio.tobytes())
        return AudioSegment(start_sample, end_sample, audio, digest.hexdigest())

    @staticmethod
    def _normalize_audio(audio: np.ndarray) -> FloatAudio:
        pcm = np.asarray(audio)
        if pcm.ndim != 1:
            raise ValueError("Expected mono audio as a one-dimensional array.")
        if pcm.dtype == np.int16:
            normalized = pcm.astype(np.float32) / 32768.0
        else:
            # The business service may reuse or mutate its receive buffer after
            # push() returns. Own the samples that have not reached an emission
            # boundary yet so cached content and its hash cannot diverge.
            normalized = np.array(pcm, dtype=np.float32, copy=True, order="C")
        return np.ascontiguousarray(normalized)
