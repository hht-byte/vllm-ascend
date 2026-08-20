from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from examples.qwen3_asr_windowed_streaming.async_engine import (
    WindowedAsyncLLMEngineAdapter,
)
from examples.qwen3_asr_windowed_streaming.prompt import Qwen3ASRPromptBuilder
from examples.qwen3_asr_windowed_streaming.window import WindowedAudioState


class RecordingEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], object, str]] = []
        self.active_calls = 0
        self.max_active_calls = 0

    async def generate(
        self,
        prompt: dict[str, Any],
        sampling_params: object,
        request_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        call_index = len(self.calls)
        self.calls.append((prompt, sampling_params, request_id))
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            yield {"request_id": request_id, "partial": True}
            await asyncio.sleep(0.01)
            yield {"request_id": request_id, "text": f"committed-{call_index}"}
        finally:
            self.active_calls -= 1


def make_adapter(engine: RecordingEngine) -> WindowedAsyncLLMEngineAdapter:
    state = WindowedAudioState(
        sample_rate=1,
        step_seconds=2,
        freeze_unit_seconds=4,
        recompute_window_seconds=8,
        cache_namespace="async-test",
    )
    return WindowedAsyncLLMEngineAdapter(
        engine=engine,
        audio_state=state,
        prompt_builder=Qwen3ASRPromptBuilder("p<audio>s", "<audio>"),
        sampling_params={"temperature": 0},
        output_to_committed_text=lambda output: output["text"],
    )


def test_push_uses_unique_requests_and_carries_committed_text() -> None:
    async def run() -> None:
        engine = RecordingEngine()
        adapter = make_adapter(engine)

        results = await adapter.push(np.arange(4, dtype=np.float32))

        assert len(results) == 2
        assert results[0].output == {
            "request_id": results[0].request_id,
            "text": "committed-0",
        }
        assert engine.calls[0][0]["prompt"] == "p<audio>s"
        assert engine.calls[1][0]["prompt"] == "p<audio>scommitted-0"
        assert engine.calls[0][2] != engine.calls[1][2]
        assert adapter.committed_text == "committed-1"

    asyncio.run(run())


def test_concurrent_pushes_are_serialized_per_stream() -> None:
    async def run() -> None:
        engine = RecordingEngine()
        adapter = make_adapter(engine)

        first, second = await asyncio.gather(
            adapter.push(np.arange(2, dtype=np.float32)),
            adapter.push(np.arange(2, 4, dtype=np.float32)),
        )

        assert len(first) == 1
        assert len(second) == 1
        assert engine.max_active_calls == 1

    asyncio.run(run())


def test_flush_executes_one_tail_request_and_then_returns_none() -> None:
    async def run() -> None:
        engine = RecordingEngine()
        adapter = make_adapter(engine)
        assert await adapter.push(np.array([0.5], dtype=np.float32)) == []

        result = await adapter.flush()

        assert result is not None
        assert result.snapshot.end_sample == 1
        assert await adapter.flush() is None
        assert len(engine.calls) == 1

    asyncio.run(run())
