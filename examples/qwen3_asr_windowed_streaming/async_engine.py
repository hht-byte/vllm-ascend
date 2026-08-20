"""AsyncLLMEngine adapter for independent windowed Qwen3-ASR requests."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .prompt import Qwen3ASRPromptBuilder
from .window import WindowedAudioSnapshot, WindowedAudioState


class AsyncGenerateEngine(Protocol):
    """The subset shared by vLLM 0.23.0's AsyncLLM and AsyncLLMEngine alias."""

    def generate(
        self,
        prompt: dict[str, Any],
        sampling_params: object,
        request_id: str,
    ) -> AsyncIterator[Any]: ...


@dataclass(frozen=True)
class WindowedInferenceResult:
    """The final engine output and inputs for one audio snapshot."""

    request_id: str
    snapshot: WindowedAudioSnapshot
    prompt: dict[str, Any]
    output: Any


class WindowedAsyncLLMEngineAdapter:
    """Submit cache-stable independent requests to one long-lived engine."""

    def __init__(
        self,
        *,
        engine: AsyncGenerateEngine,
        audio_state: WindowedAudioState,
        prompt_builder: Qwen3ASRPromptBuilder,
        sampling_params: object,
        output_to_committed_text: Callable[[Any], str] | None = None,
        request_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._engine = engine
        self._audio_state = audio_state
        self._prompt_builder = prompt_builder
        self._sampling_params = sampling_params
        self._output_to_committed_text = output_to_committed_text
        self._request_id_factory = request_id_factory or (lambda: uuid.uuid4().hex)
        self._issued_request_ids: set[str] = set()
        self._committed_text = ""
        self._lock = asyncio.Lock()

    @property
    def committed_text(self) -> str:
        return self._committed_text

    async def push(self, audio: np.ndarray) -> list[WindowedInferenceResult]:
        async with self._lock:
            snapshots = self._audio_state.push(audio)
            results: list[WindowedInferenceResult] = []
            for snapshot in snapshots:
                results.append(await self._execute(snapshot))
            return results

    async def flush(self) -> WindowedInferenceResult | None:
        async with self._lock:
            snapshot = self._audio_state.flush()
            if snapshot is None:
                return None
            return await self._execute(snapshot)

    async def _execute(
        self,
        snapshot: WindowedAudioSnapshot,
    ) -> WindowedInferenceResult:
        prompt = self._prompt_builder.build(snapshot, self._committed_text)
        request_id = self._request_id_factory()
        if request_id in self._issued_request_ids:
            raise ValueError(f"request_id_factory returned duplicate ID {request_id!r}.")
        self._issued_request_ids.add(request_id)

        final_output: Any = None
        received_output = False
        async for output in self._engine.generate(
            prompt,
            self._sampling_params,
            request_id=request_id,
        ):
            final_output = output
            received_output = True
        if not received_output:
            raise RuntimeError(f"AsyncLLMEngine produced no output for {request_id}.")

        if self._output_to_committed_text is not None:
            committed_text = self._output_to_committed_text(final_output)
            if not isinstance(committed_text, str):
                raise TypeError("output_to_committed_text must return str.")
            self._committed_text = committed_text

        return WindowedInferenceResult(
            request_id=request_id,
            snapshot=snapshot,
            prompt=prompt,
            output=final_output,
        )
