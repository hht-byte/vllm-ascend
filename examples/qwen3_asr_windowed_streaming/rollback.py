"""Official-style unstable-token rollback for Qwen3-ASR streaming."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class TextTokenizer(Protocol):
    """Tokenizer operations used by the streaming rollback policy."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


class Qwen3ASRRollbackState:
    """Track raw decoded text and return the next stable prompt prefix.

    This mirrors Qwen-ASR's realtime warmup and token rollback policy while
    remaining independent from its synchronous inference wrapper.
    """

    def __init__(
        self,
        tokenizer: TextTokenizer,
        *,
        output_to_generated_text: Callable[[Any], str] | None = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
    ) -> None:
        if unfixed_chunk_num < 0:
            raise ValueError("unfixed_chunk_num must not be negative.")
        if unfixed_token_num < 0:
            raise ValueError("unfixed_token_num must not be negative.")

        self._tokenizer = tokenizer
        self._output_to_generated_text = output_to_generated_text
        self._unfixed_chunk_num = unfixed_chunk_num
        self._unfixed_token_num = unfixed_token_num
        self.chunk_id = 0
        self.prefix = ""
        self.raw_decoded = ""

    def update(self, generated_text: str) -> str:
        """Consume one generation and return the next request's prefix."""
        if not isinstance(generated_text, str):
            raise TypeError("generated_text must be str.")

        self.raw_decoded = self.prefix + generated_text
        self.chunk_id += 1
        if self.chunk_id < self._unfixed_chunk_num:
            self.prefix = ""
        else:
            self.prefix = self._rollback(self.raw_decoded)
        return self.prefix

    def __call__(self, output: Any) -> str:
        """Adapt an engine output directly to the adapter callback contract."""
        if self._output_to_generated_text is None:
            if not isinstance(output, str):
                raise TypeError("output_to_generated_text is required for non-string outputs.")
            generated_text = output
        else:
            generated_text = self._output_to_generated_text(output)
            if not isinstance(generated_text, str):
                raise TypeError("output_to_generated_text must return str.")
        return self.update(generated_text)

    def _rollback(self, raw_decoded: str) -> str:
        token_ids = self._tokenizer.encode(raw_decoded)
        rollback_tokens = self._unfixed_token_num
        while True:
            end_idx = max(0, len(token_ids) - rollback_tokens)
            prefix = self._tokenizer.decode(token_ids[:end_idx]) if end_idx else ""
            if "\ufffd" not in prefix or end_idx == 0:
                return prefix
            rollback_tokens += 1
