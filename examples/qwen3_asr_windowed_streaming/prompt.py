"""Construct cache-stable Qwen3-ASR prompts for independent requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .window import WindowedAudioSnapshot

_AUDIO_WRAPPER_TOKENS = ("<|audio_start|>", "<|audio_end|>")


@dataclass(frozen=True)
class Qwen3ASRPromptBuilder:
    """Expand one logical audio span into adjacent cached audio items.

    ``audio_item_placeholder`` is Qwen3-ASR's inner ``<|audio_pad|>`` token,
    not the full ``<|audio_start|>...<|audio_end|>`` placeholder returned by
    ``get_placeholder_str``. Keeping the wrapper in ``prompt_template`` avoids
    inserting text tokens between frozen and active audio items.
    """

    prompt_template: str
    audio_item_placeholder: str

    def __post_init__(self) -> None:
        if not self.audio_item_placeholder:
            raise ValueError("audio_item_placeholder must not be empty.")
        if any(token in self.audio_item_placeholder for token in _AUDIO_WRAPPER_TOKENS):
            raise ValueError(
                "audio_item_placeholder must be the inner <|audio_pad|> token, "
                "not the full Qwen3-ASR audio placeholder."
            )
        if self.prompt_template.count(self.audio_item_placeholder) != 1:
            raise ValueError("prompt_template must contain exactly one audio item placeholder.")

    def build(
        self,
        snapshot: WindowedAudioSnapshot,
        committed_text: str = "",
    ) -> dict[str, Any]:
        items = (*snapshot.stable, snapshot.active)
        expanded_audio = self.audio_item_placeholder * len(items)
        prompt = self.prompt_template.replace(
            self.audio_item_placeholder,
            expanded_audio,
            1,
        )
        return {
            "prompt": prompt + committed_text,
            "multi_modal_data": {"audio": [item.audio for item in items]},
            "multi_modal_uuids": {"audio": [item.cache_id for item in items]},
        }
