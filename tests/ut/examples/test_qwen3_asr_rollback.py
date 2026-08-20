from __future__ import annotations

from examples.qwen3_asr_windowed_streaming.rollback import Qwen3ASRRollbackState


class CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def test_rollback_matches_official_warmup_and_token_trimming() -> None:
    state = Qwen3ASRRollbackState(
        tokenizer=CharacterTokenizer(),
        unfixed_chunk_num=2,
        unfixed_token_num=2,
    )

    assert state.update("abcd") == ""
    assert state.raw_decoded == "abcd"

    assert state.update("efgh") == "ef"
    assert state.raw_decoded == "efgh"

    assert state.update("ijkl") == "efij"
    assert state.raw_decoded == "efijkl"


def test_rollback_can_be_used_as_engine_output_callback() -> None:
    state = Qwen3ASRRollbackState(
        tokenizer=CharacterTokenizer(),
        output_to_generated_text=lambda output: output["text"],
        unfixed_chunk_num=1,
        unfixed_token_num=1,
    )

    assert state({"text": "abc"}) == "ab"
    assert state.raw_decoded == "abc"


def test_rollback_avoids_incomplete_unicode_decode() -> None:
    class ReplacementTokenizer(CharacterTokenizer):
        def decode(self, token_ids: list[int]) -> str:
            text = super().decode(token_ids)
            return text + ("\ufffd" if len(token_ids) == 3 else "")

    state = Qwen3ASRRollbackState(
        tokenizer=ReplacementTokenizer(),
        unfixed_chunk_num=1,
        unfixed_token_num=1,
    )

    assert state.update("abcd") == "ab"


def test_rollback_rejects_invalid_settings_and_callback_output() -> None:
    try:
        Qwen3ASRRollbackState(CharacterTokenizer(), unfixed_chunk_num=-1)
    except ValueError as error:
        assert "unfixed_chunk_num" in str(error)
    else:
        raise AssertionError("negative unfixed_chunk_num was accepted")

    state = Qwen3ASRRollbackState(
        CharacterTokenizer(),
        output_to_generated_text=lambda output: 1,
    )
    try:
        state(object())
    except TypeError as error:
        assert "must return str" in str(error)
    else:
        raise AssertionError("non-string generated text was accepted")
