"""Replay complete ASR utterances with independent greedy and MTP histories."""

from __future__ import annotations

from collections.abc import Callable, Sequence


def _without_eos(tokens: Sequence[int], eos: set[int]) -> list[int]:
    kept = []
    for token in tokens:
        if token in eos:
            break
        kept.append(token)
    return kept


def replay_utterance(utterance: dict, decode: Callable, decode_text: Callable,
                     eos_token_ids: set[int], *, rollback_tokens: int = 5,
                     unfixed_chunks: int = 2) -> dict:
    """Pass the prior *actual* unstable tokens to each arm's next chunk.

    During the first ``unfixed_chunks`` chunks the old hypothesis is discarded,
    so there is no rollback draft to verify. Later chunks keep the stable
    prefix and separately offer the removed tail for first verification.
    """
    if type(rollback_tokens) is not int or rollback_tokens < 0:
        raise ValueError("rollback_tokens must be nonnegative")
    if type(unfixed_chunks) is not int or unfixed_chunks < 0:
        raise ValueError("unfixed_chunks must be nonnegative")
    if not eos_token_ids:
        raise ValueError("eos_token_ids must not be empty")
    previous = {"baseline": [], "mtp": []}
    chunks = []
    for source in utterance["chunks"]:
        index = source["chunk_index"]
        chunk = {"chunk_index": index,
                 "audio_end_seconds": source["audio_end_seconds"]}
        for arm in ("baseline", "mtp"):
            old = previous[arm]
            stable = 0 if index < unfixed_chunks else max(0, len(old) - rollback_tokens)
            prefix = tuple(old[:stable])
            rollback = () if index < unfixed_chunks else tuple(old[stable:])
            record = {**source, "prefix_token_ids": list(prefix),
                      "target_token_ids": [min(eos_token_ids)],
                      "mtp_start_index": len(rollback)}
            result = decode(record, arm, prefix, rollback)
            tokens = result["token_ids"]
            if (not isinstance(tokens, list) or not tokens
                    or any(type(token) is not int or token < 0 for token in tokens)
                    or any(token in eos_token_ids for token in tokens[:-1])):
                raise ValueError(f"{arm} chunk {index} returned invalid token IDs")
            chunk[arm] = {**result, "prefix_token_ids": list(prefix),
                          "rollback_draft_token_ids": list(rollback)}
            previous[arm] = list(prefix) + _without_eos(tokens, eos_token_ids)
        chunk["token_exact_match"] = previous["baseline"] == previous["mtp"]
        chunks.append(chunk)
    baseline_text = decode_text(previous["baseline"])
    mtp_text = decode_text(previous["mtp"])
    return {key: utterance[key] for key in
            ("dataset", "utterance_id", "reference_text", "audio", "duration_seconds")} | {
                "baseline": {"token_ids": previous["baseline"]},
                "mtp": {"token_ids": previous["mtp"]},
                "baseline_text": baseline_text, "mtp_text": mtp_text,
                "token_exact_match": previous["baseline"] == previous["mtp"],
                "chunks": chunks,
            }
