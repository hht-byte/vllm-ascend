"""Choose external drafts using only confirmed target output tokens.

The first draft is submitted by the scheduler before the target's first
forward. After a verifier correction, the proposer calls ``next_draft`` with
the confirmed output prefix. This module is deliberately independent of vLLM
and NPU imports so the request-local policy can be tested on a CPU.
"""

from __future__ import annotations

from typing import Literal, Sequence


RejoinPolicy = Literal["aligned", "ngram_1", "ngram_2", "ngram_3"]
POLICIES = ("aligned", "ngram_1", "ngram_2", "ngram_3")


def first_mismatch(candidate: Sequence[int], confirmed: Sequence[int]) -> int | None:
    """Return the first proven mismatch; do not inspect future target output."""
    for index, (draft, target) in enumerate(zip(candidate, confirmed)):
        if draft != target:
            return index
    return None


def next_draft(
    candidate: Sequence[int],
    confirmed: Sequence[int],
    *,
    policy: RejoinPolicy = "ngram_1",
    max_draft_tokens: int,
    after_aligned_retry: bool = False,
) -> tuple[int, ...]:
    """Select a continuation after the sampled/verifier-confirmed prefix.

    First-token mismatch initially uses the candidate at the same absolute
    output position. After that aligned retry fails, callers may enable
    n-gram rejoining even though the first mismatch remains at position zero.
    Before any mismatch, the aligned remaining suffix is used.
    The caller must pass only committed target output IDs, never speculative
    IDs still waiting for verification.
    """
    if policy not in POLICIES:
        raise ValueError(f"invalid external draft rejoin policy: {policy}")
    if max_draft_tokens < 1:
        raise ValueError("max_draft_tokens must be positive")
    position = len(confirmed)
    mismatch = first_mismatch(candidate, confirmed)
    if policy == "aligned" or mismatch is None or (mismatch == 0 and
                                                not after_aligned_retry):
        return tuple(candidate[position:position + max_draft_tokens])

    n = int(policy[-1])
    if position < n:
        return ()
    anchor = tuple(confirmed[-n:])
    for next_index in range(len(candidate) - 1, n - 1, -1):
        if tuple(candidate[next_index - n:next_index]) == anchor:
            return tuple(candidate[next_index:next_index + max_draft_tokens])
    return ()
