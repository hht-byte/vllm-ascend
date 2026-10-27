"""Build formal ASR extra_args without changing the request's other options."""

from __future__ import annotations

from typing import Literal, Mapping, Sequence

RejoinPolicy = Literal["aligned", "ngram_1", "ngram_2", "ngram_3"]
POLICIES = ("aligned", "ngram_1", "ngram_2", "ngram_3")


def formal_external_draft_extra_args(
    existing: Mapping[str, object] | None,
    *,
    candidate_tokens: Sequence[int] | None,
    rollback_tokens: Sequence[int],
    kmax: int,
    rejoin_policy: RejoinPolicy = "ngram_1",
) -> dict[str, object]:
    """Mark a selected prespec candidate for first-prefill verification.

    ``candidate_tokens=None`` keeps the old rollback contract. This function
    returns a new dict so a reused SamplingParams template cannot leak the
    first-verify flag into a later rollback or shadow request.
    """
    if rejoin_policy not in POLICIES:
        raise ValueError(f"invalid prespec rejoin policy: {rejoin_policy}")
    if kmax <= 0:
        raise ValueError("kmax must be positive")
    args = dict(existing or {})
    args.pop("ascend_external_draft_first_verify", None)
    args.pop("ascend_external_draft_rejoin", None)
    draft = rollback_tokens if candidate_tokens is None else candidate_tokens
    if any(type(token) is not int or token < 0 for token in draft):
        raise ValueError("external draft must contain nonnegative token IDs")
    if len(draft) > kmax:
        raise ValueError("external draft exceeds engine kmax")
    args["ascend_external_draft"] = list(draft)
    if candidate_tokens is not None:
        if not draft:
            raise ValueError("selected candidate must not be empty")
        args["ascend_external_draft_first_verify"] = True
        args["ascend_external_draft_rejoin"] = rejoin_policy
    return args
