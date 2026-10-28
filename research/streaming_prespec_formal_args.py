"""Build formal ASR extra_args without changing the request's other options."""

from __future__ import annotations

from typing import Literal, Mapping, Sequence

RejoinPolicy = Literal["aligned", "ngram_1", "ngram_2", "ngram_3"]
POLICIES = ("aligned", "ngram_1", "ngram_2", "ngram_3")


def formal_rollback_draft_tokens(
    cur_ids: Sequence[int],
    stable_prefix_token_ids: Sequence[int],
    *,
    kmax: int,
) -> tuple[int, ...]:
    """Return C[0:] after the exact stable prefix used by the formal prompt.

    Call with the ``cur_ids`` already produced by the rollback splitter. Its
    legacy external draft starts one token later and belongs to the shadow
    path until the formal first-verification path is wired on the server.
    """
    if kmax <= 0:
        raise ValueError("kmax must be positive")
    tokens = tuple(cur_ids)
    stable = tuple(stable_prefix_token_ids)
    if tokens[:len(stable)] != stable:
        raise ValueError("stable prefix does not match rollback token IDs")
    return tokens[len(stable):len(stable) + kmax]


def formal_external_draft_extra_args(
    existing: Mapping[str, object] | None,
    *,
    candidate_tokens: Sequence[int] | None,
    rollback_tokens: Sequence[int],
    rollback_full_tokens: Sequence[int],
    kmax: int,
    rejoin_policy: RejoinPolicy = "ngram_1",
) -> dict[str, object]:
    """Mark a full candidate or full rollback for first-prefill verification.

    ``rollback_full_tokens`` must begin at the formal request's first output
    position. Formal chunk and finish must both pass it, including when a
    candidate is selected. Shadow requests do not call this helper.
    """
    if rejoin_policy not in POLICIES:
        raise ValueError(f"invalid prespec rejoin policy: {rejoin_policy}")
    if kmax <= 0:
        raise ValueError("kmax must be positive")
    args = dict(existing or {})
    args.pop("ascend_external_draft_first_verify", None)
    args.pop("ascend_external_draft_rejoin", None)
    if candidate_tokens is not None:
        draft = candidate_tokens
        first_verify = True
    else:
        draft = rollback_full_tokens
        first_verify = bool(draft)
        if not draft and rollback_tokens:
            raise ValueError("full rollback draft cannot be empty with a legacy suffix")
        if tuple(draft[1:]) != tuple(rollback_tokens[:max(0, len(draft) - 1)]):
            raise ValueError("full rollback draft does not extend the legacy suffix")
    if any(type(token) is not int or token < 0 for token in draft):
        raise ValueError("external draft must contain nonnegative token IDs")
    if len(draft) > kmax:
        raise ValueError("external draft exceeds engine kmax")
    args["ascend_external_draft"] = list(draft)
    if candidate_tokens is not None and not draft:
        raise ValueError("selected candidate must not be empty")
    if first_verify:
        args["ascend_external_draft_first_verify"] = True
        args["ascend_external_draft_rejoin"] = rejoin_policy
    return args
