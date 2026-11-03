"""CPU-testable rollback-first handoff to the spec-asr MTP proposer.

The backend owns target state. ``verify`` may advance it only through tokens
confirmed by the target, including the target correction after a mismatch.
No vLLM import or engine modification is required by this reference path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Verification:
    confirmed_tokens: tuple[int, ...]
    accepted_draft_tokens: int
    target_calls: int


class HandoffBackend(Protocol):
    def verify(self, draft: tuple[int, ...]) -> Verification: ...

    def propose_mtp(self, max_heads: int) -> Sequence[int]: ...


def confirm_draft(draft: Sequence[int], target: Sequence[int]) -> tuple[tuple[int, ...], int]:
    """Commit the matching prefix and one target correction, if needed."""
    if len(target) != len(draft) or not draft:
        raise ValueError("target comparison must cover the whole nonempty draft")
    for index, (proposed, actual) in enumerate(zip(draft, target)):
        if proposed != actual:
            return tuple(target[:index + 1]), index
    return tuple(draft), len(draft)


def _offered(tokens: Sequence[int], remaining: int, eos: set[int]) -> tuple[int, ...]:
    draft = tuple(tokens[:remaining])
    for index, token in enumerate(draft):
        if token in eos:
            return draft[:index + 1]
    return draft


def _checked_verification(draft: tuple[int, ...], result: Verification,
                          remaining: int, eos: set[int]) -> tuple[int, ...]:
    accepted = result.accepted_draft_tokens
    tokens = result.confirmed_tokens
    if (type(accepted) is not int or not 0 <= accepted <= len(draft)
            or type(result.target_calls) is not int or result.target_calls < 1
            or len(tokens) != accepted + (accepted < len(draft))
            or not tokens or len(tokens) > remaining
            or tokens[:accepted] != draft[:accepted]
            or (accepted < len(draft) and tokens[-1] == draft[accepted])
            or any(type(token) is not int or token < 0 for token in tokens)
            or any(token in eos for token in tokens[:-1])):
        raise ValueError("backend verification returned unconfirmed tokens")
    return tokens


def run_handoff(backend: HandoffBackend, rollback_draft: Sequence[int], *,
                max_new_tokens: int, kmax: int, draft_heads: int,
                eos_token_ids: set[int]) -> dict[str, object]:
    """Verify a complete rollback draft once, then draft from confirmed state.

    A rejected rollback token is replaced by the target correction before
    ``propose_mtp`` is called. Every later MTP window includes the target's
    next token followed by at most ``draft_heads`` auxiliary predictions.
    ``kmax`` counts all tokens submitted in one verification window.
    """
    if any(type(value) is not int or value < 1 for value in
           (max_new_tokens, kmax, draft_heads)):
        raise ValueError("token budgets and draft_heads must be positive integers")
    rollback = tuple(rollback_draft)
    if any(type(token) is not int or token < 0 for token in rollback):
        raise ValueError("rollback draft must contain nonnegative token IDs")
    if len(rollback) > kmax:
        raise ValueError("rollback draft exceeds engine kmax")
    if not eos_token_ids or any(type(token) is not int or token < 0
                                for token in eos_token_ids):
        raise ValueError("eos_token_ids must contain nonnegative token IDs")

    generated: list[int] = []
    windows: list[dict[str, int | str]] = []
    target_calls = 1  # prefill
    rollback_accepted = mtp_drafted = mtp_accepted = 0
    max_heads = min(draft_heads, kmax - 1)

    while len(generated) < max_new_tokens:
        phase = "rollback" if rollback else "mtp"
        if rollback:
            proposal = rollback
            rollback = ()
        else:
            proposal = tuple(backend.propose_mtp(max_heads))
            if len(proposal) > max_heads + 1:
                raise ValueError("MTP proposal exceeds engine kmax")
        proposal = _offered(proposal, max_new_tokens - len(generated), eos_token_ids)
        if not proposal or any(type(token) is not int or token < 0 for token in proposal):
            raise ValueError("proposal must contain nonnegative token IDs")
        start = len(generated)
        result = backend.verify(proposal)
        confirmed = _checked_verification(
            proposal, result, max_new_tokens - start, eos_token_ids)
        if phase == "mtp" and result.accepted_draft_tokens < 1:
            raise ValueError("MTP main token failed target verification")
        generated.extend(confirmed)
        target_calls += result.target_calls
        windows.append({"phase": phase, "start": start, "offered": len(proposal),
                        "accepted": result.accepted_draft_tokens,
                        "confirmed": len(confirmed)})
        if phase == "rollback":
            rollback_accepted = result.accepted_draft_tokens
        else:
            mtp_drafted += len(proposal) - 1
            mtp_accepted += result.accepted_draft_tokens - 1
        if generated[-1] in eos_token_ids:
            break

    return {"token_ids": generated, "rollback_offered_tokens":
            sum(window["offered"] for window in windows if window["phase"] == "rollback"),
            "rollback_accepted_tokens": rollback_accepted,
            "mtp_draft_tokens": mtp_drafted,
            "mtp_accepted_draft_tokens": mtp_accepted,
            "target_calls": target_calls,
            "windows": windows,
            "truncated": generated[-1] not in eos_token_ids}
