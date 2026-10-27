"""Estimate post-mismatch draft reuse from PRESPEC_MATCH_TOKENS logs.

Usage::

    python research/streaming_prespec_rejoin.py --log server.log \
        --out rejoin-summary.json --cases-out rejoin-cases.jsonl

This is an offline, token-ID-only replay. It can count opportunities and
idealized model steps, not actual verifier acceptance or elapsed time. A
candidate lookup may inspect only already confirmed target tokens; future
target tokens are used solely to score the proposed draft. Truncated target
logs can still support aligned replay when every candidate position is
visible; n-gram replay needs the complete target. ``estimated_saved_steps`` is
relative to one-token autoregressive continuation *after* the existing
first draft's mismatch, not an observed latency reduction. An accepted
terminal EOS by itself does not save a model step.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


MARKER = "PRESPEC_MATCH_TOKENS "
MATCH_LINE = re.compile(
    r"ws_id=(?P<ws_id>\S+) seg=(?P<seg>\d+) window=(?P<window>\d+) "
    r"mode=(?P<mode>shadow|consume) "
    r"candidate_token_ids=(?P<candidate>\([^)]*\)) "
    r"target_token_ids=(?P<target>\([^)]*\)) "
    r"target_len=(?P<target_len>\d+) "
    r"raw_matched_prefix_tokens=(?P<raw_match>\d+)"
    r"(?:\s+candidate_partial=(?P<partial>\S+))?"
)


def _token_tuple(value: str, name: str) -> tuple[int, ...]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"invalid {name}: {value!r}") from error
    if not isinstance(parsed, tuple) or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in parsed
    ):
        raise ValueError(f"{name} must be a tuple of nonnegative token IDs")
    return parsed


def _matched_prefix(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def parse_match_line(line: str) -> dict | None:
    """Parse one coordinator log line; return None for unrelated lines."""
    if MARKER not in line:
        return None
    match = MATCH_LINE.search(line.split(MARKER, 1)[1])
    if match is None:
        raise ValueError("malformed PRESPEC_MATCH_TOKENS line")
    row = match.groupdict()
    candidate = _token_tuple(row["candidate"], "candidate_token_ids")
    target = _token_tuple(row["target"], "target_token_ids")
    target_len = int(row["target_len"])
    if target_len < len(target) or len(target) > len(candidate):
        raise ValueError("target_token_ids length disagrees with log lengths")
    raw_match = int(row["raw_match"])
    if raw_match != _matched_prefix(candidate, target):
        raise ValueError("raw_matched_prefix_tokens disagrees with logged IDs")
    if row["partial"] not in (None, "True", "False"):
        raise ValueError("candidate_partial must be True or False")
    return {
        "ws_id": row["ws_id"],
        "seg": int(row["seg"]),
        "window": int(row["window"]),
        "mode": row["mode"],
        "candidate_token_ids": candidate,
        "target_token_ids": target,
        "target_len": target_len,
        "candidate_partial": (
            None if row["partial"] is None else row["partial"] == "True"
        ),
    }


def _ngram_draft(candidate: tuple[int, ...], confirmed: tuple[int, ...],
                 n: int, kmax: int) -> tuple[int, ...]:
    """Pick the rightmost matching candidate anchor without future target IDs."""
    if len(confirmed) < n:
        return ()
    anchor = confirmed[-n:]
    for next_index in range(len(candidate) - 1, n - 1, -1):
        if candidate[next_index - n:next_index] == anchor:
            return candidate[next_index:next_index + kmax]
    return ()


def _replay(candidate: tuple[int, ...], target: tuple[int, ...],
            start: int, strategy: str, kmax: int,
            eos_token_id: int, *, target_len: int | None = None) -> dict:
    horizon = len(target) if target_len is None else target_len
    position = start
    attempts = zero_accept_attempts = accepted_tokens = 0
    accepted_nonterminal_tokens = autoregressive_steps = model_steps = 0
    while position < horizon:
        if strategy == "aligned":
            draft = candidate[position:position + kmax]
        else:
            draft = _ngram_draft(candidate, target[:position],
                                 int(strategy.rsplit("_", 1)[1]), kmax)
        if draft:
            attempts += 1
            accepted = _matched_prefix(draft, target[position:])
            zero_accept_attempts += accepted == 0
            accepted_tokens += accepted
            accepted_nonterminal_tokens += sum(
                target[index] != eos_token_id
                for index in range(position, position + accepted)
            )
            position += accepted
            # The verifier emits either a correction or a bonus when one
            # more target token exists. At target end it emits no extra token.
            if position < horizon:
                position += 1
        else:
            autoregressive_steps += 1
            position += 1
        model_steps += 1
    return {
        "attempts": attempts,
        "zero_accept_attempts": zero_accept_attempts,
        "accepted_tokens": accepted_tokens,
        "accepted_nonterminal_tokens": accepted_nonterminal_tokens,
        "autoregressive_steps": autoregressive_steps,
        "estimated_model_steps": model_steps,
        "estimated_saved_steps": (horizon - start) - model_steps,
    }


def analyze_record(record: dict, *, kmax: int,
                   eos_token_id: int) -> dict:
    """Replay only after the first mismatch in the existing C[1:] draft."""
    if kmax < 1:
        raise ValueError("kmax must be positive")
    candidate = record["candidate_token_ids"]
    target = record["target_token_ids"]
    censored = record["target_len"] > len(target)
    initial_draft_matched = _matched_prefix(candidate[1:], target[1:])
    first_mismatch_index = 1 + initial_draft_matched
    if first_mismatch_index >= min(len(candidate), len(target)):
        first_mismatch_index = None
    later_same = later_nonterminal = 0
    if first_mismatch_index is not None:
        for position in range(first_mismatch_index + 1,
                              min(len(candidate), len(target))):
            if candidate[position] == target[position]:
                later_same += 1
                later_nonterminal += target[position] != eos_token_id
    if first_mismatch_index is None:
        skip_reason = "no_post_first_draft_mismatch"
    elif len(candidate) - 1 > kmax:
        skip_reason = "candidate_draft_exceeds_kmax"
    elif censored:
        skip_reason = "censored_target"
    else:
        skip_reason = None
    result = {
        "ws_id": record["ws_id"],
        "seg": record["seg"],
        "window": record["window"],
        "mode": record["mode"],
        "candidate_partial": record["candidate_partial"],
        "candidate_len": len(candidate),
        "target_len": record["target_len"],
        "censored": censored,
        "initial_draft_matched": initial_draft_matched,
        "first_mismatch_index": first_mismatch_index,
        "later_same_position_tokens": later_same,
        "later_same_position_nonterminal_tokens": later_nonterminal,
        "skip_reason": skip_reason,
        "policies": None,
    }
    # Current one-shot external drafts over kmax are rejected, rather than
    # silently shortened. A censored target still permits aligned replay if
    # all candidate positions are visible; later n-gram anchors are unknown.
    if skip_reason not in (None, "censored_target"):
        return result
    if censored and len(target) != len(candidate):
        return result
    start = first_mismatch_index + 1  # verifier's correction is confirmed
    result["policies"] = {
        "aligned": _replay(candidate, target, start, "aligned", kmax,
                           eos_token_id, target_len=record["target_len"]),
        **{
            name: (None if censored else
                   _replay(candidate, target, start, name, kmax,
                           eos_token_id))
            for name in ("ngram_1", "ngram_2", "ngram_3")
        },
    }
    return result


def summarize(cases: list[dict]) -> dict:
    fully_replayed = [row for row in cases if row["skip_reason"] is None]
    policies = {}
    for name in ("aligned", "ngram_1", "ngram_2", "ngram_3"):
        evaluated = [row["policies"][name] for row in cases
                     if row["policies"] is not None
                     and row["policies"][name] is not None]
        policies[name] = {
            key: sum(row[key] for row in evaluated)
            for key in ("attempts", "zero_accept_attempts", "accepted_tokens",
                        "accepted_nonterminal_tokens", "autoregressive_steps",
                        "estimated_model_steps", "estimated_saved_steps")
        }
        policies[name]["evaluated_request_count"] = len(evaluated)
        policies[name]["requests_with_saved_steps"] = sum(
            row["estimated_saved_steps"] > 0 for row in evaluated)
    return {
        "match_events": len(cases),
        "complete_target_count": sum(not row["censored"] for row in cases),
        "censored_target_count": sum(row["censored"] for row in cases),
        "post_first_draft_mismatch_count": sum(
            row["first_mismatch_index"] is not None for row in cases),
        "replayed_count": len(fully_replayed),
        "excluded_replay_reasons": {
            reason: sum(row["skip_reason"] == reason for row in cases)
            for reason in ("censored_target", "no_post_first_draft_mismatch",
                           "candidate_draft_exceeds_kmax")
        },
        "later_same_position_tokens": sum(
            row["later_same_position_tokens"] for row in cases),
        "later_same_position_nonterminal_tokens": sum(
            row["later_same_position_nonterminal_tokens"] for row in cases),
        "policies": policies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, nargs="+", required=True,
                        metavar="PATH")
    parser.add_argument("--out", type=Path, required=True, metavar="PATH")
    parser.add_argument("--cases-out", type=Path, metavar="PATH",
                        help="optional per-request JSONL for inspection")
    parser.add_argument("--kmax", type=int, default=15)
    parser.add_argument("--eos-token-id", type=int, default=151645)
    args = parser.parse_args()
    if args.kmax < 1 or args.eos_token_id < 0:
        parser.error("kmax must be positive; eos-token-id must be nonnegative")
    cases = []
    for path in args.log:
        with path.open(encoding="utf-8", errors="replace") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    record = parse_match_line(line)
                    if record is not None:
                        case = analyze_record(record, kmax=args.kmax,
                                              eos_token_id=args.eos_token_id)
                        case["source_file"] = str(path)
                        case["source_line"] = line_number
                        cases.append(case)
                except ValueError as error:
                    parser.error(f"{path}:{line_number}: {error}")
    if not cases:
        parser.error("no PRESPEC_MATCH_TOKENS events found")
    report = {
        "methodology": (
            "Offline token-ID replay after the first mismatch in the existing "
            "candidate[1:] draft. Censored targets with every candidate "
            "position visible support aligned replay only; n-gram replay "
            "requires complete target logs. Policy evaluated_request_count "
            "can therefore differ from replayed_count. "
            "One verifier attempt is idealized as one model step; "
            "estimated_saved_steps exclude verifier, lookup and contention "
            "cost and are not measured latency or actual acceptance. "
            "Rightmost exact candidate n-gram anchor is selected using "
            "confirmed target tokens only. Existing shadow/consume modes "
            "are counterfactual/actual candidate sources, respectively. "
            "candidate_partial=unknown means a legacy log without the "
            "optional cancellation-salvage flag. "
            "Only requests with PRESPEC_MATCH_TOKENS logs enter this report; "
            "it has no denominator for all formal requests or all commits."
        ),
        "kmax": args.kmax,
        "eos_token_id": args.eos_token_id,
        **summarize(cases),
        "by_mode": {
            mode: summarize([case for case in cases if case["mode"] == mode])
            for mode in ("shadow", "consume")
        },
        "by_candidate_partial": {
            label: summarize([case for case in cases
                              if case["candidate_partial"] is value])
            for label, value in (("true", True), ("false", False),
                                 ("unknown", None))
        },
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2)
                        + "\n", encoding="utf-8")
    if args.cases_out is not None:
        with args.cases_out.open("w", encoding="utf-8") as output:
            for case in cases:
                output.write(json.dumps(case, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
