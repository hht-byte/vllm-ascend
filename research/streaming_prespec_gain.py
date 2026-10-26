"""Compare request-level ASR formal and verifier counters from server logs.

Each relevant line contains ``ASR_FORMAL_STATS {json}`` or
``ASR_VERIFY_STEP {json}``, optionally preceded by a timestamp/log prefix.
This script uses only Python's standard library and does not infer accepted
draft tokens from text/prefix similarity or from fewer decode rounds.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from bisect import bisect_left
from datetime import datetime
from pathlib import Path


MARKERS = {"ASR_FORMAL_STATS ": "formal", "ASR_VERIFY_STEP ": "verify"}
METRICS = ("output_token_count", "decode_rounds", "tokens_per_round",
           "prefill_ms", "decode_total_ms", "submitted_draft_tokens",
           "effective_draft_tokens", "accepted_draft_tokens",
           "acceptance_rate")
SOURCES = ("none", "rollback", "prespec")
BOUNDARIES = ("chunk", "finish")
VLLM_INTERNAL_ID = re.compile(r"^(.+)-[0-9a-fA-F]{8}$")


def _integer(row: dict, key: str, where: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where}: {key} must be a nonnegative integer")
    return value


def _number(row: dict, key: str, where: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            not math.isfinite(value) or value < 0:
        raise ValueError(f"{where}: {key} must be a finite nonnegative number")
    return float(value)


def _string(row: dict, key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: {key} must be a nonempty string")
    return value


def _choice(row: dict, key: str, choices: tuple[str, ...], where: str) -> str:
    value = _string(row, key, where)
    if value not in choices:
        raise ValueError(f"{where}: {key} must be one of {choices}")
    return value


def _validated_formal(row: dict, where: str) -> dict:
    if row.get("finished") is not True:
        raise ValueError(f"{where}: finished must be true")
    result = {
        "request_id": _string(row, "request_id", where),
        "ws_id": _string(row, "ws_id", where),
        "seg": _integer(row, "seg", where),
        "boundary": _choice(row, "boundary", BOUNDARIES, where),
        "draft_source": _choice(row, "draft_source", SOURCES, where),
        "submitted_draft_tokens": _integer(row, "submitted_draft_tokens", where),
        "output_token_count": _integer(row, "output_token_count", where),
        "decode_rounds": _integer(row, "decode_rounds", where),
        "prefill_ms": _number(row, "prefill_ms", where),
        "decode_total_ms": _number(row, "decode_total_ms", where),
    }
    if (result["draft_source"] == "none") != (result["submitted_draft_tokens"] == 0):
        raise ValueError(f"{where}: draft_source disagrees with submitted_draft_tokens")
    case_id = row.get("case_id")
    if case_id is not None:
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{where}: case_id must be a nonempty string")
        result["case_id"] = case_id
    return result


def _validated_verify(row: dict, where: str) -> dict:
    source = row.get("draft_source")
    if source is not None:
        source = _choice(row, "draft_source", SOURCES[1:], where)
    complete = row.get("verifier_complete", False)
    if not isinstance(complete, bool):
        raise ValueError(f"{where}: verifier_complete must be a boolean")
    result = {
        "request_id": _string(row, "request_id", where),
        "round_index": _integer(row, "round_index", where),
        "draft_source": source,
        "effective_draft_tokens": _integer(row, "effective_draft_tokens", where),
        "accepted_draft_tokens": _integer(row, "accepted_draft_tokens", where),
        "verifier_complete": complete,
        "actual_effective_draft_tokens": None,
    }
    if result["accepted_draft_tokens"] > result["effective_draft_tokens"]:
        raise ValueError(f"{where}: accepted_draft_tokens exceeds effective_draft_tokens")
    if complete:
        result["actual_effective_draft_tokens"] = _integer(
            row, "actual_effective_draft_tokens", where)
    elif row.get("actual_effective_draft_tokens") is not None:
        raise ValueError(f"{where}: actual_effective_draft_tokens needs verifier_complete=true")
    reason = row.get("reason")
    if complete and result["actual_effective_draft_tokens"] == 0:
        reason = _string(row, "reason", where)
    elif reason is not None:
        reason = _string(row, "reason", where)
    result["reason"] = reason
    return result


def _log_time(line: str) -> datetime | None:
    """Read the millisecond timestamp used by the ASR server log, if present."""
    try:
        return datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None


def _target_submit(line: str) -> dict | None:
    """Read a formal request's submit marker for collision-safe log-order joins."""
    marker = "PRESPEC_TARGET "
    if marker not in line:
        return None
    fields = dict(re.findall(r"(?:^|\s)([a-z_]+)=([^\s]+)",
                             line.split(marker, 1)[1]))
    boundary = ("finish" if fields.get("boundary") == "formal" else
                "chunk" if fields.get("event") == "submit" and "chunk" in fields
                else None)
    if boundary is None or not all(fields.get(key) for key in
                                   ("formal_request_id", "ws_id", "seg")):
        return None
    return {"request_id": fields["formal_request_id"],
            "ws_id": fields["ws_id"], "seg": fields["seg"],
            "boundary": boundary}


def parse_run(paths: list[str], *, allow_legacy_time_join: bool = False) -> list[dict]:
    """Join events, retaining distinct requests after a legacy ID collision."""
    formals: list[tuple[dict, datetime | None, Path]] = []
    formal_lines: list[int] = []
    formal_by_id: dict[str, list[int]] = {}
    verifiers: dict[str, list[tuple[dict, datetime | None, Path, int]]] = {}
    targets: list[tuple[dict, Path, int]] = []
    for path_text in paths:
        path = Path(path_text)
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                target = _target_submit(line)
                if target is not None:
                    targets.append((target, path, line_number))
                for marker, kind in MARKERS.items():
                    if marker not in line:
                        continue
                    where = f"{path}:{line_number}"
                    payload = line.split(marker, 1)[1].strip()
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{where}: invalid JSON after {marker.strip()}: {exc}") from exc
                    if not isinstance(event, dict):
                        raise ValueError(f"{where}: {marker.strip()} must contain a JSON object")
                    if kind == "formal":
                        row = _validated_formal(event, where)
                        request_id = row["request_id"]
                        siblings = formal_by_id.setdefault(request_id, [])
                        if any((formals[index][0]["ws_id"], formals[index][0]["seg"],
                                formals[index][0]["boundary"]) ==
                               (row["ws_id"], row["seg"], row["boundary"])
                               for index in siblings):
                            raise ValueError(f"{where}: duplicate formal request_id {request_id}")
                        siblings.append(len(formals))
                        formals.append((row, _log_time(line), path))
                        formal_lines.append(line_number)
                    else:
                        row = _validated_verify(event, where)
                        verifiers.setdefault(row["request_id"], []).append(
                            (row, _log_time(line), path, line_number))
                    break
    steps_by_formal: list[list[dict]] = [[] for _ in formals]
    verifier_ids_by_formal: list[set[str]] = [set() for _ in formals]
    events_by_external_id: dict[str, list[tuple[dict, datetime | None, Path, int, str]]] = {}
    for internal_id, events in verifiers.items():
        if internal_id in formal_by_id:
            external_id = internal_id
        else:
            match = VLLM_INTERNAL_ID.fullmatch(internal_id)
            external_id = match.group(1) if match else None
            if external_id not in formal_by_id:
                raise ValueError(f"orphan verifier request_id {internal_id}")
        events_by_external_id.setdefault(external_id, []).extend(
            (row, timestamp, path, line_number, internal_id)
            for row, timestamp, path, line_number in events)
    for request_id, events in events_by_external_id.items():
        indices = formal_by_id[request_id]
        if len(indices) == 1:
            index = indices[0]
            steps_by_formal[index].extend(row for row, _, _, _, _ in events)
            verifier_ids_by_formal[index].update(raw_id for _, _, _, _, raw_id in events)
            continue
        for row, timestamp, path, line_number, raw_id in events:
            bracketed = []
            for index in indices:
                formal, _, formal_path = formals[index]
                if path != formal_path or line_number >= formal_lines[index]:
                    continue
                starts = [target_line for target, target_path, target_line in targets
                          if target_path == path and target_line < formal_lines[index]
                          and target == {"request_id": request_id,
                                         "ws_id": formal["ws_id"],
                                         "seg": str(formal["seg"]),
                                         "boundary": formal["boundary"]}]
                if len(starts) == 1 and starts[0] < line_number:
                    bracketed.append(index)
            if len(bracketed) == 1:
                index = bracketed[0]
            elif any(target["request_id"] == request_id for target, _, _ in targets):
                raise ValueError(f"ambiguous reused request_id {request_id}: "
                                 "verifier has no unique submit/completion bracket")
            else:
                if not allow_legacy_time_join:
                    raise ValueError(f"ambiguous reused request_id {request_id} has "
                                     "verifier events; use unique IDs in new runs, or "
                                     "--allow-legacy-time-join for a complete single-file log")
                # Older logs lack submit markers. Only the explicit legacy mode
                # may infer ownership from complete, timestamped single-file logs.
                sources = {formals[candidate][2] for candidate in indices}
                if len(sources) != 1 or path not in sources or timestamp is None or \
                        any(formals[candidate][1] is None for candidate in indices):
                    raise ValueError(f"ambiguous reused request_id {request_id}: "
                                     "verifier events need timestamps in one log file")
                ordered = sorted(indices, key=lambda candidate: formals[candidate][1])
                completion_times = [formals[candidate][1] for candidate in ordered]
                if len(set(completion_times)) != len(completion_times) or \
                        timestamp in completion_times:
                    raise ValueError(f"ambiguous reused request_id {request_id}: "
                                     "verifier/formal completion timestamp conflict")
                position = bisect_left(completion_times, timestamp)
                if position == len(ordered):
                    raise ValueError(f"ambiguous reused request_id {request_id}: "
                                     "verifier occurs after the last formal completion")
                index = ordered[position]
            steps_by_formal[index].append(row)
            verifier_ids_by_formal[index].add(raw_id)
    for index, raw_ids in enumerate(verifier_ids_by_formal):
        if len(raw_ids) > 1:
            raise ValueError(f"multiple verifier request_ids for formal "
                             f"{formals[index][0]['request_id']}: {sorted(raw_ids)}")
    for index, (formal, _, _) in enumerate(formals):
        request_id = formal["request_id"]
        steps = steps_by_formal[index]
        seen_rounds: set[int] = set()
        for step in steps:
            round_index = step["round_index"]
            if round_index in seen_rounds:
                raise ValueError(f"duplicate verifier round ({request_id!r}, {round_index})")
            seen_rounds.add(round_index)
        if any(step["draft_source"] is not None and
               step["draft_source"] != formal["draft_source"] for step in steps):
            raise ValueError(f"verifier draft_source differs from formal for {request_id}")
        if steps and formal["draft_source"] == "none":
            raise ValueError(f"verifier emitted for request without draft {request_id}")
        if sum(step["verifier_complete"] for step in steps) > 1:
            raise ValueError(f"duplicate verifier_complete for {request_id}")
        final_step = next((step for step in steps if step["verifier_complete"]), None)
        if final_step is not None and final_step["round_index"] != max(
                step["round_index"] for step in steps):
            raise ValueError(f"final verifier round_index is not the last for {request_id}")
        if any(step["actual_effective_draft_tokens"] is not None and
               step["actual_effective_draft_tokens"] > formal["submitted_draft_tokens"]
               for step in steps):
            raise ValueError(f"actual_effective_draft_tokens exceeds submitted_draft_tokens for {request_id}")
        if sum(step["effective_draft_tokens"] for step in steps) > formal["submitted_draft_tokens"]:
            raise ValueError(f"effective_draft_tokens exceeds submitted_draft_tokens for {request_id}")
    if not formals:
        raise ValueError("no ASR_FORMAL_STATS events found in the supplied logs")
    records = []
    for index, (formal, _, _) in enumerate(formals):
        request_id = formal["request_id"]
        steps = steps_by_formal[index]
        missing = formal["submitted_draft_tokens"] > 0 and not steps
        logged_effective = sum(step["effective_draft_tokens"] for step in steps)
        logged_accepted = sum(step["accepted_draft_tokens"] for step in steps)
        completed = next((step for step in steps if step["verifier_complete"]), None)
        actual_effective = completed["actual_effective_draft_tokens"] if completed else None
        incomplete = bool(steps) and (completed is None or
                                      logged_effective != actual_effective)
        explicit_zero = bool(steps) and not incomplete and actual_effective == 0
        if missing:
            status = "missing"
        elif incomplete:
            status = "incomplete"
        elif formal["draft_source"] == "none":
            status = "no_draft"
        elif explicit_zero:
            status = "zero_effective"
        else:
            status = "observed"
        trusted = not missing and not incomplete
        effective = logged_effective if trusted else None
        accepted = logged_accepted if trusted else None
        records.append({
            **formal,
            "tokens_per_round": formal["output_token_count"] / formal["decode_rounds"]
            if formal["decode_rounds"] else None,
            "verifier_event_count": len(steps),
            "verifier_request_ids": sorted(verifier_ids_by_formal[index]),
            "verifier_status": status,
            "missing_verifier": missing,
            "incomplete_verifier": incomplete,
            "explicit_zero_effective": explicit_zero,
            "actual_effective_draft_tokens": actual_effective,
            "logged_effective_draft_tokens": logged_effective,
            "logged_accepted_draft_tokens": logged_accepted,
            "verifier_skip_reasons": sorted({step["reason"] for step in steps
                                             if step["reason"] is not None}),
            "effective_draft_tokens": effective,
            "accepted_draft_tokens": accepted,
            "acceptance_rate": accepted / effective if effective else None,
        })
    return records


def _percentile(values: list[float], percent: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _stats(values: list[int | float]) -> dict:
    return {
        "count": len(values),
        "sum": sum(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values) if values else None,
    }


def summarize(records: list[dict]) -> dict:
    result = {
        "formal_request_count": len(records),
        "case_id_count": sum("case_id" in row for row in records),
        "verifier_event_count": sum(row["verifier_event_count"] for row in records),
        "missing_verifier_count": sum(row["missing_verifier"] for row in records),
        "missing_verifier_request_ids": sorted(row["request_id"] for row in records
                                               if row["missing_verifier"]),
        "incomplete_verifier_count": sum(row["incomplete_verifier"] for row in records),
        "incomplete_verifier_request_ids": sorted(row["request_id"] for row in records
                                                  if row["incomplete_verifier"]),
        "explicit_zero_effective_count": sum(row["explicit_zero_effective"] for row in records),
        "explicit_zero_effective_request_ids": sorted(row["request_id"] for row in records
                                                      if row["explicit_zero_effective"]),
    }
    for metric in METRICS:
        result[metric] = _stats([row[metric] for row in records if row[metric] is not None])
    total_effective = result["effective_draft_tokens"]["sum"]
    total_accepted = result["accepted_draft_tokens"]["sum"]
    result["observed_acceptance_rate"] = (total_accepted / total_effective
                                          if total_effective else None)
    return result


def summarize_run(records: list[dict]) -> dict:
    by_boundary = {boundary: summarize([row for row in records
                                         if row["boundary"] == boundary])
                   for boundary in BOUNDARIES}
    by_source = {source: summarize([row for row in records
                                    if row["draft_source"] == source])
                 for source in SOURCES}
    by_both = {boundary: {source: summarize([row for row in records
                                              if row["boundary"] == boundary
                                              and row["draft_source"] == source])
                          for source in SOURCES}
               for boundary in BOUNDARIES}
    return {"records": sorted(records, key=lambda row: row["request_id"]),
            "all": summarize(records), "by_boundary": by_boundary,
            "by_draft_source": by_source,
            "by_boundary_and_draft_source": by_both}


def _difference(before: dict, after: dict) -> dict:
    result = {}
    for metric in METRICS:
        a, b = before[metric], after[metric]
        result[metric] = {
            "mean_delta": b["mean"] - a["mean"] if a["mean"] is not None and b["mean"] is not None else None,
            "p95_delta": b["p95"] - a["p95"] if a["p95"] is not None and b["p95"] is not None else None,
        }
    a, b = before["observed_acceptance_rate"], after["observed_acceptance_rate"]
    result["observed_acceptance_rate_delta"] = b - a if a is not None and b is not None else None
    result["formal_request_count_delta"] = after["formal_request_count"] - before["formal_request_count"]
    result["missing_verifier_count_delta"] = after["missing_verifier_count"] - before["missing_verifier_count"]
    result["incomplete_verifier_count_delta"] = (after["incomplete_verifier_count"] -
                                                 before["incomplete_verifier_count"])
    result["explicit_zero_effective_count_delta"] = (after["explicit_zero_effective_count"] -
                                                      before["explicit_zero_effective_count"])
    return result


def compare(before: dict, after: dict) -> dict:
    return {
        "all": _difference(before["all"], after["all"]),
        "by_boundary": {boundary: _difference(before["by_boundary"][boundary],
                                               after["by_boundary"][boundary])
                        for boundary in BOUNDARIES},
        "by_draft_source": {source: _difference(before["by_draft_source"][source],
                                                 after["by_draft_source"][source])
                            for source in SOURCES},
        "by_boundary_and_draft_source": {
            boundary: {source: _difference(
                before["by_boundary_and_draft_source"][boundary][source],
                after["by_boundary_and_draft_source"][boundary][source])
                for source in SOURCES}
            for boundary in BOUNDARIES},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollback-log", nargs="+", required=True, metavar="PATH")
    parser.add_argument("--consume-log", nargs="+", required=True, metavar="PATH")
    parser.add_argument("--shadow-log", nargs="+", metavar="PATH")
    parser.add_argument("--out", required=True, metavar="PATH")
    parser.add_argument(
        "--allow-legacy-time-join", action="store_true",
        help="fallback for reused IDs without submit markers: infer ownership from full millisecond timestamps in one complete log file",
    )
    args = parser.parse_args()
    try:
        runs = {"rollback": summarize_run(parse_run(
                    args.rollback_log,
                    allow_legacy_time_join=args.allow_legacy_time_join)),
                "consume": summarize_run(parse_run(
                    args.consume_log,
                    allow_legacy_time_join=args.allow_legacy_time_join))}
        if args.shadow_log:
            runs["shadow"] = summarize_run(parse_run(
                args.shadow_log,
                allow_legacy_time_join=args.allow_legacy_time_join))
        comparisons = {"rollback_to_consume": compare(runs["rollback"], runs["consume"])}
        if "shadow" in runs:
            comparisons["rollback_to_shadow"] = compare(runs["rollback"], runs["shadow"])
            comparisons["shadow_to_consume"] = compare(runs["shadow"], runs["consume"])
        output = {
            "methodology": "Request-level descriptive statistics. Runs are not paired by audio/case; differences are not causal. Output tokens are raw formal DELTA token IDs (including special tokens), decode_rounds are decode iterations, and accepted_draft_tokens are verifier-confirmed external draft tokens only. Missing and incomplete verifier logs are excluded from acceptance statistics; a completed 0/0 skip is distinct from both and is not an actual acceptance-rate observation. vLLM internal -8hex request IDs are matched to service external IDs only when the external ID exists. Reused external IDs require a unique same-file submit-to-completion log bracket; without submit markers, timestamp-based joins are opt-in heuristics for complete single-file logs. Ambiguous collisions fail.",
            "legacy_time_join_enabled": args.allow_legacy_time_join,
            "runs": runs,
            "comparison": comparisons,
        }
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
