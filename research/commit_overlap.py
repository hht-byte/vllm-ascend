"""Check commit waits against prior commits and same-segment chunk inference.

DEBUG token_count lines approximate the end of chunk inference. Missing lines
mean unknown overlap, not zero overlap; time intersection is not causality.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


LOG_EVENT = re.compile(
    r"^(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)"
    r" - [A-Z]+ - \[(?P<ws>ws-[^]]+)\]\[seg-(?P<seg>\d+)\]"
    r"(?:\[chunk-(?P<chunk>\d+)\])? (?P<message>.*)$")


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp needs timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    ordered = sorted(values)
    index = (len(ordered) - 1) * 0.95
    low, high = math.floor(index), math.ceil(index)
    return {"count": len(values), "mean": sum(values) / len(values),
            "p95": ordered[low] + (ordered[high] - ordered[low]) * (index - low),
            "max": ordered[-1]}


def _log_timezone(offset: str) -> timezone:
    match = re.fullmatch(r"([+-])(\d\d):(\d\d)", offset)
    if not match or int(match[2]) > 23 or int(match[3]) > 59:
        raise ValueError("log UTC offset must look like +00:00 or +08:00")
    minutes = int(match[2]) * 60 + int(match[3])
    return timezone(timedelta(minutes=minutes if match[1] == "+" else -minutes))


def _service_intervals(lines: list[str], offset: str) -> dict[tuple[str, int], list[tuple[datetime, datetime]]]:
    zone = _log_timezone(offset)
    active: dict[tuple[str, int, int], datetime] = {}
    intervals: dict[tuple[str, int], list[tuple[datetime, datetime]]] = {}
    for line in lines:
        match = LOG_EVENT.match(line.strip())
        if not match or match["chunk"] is None:
            continue
        key = (match["ws"], int(match["seg"]), int(match["chunk"]))
        stamp = datetime.strptime(match["time"], "%Y-%m-%d %H:%M:%S.%f")
        stamp = stamp.replace(tzinfo=zone).astimezone(timezone.utc)
        message = match["message"]
        if message.startswith("inference submit"):
            active[key] = stamp
        elif message.startswith("token_count=") and key in active:
            started = active.pop(key)
            if stamp >= started:
                intervals.setdefault(key[:2], []).append((started, stamp))
    return intervals


def _covered_ms(start: datetime, end: datetime,
                intervals: list[tuple[datetime, datetime]]) -> float:
    clipped = sorted((max(start, left), min(end, right))
                     for left, right in intervals
                     if min(end, right) > max(start, left))
    total = 0.0
    current_start = current_end = None
    for left, right in clipped:
        if current_start is None:
            current_start, current_end = left, right
        elif left <= current_end:
            current_end = max(current_end, right)
        else:
            total += (current_end - current_start).total_seconds() * 1000
            current_start, current_end = left, right
    if current_start is not None:
        total += (current_end - current_start).total_seconds() * 1000
    return total


def analyze(rows: list[dict], service_log_lines: list[str] | None = None,
            log_utc_offset: str = "+00:00") -> dict:
    intervals = (_service_intervals(service_log_lines, log_utc_offset)
                 if service_log_lines is not None else None)
    formal = [row for row in rows
              if str(row.get("is_empty_commit", False)).lower() not in ("true", "1", "yes")]
    formal.sort(key=lambda row: _time(row["received_at"]))
    latest_done: dict[str, datetime] = {}
    overlaps: list[float] = []
    wait_with: list[float] = []
    wait_without: list[float] = []
    chunk_overlaps: list[float] = []
    chunk_wait_with: list[float] = []
    chunk_evidence_count = 0
    no_prior = 0
    for row in formal:
        session = str(row["ws_id"])
        received = _time(row["received_at"])
        previous = latest_done.get(session)
        wait = row.get("queue_wait_ms")
        wait_ms = float(wait) if wait is not None and wait != "" else None
        if intervals is not None:
            same_segment = intervals.get((session, int(row["seg"])), [])
            if same_segment:
                chunk_evidence_count += 1
            if wait_ms is not None:
                queue_end = received + timedelta(milliseconds=wait_ms)
                covered = _covered_ms(received, queue_end, same_segment)
                if covered > 0:
                    chunk_overlaps.append(covered)
                    chunk_wait_with.append(wait_ms)
        if previous is None:
            no_prior += 1
        if previous is not None and previous > received:
            overlaps.append((previous - received).total_seconds() * 1000)
            if wait_ms is not None:
                wait_with.append(wait_ms)
        elif wait_ms is not None:
            wait_without.append(wait_ms)
        done = row.get("done_at")
        if done:
            completed = _time(done)
            if completed < received:
                raise ValueError(f"done before received: {session}/{row.get('seg')}")
            latest_done[session] = max(previous, completed) if previous else completed
    report = {"formal_commit_count": len(formal),
            "without_previous_formal_count": no_prior,
            "previous_inference_overlap_count": len(overlaps),
            "previous_inference_overlap_fraction": len(overlaps) / len(formal)
            if formal else None,
            "overlap_ms": _stats(overlaps),
            "queue_wait_ms_when_overlap": _stats(wait_with),
            "queue_wait_ms_without_overlap": _stats(wait_without)}
    if intervals is not None:
        report.update({
            "same_segment_chunk_interval_count": sum(map(len, intervals.values())),
            "same_segment_chunk_evidence_count": chunk_evidence_count,
            "same_segment_chunk_unknown_count": len(formal) - chunk_evidence_count,
            "same_segment_chunk_overlap_count": len(chunk_overlaps),
            "same_segment_chunk_overlap_ms": _stats(chunk_overlaps),
            "queue_wait_ms_with_chunk_overlap": _stats(chunk_wait_with),
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commits", required=True, help="per-commit JSONL or CSV")
    parser.add_argument("--service-log", help="server log with DEBUG token_count completion lines")
    parser.add_argument("--log-utc-offset", default="+00:00",
                        help="timezone of service-log timestamps, e.g. +00:00 or +08:00")
    parser.add_argument("--out", help="optional JSON report; prints if omitted")
    args = parser.parse_args()
    path = Path(args.commits)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = (list(csv.DictReader(handle)) if path.suffix.lower() == ".csv"
                else [json.loads(line) for line in handle if line.strip()])
    log_lines = (Path(args.service_log).read_text(encoding="utf-8").splitlines()
                 if args.service_log else None)
    report = json.dumps(analyze(rows, log_lines, args.log_utc_offset),
                        ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
