"""Small sidecar for streaming-ASR load tests; requires only Python stdlib.

Run `sample` beside the ASR service for each concurrency stage. Optionally
export one JSONL/CSV row per formal commit and use `summarize` / `compare`.
NPU utilization is context, never a no-latency-regression verdict by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


GAUGES = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:kv_cache_usage_perc": "kv_usage",
}


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * percent / 100
    low = math.floor(pos)
    high = math.ceil(pos)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def parse_metrics(raw: str) -> tuple[dict, list[str]]:
    """Sum request gauges across engines; keep matching raw lines for audit."""
    values: dict[str, list[float]] = {name: [] for name in GAUGES.values()}
    matched = []
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        metric = parts[0].split("{", 1)[0]
        field = GAUGES.get(metric)
        if field is None:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        if math.isfinite(value):
            values[field].append(value)
            matched.append(line)
    return {
        "running": sum(values["running"]) if values["running"] else None,
        "waiting": sum(values["waiting"]) if values["waiting"] else None,
        "kv_usage_max": max(values["kv_usage"]) if values["kv_usage"] else None,
        "engine_counts": {name: len(rows) for name, rows in values.items()},
    }, matched


def read_npu(command: list[str], util_pattern: re.Pattern | None,
             timeout: float) -> dict:
    try:
        proc = subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout, check=False)
        raw = proc.stdout + proc.stderr
        result = {"status": "ok" if proc.returncode == 0 else "error",
                  "returncode": proc.returncode, "raw": raw,
                  "utilization_pct": None}
        if proc.returncode == 0:
            match = util_pattern.search(raw) if util_pattern else None
            if match:
                try:
                    utilization = float(match.group("util"))
                    if not math.isfinite(utilization) or not 0 <= utilization <= 100:
                        raise ValueError("utilization outside 0..100")
                    result["utilization_pct"] = utilization
                except (ValueError, IndexError):
                    result["status"] = "unparsed"
            else:
                result["status"] = "unparsed"
        return result
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "returncode": None,
                "raw": str(exc), "utilization_pct": None}


def read_metrics(url: str | None, timeout: float) -> dict:
    if not url:
        return {"status": "disabled", "running": None, "waiting": None,
                "kv_usage_max": None, "raw_lines": []}
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        parsed, lines = parse_metrics(raw)
        return {"status": "ok" if parsed["engine_counts"]["waiting"] else "unparsed",
                **parsed, "raw_lines": lines}
    except (OSError, UnicodeError, ValueError) as exc:
        return {"status": "error", "running": None, "waiting": None,
                "kv_usage_max": None, "raw_lines": [], "error": str(exc)}


def sample(args: argparse.Namespace) -> None:
    command = json.loads(args.npu_cmd_json)
    if not isinstance(command, list) or not command or not all(
            isinstance(part, str) and part for part in command):
        raise ValueError("--npu-cmd-json must be a nonempty JSON string array")
    pattern = re.compile(args.npu_util_regex) if args.npu_util_regex else None
    if pattern and "util" not in pattern.groupindex:
        raise ValueError("--npu-util-regex needs a named (?P<util>...) group")
    path = Path(args.out)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.duration
    next_tick = time.monotonic()
    with path.open("x", encoding="utf-8") as output:
        while time.monotonic() < deadline:
            now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            monotonic_ns = time.monotonic_ns()
            metrics = read_metrics(args.metrics_url, args.timeout)
            metrics_time_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            npu = read_npu(command, pattern, args.timeout)
            npu_time_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            output.write(json.dumps({"run_id": args.run_id, "time_utc": now,
                                     "monotonic_ns": monotonic_ns,
                                     "poll_ms": (time.monotonic_ns() - monotonic_ns) / 1e6,
                                     "metrics_time_utc": metrics_time_utc,
                                     "npu_time_utc": npu_time_utc,
                                     "metrics": metrics, "npu": npu},
                                    ensure_ascii=False) + "\n")
            output.flush()
            next_tick += args.interval
            if next_tick < time.monotonic():
                next_tick = time.monotonic() + args.interval
            time.sleep(max(0, min(next_tick - time.monotonic(),
                                  deadline - time.monotonic())))


def load_rows(path: str) -> list[dict]:
    source = Path(path)
    with source.open(encoding="utf-8", newline="") as handle:
        if source.suffix.lower() == ".csv":
            return list(csv.DictReader(handle))
        return [json.loads(line) for line in handle if line.strip()]


def _ms_between(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    first = datetime.fromisoformat(start.replace("Z", "+00:00"))
    last = datetime.fromisoformat(end.replace("Z", "+00:00"))
    if first.tzinfo is None or last.tzinfo is None:
        raise ValueError("commit timestamps must include a timezone")
    elapsed = (last - first).total_seconds() * 1000
    if elapsed < 0:
        raise ValueError("negative commit duration")
    return elapsed


def _number(row: dict, name: str) -> float | None:
    value = row.get(name)
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"invalid {name}: {value}")
    return result


def _is_true(value: object) -> bool:
    return value is True or str(value).lower() in {"true", "1", "yes"}


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp needs a timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _sample_time(row: dict, source: str) -> datetime:
    value = row.get(f"{source}_time_utc") or row.get("time_utc")
    if not value:
        raise ValueError("sample has no time_utc")
    return _utc(value)


def _nearest_sample(rows: list[dict], source: str, received: datetime,
                    max_gap_ms: float) -> dict | None:
    if not rows:
        return None
    row = min(rows, key=lambda item: abs((_sample_time(item, source) - received).total_seconds()))
    offset_ms = round((_sample_time(row, source) - received).total_seconds() * 1000, 3)
    if abs(offset_ms) > max_gap_ms:
        return None
    values = row[source]
    fields = ("waiting", "running") if source == "metrics" else ("utilization_pct",)
    return {"sample_time_utc": row.get(f"{source}_time_utc") or row["time_utc"],
            "offset_ms": offset_ms,
            **{name: values.get(name) for name in fields}}


def summarize(args: argparse.Namespace) -> None:
    samples = load_rows(args.samples)
    if any(row.get("run_id") != args.run_id for row in samples):
        raise ValueError("samples contain another run_id")
    if bool(args.test_start_utc) != bool(args.test_end_utc):
        raise ValueError("--test-start-utc and --test-end-utc must be supplied together")
    start = _utc(args.test_start_utc) if args.test_start_utc else None
    end = _utc(args.test_end_utc) if args.test_end_utc else None
    if start and start >= end:
        raise ValueError("test window must have start before end")
    def in_window(point: datetime) -> bool:
        return start is None or start <= point < end
    metrics_samples = [row for row in samples if in_window(_sample_time(row, "metrics"))]
    npu_samples = [row for row in samples if in_window(_sample_time(row, "npu"))]
    samples = [row for row in samples if in_window(_utc(row["time_utc"]))]
    waiting = [float(row["metrics"]["waiting"]) for row in metrics_samples
               if row.get("metrics", {}).get("waiting") is not None]
    running = [float(row["metrics"]["running"]) for row in metrics_samples
               if row.get("metrics", {}).get("running") is not None]
    utilization = [float(row["npu"]["utilization_pct"]) for row in npu_samples
                   if row.get("npu", {}).get("utilization_pct") is not None]
    commits = load_rows(args.commits) if args.commits else []
    if args.aligned_out and not args.commits:
        raise ValueError("--aligned-out requires --commits")
    if start or args.aligned_out:
        for row in commits:
            if not row.get("received_at"):
                raise ValueError("each commit needs received_at for time alignment")
        commits = [row for row in commits if in_window(_utc(row["received_at"]))]
    aligned = []
    commit_latency = []
    queue_wait = []
    service_queue_size = []
    audio_blocks_pending = []
    phase_values: dict[str, dict[str, list[float]]] = {}
    empty = 0
    keys = set()
    for row in commits:
        key = (row.get("ws_id"), row.get("seg"))
        if key != (None, None):
            if key in keys:
                raise ValueError(f"duplicate commit key: {key}")
            keys.add(key)
        if _is_true(row.get("is_empty_commit", False)):
            empty += 1
            continue
        if args.aligned_out:
            received = _utc(row["received_at"])
            aligned.append({**row,
                            "metrics_at_receive": _nearest_sample(
                                metrics_samples, "metrics", received, args.max_align_gap_ms),
                            "npu_at_receive": _nearest_sample(
                                npu_samples, "npu", received, args.max_align_gap_ms)})
        latency = _number(row, "commit_latency_ms")
        if latency is None:
            latency = _ms_between(row.get("received_at"), row.get("done_at"))
        wait = _number(row, "queue_wait_ms")
        if wait is None:
            wait = _ms_between(row.get("received_at"), row.get("consumed_at"))
        if latency is not None:
            commit_latency.append(latency)
        if wait is not None:
            queue_wait.append(wait)
        size = _number(row, "queue_size")
        if size is not None:
            service_queue_size.append(size)
            if size >= 1:
                audio_blocks_pending.append(size - 1)  # commit already enqueued None
        phase = row.get("phase")
        if phase in {"current", "downstream"}:
            values = phase_values.setdefault(phase, {"latency": [], "queue_wait": []})
            if latency is not None:
                values["latency"].append(latency)
            if wait is not None:
                values["queue_wait"].append(wait)
    result = {
        "run_id": args.run_id, "concurrency": args.concurrency,
        "test_start_utc": args.test_start_utc, "test_end_utc": args.test_end_utc,
        "sample_count": len(samples),
        "metrics_sample_count": len(metrics_samples),
        "npu_sample_count": len(npu_samples),
        "queue_waiting": stats(waiting), "queue_running": stats(running),
        "npu_utilization_pct": stats(utilization),
        "queue_nonzero_fraction": sum(x > 0 for x in waiting) / len(waiting)
        if waiting else None,
        "npu_missing_fraction": 1 - len(utilization) / len(npu_samples)
        if npu_samples else None,
        "formal_commit_count": len(commits) - empty,
        "empty_commit_count": empty,
        "commit_latency_ms": stats(commit_latency),
        "service_queue_wait_ms": stats(queue_wait),
        "service_queue_size": stats(service_queue_size),
        "audio_blocks_pending_at_commit": stats(audio_blocks_pending),
        "by_phase": {phase: {"commit_latency_ms": stats(values["latency"]),
                             "service_queue_wait_ms": stats(values["queue_wait"])}
                     for phase, values in phase_values.items()},
        "assessment": "descriptive_only; compare formal latency under matched load",
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    if args.aligned_out:
        Path(args.aligned_out).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in aligned),
            encoding="utf-8")


def compare(args: argparse.Namespace) -> None:
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    if baseline["concurrency"] != probe["concurrency"]:
        raise ValueError("different concurrency stages cannot be compared")
    result = {"concurrency": baseline["concurrency"],
              "baseline_run_id": baseline["run_id"],
              "probe_run_id": probe["run_id"], "deltas": {},
              "decision": "inconclusive_without_matched_load_and_latency_margin"}
    for key in ("commit_latency_ms", "service_queue_wait_ms", "queue_waiting",
                "npu_utilization_pct"):
        old = baseline[key]["p95"]
        new = probe[key]["p95"]
        result["deltas"][key + "_p95"] = new - old if old is not None and new is not None else None
    for phase in ("current", "downstream"):
        for key in ("commit_latency_ms", "service_queue_wait_ms"):
            old = baseline.get("by_phase", {}).get(phase, {}).get(key, {}).get("p95")
            new = probe.get("by_phase", {}).get(phase, {}).get(key, {}).get("p95")
            result["deltas"][phase + "_" + key + "_p95"] = (
                new - old if old is not None and new is not None else None)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                              encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    capture = sub.add_parser("sample")
    capture.add_argument("--run-id", required=True)
    capture.add_argument("--metrics-url")
    capture.add_argument("--npu-cmd-json", default='["npu-smi","info"]')
    capture.add_argument("--npu-util-regex")
    capture.add_argument("--interval", type=float, default=1.0)
    capture.add_argument("--duration", type=float, required=True)
    capture.add_argument("--timeout", type=float, default=2.0)
    capture.add_argument("--out", required=True)
    report = sub.add_parser("summarize")
    report.add_argument("--run-id", required=True)
    report.add_argument("--concurrency", type=int, required=True)
    report.add_argument("--samples", required=True)
    report.add_argument("--commits", help="optional per-formal-commit JSONL or CSV")
    report.add_argument("--test-start-utc", help="inclusive ISO-8601 test start with timezone")
    report.add_argument("--test-end-utc", help="exclusive ISO-8601 test end with timezone")
    report.add_argument("--aligned-out", help="optional per-commit JSONL with nearest monitor readings")
    report.add_argument("--max-align-gap-ms", type=float, default=3000)
    report.add_argument("--out", required=True)
    diff = sub.add_parser("compare")
    diff.add_argument("--baseline", required=True)
    diff.add_argument("--probe", required=True)
    diff.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.action == "sample":
        if args.interval <= 0 or args.duration <= 0 or args.timeout <= 0:
            parser.error("interval, duration and timeout must be positive")
        sample(args)
    elif args.action == "summarize":
        if args.max_align_gap_ms < 0:
            parser.error("--max-align-gap-ms must be nonnegative")
        summarize(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
