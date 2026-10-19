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
            npu = read_npu(command, pattern, args.timeout)
            output.write(json.dumps({"run_id": args.run_id, "time_utc": now,
                                     "monotonic_ns": monotonic_ns,
                                     "poll_ms": (time.monotonic_ns() - monotonic_ns) / 1e6,
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


def summarize(args: argparse.Namespace) -> None:
    samples = load_rows(args.samples)
    if any(row.get("run_id") != args.run_id for row in samples):
        raise ValueError("samples contain another run_id")
    waiting = [float(row["metrics"]["waiting"]) for row in samples
               if row.get("metrics", {}).get("waiting") is not None]
    running = [float(row["metrics"]["running"]) for row in samples
               if row.get("metrics", {}).get("running") is not None]
    utilization = [float(row["npu"]["utilization_pct"]) for row in samples
                   if row.get("npu", {}).get("utilization_pct") is not None]
    commits = load_rows(args.commits) if args.commits else []
    commit_latency = []
    queue_wait = []
    service_queue_size = []
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
        phase = row.get("phase")
        if phase in {"current", "downstream"}:
            values = phase_values.setdefault(phase, {"latency": [], "queue_wait": []})
            if latency is not None:
                values["latency"].append(latency)
            if wait is not None:
                values["queue_wait"].append(wait)
    result = {
        "run_id": args.run_id, "concurrency": args.concurrency,
        "sample_count": len(samples),
        "queue_waiting": stats(waiting), "queue_running": stats(running),
        "npu_utilization_pct": stats(utilization),
        "queue_nonzero_fraction": sum(x > 0 for x in waiting) / len(waiting)
        if waiting else None,
        "npu_missing_fraction": 1 - len(utilization) / len(samples)
        if samples else None,
        "formal_commit_count": len(commits) - empty,
        "empty_commit_count": empty,
        "commit_latency_ms": stats(commit_latency),
        "service_queue_wait_ms": stats(queue_wait),
        "service_queue_size": stats(service_queue_size),
        "by_phase": {phase: {"commit_latency_ms": stats(values["latency"]),
                             "service_queue_wait_ms": stats(values["queue_wait"])}
                     for phase, values in phase_values.items()},
        "assessment": "descriptive_only; compare formal latency under matched load",
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2),
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
        summarize(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
