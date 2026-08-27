"""Small Prometheus text helpers for vLLM cache counter deltas."""

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

CACHE_COUNTERS = (
    "vllm:mm_cache_queries",
    "vllm:mm_cache_hits",
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
)

_SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?P<labels>\{[^}]*\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s+\d+)?$"
)


@dataclass(frozen=True, slots=True)
class CounterDeltaSnapshot:
    """Counter deltas plus reasons why a value could not be reported."""

    values: dict[str, float | None]
    warnings: tuple[str, ...]


def _counter_series(
    text: str, metric_names: Iterable[str]
) -> dict[str, dict[str, float]]:
    wanted = {f"{name}_total": name for name in metric_names}
    series: dict[str, dict[str, float]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.fullmatch(line)
        if match is None:
            continue
        family = wanted.get(match.group("name"))
        if family is None:
            continue
        value = float(match.group("value"))
        if not math.isfinite(value):
            continue
        labels = match.group("labels") or ""
        series.setdefault(family, {})[labels] = value
    return series


def counter_deltas(
    before_text: str,
    after_text: str,
    metric_names: Iterable[str] = CACHE_COUNTERS,
) -> CounterDeltaSnapshot:
    """Return summed labeled counter deltas without inventing missing values.

    A lower after-value indicates a process or counter reset. That sample is
    reported as ``None`` because adding a guessed rollover would be misleading.
    """

    names = tuple(metric_names)
    before = _counter_series(before_text, names)
    after = _counter_series(after_text, names)
    values: dict[str, float | None] = {}
    warnings: list[str] = []
    for name in names:
        before_series = before.get(name)
        after_series = after.get(name)
        if before_series is None or after_series is None:
            missing = []
            if name not in before:
                missing.append("before")
            if name not in after:
                missing.append("after")
            values[name] = None
            warnings.append(f"{name} missing from {' and '.join(missing)} snapshot")
            continue
        before_labels = set(before_series)
        after_labels = set(after_series)
        if before_labels != after_labels:
            missing_labels = sorted(before_labels - after_labels)
            added_labels = sorted(after_labels - before_labels)
            details: list[str] = []
            if missing_labels:
                details.append(f"missing={missing_labels}")
            if added_labels:
                details.append(f"added={added_labels}")
            values[name] = None
            warnings.append(
                f"{name} counter series changed; delta is unavailable ({'; '.join(details)})"
            )
            continue
        reset_labels = sorted(
            label
            for label in before_labels
            if after_series[label] < before_series[label]
        )
        if reset_labels:
            values[name] = None
            warnings.append(
                f"{name} counter reset detected for series {reset_labels}; "
                "delta is unavailable"
            )
            continue
        values[name] = sum(
            after_series[label] - before_series[label]
            for label in sorted(before_labels)
        )
    return CounterDeltaSnapshot(values=values, warnings=tuple(warnings))
