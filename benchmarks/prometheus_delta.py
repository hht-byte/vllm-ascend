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
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s+\d+)?$"
)


@dataclass(frozen=True, slots=True)
class CounterDeltaSnapshot:
    """Counter deltas plus reasons why a value could not be reported."""

    values: dict[str, float | None]
    warnings: tuple[str, ...]


def _counter_totals(text: str, metric_names: Iterable[str]) -> dict[str, float]:
    wanted = {f"{name}_total": name for name in metric_names}
    totals: dict[str, float] = {}
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
        totals[family] = totals.get(family, 0.0) + value
    return totals


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
    before = _counter_totals(before_text, names)
    after = _counter_totals(after_text, names)
    values: dict[str, float | None] = {}
    warnings: list[str] = []
    for name in names:
        if name not in before or name not in after:
            missing = []
            if name not in before:
                missing.append("before")
            if name not in after:
                missing.append("after")
            values[name] = None
            warnings.append(f"{name} missing from {' and '.join(missing)} snapshot")
            continue
        if after[name] < before[name]:
            values[name] = None
            warnings.append(f"{name} counter reset detected; delta is unavailable")
            continue
        values[name] = after[name] - before[name]
    return CounterDeltaSnapshot(values=values, warnings=tuple(warnings))
