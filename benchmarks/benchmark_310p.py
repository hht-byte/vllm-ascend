"""Ascend 310P correctness and latency matrix for stable audio windows.

The module deliberately has no import-time vLLM, torch, or Ascend dependency so
manifest validation and ``--help`` work on a development machine.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import math
import statistics
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

_MANIFEST_FIELDS = frozenset(
    {
        "id",
        "audio_npy",
        "sample_rate",
        "checkpoints_seconds",
        "language",
        "reference",
    }
)
_SAMPLE_RATE = 16_000
_AUDIO_FEATURE_HOP_SAMPLES = 160
_MIN_SECONDS = 6.0
_MAX_SECONDS = 10.0
_NUMERIC_ERROR_SCHEMA_VERSION = "qwen3-asr-numeric-error-v1"
_NUMERIC_ERROR_FIELDS = frozenset(
    {
        "schema_version",
        "dtype",
        "kernel_provenance",
        "capture_provenance",
        "embedding",
        "logits",
    }
)
_NUMERIC_ERROR_COMPONENT_FIELDS = frozenset(
    {"max_absolute_error", "max_relative_error"}
)
_AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
_DEFAULT_PROMPT = (
    f"<|im_start|>user\n{_AUDIO_PLACEHOLDER}<|im_end|>\n<|im_start|>assistant\n"
)


class ManifestError(ValueError):
    """A line-numbered benchmark manifest validation error."""


class EquivalenceMismatch(AssertionError):
    """Cache-off and reuse generated observably different outputs."""


class LifecycleProofError(RuntimeError):
    """A target cache lifecycle event was not observably demonstrated."""


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    id: str
    audio_npy: Path
    sample_rate: int
    checkpoints_seconds: tuple[float, ...]
    language: str
    reference: str
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    mode: str
    scenario: str
    record_id: str
    language: str
    reference: str
    detected_language: str | None
    audio_duration_seconds: float
    checkpoint_seconds: float
    window_seconds: int
    hash_block_size: int
    concurrency: int
    iteration: int
    request_id: str
    sealed_window_count: int
    open_window_count: int
    token_ids: tuple[int, ...]
    text: str
    num_cached_tokens: int | None
    ttft_ms: float | None
    final_latency_ms: float
    prometheus_counter_delta: dict[str, float | None]
    prometheus_warnings: tuple[str, ...]
    peak_npu_memory_bytes: int | None
    peak_npu_memory_provenance: str | None
    warnings: tuple[str, ...]
    open_window_duration_seconds: float = 0.0
    expected_reusable_audio_tokens: int = 0
    processor_cache_queries: float | None = None
    processor_cache_hits: float | None = None
    processor_cache_misses: float | None = None
    actual_encoder_cache_hits: float | None = None
    actual_encoder_cache_misses: float | None = None
    prefix_cache_hit_tokens: int | None = None
    prefill_computed_tokens: int | None = None
    tail_character_completion_latency_ms: float = 0.0
    inference_seq: int | None = None
    prefix_cache_recomputation_observed: bool | None = None
    lru_warm_prefix_cache_hit_tokens: int | None = None
    lru_prefix_eviction_observed: bool | None = None

    def as_json(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


class _OutputChoice(Protocol):
    token_ids: Sequence[int]
    text: str


class _RequestOutput(Protocol):
    outputs: Sequence[_OutputChoice]
    prompt_token_ids: Sequence[int] | None
    num_cached_tokens: int | None


class _CounterDeltaSnapshot(Protocol):
    values: dict[str, float | None]
    warnings: tuple[str, ...]


class _WindowedAdapter(Protocol):
    def build_request(
        self,
        *,
        session_id: str,
        utterance_epoch: int,
        accumulated_audio: np.ndarray,
        sample_rate: int,
        window_sec: int,
        is_final: bool,
        prompt: str,
    ) -> dict[str, object]: ...

    def release_session(self, session_id: str, utterance_epoch: int) -> None: ...


class _Engine(Protocol):
    def generate(
        self,
        request: Mapping[str, object],
        sampling: object,
        request_id: str,
    ) -> AsyncIterator[_RequestOutput]: ...

    async def reset_prefix_cache(self) -> bool: ...

    async def reset_encoder_cache(self) -> None: ...

    def shutdown(self, timeout: float | None = None) -> None: ...


@dataclass(frozen=True, slots=True)
class _Runtime:
    async_engine_args: Any
    async_llm: Any
    sampling_params: Any


@dataclass(frozen=True, slots=True)
class _RunOptions:
    model: str
    prompt: str
    dtype: str
    quantization: str | None
    max_tokens: int
    hash_block_size: int
    concurrency: int
    iteration: int
    inference_seq: int | None = None


@dataclass(frozen=True, slots=True)
class RequestTelemetry:
    open_window_duration_seconds: float
    expected_reusable_audio_tokens: int
    processor_cache_queries: float | None
    processor_cache_hits: float | None
    processor_cache_misses: float | None
    actual_encoder_cache_hits: float | None
    actual_encoder_cache_misses: float | None
    prefix_cache_hit_tokens: int | None
    prefill_computed_tokens: int | None
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NumericErrorComponent:
    max_absolute_error: float
    max_relative_error: float


@dataclass(frozen=True, slots=True)
class NumericErrorReport:
    dtype: str
    kernel_provenance: str
    capture_provenance: str
    embedding: NumericErrorComponent
    logits: NumericErrorComponent


def _line_error(line_number: int, message: str) -> ManifestError:
    return ManifestError(f"manifest line {line_number}: {message}")


def _required_string(raw: Mapping[str, object], field: str, line_number: int) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value.strip():
        raise _line_error(line_number, f"{field} must be a non-empty string")
    return value


def _load_audio_metadata(audio_path: Path, line_number: int) -> tuple[Path, float]:
    try:
        array = np.load(audio_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise _line_error(line_number, f"cannot load audio_npy: {error}") from error
    if array.ndim != 1:
        raise _line_error(line_number, "audio_npy must be one-dimensional")
    if array.dtype != np.dtype(np.float32):
        raise _line_error(line_number, "audio_npy dtype must be float32")
    if not array.flags.c_contiguous:
        raise _line_error(line_number, "audio_npy must be C-contiguous")
    duration = float(array.size) / _SAMPLE_RATE
    if not _MIN_SECONDS <= duration <= _MAX_SECONDS:
        raise _line_error(line_number, "audio duration must be 6 to 10 seconds")
    return audio_path.resolve(), duration


def _qwen3_asr_audio_token_length(sample_count: int) -> int:
    """Apply the vLLM 0.23 Qwen3-ASR output-length formula to PCM samples."""

    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    feature_length = math.ceil(sample_count / _AUDIO_FEATURE_HOP_SAMPLES)
    remainder = feature_length % 100
    feature_remainder_length = (remainder - 1) // 2 + 1
    return (
        ((feature_remainder_length - 1) // 2 + 1 - 1) // 2
        + 1
        + (feature_length // 100) * 13
    )


def expected_reusable_audio_tokens(
    *,
    sample_count: int,
    sample_rate: int,
    window_seconds: int,
    is_final: bool,
) -> int:
    """Count immutable Qwen3-ASR audio tokens reusable by the next request.

    A non-final call may reuse only complete fixed windows. A final call makes a
    non-empty tail immutable too, so an exact final retry can reuse it.
    """

    if type(sample_rate) is not int or sample_rate != _SAMPLE_RATE:
        raise ValueError("sample_rate must equal 16000")
    if type(window_seconds) is not int or window_seconds not in (2, 4, 8):
        raise ValueError("window_seconds must be one of 2, 4, or 8")
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    window_samples = window_seconds * sample_rate
    full_windows, remainder = divmod(sample_count, window_samples)
    tokens = full_windows * _qwen3_asr_audio_token_length(window_samples)
    if is_final and remainder:
        tokens += _qwen3_asr_audio_token_length(remainder)
    return tokens


def _nonnegative_counter(
    values: Mapping[str, float | None], name: str
) -> tuple[float | None, str | None]:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{name} delta is unavailable"
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        return None, f"{name} delta is invalid"
    return numeric, None


def _nonnegative_token_count(value: int | None, name: str) -> tuple[int | None, str | None]:
    if type(value) is not int or value < 0:
        return None, f"{name} is unavailable"
    return value, None


def derive_request_telemetry(
    *,
    sample_count: int,
    sample_rate: int,
    window_seconds: int,
    is_final: bool,
    prompt_token_ids: Sequence[int] | None,
    num_cached_tokens: int | None,
    concurrency: int,
    counter_delta: _CounterDeltaSnapshot,
) -> RequestTelemetry:
    """Derive request telemetry only from public output and validated counters."""

    window_samples = window_seconds * sample_rate
    remainder = sample_count % window_samples
    open_window_duration_seconds = (
        0.0 if is_final or not remainder else remainder / sample_rate
    )
    warnings = list(counter_delta.warnings)
    if type(concurrency) is not int or concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")
    warnings.append(
        "vllm:mm_cache_queries/hits provenance is the renderer/MM processor "
        "cache, not the EngineCore encoder-output cache; actual encoder cache "
        "hits/misses are unavailable"
    )
    if concurrency > 1:
        queries = None
        hits = None
        warnings.append(
            "per-request process-global Prometheus snapshots overlap when "
            f"concurrency={concurrency}; processor-cache attribution is unavailable"
        )
    else:
        queries, queries_warning = _nonnegative_counter(
            counter_delta.values, "vllm:mm_cache_queries"
        )
        hits, hits_warning = _nonnegative_counter(
            counter_delta.values, "vllm:mm_cache_hits"
        )
        if queries_warning is not None:
            warnings.append(queries_warning)
        if hits_warning is not None:
            warnings.append(hits_warning)
    if queries is None or hits is None or hits > queries:
        if queries is not None and hits is not None and hits > queries:
            warnings.append("vllm:mm_cache_hits exceeds vllm:mm_cache_queries")
        processor_misses = None
    else:
        processor_misses = queries - hits

    prefix_cache_hit_tokens, cached_warning = _nonnegative_token_count(
        num_cached_tokens, "num_cached_tokens"
    )
    if cached_warning is not None:
        warnings.append(cached_warning)
    if prompt_token_ids is None:
        prefill_computed_tokens = None
        warnings.append("prompt_token_ids is unavailable")
    elif prefix_cache_hit_tokens is None:
        prefill_computed_tokens = None
    elif prefix_cache_hit_tokens > len(prompt_token_ids):
        prefill_computed_tokens = None
        warnings.append("cached tokens exceed prompt length")
    else:
        prefill_computed_tokens = len(prompt_token_ids) - prefix_cache_hit_tokens

    return RequestTelemetry(
        open_window_duration_seconds=open_window_duration_seconds,
        expected_reusable_audio_tokens=expected_reusable_audio_tokens(
            sample_count=sample_count,
            sample_rate=sample_rate,
            window_seconds=window_seconds,
            is_final=is_final,
        ),
        processor_cache_queries=queries,
        processor_cache_hits=hits,
        processor_cache_misses=processor_misses,
        actual_encoder_cache_hits=None,
        actual_encoder_cache_misses=None,
        prefix_cache_hit_tokens=prefix_cache_hit_tokens,
        prefill_computed_tokens=prefill_computed_tokens,
        warnings=tuple(warnings),
    )


def _numeric_component(value: object, field: str) -> NumericErrorComponent:
    if not isinstance(value, dict) or set(value) != _NUMERIC_ERROR_COMPONENT_FIELDS:
        raise ValueError(f"numeric error {field} fields mismatch")
    parsed: dict[str, float] = {}
    for name in _NUMERIC_ERROR_COMPONENT_FIELDS:
        raw = value[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"numeric error {field}.{name} must be numeric")
        number = float(raw)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"numeric error {field}.{name} must be finite and non-negative")
        parsed[name] = number
    return NumericErrorComponent(**parsed)


def load_numeric_error_report(path: Path) -> NumericErrorReport:
    """Validate a non-sensitive numeric sidecar from an external supported capture."""

    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read numeric error sidecar: {error}") from error
    if not isinstance(decoded, dict) or set(decoded) != _NUMERIC_ERROR_FIELDS:
        raise ValueError("numeric error sidecar schema fields mismatch")
    if decoded["schema_version"] != _NUMERIC_ERROR_SCHEMA_VERSION:
        raise ValueError("numeric error sidecar schema version is unsupported")
    strings: dict[str, str] = {}
    for field in ("dtype", "kernel_provenance", "capture_provenance"):
        value = decoded[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"numeric error sidecar {field} must be non-empty")
        strings[field] = value
    return NumericErrorReport(
        dtype=strings["dtype"],
        kernel_provenance=strings["kernel_provenance"],
        capture_provenance=strings["capture_provenance"],
        embedding=_numeric_component(decoded["embedding"], "embedding"),
        logits=_numeric_component(decoded["logits"], "logits"),
    )


def _checkpoints(value: object, duration: float, line_number: int) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise _line_error(line_number, "checkpoints_seconds must be a non-empty list")
    checkpoints: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise _line_error(line_number, "checkpoints_seconds must be numeric")
        point = float(item)
        if not math.isfinite(point) or not _MIN_SECONDS <= point <= duration:
            raise _line_error(
                line_number,
                "checkpoints_seconds values must be between 6 seconds and audio duration",
            )
        checkpoints.append(point)
    if any(left >= right for left, right in pairwise(checkpoints)):
        raise _line_error(
            line_number, "checkpoints_seconds must be strictly increasing"
        )
    return tuple(checkpoints)


def load_manifest(path: Path) -> list[ManifestRecord]:
    """Load and fully validate the exact JSONL benchmark schema."""

    records: list[ManifestRecord] = []
    seen_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ManifestError(f"cannot read manifest: {error}") from error
    if not lines:
        raise ManifestError("manifest is empty")
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise _line_error(line_number, "blank lines are not allowed")
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as error:
            raise _line_error(line_number, f"invalid JSON: {error.msg}") from error
        if not isinstance(decoded, dict):
            raise _line_error(line_number, "record must be a JSON object")
        raw = cast(dict[str, object], decoded)
        if set(raw) != _MANIFEST_FIELDS:
            missing = sorted(_MANIFEST_FIELDS - set(raw))
            extra = sorted(set(raw) - _MANIFEST_FIELDS)
            raise _line_error(
                line_number,
                f"schema fields mismatch; missing={missing}, extra={extra}",
            )
        record_id = _required_string(raw, "id", line_number)
        if record_id in seen_ids:
            raise _line_error(line_number, "id must be unique")
        seen_ids.add(record_id)
        sample_rate = raw["sample_rate"]
        if type(sample_rate) is not int or sample_rate != _SAMPLE_RATE:
            raise _line_error(line_number, "sample_rate must equal 16000")
        audio_value = _required_string(raw, "audio_npy", line_number)
        audio_path = Path(audio_value)
        if not audio_path.is_absolute():
            audio_path = path.parent / audio_path
        resolved_audio, duration = _load_audio_metadata(audio_path, line_number)
        records.append(
            ManifestRecord(
                id=record_id,
                audio_npy=resolved_audio,
                sample_rate=sample_rate,
                checkpoints_seconds=_checkpoints(
                    raw["checkpoints_seconds"], duration, line_number
                ),
                language=_required_string(raw, "language", line_number),
                reference=_required_string(raw, "reference", line_number),
                duration_seconds=duration,
            )
        )
    return records


def cache_off_request(
    request: Mapping[str, object], *, request_id: str
) -> dict[str, object]:
    """Copy a request and replace stable audio UUIDs with request-unique IDs."""

    uuid_groups = request.get("multi_modal_uuids")
    if not isinstance(uuid_groups, Mapping):
        raise TypeError("request multi_modal_uuids must be a mapping")
    audio_ids = uuid_groups.get("audio")
    if not isinstance(audio_ids, list) or not all(
        isinstance(identifier, str) for identifier in audio_ids
    ):
        raise ValueError("request audio UUIDs must be a list of strings")
    one_time = [
        hashlib.sha256(
            f"qwen3-asr-cache-off-v1\0{request_id}\0{index}\0{identifier}".encode()
        ).hexdigest()
        for index, identifier in enumerate(audio_ids)
    ]
    copied = dict(request)
    copied["multi_modal_uuids"] = {**uuid_groups, "audio": one_time}
    return copied


def _minimal_reproducer(
    baseline: BenchmarkResult, reuse: BenchmarkResult
) -> dict[str, object]:
    return {
        "record_id": baseline.record_id,
        "scenario": baseline.scenario,
        "window_seconds": baseline.window_seconds,
        "hash_block_size": baseline.hash_block_size,
        "checkpoint_seconds": baseline.checkpoint_seconds,
        "cache_off": {
            "token_ids": list(baseline.token_ids),
            "text": baseline.text,
            "language": baseline.detected_language,
            "manifest_language": baseline.language,
        },
        "reuse": {
            "token_ids": list(reuse.token_ids),
            "text": reuse.text,
            "language": reuse.detected_language,
            "manifest_language": reuse.language,
        },
    }


def assert_equivalent(
    pairs: Sequence[tuple[BenchmarkResult, BenchmarkResult]],
    *,
    reproducer_path: Path,
) -> None:
    """Fail on first observable mismatch and save no audio/session identifiers."""

    for baseline, reuse in pairs:
        if (
            baseline.token_ids != reuse.token_ids
            or baseline.text != reuse.text
            or baseline.detected_language != reuse.detected_language
        ):
            reproducer_path.parent.mkdir(parents=True, exist_ok=True)
            reproducer_path.write_text(
                json.dumps(
                    _minimal_reproducer(baseline, reuse),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            raise EquivalenceMismatch(
                f"cache mismatch for {baseline.record_id}, "
                f"window={baseline.window_seconds}, "
                f"checkpoint={baseline.checkpoint_seconds}, "
                f"scenario={baseline.scenario}; reproducer={reproducer_path}"
            )


def _load_runtime() -> _Runtime:
    vllm = importlib.import_module("vllm")
    async_module = importlib.import_module("vllm.v1.engine.async_llm")
    return _Runtime(
        async_engine_args=vllm.AsyncEngineArgs,
        async_llm=async_module.AsyncLLM,
        sampling_params=vllm.SamplingParams,
    )


def parse_asr_output(raw_text: str) -> tuple[str | None, str]:
    """Split Qwen3-ASR's auto-language prefix from the transcription."""

    tag = "<asr_text>"
    language_prefix = "language "
    if tag not in raw_text:
        return None, raw_text
    prefix, text = raw_text.rsplit(tag, 1)
    if not prefix.startswith(language_prefix):
        return None, raw_text
    language = prefix[len(language_prefix) :].strip()
    return (language or None), text


def _create_engine(runtime: _Runtime, options: _RunOptions, *, mode: str) -> _Engine:
    engine_kwargs: dict[str, object] = {
        "model": options.model,
        "block_size": 128,
        "enable_prefix_caching": mode == "reuse",
        "limit_mm_per_prompt": {"audio": 5},
        "dtype": options.dtype,
    }
    if options.quantization is not None:
        engine_kwargs["quantization"] = options.quantization
    engine_args = runtime.async_engine_args(**engine_kwargs)
    if mode == "reuse":
        package = importlib.import_module("qwen3_asr_window_cache")
        config = package.prepare_vllm_config(
            engine_args, hash_block_size=options.hash_block_size
        )
    else:
        config = engine_args.create_engine_config()
    return cast(_Engine, runtime.async_llm.from_vllm_config(config))


def _adapter(model: str, window_seconds: int) -> _WindowedAdapter:
    package = importlib.import_module("qwen3_asr_window_cache")
    return cast(
        _WindowedAdapter,
        package.WindowedRequestAdapter(
            package.WindowCacheConfig(
                model_fingerprint=model,
                feature_extractor_fingerprint="vllm-qwen3-asr-0.23.0",
                audio_encoder_fingerprint="qwen3-asr-1.7b-audio-tower",
                supported_window_seconds=(window_seconds,),
            )
        ),
    )


def _prometheus_text() -> tuple[str, str | None]:
    try:
        client = importlib.import_module("prometheus_client")
        payload = client.generate_latest()
    except (ImportError, AttributeError, RuntimeError) as error:
        return "", f"Prometheus snapshot unavailable: {error}"
    if not isinstance(payload, bytes):
        return "", "Prometheus snapshot unavailable: generate_latest returned non-bytes"
    return payload.decode("utf-8", errors="replace"), None


def _counter_deltas(
    before: str, after: str, *, concurrency: int
) -> _CounterDeltaSnapshot:
    module_name = "benchmarks.prometheus_delta" if __package__ else "prometheus_delta"
    module = importlib.import_module(module_name)
    helper = cast(
        Callable[..., _CounterDeltaSnapshot],
        module.counter_deltas,
    )
    return helper(before, after, concurrency=concurrency)


def _peak_npu_memory() -> tuple[int | None, str | None, str | None]:
    """Avoid attributing frontend-process memory to EngineCore workers."""

    return (
        None,
        None,
        (
            "peak NPU memory unavailable from AsyncLLM frontend; collect a "
            "supported worker-side or external npu-smi/msprof measurement"
        ),
    )


def _load_audio(record: ManifestRecord) -> np.ndarray:
    value = np.load(record.audio_npy, allow_pickle=False)
    return cast(np.ndarray, value)


async def _generate_one(
    *,
    engine: _Engine,
    sampling: object,
    adapter: _WindowedAdapter,
    record: ManifestRecord,
    audio: np.ndarray,
    checkpoint: float,
    window_seconds: int,
    mode: str,
    scenario: str,
    options: _RunOptions,
    session_id: str,
    request_id: str,
    is_final: bool,
) -> BenchmarkResult:
    sample_count = round(checkpoint * record.sample_rate)
    request = adapter.build_request(
        session_id=session_id,
        utterance_epoch=options.iteration,
        accumulated_audio=audio[:sample_count],
        sample_rate=record.sample_rate,
        window_sec=window_seconds,
        is_final=is_final,
        prompt=options.prompt,
    )
    submitted = (
        cache_off_request(request, request_id=request_id)
        if mode == "cache-off"
        else request
    )
    before, before_warning = _prometheus_text()
    started_ns = time.perf_counter_ns()
    first_token_ns: int | None = None
    final: _RequestOutput | None = None
    async for output in engine.generate(submitted, sampling, request_id):
        if output.outputs and first_token_ns is None and output.outputs[0].token_ids:
            first_token_ns = time.perf_counter_ns()
        final = output
    finished_ns = time.perf_counter_ns()
    if final is None or not final.outputs:
        raise RuntimeError(f"vLLM returned no final output for request {request_id}")
    detected_language, transcription = parse_asr_output(final.outputs[0].text)
    after, after_warning = _prometheus_text()
    delta = _counter_deltas(before, after, concurrency=options.concurrency)
    peak, provenance, memory_warning = _peak_npu_memory()
    telemetry = derive_request_telemetry(
        sample_count=sample_count,
        sample_rate=record.sample_rate,
        window_seconds=window_seconds,
        is_final=is_final,
        prompt_token_ids=getattr(final, "prompt_token_ids", None),
        num_cached_tokens=final.num_cached_tokens,
        concurrency=options.concurrency,
        counter_delta=delta,
    )
    warnings = tuple(
        warning
        for warning in (
            before_warning,
            after_warning,
            memory_warning,
            *telemetry.warnings,
        )
        if warning is not None
    )
    full_windows, remainder = divmod(sample_count, window_seconds * record.sample_rate)
    return BenchmarkResult(
        mode=mode,
        scenario=scenario,
        record_id=record.id,
        language=record.language,
        reference=record.reference,
        detected_language=detected_language,
        audio_duration_seconds=record.duration_seconds,
        checkpoint_seconds=checkpoint,
        window_seconds=window_seconds,
        hash_block_size=options.hash_block_size,
        concurrency=options.concurrency,
        iteration=options.iteration,
        request_id=request_id,
        sealed_window_count=full_windows + (1 if is_final and remainder else 0),
        open_window_count=1 if remainder and not is_final else 0,
        token_ids=tuple(int(token) for token in final.outputs[0].token_ids),
        text=transcription,
        num_cached_tokens=final.num_cached_tokens,
        ttft_ms=(
            None
            if first_token_ns is None
            else (first_token_ns - started_ns) / 1_000_000
        ),
        final_latency_ms=(finished_ns - started_ns) / 1_000_000,
        prometheus_counter_delta=delta.values,
        prometheus_warnings=delta.warnings,
        peak_npu_memory_bytes=peak,
        peak_npu_memory_provenance=provenance,
        warnings=warnings,
        open_window_duration_seconds=telemetry.open_window_duration_seconds,
        expected_reusable_audio_tokens=telemetry.expected_reusable_audio_tokens,
        processor_cache_queries=telemetry.processor_cache_queries,
        processor_cache_hits=telemetry.processor_cache_hits,
        processor_cache_misses=telemetry.processor_cache_misses,
        actual_encoder_cache_hits=telemetry.actual_encoder_cache_hits,
        actual_encoder_cache_misses=telemetry.actual_encoder_cache_misses,
        prefix_cache_hit_tokens=telemetry.prefix_cache_hit_tokens,
        prefill_computed_tokens=telemetry.prefill_computed_tokens,
        tail_character_completion_latency_ms=(finished_ns - started_ns) / 1_000_000,
        inference_seq=options.inference_seq,
    )


async def _run_record_stream(
    *,
    engine: _Engine,
    sampling: object,
    adapter: _WindowedAdapter,
    record: ManifestRecord,
    window_seconds: int,
    mode: str,
    scenario: str,
    options: _RunOptions,
    session_suffix: str,
) -> list[BenchmarkResult]:
    audio = _load_audio(record)
    session_id = _session_id(record, window_seconds, options, session_suffix)
    results: list[BenchmarkResult] = []
    for index, checkpoint in enumerate(record.checkpoints_seconds):
        request_id = (
            f"{mode}:{session_id}:{scenario}:{index}:{time.perf_counter_ns()}"
        )
        results.append(
            await _generate_one(
                engine=engine,
                sampling=sampling,
                adapter=adapter,
                record=record,
                audio=audio,
                checkpoint=checkpoint,
                window_seconds=window_seconds,
                mode=mode,
                scenario=scenario,
                options=options,
                session_id=session_id,
                request_id=request_id,
                is_final=index == len(record.checkpoints_seconds) - 1,
            )
        )
    return results


def _session_id(
    record: ManifestRecord,
    window_seconds: int,
    options: _RunOptions,
    session_suffix: str,
) -> str:
    return f"benchmark:{record.id}:{window_seconds}:{options.iteration}:{session_suffix}"


async def _run_exact_final_retry(
    *,
    engine: _Engine,
    sampling: object,
    adapter: _WindowedAdapter,
    record: ManifestRecord,
    window_seconds: int,
    mode: str,
    options: _RunOptions,
    session_suffix: str,
    scenario: str = "exact_final_retry",
) -> BenchmarkResult:
    """Retry the exact final request with a distinct request ID."""

    audio = _load_audio(record)
    session_id = _session_id(record, window_seconds, options, session_suffix)
    checkpoint = record.checkpoints_seconds[-1]
    request_id = f"{mode}:{session_id}:{scenario}:{time.perf_counter_ns()}"
    return await _generate_one(
        engine=engine,
        sampling=sampling,
        adapter=adapter,
        record=record,
        audio=audio,
        checkpoint=checkpoint,
        window_seconds=window_seconds,
        mode=mode,
        scenario=scenario,
        options=options,
        session_id=session_id,
        request_id=request_id,
        is_final=True,
    )


async def _run_mode(
    records: Sequence[ManifestRecord],
    *,
    window_seconds: int,
    mode: str,
    options: _RunOptions,
    iterations: int,
    warmup_iterations: int,
) -> list[BenchmarkResult]:
    runtime = _load_runtime()
    engine = _create_engine(runtime, options, mode=mode)
    try:
        sampling = runtime.sampling_params(
            temperature=0.0,
            top_p=1.0,
            max_tokens=options.max_tokens,
        )
        adapter = _adapter(options.model, window_seconds)
        semaphore = asyncio.Semaphore(options.concurrency)

        async def run(
            record: ManifestRecord,
            *,
            iteration: int,
            phase: str,
        ) -> list[BenchmarkResult]:
            async with semaphore:
                iteration_options = replace(options, iteration=iteration)
                session_suffix = f"{phase}-{iteration}"
                try:
                    return await _run_record_stream(
                        engine=engine,
                        sampling=sampling,
                        adapter=adapter,
                        record=record,
                        window_seconds=window_seconds,
                        mode=mode,
                        scenario="steady",
                        options=iteration_options,
                        session_suffix=session_suffix,
                    )
                finally:
                    adapter.release_session(
                        _session_id(
                            record,
                            window_seconds,
                            iteration_options,
                            session_suffix,
                        ),
                        iteration_options.iteration,
                    )

        for warmup_iteration in range(warmup_iterations):
            await asyncio.gather(
                *(
                    run(record, iteration=warmup_iteration, phase="warmup")
                    for record in records
                )
            )
        groups = await asyncio.gather(
            *(
                run(record, iteration=iteration, phase="measured")
                for iteration in range(iterations)
                for record in records
            )
        )
        return [result for group in groups for result in group]
    finally:
        engine.shutdown()


def _result_key(result: BenchmarkResult) -> tuple[object, ...]:
    return (
        result.record_id,
        result.window_seconds,
        result.hash_block_size,
        result.checkpoint_seconds,
        result.iteration,
        result.scenario,
    )


def _pair_results(
    baseline: Sequence[BenchmarkResult], reuse: Sequence[BenchmarkResult]
) -> list[tuple[BenchmarkResult, BenchmarkResult]]:
    baseline_by_key = {_result_key(result): result for result in baseline}
    reuse_by_key = {_result_key(result): result for result in reuse}
    if set(baseline_by_key) != set(reuse_by_key):
        raise RuntimeError("cache-off and reuse produced different benchmark case keys")
    return [
        (baseline_by_key[key], reuse_by_key[key]) for key in sorted(baseline_by_key)
    ]


async def run_matrix(
    *,
    model: str,
    records: Sequence[ManifestRecord],
    window_seconds: Sequence[int],
    concurrency: Sequence[int],
    iterations: int,
    warmup_iterations: int,
    max_tokens: int,
    hash_block_size: int = 32,
    prompt: str = _DEFAULT_PROMPT,
    dtype: str = "auto",
    quantization: str | None = None,
) -> list[BenchmarkResult]:
    """Run isolated cache-off and reuse engines for every matrix point."""

    _validate_selected_hash_block_size(hash_block_size)
    all_results: list[BenchmarkResult] = []
    for window in window_seconds:
        for concurrency_value in concurrency:
            options = _RunOptions(
                model=model,
                prompt=prompt,
                dtype=dtype,
                quantization=quantization,
                max_tokens=max_tokens,
                hash_block_size=hash_block_size,
                concurrency=concurrency_value,
                iteration=0,
            )
            baseline = await _run_mode(
                records,
                window_seconds=window,
                mode="cache-off",
                options=options,
                iterations=iterations,
                warmup_iterations=warmup_iterations,
            )
            reuse = await _run_mode(
                records,
                window_seconds=window,
                mode="reuse",
                options=options,
                iterations=iterations,
                warmup_iterations=warmup_iterations,
            )
            pairs = _pair_results(baseline, reuse)
            assert_equivalent(
                pairs,
                reproducer_path=Path("benchmark-results/equivalence-reproducer.json"),
            )
            all_results.extend(baseline)
            all_results.extend(reuse)
    return all_results


async def _run_validation_mode(
    records: Sequence[ManifestRecord],
    *,
    window_seconds: int,
    mode: str,
    options: _RunOptions,
    lru_pressure_requests: int,
) -> list[BenchmarkResult]:
    runtime = _load_runtime()
    engine = _create_engine(runtime, options, mode=mode)
    try:
        sampling = runtime.sampling_params(
            temperature=0.0, top_p=1.0, max_tokens=options.max_tokens
        )
        adapter = _adapter(options.model, window_seconds)
        results: list[BenchmarkResult] = []
        for record in records:
            target_suffix = "steady"
            target_session = _session_id(
                record, window_seconds, options, target_suffix
            )
            try:
                steady = await _run_record_stream(
                    engine=engine,
                    sampling=sampling,
                    adapter=adapter,
                    record=record,
                    window_seconds=window_seconds,
                    mode=mode,
                    scenario="steady",
                    options=options,
                    session_suffix=target_suffix,
                )
                results.extend(steady)
                retry = await _run_exact_final_retry(
                    engine=engine,
                    sampling=sampling,
                    adapter=adapter,
                    record=record,
                    window_seconds=window_seconds,
                    mode=mode,
                    options=options,
                    session_suffix=target_suffix,
                )
                results.append(retry)
            finally:
                adapter.release_session(target_session, options.iteration)

            if mode == "reuse":
                reset_successful = await engine.reset_prefix_cache()
                if reset_successful is not True:
                    raise LifecycleProofError(
                        "reset_prefix_cache must return success before lifecycle replay"
                    )
                await engine.reset_encoder_cache()
            try:
                reset_result = await _run_record_stream(
                    engine=engine,
                    sampling=sampling,
                    adapter=adapter,
                    record=record,
                    window_seconds=window_seconds,
                    mode=mode,
                    scenario="after_cache_reset",
                    options=options,
                    session_suffix=target_suffix,
                )
                if mode == "reuse":
                    first_cached_tokens = reset_result[0].num_cached_tokens
                    if type(first_cached_tokens) is not int:
                        raise LifecycleProofError(
                            "prefix recomputation after reset is unavailable: "
                            "first replay num_cached_tokens is missing or invalid"
                        )
                    if first_cached_tokens != 0:
                        raise LifecycleProofError(
                            "prefix recomputation after reset is inconsistent: "
                            f"first replay cached {first_cached_tokens} tokens"
                        )
                    reset_result[0] = replace(
                        reset_result[0],
                        prefix_cache_recomputation_observed=True,
                    )
                results.extend(reset_result)
                lru_warm_retry = await _run_exact_final_retry(
                    engine=engine,
                    sampling=sampling,
                    adapter=adapter,
                    record=record,
                    window_seconds=window_seconds,
                    mode=mode,
                    options=options,
                    session_suffix=target_suffix,
                    scenario="lru_warm_retry",
                )
                results.append(lru_warm_retry)
            finally:
                adapter.release_session(target_session, options.iteration)

            for pressure_index in range(lru_pressure_requests):
                pressure_suffix = f"pressure-{pressure_index}"
                pressure_session = _session_id(
                    record, window_seconds, options, pressure_suffix
                )
                try:
                    await _run_record_stream(
                        engine=engine,
                        sampling=sampling,
                        adapter=adapter,
                        record=record,
                        window_seconds=window_seconds,
                        mode=mode,
                        scenario="pressure",
                        options=options,
                        session_suffix=pressure_suffix,
                    )
                finally:
                    adapter.release_session(
                        pressure_session, options.iteration
                    )
            try:
                pressure_replay = await _run_exact_final_retry(
                    engine=engine,
                    sampling=sampling,
                    adapter=adapter,
                    record=record,
                    window_seconds=window_seconds,
                    mode=mode,
                    options=options,
                    session_suffix=target_suffix,
                    scenario="lru_pressure_replay",
                )
                if mode == "reuse":
                    warm_tokens = lru_warm_retry.num_cached_tokens
                    replay_tokens = pressure_replay.num_cached_tokens
                    eviction_observed = (
                        type(warm_tokens) is int
                        and type(replay_tokens) is int
                        and 0 <= replay_tokens < warm_tokens
                    )
                    pressure_replay = replace(
                        pressure_replay,
                        lru_warm_prefix_cache_hit_tokens=warm_tokens,
                        lru_prefix_eviction_observed=eviction_observed,
                    )
                results.append(pressure_replay)
            finally:
                adapter.release_session(target_session, options.iteration)

            recreated_suffix = "recreated"
            recreated_session = _session_id(
                record, window_seconds, options, recreated_suffix
            )
            try:
                await _run_record_stream(
                    engine=engine,
                    sampling=sampling,
                    adapter=adapter,
                    record=record,
                    window_seconds=window_seconds,
                    mode=mode,
                    scenario="after_session_recreate",
                    options=options,
                    session_suffix=recreated_suffix,
                )
            finally:
                adapter.release_session(recreated_session, options.iteration)
            try:
                recreated_again = await _run_record_stream(
                    engine=engine,
                    sampling=sampling,
                    adapter=adapter,
                    record=record,
                    window_seconds=window_seconds,
                    mode=mode,
                    scenario="after_session_recreate",
                    options=options,
                    session_suffix=recreated_suffix,
                )
                results.extend(recreated_again)
            finally:
                adapter.release_session(recreated_session, options.iteration)
        return results
    finally:
        engine.shutdown()


def _certify_lru_pressure(
    baseline: list[BenchmarkResult],
    reuse: list[BenchmarkResult],
    *,
    lru_pressure_requests: int,
) -> tuple[list[BenchmarkResult], list[BenchmarkResult]]:
    """Promote raw LRU replays only after an observable cached-token reduction."""

    reuse_replays = [
        item for item in reuse if item.scenario == "lru_pressure_replay"
    ]
    if not reuse_replays or any(
        item.lru_prefix_eviction_observed is not True for item in reuse_replays
    ):
        raise LifecycleProofError(
            "LRU prefix eviction was not observed with "
            f"{lru_pressure_requests} pressure sessions; increase configured "
            "LRU pressure and rerun the full equivalence matrix"
        )

    return (
        [
            replace(item, scenario="after_lru_pressure")
            if item.scenario == "lru_pressure_replay"
            else item
            for item in baseline
        ],
        [
            replace(item, scenario="after_lru_pressure")
            if item.scenario == "lru_pressure_replay"
            else item
            for item in reuse
        ],
    )


async def run_equivalence_validation(
    *,
    model: str,
    manifest_path: Path,
    window_seconds: Sequence[int],
    max_tokens: int,
    lru_pressure_requests: int,
    hash_block_size: int = 32,
) -> list[tuple[BenchmarkResult, BenchmarkResult]]:
    """Run steady/reset/pressure/recreate scenarios on one 310P process."""

    _validate_selected_hash_block_size(hash_block_size)
    records = load_manifest(manifest_path)
    pairs: list[tuple[BenchmarkResult, BenchmarkResult]] = []
    for window in window_seconds:
        options = _RunOptions(
            model=model,
            prompt=_DEFAULT_PROMPT,
            dtype="auto",
            quantization=None,
            max_tokens=max_tokens,
            hash_block_size=hash_block_size,
            concurrency=1,
            iteration=0,
        )
        baseline = await _run_validation_mode(
            records,
            window_seconds=window,
            mode="cache-off",
            options=options,
            lru_pressure_requests=lru_pressure_requests,
        )
        reuse = await _run_validation_mode(
            records,
            window_seconds=window,
            mode="reuse",
            options=options,
            lru_pressure_requests=lru_pressure_requests,
        )
        baseline, reuse = _certify_lru_pressure(
            baseline,
            reuse,
            lru_pressure_requests=lru_pressure_requests,
        )
        pairs.extend(_pair_results(baseline, reuse))
    return pairs


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def _summary_records(results: Sequence[BenchmarkResult]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int, int], list[BenchmarkResult]] = {}
    for result in results:
        groups.setdefault(
            (
                result.mode,
                result.window_seconds,
                result.hash_block_size,
                result.concurrency,
            ),
            [],
        ).append(result)
    summaries: list[dict[str, object]] = []
    for (mode, window, hash_block_size, concurrency_value), group in sorted(
        groups.items()
    ):
        latencies = [result.final_latency_ms for result in group]
        ttfts = [result.ttft_ms for result in group if result.ttft_ms is not None]
        summaries.append(
            {
                "record_type": "summary",
                "mode": mode,
                "window_seconds": window,
                "hash_block_size": hash_block_size,
                "concurrency": concurrency_value,
                "request_count": len(group),
                "final_latency_ms_p50": statistics.median(latencies),
                "final_latency_ms_p95": _percentile(latencies, 0.95),
                "ttft_ms_p50": statistics.median(ttfts) if ttfts else None,
                "ttft_ms_p95": _percentile(ttfts, 0.95),
            }
        )
    return summaries


def _write_results(path: Path, results: Sequence[BenchmarkResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"record_type": "request", **result.as_json()}, ensure_ascii=False)
        for result in results
    ]
    lines.extend(
        json.dumps(summary, ensure_ascii=False) for summary in _summary_records(results)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate stable-window cache equivalence and latency on Ascend 310P"
    )
    parser.add_argument("--model")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--window-seconds", nargs="+", type=int, choices=(2, 4, 8)
    )
    parser.add_argument("--concurrency", nargs="+", type=int)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--hash-block-size", type=int, choices=(32, 128), default=32)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization")
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-numeric-sidecar", type=Path)
    return parser


def _positive(values: Sequence[int], name: str) -> None:
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive integers")


def _validate_selected_hash_block_size(hash_block_size: int) -> None:
    if type(hash_block_size) is not int or hash_block_size not in (32, 128):
        raise ValueError("hash_block_size must be 32 or 128")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_numeric_sidecar is not None:
        report = load_numeric_error_report(args.validate_numeric_sidecar)
        print(
            "numeric error sidecar valid "
            f"(dtype={report.dtype}, schema={_NUMERIC_ERROR_SCHEMA_VERSION})"
        )
        return 0
    missing = [
        name
        for name, value in (
            ("--model", args.model),
            ("--manifest", args.manifest),
            ("--window-seconds", args.window_seconds),
            ("--concurrency", args.concurrency),
            ("--output", args.output),
        )
        if value is None
    ]
    if missing:
        build_parser().error(f"the following arguments are required: {', '.join(missing)}")
    assert args.model is not None
    assert args.manifest is not None
    assert args.window_seconds is not None
    assert args.concurrency is not None
    assert args.output is not None
    _positive(args.concurrency, "concurrency")
    _positive((args.iterations,), "iterations")
    _positive((args.warmup_iterations,), "warmup_iterations")
    _positive((args.max_tokens,), "max_tokens")
    records = load_manifest(args.manifest)
    results = asyncio.run(
        run_matrix(
            model=args.model,
            records=records,
            window_seconds=args.window_seconds,
            concurrency=args.concurrency,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            max_tokens=args.max_tokens,
            hash_block_size=args.hash_block_size,
            prompt=args.prompt,
            dtype=args.dtype,
            quantization=args.quantization,
        )
    )
    _write_results(args.output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
