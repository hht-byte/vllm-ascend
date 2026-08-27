import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import benchmarks.benchmark_310p as benchmark
from benchmarks.benchmark_310p import (
    BenchmarkResult,
    EquivalenceMismatch,
    assert_equivalent,
    build_parser,
    cache_off_request,
    load_manifest,
    parse_asr_output,
)


def write_audio(path: Path, *, seconds: float = 6.0) -> Path:
    np.save(path, np.zeros(int(seconds * 16_000), dtype=np.float32))
    return path


def valid_record(audio_path: Path, *, record_id: str = "zh-001") -> dict[str, object]:
    return {
        "id": record_id,
        "audio_npy": str(audio_path),
        "sample_rate": 16_000,
        "checkpoints_seconds": [6],
        "language": "zh",
        "reference": "测试文本",
    }


def write_manifest(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_valid_manifest_loads_exact_schema_and_audio_metadata(tmp_path: Path) -> None:
    audio = write_audio(tmp_path / "audio.npy", seconds=10)
    record = valid_record(audio)
    record["checkpoints_seconds"] = [6, 8, 10]
    manifest = write_manifest(tmp_path / "manifest.jsonl", [record])

    loaded = load_manifest(manifest)

    assert len(loaded) == 1
    assert loaded[0].id == "zh-001"
    assert loaded[0].audio_npy == audio.resolve()
    assert loaded[0].duration_seconds == 10.0
    assert loaded[0].checkpoints_seconds == (6.0, 8.0, 10.0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.pop("reference"), "schema"),
        (lambda record: record.__setitem__("extra", True), "schema"),
        (lambda record: record.__setitem__("sample_rate", 8_000), "sample_rate"),
        (lambda record: record.__setitem__("language", ""), "language"),
        (lambda record: record.__setitem__("reference", "  "), "reference"),
        (
            lambda record: record.__setitem__("checkpoints_seconds", [6, 6]),
            "strictly increasing",
        ),
        (
            lambda record: record.__setitem__("checkpoints_seconds", [5]),
            "between 6 seconds",
        ),
    ],
)
def test_bad_manifest_record_reports_jsonl_line_number(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    first_audio = write_audio(tmp_path / "first.npy")
    second_audio = write_audio(tmp_path / "second.npy")
    first = valid_record(first_audio, record_id="first")
    second = valid_record(second_audio, record_id="second")
    assert callable(mutation)
    mutation(second)
    manifest = write_manifest(tmp_path / "manifest.jsonl", [first, second])

    with pytest.raises(ValueError, match=rf"line 2.*{message}"):
        load_manifest(manifest)


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.zeros((2, 48_000), dtype=np.float32), "one-dimensional"),
        (np.zeros(96_000, dtype=np.float64), "float32"),
        (np.zeros(80_000, dtype=np.float32), "6 to 10 seconds"),
        (np.zeros(176_000, dtype=np.float32), "6 to 10 seconds"),
    ],
)
def test_invalid_audio_array_reports_manifest_line(
    tmp_path: Path,
    array: np.ndarray,
    message: str,
) -> None:
    audio = tmp_path / "bad.npy"
    np.save(audio, array)
    manifest = write_manifest(tmp_path / "manifest.jsonl", [valid_record(audio)])

    with pytest.raises(ValueError, match=rf"line 1.*{message}"):
        load_manifest(manifest)


def test_duplicate_id_reports_second_line(tmp_path: Path) -> None:
    first_audio = write_audio(tmp_path / "first.npy")
    second_audio = write_audio(tmp_path / "second.npy")
    manifest = write_manifest(
        tmp_path / "manifest.jsonl",
        [valid_record(first_audio), valid_record(second_audio)],
    )

    with pytest.raises(ValueError, match=r"line 2.*unique"):
        load_manifest(manifest)


def test_malformed_json_and_blank_line_report_their_line(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{}\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 1.*schema"):
        load_manifest(malformed)

    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 1.*blank"):
        load_manifest(blank)


def test_cache_off_uuid_is_one_time_without_mutating_adapter_request() -> None:
    request: dict[str, object] = {
        "prompt": "p",
        "multi_modal_data": {"audio": [np.zeros(1, dtype=np.float32)]},
        "multi_modal_uuids": {"audio": ["stable-window-id"]},
        "cache_salt": "session-namespace",
    }

    first = cache_off_request(request, request_id="req-1")
    retry = cache_off_request(request, request_id="req-2")

    assert first["prompt"] == retry["prompt"] == "p"
    assert first["multi_modal_data"] is request["multi_modal_data"]
    assert first["cache_salt"] == retry["cache_salt"] == "session-namespace"
    assert first["multi_modal_uuids"] != retry["multi_modal_uuids"]
    assert request["multi_modal_uuids"] == {"audio": ["stable-window-id"]}


def test_direct_script_help_does_not_require_vllm_npu_or_installed_package(
    tmp_path: Path,
) -> None:
    script = Path(__file__).parents[2] / "benchmarks" / "benchmark_310p.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--manifest" in completed.stdout


def test_cli_defaults_to_three_warmup_iterations() -> None:
    args = build_parser().parse_args(
        [
            "--model",
            "model",
            "--manifest",
            "manifest.jsonl",
            "--window-seconds",
            "2",
            "--concurrency",
            "1",
            "--output",
            "results.jsonl",
        ]
    )

    assert args.warmup_iterations == 3
    assert args.iterations == 20
    assert args.prompt == (
        "<|im_start|>user\n"
        "<|audio_start|><|audio_pad|><|audio_end|>"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def test_qwen3_asr_language_prefix_is_compared_separately_from_text() -> None:
    assert parse_asr_output("language Chinese<asr_text>测试文本") == (
        "Chinese",
        "测试文本",
    )
    assert parse_asr_output("already post-processed") == (
        None,
        "already post-processed",
    )


def result(*, mode: str, token_ids: tuple[int, ...], text: str) -> BenchmarkResult:
    return BenchmarkResult(
        mode=mode,
        scenario="steady",
        record_id="zh-001",
        language="zh",
        reference="测试文本",
        detected_language="Chinese",
        audio_duration_seconds=6.0,
        checkpoint_seconds=6.0,
        window_seconds=2,
        concurrency=1,
        iteration=0,
        request_id=f"{mode}-request",
        sealed_window_count=3,
        open_window_count=0,
        token_ids=token_ids,
        text=text,
        num_cached_tokens=0,
        ttft_ms=1.0,
        final_latency_ms=2.0,
        prometheus_counter_delta={},
        prometheus_warnings=(),
        peak_npu_memory_bytes=None,
        peak_npu_memory_provenance=None,
        warnings=(),
    )


def test_equivalence_mismatch_saves_minimal_non_sensitive_reproducer(
    tmp_path: Path,
) -> None:
    output = tmp_path / "reproducer.json"

    with pytest.raises(EquivalenceMismatch):
        assert_equivalent(
            [
                (
                    result(mode="cache-off", token_ids=(1,), text="a"),
                    result(mode="reuse", token_ids=(2,), text="b"),
                )
            ],
            reproducer_path=output,
        )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == {
        "record_id": "zh-001",
        "scenario": "steady",
        "window_seconds": 2,
        "checkpoint_seconds": 6.0,
        "cache_off": {
            "token_ids": [1],
            "text": "a",
            "language": "Chinese",
            "manifest_language": "zh",
        },
        "reuse": {
            "token_ids": [2],
            "text": "b",
            "language": "Chinese",
            "manifest_language": "zh",
        },
    }


def test_npu_memory_is_explicitly_unavailable_from_the_asyncllm_frontend() -> None:
    value, provenance, warning = benchmark._peak_npu_memory()

    assert value is None
    assert provenance is None
    assert warning is not None
    assert "worker-side" in warning
    assert "npu-smi" in warning
    assert "msprof" in warning


@pytest.mark.parametrize(
    ("window_seconds", "checkpoint_seconds", "is_final", "expected_tokens"),
    (
        (2, 6, False, 78),
        (4, 6, False, 52),
        (8, 6, False, 0),
        (2, 6, True, 78),
        (4, 6, True, 78),
        (8, 6, True, 78),
    ),
)
def test_expected_reusable_audio_tokens_use_pinned_qwen3_asr_lengths(
    window_seconds: int,
    checkpoint_seconds: int,
    is_final: bool,
    expected_tokens: int,
) -> None:
    assert benchmark.expected_reusable_audio_tokens(
        sample_count=checkpoint_seconds * 16_000,
        sample_rate=16_000,
        window_seconds=window_seconds,
        is_final=is_final,
    ) == expected_tokens


@pytest.mark.parametrize(
    ("tail_samples", "expected_tail_tokens"),
    ((1, 1), (159, 1), (160, 1), (161, 1)),
)
def test_final_retry_counts_padded_audio_tail_tokens(
    tail_samples: int, expected_tail_tokens: int
) -> None:
    assert benchmark.expected_reusable_audio_tokens(
        sample_count=2 * 16_000 + tail_samples,
        sample_rate=16_000,
        window_seconds=2,
        is_final=True,
    ) == 26 + expected_tail_tokens


@pytest.mark.parametrize(
    ("window_seconds", "expected_tokens"), ((2, 26), (4, 52), (8, 104))
)
def test_exact_window_audio_token_invariants_remain_pinned(
    window_seconds: int, expected_tokens: int
) -> None:
    assert benchmark.expected_reusable_audio_tokens(
        sample_count=window_seconds * 16_000,
        sample_rate=16_000,
        window_seconds=window_seconds,
        is_final=True,
    ) == expected_tokens


def test_request_telemetry_derives_cache_and_prefill_values_without_fabrication() -> None:
    telemetry = benchmark.derive_request_telemetry(
        sample_count=96_000,
        sample_rate=16_000,
        window_seconds=4,
        is_final=False,
        prompt_token_ids=tuple(range(100)),
        num_cached_tokens=40,
        counter_delta=SimpleNamespace(
            values={
                "vllm:mm_cache_queries": 7.0,
                "vllm:mm_cache_hits": 5.0,
                "vllm:prefix_cache_queries": 11.0,
                "vllm:prefix_cache_hits": 8.0,
            },
            warnings=(),
        ),
    )

    assert telemetry.open_window_duration_seconds == 2.0
    assert telemetry.expected_reusable_audio_tokens == 52
    assert telemetry.actual_encoder_cache_hits == 5.0
    assert telemetry.actual_encoder_cache_misses == 2.0
    assert telemetry.prefix_cache_hit_tokens == 40
    assert telemetry.prefill_computed_tokens == 60
    assert telemetry.warnings == ()


def test_request_telemetry_marks_missing_or_impossible_values_unavailable() -> None:
    telemetry = benchmark.derive_request_telemetry(
        sample_count=96_000,
        sample_rate=16_000,
        window_seconds=4,
        is_final=True,
        prompt_token_ids=(1, 2),
        num_cached_tokens=3,
        counter_delta=SimpleNamespace(
            values={
                "vllm:mm_cache_queries": None,
                "vllm:mm_cache_hits": 1.0,
            },
            warnings=("counter unavailable",),
        ),
    )

    assert telemetry.open_window_duration_seconds == 0.0
    assert telemetry.expected_reusable_audio_tokens == 78
    assert telemetry.actual_encoder_cache_hits is None
    assert telemetry.actual_encoder_cache_misses is None
    assert telemetry.prefix_cache_hit_tokens == 3
    assert telemetry.prefill_computed_tokens is None
    assert "counter unavailable" in telemetry.warnings
    assert any("cached tokens exceed prompt length" in item for item in telemetry.warnings)


def test_numeric_error_sidecar_requires_exact_finite_non_sensitive_schema(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "numeric-error.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "qwen3-asr-numeric-error-v1",
                "dtype": "bfloat16",
                "kernel_provenance": "target kernel metadata digest",
                "capture_provenance": "supported external capture",
                "embedding": {
                    "max_absolute_error": 0.0,
                    "max_relative_error": 0.25,
                },
                "logits": {
                    "max_absolute_error": 0.01,
                    "max_relative_error": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )

    report = benchmark.load_numeric_error_report(sidecar)

    assert report.dtype == "bfloat16"
    assert report.embedding.max_relative_error == 0.25
    assert report.logits.max_absolute_error == 0.01

    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "qwen3-asr-numeric-error-v1",
                "dtype": "bfloat16",
                "kernel_provenance": "kernel",
                "capture_provenance": "capture",
                "embedding": {"max_absolute_error": -1, "max_relative_error": 0},
                "logits": {"max_absolute_error": 0, "max_relative_error": 0},
                "session_id": "forbidden",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema fields mismatch"):
        benchmark.load_numeric_error_report(sidecar)


def test_numeric_error_sidecar_cli_is_lazy_and_strict(tmp_path: Path) -> None:
    sidecar = tmp_path / "numeric-error.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "qwen3-asr-numeric-error-v1",
                "dtype": "float16",
                "kernel_provenance": "kernel",
                "capture_provenance": "external capture",
                "embedding": {"max_absolute_error": 0.0, "max_relative_error": 0.0},
                "logits": {"max_absolute_error": 0.0, "max_relative_error": 0.0},
            }
        ),
        encoding="utf-8",
    )
    script = Path(__file__).parents[2] / "benchmarks" / "benchmark_310p.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--validate-numeric-sidecar", str(sidecar)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "numeric error sidecar valid" in completed.stdout


class FakeEngine:
    def __init__(self) -> None:
        self.requests: list[tuple[dict[str, object], str]] = []
        self.shutdown_calls = 0

    async def generate(
        self,
        request: object,
        sampling: object,
        request_id: str,
    ) -> object:
        assert isinstance(request, dict)
        self.requests.append((request, request_id))
        yield SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[101], text="language Chinese<asr_text>测试")],
            prompt_token_ids=list(range(100)),
            num_cached_tokens=0,
        )

    async def reset_prefix_cache(self) -> bool:
        return True

    async def reset_encoder_cache(self) -> None:
        return None

    def shutdown(self, timeout: float | None = None) -> None:
        del timeout
        self.shutdown_calls += 1


def _options() -> object:
    return benchmark._RunOptions(
        model="model-fingerprint",
        prompt="<|audio_start|><|audio_pad|><|audio_end|>",
        dtype="auto",
        quantization=None,
        max_tokens=8,
        concurrency=1,
        iteration=0,
    )


def _fake_runtime() -> object:
    return SimpleNamespace(sampling_params=lambda **_: object())


@pytest.mark.parametrize("runner_name", ("_run_mode", "_run_validation_mode"))
def test_engine_is_shutdown_when_adapter_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
    runner_name: str,
) -> None:
    engine = FakeEngine()
    monkeypatch.setattr(benchmark, "_load_runtime", _fake_runtime)
    monkeypatch.setattr(benchmark, "_create_engine", lambda *args, **kwargs: engine)

    def fail_adapter(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("adapter setup failed")

    monkeypatch.setattr(benchmark, "_adapter", fail_adapter)
    runner = getattr(benchmark, runner_name)
    kwargs: dict[str, object] = {
        "records": (),
        "window_seconds": 2,
        "mode": "reuse",
        "options": _options(),
    }
    if runner_name == "_run_mode":
        kwargs.update(iterations=1, warmup_iterations=0)
    else:
        kwargs.update(lru_pressure_requests=0)

    with pytest.raises(RuntimeError, match="adapter setup failed"):
        asyncio.run(runner(**kwargs))

    assert engine.shutdown_calls == 1


@pytest.mark.parametrize(
    ("mode", "retry_uuid_matches_final"),
    (("reuse", True), ("cache-off", False)),
)
def test_validation_retains_all_checkpoints_and_retries_the_exact_final_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    retry_uuid_matches_final: bool,
) -> None:
    audio = write_audio(tmp_path / "audio.npy", seconds=8)
    record = benchmark.ManifestRecord(
        id="zh-001",
        audio_npy=audio,
        sample_rate=16_000,
        checkpoints_seconds=(6.0, 8.0),
        language="zh",
        reference="测试",
        duration_seconds=8.0,
    )
    engine = FakeEngine()
    monkeypatch.setattr(benchmark, "_load_runtime", _fake_runtime)
    monkeypatch.setattr(benchmark, "_create_engine", lambda *args, **kwargs: engine)

    results = asyncio.run(
        benchmark._run_validation_mode(
            (record,),
            window_seconds=2,
            mode=mode,
            options=_options(),
            lru_pressure_requests=1,
        )
    )

    expected_full_scenarios = {
        "steady",
        "after_cache_reset",
        "after_lru_pressure",
        "after_session_recreate",
    }
    for scenario in expected_full_scenarios:
        assert [item.checkpoint_seconds for item in results if item.scenario == scenario] == [
            6.0,
            8.0,
        ]
    assert [item.checkpoint_seconds for item in results if item.scenario == "exact_final_retry"] == [8.0]
    first = next(item for item in results if item.scenario == "steady")
    assert first.open_window_duration_seconds == 0.0
    assert first.expected_reusable_audio_tokens == 78
    assert first.prefix_cache_hit_tokens == 0
    assert first.prefill_computed_tokens == 100
    assert first.tail_character_completion_latency_ms == first.final_latency_ms
    assert first.inference_seq is None

    retry_index = next(
        index for index, (_, request_id) in enumerate(engine.requests) if "retry" in request_id
    )
    retry_request, retry_id = engine.requests[retry_index]
    steady_final_request, steady_final_id = engine.requests[1]
    assert retry_id != steady_final_id
    assert retry_request["prompt"] == steady_final_request["prompt"]
    assert retry_request["cache_salt"] == steady_final_request["cache_salt"]
    retry_audio = retry_request["multi_modal_data"]
    steady_final_audio = steady_final_request["multi_modal_data"]
    assert isinstance(retry_audio, dict)
    assert isinstance(steady_final_audio, dict)
    assert all(
        np.array_equal(left, right)
        for left, right in zip(retry_audio["audio"], steady_final_audio["audio"], strict=True)
    )
    assert (
        retry_request["multi_modal_uuids"] == steady_final_request["multi_modal_uuids"]
    ) is retry_uuid_matches_final
    assert engine.shutdown_calls == 1
