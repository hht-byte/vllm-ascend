import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

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
