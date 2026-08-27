from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]


def _target_rows() -> list[str]:
    document = (PROJECT_ROOT / "docs/spec-traceability.md").read_text(encoding="utf-8")
    return [row for row in document.splitlines() if "目标机待验收" in row]


def test_target_acceptance_rows_use_executable_capture_commands() -> None:
    rows = _target_rows()
    assert rows
    for row in rows:
        assert "--help" not in row
        assert "`npu-smi info`" not in row


def test_profiling_traceability_records_before_during_after_memory_outputs() -> None:
    rows = [row for row in _target_rows() if "| 17.5 |" in row]
    assert len(rows) == 1
    row = rows[0]
    assert "/secure-validation/memory/before.txt" in row
    assert "/secure-validation/memory/during.txt" in row
    assert "/secure-validation/memory/after.txt" in row
