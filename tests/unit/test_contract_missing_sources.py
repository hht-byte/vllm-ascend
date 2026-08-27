import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
CONTRACT_FILES = (
    Path("tests/contract/conftest.py"),
    Path("tests/contract/test_upstream_clean.py"),
    Path("scripts/verify_upstream_clean.sh"),
)


def _copy_clean_contract(project_root: Path) -> None:
    for relative_path in CONTRACT_FILES:
        target = project_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative_path, target)


def test_clean_contract_reports_fetch_instruction_for_each_missing_tree(
    tmp_path: Path,
) -> None:
    cases = (
        (None, ".upstream/vllm"),
        (Path(".upstream/vllm"), ".upstream/vllm-ascend"),
    )

    for case_index, (existing_tree, missing_tree) in enumerate(cases):
        project_root = tmp_path / f"case-{case_index}"
        _copy_clean_contract(project_root)
        if existing_tree is not None:
            (project_root / existing_tree).mkdir(parents=True)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-m",
                "contract",
                "tests/contract/test_upstream_clean.py",
            ],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr
        assert result.returncode == 1, output
        assert (
            f"Failed: {missing_tree} is missing; run: bash scripts/fetch_upstream.sh"
        ) in output
