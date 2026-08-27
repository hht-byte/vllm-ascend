import os
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

PROJECT_ROOT = Path(__file__).parents[2]
VERIFY_SCRIPT = "scripts/verify_upstream_clean.sh"


def _verification_command() -> list[str]:
    if os.name != "nt":
        return ["bash", VERIFY_SCRIPT]

    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if not git_bash.is_file():
        pytest.fail(f"Git Bash is required at {git_bash}")
    return [
        str(git_bash),
        "-c",
        f"PATH=/mingw64/bin:/usr/bin:/bin; exec bash {VERIFY_SCRIPT}",
    ]


def test_upstream_trees_have_no_tracked_staged_or_untracked_changes(
    vllm_source_root: Path,
    vllm_ascend_source_root: Path,
) -> None:
    result = subprocess.run(
        _verification_command(),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output_lines = result.stdout.splitlines()
    assert any(
        re.fullmatch(r"\.upstream/vllm HEAD [0-9a-f]{40}", line)
        for line in output_lines
    )
    assert any(
        re.fullmatch(r"\.upstream/vllm-ascend HEAD [0-9a-f]{40}", line)
        for line in output_lines
    )
