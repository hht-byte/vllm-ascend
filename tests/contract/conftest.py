from collections.abc import Iterator
from pathlib import Path

import pytest

FETCH_INSTRUCTION = "bash scripts/fetch_upstream.sh"
PROJECT_ROOT = Path(__file__).parents[2]


def _require_source_tree(relative_path: str) -> Path:
    source_root = PROJECT_ROOT / relative_path
    if not source_root.is_dir():
        pytest.fail(f"{relative_path} is missing; run: {FETCH_INSTRUCTION}")
    return source_root


@pytest.fixture(scope="session")
def vllm_source_root() -> Iterator[Path]:
    yield _require_source_tree(".upstream/vllm")


@pytest.fixture(scope="session")
def vllm_ascend_source_root() -> Iterator[Path]:
    yield _require_source_tree(".upstream/vllm-ascend")
