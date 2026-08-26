from collections.abc import Callable
from importlib.metadata import PackageNotFoundError

import pytest

from qwen3_asr_window_cache import (
    validate_runtime_versions as public_validate_runtime_versions,
)
from qwen3_asr_window_cache.compatibility import (
    SUPPORTED_VLLM_ASCEND_VERSION,
    SUPPORTED_VLLM_VERSION,
    validate_runtime_versions,
)
from qwen3_asr_window_cache.errors import UnsupportedRuntimeVersion


def version_getter(versions: dict[str, str]) -> Callable[[str], str]:
    def get_version(distribution: str) -> str:
        try:
            return versions[distribution]
        except KeyError as error:
            raise PackageNotFoundError(distribution) from error

    return get_version


def test_exact_supported_runtime_versions_pass() -> None:
    public_validate_runtime_versions(
        version_getter=version_getter(
            {
                "vllm": SUPPORTED_VLLM_VERSION,
                "vllm-ascend": SUPPORTED_VLLM_ASCEND_VERSION,
            }
        )
    )


@pytest.mark.parametrize(
    ("versions", "distribution", "expected", "actual"),
    [
        (
            {"vllm-ascend": SUPPORTED_VLLM_ASCEND_VERSION},
            "vllm",
            SUPPORTED_VLLM_VERSION,
            "missing",
        ),
        (
            {"vllm": SUPPORTED_VLLM_VERSION},
            "vllm-ascend",
            SUPPORTED_VLLM_ASCEND_VERSION,
            "missing",
        ),
        (
            {"vllm": "0.23.0+local", "vllm-ascend": SUPPORTED_VLLM_ASCEND_VERSION},
            "vllm",
            SUPPORTED_VLLM_VERSION,
            "0.23.0+local",
        ),
        (
            {"vllm": "0.23.0.dev1", "vllm-ascend": SUPPORTED_VLLM_ASCEND_VERSION},
            "vllm",
            SUPPORTED_VLLM_VERSION,
            "0.23.0.dev1",
        ),
        (
            {"vllm": SUPPORTED_VLLM_VERSION, "vllm-ascend": "0.23.1"},
            "vllm-ascend",
            SUPPORTED_VLLM_ASCEND_VERSION,
            "0.23.1",
        ),
    ],
)
def test_unsupported_runtime_version_identifies_distribution_and_values(
    versions: dict[str, str],
    distribution: str,
    expected: str,
    actual: str,
) -> None:
    with pytest.raises(UnsupportedRuntimeVersion) as error:
        validate_runtime_versions(version_getter=version_getter(versions))

    message = str(error.value)
    assert distribution in message
    assert expected in message
    assert actual in message
