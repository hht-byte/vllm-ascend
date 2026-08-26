"""Compatibility checks for the pinned vLLM runtime distributions."""

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version

from .errors import UnsupportedRuntimeVersion

SUPPORTED_VLLM_VERSION = "0.23.0"
SUPPORTED_VLLM_ASCEND_VERSION = "0.23.0"


def validate_runtime_versions(
    *,
    version_getter: Callable[[str], str] = version,
) -> None:
    """Require the exact vLLM and vLLM-Ascend distribution versions."""
    _validate_distribution(
        distribution="vllm",
        expected=SUPPORTED_VLLM_VERSION,
        version_getter=version_getter,
    )
    _validate_distribution(
        distribution="vllm-ascend",
        expected=SUPPORTED_VLLM_ASCEND_VERSION,
        version_getter=version_getter,
    )


def _validate_distribution(
    *,
    distribution: str,
    expected: str,
    version_getter: Callable[[str], str],
) -> None:
    try:
        actual = version_getter(distribution)
    except PackageNotFoundError:
        actual = "missing"

    if actual != expected:
        raise UnsupportedRuntimeVersion(
            "Unsupported runtime version: "
            f"distribution={distribution!r}, expected={expected!r}, actual={actual!r}"
        )
