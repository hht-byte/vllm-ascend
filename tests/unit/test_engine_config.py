from types import SimpleNamespace
from typing import Any

import pytest

import qwen3_asr_window_cache.engine_config as engine_config_module
from qwen3_asr_window_cache import prepare_vllm_config
from qwen3_asr_window_cache.errors import (
    InvalidEngineConfiguration,
    UnsupportedRuntimeVersion,
)


class FakeEngineArgs:
    def __init__(self, config: object) -> None:
        self.config = config
        self.create_count = 0

    def create_engine_config(self) -> object:
        self.create_count += 1
        return self.config


def config(**overrides: Any) -> SimpleNamespace:
    cache_config = SimpleNamespace(
        block_size=128,
        enable_prefix_caching=True,
        hash_block_size=None,
    )
    multimodal_config = SimpleNamespace(limit_per_prompt={"audio": 5})
    result = SimpleNamespace(
        cache_config=cache_config,
        multimodal_config=multimodal_config,
        model="Qwen/Qwen3-ASR",
        device="npu",
        other_engine_setting="preserved",
    )
    for field, value in overrides.items():
        setattr(result, field, value)
    return result


def test_prepares_one_config_without_changing_other_engine_settings() -> None:
    fake_config = config()
    args = FakeEngineArgs(fake_config)

    result = prepare_vllm_config(args, validate_versions=False)

    assert result is fake_config
    assert result.cache_config.hash_block_size == 32
    assert result.model == "Qwen/Qwen3-ASR"
    assert result.device == "npu"
    assert result.other_engine_setting == "preserved"
    assert args.create_count == 1


def test_runtime_validation_happens_before_engine_config_is_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config = config()
    args = FakeEngineArgs(fake_config)

    def reject_runtime() -> None:
        raise UnsupportedRuntimeVersion("runtime rejected")

    monkeypatch.setattr(engine_config_module, "validate_runtime_versions", reject_runtime)

    with pytest.raises(UnsupportedRuntimeVersion, match="runtime rejected"):
        prepare_vllm_config(args)

    assert args.create_count == 0
    assert fake_config.cache_config.hash_block_size is None


@pytest.mark.parametrize(
    ("overrides", "hash_block_size", "required_audio_items"),
    [
        ({"cache_config": SimpleNamespace(block_size=128, enable_prefix_caching=False, hash_block_size=None)}, 32, 5),
        ({"cache_config": SimpleNamespace(block_size=64, enable_prefix_caching=True, hash_block_size=None)}, 32, 5),
        ({}, 0, 5),
        ({}, -1, 5),
        ({}, 48, 5),
        ({"multimodal_config": SimpleNamespace(limit_per_prompt={"audio": 4})}, 32, 5),
        ({}, 32, 0),
    ],
)
def test_invalid_engine_values_do_not_mutate_cache_hash(
    overrides: dict[str, object],
    hash_block_size: int,
    required_audio_items: int,
) -> None:
    fake_config = config(**overrides)
    args = FakeEngineArgs(fake_config)

    with pytest.raises(InvalidEngineConfiguration):
        prepare_vllm_config(
            args,
            hash_block_size=hash_block_size,
            required_audio_items=required_audio_items,
            validate_versions=False,
        )

    assert args.create_count == 1
    assert fake_config.cache_config.hash_block_size is None


@pytest.mark.parametrize(
    "invalid_config",
    [
        SimpleNamespace(multimodal_config=SimpleNamespace(limit_per_prompt={"audio": 5})),
        SimpleNamespace(cache_config=SimpleNamespace()),
        SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=128,
                enable_prefix_caching=True,
            ),
            multimodal_config=SimpleNamespace(limit_per_prompt={"audio": 5}),
        ),
        SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=128,
                enable_prefix_caching=True,
                hash_block_size=None,
            ),
        ),
        SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=128,
                enable_prefix_caching=True,
                hash_block_size=None,
            ),
            multimodal_config=SimpleNamespace(),
        ),
    ],
)
def test_missing_required_config_attributes_leave_cache_hash_unmodified(
    invalid_config: SimpleNamespace,
) -> None:
    args = FakeEngineArgs(invalid_config)
    cache = getattr(invalid_config, "cache_config", None)

    with pytest.raises(InvalidEngineConfiguration):
        prepare_vllm_config(args, validate_versions=False)

    assert args.create_count == 1
    if cache is not None and hasattr(cache, "hash_block_size"):
        assert cache.hash_block_size is None


def test_engine_args_without_public_config_factory_is_rejected() -> None:
    with pytest.raises(InvalidEngineConfiguration):
        prepare_vllm_config(object(), validate_versions=False)
