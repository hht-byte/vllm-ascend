"""Narrow, non-invasive vLLM Engine configuration validation."""

from collections.abc import Mapping
from typing import Any, Protocol, cast, runtime_checkable

from .compatibility import validate_runtime_versions
from .errors import InvalidEngineConfiguration

_PHYSICAL_BLOCK_SIZE = 128
_MINIMUM_AUDIO_ITEMS = 5


@runtime_checkable
class _EngineArgs(Protocol):
    def create_engine_config(self) -> object: ...


class _CacheConfig(Protocol):
    block_size: int
    enable_prefix_caching: bool
    hash_block_size: int | None


def prepare_vllm_config(
    engine_args: object,
    *,
    hash_block_size: int = 32,
    required_audio_items: int = 5,
    validate_versions: bool = True,
) -> object:
    """Validate an EngineArgs-produced config before setting its cache hash size."""
    if validate_versions:
        validate_runtime_versions()

    config = _create_engine_config(engine_args)
    cache = _read_cache_config(config)
    audio_limit = _read_audio_limit(config)

    _validate_hash_block_size(hash_block_size)
    _validate_required_audio_items(required_audio_items)
    _validate_cache(cache)
    if audio_limit < required_audio_items:
        raise InvalidEngineConfiguration("audio item limit is smaller than required")

    cache.hash_block_size = hash_block_size
    return config


def _create_engine_config(engine_args: object) -> object:
    if not isinstance(engine_args, _EngineArgs):
        raise InvalidEngineConfiguration(
            "engine_args must provide create_engine_config()"
        )
    return engine_args.create_engine_config()


def _read_cache_config(config: object) -> _CacheConfig:
    try:
        config_values = cast(Any, config)
        cache = cast(_CacheConfig, config_values.cache_config)
        _ = (cache.block_size, cache.enable_prefix_caching, cache.hash_block_size)
    except AttributeError as error:
        raise InvalidEngineConfiguration("cache_config is missing required fields") from error
    return cache


def _read_audio_limit(config: object) -> int:
    try:
        config_values = cast(Any, config)
        limit_per_prompt = config_values.multimodal_config.limit_per_prompt
    except AttributeError as error:
        raise InvalidEngineConfiguration(
            "multimodal_config is missing limit_per_prompt"
        ) from error

    if not isinstance(limit_per_prompt, Mapping):
        raise InvalidEngineConfiguration("limit_per_prompt must be a mapping")

    audio_limit = limit_per_prompt.get("audio", 0)
    if type(audio_limit) is not int:
        raise InvalidEngineConfiguration("audio item limit must be an integer")
    return audio_limit


def _validate_hash_block_size(hash_block_size: int) -> None:
    if type(hash_block_size) is not int or hash_block_size <= 0:
        raise InvalidEngineConfiguration("hash_block_size must be a positive integer")
    if _PHYSICAL_BLOCK_SIZE % hash_block_size:
        raise InvalidEngineConfiguration(
            "hash_block_size must divide the physical block size"
        )


def _validate_required_audio_items(required_audio_items: int) -> None:
    if (
        type(required_audio_items) is not int
        or required_audio_items < _MINIMUM_AUDIO_ITEMS
    ):
        raise InvalidEngineConfiguration(
            "required_audio_items must be at least five"
        )


def _validate_cache(cache: _CacheConfig) -> None:
    block_size = cache.block_size
    prefix_caching_enabled = cache.enable_prefix_caching
    if type(block_size) is not int or block_size != _PHYSICAL_BLOCK_SIZE:
        raise InvalidEngineConfiguration(
            "block_size must be 128 and prefix caching must be enabled"
        )
    if prefix_caching_enabled is not True:
        raise InvalidEngineConfiguration(
            "block_size must be 128 and prefix caching must be enabled"
        )
