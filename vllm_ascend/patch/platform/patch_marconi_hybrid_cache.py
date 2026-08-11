# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Backport Marconi-style admission for aligned hybrid KV caches."""

import inspect

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec


def _has_native_marconi_support(scheduler_cls: type) -> bool:
    parameters = inspect.signature(scheduler_cls._mamba_block_aligned_split).parameters
    return "num_uncached_common_prefix_tokens" in parameters


_NATIVE_MARCONI_SUPPORT = _has_native_marconi_support(Scheduler)
_ORIGINAL_MAMBA_BLOCK_ALIGNED_SPLIT = Scheduler._mamba_block_aligned_split


def _patched_find_longest_cache_hit(
    self,
    block_hashes: list[BlockHash],
    max_cache_hit_length: int,
) -> tuple[tuple[list[KVCacheBlock], ...], int]:
    def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
        if kv_cache_spec.block_size == self.hash_block_size:
            return block_hashes
        return BlockHashListWithBlockSize(
            block_hashes,
            self.hash_block_size,
            kv_cache_spec.block_size,
        )

    num_groups = len(self.kv_cache_config.kv_cache_groups)
    hit_length = max_cache_hit_length
    longest_hit_length = 0
    hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
    is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(self.attention_groups[0][0], FullAttentionSpec)

    while True:
        curr_hit_length = hit_length
        for spec, group_ids, manager_cls in self.attention_groups:
            is_full_attn = isinstance(spec, FullAttentionSpec)
            cached_blocks = hit_blocks_by_group[group_ids[0]]
            if is_full_attn and cached_blocks is not None:
                num_blocks = curr_hit_length // spec.block_size
                curr_hit_length = num_blocks * spec.block_size
            else:
                hit_blocks = manager_cls.find_longest_cache_hit(
                    block_hashes=_get_block_hashes(spec),
                    max_length=curr_hit_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    use_eagle=self.use_eagle,
                    alignment_tokens=self.lcm_block_size,
                )
                curr_hit_length = len(hit_blocks[0]) * spec.block_size
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                longest_hit_length = max(longest_hit_length, curr_hit_length)

        if curr_hit_length >= hit_length:
            break
        hit_length = curr_hit_length
        if is_simple_hybrid:
            break

    spec, group_ids, _ = self.attention_groups[0]
    if isinstance(spec, FullAttentionSpec):
        num_blocks = hit_length // spec.block_size
        for group_id in group_ids:
            if (blocks := hit_blocks_by_group[group_id]) is not None:
                del blocks[num_blocks:]

    self.num_uncached_common_prefix_tokens = longest_hit_length - hit_length
    return tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group), hit_length


def _patched_mamba_block_aligned_split(
    self,
    request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    num_uncached_common_prefix_tokens: int | None = None,
) -> int:
    adjusted_tokens = _ORIGINAL_MAMBA_BLOCK_ALIGNED_SPLIT(
        self,
        request,
        num_new_tokens,
        num_new_local_computed_tokens,
        num_external_computed_tokens,
    )
    if num_uncached_common_prefix_tokens is None:
        if request.num_computed_tokens != 0:
            return adjusted_tokens
        coordinator = getattr(self.kv_cache_manager, "coordinator", None)
        num_uncached_common_prefix_tokens = getattr(
            coordinator,
            "num_uncached_common_prefix_tokens",
            0,
        )

    num_computed_tokens = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
    is_prefill = num_computed_tokens < max(
        request.num_prompt_tokens,
        request.num_tokens - 1,
    )
    block_size = self.cache_config.block_size
    if (
        is_prefill
        and num_uncached_common_prefix_tokens >= block_size
        and adjusted_tokens > num_uncached_common_prefix_tokens
    ):
        return num_uncached_common_prefix_tokens // block_size * block_size
    return adjusted_tokens


if not _NATIVE_MARCONI_SUPPORT:
    HybridKVCacheCoordinator.find_longest_cache_hit = _patched_find_longest_cache_hit
    Scheduler._mamba_block_aligned_split = _patched_mamba_block_aligned_split
