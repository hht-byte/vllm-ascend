# SPDX-License-Identifier: Apache-2.0
"""Fixed auxiliary pages, independent of target KV/block allocation."""

import inspect
from math import ceil

import torch

try:
    from .asr_mtp_heads import CacheArena, HeadHistory, RequestHistory
except ImportError:
    from asr_mtp_heads import CacheArena, HeadHistory, RequestHistory


def scatter_pa_supports_nd(operation):
    """Inspect once at startup; old NZ-only ScatterPa schemas lack cache_mode."""
    try:
        parameters = inspect.signature(operation).parameters
        if "cache_mode" in parameters:
            return True
        if not any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        ):
            return False
    except (TypeError, ValueError):
        pass
    packet = getattr(torch.ops.npu, "npu_scatter_pa_kv_cache", None)
    schema = getattr(getattr(packet, "default", None), "_schema", None)
    return schema is not None and any(a.name == "cache_mode" for a in schema.arguments)


class PagedCacheArena(CacheArena):
    """Hidden-only flat arena and one shared ND paged KV allocation.

    HeadHistory exposes strided views into the same pages for eager fallback.
    Graphs never convert the whole historical cache into contiguous TND data.
    Padding writes use distinct dummy token slots, not a live page or -1 slot.
    """

    @staticmethod
    def budget_elements(config, heads, context, block_size, dummy_tokens):
        rounded = ceil(context / block_size) * block_size
        dummy = ceil(dummy_tokens / block_size) * block_size
        kv_width = config.num_key_value_heads * config.head_dim
        return (
            (heads + 1) * context * config.hidden_size + 2 * heads * rounded * kv_width,
            2 * heads * dummy * kv_width,
        )

    def __init__(
        self,
        config,
        heads,
        *,
        max_context,
        slots,
        block_size,
        dummy_tokens,
        device,
        dtype,
        scratch_elements=0,
        write_backend="auto",
        npu_api=None,
    ):
        if any(
            type(n) is not int or n < 1
            for n in (heads, max_context, slots, block_size, dummy_tokens)
        ):
            raise ValueError("paged MTP cache dimensions must be positive integers")
        if type(scratch_elements) is not int or scratch_elements < 0:
            raise ValueError("paged MTP scratch_elements must be nonnegative")
        if write_backend not in ("auto", "scatter_pa", "scatter_nd"):
            raise ValueError(
                "asr_mtp.kv_write_backend must be auto, scatter_pa or scatter_nd"
            )
        self.config, self.num_heads, self.max_context = config, heads, max_context
        self.block_size = block_size
        self.blocks_per_slot = ceil(max_context / block_size)
        self.slot_capacity = self.blocks_per_slot * block_size
        self.dummy_blocks = ceil(dummy_tokens / block_size)
        self.dummy_base = slots * self.slot_capacity
        self.total_blocks = slots * self.blocks_per_slot + self.dummy_blocks
        self.token_capacity = self.total_blocks * block_size
        self.paged_kv = True
        hidden_per_slot = (heads + 1) * max_context * config.hidden_size
        self.scratch_offset = slots * hidden_per_slot
        self.graph_storage = torch.empty(
            self.scratch_offset + scratch_elements, device=device, dtype=dtype
        )
        self.storage = self.graph_storage[: self.scratch_offset].view(
            slots, hidden_per_slot
        )
        self.kv = torch.zeros(
            (
                heads,
                2,
                self.total_blocks,
                block_size,
                config.num_key_value_heads,
                config.head_dim,
            ),
            device=device,
            dtype=dtype,
        )
        width = max(self.blocks_per_slot, self.dummy_blocks)
        self.block_rows = []
        for i in range(slots + 1):
            begin = i * self.blocks_per_slot
            count = self.blocks_per_slot if i < slots else self.dummy_blocks
            self.block_rows.append(
                tuple(range(begin, begin + count)) + (begin,) * (width - count)
            )
        self.free, self.leased = list(reversed(range(slots))), set()
        self.api = npu_api
        if self.storage.device.type == "npu" and self.api is None:
            import torch_npu

            self.api = torch_npu
        self.scatter_pa = None
        self.scatter_nd = None
        if self.api is not None:
            operation = getattr(self.api, "npu_scatter_pa_kv_cache", None)
            if (
                write_backend != "scatter_nd"
                and operation is not None
                and scatter_pa_supports_nd(operation)
            ):
                self.scatter_pa = operation
            elif write_backend == "scatter_pa":
                raise ValueError(
                    "paged MTP scatter_pa requires the ND cache_mode=Norm API"
                )
            if self.scatter_pa is None:
                self.scatter_nd = getattr(self.api, "npu_scatter_nd_update_", None)
                if not callable(self.scatter_nd):
                    raise ValueError(
                        "paged MTP requires a combined ScatterNd or ND ScatterPa writer"
                    )
        self.write_backend = (
            "scatter_pa" if self.scatter_pa is not None else "scatter_nd"
        )

    @property
    def cache_bytes(self):
        return (self.scratch_offset + self.kv.numel()) * self.storage.element_size()

    def acquire(self):
        if not self.free:
            return None
        slot = self.free.pop()
        self.leased.add(slot)
        hidden = self.storage[slot].view(
            self.num_heads + 1, self.max_context, self.config.hidden_size
        )
        begin = slot * self.blocks_per_slot
        histories = []
        for d in range(self.num_heads):
            views = []
            for kind in range(2):
                values = self.kv[d, kind, begin : begin + self.blocks_per_slot].view(
                    self.slot_capacity,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
                views.append(values[: self.max_context].permute(1, 0, 2))
            histories.append(HeadHistory(hidden[d + 1], *views))
        return RequestHistory(slot, hidden[0], histories)

    def write_kv(self, head, key, value, slot_mapping):
        """One write call per head, proportional to fresh positions only."""
        if self.scatter_pa is not None:
            self.scatter_pa(
                key=key.contiguous(),
                value=value.contiguous(),
                key_cache=self.kv[head, 0],
                value_cache=self.kv[head, 1],
                slot_mapping=slot_mapping,
                cache_mode="Norm",
            )
            return
        indices = torch.cat((slot_mapping, slot_mapping + self.token_capacity)).long()
        updates = torch.cat((key, value)).reshape(2 * key.shape[0], -1)
        destination = self.kv[head].view(2 * self.token_capacity, -1)
        if self.scatter_nd is not None:
            self.scatter_nd(destination, indices[:, None], updates)
        else:
            destination.index_copy_(0, indices, updates)

    def attention_kv(self, head):
        return tuple(
            self.kv[head, kind].view(self.total_blocks, self.block_size, -1)
            for kind in range(2)
        )
