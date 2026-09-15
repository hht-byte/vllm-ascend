# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental token-bucket graphs for 310P attention.

The planner is deliberately independent of vLLM and torch_npu so it can be
tested on CPU. Hardware support for zero-query rows / descriptor updates is
NOT inferred from the successful context_lens-only experiment. Token layout
keeps operator qLens constant and expresses causality using per-token contexts.
"""

from dataclasses import dataclass, field
from typing import Any

import torch

CONFIG_KEY = "token_graph_310p"
COMPRESSED_MASK_SIZE = 2048
COMPRESSED_MASK_TYPE = 5
PAD_SLOT = -1


@dataclass(frozen=True)
class TokenGraphConfig:
    enabled: bool = False
    update_mode: str = "inplace"
    request_layout: str = "token"

    @classmethod
    def from_vllm(cls, config):
        values = (config.additional_config or {}).get(CONFIG_KEY, {})
        if not isinstance(values, dict):
            raise ValueError(f"{CONFIG_KEY} must be an object")
        unknown = set(values) - {"enabled", "update_mode", "request_layout"}
        if unknown:
            raise ValueError(f"Unknown {CONFIG_KEY} settings: {sorted(unknown)}")
        result = cls(**values)
        if type(result.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if result.update_mode not in ("inplace", "task_update"):
            raise ValueError("update_mode must be inplace or task_update")
        if result.request_layout not in ("token", "fixed", "active"):
            raise ValueError("request_layout must be token, fixed or active")
        if result.request_layout == "active" and result.update_mode != "task_update":
            raise ValueError("active request views require task_update")
        if result.enabled and result.request_layout == "fixed" and result.update_mode == "inplace":
            raise ValueError("fixed/inplace cannot replay changing host qLens; use request_layout=token")
        return result


@dataclass(frozen=True)
class TokenGraphPlan:
    bucket: int
    actual_tokens: int
    actual_reqs: int
    query_lens: tuple[int, ...]
    context_lens: tuple[int, ...]
    query_start_loc: tuple[int, ...]
    row_requests: tuple[int, ...] = ()

    @classmethod
    def create(cls, bucket, query_lens, context_lens, max_reqs, max_context, layout="fixed"):
        q = tuple(int(x) for x in query_lens)
        c = tuple(int(x) for x in context_lens)
        if layout not in ("token", "fixed", "active"):
            raise ValueError("Unknown request layout")
        if not q or len(q) != len(c) or len(q) > max_reqs:
            raise ValueError("Invalid real request count / context_lens length")
        if any(x <= 0 for x in q) or any(k < n or k > max_context for n, k in zip(q, c)):
            raise ValueError("Real requests require 0 < query_len <= context_len <= max_context")
        actual = sum(q)
        if not actual <= bucket <= max_context:
            raise ValueError("Require actual_tokens <= bucket <= max_context")
        reqs = len(q)
        if layout == "token":
            # The j-th query of request r may see exactly C_r - Q_r + j + 1
            # keys, even though all new K/V are written before attention runs.
            contexts = tuple(length for n, end in zip(q, c) for length in range(end - n + 1, end + 1))
            owners = tuple(row for row, n in enumerate(q) for _ in range(n))
            padding = bucket - actual
            return cls(
                bucket,
                actual,
                reqs,
                (1,) * bucket,
                contexts + (1,) * padding,
                tuple(range(bucket + 1)),
                owners + (-1,) * padding,
            )
        if bucket > actual:
            # Read-only dummy: repeated block 0 covers the virtual context.
            q += (bucket - actual,)
            c += (bucket - actual,)
        if layout == "fixed":
            capacity = min(bucket, max_reqs + 1)
            q += (0,) * (capacity - len(q))
            c += (1,) * (capacity - len(c))
        starts = [0]
        for length in q:
            starts.append(starts[-1] + length)
        return cls(bucket, actual, reqs, q, c, tuple(starts))


class TokenGraphArena:
    """One shared metadata allocation; bucket states retain views and tasks.

    Caller must drain the previous replay before rewriting pinned host buffers.
    Device copies and graph replay must be ordered on the same stream.
    """

    def __init__(self, max_tokens, max_reqs, max_blocks, max_context, device, config):
        self.max_tokens = max_tokens
        self.max_reqs = max_reqs
        self.max_blocks = max_blocks
        self.max_context = max_context
        self.config = config
        self.device = torch.device(device)
        self.capacity = max_tokens if config.request_layout == "token" else min(max_tokens, max_reqs + 1)
        pinned = self.device.type != "cpu"
        self.qlens = torch.ones(self.capacity, dtype=torch.int32, pin_memory=pinned)
        self.context_cpu = torch.ones(self.capacity, dtype=torch.int32, pin_memory=pinned)
        self.starts_cpu = torch.zeros(self.capacity + 1, dtype=torch.int32, pin_memory=pinned)
        self.context = torch.ones(self.capacity, dtype=torch.int32, device=device)
        self.starts = torch.zeros(self.capacity + 1, dtype=torch.int32, device=device)
        self.blocks = torch.zeros((self.capacity, max_blocks), dtype=torch.int32, device=device)
        self.slots = torch.full((max_tokens,), PAD_SLOT, dtype=torch.int32, device=device)
        self.states: dict[int, TokenGraphState] = {}

    def prepare(self, bucket, query_lens, context_lens, blocks, slots):
        query_lens = tuple(int(length) for length in query_lens)
        if bucket > self.max_tokens:
            raise ValueError("Bucket exceeds allocated arena")
        plan = TokenGraphPlan.create(
            bucket, query_lens, context_lens, self.max_reqs, self.max_context, self.config.request_layout
        )
        rows = len(plan.query_lens)
        if blocks.ndim != 2 or blocks.shape[0] < plan.actual_reqs or blocks.shape[1] != self.max_blocks:
            raise ValueError("Block table does not cover real requests or has changed width")
        if slots.ndim != 1 or slots.numel() < plan.actual_tokens:
            raise ValueError("Slot mapping does not cover real tokens")
        if self.config.request_layout != "token":
            self.qlens[:rows].copy_(torch.tensor(plan.query_lens, dtype=torch.int32))
        self.context_cpu[:rows].copy_(torch.tensor(plan.context_lens, dtype=torch.int32))
        self.starts_cpu[: rows + 1].copy_(torch.tensor(plan.query_start_loc, dtype=torch.int32))
        self.context[:rows].copy_(self.context_cpu[:rows], non_blocking=True)
        self.starts[: rows + 1].copy_(self.starts_cpu[: rows + 1], non_blocking=True)
        self.blocks[:rows].zero_()
        if self.config.request_layout == "token":
            offset = 0
            for row, length in enumerate(query_lens):
                # Broadcast the real request's table to each of its query rows.
                self.blocks[offset : offset + length].copy_(blocks[row], non_blocking=True)
                offset += length
        else:
            self.blocks[: plan.actual_reqs].copy_(blocks[: plan.actual_reqs], non_blocking=True)
        self.slots[:bucket].fill_(PAD_SLOT)
        self.slots[: plan.actual_tokens].copy_(slots[: plan.actual_tokens], non_blocking=True)
        if bucket not in self.states:
            self.states[bucket] = TokenGraphState(self, plan)
        state = self.states[bucket]
        state.plan = plan
        return state


@dataclass
class SplitfuseTask:
    handle: Any
    kwargs: dict[str, Any]


@dataclass
class TokenGraphState:
    arena: TokenGraphArena
    plan: TokenGraphPlan
    tasks: dict[str, SplitfuseTask] = field(default_factory=dict)

    def metadata_kwargs(self):
        rows = len(self.plan.query_lens)
        return dict(
            block_table=self.arena.blocks[:rows],
            context_lens=self.arena.context[:rows],
            seq_len=self.arena.qlens[:rows],
        )

    def attach(self, metadata):
        rows = len(self.plan.query_lens)
        metadata.token_graph_state = self
        metadata.num_actual_tokens = self.plan.bucket  # Graph-local write extent, not scheduler count.
        metadata.slot_mapping = self.arena.slots[: self.plan.bucket]
        metadata.block_tables = self.arena.blocks[:rows]
        metadata.seq_lens = self.arena.context[:rows]
        metadata.query_start_loc = self.arena.starts[: rows + 1]

    def attention(self, backend, layer_name, query, key_cache, value_cache, mask, output, heads, kv_heads, scale):
        kwargs = dict(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            mask=mask,
            out=output,
            num_heads=heads,
            num_kv_heads=kv_heads,
            scale_value=scale,
            mask_type=COMPRESSED_MASK_TYPE,
            **self.metadata_kwargs(),
        )
        capturing = backend.npu.is_current_stream_capturing()
        handle = None
        if capturing and self.arena.config.update_mode == "task_update":
            stream = backend.npu.current_stream()
            backend.npu.graph_task_group_begin(stream)
            try:
                backend._npu_paged_attention_splitfuse_v2(**kwargs)
            finally:
                handle = backend.npu.graph_task_group_end(stream)
        else:
            backend._npu_paged_attention_splitfuse_v2(**kwargs)
        if capturing:
            if layer_name in self.tasks:
                raise RuntimeError("Repeated attention layer in token graph capture is unsupported")
            # Strong references retain graph inputs/output and all metadata views.
            self.tasks[layer_name] = SplitfuseTask(handle, kwargs)
        return output

    def update(self, backend):
        if self.arena.config.update_mode == "inplace":
            return
        stream = backend.npu.current_stream()
        for task in self.tasks.values():
            kwargs = dict(task.kwargs, **self.metadata_kwargs())
            backend.npu.graph_task_update_begin(stream, task.handle)
            try:
                backend._npu_paged_attention_splitfuse_v2(**kwargs)
            finally:
                backend.npu.graph_task_update_end(stream)
            task.kwargs = kwargs


def validate_token_graph_config(config):
    """Fail explicitly for combinations not implemented by this first version."""
    if config.compilation_config.cudagraph_mode.name != "FULL":
        raise ValueError("token_graph_310p requires cudagraph_mode=FULL")
    if config.scheduler_config.async_scheduling:
        raise ValueError("token_graph_310p requires --no-async-scheduling")
    parallel = config.parallel_config
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"token_graph_310p does not yet support {name}>1")
    if config.lora_config is not None or config.kv_transfer_config is not None:
        raise ValueError("token_graph_310p does not yet support LoRA or KV transfer")
    spec = config.speculative_config
    if spec is not None and spec.method != "ngram":
        raise ValueError("token_graph_310p initially supports only ngram speculative decoding")
    model = config.model_config
    if model.use_mla or model.runner_type != "generate" or model.is_encoder_decoder:
        raise ValueError("token_graph_310p requires a dense causal decoder")
    if config.cache_config.cache_dtype not in ("auto", "float16", "bfloat16"):
        raise ValueError("token_graph_310p does not support quantized KV cache")
    sizes = config.compilation_config.cudagraph_capture_sizes
    if not sizes or max(sizes) > min(COMPRESSED_MASK_SIZE, model.max_model_len):
        raise ValueError("token_graph_310p capture sizes must fit the compressed mask and model context")
