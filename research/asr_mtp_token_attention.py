# SPDX-License-Identifier: Apache-2.0
"""Packed auxiliary attention with isolated request segments and FIA task updates.

The NPU path follows the v0.23 Ascend FULL attention contract: a fixed-shape
TND operation, a startup maximum workspace, and CPU length-list task updates.
CPU execution is a correctness reference, not an online per-request fallback.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

CAUSAL_MASK_SIZE = 2048
UNLIMITED_PRE_TOKENS = 2147483647


@dataclass
class _Task:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    output: torch.Tensor
    softmax_lse: torch.Tensor
    workspace: torch.Tensor
    scale: float
    handle: object = None
    event: object = None


class PackedAttention:
    """One packed attention call per auxiliary head, with fixed tensor capacities.

    ``stage`` receives cumulative CPU lengths. Each query segment is the tail
    of its matching KV segment. A terminal positive-query dummy segment covers
    all remaining physical KV rows. Non-PA TND requires exact Q/KV coverage;
    no device tensor is inspected to stage boundaries or update graph tasks.
    """

    def __init__(
        self,
        num_heads,
        num_kv_heads,
        head_dim,
        *,
        device,
        dtype,
        heads,
        npu_api=None,
        npu_runtime=None,
    ):
        if (
            min(num_heads, num_kv_heads, head_dim, heads) < 1
            or num_heads % num_kv_heads
        ):
            raise ValueError("invalid packed attention geometry")
        self.num_heads, self.num_kv_heads, self.head_dim = (
            num_heads,
            num_kv_heads,
            head_dim,
        )
        self.heads, self.device, self.dtype = heads, torch.device(device), dtype
        self.lengths, self.tasks, self.workspaces = {}, {}, {}
        self.recording = False
        self.is_npu = self.device.type == "npu" or npu_api is not None
        self.api, self.runtime, self.mask = None, None, None
        if self.is_npu:
            if npu_api is None:
                import torch_npu

                npu_api = torch_npu
            self.api = npu_api
            self.runtime = npu_runtime if npu_runtime is not None else torch.npu
            # FIA TND (without separate RoPE) on 910B supports D=128/192.
            # The old rectangular SDPA backend has a different geometry range.
            if head_dim not in (128, 192):
                raise ValueError("910B packed FIA requires head_dim 128 or 192")
            if self.device.type == "npu" and dtype not in (
                torch.float16,
                torch.bfloat16,
            ):
                raise ValueError("910B packed FIA requires float16 or bfloat16")
            operation = getattr(self.api, "npu_fused_infer_attention_score", None)
            if (
                operation is None
                or not callable(getattr(operation, "out", None))
                or not callable(
                    getattr(
                        self.api,
                        "_npu_fused_infer_attention_score_get_max_workspace",
                        None,
                    )
                )
                or any(
                    not callable(getattr(self.runtime, name, None))
                    for name in (
                        "current_stream",
                        "ExternalEvent",
                        "graph_task_group_begin",
                        "graph_task_group_end",
                        "graph_task_update_begin",
                        "graph_task_update_end",
                    )
                )
            ):
                raise ValueError(
                    "packed MTP requires FIA out/workspace and graph task APIs"
                )
            self.mask = torch.ones(
                (CAUSAL_MASK_SIZE, CAUSAL_MASK_SIZE),
                dtype=torch.bool,
                device=self.device,
            ).triu_(1)
            # A caller may supply "npu" without an index. Tensor allocations
            # resolve it to e.g. npu:0; compare against that canonical device.
            self.device = self.mask.device

    def stage(self, head, q_ends, kv_ends):
        if type(head) is not int or not 0 <= head < self.heads:
            raise ValueError("invalid packed attention head")
        q_ends, kv_ends = tuple(q_ends), tuple(kv_ends)
        if not q_ends or len(q_ends) != len(kv_ends) or len(q_ends) > 4096:
            raise ValueError("packed Q/KV segments must be aligned and nonempty")
        previous_q = previous_kv = 0
        for q_end, kv_end in zip(q_ends, kv_ends):
            if (
                type(q_end) is not int
                or type(kv_end) is not int
                or q_end <= previous_q
                or kv_end <= previous_kv
                or q_end - previous_q > kv_end - previous_kv
            ):
                raise ValueError(
                    "packed segments require positive Q lengths no larger than KV"
                )
            previous_q, previous_kv = q_end, kv_end
        self.lengths[head] = q_ends, kv_ends

    def _validate(self, head, query, key, value):
        if head not in self.lengths:
            raise ValueError("packed attention lengths were not staged")
        q_ends, kv_ends = self.lengths[head]
        if (
            query.ndim != 3
            or query.shape[1:] != (self.num_heads, self.head_dim)
            or key.ndim != 3
            or key.shape[1:] != (self.num_kv_heads, self.head_dim)
            or value.shape != key.shape
            or q_ends[-1] != query.shape[0]
            or kv_ends[-1] != key.shape[0]
            or query.device != self.device
            or key.device != self.device
            or value.device != self.device
            or query.dtype != self.dtype
            or key.dtype != self.dtype
            or value.dtype != self.dtype
            or (
                self.is_npu
                and not all(tensor.is_contiguous() for tensor in (query, key, value))
            )
        ):
            raise ValueError(
                "packed attention tensors do not cover the staged segments"
            )

    @torch.inference_mode()
    def __call__(self, head, query, key, value, scale):
        self._validate(head, query, key, value)
        if not self.is_npu:
            result = torch.empty_like(query)
            previous_q = previous_kv = 0
            groups = self.num_heads // self.num_kv_heads
            for q_end, kv_end in zip(*self.lengths[head]):
                q = query[previous_q:q_end].transpose(0, 1)
                k = key[previous_kv:kv_end].transpose(0, 1)
                v = value[previous_kv:kv_end].transpose(0, 1)
                q_length, kv_length = q_end - previous_q, kv_end - previous_kv
                allowed = torch.arange(kv_length, device=self.device)[None, :] <= (
                    torch.arange(q_length, device=self.device)[:, None]
                    + kv_length
                    - q_length
                )
                result[previous_q:q_end].copy_(
                    F.scaled_dot_product_attention(
                        q,
                        k.repeat_interleave(groups, 0),
                        v.repeat_interleave(groups, 0),
                        attn_mask=allowed[None],
                        dropout_p=0.0,
                        scale=scale,
                    ).transpose(0, 1)
                )
                previous_q, previous_kv = q_end, kv_end
            return result

        task = self.tasks.get(head)
        if task is None:
            if self.recording:
                raise RuntimeError("packed attention must be warmed before capture")
            geometry = query.shape, key.shape, value.shape, float(scale)
            workspace = self.workspaces.get(geometry)
            if workspace is None:
                workspace = self.api._npu_fused_infer_attention_score_get_max_workspace(
                    **self._arguments(head, query, key, value, scale)
                )
                self.workspaces[geometry] = workspace
            task = _Task(
                query,
                key,
                value,
                torch.empty_like(query),
                torch.empty(1, device=query.device, dtype=query.dtype),
                workspace,
                float(scale),
            )
            self.tasks[head] = task
        elif (
            task.query.shape != query.shape
            or task.key.shape != key.shape
            or task.value.shape != value.shape
            or task.scale != scale
        ):
            raise ValueError("packed attention capture geometry changed")
        task.query, task.key, task.value = query, key, value
        if self.recording:
            if task.handle is not None:
                raise RuntimeError("packed attention head was captured twice")
            stream = self.runtime.current_stream()
            task.event = self.runtime.ExternalEvent()
            task.event.wait(stream)
            task.event.reset(stream)
            self.runtime.graph_task_group_begin(stream)
            self._execute(head, task)
            task.handle = self.runtime.graph_task_group_end(stream)
        else:
            self._execute(head, task)
        return task.output

    def _arguments(self, head, query, key, value, scale):
        q_ends, kv_ends = self.lengths[head]
        return {
            "query": query,
            "key": key,
            "value": value,
            "atten_mask": self.mask,
            "input_layout": "TND",
            "actual_seq_lengths": q_ends,
            "actual_seq_lengths_kv": kv_ends,
            "num_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "scale": scale,
            "sparse_mode": 3,
            "pre_tokens": UNLIMITED_PRE_TOKENS,
            "next_tokens": 0,
        }

    def _execute(self, head, task):
        self.api.npu_fused_infer_attention_score.out(
            **self._arguments(head, task.query, task.key, task.value, task.scale),
            workspace=task.workspace,
            out=[task.output, task.softmax_lse],
        )

    def begin_capture(self):
        if self.recording or any(
            task.handle is not None for task in self.tasks.values()
        ):
            raise RuntimeError("packed attention capture is already active or complete")
        self.recording = True

    def end_capture(self):
        self.recording = False

    @torch.inference_mode()
    def update(self):
        if not self.is_npu:
            return
        if self.recording or set(self.tasks) != set(range(self.heads)):
            raise RuntimeError("packed attention graph capture is incomplete")
        # Preflight every head before enqueueing any task update. Workspaces
        # cover the fixed Q/KV capacities and are never allocated during replay.
        for head, task in self.tasks.items():
            if task.handle is None:
                raise RuntimeError("packed attention graph capture is incomplete")
            self._validate(head, task.query, task.key, task.value)
        stream = self.runtime.current_stream()
        for head, task in self.tasks.items():
            self.runtime.graph_task_update_begin(stream, task.handle)
            self._execute(head, task)
            self.runtime.graph_task_update_end(stream)
            task.event.record(stream)


class PackedNPUCapture:
    """Serial auxiliary graph capture with per-head FIA length task updates."""

    def __init__(self):
        self.pool = torch.npu.graph_pool_handle()

    def capture(self, program):
        attention = program.attention
        if not attention.is_npu:
            raise RuntimeError("packed NPU capture requires an NPU attention backend")
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(2):
                program.run()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        attention.begin_capture()
        try:
            with torch.npu.graph(graph, pool=self.pool):
                program.run()
        finally:
            attention.end_capture()
        torch.npu.synchronize()

        def replay():
            attention.update()
            graph.replay()

        return replay
