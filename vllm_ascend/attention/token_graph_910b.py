# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in PA token graphs: CPU lengths are refreshed through task update."""

import logging
from dataclasses import dataclass, field
from typing import Any

from vllm_ascend._310p.token_graph import TokenGraphArena, TokenGraphConfig


def enabled(config):
    values = (config.additional_config or {}).get("token_graph_910b", {})
    if not isinstance(values, dict) or set(values) - {"enabled"}:
        raise ValueError("token_graph_910b accepts only the boolean enabled setting")
    value = values.get("enabled", False)
    if type(value) is not bool:
        raise ValueError("token_graph_910b.enabled must be boolean")
    return value


class PAGraphArena(TokenGraphArena):
    def __init__(self, *args):
        super().__init__(*args, config=TokenGraphConfig(request_layout="token", update_mode="task_update"))

    def prepare(self, *args):
        state = super().prepare(*args)
        if not isinstance(state, PAGraphState):
            state = PAGraphState(self, state.plan)
            self.states[state.plan.bucket] = state
        return state


@dataclass
class PATask:
    handle: Any
    event: Any
    kwargs: dict
    workspace: Any
    capture_workspace: Any


def workspace_key(kwargs):
    """Sharing excludes addresses but includes all tensor layouts and PA attributes.

    Within one state, every layer uses the same block table and context lengths.
    The runner serializes updates and replays; overlapping graphs are unsupported.
    """
    tensors = tuple((tuple(kwargs[name].shape), tuple(kwargs[name].stride()),
                     kwargs[name].dtype, kwargs[name].device, kwargs[name].storage_offset())
                    for name in ("query", "key_cache", "value_cache", "out", "block_table", "context_lens"))
    return tensors, kwargs["num_heads"], kwargs["num_kv_heads"], kwargs["scale_value"]


@dataclass
class PAGraphState:
    arena: PAGraphArena
    plan: Any
    tasks: dict[str, PATask] = field(default_factory=dict)
    updates: int = 0
    capture_workspaces: dict = field(default_factory=dict)

    def attach(self, metadata):
        # Do not replace scheduler/common metadata used by sampling or rollback.
        metadata.pa_token_graph_state = self
        metadata.num_actual_tokens = self.plan.bucket
        metadata.slot_mapping = self.arena.slots[:self.plan.bucket]

    def attention(self, backend, layer_name, query, key_cache, value_cache, output, heads, kv_heads, scale):
        bucket = self.plan.bucket
        kwargs = dict(query=query.view(bucket, heads, -1), key_cache=key_cache, value_cache=value_cache,
                      out=output.view(bucket, heads, -1), num_heads=heads, num_kv_heads=kv_heads,
                      scale_value=scale, block_table=self.arena.blocks[:bucket],
                      context_lens=self.arena.context_cpu[:bucket])
        if not backend.npu.is_current_stream_capturing():
            backend._npu_paged_attention(**kwargs)
            return output
        if layer_name in self.tasks:
            raise RuntimeError(f"Repeated PA capture for layer {layer_name}")
        key = workspace_key(kwargs)
        if key not in self.capture_workspaces:
            self.capture_workspaces[key] = backend._npu_paged_attention_get_workspace(**kwargs)
            logging.getLogger(__name__).info(
                "910B PA capture workspace: bucket=%s geometry=%s bytes=%s first_layer=%s",
                bucket, len(self.capture_workspaces), getattr(self.capture_workspaces[key], "nbytes", None), layer_name,
            )
        workspace = self.capture_workspaces[key]
        stream = backend.npu.current_stream()
        event = backend.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        backend.npu.graph_task_group_begin(stream)
        backend._npu_paged_attention(**kwargs, workspace=workspace)
        handle = backend.npu.graph_task_group_end(stream)
        self.tasks[layer_name] = PATask(handle, event, kwargs, workspace, workspace)
        return output

    def update(self, backend, stream):
        # Caller drains previous replay and input copies. Strong refs retain all
        # captured activations, caches, host buffers and workspace allocations.
        # get_workspace allocates a tensor, rather than merely querying bytes.
        # Reuse it across matching, serial layer tasks. A fresh per-update map
        # still accounts for changing lengths/tiling; do not assume a size bound.
        workspaces = {}
        with backend.npu.stream(stream):
            for task in self.tasks.values():
                key = workspace_key(task.kwargs)
                if key not in workspaces:
                    workspaces[key] = backend._npu_paged_attention_get_workspace(**task.kwargs)
                workspace = workspaces[key]
                backend.npu.graph_task_update_begin(stream, task.handle)
                backend._npu_paged_attention(**task.kwargs, workspace=workspace)
                backend.npu.graph_task_update_end(stream)
                task.event.record(stream)
                stream.synchronize()
                task.workspace = workspace
        self.updates += 1
