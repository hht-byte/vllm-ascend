# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in PA token graphs: CPU lengths are refreshed through task update."""

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


@dataclass
class PAGraphState:
    arena: PAGraphArena
    plan: Any
    tasks: dict[str, PATask] = field(default_factory=dict)
    updates: int = 0

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
        workspace = backend._npu_paged_attention_get_workspace(**kwargs)
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
        with backend.npu.stream(stream):
            for task in self.tasks.values():
                workspace = backend._npu_paged_attention_get_workspace(**task.kwargs)
                backend.npu.graph_task_update_begin(stream, task.handle)
                backend._npu_paged_attention(**task.kwargs, workspace=workspace)
                backend.npu.graph_task_update_end(stream)
                task.event.record(stream)
                stream.synchronize()
                task.workspace = workspace
        self.updates += 1
