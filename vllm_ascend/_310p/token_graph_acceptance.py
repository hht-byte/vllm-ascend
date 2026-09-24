# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker extension used only by the offline token-graph acceptance script."""


def worker_audit(worker, action="snapshot", phase=""):
    """Called by a named worker extension RPC, without sending Python callables."""
    import inspect

    import torch
    from vllm.forward_context import get_forward_context

    from vllm_ascend.compilation import acl_graph

    runner = worker.model_runner
    if action == "install":
        native_full = phase == "native_full"
        if hasattr(worker, "_token_acceptance"):
            raise RuntimeError("Acceptance audit already installed")
        worker._token_acceptance = {"events": [], "phase": "warmup"}
        determine = runner._determine_batch_execution_and_padding
        signature = inspect.signature(determine)

        def traced_dispatch(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            result = determine(*args, **kwargs)
            values = bound.arguments
            scheduled = values["num_scheduled_tokens_np"][: values["num_reqs"]].tolist()
            worker._token_acceptance["events"].append(
                dict(
                    event="dispatch",
                    mode=result[0].name,
                    bucket=result[1].num_tokens,
                    family=getattr(result[1], "attention_family", "token"),
                    actual_reqs=values["num_reqs"],
                    scheduled=scheduled,
                    phase=worker._token_acceptance["phase"],
                )
            )
            return result

        runner._determine_batch_execution_and_padding = traced_dispatch
        original = acl_graph.ACLGraphWrapper.__call__

        def traced(wrapper, *args, **kwargs):
            context = get_forward_context()
            desc = context.batch_descriptor
            full = context.cudagraph_runtime_mode.name == "FULL" and wrapper.runtime_mode.name == "FULL"
            event = None
            if full and native_full:
                entry = wrapper.concrete_aclgraph_entries.get(desc)
                dispatch = next(e for e in reversed(worker._token_acceptance["events"])
                                if e["event"] == "dispatch")
                kind = "replay" if entry is not None and entry.aclgraph is not None else "capture"
                event = dict(dispatch, event=kind,
                             family="native_full", actual_tokens=sum(dispatch["scheduled"]))
            elif full:
                entry = wrapper.concrete_aclgraph_entries.get(desc)
                family = getattr(desc, "attention_family", "token")
                arena = (getattr(runner, "_native_graph_arenas", {}).get(family) if family != "token"
                         else getattr(runner, "_token_graph_arena", None))
                state = arena.states.get(desc.num_tokens) if arena is not None else None
                if state is None:
                    raise RuntimeError(f"FULL replay without {family} arena state")
                if family != "token":
                    counts = list(state.scheduled_query_lens)
                else:
                    counts = [0] * state.plan.actual_reqs
                    for row in state.plan.row_requests[: state.plan.actual_tokens]:
                        counts[row] += 1
                event = dict(
                    event="replay" if entry is not None and entry.aclgraph is not None else "capture",
                    bucket=desc.num_tokens,
                    family=family,
                    operator=("_npu_paged_attention" if getattr(runner, "token_graph_910b_enabled", False)
                              else {"prefill": "_npu_flash_attention", "decode": "_npu_paged_attention",
                                    "token": "_npu_paged_attention_splitfuse_v2"}[family]),
                    actual_tokens=state.plan.actual_tokens,
                    actual_reqs=state.plan.actual_reqs,
                    scheduled=counts,
                    task_updates=getattr(state, "updates", 0),
                    decode_only=getattr(runner.attn_state, "name", "") == "DecodeOnly",
                    phase=worker._token_acceptance["phase"],
                )
            output = original(wrapper, *args, **kwargs)
            if event is not None:
                worker._token_acceptance["events"].append(event)
            return output

        acl_graph.ACLGraphWrapper.__call__ = traced
    elif action == "reset":
        worker._token_acceptance["events"] = []
        worker._token_acceptance["phase"] = phase
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
    elif action != "snapshot":
        raise ValueError(action)
    torch.npu.synchronize()
    entries = []
    for wrapper in list(acl_graph._acl_graph_wrappers):
        if wrapper.runtime_mode.name != "FULL":
            continue
        for desc, entry in wrapper.concrete_aclgraph_entries.items():
            if entry.aclgraph is not None:
                entries.append(
                    dict(
                        bucket=desc.num_tokens,
                        family=getattr(desc, "attention_family", "token"),
                        num_reqs=desc.num_reqs,
                        uniform=desc.uniform,
                        graph_id=id(entry.aclgraph),
                    )
                )
    entries.sort(key=lambda item: (item["bucket"], item["graph_id"]))
    return dict(
        graphs=entries,
        audit=list(worker._token_acceptance["events"]),
        memory=dict(
            allocated=torch.npu.memory_allocated(),
            reserved=torch.npu.memory_reserved(),
            peak_allocated=torch.npu.max_memory_allocated(),
            peak_reserved=torch.npu.max_memory_reserved(),
        ),
    )


class TokenGraphAcceptanceWorker:
    def token_graph_acceptance(self, action="snapshot", phase=""):
        return worker_audit(self, action, phase)
