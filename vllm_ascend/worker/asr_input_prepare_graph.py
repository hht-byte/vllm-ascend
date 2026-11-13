# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in NPU graph for synchronous ASR input preparation, separate from forward."""

from bisect import bisect_left
from collections.abc import Mapping
from time import monotonic

import torch


def settings(runner, *, prepare_inputs):
    config = runner.vllm_config.additional_config or {}
    opts = config.get("asr_input_prepare_graph", {})
    if not isinstance(opts, Mapping):
        raise TypeError("asr_input_prepare_graph must be a mapping")
    if set(opts) - {"enabled", "token_sizes", "pack_spec_metadata"}:
        raise ValueError("unknown asr_input_prepare_graph setting")
    on = opts.get("enabled", False)
    if type(on) is not bool:
        raise TypeError("asr_input_prepare_graph.enabled must be a bool")
    if not on:
        return None
    if type(opts.get("pack_spec_metadata", True)) is not bool:
        raise TypeError("asr_input_prepare_graph.pack_spec_metadata must be a bool")
    if config.get("asr_sync_input_packing", False):
        raise ValueError("disable asr_sync_input_packing before enabling asr_input_prepare_graph")
    if not isinstance(config.get("asr_mtp"), Mapping):
        raise ValueError("asr_input_prepare_graph requires additional_config.asr_mtp")
    # vLLM allocates num_accepted_tokens_event for every speculative runner,
    # including synchronous dense models. Allocation is not an async/hybrid flag.
    unsupported = [
        f"{name}={value!r}"
        for name, value in {
            "use_async_scheduling": runner.use_async_scheduling,
            "use_async_spec_decode": runner.use_async_spec_decode,
            "model_config.is_hybrid": getattr(runner.model_config, "is_hybrid", False),
            "enforce_eager": getattr(runner.model_config, "enforce_eager", False),
            **{name: getattr(runner, name, False) for name in ("_has_gdn", "use_sparse", "use_compress", "is_hybrid")},
        }.items()
        if value
    ]
    unsupported.extend(
        f"{name}={value!r}"
        for name, value in {
            "pcp_size": runner.pcp_size,
            "dcp_size": runner.dcp_size,
            **{
                name: getattr(runner.parallel_config, name)
                for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")
            },
        }.items()
        if value != 1
    )
    mode = runner.compilation_config.cudagraph_mode.name
    if mode != "FULL":
        unsupported.append(f"cudagraph_mode={mode!r}")
    if unsupported:
        raise ValueError(
            "asr_input_prepare_graph requires synchronous single-device dense ASR with FULL graphs; "
            "unsupported settings: " + ", ".join(unsupported)
        )
    if runner._prepare_inputs.__func__ is not prepare_inputs:
        raise ValueError("asr_input_prepare_graph does not support overridden _prepare_inputs")
    capacity = min(runner.max_num_tokens, runner.positions.numel())
    default = sorted({min(n, capacity) for n in (16, 64, 256, 1024)})
    sizes = opts.get("token_sizes", default)
    if (
        not isinstance(sizes, (list, tuple))
        or not sizes
        or any(type(n) is not int or not 0 < n <= capacity for n in sizes)
        or len(set(sizes)) != len(sizes)
    ):
        raise ValueError("token_sizes must be unique positive integers within the runner token capacity")
    return tuple(sorted(sizes))


class NPUCapture:
    def __init__(self):
        self.pool = torch.npu.graph_pool_handle()
        self.event = torch.npu.Event()

    def reserved(self):
        return torch.npu.memory_reserved()

    def capture(self, fn):
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(2):
                fn()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, pool=self.pool):
            fn()
        torch.npu.synchronize()
        return graph


class InputPrepareGraph:
    """Fixed metadata addresses and fixed request grid; token buckets only.

    CPU lengths and positions are already authoritative in the synchronous
    runner. Upload them once, then replay all device copies/fills/slot mapping.
    Graphs share one pool and output storage and must execute serially on the
    runner stream. No runtime capture, host length readback or device-wide sync.
    """

    def __init__(self, runner, token_sizes, *, backend=None):
        self.runner, self.token_sizes = runner, token_sizes
        self.backend = backend if backend is not None else NPUCapture()
        self.entries = {}
        self.tables = None
        self.ready = False
        self.replays = self.misses = self.reserved_bytes = 0
        self.pending_copy = False
        self.last_stats_time = None
        self.max_requests = runner.seq_lens.numel()
        self.outputs = {
            "counts": runner.num_computed_tokens,
            "accepted": runner.num_accepted_tokens.gpu,
            "seq": runner.seq_lens,
            "qsl": runner.query_start_loc.gpu,
            "discard_indices": runner.discard_request_indices.gpu,
            "discard_mask": runner.discard_request_mask.gpu,
            "positions": runner.positions,
        }
        expected = {
            "counts": torch.int32,
            "accepted": torch.int32,
            "seq": torch.int32,
            "qsl": torch.int32,
            "discard_indices": torch.int64,
            "discard_mask": torch.bool,
            "positions": torch.int64,
        }
        for name, out in self.outputs.items():
            sizes = (self.max_requests + 1, self.max_requests + 2) if name == "qsl" else (self.max_requests,)
            if (
                out.ndim != 1
                or not out.is_contiguous()
                or out.dtype != expected[name]
                or (name != "positions" and out.numel() not in sizes)
            ):
                raise ValueError(f"unsupported input preparation layout: {name}")
        # Every field is independently aligned. Positions last: only the selected
        # bucket prefix is transferred. Separate private qsl avoids FIA rewrites.
        layout, offset = {}, 0
        for name in ("counts", "accepted", "seq", "qsl", "discard_indices", "discard_mask", "slot_qsl", "positions"):
            out = self.outputs["qsl" if name == "slot_qsl" else name]
            offset = (offset + 7) // 8 * 8
            size = out.numel() * out.element_size()
            layout[name] = (offset, size, out.dtype)
            offset += size
        self.host = torch.zeros(offset, dtype=torch.uint8, pin_memory=runner.pin_memory)
        self.device = torch.zeros(offset, dtype=torch.uint8, device=runner.device)
        self.inputs, self.cpu = {}, {}
        for name, (start, size, dtype) in layout.items():
            self.inputs[name] = self.device[start : start + size].view(dtype)
            self.cpu[name] = self.host[start : start + size].view(dtype).numpy()
        self.position_offset = layout["positions"][0]

    def select(self, total_tokens):
        if not self.ready:
            return None
        if self.runner.input_batch.block_table is not self.tables:
            raise RuntimeError("input preparation graph block table was replaced after capture")
        index = bisect_left(self.token_sizes, total_tokens)
        if total_tokens <= 0 or index == len(self.token_sizes):
            self.misses += 1
            return None
        return self.token_sizes[index]

    def should_log(self, *, now=None):
        if not self.ready or self.replays + self.misses == 0:
            return False
        now = monotonic() if now is None else now
        if self.last_stats_time is not None and now - self.last_stats_time < 5.0:
            return False
        self.last_stats_time = now
        return True

    def _run(self, tokens):
        for name, out in self.outputs.items():
            if name == "positions":
                out[:tokens].copy_(self.inputs[name][:tokens])
            else:
                out.copy_(self.inputs[name])
        for table in self.tables.block_tables:
            # The slot kernel clears tokens:capacity itself. It does not clear
            # actual_tokens:tokens for a fixed bucket: explicitly clear that too.
            table.slot_mapping.gpu[:tokens].fill_(-1)
        self.tables.compute_slot_mapping(self.max_requests, self.inputs["slot_qsl"], self.outputs["positions"][:tokens])

    @torch.inference_mode()
    def capture(self):
        if self.ready:
            return 0
        self.tables = self.runner.input_batch.block_table
        if not self.tables.block_tables or any(
            table.is_mamba_group
            or table.block_table.gpu.shape[0] < self.max_requests
            or table.slot_mapping.gpu.numel() < self.token_sizes[-1]
            or table.max_num_batched_tokens < self.token_sizes[-1]
            for table in self.tables.block_tables
        ):
            raise ValueError("unsupported input preparation block table layout")
        before = self.backend.reserved()
        # Zero private qsl makes every request inactive during capture. Runtime
        # lengths are device inputs to the fixed-grid existing slot kernel.
        self.device.zero_()
        for tokens in self.token_sizes:
            self.entries[tokens] = self.backend.capture(lambda tokens=tokens: self._run(tokens))
        self.reserved_bytes = max(0, self.backend.reserved() - before)
        self.ready = True
        return self.reserved_bytes

    @torch.inference_mode()
    def replay(self, bucket, counts, seq_lens, positions, qsl, discard_mask, discard_indices):
        n, t = len(counts), len(positions)
        if (
            not self.ready
            or bucket not in self.entries
            or not 0 < n <= self.max_requests
            or not 0 < t <= bucket
            or len(seq_lens) != n
            or len(qsl) != n + 1
            or qsl[0] != 0
            or qsl[-1] != t
            or len(discard_mask) != n
            or len(discard_indices) > n
        ):
            raise ValueError("input preparation metadata exceeds captured capacity or has inconsistent lengths")
        event = self.backend.event
        if self.pending_copy and not event.query():
            event.synchronize()  # Protect pinned host reuse, only the preceding copy.
        for name, values in (
            ("counts", counts),
            ("seq", seq_lens),
            ("discard_indices", discard_indices),
            ("discard_mask", discard_mask),
        ):
            self.cpu[name].fill(0)
            self.cpu[name][: len(values)] = values
        # The runner retains its original event wait / CPU synchronization.
        # Only the device transfer moves into this graph; do not replace accepted
        # counts with ones just because the supported model is dense.
        self.cpu["accepted"].fill(1)
        self.cpu["accepted"][:n] = self.runner.num_accepted_tokens.np[:n]
        self.cpu["qsl"].fill(-1)
        self.cpu["qsl"][: n + 1] = qsl
        self.cpu["slot_qsl"].fill(t)
        self.cpu["slot_qsl"][: n + 1] = qsl
        self.cpu["positions"][:t] = positions
        self.cpu["positions"][t:bucket] = 0
        size = self.position_offset + bucket * 8
        self.device[:size].copy_(self.host[:size], non_blocking=True)
        event.record()
        self.pending_copy = True
        self.entries[bucket].replay()
        self.replays += 1


def install(runner, *, prepare_inputs):
    sizes = settings(runner, prepare_inputs=prepare_inputs)
    return None if sizes is None else InputPrepareGraph(runner, sizes)
