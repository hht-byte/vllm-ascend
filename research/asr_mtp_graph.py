# SPDX-License-Identifier: Apache-2.0
"""Bounded, independent auxiliary graphs; target forward/verification is untouched."""

import logging
from collections import Counter
from dataclasses import dataclass
from math import gcd
from time import perf_counter

import torch
from torch.nn import functional as F

try:
    from .asr_mtp_heads import rotate_half
except ImportError:
    from asr_mtp_heads import rotate_half

logger = logging.getLogger(__name__)
MAX_MTP_GRAPHS = 64


def validate_graph_runner(runner):
    if getattr(runner.model_config, "enforce_eager", False):
        raise ValueError("asr_mtp.graph_mode=full conflicts with enforce_eager=True")
    if any(
        cls.__module__.startswith("vllm_ascend._310p") for cls in type(runner).__mro__
    ):
        raise ValueError(
            "ASR MTP full graphs require the 910B worker memory path; "
            "310P auxiliary graph memory accounting is not supported"
        )


@dataclass(frozen=True)
class GraphOptions:
    max_batch: int
    queries: tuple[int, ...]
    context: int

    def scratch_elements(self, config):
        return (
            self.max_batch
            * max(self.queries)
            * max(config.hidden_size, config.num_key_value_heads * config.head_dim)
        )


def graph_options(settings, *, context, slots, max_batch):
    mode = settings.get("graph_mode", "off")
    if mode not in ("off", "full"):
        raise ValueError("asr_mtp.graph_mode must be off or full")
    if mode == "off":
        return None
    capacity = settings.get("graph_context_tokens", min(128, context))
    batch = settings.get("graph_max_batch_size", 8)
    if type(capacity) is not int or not 0 < capacity <= context:
        raise ValueError(
            "asr_mtp.graph_context_tokens exceeds the auxiliary context capacity"
        )
    queries = settings.get("graph_query_sizes", (min(8, capacity), min(64, capacity)))
    if type(batch) is not int or batch < 1:
        raise ValueError("asr_mtp.graph_max_batch_size must be a positive integer")
    if (
        not isinstance(queries, (list, tuple))
        or not queries
        or any(type(q) is not int or not 0 < q <= capacity for q in queries)
    ):
        raise ValueError(
            "asr_mtp.graph_query_sizes must contain positive query capacities"
        )
    queries = tuple(sorted(set(queries)))
    batch = min(batch, slots, max_batch)
    if batch < 1 or batch * len(queries) > MAX_MTP_GRAPHS:
        raise ValueError(f"asr_mtp graph count must be between 1 and {MAX_MTP_GRAPHS}")
    return GraphOptions(batch, queries, capacity)


class FixedMTPProgram:
    """Fixed B/Q/K tensor program with device-side lengths and cache addresses.

    Persistent inputs/outputs/arena are allocated outside graph pools. Cache
    transfers address aligned vector rows instead of individual scalars. Invalid
    query writes have unique scratch rows, including at the context cap;
    clamping padded writes onto the last live position would corrupt that row.
    """

    def __init__(
        self, heads, arena, embedding, logits, *, batch_size, query_size, context_size
    ):
        cfg = heads.config
        if not 0 < query_size <= context_size <= arena.max_context:
            raise ValueError("invalid fixed MTP query/context capacity")
        if not 0 < batch_size <= arena.storage.shape[0]:
            raise ValueError("invalid fixed MTP batch capacity")
        needed = (
            batch_size
            * query_size
            * max(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim)
        )
        if arena.graph_storage.numel() - arena.scratch_offset < needed:
            raise ValueError("MTP arena lacks the graph scratch suffix")
        self.heads, self.arena = heads, arena
        self.embedding, self.logits = embedding, logits
        self.batch, self.queries, self.context = batch_size, query_size, context_size
        device = arena.storage.device
        # Both hidden and KV regions, including each slot and the scratch suffix,
        # are aligned to this width without changing the existing arena layout.
        self.row_width = gcd(cfg.hidden_size, cfg.head_dim)
        row_elements = arena.graph_storage.numel() // self.row_width * self.row_width
        self.cache_rows = arena.graph_storage[:row_elements].view(-1, self.row_width)
        self.npu_row_update = None
        if device.type == "npu":
            import torch_npu

            self.npu_row_update = getattr(torch_npu, "npu_scatter_nd_update_", None)
            if self.npu_row_update is None:
                raise ValueError("MTP graphs require torch_npu.npu_scatter_nd_update_")
        self.metadata = torch.zeros(
            (batch_size, 3 + heads.num_heads + context_size + heads.num_heads + 1),
            device=device,
            dtype=torch.long,
        )
        self.output = torch.empty(
            (batch_size, heads.num_heads), device=device, dtype=torch.long
        )
        self.query_index = torch.arange(query_size, device=device)
        self.key_index = torch.arange(context_size, device=device)
        self.hidden_index = torch.arange(
            cfg.hidden_size // self.row_width, device=device
        )
        self.kv_head_index = torch.arange(cfg.num_key_value_heads, device=device)
        self.dim_index = torch.arange(cfg.head_dim // self.row_width, device=device)
        self.hidden_sink = (
            arena.scratch_offset // self.row_width
            + torch.arange(
                batch_size * query_size * cfg.hidden_size // self.row_width,
                device=device,
            )
        ).view(batch_size, query_size, cfg.hidden_size // self.row_width)
        self.kv_sink = (
            arena.scratch_offset // self.row_width
            + torch.arange(
                batch_size
                * cfg.num_key_value_heads
                * query_size
                * cfg.head_dim
                // self.row_width,
                device=device,
            )
        ).view(
            batch_size,
            cfg.num_key_value_heads,
            query_size,
            cfg.head_dim // self.row_width,
        )
        self.inverse = 1.0 / (
            cfg.rope_theta
            ** (
                torch.arange(0, cfg.head_dim, 2, device=device, dtype=torch.float32)
                / cfg.head_dim
            )
        )

    def stage(self, states, inputs, limits):
        if not len(states) == len(inputs) == len(limits) == self.batch:
            raise ValueError("fixed MTP requires the exact captured batch size")
        slots = [state.slot for state in states]
        if len(set(slots)) != len(slots) or any(
            not 0 <= slot < self.arena.storage.shape[0] for slot in slots
        ):
            raise ValueError("fixed MTP requires distinct live arena slots")
        rows = []
        depth = self.heads.num_heads
        for state, tokens, limit in zip(states, inputs, limits):
            if (
                type(limit) is not int
                or not 0 < limit <= depth
                or not 0 < state.length <= self.context
                or len(tokens) != state.length + 1
            ):
                raise ValueError("fixed MTP history/drive/budget mismatch")
            starts = [head.valid for head in state.heads]
            if any(
                not 0 <= start < state.length or state.length - start > self.queries
                for start in starts[:limit]
            ):
                raise ValueError("fixed MTP fresh queries exceed captured capacity")
            rows.append(
                [
                    state.slot,
                    state.length,
                    *starts,
                    limit,
                    *tokens,
                    *([0] * (self.context + depth + 1 - len(tokens))),
                ]
            )
        # Fresh pinned storage per upload: never mutate a staging buffer whose
        # asynchronous H2D might still be pending from an earlier replay.
        host = torch.tensor(
            rows, dtype=torch.long, pin_memory=self.metadata.device.type == "npu"
        )
        self.metadata.copy_(host, non_blocking=True)

    def _hidden_indices(self, base, positions):
        return (
            base[:, None, None]
            + positions[:, :, None] * (self.heads.config.hidden_size // self.row_width)
            + self.hidden_index[None, None, :]
        )

    def _kv_indices(self, base, positions):
        cfg = self.heads.config
        return (
            base[:, None, None, None]
            + self.kv_head_index[None, :, None, None]
            * self.arena.max_context
            * (cfg.head_dim // self.row_width)
            + positions[:, None, :, None] * (cfg.head_dim // self.row_width)
            + self.dim_index[None, None, None, :]
        )

    def _read(self, indices):
        return self.cache_rows.index_select(0, indices.reshape(-1)).view(
            *indices.shape[:-1], indices.shape[-1] * self.row_width
        )

    def _write(self, indices, value, valid, sink):
        addresses = torch.where(valid, indices, sink)
        updates = value.reshape(-1, self.row_width)
        if self.npu_row_update is not None:
            # Explicit device API avoids legacy index_copy_ implementations that
            # fetch each index on the host and cannot be safely captured.
            self.npu_row_update(self.cache_rows, addresses.reshape(-1, 1), updates)
        else:
            self.cache_rows.index_copy_(0, addresses.reshape(-1), updates)

    @torch.inference_mode()
    def run(self):
        cfg, heads = self.heads.config, self.heads.num_heads
        cap, width = self.arena.max_context, cfg.hidden_size
        hidden_width = cap * width // self.row_width
        kv_width = cfg.num_key_value_heads * cap * cfg.head_dim // self.row_width
        metadata = self.metadata
        base = metadata[:, 0] * (self.arena.storage.shape[1] // self.row_width)
        lengths, limits = metadata[:, 1], metadata[:, 2 + heads]
        sequence = metadata[:, 3 + heads :].clone()
        for depth, block in enumerate(self.heads.blocks, 1):
            starts = metadata[:, 1 + depth]
            active = limits >= depth
            anchors = starts[:, None] + self.query_index[None, :]
            valid = active[:, None] & (anchors < lengths[:, None])
            positions = anchors.clamp(0, cap - 1)
            source_offset = (
                0
                if depth == 1
                else hidden_width + (depth - 2) * (hidden_width + 2 * kv_width)
            )
            source = self._read(self._hidden_indices(base + source_offset, positions))
            source = torch.where(valid[:, :, None], source, 0)
            indices = (anchors + depth).clamp(0, sequence.shape[1] - 1)
            shifted = sequence.gather(1, indices)
            token_embeddings = self.embedding(shifted.reshape(-1)).view(
                self.batch, self.queries, width
            )
            fused = block.fusion(
                torch.cat(
                    (block.hidden_norm(source), block.token_norm(token_embeddings)), -1
                )
            )
            normed = block.decoder.input_layernorm(fused)
            attn = block.decoder.self_attn
            q = attn.q_norm(
                attn.q_proj(normed).view(
                    self.batch, self.queries, cfg.num_attention_heads, cfg.head_dim
                )
            ).transpose(1, 2)
            k = attn.k_norm(
                attn.k_proj(normed).view(
                    self.batch, self.queries, cfg.num_key_value_heads, cfg.head_dim
                )
            ).transpose(1, 2)
            v = (
                attn.v_proj(normed)
                .view(self.batch, self.queries, cfg.num_key_value_heads, cfg.head_dim)
                .transpose(1, 2)
            )
            frequencies = (anchors + depth).float().unsqueeze(-1) * self.inverse
            angles = torch.cat((frequencies, frequencies), -1)
            cos = angles.cos().to(fused.dtype)[:, None]
            sin = angles.sin().to(fused.dtype)[:, None]
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
            hidden_base = (
                base + hidden_width + (depth - 1) * (hidden_width + 2 * kv_width)
            )
            key_base, value_base = (
                hidden_base + hidden_width,
                hidden_base + hidden_width + kv_width,
            )
            for offset, value in ((key_base, k), (value_base, v)):
                self._write(
                    self._kv_indices(offset, positions),
                    value,
                    valid[:, None, :, None],
                    self.kv_sink,
                )
            key_positions = self.key_index[None, :].expand(self.batch, -1)
            cache_valid = active[:, None] & (key_positions < lengths[:, None])
            past_k = torch.where(
                cache_valid[:, None, :, None],
                self._read(self._kv_indices(key_base, key_positions)),
                0,
            )
            past_v = torch.where(
                cache_valid[:, None, :, None],
                self._read(self._kv_indices(value_base, key_positions)),
                0,
            )
            allowed = (self.key_index[None, None, :] <= anchors[:, :, None]) & (
                self.key_index[None, None, :] < lengths[:, None, None]
            )
            # Inactive/padded queries still execute fixed-shape kernels. Give
            # them one finite key; never depend on backend all-masked behavior.
            allowed = torch.where(
                valid[:, :, None], allowed, self.key_index[None, None, :] == 0
            )
            groups = cfg.num_attention_heads // cfg.num_key_value_heads
            attended = F.scaled_dot_product_attention(
                q,
                past_k.repeat_interleave(groups, 1),
                past_v.repeat_interleave(groups, 1),
                attn_mask=allowed[:, None],
                dropout_p=0.0,
                scale=cfg.head_dim**-0.5,
            )
            residual = fused + attn.o_proj(
                attended.transpose(1, 2).reshape(self.batch, self.queries, width)
            )
            fresh = block.output_norm(
                residual
                + block.decoder.mlp(block.decoder.post_attention_layernorm(residual))
            )
            last_index = (lengths - starts - 1).clamp(0, self.queries - 1)
            last = fresh.gather(
                1, last_index[:, None, None].expand(-1, 1, width)
            ).squeeze(1)
            predicted = torch.where(active, self.logits(last).argmax(-1), 0)
            self.output[:, depth - 1].copy_(predicted)
            if depth < heads:
                # Only a later head consumes this hidden state and token. Keep
                # the final head's KV above for the next round, but omit its
                # unused persistent hidden and local sequence updates.
                sequence.scatter_(1, (lengths + depth)[:, None], predicted[:, None])
                self._write(
                    self._hidden_indices(hidden_base, positions),
                    fresh,
                    valid[:, :, None],
                    self.hidden_sink,
                )
        return self.output

    def commit(self, states, limits):
        # CPU bookkeeping only, after the replay was successfully enqueued.
        for state, limit in zip(states, limits):
            for head in state.heads[:limit]:
                self.heads.processed_positions += state.length - head.valid
                self.heads.reused_positions += head.valid
                head.valid = state.length


class NPUCapture:
    """One serial auxiliary pool, independent of target ACLGraphWrapper state.

    Graphs have no persistent intermediates in this pool: metadata, outputs,
    weights and arena all live outside it. A replay's output must be consumed on
    the current stream before another replay. No cross-stream graph execution.
    """

    def __init__(self):
        self.pool = torch.npu.graph_pool_handle()

    def capture(self, program):
        # Startup only; warm lazy Torch/NPU kernels before graph recording.
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(2):
                program.run()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, pool=self.pool):
            program.run()
        torch.npu.synchronize()
        # Hold the graph itself; its bound replay method keeps it alive.
        return graph.replay


class MTPGraphs:
    def __init__(self, options, heads, arena, embedding, logits):
        self.options, self.heads, self.arena = options, heads, arena
        self.embedding, self.logits = embedding, logits
        self.entries, self.stats = {}, Counter()
        self.ready = False
        self.capture_backend = None
        self.reserved_bytes = 0

    @torch.inference_mode()
    def prepare(self, capture=None):
        if self.ready:
            return
        if self.arena.leased:
            raise RuntimeError("MTP graph capture must precede live requests")
        if capture is None:
            if self.arena.storage.device.type != "npu":
                raise RuntimeError("MTP full graph capture requires an NPU worker")
            capture = NPUCapture()
        self.capture_backend = capture
        is_npu = self.arena.storage.device.type == "npu"
        if is_npu:
            torch.npu.synchronize()
            torch.npu.empty_cache()
        reserved_before = torch.npu.memory_reserved() if is_npu else 0
        before = self.heads.processed_positions, self.heads.reused_positions
        start_time = perf_counter()
        self.arena.storage.zero_()
        states = []
        try:
            for _ in range(self.options.max_batch):
                states.append(self.arena.acquire())
            # Largest first to reuse temporary pool allocations. Live graph
            # inputs/outputs are deliberately outside that shared pool.
            for queries in reversed(self.options.queries):
                for count in range(self.options.max_batch, 0, -1):
                    warm = states[:count]
                    for state in warm:
                        state.length = self.options.context
                        for head in state.heads:
                            head.valid = self.options.context - queries
                    program = FixedMTPProgram(
                        self.heads,
                        self.arena,
                        self.embedding,
                        self.logits,
                        batch_size=count,
                        query_size=queries,
                        context_size=self.options.context,
                    )
                    program.stage(
                        warm,
                        [[0] * (state.length + 1) for state in warm],
                        [self.heads.num_heads] * count,
                    )
                    replay = capture.capture(program)
                    self.entries[count, queries] = program, replay
        finally:
            for state in states:
                self.arena.release(state)
            self.heads.processed_positions, self.heads.reused_positions = before
        self.ready = True
        if is_npu:
            torch.npu.synchronize()
            # Release ordinary warmup cache, retaining the live private pool
            # and graph buffers. Reserve their full delta conservatively.
            torch.npu.empty_cache()
            self.reserved_bytes = max(0, torch.npu.memory_reserved() - reserved_before)
        logger.info(
            "ASR_MTP_GRAPH_READY graphs=%d max_batch=%d query_sizes=%s context=%d "
            "capture_ms=%.2f reserved_mib=%.2f",
            len(self.entries),
            self.options.max_batch,
            self.options.queries,
            self.options.context,
            (perf_counter() - start_time) * 1000,
            self.reserved_bytes / 1024**2,
        )

    @torch.inference_mode()
    def propose(self, states, inputs, limits):
        if not self.ready:
            raise RuntimeError("MTP graphs were not prepared during startup profiling")
        if not states or len(states) > self.options.max_batch:
            self.stats["miss_batch"] += 1
            return None
        if max(limits) != self.heads.num_heads:
            self.stats["miss_budget"] += 1
            return None
        max_context = max(state.length for state in states)
        if max_context > self.options.context:
            self.stats["miss_context"] += 1
            if logger.isEnabledFor(logging.INFO):
                # CPU metadata only; do not fetch tensor values or synchronize
                # the device just to diagnose an eager fallback.
                max_fresh_query = max(
                    state.length - head.valid
                    for state, limit in zip(states, limits)
                    for head in state.heads[:limit]
                )
                logger.info(
                    "ASR_MTP_CONTEXT_MISS batch_size=%d max_context=%d context_limit=%d "
                    "max_fresh_query=%d query_limit=%d miss_count=%d",
                    len(states),
                    max_context,
                    self.options.context,
                    max_fresh_query,
                    max(self.options.queries),
                    self.stats["miss_context"],
                )
            return None
        fresh = max(
            state.length - head.valid
            for state, limit in zip(states, limits)
            for head in state.heads[:limit]
        )
        queries = next((q for q in self.options.queries if fresh <= q), None)
        if queries is None:
            self.stats["miss_query"] += 1
            return None
        program, replay = self.entries[len(states), queries]
        program.stage(states, inputs, limits)
        replay()
        program.commit(states, limits)
        self.stats["replays"] += 1
        self.stats["requests"] += len(states)
        if self.stats["replays"] == 1:
            logger.info(
                "ASR_MTP_GRAPH_REPLAY batch=%d query=%d context=%d heads=%d",
                len(states),
                queries,
                self.options.context,
                self.heads.num_heads,
            )
        return program.output
