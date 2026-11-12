# SPDX-License-Identifier: Apache-2.0
"""Packed auxiliary MTP graphs keyed by execution depth and fresh-token capacity."""

import logging
from collections import Counter
from math import gcd
from time import perf_counter

import torch

try:
    from .asr_mtp_fused import add_rms_norm
    from .asr_mtp_graph import captured_depth, depth_key
    from .asr_mtp_heads import project_norm_rope
    from .asr_mtp_row_copy import prepare_row_writer
    from .asr_mtp_token_attention import PackedAttention, PackedNPUCapture
except ImportError:
    from asr_mtp_fused import add_rms_norm
    from asr_mtp_graph import captured_depth, depth_key
    from asr_mtp_heads import project_norm_rope
    from asr_mtp_row_copy import prepare_row_writer
    from asr_mtp_token_attention import PackedAttention, PackedNPUCapture

logger = logging.getLogger(__name__)


def required_tokens(states, limits, heads):
    """Largest packed query count among sequential auxiliary heads (CPU only)."""
    return max(
        sum(
            state.length - state.heads[d].valid
            for state, limit in zip(states, limits)
            if limit > d
        )
        for d in range(heads)
    )


class PackedMTPProgram:
    """Fixed T/H program with per-head packed request boundaries.

    Bcap is a preallocation and logits-row bound, not a graph key. Each head has
    its own packed queries because rejection and budgets change fresh prefixes.
    CPU staging uploads only request metadata; row maps are derived on device.
    """

    def __init__(
        self,
        heads,
        arena,
        embedding,
        logits,
        *,
        max_batch,
        token_size,
        context_size,
        row_write_backend="auto",
        execution_heads=None,
    ):
        cfg = heads.config
        self.execution_heads = (
            heads.num_heads if execution_heads is None else execution_heads
        )
        if (
            type(self.execution_heads) is not int
            or not 0 < self.execution_heads <= heads.num_heads
        ):
            raise ValueError("invalid MTP execution depth")
        if (
            type(max_batch) is not int
            or not 0 < max_batch <= arena.storage.shape[0]
            or type(token_size) is not int
            or token_size < 1
            or type(context_size) is not int
            or not 0 < context_size <= arena.max_context
        ):
            raise ValueError("invalid packed MTP capacities")
        query_capacity = token_size + 1
        needed = query_capacity * max(
            cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim
        )
        if arena.graph_storage.numel() - arena.scratch_offset < needed:
            raise ValueError("MTP arena lacks the packed graph scratch suffix")
        self.heads, self.arena = heads, arena
        self.embedding, self.logits = embedding, logits
        self.max_batch, self.tokens, self.context = max_batch, token_size, context_size
        self.query_capacity = query_capacity
        self.paged_kv = getattr(arena, "paged_kv", False)
        self.kv_capacity = (
            arena.token_capacity
            if self.paged_kv
            else max_batch * context_size + query_capacity
        )
        # Paged K/V is separate: the flat arena contains only full hidden
        # vectors. Express their contiguous H elements as one row, rather than
        # H/gcd independently indexed fragments. Mixed hidden/KV TND keeps its
        # shared alignment because not every KV region is H-aligned.
        self.row_width = (
            cfg.hidden_size if self.paged_kv else gcd(cfg.hidden_size, cfg.head_dim)
        )
        row_elements = arena.graph_storage.numel() // self.row_width * self.row_width
        self.cache_rows = arena.graph_storage[:row_elements].view(-1, self.row_width)
        device, dtype = arena.storage.device, arena.storage.dtype
        self.npu_row_update, self.row_write_backend = prepare_row_writer(
            device, backend=row_write_backend
        )
        count, requests = heads.num_heads, max_batch + 1
        self.sequence_width = context_size + count + 1
        size = 5 * count * requests + 3 * requests + requests * self.sequence_width
        self.metadata = torch.zeros(size, device=device, dtype=torch.long)
        offset = 0

        def view(shape):
            nonlocal offset
            n = 1
            for extent in shape:
                n *= extent
            value = self.metadata[offset : offset + n].view(shape)
            offset += n
            return value

        self.q_ends, self.kv_ends, self.order, self.starts, self.last = (
            view((count, requests)) for _ in range(5)
        )
        self.slots, self.lengths, self.limits = (view((requests,)) for _ in range(3))
        self.sequence = view((requests, self.sequence_width))
        self.output = torch.empty((max_batch, count), device=device, dtype=torch.long)
        self.token_index = torch.arange(query_capacity, device=device)
        self.key_index = (
            None if self.paged_kv else torch.arange(self.kv_capacity, device=device)
        )
        self.hidden_index = torch.arange(
            cfg.hidden_size // self.row_width, device=device
        )
        self.kv_head_index = torch.arange(cfg.num_key_value_heads, device=device)
        self.dim_index = torch.arange(cfg.head_dim // self.row_width, device=device)
        self.zero = torch.zeros(1, device=device, dtype=torch.long)
        self.hidden_sink = (
            arena.scratch_offset // self.row_width
            + torch.arange(
                query_capacity * cfg.hidden_size // self.row_width, device=device
            )
        ).view(query_capacity, cfg.hidden_size // self.row_width)
        self.kv_sink = None
        if not self.paged_kv:
            self.kv_sink = (
                arena.scratch_offset // self.row_width
                + torch.arange(
                    query_capacity
                    * cfg.num_key_value_heads
                    * cfg.head_dim
                    // self.row_width,
                    device=device,
                )
            ).view(
                query_capacity, cfg.num_key_value_heads, cfg.head_dim // self.row_width
            )
        self.inverse = 1.0 / (
            cfg.rope_theta
            ** (
                torch.arange(0, cfg.head_dim, 2, device=device, dtype=torch.float32)
                / cfg.head_dim
            )
        )
        attention_class, extra = PackedAttention, {}
        if self.paged_kv:
            try:
                from .asr_mtp_paged_attention import PagedAttention
            except ImportError:
                from asr_mtp_paged_attention import PagedAttention
            attention_class = PagedAttention
            extra = {
                "max_batch": max_batch,
                "block_size": arena.block_size,
                "num_blocks": arena.total_blocks,
                "max_blocks": len(arena.block_rows[0]),
            }
        self.attention = attention_class(
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            device=device,
            dtype=dtype,
            heads=self.execution_heads,
            **extra,
        )

    def stage(self, states, inputs, limits):
        n, h, r = len(states), self.heads.num_heads, self.max_batch + 1
        if not n or not n == len(inputs) == len(limits) or n > self.max_batch:
            raise ValueError("packed MTP requires aligned bounded requests")
        slots = [state.slot for state in states]
        if len(set(slots)) != n or any(
            type(slot) is not int or not 0 <= slot < self.arena.storage.shape[0]
            for slot in slots
        ):
            raise ValueError("packed MTP requires distinct live arena slots")
        for state, tokens, limit in zip(states, inputs, limits):
            if (
                type(limit) is not int
                or not 0 < limit <= self.execution_heads
                or not 0 < state.length <= self.context
                or len(tokens) != state.length + 1
                or any(
                    not 0 <= head.valid < state.length for head in state.heads[:limit]
                )
            ):
                raise ValueError("packed MTP history/drive/budget mismatch")
        if required_tokens(states, limits, h) > self.tokens:
            raise ValueError("packed MTP queries exceed token capacity")
        q_rows, kv_rows, orders, start_rows, last_rows, segments = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for d in range(h):
            q_ends, kv_ends, order = [], [], []
            last, starts = [0] * r, [0] * r
            q_total = kv_total = 0
            for i, (state, limit) in enumerate(zip(states, limits)):
                starts[i] = state.heads[d].valid
                if d >= limit:
                    continue
                q_total += state.length - state.heads[d].valid
                kv_total += state.length
                order.append(i)
                q_ends.append(q_total)
                kv_ends.append(kv_total)
                last[i] = q_total - 1
            # A positive dummy Q exists even at the exact logical token limit.
            # Its zero KV segment covers the remaining physical K/V capacity,
            # satisfying strict non-PA TND tilers without changing tensor shapes.
            q_ends.append(self.query_capacity)
            kv_ends.append(
                kv_total + self.query_capacity - q_total
                if self.paged_kv
                else self.kv_capacity
            )
            order.append(self.max_batch)
            segments.append((tuple(q_ends), tuple(kv_ends)))
            q_rows.append(q_ends + [self.query_capacity] * (r - len(q_ends)))
            kv_rows.append(kv_ends + [self.kv_capacity] * (r - len(kv_ends)))
            orders.append(order + [self.max_batch] * (r - len(order)))
            start_rows.append(starts)
            last_rows.append(last)

        def flatten(rows):
            return [x for row in rows for x in row]

        sequences = [
            list(tokens) + [0] * (self.sequence_width - len(tokens))
            for tokens in inputs
        ]
        sequences.extend([[0] * self.sequence_width for _ in range(r - n)])
        payload = [
            *flatten(q_rows),
            *flatten(kv_rows),
            *flatten(orders),
            *flatten(start_rows),
            *flatten(last_rows),
            *slots,
            *([0] * (r - n)),
            *[state.length for state in states],
            *([0] * (r - n)),
            *limits,
            *([0] * (r - n)),
            *flatten(sequences),
        ]
        # All validation precedes upload. The fresh pinned buffer is never
        # mutated while the asynchronous transfer may still be pending.
        host = torch.tensor(
            payload, dtype=torch.long, pin_memory=self.metadata.device.type == "npu"
        )
        self.metadata.copy_(host, non_blocking=True)
        if self.paged_kv:
            tables = [
                [
                    self.arena.block_rows[slots[i]]
                    if i < n
                    else self.arena.block_rows[-1]
                    for i in order[: len(segment[0])]
                ]
                for order, segment in zip(
                    orders[: self.execution_heads], segments[: self.execution_heads]
                )
            ]
            self.attention.set_tables(tables)
        for d, (q_ends, kv_ends) in enumerate(segments[: self.execution_heads]):
            self.attention.stage(d, q_ends, kv_ends)

    def _segments(self, index, ends):
        # Bcap is small and fixed. Avoid searchsorted implementations that may
        # dispatch to a host compatibility path on some installed NPU versions.
        segment = (index[:, None] >= ends[None, :]).sum(-1).clamp_max(ends.shape[0] - 1)
        begins = torch.cat((self.zero, ends[:-1]))
        return segment, index - begins.index_select(0, segment)

    def _hidden_indices(self, base, position):
        return (
            base[:, None]
            + position[:, None] * (self.heads.config.hidden_size // self.row_width)
            + self.hidden_index[None, :]
        )

    def _kv_indices(self, base, position):
        cfg = self.heads.config
        return (
            base[:, None, None]
            + self.kv_head_index[None, :, None]
            * self.arena.max_context
            * (cfg.head_dim // self.row_width)
            + position[:, None, None] * (cfg.head_dim // self.row_width)
            + self.dim_index[None, None, :]
        )

    def _read(self, indices):
        return self.cache_rows.index_select(0, indices.reshape(-1)).view(
            *indices.shape[:-1], indices.shape[-1] * self.row_width
        )

    def _write(self, indices, value, valid, sink):
        addresses = torch.where(valid, indices, sink)
        updates = value.reshape(-1, self.row_width)
        if self.npu_row_update is not None:
            self.npu_row_update(self.cache_rows, addresses.reshape(-1, 1), updates)
        else:
            self.cache_rows.index_copy_(0, addresses.reshape(-1), updates)

    @torch.inference_mode()
    def run(self):
        cfg, count = self.heads.config, self.heads.num_heads
        cap, width = self.arena.max_context, cfg.hidden_size
        hidden_width = cap * width // self.row_width
        kv_width = (
            0
            if self.paged_kv
            else cfg.num_key_value_heads * cap * cfg.head_dim // self.row_width
        )
        stride = self.arena.storage.shape[1] // self.row_width
        sequence = self.sequence.clone()
        for d, block in enumerate(self.heads.blocks[: self.execution_heads]):
            depth = d + 1
            segment, relative = self._segments(self.token_index, self.q_ends[d])
            request = self.order[d].index_select(0, segment)
            anchor = self.starts[d].index_select(0, request) + relative
            valid = self.limits.index_select(0, request) >= depth
            positions = anchor.clamp(0, cap - 1)
            base = self.slots.index_select(0, request) * stride
            source_offset = (
                0 if d == 0 else hidden_width + (d - 1) * (hidden_width + 2 * kv_width)
            )
            source = self._read(self._hidden_indices(base + source_offset, positions))
            source = torch.where(valid[:, None], source, 0)
            shifted_index = request * self.sequence_width + (anchor + depth).clamp(
                0, self.sequence_width - 1
            )
            shifted = sequence.reshape(-1).index_select(0, shifted_index)
            embedded = self.embedding(shifted)
            fused = block.fusion(
                torch.cat((block.hidden_norm(source), block.token_norm(embedded)), -1)
            )
            normed = block.decoder.input_layernorm(fused)
            attn = block.decoder.self_attn
            q, k, v = project_norm_rope(
                self.heads, attn, normed, anchor, depth, self.inverse, cap
            )
            hidden_base = base + hidden_width + d * (hidden_width + 2 * kv_width)
            if self.paged_kv:
                slot = self.slots.index_select(0, request)
                addresses = torch.where(
                    valid,
                    slot * self.arena.slot_capacity + anchor,
                    self.arena.dummy_base + self.token_index,
                ).to(torch.int32)
                self.arena.write_kv(
                    d,
                    torch.where(valid[:, None, None], k, 0),
                    torch.where(valid[:, None, None], v, 0),
                    addresses,
                )
                past_k, past_v = self.arena.attention_kv(d)
            else:
                past_k, past_v = self._contiguous_kv(
                    d, k, v, hidden_base, positions, valid
                )
            attended = self.attention(d, q, past_k, past_v, cfg.head_dim**-0.5)
            normed_residual, residual = add_rms_norm(
                block.decoder.post_attention_layernorm,
                fused,
                attn.o_proj(attended.reshape(self.query_capacity, -1)),
            )
            fresh, _ = add_rms_norm(
                block.output_norm, residual, block.decoder.mlp(normed_residual)
            )
            last = fresh.index_select(0, self.last[d, : self.max_batch])
            active = self.limits[: self.max_batch] >= depth
            predicted = torch.where(active, self.logits(last).argmax(-1), 0)
            self.output[:, d].copy_(predicted)
            if depth < count:
                sequence[: self.max_batch].scatter_(
                    1,
                    (self.lengths[: self.max_batch] + depth)[:, None],
                    predicted[:, None],
                )
                self._write(
                    self._hidden_indices(hidden_base, positions),
                    fresh,
                    valid[:, None],
                    self.hidden_sink,
                )
        return self.output

    def _contiguous_kv(self, d, k, v, hidden_base, positions, valid):
        cfg, cap = self.heads.config, self.arena.max_context
        hidden_width = cap * cfg.hidden_size // self.row_width
        kv_width = cfg.num_key_value_heads * cap * cfg.head_dim // self.row_width
        stride = self.arena.storage.shape[1] // self.row_width
        depth = d + 1
        key_base, value_base = (
            hidden_base + hidden_width,
            hidden_base + hidden_width + kv_width,
        )
        for offset, value in ((key_base, k), (value_base, v)):
            self._write(
                self._kv_indices(offset, positions),
                value,
                valid[:, None, None],
                self.kv_sink,
            )
        key_segment, key_position = self._segments(self.key_index, self.kv_ends[d])
        key_request = self.order[d].index_select(0, key_segment)
        key_valid = self.limits.index_select(0, key_request) >= depth
        key_position = key_position.clamp(0, cap - 1)
        key_hidden_base = (
            self.slots.index_select(0, key_request) * stride
            + hidden_width
            + d * (hidden_width + 2 * kv_width)
        )
        past_k = torch.where(
            key_valid[:, None, None],
            self._read(self._kv_indices(key_hidden_base + hidden_width, key_position)),
            0,
        )
        past_v = torch.where(
            key_valid[:, None, None],
            self._read(
                self._kv_indices(
                    key_hidden_base + hidden_width + kv_width, key_position
                )
            ),
            0,
        )
        return past_k, past_v

    def commit(self, states, limits):
        for state, limit in zip(states, limits):
            for head in state.heads[:limit]:
                self.heads.processed_positions += state.length - head.valid
                self.heads.reused_positions += head.valid
                head.valid = state.length


class TokenMTPGraphs:
    def __init__(self, options, heads, arena, embedding, logits):
        self.options, self.heads, self.arena = options, heads, arena
        self.embedding, self.logits = embedding, logits
        self.entries, self.stats = {}, Counter()
        self.ready = False
        self.capture_backend = None
        self.reserved_bytes = 0
        self.allocated_bytes = 0

    @torch.inference_mode()
    def prepare(self, capture=None):
        if self.ready:
            return
        if self.arena.leased:
            raise RuntimeError("MTP graph capture must precede live requests")
        if capture is None:
            if self.arena.storage.device.type != "npu":
                raise RuntimeError("MTP token graph capture requires an NPU worker")
            capture = PackedNPUCapture()
        self.capture_backend = capture
        is_npu = self.arena.storage.device.type == "npu"
        if is_npu:
            torch.npu.synchronize()
            torch.npu.empty_cache()
        reserved = torch.npu.memory_reserved() if is_npu else 0
        allocated = torch.npu.memory_allocated() if is_npu else 0
        before = self.heads.processed_positions, self.heads.reused_positions
        started = perf_counter()
        self.arena.storage.zero_()
        states = []
        try:
            states = [self.arena.acquire() for _ in range(self.options.max_batch)]
            for depth in reversed(
                getattr(self.options, "head_counts", ()) or (self.heads.num_heads,)
            ):
                for tokens in reversed(self.options.tokens):
                    count = min(self.options.max_batch, tokens)
                    warm = states[:count]
                    available = min(tokens, count * self.options.context)
                    for i, state in enumerate(warm):
                        fresh = min(self.options.context, available - (count - i - 1))
                        available -= fresh
                        state.length = self.options.context
                        for head in state.heads:
                            head.valid = state.length - fresh
                    program = PackedMTPProgram(
                        self.heads,
                        self.arena,
                        self.embedding,
                        self.logits,
                        max_batch=self.options.max_batch,
                        token_size=tokens,
                        context_size=self.options.context,
                        row_write_backend=getattr(
                            self.options, "row_write_backend", "auto"
                        ),
                        execution_heads=depth,
                    )
                    program.profile_stages = getattr(
                        self.options, "profile_stages", False
                    )
                    program.stage(
                        warm,
                        [[0] * (s.length + 1) for s in warm],
                        [depth] * count,
                    )
                    self.entries[depth_key(self.options, depth, tokens)] = (
                        program,
                        capture.capture(program),
                    )
        finally:
            for state in states:
                self.arena.release(state)
            self.heads.processed_positions, self.heads.reused_positions = before
        self.ready = True
        if is_npu:
            torch.npu.synchronize()
            torch.npu.empty_cache()
            self.reserved_bytes = max(0, torch.npu.memory_reserved() - reserved)
            self.allocated_bytes = max(0, torch.npu.memory_allocated() - allocated)
        logger.info(
            "ASR_MTP_GRAPH_READY layout=tokens graphs=%d max_batch=%d token_sizes=%s "
            "query_capacities=%s context=%d capture_ms=%.2f reserved_mib=%.2f "
            "allocated_mib=%.2f row_write_backend=%s row_width=%d head_counts=%s",
            len(self.entries),
            self.options.max_batch,
            self.options.tokens,
            tuple(t + 1 for t in self.options.tokens),
            self.options.context,
            (perf_counter() - started) * 1000,
            self.reserved_bytes / 1024**2,
            self.allocated_bytes / 1024**2,
            next(iter(self.entries.values()))[0].row_write_backend,
            next(iter(self.entries.values()))[0].row_width,
            getattr(self.options, "head_counts", ()) or (self.heads.num_heads,),
        )

    @torch.inference_mode()
    def propose(self, states, inputs, limits):
        if not self.ready:
            raise RuntimeError("MTP token graphs were not prepared during startup")
        if not states or len(states) > self.options.max_batch:
            self.stats["miss_batch"] += 1
            return None
        if len(states) != len(inputs) or len(states) != len(limits):
            raise ValueError("aligned MTP graph requests required")
        depth = captured_depth(self.options, self.heads.num_heads, max(limits))
        if depth is None:
            self.stats["miss_budget"] += 1
            return None
        history = max(state.length for state in states)
        if history > self.options.context:
            self.stats["miss_context"] += 1
            if logger.isEnabledFor(logging.INFO):
                fresh = max(
                    state.length - head.valid
                    for state, limit in zip(states, limits)
                    for head in state.heads[:limit]
                )
                logger.info(
                    "ASR_MTP_CONTEXT_MISS layout=tokens batch_size=%d max_context=%d "
                    "context_limit=%d max_fresh_query=%d token_limit=%d miss_count=%d",
                    len(states),
                    history,
                    self.options.context,
                    fresh,
                    max(self.options.tokens),
                    self.stats["miss_context"],
                )
            return None
        needed = required_tokens(states, limits, self.heads.num_heads)
        bucket = next((t for t in self.options.tokens if needed <= t), None)
        if bucket is None:
            self.stats["miss_tokens"] += 1
            return None
        program, replay = self.entries[depth_key(self.options, depth, bucket)]
        profiled = getattr(self.options, "profile_stages", False)
        stage_started = perf_counter() if profiled else None
        program.stage(states, inputs, limits)
        replay_started = perf_counter() if profiled else None
        replay()
        commit_started = perf_counter() if profiled else None
        program.commit(states, limits)
        if profiled:
            timings = getattr(replay, "last_timings", {})
            logger.info(
                "ASR_MTP_GRAPH_HOST_STAGE requests=%d tokens=%d bucket=%d "
                "query_capacity=%d kv_capacity=%d max_batch=%d stage_host_ms=%.3f "
                "replay_host_ms=%.3f task_update_host_ms=%.3f "
                "replay_enqueue_host_ms=%.3f commit_host_ms=%.3f "
                "task_updates=%d task_update_reuses=%d",
                len(states),
                needed,
                bucket,
                program.query_capacity,
                program.kv_capacity,
                self.options.max_batch,
                (replay_started - stage_started) * 1000,
                (commit_started - replay_started) * 1000,
                timings.get("task_update_host_ms", 0.0),
                timings.get("replay_enqueue_host_ms", 0.0),
                (perf_counter() - commit_started) * 1000,
                timings.get("task_updates", 0),
                timings.get("task_update_reuses", 0),
            )
        self.stats[f"replays_h{depth}"] += 1
        self.stats["replays"] += 1
        self.stats["requests"] += len(states)
        if self.stats["replays"] == 1:
            logger.info(
                "ASR_MTP_GRAPH_REPLAY layout=tokens batch=%d tokens=%d bucket=%d "
                "query_capacity=%d kv_capacity=%d context=%d heads=%d",
                len(states),
                needed,
                bucket,
                program.query_capacity,
                program.kv_capacity,
                self.options.context,
                depth,
            )
        return program.output[: len(states)]
