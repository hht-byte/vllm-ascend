# SPDX-License-Identifier: Apache-2.0
"""FIA TND queries reading auxiliary PA cache directly through fixed block tables."""

import torch
from torch.nn import functional as F

try:
    from .asr_mtp_token_attention import PackedAttention
except ImportError:
    from asr_mtp_token_attention import PackedAttention


class PagedAttention(PackedAttention):
    def __init__(self, *args, max_batch, block_size, num_blocks, max_blocks, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size, self.num_blocks = block_size, num_blocks
        self.tables = torch.zeros(
            (self.heads, max_batch + 1, max_blocks),
            dtype=torch.int32,
            device=self.device,
        )
        self.table_rows = None
        self.kv_lengths = {}

    def set_tables(self, rows):
        """Upload all heads in one transfer; values originate in CPU slot metadata."""
        if len(rows) != self.heads:
            raise ValueError("paged MTP needs block tables for every head")
        payload = []
        for table in rows:
            if not table or len(table) > self.tables.shape[1]:
                raise ValueError("paged MTP block table batch mismatch")
            for row in table:
                if len(row) != self.tables.shape[2] or any(
                    type(b) is not int or not 0 <= b < self.num_blocks for b in row
                ):
                    raise ValueError("paged MTP block id or row width out of range")
                payload.extend(row)
            payload.extend(
                [0] * ((self.tables.shape[1] - len(table)) * self.tables.shape[2])
            )
        normalized = tuple(tuple(tuple(row) for row in table) for table in rows)
        if normalized == self.table_rows:
            return
        host = torch.tensor(
            payload, dtype=torch.int32, pin_memory=self.device.type == "npu"
        )
        self.tables.view(-1).copy_(host, non_blocking=True)
        self.table_rows = normalized

    def stage(self, head, q_ends, kv_ends):
        super().stage(head, q_ends, kv_ends)
        previous = (0, *kv_ends[:-1])
        self.kv_lengths[head] = tuple(
            end - begin for begin, end in zip(previous, kv_ends)
        )

    def _validate(self, head, query, key, value):
        if head not in self.lengths or self.table_rows is None:
            raise ValueError("paged MTP lengths and block tables must be staged")
        q_ends, _ = self.lengths[head]
        if (
            query.ndim != 3
            or query.shape[1:] != (self.num_heads, self.head_dim)
            or q_ends[-1] != query.shape[0]
            or key.shape
            != (self.num_blocks, self.block_size, self.num_kv_heads * self.head_dim)
            or value.shape != key.shape
            or len(self.table_rows[head]) != len(q_ends)
            or max(self.kv_lengths[head]) > self.tables.shape[2] * self.block_size
            or any(
                t.device != self.device
                or t.dtype != self.dtype
                or not t.is_contiguous()
                for t in (query, key, value)
            )
        ):
            raise ValueError("paged MTP query/cache/table geometry mismatch")

    def _arguments(self, head, query, key, value, scale):
        result = super()._arguments(head, query, key, value, scale)
        result.update(
            block_table=self.tables[head, : len(self.lengths[head][0])],
            block_size=self.block_size,
            actual_seq_lengths_kv=self.kv_lengths[head],
        )
        return result

    @torch.inference_mode()
    def __call__(self, head, query, key, value, scale):
        if self.is_npu:
            return super().__call__(head, query, key, value, scale)
        self._validate(head, query, key, value)
        # CPU-only correctness reference. Production FIA reads these pages
        # itself; there is no index_select/history materialization on NPU.
        result, start = torch.empty_like(query), 0
        groups = self.num_heads // self.num_kv_heads
        for i, (end, length) in enumerate(
            zip(self.lengths[head][0], self.kv_lengths[head])
        ):
            blocks = self.table_rows[head][i][
                : (length + self.block_size - 1) // self.block_size
            ]
            index = torch.tensor(blocks, dtype=torch.long)
            k = (
                key.index_select(0, index)
                .reshape(-1, self.num_kv_heads, self.head_dim)[:length]
                .transpose(0, 1)
            )
            v = (
                value.index_select(0, index)
                .reshape(-1, self.num_kv_heads, self.head_dim)[:length]
                .transpose(0, 1)
            )
            q_length = end - start
            mask = (
                torch.arange(length)[None]
                <= torch.arange(q_length)[:, None] + length - q_length
            )
            result[start:end].copy_(
                F.scaled_dot_product_attention(
                    query[start:end].transpose(0, 1),
                    k.repeat_interleave(groups, 0),
                    v.repeat_interleave(groups, 0),
                    attn_mask=mask[None],
                    dropout_p=0.0,
                    scale=scale,
                ).transpose(0, 1)
            )
            start = end
        return result
