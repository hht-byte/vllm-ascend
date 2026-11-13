# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One upload for CPU-authoritative synchronous speculative verifier metadata."""

import numpy as np
import torch


class SyncSpecDecodeMetadata:
    """Reuse fixed host/device storage on the synchronous runner stream.

    All consumers must finish using these views before the next engine step.
    The copy event protects pinned host reuse, not device consumers: those are
    ordered by the runner stream. Never use this for GPU-repaired input IDs.
    """

    def __init__(self, runner, *, event=None):
        self.max_requests = runner.seq_lens.numel()
        self.max_tokens = runner.max_num_tokens
        self.max_drafts = runner.num_spec_tokens
        self.event = event if event is not None else torch.npu.Event()
        self.pending = False
        self.uploads = 0
        self.arange = np.arange(self.max_tokens + self.max_requests, dtype=np.int64)
        sample_capacity = min(self.max_tokens, self.max_requests * (self.max_drafts + 1))
        draft_capacity = min(self.max_tokens, self.max_requests * self.max_drafts)
        fields = (
            ("cu_num_draft_tokens", self.max_requests, torch.int32),
            ("cu_num_sampled_tokens", self.max_requests, torch.int32),
            ("bonus_logits_indices", self.max_requests, torch.int32),
            ("logits_indices", sample_capacity, torch.int64),
            ("target_logits_indices", draft_capacity, torch.int64),
            ("draft_token_ids", draft_capacity, torch.int32),
        )
        layout, offset = {}, 0
        for name, capacity, dtype in fields:
            offset = (offset + 7) // 8 * 8
            size = capacity * torch.empty((), dtype=dtype).element_size()
            layout[name] = offset, size, dtype
            offset += size
        self.host = torch.zeros(offset, dtype=torch.uint8, pin_memory=runner.pin_memory)
        self.device = torch.zeros(offset, dtype=torch.uint8, device=runner.device)
        self.cpu, self.gpu = {}, {}
        for name, (start, size, dtype) in layout.items():
            self.cpu[name] = self.host[start : start + size].view(dtype).numpy()
            self.gpu[name] = self.device[start : start + size].view(dtype)
        self.draft_offset = layout["draft_token_ids"][0]

    @torch.inference_mode()
    def prepare(self, drafts, scheduled_ends, input_ids, metadata_type):
        n = len(drafts)
        lengths = np.diff(scheduled_ends, prepend=0)
        if (
            not 0 < n <= self.max_requests
            or len(scheduled_ends) != n
            or drafts.ndim != 1
            or scheduled_ends.ndim != 1
            or np.any(drafts < 0)
            or np.any(drafts > self.max_drafts)
            or np.any(lengths < drafts + 1)
            or scheduled_ends[-1] > min(self.max_tokens, len(input_ids))
        ):
            raise ValueError("speculative metadata exceeds capacity or has inconsistent lengths")
        if self.pending and not self.event.query():
            self.event.synchronize()
        cu_draft = self.cpu["cu_num_draft_tokens"][:n]
        cu_sample = self.cpu["cu_num_sampled_tokens"][:n]
        np.cumsum(drafts, dtype=np.int32, out=cu_draft)
        np.add(cu_draft, self.arange[1 : n + 1], out=cu_sample, casting="unsafe")
        d, s = int(cu_draft[-1]), int(cu_sample[-1])
        sampled = drafts + 1
        logits = self.cpu["logits_indices"][:s]
        logits[:] = (
            np.repeat(scheduled_ends - sampled, sampled) + self.arange[:s] - np.repeat(cu_sample - sampled, sampled)
        )
        target = self.cpu["target_logits_indices"][:d]
        target[:] = np.repeat(cu_sample - sampled, drafts) + self.arange[:d] - np.repeat(cu_draft - drafts, drafts)
        np.subtract(cu_sample, 1, out=self.cpu["bonus_logits_indices"][:n])
        # The synchronous runner already prepared these exact CPU input IDs.
        # NumPy indexing replaces two NPU Index/IndexCheck launches and Add.
        self.cpu["draft_token_ids"][:d] = input_ids[logits[target + 1]]
        size = self.draft_offset + d * 4
        self.device[:size].copy_(self.host[:size], non_blocking=True)
        self.event.record()
        self.pending = True
        self.uploads += 1
        sizes = {
            "cu_num_draft_tokens": n,
            "cu_num_sampled_tokens": n,
            "bonus_logits_indices": n,
            "logits_indices": s,
            "target_logits_indices": d,
            "draft_token_ids": d,
        }
        return metadata_type(
            num_draft_tokens=drafts.tolist(), **{key: self.gpu[key][:length] for key, length in sizes.items()}
        )
