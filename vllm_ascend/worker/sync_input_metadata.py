# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed metadata for synchronous ASR; no changes to scheduler or graph keys."""

from collections.abc import Mapping

import numpy as np
import torch


def enabled(runner) -> bool:
    config = runner.vllm_config.additional_config or {}
    value = config.get("asr_sync_input_packing", False)
    if type(value) is not bool:
        raise TypeError("asr_sync_input_packing must be a bool")
    if not value:
        return False
    if not isinstance(config.get("asr_mtp"), Mapping):
        raise ValueError("asr_sync_input_packing requires additional_config.asr_mtp")
    if (
        runner.use_async_scheduling
        or runner.use_async_spec_decode
        or runner.pcp_size != 1
        or runner.dcp_size != 1
        or getattr(getattr(runner, "model_config", None), "is_hybrid", False)
        or any(
            getattr(runner.parallel_config, field) != 1
            for field in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")
        )
        or any(getattr(runner, field, False) for field in ("_has_gdn", "use_sparse", "use_compress", "is_hybrid"))
    ):
        raise ValueError("asr_sync_input_packing requires synchronous single-device dense ASR")
    return True


class SyncInputMetadata:
    """One H2D for counts, padded seq_lens and the active positions prefix.

    Device views remain contiguous at stable addresses across capture/replay.
    Host storage cannot be reused until the preceding copy completes. The event
    covers only that copy; no device-wide or model-forward synchronization is used.
    """

    def __init__(self, max_tokens, max_requests, device, *, pin_memory, copy_event=None):
        self.max_tokens = max_tokens
        self.max_requests = max_requests
        # Two int32 request arrays naturally align the int64 positions to 8 bytes.
        self.position_offset = max_requests * 8
        size = self.position_offset + max_tokens * 8
        self.host = torch.zeros(size, dtype=torch.uint8, pin_memory=pin_memory)
        self.device = torch.zeros(size, dtype=torch.uint8, device=device)
        self.num_computed_tokens = self.device[: max_requests * 4].view(torch.int32)
        self.seq_lens = self.device[max_requests * 4 : self.position_offset].view(torch.int32)
        self.positions = self.device[self.position_offset :].view(torch.int64)
        host = self.host.numpy()
        self.counts_np = host[: max_requests * 4].view(np.int32)
        self.seq_lens_np = host[max_requests * 4 : self.position_offset].view(np.int32)
        self.positions_np = host[self.position_offset :].view(np.int64)
        self.copy_event = copy_event
        self.copy_pending = False
        self.last_copy_bytes = 0

    def prepare(self, counts, seq_lens, positions):
        n, t = len(counts), len(positions)
        if not 0 < n <= self.max_requests or not 0 < t <= self.max_tokens or len(seq_lens) != n:
            raise ValueError("input metadata exceeds captured buffer capacity or has inconsistent lengths")
        if self.copy_pending and not self.copy_event.query():
            self.copy_event.synchronize()
        self.counts_np[:n] = counts
        self.counts_np[n:] = 0
        self.seq_lens_np[:n] = seq_lens
        self.seq_lens_np[n:] = 0
        self.positions_np[:t] = positions
        self.last_copy_bytes = self.position_offset + t * 8
        self.device[: self.last_copy_bytes].copy_(self.host[: self.last_copy_bytes], non_blocking=True)
        if self.copy_event is not None:
            self.copy_event.record()
            self.copy_pending = True


def install(runner, *, prepare_inputs):
    """Called once at runner initialization, before model load and graph capture."""
    if not enabled(runner):
        return None
    if runner._prepare_inputs.__func__ is not prepare_inputs:
        raise ValueError("asr_sync_input_packing does not support overridden _prepare_inputs")
    counts, seq, pos = runner.num_computed_tokens, runner.seq_lens, runner.positions
    if (
        counts.dtype != torch.int32
        or seq.dtype != torch.int32
        or pos.dtype != torch.int64
        or counts.ndim != 1
        or seq.ndim != 1
        or pos.ndim != 1
        or counts.shape != seq.shape
    ):
        raise ValueError("unsupported runner metadata layout for asr_sync_input_packing")
    storage = SyncInputMetadata(
        pos.numel(), seq.numel(), runner.device, pin_memory=runner.pin_memory, copy_event=torch.npu.Event()
    )
    runner.num_computed_tokens = storage.num_computed_tokens
    runner.seq_lens = storage.seq_lens
    runner.positions = storage.positions
    return storage
