# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded pinned readback for synchronous ASR rejection bookkeeping."""

import torch


def install(runner):
    opts = (runner.vllm_config.additional_config or {}).get("asr_input_prepare_graph", {})
    if (
        runner._input_prepare_graph is None
        or not opts.get("reuse_rejection_buffer", True)
        or runner.use_async_scheduling
        or runner.use_async_spec_decode
        or runner.num_spec_tokens <= 0
    ):
        return None
    return RejectionOutputBuffer(runner.max_num_reqs, runner.num_spec_tokens + 1, device=runner.device)


class RejectionOutputBuffer:
    def __init__(self, max_requests, max_tokens, *, device, event=None):
        self.max_requests, self.max_tokens = max_requests, max_tokens
        self.host = torch.empty(
            max_requests * max_tokens,
            dtype=torch.int32,
            device="cpu",
            pin_memory=device.type == "npu",
        )
        self.event = event
        if self.event is None and device.type == "npu":
            self.event = torch.npu.Event()
        self.copies = self.fallbacks = 0

    def parse(self, tokens, vocab_size, discard, *, parser, logprobs_tensors):
        if (
            logprobs_tensors is not None
            or tokens.dtype != torch.int32
            or tokens.ndim != 2
            or not 0 < tokens.shape[0] <= self.max_requests
            or not 1 < tokens.shape[1] <= self.max_tokens
        ):
            self.fallbacks += 1
            return parser(tokens, vocab_size, discard, logprobs_tensors=logprobs_tensors)
        # Flat prefix makes each changing (requests, accepted-width) contiguous.
        # No device dtype cast, fresh host tensor, or per-request transfer.
        host = self.host[: tokens.numel()].view(tokens.shape)
        host.copy_(tokens, non_blocking=True)
        if self.event is not None:
            self.event.record()
            self.event.synchronize()
        self.copies += 1
        # The upstream parser's .cpu() is now a no-op. Keep its exact filtering
        # rules rather than duplicating placeholders, vocab or discard semantics.
        return parser(host, vocab_size, discard, logprobs_tensors=None)
