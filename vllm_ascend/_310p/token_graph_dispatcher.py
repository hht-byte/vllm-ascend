# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher


class TokenGraphDispatcher310(CudagraphDispatcher):
    """Use the upstream FULL bucket machinery with request-independent keys."""

    def _create_padded_batch_descriptor(self, num_tokens, uniform_decode, has_lora, num_active_loras=0):
        descriptor = super()._create_padded_batch_descriptor(num_tokens, False, has_lora, num_active_loras)
        return replace(descriptor, num_reqs=None, uniform=False)
