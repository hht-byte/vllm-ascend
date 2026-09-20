# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import asdict, dataclass, replace

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher


@dataclass(frozen=True)
class NativeBatchDescriptor310(BatchDescriptor):
    # Additional identity only; request counts and qLens are never graph keys.
    attention_family: str = "decode"


def native_batch_descriptor(descriptor, family):
    return NativeBatchDescriptor310(**asdict(descriptor), attention_family=family)


class TokenGraphDispatcher310(CudagraphDispatcher):
    """Use the upstream FULL bucket machinery with request-independent keys."""

    attention_family = "token"

    def initialize_cudagraph_keys(self, cudagraph_mode, uniform_decode_query_len=1):
        self.attention_family = "token"
        super().initialize_cudagraph_keys(cudagraph_mode, uniform_decode_query_len)
        config = (getattr(self.vllm_config, "additional_config", None) or {}).get("token_graph_310p", {})
        if config.get("phase_routing", False):
            for descriptor in tuple(self.cudagraph_keys[CUDAGraphMode.FULL]):
                for family in ("prefill", "decode"):
                    self.add_cudagraph_key(CUDAGraphMode.FULL, native_batch_descriptor(descriptor, family))

    def _create_padded_batch_descriptor(self, num_tokens, uniform_decode, has_lora, num_active_loras=0):
        descriptor = super()._create_padded_batch_descriptor(num_tokens, False, has_lora, num_active_loras)
        descriptor = replace(descriptor, num_reqs=None, uniform=False)
        if self.attention_family != "token":
            return native_batch_descriptor(descriptor, self.attention_family)
        return descriptor
