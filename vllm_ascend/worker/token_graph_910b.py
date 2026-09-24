# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental single-device 910B FULL graphs keyed only by total tokens."""

import inspect

import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend._310p.token_graph_dispatcher import TokenGraphDispatcher310
from vllm_ascend.attention.token_graph_910b import PAGraphArena
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.utils import copy_snapshot_to_gpu


class TokenGraphRunner910B(NPUModelRunner):
    def __init__(self, vllm_config, device):
        config = vllm_config
        if "910B" not in torch.npu.get_device_name(device):
            raise ValueError("token_graph_910b requires Ascend910B")
        if config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL:
            raise ValueError("token_graph_910b requires FULL graphs")
        if config.scheduler_config.async_scheduling:
            raise ValueError("token_graph_910b requires async_scheduling=False")
        for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                     "prefill_context_parallel_size", "decode_context_parallel_size"):
            if getattr(config.parallel_config, name, 1) != 1:
                raise ValueError(f"token_graph_910b requires {name}=1")
        if config.lora_config or config.kv_transfer_config:
            raise ValueError("token_graph_910b does not support LoRA or KV transfer")
        spec = config.speculative_config
        if spec is not None and spec.method not in ("ngram", "custom_class"):
            raise ValueError("token_graph_910b supports ngram/custom_class speculation only")
        model = config.model_config
        if model.use_mla or model.is_encoder_decoder or model.runner_type != "generate":
            raise ValueError("token_graph_910b requires a causal decoder with ordinary attention")
        if config.cache_config.cache_dtype not in ("auto", "float16", "bfloat16"):
            raise ValueError("token_graph_910b requires floating-point KV cache")
        if (config.additional_config or {}).get("token_graph_310p", {}).get("enabled", False):
            raise ValueError("310P and 910B token graphs cannot both be enabled")
        super().__init__(vllm_config, device)
        if self.enable_enpu:
            raise ValueError("token_graph_910b does not support ENPU")
        sizes = self.compilation_config.cudagraph_capture_sizes
        if not sizes or min(sizes) < 1 or max(sizes) > self.max_model_len:
            raise ValueError("token_graph_910b requires positive buckets within max_model_len")
        self.token_graph_910b_enabled = True
        self._token_graph_arena = None
        self._token_graph_active = False
        self._token_graph_warmup = False
        self._token_graph_dummy = False
        self._pa_update_stream = torch.npu.Stream(device=device)
        self.cudagraph_dispatcher = TokenGraphDispatcher310(vllm_config)

    def _check_and_update_cudagraph_mode(self, attention_backends, kv_cache_groups):
        # Protect small buckets against K+1 rounding without changing the draft
        # count or the custom speculative class used for actual execution.
        previous = self.uniform_decode_query_len
        self.uniform_decode_query_len = 1
        try:
            super()._check_and_update_cudagraph_mode(attention_backends, kv_cache_groups)
        finally:
            self.uniform_decode_query_len = previous

    def _determine_batch_execution_and_padding(
        self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens,
        use_cascade_attn, allow_microbatching=False, force_eager=False, force_uniform_decode=None,
        force_has_lora=None, force_num_active_loras=None, num_encoder_reqs=0,
    ):
        result = super()._determine_batch_execution_and_padding(
            num_tokens=num_tokens, num_reqs=num_reqs, num_scheduled_tokens_np=num_scheduled_tokens_np,
            max_num_scheduled_tokens=max_num_scheduled_tokens, use_cascade_attn=use_cascade_attn,
            allow_microbatching=False, force_eager=force_eager, force_uniform_decode=False,
            force_has_lora=force_has_lora, force_num_active_loras=force_num_active_loras,
            num_encoder_reqs=num_encoder_reqs,
        )
        self._token_graph_active = result[0] == CUDAGraphMode.FULL
        return result

    def _dummy_run(self, *args, **kwargs):
        values = inspect.signature(NPUModelRunner._dummy_run).bind(self, *args, **kwargs)
        values.apply_defaults()
        values = values.arguments
        self._token_graph_dummy = True
        self._token_graph_warmup = (values["force_attention"] and not values["is_profile"]
                                    and values["num_tokens"] in self.compilation_config.cudagraph_capture_sizes)
        try:
            return super()._dummy_run(*args, **kwargs)
        finally:
            self._token_graph_dummy = self._token_graph_warmup = self._token_graph_active = False

    def _pad_query_start_loc_for_fia(self, query_start_loc, num_tokens_padded, num_reqs_padded,
                                    num_reqs, cudagraph_runtime_mode=None, batch_desc_num_reqs=None):
        if cudagraph_runtime_mode == CUDAGraphMode.FULL or self._token_graph_warmup:
            copy_snapshot_to_gpu(query_start_loc)
            return num_reqs
        return super()._pad_query_start_loc_for_fia(
            query_start_loc, num_tokens_padded, num_reqs_padded, num_reqs,
            cudagraph_runtime_mode, batch_desc_num_reqs,
        )

    def _build_attention_metadata(self, *args, **kwargs):
        active = self._token_graph_active or self._token_graph_warmup
        if active:
            torch.npu.synchronize()
        result = super()._build_attention_metadata(*args, **kwargs)
        if not active:
            return result
        metadata, _ = result
        groups = self.kv_cache_config.kv_cache_groups
        if len(groups) != 1 or type(groups[0].kv_cache_spec) is not FullAttentionSpec:
            raise ValueError("token_graph_910b requires one ordinary full-attention KV group")
        if groups[0].kv_cache_spec.sliding_window is not None:
            raise ValueError("token_graph_910b does not support sliding windows")
        if not isinstance(metadata, dict) or len({id(value) for value in metadata.values()}) != 1:
            raise ValueError("token_graph_910b requires homogeneous attention metadata")
        reqs = kwargs["num_reqs"]
        bucket = kwargs.get("num_tokens_padded") or kwargs["num_tokens"]
        qlens = self.query_start_loc.cpu[:reqs + 1].diff().tolist()
        contexts = self.optimistic_seq_lens_cpu[:reqs].tolist()
        if self._token_graph_dummy:
            # Mainline PA dummy lengths can exceed a small model's context.
            # These graphs refresh workspace and lengths on every replay.
            contexts = qlens
        table = self.input_batch.block_table[0]
        blocks = table.get_device_tensor()
        if self._token_graph_arena is None:
            self._token_graph_arena = PAGraphArena(
                max(self.compilation_config.cudagraph_capture_sizes), self.max_num_reqs,
                blocks.shape[1], self.max_model_len, self.device,
            )
        state = self._token_graph_arena.prepare(bucket, qlens, contexts, blocks, table.slot_mapping.gpu)
        if self._token_graph_dummy:
            state.arena.slots[:bucket].fill_(-1)
        state.attach(next(iter(metadata.values())))
        return result

    def _model_forward(self, num_tokens_padded, input_ids=None, positions=None,
                       intermediate_tensors=None, inputs_embeds=None, **model_kwargs):
        context = get_forward_context()
        if context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            return super()._model_forward(num_tokens_padded, input_ids, positions,
                                          intermediate_tensors, inputs_embeds, **model_kwargs)
        state = self._token_graph_arena.states[num_tokens_padded]
        if not self._token_graph_dummy and not state.tasks:
            raise RuntimeError("910B token graph was not captured at startup")
        torch.npu.synchronize()
        if state.tasks:
            state.update(torch_npu, self._pa_update_stream)
        # The wrapper replays KV writes and PA in their captured order. Bypass
        # mainline FIA/PA parameter updates: these tasks are owned by this state.
        return self.model(input_ids=input_ids, positions=positions, intermediate_tensors=intermediate_tensors,
                          inputs_embeds=inputs_embeds, **model_kwargs)
