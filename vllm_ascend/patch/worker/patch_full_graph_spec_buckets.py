# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve mixed FULL token buckets for variable-length custom proposals.

Install only after NPUModelRunner is defined. No runner import here: the worker
patch package is itself imported while that runner module is initializing.
"""

import inspect
from functools import wraps


def install_full_graph_spec_buckets(runner_cls):
    if getattr(runner_cls, "_full_graph_spec_buckets_installed", False):
        return
    original = getattr(runner_cls, "_check_and_update_cudagraph_mode", None)
    native_dispatch = getattr(runner_cls, "_determine_batch_execution_and_padding", None)
    if (
        original is None
        or native_dispatch is None
        or tuple(inspect.signature(original).parameters)
        != (
            "self",
            "attention_backends",
            "kv_cache_groups",
        )
    ):
        raise RuntimeError("FULL token buckets require the supported v0.23 graph setup API")

    @wraps(original)
    def setup(self, attention_backends, kv_cache_groups):
        # Specialized runners (including 310P and token PA graphs) own their
        # dispatch restrictions and bucket setup; do not override those policies.
        if type(self)._determine_batch_execution_and_padding is not native_dispatch:
            return original(self, attention_backends, kv_cache_groups)
        spec = self.speculative_config
        if spec is None or spec.method != "custom_class" or self.compilation_config.cudagraph_mode.name != "FULL":
            return original(self, attention_backends, kv_cache_groups)
        # Keep distributed/SP alignment and cross-rank graph coordination owned
        # by the original setup; this optimization is limited to one device.
        parallel = self.vllm_config.parallel_config
        if any(
            getattr(parallel, name, 1) != 1
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        ):
            return original(self, attention_backends, kv_cache_groups)
        # Requested FULL can be downgraded to uniform decode graphs. Keep the
        # original K+1 safety checks and rounding whenever mixed FULL is unsupported.
        for backends, group in zip(attention_backends, kv_cache_groups):
            if any(
                backend.get_builder_cls().get_cudagraph_support(self.vllm_config, group.kv_cache_spec).name != "ALWAYS"
                for backend in backends
            ):
                return original(self, attention_backends, kv_cache_groups)
        # Like the 310P token graph setup, mixed FULL keys use total tokens,
        # not K+1 per request. Runtime attention metadata retains each real qLen.
        previous = self.uniform_decode_query_len
        self.uniform_decode_query_len = 1
        try:
            return original(self, attention_backends, kv_cache_groups)
        finally:
            self.uniform_decode_query_len = previous

    runner_cls._check_and_update_cudagraph_mode = setup
    runner_cls._full_graph_spec_buckets_installed = True
