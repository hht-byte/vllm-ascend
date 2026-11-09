# SPDX-License-Identifier: Apache-2.0
"""Optional worker hooks; import from worker.py AFTER its NPUModelRunner import.

Do not import here from patch/worker/__init__.py: model_runner_v1 imports that
package before defining NPUModelRunner, causing a circular import.

The installed target runner, sampler, rejection sampler and paged KV are not
replaced. Without additional_config.asr_mtp, every original method is retained.
"""
import inspect
import logging

from vllm_ascend.spec_decode.asr_mtp_worker import load_worker, request_enabled
from vllm_ascend.spec_decode.external_draft_proposer import AscendExternalDraftProposer
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

EXTERNAL_CLASS = ('vllm_ascend.spec_decode.external_draft_proposer.'
                  'AscendExternalDraftProposer')
logger = logging.getLogger(__name__)


class AscendExternalDraftMTPProposer(AscendExternalDraftProposer):
    def __init__(self, config, runner):
        super().__init__(config, runner)
        self.settings = config.additional_config['asr_mtp']
        self.online = None

    def load_model(self, target, *args, **kwargs):
        self.online = load_worker(self.runner, target, self.settings)

    def dummy_run(self, *args, is_profile=False, **kwargs):
        # Independent MTP graphs are prepared once during memory profiling,
        # before target KV sizing. Never capture inside target graph dummy runs.
        if is_profile and self.online is not None:
            self.online.profile()

    def propose_engine(self, sampled, scheduler_output, hidden):
        if not isinstance(sampled, list) or len(sampled) != len(self.runner.input_batch.req_ids):
            raise RuntimeError('ASR MTP requires synchronous CPU sampled token lists')
        enabled = [request_enabled(self.runner.requests[req_id].sampling_params)
                   for req_id in self.runner.input_batch.req_ids]
        # Formal first-verify rows already consumed their initial draft. Legacy
        # MTP shadow rows submit theirs once inside online.propose before MTP.
        # Opt-out shadow/formal rows preserve the original proposer behavior.
        outputs = super().propose([[] if active else row for active, row in zip(enabled, sampled)])
        if any(enabled):
            if self.online is None:
                raise RuntimeError('ASR MTP weights were not loaded by the worker')
            mtp = self.online.propose(sampled, scheduler_output, hidden)
            for i, active in enumerate(enabled):
                if active:
                    outputs[i] = mtp[i]
        return outputs


def install():
    cls = NPUModelRunner
    if getattr(cls, '_asr_mtp_patch_installed', False):
        return
    original_get = cls._get_drafter
    original_propose = cls.propose_draft_token_ids
    original_update = cls._update_states
    original_graph_setup = getattr(cls, '_check_and_update_cudagraph_mode', None)
    expected = ('self', 'valid_sampled_token_ids', 'sampling_metadata', 'scheduler_output',
                'spec_decode_metadata', 'spec_decode_common_attn_metadata', 'positions',
                'num_scheduled_tokens', 'hidden_states', 'aux_hidden_states',
                'sample_hidden_states', 'target_model_batch_desc')
    if tuple(inspect.signature(original_propose).parameters) != expected:
        raise RuntimeError('ASR MTP patch requires the supported v0.23 proposal dispatch API')
    if (original_graph_setup is None or tuple(inspect.signature(original_graph_setup).parameters) !=
            ('self', 'attention_backends', 'kv_cache_groups')):
        raise RuntimeError('ASR MTP patch requires the supported v0.23 graph setup API')

    def get_drafter(self):
        additional = self.vllm_config.additional_config
        if isinstance(additional, dict) and 'asr_mtp' in additional:
            config = self.speculative_config
            if config.method != 'custom_class' or config.model != EXTERNAL_CLASS:
                raise ValueError('asr_mtp requires the existing external-draft custom_class config')
            return AscendExternalDraftMTPProposer(self.vllm_config, self)
        return original_get(self)

    def propose(self, valid_sampled_token_ids, sampling_metadata, scheduler_output,
                spec_decode_metadata, spec_decode_common_attn_metadata, positions,
                num_scheduled_tokens, hidden_states, aux_hidden_states=None,
                sample_hidden_states=None, target_model_batch_desc=None):
        if isinstance(self.drafter, AscendExternalDraftMTPProposer):
            return self.drafter.propose_engine(valid_sampled_token_ids, scheduler_output, hidden_states)
        return original_propose(
            self, valid_sampled_token_ids, sampling_metadata, scheduler_output,
            spec_decode_metadata, spec_decode_common_attn_metadata, positions,
            num_scheduled_tokens, hidden_states, aux_hidden_states,
            sample_hidden_states, target_model_batch_desc)

    def update(self, scheduler_output):
        drafter = self.drafter
        if isinstance(drafter, AscendExternalDraftMTPProposer) and drafter.online is not None:
            # Before original_update: abort+resubmit of the same ID is a new
            # lifetime. Batch removal for mere unscheduling must not evict state.
            drafter.online.finish(scheduler_output.finished_req_ids)
        return original_update(self, scheduler_output)

    def graph_setup(self, attention_backends, kv_cache_groups):
        if (not isinstance(self.drafter, AscendExternalDraftMTPProposer) or
                self.compilation_config.cudagraph_mode.name != 'FULL'):
            return original_graph_setup(self, attention_backends, kv_cache_groups)
        # A requested FULL mode can resolve to uniform decode-only graphs.
        # Preserve upstream capability checks and K+1 rounding on that path.
        for backends, group in zip(attention_backends, kv_cache_groups):
            if any(backend.get_builder_cls().get_cudagraph_support(
                    self.vllm_config, group.kv_cache_spec).name != 'ALWAYS' for backend in backends):
                return original_graph_setup(self, attention_backends, kv_cache_groups)
        # As in 310P token graph commit 46fa2fb8f: FULL uses mixed total-token
        # buckets, so setup must not round every size to K+1. This overrides
        # only initialization; the actual draft budget and runtime query length
        # are unchanged. Decode-only modes still need their uniform K+1 sizes.
        previous = self.uniform_decode_query_len
        self.uniform_decode_query_len = 1
        try:
            return original_graph_setup(self, attention_backends, kv_cache_groups)
        finally:
            self.uniform_decode_query_len = previous

    cls._get_drafter = get_drafter
    cls.propose_draft_token_ids = propose
    cls._update_states = update
    cls._check_and_update_cudagraph_mode = graph_setup
    cls._asr_mtp_patch_installed = True


def install_worker_memory_hook(cls):
    """Called after NPUWorker is defined; only changes startup KV accounting."""
    if getattr(cls, '_asr_mtp_graph_memory_installed', False):
        return
    original = cls.determine_available_memory
    if tuple(inspect.signature(original).parameters) != ('self',):
        raise RuntimeError('ASR MTP graph memory hook requires the v0.23 worker API')

    def determine(self):
        available = original(self)
        drafter = getattr(self.model_runner, 'drafter', None)
        online = drafter.online if isinstance(drafter, AscendExternalDraftMTPProposer) else None
        graphs = getattr(online, 'graphs', None)
        if graphs is None or not graphs.ready:
            return available
        reserve = graphs.reserved_bytes
        manual = bool(self.cache_config.kv_cache_memory_bytes)
        logger.info('ASR_MTP_GRAPH_MEMORY reserved_mib=%.2f manual_kv=%s',
                    reserve / 1024 ** 2, manual)
        if manual:
            return available  # Explicit user KV bytes remain authoritative.
        if available <= reserve:
            raise ValueError('insufficient automatic KV budget after reserving ASR MTP graphs')
        self.available_kv_cache_memory_bytes = available - reserve
        return self.available_kv_cache_memory_bytes

    cls.determine_available_memory = determine
    cls._asr_mtp_graph_memory_installed = True


install()
