# SPDX-License-Identifier: Apache-2.0
"""Optional, idempotent worker hooks; import after the first-verify patch.

The installed target runner, sampler, rejection sampler and paged KV are not
replaced. Without additional_config.asr_mtp, every original method is retained.
"""
import inspect

from vllm_ascend.spec_decode.external_draft_proposer import AscendExternalDraftProposer
from vllm_ascend.spec_decode.asr_mtp_worker import request_enabled, load_worker
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

EXTERNAL_CLASS = ('vllm_ascend.spec_decode.external_draft_proposer.'
                  'AscendExternalDraftProposer')


class AscendExternalDraftMTPProposer(AscendExternalDraftProposer):
    def __init__(self, config, runner):
        super().__init__(config, runner)
        self.settings = config.additional_config['asr_mtp']
        self.online = None

    def load_model(self, target, *args, **kwargs):
        self.online = load_worker(self.runner, target, self.settings)

    def dummy_run(self, *args, is_profile=False, **kwargs):
        # Target graphs stay unchanged; auxiliary heads run outside capture.
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
    expected = ('self', 'valid_sampled_token_ids', 'sampling_metadata', 'scheduler_output',
                'spec_decode_metadata', 'spec_decode_common_attn_metadata', 'positions',
                'num_scheduled_tokens', 'hidden_states', 'aux_hidden_states',
                'sample_hidden_states', 'target_model_batch_desc')
    if tuple(inspect.signature(original_propose).parameters) != expected:
        raise RuntimeError('ASR MTP patch requires the supported v0.23 proposal dispatch API')

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

    cls._get_drafter = get_drafter
    cls.propose_draft_token_ids = propose
    cls._update_states = update
    cls._asr_mtp_patch_installed = True


install()
