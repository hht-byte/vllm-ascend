# SPDX-License-Identifier: Apache-2.0
"""Bounded external drafts for the synchronous vLLM-Ascend v0.23 runner.

Installed as ``vllm_ascend.spec_decode.external_draft_proposer``. Inheriting
the Ascend ngram type reuses its existing runner dispatch; the ngram search
and its initialization buffers are not used. Bookkeeping owns all writes to
the CPU token table before this proposer runs.
"""
from collections.abc import Mapping

from vllm_ascend.spec_decode.ngram_proposer import AscendNgramProposer

try:
    from vllm_ascend.spec_decode.external_draft_rejoin import first_mismatch, next_draft
except ImportError:
    from external_draft_rejoin import first_mismatch, next_draft


EXTERNAL_CLASS = 'vllm_ascend.spec_decode.external_draft_proposer.AscendExternalDraftProposer'
EXTERNAL_DRAFT_KEY = 'ascend_external_draft'
EXTERNAL_DRAFT_ATTR = '_external_draft_tokens'
EXTERNAL_DRAFT_FIRST_VERIFY_KEY = 'ascend_external_draft_first_verify'
EXTERNAL_DRAFT_POLICY_KEY = 'ascend_external_draft_rejoin'
EXTERNAL_DRAFT_FIRST_VERIFY_ATTR = '_external_draft_first_verify'
EXTERNAL_DRAFT_POLICIES_ATTR = '_external_draft_policies'
EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR = '_external_draft_last_proposal'
EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR = '_external_draft_aligned_blocked'
EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_KEY = 'ascend_external_draft_ngram_after_first_mismatch'
EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR = '_external_draft_first_mismatch_ngram'
EXTERNAL_DRAFT_STATES_ATTR = '_external_draft_request_states'
EXTERNAL_DRAFT_STATE_ATTR = '_blue_external_draft_state'
REQUEST_STATE_ATTRS = (
    EXTERNAL_DRAFT_ATTR, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR, EXTERNAL_DRAFT_POLICIES_ATTR,
    EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR, EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR,
    EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR,
    EXTERNAL_DRAFT_STATES_ATTR,
)
POLICIES = frozenset(('aligned', 'ngram_1', 'ngram_2', 'ngram_3'))


def request_metadata(batch, request):
    """Validate before InputBatch allocation and return an immutable snapshot."""
    params = getattr(request, 'sampling_params', None)
    extra = getattr(params, 'extra_args', None)
    if extra is None:
        extra = {}
    if not isinstance(extra, Mapping):
        raise TypeError('external draft extra_args must be a mapping')
    draft = extra.get(EXTERNAL_DRAFT_KEY, ())
    if not isinstance(draft, (list, tuple)):
        raise TypeError('external draft must be a token list or tuple')
    if any(type(token) is not int or token < 0 or token >= batch.vocab_size for token in draft):
        raise ValueError('external draft contains an invalid token ID')
    first = extra.get(EXTERNAL_DRAFT_FIRST_VERIFY_KEY, False)
    ngram = extra.get(EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_KEY, False)
    if type(first) is not bool or type(ngram) is not bool:
        raise TypeError('external draft verification markers must be bools')
    policy = extra.get(EXTERNAL_DRAFT_POLICY_KEY, 'ngram_1')
    if policy not in POLICIES:
        raise ValueError('invalid external draft rejoin policy')
    if first and not draft:
        raise ValueError('first-verify external draft must be nonempty')
    if ngram and (not first or policy == 'aligned'):
        raise ValueError('first-mismatch ngram requires first verification and an ngram policy')
    if draft:
        k = getattr(batch, '_external_draft_kmax', 0)
        if type(k) is not int or k <= 0 or len(draft) > k:
            raise ValueError('external draft exceeds speculative kmax or backend is disabled')
    return tuple(draft), first, policy, ngram


class AscendExternalDraftProposer(AscendNgramProposer):
    def __init__(self, vllm_config, runner=None):
        k = vllm_config.speculative_config.num_speculative_tokens
        if type(k) is not int or k <= 0:
            raise ValueError('external drafts require positive speculative kmax')
        self.runner = runner
        self.k = k
        # GPUModelRunner's parent initializer constructs custom proposers with
        # config only. NPUModelRunner replaces this inert instance with the
        # runner-bound Ascend drafter after its own buffers exist.
        if runner is None:
            return
        if runner.use_async_scheduling:
            raise ValueError('external drafts require async_scheduling=False')
        runner.input_batch._external_draft_kmax = k

    def propose(self, sampled_token_ids, num_tokens_no_spec=None, token_ids_cpu=None,
                slot_mappings=None):
        if self.runner is None:
            raise RuntimeError('external draft proposer has not been bound to an Ascend runner')
        batch = self.runner.input_batch
        if not isinstance(sampled_token_ids, list) or len(sampled_token_ids) != len(batch.req_ids):
            raise TypeError('external drafts require synchronous CPU lists for the full current batch')
        # Validate every row before one-shot candidates or retry state can change.
        for row in sampled_token_ids:
            if not isinstance(row, list):
                raise TypeError('external drafts require CPU sampled token lists')
            if any(type(token) is not int or token < 0 or token >= batch.vocab_size for token in row):
                raise ValueError('sampled rows contain invalid token IDs')
        outputs = [[] for _ in sampled_token_ids]
        candidates = getattr(batch, EXTERNAL_DRAFT_ATTR, {})
        first = getattr(batch, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR, set())
        policies = getattr(batch, EXTERNAL_DRAFT_POLICIES_ATTR, {})
        ngram = getattr(batch, EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR, set())
        previous = getattr(batch, EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR, {})
        blocked = getattr(batch, EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR, set())
        request_states = getattr(batch, EXTERNAL_DRAFT_STATES_ATTR, {})
        for i, sampled in enumerate(sampled_token_ids):
            req_id = batch.req_ids[i]
            if not sampled or req_id in batch.spec_decode_unsupported_reqs:
                continue
            length = int(batch.num_tokens_no_spec[i])
            capacity = min(self.k, batch.max_model_len - length)
            if capacity <= 0:
                continue
            if req_id not in first:
                outputs[i] = list(candidates.pop(req_id, ())[:capacity])
                state = request_states.get(req_id)
                if state is not None:
                    state['remaining'] = ()
                continue
            candidate = candidates.get(req_id, ())
            prompt = int(batch.num_prompt_tokens[i])
            if length < prompt:
                continue
            confirmed = tuple(int(token) for token in batch.token_ids_cpu[i, prompt:length])
            mismatch = first_mismatch(candidate, confirmed)
            last = previous.get(req_id)
            if req_id in ngram and mismatch == 0:
                if first_mismatch(candidate[1:], confirmed[1:]) is not None:
                    blocked.add(req_id)
            elif last and last[0] == 'aligned' and len(sampled) <= 1:
                blocked.add(req_id)
            state = request_states.get(req_id)
            if state is not None:
                state['blocked'] = req_id in blocked
            after_retry = req_id in ngram and mismatch == 0 and req_id in blocked
            strategy = policies.get(req_id, 'ngram_1') if after_retry or mismatch not in (None, 0) else 'aligned'
            if strategy == 'aligned' and req_id in blocked:
                continue
            outputs[i] = list(next_draft(candidate, confirmed,
                policy=policies.get(req_id, 'ngram_1'), max_draft_tokens=capacity,
                after_aligned_retry=after_retry))
            if (outputs[i] and not after_retry and last and len(sampled) <= 1
                    and tuple(outputs[i]) == last[1]):
                outputs[i] = []
            if outputs[i]:
                previous[req_id] = (strategy, tuple(outputs[i]))
                if state is not None:
                    state['last'] = previous[req_id]
        return outputs
