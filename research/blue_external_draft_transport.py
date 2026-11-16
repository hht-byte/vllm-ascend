# SPDX-License-Identifier: Apache-2.0
"""Install validated external-draft metadata on worker-local InputBatch rows.

Installed as ``vllm_ascend.patch.worker.patch_external_draft`` before the
existing first-verify transport hook. No runner import occurs here.

CachedRequestState owns candidate consumption and retry history. InputBatch
only references the live row's state, so temporary unscheduling or a batch
rebuild cannot restart a proposal. Finished/aborted request disposal releases
the holder; a new request object with a reused ID starts a new lifetime.
"""
from functools import wraps
from inspect import signature

from vllm_ascend.spec_decode.external_draft_proposer import (
    EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR,
    EXTERNAL_DRAFT_ATTR,
    EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR,
    EXTERNAL_DRAFT_FIRST_VERIFY_ATTR,
    EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR,
    EXTERNAL_DRAFT_POLICIES_ATTR,
    EXTERNAL_DRAFT_STATE_ATTR,
    EXTERNAL_DRAFT_STATES_ATTR,
    REQUEST_STATE_ATTRS,
    request_metadata,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch


def install(cls=NPUInputBatch):
    if getattr(cls, '_blue_external_draft_transport_installed', False):
        return
    original_init = cls.__init__
    init_signature = signature(original_init)
    if 'num_speculative_tokens' not in init_signature.parameters:
        raise RuntimeError('external drafts require the v0.23 InputBatch speculative token bound')
    original_add = cls.add_request
    original_remove = cls.remove_request

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        bound = init_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        k = bound.arguments['num_speculative_tokens']
        if type(k) is not int or k < 0:
            raise ValueError('InputBatch speculative token bound must be nonnegative')
        original_init(self, *args, **kwargs)
        # initialize_kv_cache can replace the input batch after drafter creation.
        # Capture each new batch's own bound rather than relying on the drafter.
        self._external_draft_kmax = k

    @wraps(original_add)
    def add(self, request, *args, **kwargs):
        metadata = request_metadata(self, request)
        draft, first, policy, ngram = metadata
        state = getattr(request, EXTERNAL_DRAFT_STATE_ATTR, None)
        if state is not None and state['metadata'] != metadata:
            raise ValueError('external draft metadata changed within a cached request lifetime')
        index = original_add(self, request, *args, **kwargs)
        if state is None and draft:
            state = {'metadata': metadata, 'remaining': draft, 'last': None, 'blocked': False}
            setattr(request, EXTERNAL_DRAFT_STATE_ATTR, state)
        defaults = {EXTERNAL_DRAFT_ATTR: dict, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR: set,
                    EXTERNAL_DRAFT_POLICIES_ATTR: dict, EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR: dict,
                    EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR: set,
                    EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR: set,
                    EXTERNAL_DRAFT_STATES_ATTR: dict}
        for name, factory in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, factory())
            row_values = getattr(self, name)
            if isinstance(row_values, dict):
                row_values.pop(request.req_id, None)
            else:
                row_values.discard(request.req_id)
        if state is None:
            return index
        getattr(self, EXTERNAL_DRAFT_STATES_ATTR)[request.req_id] = state
        if state['remaining']:
            getattr(self, EXTERNAL_DRAFT_ATTR)[request.req_id] = state['remaining']
        if first:
            getattr(self, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR).add(request.req_id)
            getattr(self, EXTERNAL_DRAFT_POLICIES_ATTR)[request.req_id] = policy
        if ngram:
            getattr(self, EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR).add(request.req_id)
        if state['last'] is not None:
            getattr(self, EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR)[request.req_id] = state['last']
        if state['blocked']:
            getattr(self, EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR).add(request.req_id)
        return index

    @wraps(original_remove)
    def remove(self, req_id, *args, **kwargs):
        index = original_remove(self, req_id, *args, **kwargs)
        for name in REQUEST_STATE_ATTRS:
            state = getattr(self, name, None)
            if isinstance(state, dict):
                state.pop(req_id, None)
            elif state is not None:
                state.discard(req_id)
        return index

    cls.__init__ = initialize
    cls.add_request = add
    cls.remove_request = remove
    cls._blue_external_draft_transport_installed = True


install()
