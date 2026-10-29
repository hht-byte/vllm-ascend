"""Layer first-verification metadata onto the existing external-draft patch.

Import this module after ``patch_external_draft`` in each NPU worker process.
The original patch still owns proposer registration and draft transport.
"""

from __future__ import annotations

from vllm_ascend.spec_decode.external_draft_proposer import (
    EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR,
    EXTERNAL_DRAFT_ATTR,
    EXTERNAL_DRAFT_FIRST_VERIFY_ATTR,
    EXTERNAL_DRAFT_FIRST_VERIFY_KEY,
    EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR,
    EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_KEY,
    EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR,
    EXTERNAL_DRAFT_POLICIES_ATTR,
    EXTERNAL_DRAFT_POLICY_KEY,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch


_original_add_request = NPUInputBatch.add_request
_original_remove_request = NPUInputBatch.remove_request
_POLICIES = frozenset(("aligned", "ngram_1", "ngram_2", "ngram_3"))


def _patched_add_request(self, request, *args, **kwargs):
    params = getattr(request, "sampling_params", None)
    extra = getattr(params, "extra_args", None) or {}
    first_verify = extra.get(EXTERNAL_DRAFT_FIRST_VERIFY_KEY, False)
    first_mismatch_ngram = extra.get(EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_KEY, False)
    if type(first_mismatch_ngram) is not bool:
        raise TypeError("first-mismatch ngram marker must be a bool")
    if first_mismatch_ngram and not first_verify:
        raise ValueError("first-mismatch ngram requires first verification")
    if first_verify:
        if not extra.get("ascend_external_draft"):
            raise ValueError("first-verify candidate must be nonempty")
        policy = extra.get(EXTERNAL_DRAFT_POLICY_KEY, "ngram_1")
        if policy not in _POLICIES:
            raise ValueError(f"invalid external draft rejoin policy: {policy}")
        if first_mismatch_ngram and policy == "aligned":
            raise ValueError("first-mismatch ngram requires an ngram policy")
    req_index = _original_add_request(self, request, *args, **kwargs)
    if not first_verify:
        return req_index
    if request.req_id not in getattr(self, EXTERNAL_DRAFT_ATTR, {}):
        raise ValueError("first-verify candidate missing from base draft transport")
    if not hasattr(self, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR):
        setattr(self, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR, set())
    if not hasattr(self, EXTERNAL_DRAFT_POLICIES_ATTR):
        setattr(self, EXTERNAL_DRAFT_POLICIES_ATTR, {})
    if not hasattr(self, EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR):
        setattr(self, EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR, {})
    if not hasattr(self, EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR):
        setattr(self, EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR, set())
    if not hasattr(self, EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR):
        setattr(self, EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR, set())
    getattr(self, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR).add(request.req_id)
    getattr(self, EXTERNAL_DRAFT_POLICIES_ATTR)[request.req_id] = policy
    if first_mismatch_ngram:
        getattr(self, EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR).add(request.req_id)
    return req_index


def _patched_remove_request(self, req_id, *args, **kwargs):
    first_verify = getattr(self, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR, None)
    if first_verify is not None:
        first_verify.discard(req_id)
    policies = getattr(self, EXTERNAL_DRAFT_POLICIES_ATTR, None)
    if policies is not None:
        policies.pop(req_id, None)
    last_proposals = getattr(self, EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR, None)
    if last_proposals is not None:
        last_proposals.pop(req_id, None)
    aligned_blocked = getattr(self, EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR, None)
    if aligned_blocked is not None:
        aligned_blocked.discard(req_id)
    first_mismatch_ngram = getattr(self, EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR, None)
    if first_mismatch_ngram is not None:
        first_mismatch_ngram.discard(req_id)
    return _original_remove_request(self, req_id, *args, **kwargs)


NPUInputBatch.add_request = _patched_add_request
NPUInputBatch.remove_request = _patched_remove_request
