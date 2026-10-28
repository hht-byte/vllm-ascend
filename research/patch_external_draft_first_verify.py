"""Layer first-verification metadata onto the existing external-draft patch.

Import this module after ``patch_external_draft`` in each NPU worker process.
The original patch still owns proposer registration and draft transport.
"""

from __future__ import annotations

from vllm_ascend.spec_decode.external_draft_proposer import (
    EXTERNAL_DRAFT_ATTR,
    EXTERNAL_DRAFT_FIRST_VERIFY_ATTR,
    EXTERNAL_DRAFT_FIRST_VERIFY_KEY,
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
    if first_verify:
        if not extra.get("ascend_external_draft"):
            raise ValueError("first-verify candidate must be nonempty")
        policy = extra.get(EXTERNAL_DRAFT_POLICY_KEY, "ngram_1")
        if policy not in _POLICIES:
            raise ValueError(f"invalid external draft rejoin policy: {policy}")
    req_index = _original_add_request(self, request, *args, **kwargs)
    if not first_verify:
        return req_index
    if request.req_id not in getattr(self, EXTERNAL_DRAFT_ATTR, {}):
        raise ValueError("first-verify candidate missing from base draft transport")
    if not hasattr(self, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR):
        setattr(self, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR, set())
    if not hasattr(self, EXTERNAL_DRAFT_POLICIES_ATTR):
        setattr(self, EXTERNAL_DRAFT_POLICIES_ATTR, {})
    getattr(self, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR).add(request.req_id)
    getattr(self, EXTERNAL_DRAFT_POLICIES_ATTR)[request.req_id] = policy
    return req_index


def _patched_remove_request(self, req_id, *args, **kwargs):
    first_verify = getattr(self, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR, None)
    if first_verify is not None:
        first_verify.discard(req_id)
    policies = getattr(self, EXTERNAL_DRAFT_POLICIES_ATTR, None)
    if policies is not None:
        policies.pop(req_id, None)
    return _original_remove_request(self, req_id, *args, **kwargs)


NPUInputBatch.add_request = _patched_add_request
NPUInputBatch.remove_request = _patched_remove_request
