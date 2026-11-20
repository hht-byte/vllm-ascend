"""Install first-verification metadata after the base draft transport."""

from functools import wraps


def install(runner_cls=None):
    """Layer request metadata onto the initialized batch class, idempotently.

    ``runner_cls`` is accepted for the same late-install interface as the
    base patch; this layer only needs the input batch and never imports a
    model runner.
    """
    from vllm_ascend.patch.worker.patch_external_draft import (
        EXTERNAL_DRAFT_REQUESTS_ATTR,
        EXTERNAL_DRAFT_RESUME_ATTR,
    )
    from vllm_ascend.spec_decode.external_draft_proposer import (
        EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR,
        EXTERNAL_DRAFT_ATTR,
        EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR,
        EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_KEY,
        EXTERNAL_DRAFT_FIRST_VERIFY_ATTR,
        EXTERNAL_DRAFT_FIRST_VERIFY_KEY,
        EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR,
        EXTERNAL_DRAFT_POLICIES_ATTR,
        EXTERNAL_DRAFT_POLICY_KEY,
    )
    from vllm_ascend.spec_decode.external_draft_rejoin import POLICIES
    from vllm_ascend.worker.npu_input_batch import NPUInputBatch

    if getattr(NPUInputBatch, "_external_draft_first_verify_installed", False):
        return
    original_add = NPUInputBatch.add_request
    original_remove = NPUInputBatch.remove_request

    @wraps(original_add)
    def add_request(self, request, *args, **kwargs):
        params = getattr(request, "sampling_params", None)
        extra = getattr(params, "extra_args", None) or {}
        first_verify = extra.get(EXTERNAL_DRAFT_FIRST_VERIFY_KEY, False)
        if type(first_verify) is not bool:
            raise TypeError("first-verify marker must be a bool")
        first_mismatch_ngram = extra.get(EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_KEY, False)
        if type(first_mismatch_ngram) is not bool:
            raise TypeError("first-mismatch ngram marker must be a bool")
        if first_mismatch_ngram and not first_verify:
            raise ValueError("first-mismatch ngram requires first verification")
        if first_verify:
            if not extra.get("ascend_external_draft"):
                raise ValueError("first-verify candidate must be nonempty")
            policy = extra.get(EXTERNAL_DRAFT_POLICY_KEY, "ngram_1")
            if policy not in POLICIES:
                raise ValueError(f"invalid external draft rejoin policy: {policy}")
            if first_mismatch_ngram and policy == "aligned":
                raise ValueError("first-mismatch ngram requires an ngram policy")
        req_index = original_add(self, request, *args, **kwargs)
        if not first_verify:
            return req_index
        if request.req_id not in getattr(self, EXTERNAL_DRAFT_ATTR, {}):
            raise ValueError("first-verify candidate missing from base draft transport")
        for attr, factory in (
            (EXTERNAL_DRAFT_FIRST_VERIFY_ATTR, set),
            (EXTERNAL_DRAFT_POLICIES_ATTR, dict),
            (EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR, dict),
            (EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR, set),
            (EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR, set),
        ):
            if not hasattr(self, attr):
                setattr(self, attr, factory())
        getattr(self, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR).add(request.req_id)
        resume = getattr(request, EXTERNAL_DRAFT_RESUME_ATTR, None) or {}
        getattr(self, EXTERNAL_DRAFT_POLICIES_ATTR)[request.req_id] = resume.get(EXTERNAL_DRAFT_POLICIES_ATTR, policy)
        previous = resume.get(EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR)
        if previous is not None:
            getattr(self, EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR)[request.req_id] = previous
        if resume.get(EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR, False):
            getattr(self, EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR).add(request.req_id)
        if resume.get(EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR, first_mismatch_ngram):
            getattr(self, EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR).add(request.req_id)
        return req_index

    @wraps(original_remove)
    def remove_request(self, req_id, *args, **kwargs):
        owner = getattr(self, EXTERNAL_DRAFT_REQUESTS_ATTR, {}).get(req_id)
        if owner is not None and req_id in getattr(self, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR, set()):
            # The base wrapper saves draft consumption after this layer has
            # cleared its maps; preserve continuation state before clearing.
            resume = dict(getattr(owner, EXTERNAL_DRAFT_RESUME_ATTR, None) or {})
            for attr in (EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR, EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR):
                resume[attr] = req_id in getattr(self, attr, set())
            for attr in (EXTERNAL_DRAFT_POLICIES_ATTR, EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR):
                resume[attr] = getattr(self, attr, {}).get(req_id)
            setattr(owner, EXTERNAL_DRAFT_RESUME_ATTR, resume)
        for attr in (
            EXTERNAL_DRAFT_FIRST_VERIFY_ATTR,
            EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR,
            EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR,
        ):
            state = getattr(self, attr, None)
            if state is not None:
                state.discard(req_id)
        for attr in (EXTERNAL_DRAFT_POLICIES_ATTR, EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR):
            state = getattr(self, attr, None)
            if state is not None:
                state.pop(req_id, None)
        return original_remove(self, req_id, *args, **kwargs)

    NPUInputBatch.add_request = add_request
    NPUInputBatch.remove_request = remove_request
    NPUInputBatch._external_draft_first_verify_installed = True
