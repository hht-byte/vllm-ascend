"""Install request-provided drafts after the NPU runner has been imported.

The worker calls ``install(NPUModelRunner)`` at its late initialization
boundary. Importing this module alone never imports or patches the runner.
"""

from functools import wraps

EXTERNAL_CLASS = "vllm_ascend.spec_decode.external_draft_proposer.AscendExternalDraftProposer"
EXTERNAL_DRAFT_REQUESTS_ATTR = "_external_draft_requests"
EXTERNAL_DRAFT_RESUME_ATTR = "_ascend_external_draft_resume_state"


def install(runner_cls):
    """Register the external custom class and its request-local transport once."""
    from vllm_ascend.spec_decode.external_draft_proposer import (
        EXTERNAL_DRAFT_ATTR,
        EXTERNAL_DRAFT_KEY,
        AscendExternalDraftProposer,
    )
    from vllm_ascend.worker.npu_input_batch import NPUInputBatch

    if not getattr(NPUInputBatch, "_external_draft_transport_installed", False):
        original_add = NPUInputBatch.add_request
        original_remove = NPUInputBatch.remove_request

        @wraps(original_add)
        def add_request(self, request, *args, **kwargs):
            params = getattr(request, "sampling_params", None)
            extra = getattr(params, "extra_args", None) or {}
            draft = extra.get(EXTERNAL_DRAFT_KEY)
            if draft is not None:
                if not isinstance(draft, (list, tuple)) or any(type(token) is not int or token < 0 for token in draft):
                    raise ValueError("external draft must contain nonnegative integer token IDs")
                draft = list(draft)
            resume = getattr(request, EXTERNAL_DRAFT_RESUME_ATTR, None)
            if resume is not None:
                # A temporarily evicted row retains its request owner. An
                # absent saved draft means its legacy offer was consumed.
                draft = resume[EXTERNAL_DRAFT_ATTR]
            req_index = original_add(self, request, *args, **kwargs)
            if draft:
                if not hasattr(self, EXTERNAL_DRAFT_ATTR):
                    setattr(self, EXTERNAL_DRAFT_ATTR, {})
                getattr(self, EXTERNAL_DRAFT_ATTR)[request.req_id] = list(draft)
            if draft or resume is not None:
                if not hasattr(self, EXTERNAL_DRAFT_REQUESTS_ATTR):
                    setattr(self, EXTERNAL_DRAFT_REQUESTS_ATTR, {})
                getattr(self, EXTERNAL_DRAFT_REQUESTS_ATTR)[request.req_id] = request
            return req_index

        @wraps(original_remove)
        def remove_request(self, req_id, *args, **kwargs):
            drafts = getattr(self, EXTERNAL_DRAFT_ATTR, None)
            draft = drafts.pop(req_id, None) if drafts is not None else None
            owners = getattr(self, EXTERNAL_DRAFT_REQUESTS_ATTR, {})
            owner = owners.pop(req_id, None)
            if owner is not None:
                # GPUInputBatch removal also occurs during unscheduling and
                # preemption. Persist on CachedRequestState, not batch rows;
                # a new owner reusing this ID starts with a fresh lifetime.
                resume = dict(getattr(owner, EXTERNAL_DRAFT_RESUME_ATTR, None) or {})
                resume[EXTERNAL_DRAFT_ATTR] = tuple(draft) if draft is not None else None
                setattr(owner, EXTERNAL_DRAFT_RESUME_ATTR, resume)
            return original_remove(self, req_id, *args, **kwargs)

        NPUInputBatch.add_request = add_request
        NPUInputBatch.remove_request = remove_request
        NPUInputBatch._external_draft_transport_installed = True

    if getattr(runner_cls, "_external_draft_registration_installed", False):
        return
    original_get_drafter = runner_cls._get_drafter

    @wraps(original_get_drafter)
    def get_drafter(self, *args, **kwargs):
        config = self.speculative_config
        if config is not None and config.method == "custom_class" and config.model == EXTERNAL_CLASS:
            return AscendExternalDraftProposer(self.vllm_config, self)
        return original_get_drafter(self, *args, **kwargs)

    runner_cls._get_drafter = get_drafter
    runner_cls._external_draft_registration_installed = True
