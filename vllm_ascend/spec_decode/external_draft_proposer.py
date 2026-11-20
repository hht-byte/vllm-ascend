"""Request-local external drafts, including confirmed-token rejoining.

This backend inherits the Ascend ngram lifecycle and runner dispatch while
keeping the caller-supplied candidate pool until request removal.
"""

from __future__ import annotations

import logging

import torch

from vllm_ascend.spec_decode.external_draft_rejoin import first_mismatch, next_draft
from vllm_ascend.spec_decode.ngram_proposer import AscendNgramProposer

EXTERNAL_DRAFT_KEY = "ascend_external_draft"
EXTERNAL_DRAFT_ATTR = "_external_draft_tokens"
EXTERNAL_DRAFT_FIRST_VERIFY_KEY = "ascend_external_draft_first_verify"
EXTERNAL_DRAFT_POLICY_KEY = "ascend_external_draft_rejoin"
EXTERNAL_DRAFT_FIRST_VERIFY_ATTR = "_external_draft_first_verify"
EXTERNAL_DRAFT_POLICIES_ATTR = "_external_draft_policies"

logger = logging.getLogger(__name__)
EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR = "_external_draft_last_proposal"
EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR = "_external_draft_aligned_blocked"
EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_KEY = "ascend_external_draft_ngram_after_first_mismatch"
EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR = "_external_draft_first_mismatch_ngram"


class AscendExternalDraftProposer(AscendNgramProposer):
    """Use caller-provided token IDs through the Ascend ngram runner contract."""

    def __init__(self, vllm_config, runner=None):
        # The upstream custom factory first creates a config-only placeholder;
        # NPUModelRunner replaces it with the runner-bound drafter after super.
        # Inherit Ascend dispatch and no-op model lifecycle, but not ngram's
        # lookup buffers or JIT warmup: custom_class has no lookup window,
        # and the runner has not created its input batch at this point.
        config = vllm_config.speculative_config
        if config is None or type(config.num_speculative_tokens) is not int or config.num_speculative_tokens <= 0:
            raise ValueError("external draft requires a positive speculative token budget")
        self.runner = runner
        self.k = config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

    @torch.inference_mode()
    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec=None,
        token_ids_cpu=None,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    ) -> list[list[int]]:
        input_batch = self.runner.input_batch
        outputs: list[list[int]] = [[] for _ in sampled_token_ids]
        external_drafts = getattr(input_batch, EXTERNAL_DRAFT_ATTR, None)
        if not external_drafts:
            return outputs

        first_verify = getattr(input_batch, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR, set())
        policies = getattr(input_batch, EXTERNAL_DRAFT_POLICIES_ATTR, {})
        first_mismatch_ngram = getattr(input_batch, EXTERNAL_DRAFT_FIRST_MISMATCH_NGRAM_ATTR, set())
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                continue
            req_id = input_batch.req_ids[i]
            if req_id in input_batch.spec_decode_unsupported_reqs:
                continue
            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= input_batch.max_model_len:
                continue
            if req_id not in first_verify:
                # Preserve the old rollback/shadow one-shot contract.
                # v0.23 bookkeeping has already appended sampled_ids and
                # advanced num_tokens_no_spec for every request path.
                max_draft_len = min(self.k, input_batch.max_model_len - num_tokens)
                if max_draft_len <= 0:
                    continue
                draft = external_drafts.pop(req_id, [])
                outputs[i] = list(draft[:max_draft_len])
                continue

            candidate = external_drafts.get(req_id, [])
            prompt_len = input_batch.num_prompt_tokens[i]
            # _bookkeeping_sync already wrote sampled_ids and advanced
            # num_tokens_no_spec before calling this proposer. Do not write
            # sampled_ids a second time or include unconfirmed tail slots.
            if num_tokens < prompt_len:
                continue
            confirmed = tuple(int(token) for token in input_batch.token_ids_cpu[i, prompt_len:num_tokens])
            mismatch = first_mismatch(candidate, confirmed)
            last_proposals = getattr(input_batch, EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR, {})
            aligned_blocked = getattr(input_batch, EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR, set())
            previous = last_proposals.get(req_id)
            # The scheduler can verify only a prefix of C[1:] and emit a
            # bonus token. Switch only after a confirmed token differs at a
            # post-first position, regardless of the scheduled draft length.
            if req_id in first_mismatch_ngram and mismatch == 0:
                if first_mismatch(candidate[1:], confirmed[1:]) is not None:
                    aligned_blocked.add(req_id)
            elif previous and previous[0] == "aligned" and len(sampled_ids) <= 1:
                # Preserve the old rollback and non-first-mismatch stop.
                aligned_blocked.add(req_id)
            after_aligned_retry = req_id in first_mismatch_ngram and mismatch == 0 and req_id in aligned_blocked
            strategy = (
                policies.get(req_id, "ngram_1") if after_aligned_retry or mismatch not in (None, 0) else "aligned"
            )
            max_draft_len = min(self.k, input_batch.max_model_len - num_tokens)
            if max_draft_len > 0 and not (strategy == "aligned" and req_id in aligned_blocked):
                outputs[i] = list(
                    next_draft(
                        candidate,
                        confirmed,
                        policy=policies.get(req_id, "ngram_1"),
                        max_draft_tokens=max_draft_len,
                        after_aligned_retry=after_aligned_retry,
                    )
                )
                if (
                    outputs[i]
                    and not after_aligned_retry
                    and previous
                    and len(sampled_ids) <= 1
                    and tuple(outputs[i]) == previous[1]
                ):
                    outputs[i] = []
            if outputs[i]:
                last_proposals[req_id] = (strategy, tuple(outputs[i]))
            logger.debug(
                "EXTERNAL_DRAFT_REJOIN request_id=%s confirmed_tokens=%d "
                "first_mismatch=%s strategy=%s proposed_tokens=%d",
                req_id,
                len(confirmed),
                mismatch,
                strategy,
                len(outputs[i]),
            )
        return outputs
