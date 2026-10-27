"""Generate the vLLM 0.23 / vLLM-Ascend first-prefill draft patch.

Run from this repository with paths to the *unmodified* engine checkouts::

    python research/build_external_draft_first_verify_patch.py \
      --vllm-root /path/to/vllm --ascend-root /path/to/vllm-ascend \
      --out research/external_draft_first_verify_v023.patch

The script checks exact anchors and only writes the patch; it never edits the
engine source. Apply the vllm/ and vllm_ascend/ paths separately in their
respective checkouts after reviewing the diff.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one anchor, got {count}")
    return source.replace(old, new, 1)


def replace_last(source: str, old: str, new: str, label: str) -> str:
    index = source.rfind(old)
    if index < 0:
        raise ValueError(f"{label}: anchor missing")
    return source[:index] + new + source[index + len(old):]


def scheduler_common(source: str) -> str:
    marker = "    def add_request(self, request: Request) -> None:\n"
    helper = '''    def _prepare_initial_external_draft(self, request: Request) -> None:
        """Seed first-pass verification before the target is ever executed."""
        params = request.sampling_params
        extra = getattr(params, "extra_args", None) or {}
        if not extra.get("ascend_external_draft_first_verify", False):
            return
        draft = extra.get("ascend_external_draft")
        if (not isinstance(draft, (list, tuple)) or not draft
                or any(type(token) is not int or token < 0 for token in draft)):
            raise ValueError("first-verify external draft must be nonempty token IDs")
        if not self.speculative_config or len(draft) > self.num_spec_tokens:
            raise ValueError("first-verify draft exceeds speculative kmax")
        if request.num_tokens + len(draft) >= self.max_model_len:
            raise ValueError("first-verify draft exceeds model context")
        request.spec_token_ids = list(draft)

'''
    source = replace_once(source, marker, helper + marker, "scheduler helper")
    # New requests only; existing streaming request updates retain their path.
    source = replace_once(
        source,
        "        else:\n            if request.resumable:\n                request.streaming_queue = deque()\n",
        "        else:\n            if request.resumable:\n                request.streaming_queue = deque()\n"
        "            self._prepare_initial_external_draft(request)\n",
        "scheduler add_request",
    )
    return source


def base_scheduler(source: str) -> str:
    source = scheduler_common(source)
    source = replace_once(
        source,
        "                    num_new_tokens = request.num_tokens - num_computed_tokens\n"
        "                    threshold = self.scheduler_config.long_prefill_token_threshold\n",
        "                    num_new_tokens = request.num_tokens - num_computed_tokens\n"
        "                    initial_draft = bool(request.spec_token_ids) and bool(\n"
        "                        (getattr(request.sampling_params, 'extra_args', None) or {})\n"
        "                        .get('ascend_external_draft_first_verify', False))\n"
        "                    threshold = self.scheduler_config.long_prefill_token_threshold\n"
        "                    if initial_draft:\n"
        "                        full = request.num_tokens_with_spec - num_computed_tokens\n"
        "                        if full <= token_budget and (threshold <= 0 or full <= threshold):\n"
        "                            num_new_tokens = full\n"
        "                        else:\n"
        "                            request.spec_token_ids = []\n"
        "                            logger.info('INITIAL_DRAFT request_id=%s status=fallback reason=budget', request.request_id)\n",
        "base waiting token count",
    )
    # Encoder scheduling may shrink num_new_tokens. Never validate a partial
    # external draft as though the full candidate had been verified.
    marker = "                new_blocks = self.kv_cache_manager.allocate_slots(\n"
    guard = '''                if (request.spec_token_ids and
                        (getattr(request.sampling_params, "extra_args", None) or {})
                        .get("ascend_external_draft_first_verify", False) and
                        num_computed_tokens + num_new_tokens < request.num_tokens_with_spec):
                    request.spec_token_ids = []
                    num_new_tokens = min(num_new_tokens, request.num_tokens - num_computed_tokens)
                    logger.info("INITIAL_DRAFT request_id=%s status=fallback reason=encoder_or_chunk", request.request_id)

'''
    source = replace_last(source, marker, guard + marker, "base post-encoder guard")
    marker = "                self.running.append(request)\n"
    schedule = '''                if (request.spec_token_ids and
                        (getattr(request.sampling_params, "extra_args", None) or {})
                        .get("ascend_external_draft_first_verify", False)):
                    assert num_computed_tokens + num_new_tokens == request.num_tokens_with_spec
                    scheduled_spec_decode_tokens[request_id] = request.spec_token_ids
                    request.spec_token_ids = []
                    logger.info("INITIAL_DRAFT request_id=%s status=scheduled draft_tokens=%d",
                                request_id, len(scheduled_spec_decode_tokens[request_id]))

'''
    source = replace_once(source, marker, schedule + marker, "base waiting spec schedule")
    return source


def recompute_scheduler(source: str) -> str:
    source = replace_once(
        source,
        "            if request.resumable:\n                request.streaming_queue = deque()\n"
        "            # Fill in placeholder tokens",
        "            if request.resumable:\n                request.streaming_queue = deque()\n"
        "            self._prepare_initial_external_draft(request)\n"
        "            # Fill in placeholder tokens",
        "recompute add_request",
    )
    source = replace_once(
        source,
        "            if self.is_mtp_kv_consumer and (self.max_model_len >= (request.num_tokens + self.num_spec_tokens)):\n",
        "            if (not request.spec_token_ids and self.is_mtp_kv_consumer\n"
        "                    and self.max_model_len >= request.num_tokens + self.num_spec_tokens):\n",
        "recompute placeholder precedence",
    )
    source = replace_once(
        source,
        "                    if self.is_mtp_kv_consumer:\n"
        "                        num_new_tokens = request.num_tokens_with_spec - num_computed_tokens\n"
        "                    else:\n"
        "                        num_new_tokens = request.num_tokens - num_computed_tokens\n"
        "                    threshold = self.scheduler_config.long_prefill_token_threshold\n",
        "                    initial_draft = bool(request.spec_token_ids) and bool(\n"
        "                        (getattr(request.sampling_params, 'extra_args', None) or {})\n"
        "                        .get('ascend_external_draft_first_verify', False))\n"
        "                    if self.is_mtp_kv_consumer or initial_draft:\n"
        "                        num_new_tokens = request.num_tokens_with_spec - num_computed_tokens\n"
        "                    else:\n"
        "                        num_new_tokens = request.num_tokens - num_computed_tokens\n"
        "                    threshold = self.scheduler_config.long_prefill_token_threshold\n"
        "                    if initial_draft and (num_new_tokens > token_budget or\n"
        "                                          0 < threshold < num_new_tokens):\n"
        "                        request.spec_token_ids = []\n"
        "                        num_new_tokens = request.num_tokens - num_computed_tokens\n"
        "                        logger.info('INITIAL_DRAFT request_id=%s status=fallback reason=budget', request.request_id)\n",
        "recompute waiting token count",
    )
    marker = "                new_blocks = self.kv_cache_manager.allocate_slots(\n"
    guard = '''                if (request.spec_token_ids and
                        (getattr(request.sampling_params, "extra_args", None) or {})
                        .get("ascend_external_draft_first_verify", False) and
                        num_computed_tokens + num_new_tokens < request.num_tokens_with_spec):
                    request.spec_token_ids = []
                    num_new_tokens = min(num_new_tokens, request.num_tokens - num_computed_tokens)
                    logger.info("INITIAL_DRAFT request_id=%s status=fallback reason=encoder_or_chunk", request.request_id)

'''
    source = replace_last(source, marker, guard + marker, "recompute post-encoder guard")
    source = replace_once(
        source,
        "                if self.is_mtp_kv_consumer and request.spec_token_ids:\n",
        "                if (self.is_mtp_kv_consumer or\n"
        "                    (getattr(request.sampling_params, 'extra_args', None) or {})\n"
        "                    .get('ascend_external_draft_first_verify', False)) and request.spec_token_ids:\n",
        "recompute waiting spec schedule",
    )
    source = replace_once(
        source,
        "                    num_scheduled_spec_tokens = (\n"
        "                        num_new_tokens\n"
        "                        + request.num_computed_tokens\n"
        "                        - request.num_tokens\n"
        "                        - request.num_output_placeholders\n"
        "                    )\n",
        "                    num_scheduled_spec_tokens = (\n"
        "                        num_new_tokens\n"
        "                        + num_computed_tokens\n"
        "                        - request.num_tokens\n"
        "                        - request.num_output_placeholders\n"
        "                    )\n",
        "recompute cached-prefix spec length",
    )
    return source


def proposer(source: str) -> str:
    source = replace_once(
        source,
        "import torch\n",
        "import logging\n\nimport torch\n",
        "proposer logging import",
    )
    source = replace_once(
        source,
        "from vllm.v1.spec_decode.ngram_proposer import NgramProposer\n",
        "from vllm.v1.spec_decode.ngram_proposer import NgramProposer\n"
        "from vllm_ascend.spec_decode.external_draft_rejoin import first_mismatch, next_draft\n",
        "proposer import",
    )
    source = replace_once(
        source,
        'EXTERNAL_DRAFT_ATTR = "_external_draft_tokens"\n',
        'EXTERNAL_DRAFT_ATTR = "_external_draft_tokens"\n'
        'EXTERNAL_DRAFT_FIRST_VERIFY_KEY = "ascend_external_draft_first_verify"\n'
        'EXTERNAL_DRAFT_POLICY_KEY = "ascend_external_draft_rejoin"\n'
        'EXTERNAL_DRAFT_FIRST_VERIFY_ATTR = "_external_draft_first_verify"\n'
        'EXTERNAL_DRAFT_POLICIES_ATTR = "_external_draft_policies"\n',
        "proposer constants",
    )
    source = replace_once(
        source,
        'EXTERNAL_DRAFT_POLICIES_ATTR = "_external_draft_policies"\n',
        'EXTERNAL_DRAFT_POLICIES_ATTR = "_external_draft_policies"\n'
        '\nlogger = logging.getLogger(__name__)\n',
        "proposer logger",
    )
    start = source.index("    @torch.inference_mode()\n    def propose(")
    source = source[:start] + '''    @torch.inference_mode()
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
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                continue
            req_id = input_batch.req_ids[i]
            if req_id in input_batch.spec_decode_unsupported_reqs:
                continue
            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= input_batch.max_model_len:
                continue
            end_idx = num_tokens + len(sampled_ids)
            write_end = min(end_idx, input_batch.max_model_len)
            input_batch.token_ids_cpu[i, num_tokens:write_end] = sampled_ids[:write_end - num_tokens]
            remaining = input_batch.max_model_len - end_idx
            max_draft_len = min(self.k, remaining)
            if max_draft_len <= 0:
                continue

            if req_id not in first_verify:
                # Preserve the old rollback/shadow one-shot contract.
                draft = external_drafts.pop(req_id, [])
                outputs[i] = list(draft[:max_draft_len])
                continue

            candidate = external_drafts.get(req_id, [])
            prompt_len = input_batch.num_prompt_tokens[i]
            # token_ids_cpu before end_idx is confirmed target output only;
            # speculative IDs at later positions must not influence lookup.
            if end_idx < prompt_len:
                continue
            confirmed = tuple(int(token) for token in
                              input_batch.token_ids_cpu[i, prompt_len:end_idx])
            outputs[i] = list(next_draft(
                candidate, confirmed, policy=policies.get(req_id, "ngram_1"),
                max_draft_tokens=max_draft_len))
            mismatch = first_mismatch(candidate, confirmed)
            strategy = ("aligned" if mismatch is None or mismatch == 0
                        else policies.get(req_id, "ngram_1"))
            logger.debug("EXTERNAL_DRAFT_REJOIN request_id=%s confirmed_tokens=%d "
                         "first_mismatch=%s strategy=%s proposed_tokens=%d",
                         req_id, len(confirmed), mismatch, strategy, len(outputs[i]))
        return outputs
'''
    return source


def worker_patch(source: str) -> str:
    source = replace_once(
        source,
        "    EXTERNAL_DRAFT_ATTR,\n    EXTERNAL_DRAFT_EXTRA_ARGS_KEY,\n",
        "    EXTERNAL_DRAFT_ATTR,\n    EXTERNAL_DRAFT_EXTRA_ARGS_KEY,\n"
        "    EXTERNAL_DRAFT_FIRST_VERIFY_ATTR,\n    EXTERNAL_DRAFT_FIRST_VERIFY_KEY,\n"
        "    EXTERNAL_DRAFT_POLICIES_ATTR,\n    EXTERNAL_DRAFT_POLICY_KEY,\n",
        "worker imports",
    )
    source = replace_once(
        source,
        "    if not hasattr(self, EXTERNAL_DRAFT_ATTR):\n"
        "        setattr(self, EXTERNAL_DRAFT_ATTR, {})\n",
        "    if not hasattr(self, EXTERNAL_DRAFT_ATTR):\n"
        "        setattr(self, EXTERNAL_DRAFT_ATTR, {})\n"
        "    if not hasattr(self, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR):\n"
        "        setattr(self, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR, set())\n"
        "    if not hasattr(self, EXTERNAL_DRAFT_POLICIES_ATTR):\n"
        "        setattr(self, EXTERNAL_DRAFT_POLICIES_ATTR, {})\n",
        "worker add stores",
    )
    source = replace_once(
        source,
        "        if tokens:\n            drafts[request.req_id] = list(tokens)\n",
        "        if tokens:\n            drafts[request.req_id] = list(tokens)\n"
        "            if extra_args.get(EXTERNAL_DRAFT_FIRST_VERIFY_KEY, False):\n"
        "                getattr(self, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR).add(request.req_id)\n"
        "                policy = extra_args.get(EXTERNAL_DRAFT_POLICY_KEY, 'ngram_1')\n"
        "                if policy not in ('aligned', 'ngram_1', 'ngram_2', 'ngram_3'):\n"
        "                    raise ValueError(f'invalid external draft rejoin policy: {policy}')\n"
        "                getattr(self, EXTERNAL_DRAFT_POLICIES_ATTR)[request.req_id] = policy\n",
        "worker policy transport",
    )
    source = replace_once(
        source,
        "    if drafts is not None:\n        drafts.pop(req_id, None)\n"
        "    return _original_remove_request(self, req_id, *args, **kwargs)\n",
        "    if drafts is not None:\n        drafts.pop(req_id, None)\n"
        "    first_verify = getattr(self, EXTERNAL_DRAFT_FIRST_VERIFY_ATTR, None)\n"
        "    if first_verify is not None:\n        first_verify.discard(req_id)\n"
        "    policies = getattr(self, EXTERNAL_DRAFT_POLICIES_ATTR, None)\n"
        "    if policies is not None:\n        policies.pop(req_id, None)\n"
        "    return _original_remove_request(self, req_id, *args, **kwargs)\n",
        "worker cleanup",
    )
    return source


def model_runner(source: str) -> str:
    source = replace_once(
        source,
        "from vllm_ascend.spec_decode.ngram_proposer import AscendNgramProposer\n",
        "from vllm_ascend.spec_decode.external_draft_proposer import AscendExternalDraftProposer\n"
        "from vllm_ascend.spec_decode.ngram_proposer import AscendNgramProposer\n",
        "model runner import",
    )
    source = replace_once(
        source,
        "        elif isinstance(self.drafter, AscendNgramProposer):\n",
        "        elif isinstance(self.drafter, AscendExternalDraftProposer):\n"
        "            draft_token_ids = self.drafter.propose(valid_sampled_token_ids)\n"
        "        elif isinstance(self.drafter, AscendNgramProposer):\n",
        "model runner proposer dispatch",
    )
    return source


def make_patch(vllm_root: Path, ascend_root: Path, helper: Path) -> str:
    sources = [
        (vllm_root / "vllm/v1/core/sched/scheduler.py",
         "vllm/v1/core/sched/scheduler.py", base_scheduler),
        (ascend_root / "vllm_ascend/core/recompute_scheduler.py",
         "vllm_ascend/core/recompute_scheduler.py", recompute_scheduler),
        (ascend_root / "vllm_ascend/spec_decode/external_draft_proposer.py",
         "vllm_ascend/spec_decode/external_draft_proposer.py", proposer),
        (ascend_root / "vllm_ascend/patch/worker/patch_external_draft.py",
         "vllm_ascend/patch/worker/patch_external_draft.py", worker_patch),
        (ascend_root / "vllm_ascend/worker/model_runner_v1.py",
         "vllm_ascend/worker/model_runner_v1.py", model_runner),
    ]
    chunks = []
    for path, name, transform in sources:
        old = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        new = transform(old)
        chunks.extend(difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile="a/" + name, tofile="b/" + name,
        ))
    name = "vllm_ascend/spec_decode/external_draft_rejoin.py"
    helper_text = helper.read_text(encoding="utf-8")
    chunks.extend(difflib.unified_diff(
        [], helper_text.splitlines(keepends=True),
        fromfile="/dev/null", tofile="b/" + name,
    ))
    # Keep the exported patch clean even when indented snippets contain
    # whitespace-only separator lines.
    return "".join("+\n" if line.startswith("+") and not line[1:].strip()
                   else line for line in chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument("--ascend-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    patch = make_patch(args.vllm_root, args.ascend_root,
                       Path(__file__).with_name("external_draft_rejoin.py"))
    args.out.write_bytes(patch.encode("utf-8"))
    print(f"wrote {args.out} ({len(patch.splitlines())} lines)")


if __name__ == "__main__":
    main()
