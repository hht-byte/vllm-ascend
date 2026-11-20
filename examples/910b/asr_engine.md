# Complete external-draft and ASR MTP engine

This branch now contains the runtime modules for external drafts, first-verify
metadata/rejoining, and the optional ASR MTP extension. The worker installs them
in that order, after `NPUModelRunner` has finished importing. MTP memory hooks
are registered only after `NPUWorker` exists. No source from the streaming
service repository is needed to supply these engine modules.

The existing native FULL and input-preparation integrations are retained.
The delivery branch name `310P-graph` does not imply that the ASR preparation
and MTP graph path supports actual 310P hardware; these additions use the native
Ascend v1 runner. The completed device checks ran on Atlas A3, SoC
`Ascend910_9362`; they do not establish 910B performance.

## Install

Use the vLLM 0.23.0 / compatible Ascend environment and this branch's source.
The runtime Python files are already present. Do **not** apply the historical
`asr_mtp_v023.patch`, `asr_mtp_graph_v023.patch`, or the other cumulative/upgrade
patches over this checkout. Those files are retained for older installations.

First-prefill verification additionally changes the scheduler in the separate
upstream vLLM repository. Apply only the accompanying upstream patch from the
**vLLM repository root**:

```bash
git apply --check /path/to/vllm-ascend/examples/910b/external_draft_first_verify_vllm_v023.patch
git apply /path/to/vllm-ascend/examples/910b/external_draft_first_verify_vllm_v023.patch
```

If the scheduler was patched previously, check for its
`_prepare_initial_external_draft` method and compare the existing behavior;
`git apply --reverse --check` can identify an exact prior installation. Do not
force a mismatching patch. The Ascend recompute scheduler is already updated
in this branch and does not need a second patch.

The scheduler patch adds no wait or new batching policy. Requests without
`ascend_external_draft_first_verify` keep ordinary scheduling. Marked requests
verify a complete prompt plus draft when the budget permits; encoder/chunk
or budget limits fall back to ordinary generation rather than falsely treating
a partial prompt as completed verification.

## Configure the engine

All configuration must be supplied before engine construction. The optional
ASR extension is disabled when `additional_config.asr_mtp` is absent. Ordinary
ngram/model proposers and unrelated custom classes keep their original path.
External draft transport is active only for requests carrying its extra args.

Keep the external-draft backend; this checkpoint is not a native `method="mtp"`
checkpoint:

```python
speculative_config = {
    "method": "custom_class",
    "model": "vllm_ascend.spec_decode.external_draft_proposer.AscendExternalDraftProposer",
    "num_speculative_tokens": 15,
}

additional_config = {
    "asr_mtp": {
        "checkpoint": "/path/to/best.pt",
        "num_heads": 5,
        "warm_heads": 5,
        "graph_mode": "full",
        "graph_token_sizes": [8, 32, 64, 128, 256, 512, 1024],
        "graph_context_tokens": 160,
        "kv_cache_backend": "paged",
        "device_rejection": True,
    },
    "asr_input_prepare_graph": {"enabled": True},
}
```

The listed token buckets are an example, not a new performance recommendation;
keep the buckets already validated for the workload when upgrading. MTP graph,
preparation graph and target decoder graph have independent controls. Set the
target compilation mode to `FULL`, keep `enforce_eager=False`,
`async_scheduling=False` and `enable_prefix_caching=False`. The current combined
path requires a dense, unquantized Qwen3-ASR target without LoRA and
TP/PP/DP/PCP/DCP equal to one.

MTP is enabled per request by `SamplingParams.extra_args["ascend_mtp"] = True`.
An explicit `False` disables it for that request. When absent, the existing
first-verify marker is used for compatibility. For an initial external draft,
also provide the complete token list in `ascend_external_draft`, set
`ascend_external_draft_first_verify=True`, and choose
`ascend_external_draft_rejoin` from `aligned/ngram_1/ngram_2/ngram_3`.
Existing callers that do not send first-verify metadata retain the legacy
external-draft contract. No service/parser code is included in this change.

`cold_heads` defaults to the loaded head count; `warm_heads` defaults to at most
three. Per-kind `chunk_*_heads` and `finish_*_heads` inherit those values unless
overridden. EOS and remaining output/model budgets may shorten actual drafts.
The complete policy and capacity settings remain in the existing MTP modules.

## Validate and upgrade

See [the engine validation record](asr_engine_validation.md) for completed
checks, their limits, and the pending Chinese actual-engine acceptance.

- Use the tests under `tests/ut/worker`, `tests/ut/spec_decode` and
  `tests/ut/core` for registration, default-off behavior, transport, lifecycle
  and first-verification fallback contracts.
- The existing `examples/910b/tests` cover the unchanged MTP graph/cache/math,
  mixed prefill handoff, head policies, memory accounting and acceptance data.
- `ASR_MTP_CONFIG` and `ASR_MTP_GRAPH_READY` indicate loading/capture;
  `ASR_MTP_DEVICE_REJECTION` counters establish actual handoff execution.
  An ordinary scheduler's `INITIAL_DRAFT ... status=scheduled` establishes
  initial verification; the recompute path does not emit that same success log.
- See `asr_cpu_prepare.md` for the already validated CPU turnaround changes.
  Its hook upgrade is already incorporated in this branch's runtime source.

Upgrading from locally patched copies may produce ordinary Git conflicts in
these same runtime files. Preserve local edits, compare them against tracked
source, and resolve the version differences before restarting through the
normal deployment process. Merely importing the worker does not load MTP
weights, allocate its arena or capture its graphs.
