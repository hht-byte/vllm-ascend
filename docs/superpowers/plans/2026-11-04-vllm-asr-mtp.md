# Online vLLM ASR MTP implementation plan

> **For agentic workers:** Use superpowers:executing-plans and test-driven-development. The user's explicit instruction authorizes implementation in the current codex feature branch; preserve earlier uncommitted reference work.

**Goal:** Run the supplied shifted-embedding MTP checkpoint inside the ASR vLLM-Ascend worker after external rollback/prespec verification, using the existing target verifier.

**Spec:** The user's latest request in this chat: online engine integration, minimal vLLM-Ascend intrusion, low overhead. No more independent replay work.

**Architecture / choice:** Add a separate drafter selected by existing custom_class config plus `additional_config.asr_mtp`. A small import-side worker patch wraps drafter selection, proposal dispatch and request lifecycle; no model_runner source rewrite or new scheduler changes. Existing first-verify scheduler patch remains prerequisite. The target's last sampled correction/bonus is the drive token; auxiliary heads predict only tokens after it. Preserve only confirmed target hidden rows, never rejected suffix rows. Batch auxiliary heads across active requests and transfer proposal IDs to CPU once per batch. Share target embedding/logits; never load a second backbone or audio encoder.

**Alternatives:** Native method=mtp cannot load this custom SequentialMTP checkpoint without model conversion; calling an HF generate loop from the service would duplicate the backbone/prefill and miss worker state. Use the small worker adapter, as requested.

**Initial supported configuration:** v0.23 runner API, sync scheduling, TP=PP=PCP=DCP=1, unquantized Qwen3-ASR, prefix caching disabled (KV-only prefix hits have no full hidden history), greedy requests. Unsupported engine settings fail at startup; request cache/context capacity falls back to normal target decoding. Formal first-verify requests default to MTP only when the engine extension is configured; explicit per-request `ascend_mtp` can enable draftless formal requests or opt out. Shadow requests have neither marker. Auxiliary model parameters are loaded once and auxiliary history lives in a fixed preallocated arena counted during model loading. The target paged KV remains engine-owned.

## Tasks / interfaces

- [x] TDD: `asr_mtp_heads.py` implements checkpoint-compatible auxiliary blocks and batched incremental proposal. Compare against the supplied unmodified SequentialMTP with real tiny Qwen3 attention; mixed lengths, reject crop, first-token mismatch, head limits. No dependency on the independent replay scripts.
- [x] TDD: `asr_mtp_worker.py` owns bounded request slots, confirmed-row extraction, external-to-MTP handoff, load/metadata checks, fallback and cleanup. Worker hidden history excludes the last sampled token, which has not been executed. No extra target forward.
- [x] TDD: `patch_external_draft_mtp.py` wraps only three worker methods, preserves disabled behavior and is idempotent. `streaming_prespec_formal_args.py` adds explicit formal request marker; both chunk/finish use the same helper, shadow remains off.
- [x] Add deterministic patch exporter/installer, online engine launch/ASR wiring instructions and actual integration checks against local runner source. Compile, run scoped tests, review the engine patch independently.

## Review focus

- Initial external rejection at position zero, partial acceptance, all accepted + bonus: target hidden end is confirmed length minus one, not the speculative query end.
- Chunked prefill, prefix hit/missing hidden, reordered batches, preemption, abort and reused request IDs must not consume stale state.
- Capacity and head/output/context limits must bound allocations and proposals; unsupported settings must not silently run a different algorithm.
- No per-request/per-head GPU item()/tolist(), target forward, model reload or CPU copy of hidden states in the hot path.
- Checkpoint key shapes/signature and actual worker hook signatures must be checked before serving.

## Ledger

- Read-only reference source: E:/26B-multimodal/vllm and E:/26B-multimodal/opencode-project/asr-optimize/vllm-ascend; implementation/export stays in this writable repository.
- Supplied MTP source: hht-byte/asr-mtp 3a646be6178317d11a350a44a92d988ea18ded40. CPU tiny-model checks cannot establish 910B latency or full-checkpoint acceptance.
- Resumed by the user's one-shot 20:00 heartbeat. Kept all prior changes. No server connections, engine-checkout edits, commits, pushes or restarts.
- TDD ledger: missing batched-head modules/worker, profile and request marker, three hooks, formal helper marker and exporter were each observed RED before implementation, then GREEN. The release-artifact freshness test also failed on the stale export before regeneration.
- Independent architectural review checked real runner post-bookkeeping order, load/profile hooks, confirmed hidden slicing, L-d head crop and lifecycle. 600 randomized batched full-state comparisons found no discrepancy. One P2 (accepted EOS followed by bonus still triggered auxiliary heads) was reproduced RED, fixed GREEN, and covered alongside ignore_eos/min_tokens/explicit stop semantics.
- Final scoped suite: 78/78 passed (38 online/core/helper/export checks plus the earlier 40 reference checks). Includes real tiny Qwen3 target execution for initial external acceptance 0/1/all, worker cache crop and subsequent MTP; output matches greedy. Real row reorder, bounded slots/context/output budget, EOS, checkpoint load/signature rejection and disabled-mode dispatch covered.
- Export applied and compiled in a temporary fixture constructed from the actual local runner and first-verify registry; git apply --check passed, applied diff passed compileall, second export was empty. Committed-as-artifact module contents exactly equal current production module sources. git diff --check passed.
- Broader research run: 203 tests, with the same 1 failure + 19 errors already reproduced on unmodified HEAD earlier in this chat. No new failing IDs; final release freshness test separately passes. Existing unrelated fixture failures were preserved.
- Delivery: research/asr_mtp_v023.patch and asr_mtp_engine.md. The engine adapter is implemented; applying it/configuration to the actual 910B service and testing full weights/BF16/NPU SDPA/latency remain device validation, not claimed complete deployment.
- Push scope (2026-11-05): user requested pushing the online integration. Only the online modules, exported patch, formal helper, related tests/docs and ignored local runtime paths are included; earlier independent-reference edits stay local. The online-only suite passes 38/38 immediately before commit, with compileall and whitespace checks passing.
