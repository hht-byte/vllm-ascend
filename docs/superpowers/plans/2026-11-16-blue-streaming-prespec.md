# Blue streaming prespec optimization plan

> **For agentic workers:** Use the existing isolated target checkout; implement bounded tasks with regression tests and independent review.

**Goal:** Attribute and reduce recurring host work in the current device-rejection handoff while preserving streaming ASR token correctness.

**Architecture:** Keep canonical research modules and the generated vLLM deployment patch as separate deliverables. Measure the pinned baseline and candidate on the same blue NPU; simulated or component evidence does not establish yellow-product gains.

**Tech stack:** Python 3.12, existing torch/torch-npu/vLLM stack, NPU graph and FIA. No global dependency replacement.

**Spec:** `research/asr_mtp_device_rejection.md` and the requested blue-zone validation workflow.

## Constraints

- Baseline source: `41a2dde544690c430355f00f502d0f9a3b0de8c9`; deployment reference: `271fa79db29e6064430e84c77673b4bf7d26208e`.
- Preserve per-head FIA events, upload lifetime fences, EOS/budget behavior, cold/warm and chunked-prefill semantics.
- Raw traces, model weights, service payloads and credentials stay outside tracked files.
- Trained MTP evidence requires a compatible, pinned auxiliary checkpoint and base model. Missing weights remain a reported limitation.
- Profiling and benchmark runs are separate; compare correctness before reporting latency.

## Review focus

- Pending rows must validate the sampled suffix without materializing unused histories; fallback proposals retain complete histories.
- Shrinking batches must not expose stale samples/drafts or omit the upload completion fence.
- Direct staging must wait before mutating pinned storage and preserve padded metadata byte for byte.
- Diagnostics must be opt-in and must not add events, synchronization or timer reads when disabled.
- Delivery patches must apply to the documented engine baseline and include all canonical changes.

## Tasks

- [x] Freeze baseline, run existing CPU/NPU parity tests, measure recurring staging costs using a reproducible component workload.
- [x] Add targeted regression tests for pending bookkeeping and bounded readback; implement bounded changes.
- [x] Compare padded metadata staging with direct pinned-array filling; retain with correctness and measured evidence.
- [x] Add opt-in attribution for the early handoff path and fallback summaries; distinguish enqueue spans from completion wait.
- [x] Run real native ASR simulation with the matching public base; safely load the trained checkpoint and verify real five-head graph parity on synthetic component inputs.
- [x] Run candidate parity and paired measurements, independently review, generate and validate deployment increments, and prepare source/delivery branches for publication.

## Evidence and decisions

- Implementation: `6f39716af53e75a19be8055d71a3e37646fa9cfc`.
- Final paired staging measurement: 2.43–2.57 ms to 0.24–0.67 ms across the fixed fixture, not an end-to-end ASR gain.
- Actual NPU MTP suite: 222 tests, 17 skips, no failures/errors. Project non-contract/non-NPU tests pass. Wider research collection retains documented baseline budget/concurrency issues.
- Five trained heads pass initial proposal, partial-rejection and confirmed-cache component checks. Public native ASR completes two cumulative inputs with speculation off.
- Native full MTP service needs the matching external-draft/first-verify scheduler and transport prerequisites. Their patches do not apply cleanly to the stock blue stack; this performance change does not port that separate integration.
- Raw traces, weights, audio and outputs remain in local runtime storage. Existing services and shared packages were not modified.
