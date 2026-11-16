# Blue ASR + MTP integration and end-to-end comparison

**Goal:** Run real ASR with trained MTP heads in isolation, establish token/text parity, then compare the previous and optimized MTP implementations under the same workload.

**Architecture:** Copy the installed, pinned vLLM and vLLM-Ascend packages into versioned experiment overlays. A common bounded transport/scheduler port serves both variants; only the canonical MTP module set differs. Run trusted preparation/control outside candidate namespaces, and keep model/service execution inside a private filesystem/network/PID/IPC environment. Reuse the generic `LLM_Optimize` worker/profile/ledger interfaces without adding ASR assumptions to controller core.

**Pinned inputs:** vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`; vLLM-Ascend `1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`; prior canonical MTP `41a2dde544690c430355f00f502d0f9a3b0de8c9`; optimized MTP `6f39716af53e75a19be8055d71a3e37646fa9cfc`. Backbone and trained checkpoint retain the content hashes in `research/blue_mtp_trained_validation.md`.

**Workload:** User requested Portuguese, four 15-second recordings, VAD segments of 2–6 seconds, ramping one request/client per second to 20 concurrent clients, then three steady-state measurements. Public dataset selection is authorized. Freeze source rows, crops, VAD version/settings, generated WAV hashes, request order, language and sampling limits before formal runs. Public sample smoke checks do not substitute for this workload.

## Implementation and gates

- [x] Read actual clean installed source, freeze interfaces and file hashes, and inspect previous compatibility failures.
- [x] Add scheduler port with executable contracts for budgets, preemption, remote KV and native-session rejection.
- [x] Add exact-class external-draft transport through the existing typed ngram dispatch; preserve request-owned state across row eviction and replacement batches.
- [x] Build immutable overlays; verify request-state compatibility and child import origins before NPU execution.
- [x] Prepare four public Portuguese recordings and deterministic VAD manifest outside Git.
- [ ] Run real AR, baseline MTP and optimized MTP correctness. Require exact token IDs and text per case; preserve mismatch evidence and do not benchmark a failing configuration.
- [ ] Execute the specified ramp and three steady-state repeats; compare request latency distributions, completed requests/tokens per second, and separately defined device/Torch memory statistics.
- [ ] Run targeted CPU/NPU regression checks, validate cleanup, freeze full manifests and restricted summaries, independently review and push code/configuration to the same repositories.

## Constraints and decisions

- Shared engine files, shared packages and existing services remain outside the experiment's write set.
- The new Docker-container probe failed in an OCI prestart hook with ENOSPC despite free host blocks and inodes. The namespace/chroot alternative inside the existing container passed read-only input, import origin, zero-capability, loopback-only and real NPU checks. Only physical card 6 is exposed, addressed as logical device 0 inside the namespace.
- Native resumable sessions with first verification are rejected; formal ASR chunks use distinct request IDs. Supporting native session continuation would require a separate lifetime protocol.
- Candidate text is never an execution instruction. Raw audio, model tensors, transcripts, runtime logs and profiling stay out of tracked files. Controller export approval remains a human operation.
- Profiling is separate from benchmark measurements. Missing values are errors, and no synthetic timings establish real gains.
