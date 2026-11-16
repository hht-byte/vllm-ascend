# MTP host staging and device-handoff bookkeeping

This focused upgrade reduces recurring host work in the existing optional ASR
MTP extension. It directly fills reusable pinned metadata buffers, copies only
the active device-result rows, and validates pending sampled suffixes without
materializing the complete token history. It also packs strided fused-QKV value
rows before the existing native cache-row writer; dense rows keep their storage.
The existing upload lifetime fence,
padding contract, fallback histories, FIA events and request lifecycle remain
part of the execution contract.

The deployment baseline is `310P-graph` at
`271fa79db29e6064430e84c77673b4bf7d26208e`, with the optional first-verify/MTP
extension already installed and its MTP modules matching canonical source
`41a2dde544690c430355f00f502d0f9a3b0de8c9`. A clean checkout of that delivery
commit does not contain the optional MTP modules: pulling this branch alone
does not install the extension.

The corresponding canonical implementation is
[`6f39716af53e75a19be8055d71a3e37646fa9cfc`](https://github.com/hht-byte/vllm-ascend/commit/6f39716af53e75a19be8055d71a3e37646fa9cfc).
The source branch `codex/streaming-prespec-blue-opt` also carries the portable
component benchmark, aggregate results, real-weight verifier and public ASR
simulation. Its [validation report](https://github.com/hht-byte/vllm-ascend/blob/codex/streaming-prespec-blue-opt/research/blue_mtp_host_staging.md)
distinguishes component gains from full-service performance. The final 910B3
component experiment measured staging at 2.43–2.57 ms before and 0.24–0.67 ms
after across the stated fixtures; it does not establish an ASR latency gain.

## Install or roll back

During the normal code-update window, from the installed vLLM-Ascend source
root, check and apply the focused upgrade:

```bash
git apply --check examples/910b/asr_mtp_host_staging_v023.patch
git apply examples/910b/asr_mtp_host_staging_v023.patch
```

The patch changes only `asr_mtp_worker.py`, `asr_mtp_device_rejection.py`,
`asr_mtp_staging.py`, `asr_mtp_token_graph.py` and `asr_mtp_graph.py` under
`vllm_ascend/spec_decode/`. Deploy the five files together, then use the usual
worker restart procedure. Existing MTP, Prespec, head-count, graph and sampling
settings apply unchanged. The device rejection path retains its existing
opt-in switch and eligibility checks.

For a different installed MTP version, generate an upgrade from the matching
updated canonical source checkout against the actual engine directory:

```bash
python research/build_asr_mtp_patch.py --ascend-root /path/to/vllm-ascend --out /path/to/asr-mtp-upgrade.patch
git -C /path/to/vllm-ascend apply --check /path/to/asr-mtp-upgrade.patch
git -C /path/to/vllm-ascend apply /path/to/asr-mtp-upgrade.patch
```

Review the resulting diff before installation. Fresh-install and cumulative
MTP patches are alternative installation paths. Do not stack this focused
upgrade on top of a cumulative patch that already contains these changes.

To restore the documented installed baseline, use the same code-update window:

```bash
git apply --check --reverse examples/910b/asr_mtp_host_staging_v023.patch
git apply --reverse examples/910b/asr_mtp_host_staging_v023.patch
```

Then restart workers normally. Reverse application must pass against the exact
installed files; subsequent local changes require a reviewed rollback diff.

## Regression checks and attribution

After installing the patch, run the focused CPU tests using a Python environment
with the existing Torch dependencies. This command works without shell-specific
environment-variable syntax:

```bash
python -c "import sys, unittest; sys.path[:0] = ['vllm_ascend/spec_decode', 'examples/910b/tests']; unittest.main(module=None, argv=['unittest', 'test_asr_mtp_direct_staging', 'test_asr_mtp_pending_bookkeeping', 'test_asr_mtp_handoff_timing', 'test_asr_mtp_strided_write'], verbosity=2)"
```

The tests cover exact padded metadata, pinned-buffer reuse fences, reordered and
shrinking batches, bounded sampled-suffix reads, complete fallback histories,
single-result consumption, disabled timing, early-handoff attribution and
strided/dense fused-QKV cache writes. They
use CPU/fake-device execution and do not establish actual service performance.

Optional early-handoff timing uses the existing
`additional_config.asr_mtp.profile_stages=True` setting. It reports host enqueue
and CPU wait/conversion spans; enqueue time is not completed NPU kernel time.
Disabled attribution adds no timer reads or new device events. Preserve all
other settings and workloads when comparing versions, establish token/text
parity first, and collect full-step/service latency separately from profiling.
Component staging measurements alone do not establish an end-to-end ASR gain.

`310P-graph` is the delivery branch name. It does not expand auxiliary full-graph
support to `NPUWorker310`; the existing worker/platform restrictions still apply.
