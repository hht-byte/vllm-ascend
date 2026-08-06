# Marconi Admission Policy Backport Design

## Goal

Backport the Marconi-style shared-prefix admission policy from upstream vLLM PR
#37898 to the vLLM 0.19.1 compatibility path used by vLLM Ascend 0.19.1rc1.
The feature must work with the default scheduler and the Ascend Balance,
Recompute, and ProfilingChunk schedulers without changing NPU kernels or cache
tensor layouts.

## Scope

- Enable the policy only when upstream vLLM does not already expose the native
  `num_uncached_common_prefix_tokens` scheduler parameter.
- Detect when a full-attention cache group has a longer cache hit than the
  common hybrid cache hit.
- In Mamba `align` mode, split prefill at the detected shared-prefix boundary
  so the linear-attention state is admitted there.
- Preserve the existing block-aligned granularity. A configured effective
  block size of 512 admits only at multiples of 512.
- Cover the default scheduler plus the Ascend Balance, Recompute, and
  ProfilingChunk schedulers through a shared base-method patch.

## Non-goals

- Add Mamba align support to `SchedulerDynamicBatch`, which currently does not
  call `_mamba_block_aligned_split`.
- Change P/D disaggregation support. Existing restrictions on external cache
  hits in Mamba align mode remain unchanged.
- Modify GDN, Mamba, attention kernels, model runners, or KV-cache memory
  formats.
- Backport unrelated changes from upstream vLLM main.

## Architecture

Add a focused platform patch module under `vllm_ascend/patch/platform/`.
Activation is capability-based: if upstream `Scheduler._mamba_block_aligned_split`
already contains `num_uncached_common_prefix_tokens`, the patch is skipped.
This prevents duplicate behavior when vLLM Ascend is tested against a newer
upstream commit that already contains PR #37898.

For vLLM 0.19.1, the module patches two upstream methods:

1. `HybridKVCacheCoordinator.find_longest_cache_hit` follows the 0.19.1
   fixed-point algorithm and records the difference between the longest
   per-group hit and the final common hybrid hit.
2. `Scheduler._mamba_block_aligned_split` delegates to the original 0.19.1
   method, then applies the Marconi admission boundary. When callers omit the
   new hint, the wrapper reads the coordinator value only for a new request
   immediately following cache lookup. This allows existing Ascend scheduler
   overrides to inherit the behavior without duplicating their large
   `schedule()` implementations.

The wrapper keeps an optional explicit hint parameter so tests and future
callers can exercise the same interface as upstream.

## Data Flow

1. A waiting request queries `KVCacheManager.get_computed_blocks`.
2. The hybrid coordinator computes cache hits for every cache group.
3. The coordinator stores
   `longest_group_hit - final_common_hit` as the uncached shared-prefix hint.
4. The scheduler calls `_mamba_block_aligned_split`.
5. For a new prefill request in Mamba align mode, the wrapper shortens the
   scheduled token count to the shared-prefix boundary when that boundary is
   at least one block and lies before the current chunk end.
6. The existing cache allocation and state-update flow admits the recurrent
   state at that boundary. No kernel changes are required.

## Safety and Compatibility

- The coordinator hint is consumed only for a new request
  (`request.num_computed_tokens == 0`); running requests cannot inherit stale
  state from another lookup.
- Admission length is rounded down to `cache_config.block_size`.
- Prefix caching disabled, non-hybrid models, non-align Mamba modes, decode,
  and prefixes shorter than one block retain existing behavior.
- Native upstream support wins through capability detection.
- Imports stay at module scope and no new environment variable or mutable
  process-wide configuration is introduced.

## Testing

Unit tests will verify:

- capability detection skips patching a scheduler with native support;
- a longer full-attention hit produces the expected uncached common-prefix
  hint;
- a 512-token block size shortens a larger prefill chunk to 512 or another
  aligned shared-prefix boundary;
- short, stale, decode, and non-crossing hints do not change scheduling;
- the inherited wrapper works for scheduler subclasses that do not override
  `_mamba_block_aligned_split`.

Validation will run the focused unit-test file, relevant existing scheduler
tests, Ruff checks for changed Python files, and the repository formatting
check where available. NPU end-to-end validation will be reported separately
if the local environment has no Ascend runtime.

## Delivery

Use branch `codex/marconi-admission-v0191`, create signed-off conventional
commits, and push it to the `hht-byte` remote. The current
`codex/qwen35-p0-optimizations` branch remains unchanged.
