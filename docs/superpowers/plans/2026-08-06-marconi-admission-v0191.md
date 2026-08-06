# Marconi Admission Policy v0.19.1 Backport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add upstream vLLM PR #37898 shared-prefix admission behavior to the vLLM 0.19.1 compatibility path used by vLLM Ascend.

**Architecture:** Add one capability-gated platform patch. It patches the v0.19.1 hybrid coordinator to expose the uncached shared-prefix length and wraps the v0.19.1 Mamba block split method so default and inherited Ascend schedulers consume that hint without copying their `schedule()` bodies.

**Tech Stack:** Python 3.10/3.11, vLLM 0.19.1 scheduler and KV-cache interfaces, pytest, Ruff, vLLM Ascend platform patching.

## Global Constraints

- Support vLLM Ascend v0.19.1rc1 with vLLM v0.19.1.
- Skip the backport when upstream `Scheduler._mamba_block_aligned_split` already has `num_uncached_common_prefix_tokens`.
- Do not modify NPU kernels, model runners, KV-cache tensor layouts, P/D restrictions, or DynamicBatch Mamba behavior.
- Preserve effective `cache_config.block_size` alignment, including a requested effective size of 512.
- Add no environment variable or mutable global configuration.
- Keep all imports at module scope.
- All commits must use Conventional Commits and `git commit -s`.

---

### Task 1: Write Marconi Patch Regression Tests

**Files:**
- Create: `tests/ut/patch/platform/test_patch_marconi_hybrid_cache.py`

**Interfaces:**
- Consumes: `vllm_ascend.patch.platform.patch_marconi_hybrid_cache`.
- Verifies: `_has_native_marconi_support`, `_patched_find_longest_cache_hit`, and `_patched_mamba_block_aligned_split`.

- [ ] **Step 1: Add the failing capability and scheduler tests**

Create the test file with imports for `inspect`, `SimpleNamespace`, `pytest`, the vLLM scheduler/spec types, and the missing patch module. Add these cases:

```python
def test_native_support_detection_uses_scheduler_signature():
    class OldScheduler:
        def _mamba_block_aligned_split(self, request, num_new_tokens):
            return num_new_tokens

    class NativeScheduler:
        def _mamba_block_aligned_split(
            self,
            request,
            num_new_tokens,
            num_uncached_common_prefix_tokens=0,
        ):
            return num_new_tokens

    assert not marconi_patch._has_native_marconi_support(OldScheduler)
    assert marconi_patch._has_native_marconi_support(NativeScheduler)


@pytest.mark.parametrize(
    ("hint", "num_new_tokens", "expected"),
    [(512, 1536, 512), (700, 1536, 512), (511, 1536, 1536), (512, 512, 512)],
)
def test_marconi_split_uses_512_aligned_shared_prefix(
    monkeypatch, hint, num_new_tokens, expected
):
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=2048,
        num_tokens=2048,
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=512),
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(
                num_uncached_common_prefix_tokens=hint,
            )
        ),
    )
    monkeypatch.setattr(
        marconi_patch,
        "_ORIGINAL_MAMBA_BLOCK_ALIGNED_SPLIT",
        lambda self, request, tokens, local, external: tokens,
    )

    result = marconi_patch._patched_mamba_block_aligned_split(
        scheduler,
        request,
        num_new_tokens,
    )

    assert result == expected
```

Add safety cases showing that a running request ignores a stale coordinator hint and that decode does not change:

```python
def test_marconi_split_ignores_stale_hint_for_running_request(monkeypatch):
    request = SimpleNamespace(
        num_computed_tokens=512,
        num_prompt_tokens=2048,
        num_tokens=2048,
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=512),
        kv_cache_manager=SimpleNamespace(
            coordinator=SimpleNamespace(num_uncached_common_prefix_tokens=512)
        ),
    )
    monkeypatch.setattr(
        marconi_patch,
        "_ORIGINAL_MAMBA_BLOCK_ALIGNED_SPLIT",
        lambda self, request, tokens, local, external: tokens,
    )

    assert (
        marconi_patch._patched_mamba_block_aligned_split(
            scheduler, request, 1536
        )
        == 1536
    )
```

- [ ] **Step 2: Add the failing coordinator test**

Use `object.__new__(HybridKVCacheCoordinator)` with fake full-attention and Mamba manager classes. The full manager returns two 512-token blocks while the Mamba manager returns no blocks. Assert the patched function returns a zero common hit and stores a 1024-token hint:

```python
blocks, hit_length = marconi_patch._patched_find_longest_cache_hit(
    coordinator,
    block_hashes=[],
    max_cache_hit_length=1536,
)

assert hit_length == 0
assert coordinator.num_uncached_common_prefix_tokens == 1024
assert blocks == ([], [])
```

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
pytest -sv tests/ut/patch/platform/test_patch_marconi_hybrid_cache.py
```

Expected: collection fails because `patch_marconi_hybrid_cache` does not exist.

---

### Task 2: Implement the Capability-Gated Platform Patch

**Files:**
- Create: `vllm_ascend/patch/platform/patch_marconi_hybrid_cache.py`
- Test: `tests/ut/patch/platform/test_patch_marconi_hybrid_cache.py`

**Interfaces:**
- Produces: `_has_native_marconi_support(scheduler_cls: type) -> bool`.
- Produces: `_patched_find_longest_cache_hit(self, block_hashes, max_cache_hit_length)`.
- Produces: `_patched_mamba_block_aligned_split(self, request, num_new_tokens, num_new_local_computed_tokens=0, num_external_computed_tokens=0, num_uncached_common_prefix_tokens=None) -> int`.

- [ ] **Step 1: Add capability detection and preserve original methods**

Implement:

```python
import inspect

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec


def _has_native_marconi_support(scheduler_cls: type) -> bool:
    parameters = inspect.signature(
        scheduler_cls._mamba_block_aligned_split
    ).parameters
    return "num_uncached_common_prefix_tokens" in parameters


_NATIVE_MARCONI_SUPPORT = _has_native_marconi_support(Scheduler)
_ORIGINAL_MAMBA_BLOCK_ALIGNED_SPLIT = Scheduler._mamba_block_aligned_split
```

- [ ] **Step 2: Implement the v0.19.1 coordinator algorithm with lag tracking**

Implement the complete fixed-point lookup so the only semantic difference from
v0.19.1 is tracking `longest_hit_length`:

```python
def _patched_find_longest_cache_hit(
    self,
    block_hashes: list[BlockHash],
    max_cache_hit_length: int,
) -> tuple[tuple[list[KVCacheBlock], ...], int]:
    def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
        if kv_cache_spec.block_size == self.hash_block_size:
            return block_hashes
        return BlockHashListWithBlockSize(
            block_hashes,
            self.hash_block_size,
            kv_cache_spec.block_size,
        )

    num_groups = len(self.kv_cache_config.kv_cache_groups)
    hit_length = max_cache_hit_length
    longest_hit_length = 0
    hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
    is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
        self.attention_groups[0][0], FullAttentionSpec
    )

    while True:
        curr_hit_length = hit_length
        for spec, group_ids, manager_cls in self.attention_groups:
            is_full_attn = isinstance(spec, FullAttentionSpec)
            cached_blocks = hit_blocks_by_group[group_ids[0]]
            if is_full_attn and cached_blocks is not None:
                num_blocks = curr_hit_length // spec.block_size
                curr_hit_length = num_blocks * spec.block_size
            else:
                hit_blocks = manager_cls.find_longest_cache_hit(
                    block_hashes=_get_block_hashes(spec),
                    max_length=curr_hit_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    use_eagle=self.use_eagle,
                    alignment_tokens=self.lcm_block_size,
                )
                curr_hit_length = len(hit_blocks[0]) * spec.block_size
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                longest_hit_length = max(
                    longest_hit_length,
                    curr_hit_length,
                )

        if curr_hit_length >= hit_length:
            break
        hit_length = curr_hit_length
        if is_simple_hybrid:
            break

    spec, group_ids, _ = self.attention_groups[0]
    if isinstance(spec, FullAttentionSpec):
        num_blocks = hit_length // spec.block_size
        for group_id in group_ids:
            if (blocks := hit_blocks_by_group[group_id]) is not None:
                del blocks[num_blocks:]

    self.num_uncached_common_prefix_tokens = (
        longest_hit_length - hit_length
    )
    return tuple(
        blocks if blocks is not None else []
        for blocks in hit_blocks_by_group
    ), hit_length
```

- [ ] **Step 3: Implement the scheduler wrapper**

Delegate first, then apply admission only during prefill. Derive the coordinator hint only for a new request:

```python
adjusted_tokens = _ORIGINAL_MAMBA_BLOCK_ALIGNED_SPLIT(
    self,
    request,
    num_new_tokens,
    num_new_local_computed_tokens,
    num_external_computed_tokens,
)
if num_uncached_common_prefix_tokens is None:
    if request.num_computed_tokens != 0:
        return adjusted_tokens
    coordinator = getattr(self.kv_cache_manager, "coordinator", None)
    num_uncached_common_prefix_tokens = getattr(
        coordinator,
        "num_uncached_common_prefix_tokens",
        0,
    )

num_computed_tokens = (
    request.num_computed_tokens
    + num_new_local_computed_tokens
    + num_external_computed_tokens
)
is_prefill = num_computed_tokens < max(
    request.num_prompt_tokens,
    request.num_tokens - 1,
)
block_size = self.cache_config.block_size
if (
    is_prefill
    and num_uncached_common_prefix_tokens >= block_size
    and adjusted_tokens > num_uncached_common_prefix_tokens
):
    return (
        num_uncached_common_prefix_tokens // block_size * block_size
    )
return adjusted_tokens
```

- [ ] **Step 4: Activate only for vLLM without native support**

At module scope:

```python
if not _NATIVE_MARCONI_SUPPORT:
    HybridKVCacheCoordinator.find_longest_cache_hit = (
        _patched_find_longest_cache_hit
    )
    Scheduler._mamba_block_aligned_split = (
        _patched_mamba_block_aligned_split
    )
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
pytest -sv tests/ut/patch/platform/test_patch_marconi_hybrid_cache.py
```

Expected: all tests pass.

- [ ] **Step 6: Verify regression-test sensitivity**

Temporarily replace the admission branch with `return adjusted_tokens`, run the 512-alignment test and confirm it fails. Restore the implementation and confirm it passes again.

---

### Task 3: Register and Document the Backport

**Files:**
- Modify: `vllm_ascend/patch/platform/__init__.py`
- Modify: `vllm_ascend/patch/__init__.py`
- Modify: `tests/ut/patch/platform/test_patch_marconi_hybrid_cache.py`

**Interfaces:**
- Consumes: import-time behavior from `patch_marconi_hybrid_cache`.
- Produces: automatic platform-patch activation during `adapt_patch(is_global_patch=True)`.

- [ ] **Step 1: Add an activation regression test**

Assert the platform initializer contains the unconditional module import:

```python
def test_platform_initializer_registers_marconi_patch():
    init_path = Path(marconi_patch.__file__).with_name("__init__.py")
    source = init_path.read_text(encoding="utf-8")
    assert "import vllm_ascend.patch.platform.patch_marconi_hybrid_cache" in source
```

- [ ] **Step 2: Run the registration test and verify RED**

Run:

```bash
pytest -sv tests/ut/patch/platform/test_patch_marconi_hybrid_cache.py::test_platform_initializer_registers_marconi_patch
```

Expected: FAIL because the initializer does not import the new module.

- [ ] **Step 3: Register the patch**

Add this module-scope import beside the other platform patches:

```python
import vllm_ascend.patch.platform.patch_marconi_hybrid_cache  # noqa
```

- [ ] **Step 4: Add the patch inventory entry**

Document the patched coordinator and scheduler, why v0.19.1 needs the backport, upstream PR #37898, capability gating, and removal once vLLM 0.19.1 support is dropped.

- [ ] **Step 5: Run the full focused test file**

Run:

```bash
pytest -sv tests/ut/patch/platform/test_patch_marconi_hybrid_cache.py
```

Expected: all tests pass.

- [ ] **Step 6: Create the signed implementation commit**

```bash
git add \
  vllm_ascend/patch/platform/patch_marconi_hybrid_cache.py \
  vllm_ascend/patch/platform/__init__.py \
  vllm_ascend/patch/__init__.py \
  tests/ut/patch/platform/test_patch_marconi_hybrid_cache.py
git commit -s -m "feat(cache): backport Marconi admission policy"
```

---

### Task 4: Validate the Branch

**Files:**
- Verify: all files changed since `codex/qwen35-p0-optimizations`.

**Interfaces:**
- Consumes: committed patch and tests.
- Produces: evidence suitable for the final push summary.

- [ ] **Step 1: Run focused and adjacent tests**

```bash
pytest -sv tests/ut/patch/platform/test_patch_marconi_hybrid_cache.py
pytest -sv tests/ut/worker/test_block_table.py
```

- [ ] **Step 2: Run lint and formatting checks**

```bash
ruff check \
  vllm_ascend/patch/platform/patch_marconi_hybrid_cache.py \
  tests/ut/patch/platform/test_patch_marconi_hybrid_cache.py
bash format.sh ci
```

- [ ] **Step 3: Review the complete diff and commit metadata**

```bash
git diff --check codex/qwen35-p0-optimizations...HEAD
git diff --stat codex/qwen35-p0-optimizations...HEAD
git log --show-signature --format=fuller codex/qwen35-p0-optimizations..HEAD
git status --short --branch
```

Expected: no whitespace errors, only design/plan/patch/test files changed, all commits contain `Signed-off-by`, and the worktree is clean.

- [ ] **Step 4: Push the new branch**

```bash
git push -u hht-byte codex/marconi-admission-v0191
```

Expected: the remote branch is created and local tracking is configured.
