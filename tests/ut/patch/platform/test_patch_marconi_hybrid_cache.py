# SPDX-License-Identifier: Apache-2.0

import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class _FakeFullAttentionSpec:
    def __init__(self, block_size: int):
        self.block_size = block_size


class _FakeKVCacheSpec:
    def __init__(self, block_size: int):
        self.block_size = block_size


def _fake_module(name: str, **members) -> ModuleType:
    module = ModuleType(name)
    for member_name, member in members.items():
        setattr(module, member_name, member)
    return module


def _load_marconi_patch(monkeypatch, scheduler_cls=None):
    if scheduler_cls is None:

        class OldRuntimeScheduler:
            def _mamba_block_aligned_split(
                self,
                request,
                num_new_tokens,
                num_new_local_computed_tokens=0,
                num_external_computed_tokens=0,
            ):
                return num_new_tokens

        scheduler_cls = OldRuntimeScheduler

    coordinator_cls = type("FakeHybridKVCacheCoordinator", (), {})
    modules = {
        "vllm": _fake_module("vllm"),
        "vllm.v1": _fake_module("vllm.v1"),
        "vllm.v1.core": _fake_module("vllm.v1.core"),
        "vllm.v1.core.kv_cache_coordinator": _fake_module(
            "vllm.v1.core.kv_cache_coordinator",
            HybridKVCacheCoordinator=coordinator_cls,
        ),
        "vllm.v1.core.kv_cache_utils": _fake_module(
            "vllm.v1.core.kv_cache_utils",
            BlockHash=tuple,
            BlockHashList=list,
            BlockHashListWithBlockSize=lambda hashes, *_: hashes,
            KVCacheBlock=object,
        ),
        "vllm.v1.core.sched": _fake_module("vllm.v1.core.sched"),
        "vllm.v1.core.sched.scheduler": _fake_module("vllm.v1.core.sched.scheduler", Scheduler=scheduler_cls),
        "vllm.v1.kv_cache_interface": _fake_module(
            "vllm.v1.kv_cache_interface",
            FullAttentionSpec=_FakeFullAttentionSpec,
            KVCacheSpec=_FakeKVCacheSpec,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_marconi_patch_under_test"
    module_path = Path(__file__).parents[4] / "vllm_ascend" / "patch" / "platform" / "patch_marconi_hybrid_cache.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, scheduler_cls, coordinator_cls


@pytest.fixture
def marconi_patch(monkeypatch):
    module, _, _ = _load_marconi_patch(monkeypatch)
    return module


def test_native_support_detection_uses_scheduler_signature(marconi_patch):
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


def test_native_support_is_not_overwritten(monkeypatch):
    class NativeScheduler:
        def _mamba_block_aligned_split(
            self,
            request,
            num_new_tokens,
            num_uncached_common_prefix_tokens=0,
        ):
            return num_new_tokens

    native_method = NativeScheduler._mamba_block_aligned_split
    _, scheduler_cls, coordinator_cls = _load_marconi_patch(monkeypatch, NativeScheduler)

    assert scheduler_cls._mamba_block_aligned_split is native_method
    assert not hasattr(coordinator_cls, "find_longest_cache_hit")


@pytest.mark.parametrize(
    ("hint", "num_new_tokens", "expected"),
    [(512, 1536, 512), (700, 1536, 512), (511, 1536, 1536), (512, 512, 512)],
)
def test_marconi_split_uses_512_aligned_shared_prefix(monkeypatch, marconi_patch, hint, num_new_tokens, expected):
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


def test_marconi_split_ignores_stale_hint_for_running_request(monkeypatch, marconi_patch):
    request = SimpleNamespace(
        num_computed_tokens=512,
        num_prompt_tokens=2048,
        num_tokens=2048,
    )
    scheduler = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=512),
        kv_cache_manager=SimpleNamespace(coordinator=SimpleNamespace(num_uncached_common_prefix_tokens=512)),
    )
    monkeypatch.setattr(
        marconi_patch,
        "_ORIGINAL_MAMBA_BLOCK_ALIGNED_SPLIT",
        lambda self, request, tokens, local, external: tokens,
    )

    assert (
        marconi_patch._patched_mamba_block_aligned_split(
            scheduler,
            request,
            1536,
        )
        == 1536
    )


def test_marconi_split_does_not_change_decode(monkeypatch, marconi_patch):
    request = SimpleNamespace(
        num_computed_tokens=2048,
        num_prompt_tokens=2048,
        num_tokens=2049,
    )
    scheduler = SimpleNamespace(cache_config=SimpleNamespace(block_size=512))
    monkeypatch.setattr(
        marconi_patch,
        "_ORIGINAL_MAMBA_BLOCK_ALIGNED_SPLIT",
        lambda self, request, tokens, local, external: tokens,
    )

    assert (
        marconi_patch._patched_mamba_block_aligned_split(
            scheduler,
            request,
            1,
            num_uncached_common_prefix_tokens=512,
        )
        == 1
    )


def test_coordinator_exposes_uncached_shared_prefix_length(marconi_patch):
    class FullAttentionManager:
        @staticmethod
        def find_longest_cache_hit(**kwargs):
            return [[object(), object()]]

    class MambaManager:
        @staticmethod
        def find_longest_cache_hit(**kwargs):
            return [[]]

    coordinator = object.__new__(marconi_patch.HybridKVCacheCoordinator)
    coordinator.hash_block_size = 512
    coordinator.kv_cache_config = SimpleNamespace(kv_cache_groups=[object(), object()])
    coordinator.attention_groups = [
        (_FakeFullAttentionSpec(block_size=512), [0], FullAttentionManager),
        (_FakeKVCacheSpec(block_size=512), [1], MambaManager),
    ]
    coordinator.block_pool = object()
    coordinator.use_eagle = False
    coordinator.lcm_block_size = 512

    blocks, hit_length = marconi_patch._patched_find_longest_cache_hit(
        coordinator,
        block_hashes=[],
        max_cache_hit_length=1536,
    )

    assert hit_length == 0
    assert coordinator.num_uncached_common_prefix_tokens == 1024
    assert blocks == ([], [])


def test_platform_initializer_imports_marconi_patch(monkeypatch):
    imported_modules = set()
    original_import = builtins.__import__
    fake_envs = SimpleNamespace(VLLM_ASCEND_BALANCE_SCHEDULING=False)
    fake_vllm_ascend = _fake_module("vllm_ascend", envs=fake_envs)
    fake_utils = _fake_module("vllm_ascend.utils", is_310p=lambda: False)

    def recording_import(name, globals=None, locals=None, fromlist=(), level=0):
        if not name.startswith("vllm_ascend"):
            return original_import(name, globals, locals, fromlist, level)
        imported_modules.add(name)
        if name == "vllm_ascend":
            return fake_vllm_ascend
        if name == "vllm_ascend.utils":
            return fake_utils
        return _fake_module(name)

    monkeypatch.setattr(builtins, "__import__", recording_import)
    init_path = Path(__file__).parents[4] / "vllm_ascend" / "patch" / "platform" / "__init__.py"

    exec(compile(init_path.read_bytes(), init_path, "exec"), {})

    assert "vllm_ascend.patch.platform.patch_marconi_hybrid_cache" in imported_modules
