# SPDX-License-Identifier: Apache-2.0
"""CPU delivery and default-off contracts for the optional online MTP worker.

Run with unittest to avoid the NPU/vLLM setup in tests/ut/conftest.py. Namespace
packages isolate the delivered modules from unrelated speculative proposers;
the MTP modules themselves and their relative imports execute unchanged.
"""

import builtins
import importlib
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch  # noqa: F401  # Load the CPU dependency before guarding native imports.

ROOT = Path(__file__).resolve().parents[3]
MODULES = {
    "asr_mtp_heads": "BatchedMTPHeads",
    "asr_mtp_fused": "prepare_fused_heads",
    "asr_mtp_qkv_rope": "prepare_qkv_rope",
    "asr_mtp_row_copy": "prepare_row_writer",
    "asr_mtp_graph": "MTPGraphs",
    "asr_mtp_staging": "UploadBuffer",
    "asr_mtp_device_rejection": "DeviceRejectionHandoff",
    "asr_mtp_token_attention": "PackedAttention",
    "asr_mtp_token_graph": "TokenMTPGraphs",
    "asr_mtp_paged_cache": "PagedCacheArena",
    "asr_mtp_paged_attention": "PagedAttention",
    "asr_mtp_worker": "MTPWorker",
    "asr_mtp_metrics": "AcceptanceTracker",
}


@contextmanager
def runtime_modules():
    namespaces = {}
    for name, relative in (
        ("vllm_ascend", "vllm_ascend"),
        ("vllm_ascend.spec_decode", "vllm_ascend/spec_decode"),
        ("vllm_ascend.patch", "vllm_ascend/patch"),
        ("vllm_ascend.patch.worker", "vllm_ascend/patch/worker"),
        ("vllm_ascend.worker", "vllm_ascend/worker"),
    ):
        namespace = ModuleType(name)
        namespace.__path__ = [str(ROOT / relative)]
        namespaces[name] = namespace
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in ("torch_npu", "triton"):
            raise AssertionError(f"optional native dependency imported during CPU startup: {name}")
        return original_import(name, *args, **kwargs)

    with patch.dict(sys.modules, namespaces), patch("builtins.__import__", guarded_import):
        for name in tuple(sys.modules):
            if name.startswith("vllm_ascend.") and name not in namespaces:
                del sys.modules[name]
        yield


def runtime(name):
    source = ROOT / "vllm_ascend/spec_decode" / f"{name}.py"
    assert source.is_file(), f"online MTP delivery is missing {source.name}"
    return importlib.import_module(f"vllm_ascend.spec_decode.{name}")


class DeliveryTests(unittest.TestCase):
    def test_runtime_modules_import_without_native_kernels(self):
        with runtime_modules():
            for name, exported in MODULES.items():
                with self.subTest(module=name):
                    self.assertTrue(callable(getattr(runtime(name), exported)))
            self.assertTrue((ROOT / "vllm_ascend/spec_decode/asr_mtp_row_copy_kernel.py").is_file())
            self.assertNotIn("vllm_ascend.spec_decode.asr_mtp_row_copy_kernel", sys.modules)

    def test_request_and_graph_features_default_off(self):
        with runtime_modules():
            worker = runtime("asr_mtp_worker")
            self.assertFalse(worker.request_enabled(SimpleNamespace(extra_args=None)))
            self.assertFalse(worker.request_enabled(SimpleNamespace(extra_args={})))
            self.assertTrue(worker.request_enabled(SimpleNamespace(extra_args={"ascend_mtp": True})))
            self.assertTrue(
                worker.request_enabled(SimpleNamespace(extra_args={"ascend_external_draft_first_verify": True}))
            )
            with self.assertRaises(TypeError):
                worker.request_enabled(SimpleNamespace(extra_args={"ascend_mtp": 1}))
            self.assertIsNone(runtime("asr_mtp_graph").graph_options({}, context=32, slots=2, max_batch=2))
            worker.validate_device_handoff_hook(SimpleNamespace(), enabled=False)
            with self.assertRaisesRegex(RuntimeError, "patch_external_draft_mtp.py"):
                worker.validate_device_handoff_hook(SimpleNamespace(), enabled=True)


class OriginalRunner:
    def _get_drafter(self):
        return "original drafter"

    def _update_states(self, scheduler_output):
        return ("original update", scheduler_output)

    def _bookkeeping_sync(
        self, scheduler_output, sampler_output, logits, hidden_states, num_scheduled_tokens, spec_decode_metadata
    ):
        return ("original bookkeeping", sampler_output)

    def _check_and_update_cudagraph_mode(self, attention_backends, kv_cache_groups):
        return ("original graph setup", attention_backends, kv_cache_groups)

    def propose_draft_token_ids(
        self,
        valid_sampled_token_ids,
        sampling_metadata,
        scheduler_output,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        positions,
        num_scheduled_tokens,
        hidden_states,
        aux_hidden_states=None,
        sample_hidden_states=None,
        target_model_batch_desc=None,
    ):
        return ("original proposal", valid_sampled_token_ids)


class WorkerHookTests(unittest.TestCase):
    def load_hook(self):
        source = ROOT / "vllm_ascend/patch/worker/patch_external_draft_mtp.py"
        self.assertTrue(source.is_file(), "online MTP delivery is missing the late worker hook")
        runner_module = ModuleType("vllm_ascend.worker.model_runner_v1")
        external = ModuleType("vllm_ascend.spec_decode.external_draft_proposer")

        class Runner(OriginalRunner):
            pass

        class ExternalProposer:
            def __init__(self, config, runner):
                self.runner = runner

        runner_module.NPUModelRunner = Runner
        external.AscendExternalDraftProposer = ExternalProposer
        sys.modules[runner_module.__name__] = runner_module
        sys.modules[external.__name__] = external
        return Runner, importlib.import_module("vllm_ascend.patch.worker.patch_external_draft_mtp")

    def test_imported_late_hook_preserves_default_runner_behavior(self):
        with runtime_modules():
            Runner, hook = self.load_hook()
            runner = Runner()
            runner.vllm_config = SimpleNamespace(additional_config={})
            runner.drafter = runner._get_drafter()
            self.assertEqual(runner.drafter, "original drafter")
            self.assertEqual(runner._update_states("schedule"), ("original update", "schedule"))
            sampled = [[3]]
            self.assertEqual(
                runner.propose_draft_token_ids(sampled, None, None, None, None, None, 1, None),
                ("original proposal", sampled),
            )
            sampler = object()
            self.assertEqual(
                runner._bookkeeping_sync(None, sampler, None, None, 1, None), ("original bookkeeping", sampler)
            )
            self.assertEqual(
                runner._check_and_update_cudagraph_mode("backends", "groups"),
                ("original graph setup", "backends", "groups"),
            )
            previous = Runner._bookkeeping_sync
            hook.install()
            self.assertIs(Runner._bookkeeping_sync, previous)

    def test_mtp_configuration_requires_external_draft_and_installs_handoff(self):
        with runtime_modules():
            Runner, hook = self.load_hook()
            runner = Runner()
            runner.vllm_config = SimpleNamespace(additional_config={"asr_mtp": {}})
            runner.speculative_config = SimpleNamespace(method="draft_model", model="wrong")
            with self.assertRaisesRegex(ValueError, "external-draft"):
                runner._get_drafter()
            runner.speculative_config = SimpleNamespace(method="custom_class", model=hook.EXTERNAL_CLASS)
            runner.drafter = runner._get_drafter()
            self.assertIsInstance(runner.drafter, hook.AscendExternalDraftMTPProposer)
            self.assertIsNone(runner.drafter.online)
            runtime("asr_mtp_worker").validate_device_handoff_hook(runner, enabled=True)

    def test_default_worker_memory_budget_is_unchanged(self):
        with runtime_modules():
            Runner, hook = self.load_hook()

            class Worker:
                def determine_available_memory(self):
                    return 4096

            hook.install_worker_memory_hook(Worker)
            worker = Worker()
            worker.model_runner = Runner()
            worker.model_runner.drafter = "original drafter"
            self.assertEqual(worker.determine_available_memory(), 4096)


if __name__ == "__main__":
    unittest.main()
