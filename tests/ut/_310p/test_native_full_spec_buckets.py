# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU setup and real-dispatch contracts; these do not execute an NPU graph.

Run directly with --vllm-root /path/to/vllm for a source-only core checkout.
Otherwise use the installed vLLM package, or an adjacent vllm checkout.
"""

import __future__

import argparse
import ast
import importlib.util
import sys
import unittest
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VLLM_ROOT = None
CAPTURE_SIZES = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    32,
    40,
    48,
    64,
    80,
    96,
    128,
    144,
    160,
    180,
    200,
    256,
    300,
    384,
    512,
    768,
    1024,
)


class Mode(Enum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2
    FULL_DECODE_ONLY = 3
    FULL_AND_PIECEWISE = 4

    def mixed_mode(self):
        if self == Mode.FULL_DECODE_ONLY:
            return Mode.NONE
        return Mode.PIECEWISE if self == Mode.FULL_AND_PIECEWISE else self

    def decode_mode(self):
        return Mode.FULL if self.separate_routine() else self

    def separate_routine(self):
        return self in (Mode.FULL_DECODE_ONLY, Mode.FULL_AND_PIECEWISE)

    def requires_piecewise_compilation(self):
        return self in (Mode.PIECEWISE, Mode.FULL_AND_PIECEWISE)

    def has_mode(self, other):
        return other in (self.mixed_mode(), self.decode_mode())

    @classmethod
    def valid_runtime_modes(cls):
        return {cls.NONE, cls.FULL, cls.PIECEWISE}


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def graph_inputs(support="ALWAYS"):
    builder = SimpleNamespace(get_cudagraph_support=lambda *_: SimpleNamespace(name=support))
    backend = SimpleNamespace(get_builder_cls=lambda: builder)
    return [[backend]], [SimpleNamespace(kv_cache_spec=object())]


def runner_class():
    class Runner:
        def _determine_batch_execution_and_padding(self, *args, **kwargs):
            raise NotImplementedError

        def _check_and_update_cudagraph_mode(self, attention_backends, kv_cache_groups):
            self.setup_query_len = multiple = self.uniform_decode_query_len
            config = self.compilation_config
            if multiple > 1:
                config.cudagraph_capture_sizes = sorted(
                    {
                        (size + multiple - 1) // multiple * multiple
                        for size in config.cudagraph_capture_sizes
                        if (size + multiple - 1) // multiple * multiple <= config.max_cudagraph_capture_size
                    }
                )
            if getattr(self, "fail_setup", False):
                raise RuntimeError("setup failed")
            if hasattr(self, "cudagraph_dispatcher"):
                self.cudagraph_dispatcher.initialize_cudagraph_keys(config.cudagraph_mode, multiple)
            return "ready"

    return Runner


class RunnerFixture:
    def make_runner(self, method="custom_class", mode=Mode.FULL):
        path = ROOT / "vllm_ascend/patch/worker/patch_full_graph_spec_buckets.py"
        self.assertTrue(path.exists(), "native FULL bucket initialization patch is missing")
        extension = load(path, "native_full_bucket_test")
        cls = runner_class()
        extension.install_full_graph_spec_buckets(cls)
        runner = cls()
        runner.uniform_decode_query_len = 16
        runner.speculative_config = SimpleNamespace(method=method, num_speculative_tokens=15) if method else None
        runner.compilation_config = SimpleNamespace(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=list(CAPTURE_SIZES),
            max_cudagraph_capture_size=1024,
            compile_sizes=[],
        )
        runner.vllm_config = SimpleNamespace(
            compilation_config=runner.compilation_config,
            speculative_config=runner.speculative_config,
            lora_config=None,
            scheduler_config=SimpleNamespace(max_num_seqs=20),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
        )
        return runner, extension


class SetupTests(RunnerFixture, unittest.TestCase):
    def test_preserves_buckets_and_restores_real_budget(self):
        runner, _ = self.make_runner()
        self.assertEqual(runner._check_and_update_cudagraph_mode(*graph_inputs()), "ready")
        self.assertEqual(runner.compilation_config.cudagraph_capture_sizes, list(CAPTURE_SIZES))
        self.assertEqual(runner.setup_query_len, 1)
        self.assertEqual(runner.uniform_decode_query_len, 16)
        self.assertEqual(runner.speculative_config.num_speculative_tokens, 15)

    def test_restores_query_length_on_initialization_error(self):
        runner, _ = self.make_runner()
        runner.fail_setup = True
        with self.assertRaisesRegex(RuntimeError, "setup failed"):
            runner._check_and_update_cudagraph_mode(*graph_inputs())
        self.assertEqual(runner.uniform_decode_query_len, 16)
        self.assertEqual(runner.compilation_config.cudagraph_capture_sizes, list(CAPTURE_SIZES))

    def test_other_methods_modes_and_backend_downgrades_keep_original_setup(self):
        cases = [(method, Mode.FULL, "ALWAYS") for method in (None, "mtp", "eagle3", "ngram")]
        cases += [
            ("custom_class", mode, "ALWAYS")
            for mode in (Mode.NONE, Mode.PIECEWISE, Mode.FULL_DECODE_ONLY, Mode.FULL_AND_PIECEWISE)
        ]
        cases += [
            ("custom_class", Mode.FULL, support)
            for support in ("NEVER", "UNIFORM_BATCH", "UNIFORM_SINGLE_TOKEN_DECODE")
        ]
        for method, mode, support in cases:
            with self.subTest(method=method, mode=mode, support=support):
                runner, _ = self.make_runner(method, mode)
                runner._check_and_update_cudagraph_mode(*graph_inputs(support))
                self.assertEqual(runner.setup_query_len, 16)
                self.assertEqual(len(runner.compilation_config.cudagraph_capture_sizes), 17)

    def test_install_is_idempotent_and_requires_supported_api(self):
        runner, extension = self.make_runner()
        installed = type(runner)._check_and_update_cudagraph_mode
        extension.install_full_graph_spec_buckets(type(runner))
        self.assertIs(type(runner)._check_and_update_cudagraph_mode, installed)
        with self.assertRaisesRegex(RuntimeError, "graph setup API"):
            extension.install_full_graph_spec_buckets(type("Unsupported", (), {}))

    def test_worker_registration_is_late_and_version_gated(self):
        _, extension = self.make_runner()
        source = ast.parse((ROOT / "vllm_ascend/worker/worker.py").read_text(encoding="utf-8"))
        runner_import = next(
            node
            for node in source.body
            if isinstance(node, ast.ImportFrom) and node.module == "vllm_ascend.worker.model_runner_v1"
        )
        registration = next(
            node
            for node in source.body
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "install_full_graph_spec_buckets"
                for child in ast.walk(node)
            )
        )
        self.assertGreater(registration.lineno, runner_import.lineno)
        for supported in (False, True):
            cls = runner_class()
            namespace = dict(NPUModelRunner=cls, vllm_version_is=lambda _, supported=supported: supported)
            with patch.dict(sys.modules, {"vllm_ascend.patch.worker.patch_full_graph_spec_buckets": extension}):
                exec(
                    compile(ast.Module(body=[registration], type_ignores=[]), "worker registration", "exec"), namespace
                )
            self.assertEqual(getattr(cls, "_full_graph_spec_buckets_installed", False), supported)

    def test_distributed_settings_keep_original_bucket_alignment(self):
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            with self.subTest(parallel_setting=name):
                runner, _ = self.make_runner()
                setattr(runner.vllm_config.parallel_config, name, 2)
                runner._check_and_update_cudagraph_mode(*graph_inputs())
                self.assertEqual(runner.setup_query_len, 16)
                self.assertEqual(len(runner.compilation_config.cudagraph_capture_sizes), 17)

    def test_specialized_runner_keeps_its_own_graph_setup(self):
        runner, _ = self.make_runner()

        class Specialized(type(runner)):
            def _determine_batch_execution_and_padding(self, *args, **kwargs):
                return "specialized dispatch"

        specialized = Specialized()
        specialized.__dict__.update(runner.__dict__)
        specialized._check_and_update_cudagraph_mode(*graph_inputs())
        self.assertEqual(specialized.setup_query_len, 16)
        self.assertEqual(len(specialized.compilation_config.cudagraph_capture_sizes), 17)


class MixedDispatchTests(RunnerFixture, unittest.TestCase):
    def core_path(self):
        candidates = [VLLM_ROOT, ROOT.parent / "vllm"]
        spec = importlib.util.find_spec("vllm")
        if spec is not None and spec.origin:
            candidates.append(Path(spec.origin).resolve().parent.parent)
        for candidate in candidates:
            if candidate and (candidate / "vllm/v1/cudagraph_dispatcher.py").is_file():
                return candidate
        self.skipTest("install vLLM or pass --vllm-root for real dispatcher contracts")

    def dispatcher(self, config):
        core = self.core_path()
        stubs = {
            "vllm": module("vllm"),
            "vllm.config": module("vllm.config", CUDAGraphMode=Mode, VllmConfig=object),
            "vllm.forward_context": module("vllm.forward_context", BatchDescriptor=Descriptor),
            "vllm.logger": module("vllm.logger", init_logger=lambda _: Mock()),
            "vllm.lora.utils": module("vllm.lora.utils", get_captured_lora_counts=Mock()),
            "vllm.compilation.breakable_cudagraph": module(
                "vllm.compilation.breakable_cudagraph", is_breakable_cudagraph_enabled=lambda: False
            ),
        }
        with patch.dict(sys.modules, stubs):
            upstream = load(core / "vllm/v1/cudagraph_dispatcher.py", "core_dispatcher_test")
            sys.modules["vllm.v1.cudagraph_dispatcher"] = upstream
            load(ROOT / "vllm_ascend/patch/worker/patch_cudagraph.py", "ascend_dispatcher_test")
            return upstream.CudagraphDispatcher(config)

    def test_actual_runner_dispatches_prefill_and_variable_drafts_together(self):
        runner, _ = self.make_runner()
        runner.cudagraph_dispatcher = self.dispatcher(runner.vllm_config)
        runner._check_and_update_cudagraph_mode(*graph_inputs())
        self.assertEqual(len(runner.cudagraph_dispatcher.get_capture_descs()[0][1]), 31)
        source = ROOT / "vllm_ascend/worker/model_runner_v1.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "_determine_batch_execution_and_padding"
        )
        namespace = dict(np=np, CUDAGraphMode=Mode, BatchDescriptor=Descriptor, enable_sp=lambda _: False)
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]),
                str(source),
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            namespace,
        )
        runner._pad_for_sequence_parallelism = lambda tokens: tokens
        runner.model_config = SimpleNamespace(is_encoder_decoder=False)
        runner.parallel_config = SimpleNamespace(data_parallel_size=1)
        runner.vllm_config.parallel_config = runner.parallel_config
        runner.input_batch = SimpleNamespace(lora_id_to_lora_request={})
        runner.vllm_config.observability_config = SimpleNamespace(cudagraph_metrics=False)
        determine = namespace[method.name]
        for queries, computed, prompts in [
            ([59, 1, 4], [0, 80, 96], [59, 20, 20]),
            ([60, 2, 2], [10, 80, 96], [70, 20, 20]),
            ([5, 6, 7], [0, 80, 96], [5, 20, 20]),
            ([16, 16, 16], [0, 80, 96], [16, 20, 20]),
        ]:
            with self.subTest(queries=queries):
                stages = ["prefill" if done < prompt else "decode" for done, prompt in zip(computed, prompts)]
                self.assertEqual(set(stages), {"prefill", "decode"})
                runner.input_batch.num_computed_tokens_cpu = np.array(computed)
                scheduled = np.array(queries)
                result = determine(runner, sum(queries), len(queries), scheduled, max(queries), False)
                self.assertEqual(result[0], Mode.FULL)
                self.assertFalse(result[1].uniform)
                self.assertEqual(result[1].num_tokens, sum(queries))
                self.assertIn(result[1], runner.cudagraph_dispatcher.cudagraph_keys[Mode.FULL])
                np.testing.assert_array_equal(scheduled, queries)
                self.assertEqual(runner.uniform_decode_query_len, 16)
        self.assertEqual(runner.cudagraph_dispatcher.dispatch(6)[1].num_tokens, 6)
        self.assertEqual(runner.cudagraph_dispatcher.dispatch(1025)[0], Mode.NONE)

    def test_tp8_sequence_parallel_mixed_batch_keeps_aligned_bucket(self):
        runner, _ = self.make_runner()
        runner.vllm_config.parallel_config = SimpleNamespace(tensor_parallel_size=8, data_parallel_size=1)
        runner.cudagraph_dispatcher = self.dispatcher(runner.vllm_config)
        runner._check_and_update_cudagraph_mode(*graph_inputs())
        source = ROOT / "vllm_ascend/worker/model_runner_v1.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "_determine_batch_execution_and_padding"
        )
        namespace = dict(np=np, CUDAGraphMode=Mode, BatchDescriptor=Descriptor, enable_sp=lambda _: True)
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]),
                str(source),
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            namespace,
        )
        runner._pad_for_sequence_parallelism = lambda tokens: tokens
        runner.model_config = SimpleNamespace(is_encoder_decoder=False)
        runner.input_batch = SimpleNamespace(lora_id_to_lora_request={}, num_computed_tokens_cpu=np.array([0, 80, 96]))
        runner.vllm_config.observability_config = SimpleNamespace(cudagraph_metrics=False)
        result = namespace[method.name](runner, 176, 3, np.array([160, 1, 15]), 160, False)
        self.assertEqual(result[0], Mode.FULL)
        self.assertEqual(result[1].num_tokens, 192)
        self.assertEqual(runner.setup_query_len, 16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vllm-root", type=Path)
    args, remaining = parser.parse_known_args()
    VLLM_ROOT = args.vllm_root
    unittest.main(argv=[sys.argv[0], *remaining])
