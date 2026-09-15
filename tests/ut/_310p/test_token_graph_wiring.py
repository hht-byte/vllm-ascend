# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU boundary tests for the real runner methods and dispatcher sources.

Runner methods are compiled from their AST into a minimal parent harness;
this tests argument/data flow, not NPU execution or importing the full engine.
The adjacent vLLM v0.23.0 checkout is required for dispatcher contract tests.
"""

import ast
import importlib.util
import sys
import unittest
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]


class Mode(Enum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2

    def mixed_mode(self):
        return self

    def decode_mode(self):
        return self

    def separate_routine(self):
        return False

    def requires_piecewise_compilation(self):
        return False

    def has_mode(self, other):
        return self == other

    @classmethod
    def valid_runtime_modes(cls):
        return set(cls)


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
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


class TestDispatcher(unittest.TestCase):
    def test_real_upstream_initialization_and_dispatch(self):
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
            load(ROOT.parent / "vllm/vllm/v1/cudagraph_dispatcher.py", "vllm.v1.cudagraph_dispatcher")
            plugin = load(ROOT / "vllm_ascend/_310p/token_graph_dispatcher.py", "token_dispatch_test")
            config = SimpleNamespace(
                compilation_config=SimpleNamespace(
                    cudagraph_mode=Mode.FULL,
                    max_cudagraph_capture_size=192,
                    cudagraph_capture_sizes=[20, 80, 192],
                    compile_sizes=[],
                ),
                speculative_config=None,
                lora_config=None,
                scheduler_config=SimpleNamespace(max_num_seqs=20),
            )
            dispatcher = plugin.TokenGraphDispatcher310(config)
            dispatcher.initialize_cudagraph_keys(Mode.FULL)
            self.assertEqual(len(dispatcher.cudagraph_keys[Mode.FULL]), 3)
            for tokens, bucket in ((10, 20), (18, 20), (20, 20), (79, 80), (180, 192)):
                mode, desc = dispatcher.dispatch(tokens)
                self.assertEqual(mode, Mode.FULL)
                self.assertEqual(desc.num_tokens, bucket)
                self.assertIsNone(desc.num_reqs)
                self.assertFalse(desc.uniform)
            self.assertEqual(dispatcher.dispatch(193)[0], Mode.NONE)
            self.assertEqual(dispatcher.dispatch(20, invalid_modes={Mode.FULL})[0], Mode.NONE)


def runner_harness(parent, namespace):
    source = ast.parse((ROOT / "vllm_ascend/_310p/model_runner_310p.py").read_text(encoding="utf-8"))
    original = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner310")
    wanted = {
        "_determine_batch_execution_and_padding",
        "_pad_query_start_loc_for_fia",
        "_model_forward",
        "_build_attention_metadata",
    }
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    node = ast.ClassDef(
        name="Runner", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[]
    )
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    env = dict(namespace, Parent=parent)
    # Future annotations let the harness omit irrelevant vLLM annotation imports.
    import __future__

    exec(compile(tree, "runner_boundary", "exec", flags=__future__.annotations.compiler_flag), env)
    return env["Runner"]


class TestRunnerBoundaries(unittest.TestCase):
    def test_warmup_build_uses_graph_metadata_without_changing_scheduler_count(self):
        tg = load(ROOT / "vllm_ascend/_310p/token_graph.py", "token_graph_wiring_runtime")

        class FullSpec:
            sliding_window = None

        class Parent:
            def _build_attention_metadata(self, **kwargs):
                return {"layer": SimpleNamespace()}, SimpleNamespace(num_actual_tokens=10)

        gpu = SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: None)))
        runner = runner_harness(
            Parent, dict(torch=gpu, TokenGraphArena=tg.TokenGraphArena, FullAttentionSpec=FullSpec)
        )()
        runner._spec_dummy_capture = False
        runner._token_graph_active = False
        runner._token_graph_warmup = True
        runner._token_graph_dummy = True
        runner._token_graph_arena = None
        runner.token_graph_config = tg.TokenGraphConfig(True)
        runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=FullSpec())])
        runner.compilation_config = SimpleNamespace(cudagraph_capture_sizes=[20])
        runner.max_num_reqs = 20
        runner.max_model_len = 512
        runner.device = "cpu"
        runner.query_start_loc = SimpleNamespace(cpu=torch.tensor([0, 5, 10], dtype=torch.int32))
        runner.optimistic_seq_lens_cpu = torch.tensor([128, 128])
        runner.input_batch = SimpleNamespace(
            block_table=[
                SimpleNamespace(
                    get_device_tensor=lambda: torch.zeros(20, 4, dtype=torch.int32),
                    slot_mapping=SimpleNamespace(gpu=torch.arange(20, dtype=torch.int32)),
                )
            ]
        )
        metadata, common = runner._build_attention_metadata(num_tokens=10, num_tokens_padded=20, num_reqs=2)
        self.assertEqual(common.num_actual_tokens, 10)
        self.assertEqual(metadata["layer"].num_actual_tokens, 20)
        self.assertEqual(metadata["layer"].slot_mapping.tolist(), [-1] * 20)

    def test_graph_selection_preserves_force_eager(self):
        class Parent:
            def _determine_batch_execution_and_padding(self, **kwargs):
                self.received = kwargs
                return (Mode.NONE if kwargs["force_eager"] else Mode.FULL, Descriptor(20), False, None, None)

        runner = runner_harness(Parent, dict(CUDAGraphMode=Mode))()
        runner.token_graph_config = SimpleNamespace(enabled=True)
        for eager in (False, True):
            result = runner._determine_batch_execution_and_padding(
                18, 6, [1, 1, 1, 1, 7, 7], 7, False, force_eager=eager
            )
            self.assertEqual(runner._token_graph_active, not eager)
            self.assertEqual(result[0], Mode.NONE if eager else Mode.FULL)
            self.assertFalse(runner.received["force_uniform_decode"])
            self.assertFalse(runner.received["allow_microbatching"])

    def test_padding_keeps_real_request_count(self):
        copy = Mock()
        runner = runner_harness(object, dict(CUDAGraphMode=Mode, copy_snapshot_to_gpu=copy))()
        runner.token_graph_config = SimpleNamespace(enabled=True)
        self.assertEqual(runner._pad_query_start_loc_for_fia("qsl", 20, 20, 8, Mode.FULL, None), 8)
        copy.assert_called_once_with("qsl")

    def test_update_precedes_replay_and_skips_mainline_update(self):
        from functools import partial

        calls = []
        state = SimpleNamespace(
            update=lambda backend: calls.append("update"),
            tasks={},
            plan=SimpleNamespace(actual_tokens=20, actual_reqs=8, query_lens=[7, 7, 1, 1, 1, 1, 1, 1]),
        )
        backend = SimpleNamespace(
            npu=SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: calls.append("sync")))
        )
        env = dict(
            CUDAGraphMode=Mode,
            logger=Mock(),
            partial=partial,
            torch=backend,
            torch_npu=backend,
            get_forward_context=lambda: SimpleNamespace(cudagraph_runtime_mode=Mode.FULL),
        )
        runner = runner_harness(object, env)()
        runner.token_graph_config = SimpleNamespace(enabled=True)
        runner.uses_mrope = False
        runner._token_graph_arena = SimpleNamespace(states={20: state})
        runner.model = lambda **kwargs: calls.append("replay")
        runner._update_full_graph_params_if_needed = Mock(side_effect=AssertionError("Mainline update must not run"))
        runner._model_forward(20, positions=torch.arange(20))
        self.assertEqual(calls, ["update", "sync", "replay"])


if __name__ == "__main__":
    unittest.main()
