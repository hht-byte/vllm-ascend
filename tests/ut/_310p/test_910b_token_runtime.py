# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lifetime/ordering tests; run directly without vLLM or an NPU."""

import ast
import importlib.util
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load("vllm_ascend._310p.token_graph", "vllm_ascend/_310p/token_graph.py")
pa = load("pa_under_test", "vllm_ascend/attention/token_graph_910b.py")


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.arena = pa.PAGraphArena(20, 20, 4, 512, "cpu")
        self.blocks = torch.arange(80, dtype=torch.int32).view(20, 4)
        self.slots = torch.arange(20, dtype=torch.int32)
        self.state = self.arena.prepare(20, [3, 2], [130, 4], self.blocks, self.slots)
        self.backend = Mock()
        self.stream = Mock()
        self.backend.npu.stream.return_value = nullcontext()
        self.backend.npu.is_current_stream_capturing.return_value = True
        self.backend.npu.current_stream.return_value = self.stream
        self.q = torch.empty(20, 8, 64)
        self.cache = torch.empty(81, 128, 2, 64)
        self.out = torch.empty_like(self.q)

    def capture(self, layer="layer0"):
        return self.state.attention(self.backend, layer, self.q, self.cache, self.cache,
                                    self.out, 8, 2, 0.125)

    def test_lengths_change_without_replacing_task_inputs(self):
        self.capture()
        task = self.state.tasks["layer0"]
        old_ptr = task.kwargs["context_lens"].data_ptr()
        state = self.arena.prepare(20, [2, 3], [4, 130], self.blocks.flip(0), self.slots)
        self.assertIs(state, self.state)
        self.assertEqual(task.kwargs["context_lens"][:5].tolist(), [3, 4, 128, 129, 130])
        self.assertEqual(task.kwargs["context_lens"].data_ptr(), old_ptr)
        self.assertEqual(self.arena.slots[5:].tolist(), [-1] * 15)
        self.assertTrue(torch.equal(task.kwargs["block_table"][0], self.blocks[-1]))

    def test_attach_preserves_scheduler_boundaries(self):
        qsl = torch.tensor([0, 3, 5])
        meta = SimpleNamespace(query_start_loc=qsl)
        self.state.attach(meta)
        self.assertIs(meta.query_start_loc, qsl)
        self.assertEqual(meta.num_actual_tokens, 20)
        self.assertIs(meta.pa_token_graph_state, self.state)

    def test_update_order_and_workspace_lifetime(self):
        self.capture()
        task = self.state.tasks["layer0"]
        capture_workspace = task.capture_workspace
        self.backend.reset_mock()
        self.state.update(self.backend, self.stream)
        names = [call[0] for call in self.backend.mock_calls]
        self.assertLess(names.index("npu.graph_task_update_begin"), names.index("_npu_paged_attention"))
        self.assertLess(names.index("_npu_paged_attention"), names.index("npu.graph_task_update_end"))
        task.event.record.assert_called_once_with(self.stream)
        self.stream.synchronize.assert_called_once()
        self.assertIs(task.capture_workspace, capture_workspace)
        self.assertEqual(self.state.updates, 1)

    def test_failure_preserves_first_error(self):
        self.capture()
        self.backend.reset_mock()
        self.backend._npu_paged_attention.side_effect = RuntimeError("first PA failure")
        with self.assertRaisesRegex(RuntimeError, "first PA failure"):
            self.state.update(self.backend, self.stream)
        self.backend.npu.graph_task_update_end.assert_not_called()
        self.assertEqual(self.state.updates, 0)

    def test_layer_identity_and_duplicate_guard(self):
        self.capture("layer10")
        self.capture("layer2")
        self.assertEqual(list(self.state.tasks), ["layer10", "layer2"])
        with self.assertRaisesRegex(RuntimeError, "Repeated PA capture"):
            self.capture("layer2")

    def test_eager_does_not_register_tasks(self):
        self.backend.npu.is_current_stream_capturing.return_value = False
        self.assertIs(self.capture(), self.out)
        self.assertFalse(self.state.tasks)
        self.backend.npu.graph_task_group_begin.assert_not_called()

    def test_config_is_strict_and_default_off(self):
        self.assertFalse(pa.enabled(SimpleNamespace(additional_config={})))
        self.assertTrue(pa.enabled(SimpleNamespace(additional_config={"token_graph_910b": {"enabled": True}})))
        for value in ("yes", {"enabled": "true"}, {"update_mode": "inplace"}):
            with self.assertRaises(ValueError):
                pa.enabled(SimpleNamespace(additional_config={"token_graph_910b": value}))


class RunnerBoundaryTests(unittest.TestCase):
    def setUp(self):
        class Parent:
            def _check_and_update_cudagraph_mode(self, *args):
                self.observed_qlen = self.uniform_decode_query_len
                raise RuntimeError("resolution failure")

        tree = ast.parse((ROOT / "vllm_ascend/worker/token_graph_910b.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        self.api = Mock()
        self.context = SimpleNamespace(cudagraph_runtime_mode="FULL")
        namespace = dict(NPUModelRunner=Parent, torch=SimpleNamespace(npu=self.api),
                         torch_npu=SimpleNamespace(npu=self.api), CUDAGraphMode=SimpleNamespace(FULL="FULL"),
                         get_forward_context=lambda: self.context)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "runner", "exec"), namespace)
        self.runner = namespace["TokenGraphRunner910B"].__new__(namespace["TokenGraphRunner910B"])

    def test_bucket_resolution_restores_speculative_length_on_failure(self):
        self.runner.uniform_decode_query_len = 16
        with self.assertRaisesRegex(RuntimeError, "resolution failure"):
            self.runner._check_and_update_cudagraph_mode([], [])
        self.assertEqual(self.runner.observed_qlen, 1)
        self.assertEqual(self.runner.uniform_decode_query_len, 16)

    def test_update_precedes_model_replay(self):
        order = []
        state = SimpleNamespace(tasks={"layer": object()}, update=lambda *a: order.append("update"))
        self.runner._token_graph_arena = SimpleNamespace(states={20: state})
        self.runner._token_graph_dummy = False
        self.runner._pa_update_stream = object()
        self.runner.model = lambda **kw: order.append("replay")
        self.api.synchronize.side_effect = lambda: order.append("sync")
        self.runner._model_forward(20)
        self.assertEqual(order, ["sync", "update", "replay"])

    def test_runtime_cannot_capture_missing_bucket(self):
        self.runner._token_graph_arena = SimpleNamespace(states={20: SimpleNamespace(tasks={})})
        self.runner._token_graph_dummy = False
        with self.assertRaisesRegex(RuntimeError, "not captured at startup"):
            self.runner._model_forward(20)


if __name__ == "__main__":
    unittest.main()
