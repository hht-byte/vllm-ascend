# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; importing the NPU worker is deliberately unnecessary."""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/asr_rejection_output.py"


def load():
    assert SOURCE.exists(), "reusable rejection readback is not implemented"
    spec = importlib.util.spec_from_file_location("asr_rejection_output", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Event:
    def __init__(self):
        self.calls = []

    def record(self):
        self.calls.append("record")

    def synchronize(self):
        self.calls.append("wait")


def reference_parser(tokens, vocab, discard, *, logprobs_tensors):
    # vLLM 0.23 parse_output mask, including its exact negative-id semantics.
    array = tokens.cpu().numpy()
    mask = (array != -1) & (array < vocab)
    mask[discard] = False
    return [row[mask[i]].tolist() for i, row in enumerate(array)], logprobs_tensors


class RejectionTests(unittest.TestCase):
    def test_runner_bookkeeping_preserves_tokens_discard_and_cpu_state(self):
        source = SOURCE.with_name("model_runner_v1.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_bookkeeping_sync")
        namespace = {"RejectionSampler": NS(parse_output=reference_parser)}
        exec("from __future__ import annotations\n" + ast.unparse(fn), namespace)
        values = torch.tensor([[4, 9, -1], [6, 7, 8]], dtype=torch.int32)
        outputs = []
        for enabled in (False, True):
            buffer, _ = self.make()
            requests = {r: NS(output_token_ids=[]) for r in ("a", "b")}
            batch = NS(
                generators={},
                req_ids=["a", "b"],
                req_id_to_index={"a": 0, "b": 1},
                vocab_size=10,
                num_tokens_no_spec=np.array([2, 3]),
                num_tokens=np.array([2, 3]),
                token_ids_cpu=np.zeros((2, 20), np.int32),
                is_token_ids=np.zeros((2, 20), bool),
            )
            runner = NS(
                discard_request_indices=NS(np=np.array([1])),
                num_discarded_requests=1,
                use_async_scheduling=False,
                routed_experts_initialized=False,
                input_batch=batch,
                requests=requests,
                max_model_len=20,
                _rejection_output_buffer=buffer if enabled else None,
                _get_prompt_logprobs_dict=lambda *args: {},
            )
            result = namespace["_bookkeeping_sync"](
                runner,
                NS(num_scheduled_tokens={}),
                NS(sampled_token_ids=values, logprobs_tensors=None),
                None,
                torch.zeros(2, 1),
                2,
                None,
            )
            outputs.append(
                (
                    result,
                    batch.num_tokens.tolist(),
                    batch.token_ids_cpu.tolist(),
                    [r.output_token_ids for r in requests.values()],
                )
            )
            self.assertEqual(buffer.copies, int(enabled))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1][0][1], [[4, 9], []])

    def test_settings_rejects_nonboolean_ablation_option(self):
        from test_asr_input_prepare_graph_standalone import load as load_prepare
        from test_asr_input_prepare_graph_standalone import prepare_inputs, runner

        r = runner()
        r.vllm_config.additional_config["asr_input_prepare_graph"]["reuse_rejection_buffer"] = "false"
        with self.assertRaisesRegex(TypeError, "reuse_rejection_buffer"):
            load_prepare().settings(r, prepare_inputs=prepare_inputs)

    @unittest.skipUnless(importlib.util.find_spec("torch_npu"), "requires real NPU")
    def test_npu_changing_batch_width_and_logprobs_fallback(self):
        import torch_npu  # noqa: F401

        buffer = load().RejectionOutputBuffer(8, 16, device=torch.device("npu"))
        self.assertTrue(buffer.host.is_pinned())
        ptr = buffer.host.data_ptr()
        for n, k in [(8, 16), (1, 2), (4, 6), (2, 16)] * 4:
            cpu = torch.arange(n * k, dtype=torch.int32).view(n, k)
            cpu[:, -1] = -1
            actual = buffer.parse(cpu.to("npu"), 40, [0], parser=reference_parser, logprobs_tensors=None)
            self.assertEqual(actual, reference_parser(cpu, 40, [0], logprobs_tensors=None))
        self.assertEqual(ptr, buffer.host.data_ptr())

    def test_install_is_opt_in_and_can_be_disabled_independently(self):
        module = load()
        runner = NS(
            _input_prepare_graph=None,
            device=torch.device("cpu"),
            max_num_reqs=4,
            num_spec_tokens=5,
            use_async_scheduling=False,
            use_async_spec_decode=False,
            vllm_config=NS(additional_config={}),
        )
        self.assertIsNone(module.install(runner))
        runner._input_prepare_graph = object()
        runner.vllm_config.additional_config = {"asr_input_prepare_graph": {"reuse_rejection_buffer": False}}
        self.assertIsNone(module.install(runner))
        runner.vllm_config.additional_config["asr_input_prepare_graph"]["reuse_rejection_buffer"] = True
        self.assertEqual(module.install(runner).host.numel(), 24)
        runner.use_async_scheduling = True
        self.assertIsNone(module.install(runner))

    def make(self):
        event = Event()
        return load().RejectionOutputBuffer(4, 6, device=torch.device("cpu"), event=event), event

    def test_original_parser_gets_host_view_only_after_event_wait(self):
        buffer, event = self.make()
        values = torch.tensor([[1, 2, -1, 99], [-2, 3, 4, 5]], dtype=torch.int32)

        def parser(t, *args, **kwargs):
            self.assertEqual(event.calls, ["record", "wait"])
            self.assertEqual(t.device.type, "cpu")
            self.assertNotEqual(t.data_ptr(), values.data_ptr())
            return reference_parser(t, *args, **kwargs)

        actual = buffer.parse(values, 10, [1], parser=parser, logprobs_tensors=None)
        self.assertEqual(actual, reference_parser(values, 10, [1], logprobs_tensors=None))
        self.assertEqual(buffer.copies, 1)

    def test_shrinking_and_reordering_reuse_storage_without_stale_tail(self):
        buffer, _ = self.make()
        ptr = buffer.host.data_ptr()
        first = torch.arange(24, dtype=torch.int32).view(4, 6)
        result = buffer.parse(first, 20, [], parser=reference_parser, logprobs_tensors=None)
        second = torch.tensor([[9, -1], [7, 6]], dtype=torch.int32)
        self.assertEqual(buffer.parse(second, 20, [], parser=reference_parser, logprobs_tensors=None)[0], [[9], [7, 6]])
        self.assertEqual(result[0][0], list(range(6)))
        self.assertEqual(ptr, buffer.host.data_ptr())

    def test_logprobs_dtype_and_oversize_fall_back_without_extra_copy(self):
        buffer, event = self.make()
        for value, probs in [
            (torch.ones(1, 2, dtype=torch.int32), object()),
            (torch.ones(1, 2, dtype=torch.int64), None),
            (torch.ones(5, 6, dtype=torch.int32), None),
        ]:

            def parser(t, *args, value=value, probs=probs, **kwargs):
                self.assertIs(t, value)
                self.assertIs(kwargs["logprobs_tensors"], probs)
                return "original"

            self.assertEqual(buffer.parse(value, 10, [], parser=parser, logprobs_tensors=probs), "original")
        self.assertEqual(event.calls, [])
        self.assertEqual((buffer.copies, buffer.fallbacks), (0, 3))


if __name__ == "__main__":
    unittest.main()
