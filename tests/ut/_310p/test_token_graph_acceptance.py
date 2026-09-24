# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the acceptance gates; these do not execute a model."""

import copy
import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[3] / "examples/310p/accept_token_graph.py"
SPEC = importlib.util.spec_from_file_location("accept_token_graph_test", SOURCE)
accept = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(accept)


def reports():
    run = dict(key="prefill/b8/r0", outputs=[dict(token_ids=[1, 2], finish_reason="length")], seconds=1)
    eager = dict(runs=[copy.deepcopy(run)])
    graph = dict(
        runs=[copy.deepcopy(run)],
        initial_graphs=[dict(bucket=20, num_reqs=None, uniform=False, graph_id=1)],
        final_graphs=[dict(bucket=20, num_reqs=None, uniform=False, graph_id=1)],
        audit=[dict(event="replay", bucket=20, scheduled=[2, 1], phase="prefill", actual_reqs=2)],
    )
    return eager, graph


class TestAcceptance(unittest.TestCase):
    def test_910b_requires_startup_capture_and_updated_pa(self):
        eager, graph = reports()
        graph["backend"] = "910b"
        graph["startup"] = dict(graphs=copy.deepcopy(graph["initial_graphs"]))
        self.assertFalse(accept.compare_reports(eager, graph, [20])["passed"])
        graph["audit"][0].update(operator="_npu_paged_attention", task_updates=1)
        self.assertTrue(accept.compare_reports(eager, graph, [20])["passed"])
        graph["startup"]["graphs"] = []
        self.assertFalse(accept.compare_reports(eager, graph, [20])["passed"])

    def test_three_families_require_three_stable_graphs_and_real_replays(self):
        eager, graph = reports()
        graph["phase_routing"] = True
        graph["initial_graphs"] = [dict(bucket=20, family=family, num_reqs=None, uniform=False, graph_id=i)
                                   for i, family in enumerate(("token", "prefill", "decode"))]
        graph["final_graphs"] = copy.deepcopy(graph["initial_graphs"])
        result = accept.compare_reports(eager, graph, [20])
        self.assertTrue(result["passed"])
        self.assertIn("family prefill: no actual replay", result["missing_coverage"])
        graph["audit"].extend([
            dict(event="replay", bucket=20, family="prefill", actual_reqs=8, scheduled=[2] * 8),
            dict(event="replay", bucket=20, family="decode", actual_reqs=10, scheduled=[1] * 10,
                 phase="decode", decode_only=True),
        ])
        self.assertTrue(accept.compare_reports(eager, graph, [20])["full_acceptance"])
        graph["audit"].append(dict(event="capture"))
        self.assertFalse(accept.compare_reports(eager, graph, [20])["passed"])

    def test_identical_outputs_and_real_replay_pass(self):
        eager, graph = reports()
        self.assertTrue(accept.compare_reports(eager, graph, [20])["passed"])

    def test_changed_tokens_fail(self):
        eager, graph = reports()
        graph["runs"][0]["outputs"][0]["token_ids"] = [1, 3]
        self.assertFalse(accept.compare_reports(eager, graph, [20])["passed"])

    def test_all_eager_fallback_fails_even_if_outputs_match(self):
        eager, graph = reports()
        graph["audit"] = []
        self.assertFalse(accept.compare_reports(eager, graph, [20])["passed"])

    def test_request_count_key_or_recapture_fails(self):
        for field, value in (("num_reqs", 8), ("graph_id", 2)):
            eager, graph = reports()
            graph["final_graphs"][0][field] = value
            self.assertFalse(accept.compare_reports(eager, graph, [20])["passed"])

    def test_missing_case_fails(self):
        eager, graph = reports()
        graph["runs"] = []
        self.assertFalse(accept.compare_reports(eager, graph, [20])["passed"])

    def test_missing_bucket_fails(self):
        eager, graph = reports()
        self.assertFalse(accept.compare_reports(eager, graph, [20, 80])["passed"])

    def test_no_decode_does_not_pass_full_coverage(self):
        eager, graph = reports()
        result = accept.compare_reports(eager, graph, [20])
        self.assertTrue(any("decode-only" in item for item in result["missing_coverage"]))
        self.assertFalse(result["full_acceptance"])

    def test_named_worker_rpc_records_real_replay_and_dispatch(self):
        worker_source = SOURCE.parents[2] / "vllm_ascend/_310p/token_graph_acceptance.py"
        spec = importlib.util.spec_from_file_location("acceptance_worker_test", worker_source)
        worker_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker_module)
        descriptor = namedtuple("Descriptor", "num_tokens num_reqs uniform")(20, None, False)
        full = SimpleNamespace(name="FULL")

        class Wrapper:
            runtime_mode = full
            concrete_aclgraph_entries = {descriptor: SimpleNamespace(aclgraph=object())}

            def __call__(self):
                return "model-output"

        class Numbers(list):
            def __getitem__(self, index):
                return Numbers(super().__getitem__(index)) if isinstance(index, slice) else super().__getitem__(index)

            def tolist(self):
                return list(self)

        class Runner:
            attn_state = SimpleNamespace(name="DecodeOnly")
            _token_graph_arena = SimpleNamespace(
                states={20: SimpleNamespace(plan=SimpleNamespace(actual_reqs=2, actual_tokens=2, row_requests=(0, 1)))}
            )

            def _determine_batch_execution_and_padding(self, num_tokens, num_reqs, num_scheduled_tokens_np):
                return full, descriptor

        def module(name, **values):
            result = ModuleType(name)
            result.__dict__.update(values)
            return result

        npu = SimpleNamespace(
            synchronize=lambda: None,
            reset_peak_memory_stats=lambda: None,
            memory_allocated=lambda: 10,
            memory_reserved=lambda: 20,
            max_memory_allocated=lambda: 30,
            max_memory_reserved=lambda: 40,
        )
        wrapper = Wrapper()
        graph_module = SimpleNamespace(ACLGraphWrapper=Wrapper, _acl_graph_wrappers=[wrapper])
        stubs = {
            "torch": module("torch", npu=npu),
            "vllm": module("vllm"),
            "vllm.forward_context": module(
                "vllm.forward_context",
                get_forward_context=lambda: SimpleNamespace(batch_descriptor=descriptor, cudagraph_runtime_mode=full),
            ),
            "vllm_ascend": module("vllm_ascend"),
            "vllm_ascend.compilation": module("vllm_ascend.compilation", acl_graph=graph_module),
        }
        worker = worker_module.TokenGraphAcceptanceWorker()
        worker.model_runner = Runner()
        with patch.dict(sys.modules, stubs):
            worker.token_graph_acceptance("install")
            worker.token_graph_acceptance("reset", "decode")
            worker.model_runner._determine_batch_execution_and_padding(2, 2, Numbers([1, 1]))
            self.assertEqual(wrapper(), "model-output")
            result = worker.token_graph_acceptance()
        self.assertEqual([event["event"] for event in result["audit"]], ["dispatch", "replay"])
        self.assertTrue(result["audit"][1]["decode_only"])
        self.assertEqual(result["graphs"][0]["num_reqs"], None)
        self.assertEqual(result["memory"]["peak_allocated"], 30)


if __name__ == "__main__":
    unittest.main()
