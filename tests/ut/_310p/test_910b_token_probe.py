# SPDX-License-Identifier: Apache-2.0
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PATH = Path(__file__).resolve().parents[3] / "examples/910b/probe_token_graph.py"
spec = importlib.util.spec_from_file_location("pa_probe", PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class TestProbe(unittest.TestCase):
    def test_task_update_uses_handle_and_records_event_after_update(self):
        calls = []
        api = SimpleNamespace(
            graph_task_update_begin=lambda s, h: calls.append(("begin", s, h)),
            graph_task_update_end=lambda s: calls.append(("end", s)))
        event = SimpleNamespace(record=lambda s: calls.append(("record", s)))
        probe.update_task(api, "stream", "handle", lambda: calls.append("PA"), event)
        self.assertEqual(calls, [("begin", "stream", "handle"), "PA",
                                 ("end", "stream"), ("record", "stream")])

    def test_update_failure_does_not_record_success_event(self):
        calls = []
        api = SimpleNamespace(graph_task_update_begin=lambda *a: None,
                              graph_task_update_end=lambda *a: calls.append("end"))
        event = SimpleNamespace(record=lambda *a: calls.append("record"))

        def fail():
            raise RuntimeError("PA first error")

        with self.assertRaisesRegex(RuntimeError, "PA first error"):
            probe.update_task(api, None, None, fail, event)
        self.assertEqual(calls, [])

    def test_expansion_padding_and_request_boundaries(self):
        blocks = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        contexts, tables = probe.expand(8, [3, 2], [5, 3], blocks)
        self.assertEqual(contexts.tolist(), [3, 4, 5, 2, 3, 1, 1, 1])
        self.assertEqual(tables[:, 0].tolist(), [1, 1, 1, 3, 3, 0, 0, 0])

    def test_reference_matches_known_causal_means(self):
        q = torch.zeros(3, 2, 16)
        k = torch.zeros(2, 4, 1, 16)
        v = torch.arange(8.).reshape(2, 4, 1, 1).expand_as(k)
        result = probe.reference(q, k, v, torch.tensor([[1]]), [3], [4])
        torch.testing.assert_close(result[:, 0, 0], torch.tensor([4.5, 5., 5.5]))

    def test_invalid_lengths_rejected(self):
        with self.assertRaises(ValueError):
            probe.expand(2, [3], [3], torch.zeros(1, 1, dtype=torch.int32))
        with self.assertRaises(ValueError):
            probe.expand(4, [3], [2], torch.zeros(1, 1, dtype=torch.int32))


if __name__ == "__main__":
    unittest.main()
