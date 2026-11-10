"""Packed attention boundaries and graph task-update contracts."""

import importlib.util
import unittest
from types import SimpleNamespace


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Torch")
class PackedAttentionTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.t = torch
        torch.set_num_threads(1)

    def backend(self, **kwargs):
        try:
            from asr_mtp_token_attention import PackedAttention
        except ImportError:
            self.fail("packed attention backend is not implemented")
        return PackedAttention(
            2,
            1,
            kwargs.pop("head_dim", 4),
            device="cpu",
            dtype=self.t.float32,
            heads=kwargs.pop("heads", 1),
            **kwargs,
        )

    def test_nonpa_tnd_rejects_unaccounted_physical_kv_capacity(self):
        attention = self.backend()
        attention.stage(0, [8], [1280])
        with self.assertRaises(ValueError):
            attention(
                0,
                self.t.zeros(8, 2, 4),
                self.t.zeros(2304, 1, 4),
                self.t.zeros(2304, 1, 4),
                0.5,
            )

    def test_packed_gqa_uses_positive_dummy_for_entire_physical_kv_padding(self):
        attention = self.backend()
        attention.stage(0, [2, 3, 5], [4, 6, 10])
        q = self.t.zeros(5, 2, 4)
        k = self.t.zeros(10, 1, 4)
        v = self.t.tensor([1, 3, 5, 7, 10, 20, 0, 0, 0, 0], dtype=q.dtype)
        v = v[:, None, None].expand(-1, 1, 4).contiguous()
        output = attention(0, q, k, v, 0.5)
        expected = self.t.tensor([3, 4, 15, 0, 0], dtype=q.dtype)[:, None, None].expand_as(q)
        self.t.testing.assert_close(output, expected)

    def test_metadata_rejection_preserves_previous_valid_segments(self):
        attention = self.backend()
        attention.stage(0, [1, 2], [2, 4])
        for qends, kvends in (([2, 1], [2, 4]), ([1], [2, 4]), ([2], [1])):
            with self.subTest(qends=qends), self.assertRaises(ValueError):
                attention.stage(0, qends, kvends)
        q = self.t.zeros(2, 2, 4)
        k = self.t.zeros(4, 1, 4)
        v = self.t.tensor([2, 4, 10, 20], dtype=q.dtype)[:, None, None].expand(-1, 1, 4)
        self.t.testing.assert_close(
            attention(0, q, k, v, 1.0),
            self.t.tensor([3, 15], dtype=q.dtype)[:, None, None].expand_as(q),
        )

    def test_staged_boundaries_change_without_changing_tensor_capacities(self):
        attention = self.backend()
        q = self.t.zeros(4, 2, 4)
        k = self.t.zeros(8, 1, 4)
        v = self.t.tensor([1, 3, 5, 7, 10, 20, 0, 0], dtype=q.dtype)[:, None, None].expand(-1, 1, 4)
        attention.stage(0, [2, 3, 4], [4, 6, 8])
        first = attention(0, q, k, v, 1.0)
        attention.stage(0, [1, 3, 4], [2, 6, 8])
        second = attention(0, q, k, v, 1.0)
        self.t.testing.assert_close(
            first,
            self.t.tensor([3, 4, 15, 0], dtype=q.dtype)[:, None, None].expand_as(q),
        )
        self.t.testing.assert_close(second, self.t.tensor([2, 22 / 3, 10.5, 0])[:, None, None].expand_as(q))

    def test_call_rejects_uncovered_query_and_kv_capacities(self):
        attention = self.backend()
        attention.stage(0, [3], [4])
        with self.assertRaises(ValueError):
            attention(
                0,
                self.t.zeros(2, 2, 4),
                self.t.zeros(4, 1, 4),
                self.t.zeros(4, 1, 4),
                1.0,
            )
        attention.stage(0, [2], [5])
        with self.assertRaises(ValueError):
            attention(
                0,
                self.t.zeros(2, 2, 4),
                self.t.zeros(4, 1, 4),
                self.t.zeros(4, 1, 4),
                1.0,
            )

    def test_npu_task_updates_reuse_preallocated_outputs_and_workspace(self):
        runtime = FakeRuntime()
        api = FakeAPI(self.t, runtime)
        attention = self.backend(head_dim=128, heads=2, npu_api=api, npu_runtime=runtime)
        q = self.t.zeros(4, 2, 128)
        k = self.t.zeros(8, 1, 128)
        v = self.t.zeros_like(k)
        for head in range(2):
            attention.stage(head, [1, 3, 4], [2, 6, 8])
            attention(head, q, k, v, 0.25)
        attention.begin_capture()
        outputs = [attention(head, q, k, v, 0.25) for head in range(2)]
        attention.end_capture()
        before = api.workspaces
        for head in range(2):
            attention.stage(head, [2, 3, 4], [4, 6, 8])
        attention.update()
        self.assertEqual(api.workspaces, before)
        updates = api.calls[-2:]
        for head, call in enumerate(updates):
            self.assertEqual(call["actual_seq_lengths"], (2, 3, 4))
            self.assertEqual(call["actual_seq_lengths_kv"], (4, 6, 8))
            self.assertEqual(call["input_layout"], "TND")
            self.assertEqual(call["sparse_mode"], 3)
            self.assertEqual(call["out"][0].data_ptr(), outputs[head].data_ptr())
        self.assertEqual(runtime.events.count("update_begin"), 2)
        self.assertEqual(runtime.events.count("event_record"), 2)

    def test_npu_update_validates_every_head_before_updating_any_task(self):
        runtime = FakeRuntime()
        api = FakeAPI(self.t, runtime)
        attention = self.backend(head_dim=128, heads=2, npu_api=api, npu_runtime=runtime)
        q = self.t.zeros(2, 2, 128)
        k = self.t.zeros(4, 1, 128)
        for head in range(2):
            attention.stage(head, [2], [4])
            attention(head, q, k, k, 1.0)
        attention.begin_capture()
        for head in range(2):
            attention(head, q, k, k, 1.0)
        attention.end_capture()
        attention.stage(1, [3], [4])
        with self.assertRaises(ValueError):
            attention.update()
        self.assertNotIn("update_begin", runtime.events)

    def test_backend_device_alias_is_resolved_to_the_allocated_mask_device(self):
        from asr_mtp_token_attention import PackedAttention

        runtime = FakeRuntime()
        api = FakeAPI(self.t, runtime)
        attention = PackedAttention(
            2,
            1,
            128,
            device="cpu:0",
            dtype=self.t.float32,
            heads=1,
            npu_api=api,
            npu_runtime=runtime,
        )
        attention.stage(0, [2], [4])
        query = self.t.zeros(2, 2, 128)
        key = self.t.zeros(4, 1, 128)
        self.assertEqual(attention(0, query, key, key, 1.0).device, query.device)


class FakeEvent:
    def __init__(self, runtime):
        self.runtime = runtime

    def wait(self, stream):
        self.runtime.events.append("event_wait")

    def reset(self, stream):
        self.runtime.events.append("event_reset")

    def record(self, stream):
        self.runtime.events.append("event_record")


class FakeRuntime:
    def __init__(self):
        self.events = []
        self.stream = object()

    def current_stream(self):
        return self.stream

    def ExternalEvent(self):
        return FakeEvent(self)

    def graph_task_group_begin(self, stream):
        self.events.append("group_begin")

    def graph_task_group_end(self, stream):
        self.events.append("group_end")
        return object()

    def graph_task_update_begin(self, stream, handle):
        self.events.append("update_begin")

    def graph_task_update_end(self, stream):
        self.events.append("update_end")


class FakeAPI:
    def __init__(self, torch, runtime):
        self.t, self.runtime = torch, runtime
        self.workspaces = 0
        self.calls = []
        self.npu_fused_infer_attention_score = SimpleNamespace(out=self.out)

    def _npu_fused_infer_attention_score_get_max_workspace(self, **kwargs):
        self.validate_tnd(kwargs)
        self.workspaces += 1
        return self.t.empty(64, dtype=self.t.uint8)

    def out(self, **kwargs):
        self.validate_tnd(kwargs)
        self.calls.append(
            {
                **kwargs,
                "actual_seq_lengths": tuple(kwargs["actual_seq_lengths"]),
                "actual_seq_lengths_kv": tuple(kwargs["actual_seq_lengths_kv"]),
            }
        )
        kwargs["out"][0].zero_()
        return kwargs["out"]

    def validate_tnd(self, kwargs):
        if kwargs.get("input_layout") == "TND":
            assert kwargs["actual_seq_lengths"][-1] == kwargs["query"].shape[0]
            assert kwargs["actual_seq_lengths_kv"][-1] == kwargs["key"].shape[0]
            assert kwargs["actual_seq_lengths_kv"][-1] == kwargs["value"].shape[0]


def npu_available():
    if importlib.util.find_spec("torch_npu") is None:
        return False
    import torch
    import torch_npu  # noqa: F401

    return torch.npu.is_available()


@unittest.skipUnless(npu_available(), "requires torch-npu and an available NPU")
class ActualPackedAttentionReplayTests(unittest.TestCase):
    def test_task_update_changes_positive_segments_while_covering_physical_kv(self):
        import torch
        from asr_mtp_token_attention import PackedAttention, PackedNPUCapture

        class Program:
            def __init__(self):
                self.attention = PackedAttention(2, 1, 128, device="npu", dtype=torch.float16, heads=1)
                self.query = torch.randn(9, 2, 128, device="npu", dtype=torch.float16)
                self.key = torch.randn(20, 1, 128, device="npu", dtype=torch.float16)
                self.value = torch.randn_like(self.key)

            def run(self):
                self.output = self.attention(0, self.query, self.key, self.value, 128**-0.5)
                return self.output

        torch.manual_seed(318)
        with torch.inference_mode():
            program = Program()
            program.query[8:] = program.key[12:] = program.value[12:] = 0
            program.attention.stage(0, [2, 5, 8, 9], [4, 9, 12, 20])
            replay = PackedNPUCapture().capture(program)
            reference = PackedAttention(2, 1, 128, device="cpu", dtype=torch.float32, heads=1)
            for qends, kvends in (
                ([2, 5, 8, 9], [4, 9, 12, 20]),
                ([1, 8, 9], [3, 10, 20]),
                ([8, 9], [12, 20]),
            ):
                program.query.copy_(torch.randn_like(program.query))
                program.key.copy_(torch.randn_like(program.key))
                program.value.copy_(torch.randn_like(program.value))
                program.query[qends[-2] :] = 0
                program.key[kvends[-2] :] = program.value[kvends[-2] :] = 0
                program.attention.stage(0, qends, kvends)
                replay()
                reference.stage(0, qends, kvends)
                expected = reference(
                    0,
                    program.query.cpu().float(),
                    program.key.cpu().float(),
                    program.value.cpu().float(),
                    128**-0.5,
                )
                torch.testing.assert_close(program.output.cpu().float(), expected, atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    unittest.main()
