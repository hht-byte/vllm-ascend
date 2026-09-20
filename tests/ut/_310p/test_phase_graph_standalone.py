# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for native attention graph routing and metadata."""

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("phase_token_test", ROOT / "vllm_ascend/_310p/token_graph.py")
tg = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tg
spec.loader.exec_module(tg)


class TestPhaseGraphs(unittest.TestCase):
    def test_real_prefill_forward_passes_host_lengths_to_flash(self):
        source = ROOT / "vllm_ascend/_310p/attention/attention_v1.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "AscendAttentionBackendImpl310")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_prefill_310")
        calls = []
        env = {"torch_npu": SimpleNamespace(_npu_flash_attention=lambda **kwargs: calls.append(kwargs))}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), env)
        lengths = torch.tensor([20], dtype=torch.int32)
        metadata = SimpleNamespace(native_graph_state=SimpleNamespace(flash_seq_lens=lengths),
                                   seq_lens=object(), attn_mask=object())
        impl = SimpleNamespace(num_heads=8, num_kv_heads=2, scale=0.125)
        output = object()
        self.assertIs(env["forward_prefill_310"](impl, None, None, None, metadata, output), output)
        self.assertIs(calls[0]["seq_len"], lengths)
        self.assertIs(calls[0]["mask"], metadata.attn_mask)

    def test_normal_masks_are_aligned_and_owned_by_bucket(self):
        arena = tg.NativeGraphArena(80, 10, 4, 512, "cpu", "prefill")
        blocks = torch.zeros(10, 4, dtype=torch.int32)
        small = arena.prepare(20, [7, 5], [7, 5], blocks, torch.arange(12))
        meta = SimpleNamespace()
        small.attach(meta)
        mask = meta.attn_mask
        self.assertEqual(tuple(mask.shape), (1, 2, 32, 16))
        large = arena.prepare(80, [70], [70], blocks, torch.arange(70))
        large.attach(meta)
        self.assertEqual(tuple(meta.attn_mask.shape), (1, 5, 80, 16))
        small.attach(meta)
        self.assertIs(meta.attn_mask, mask)
        again = arena.prepare(20, [2] * 10, [2] * 10, blocks, torch.arange(20))
        again.attach(meta)
        self.assertIs(meta.attn_mask, mask)

    def test_flash_lengths_are_immutable_host_tensors_owned_by_each_bucket(self):
        arena = tg.NativeGraphArena(80, 10, 4, 512, "cpu", "prefill")
        blocks = torch.zeros(10, 4, dtype=torch.int32)
        small = arena.prepare(20, [7, 5], [7, 5], blocks, torch.arange(12))
        lengths = small.flash_seq_lens
        self.assertEqual(lengths.device.type, "cpu")
        self.assertEqual(lengths.dtype, torch.int32)
        self.assertEqual(lengths.tolist(), [20])
        large = arena.prepare(80, [70], [70], blocks, torch.arange(70))
        self.assertEqual(large.flash_seq_lens.tolist(), [80])
        self.assertEqual(lengths.tolist(), [20])
        self.assertNotEqual(large.flash_seq_lens.data_ptr(), lengths.data_ptr())
        again = arena.prepare(20, [2] * 10, [2] * 10, blocks, torch.arange(20))
        self.assertIs(again.flash_seq_lens, lengths)

    def test_packed_flash_mask_matches_independent_request_attention(self):
        torch.manual_seed(310)
        arena = tg.NativeGraphArena(20, 10, 4, 512, "cpu", "prefill")
        blocks = torch.zeros(10, 4, dtype=torch.int32)
        query, key, value = (torch.randn(20, 8) for _ in range(3))
        for lengths in ([7, 5], [2] * 10, [3] * 4 + [2] * 4, [20]):
            arena.prepare(20, lengths, lengths, blocks, torch.arange(sum(lengths)))
            scores = query @ key.T / 8**0.5
            actual = (scores + arena.prefill_mask_cpu[:20, :20]).softmax(-1) @ value
            offset = 0
            for n in lengths:
                q, k, v = (t[offset:offset + n] for t in (query, key, value))
                mask = torch.ones(n, n, dtype=torch.bool).triu(1)
                expected = (q @ k.T / 8**0.5).masked_fill(mask, float("-inf")).softmax(-1) @ v
                torch.testing.assert_close(actual[offset:offset + n], expected)
                offset += n

    def test_phase_is_not_inferred_from_query_length_alone(self):
        cases = [
            (([1], [0], [1]), "prefill"),
            (([102], [0], [102]), "prefill"),
            (([20], [10], [100]), "token"),
            (([3, 4], [0, 10], [3, 30]), "token"),
            (([1, 1], [44, 71], [44, 70]), "decode"),
            (([1, 5], [44, 0], [44, 5]), "token"),
            (([3, 3], [44, 70], [44, 70]), "token"),
            (([3], [9], [10]), "token"),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(tg.classify_graph_family(*args), expected)

    def test_native_prefill_preserves_request_layout_and_safe_padding(self):
        arena = tg.NativeGraphArena(20, 10, 4, 512, "cpu", "prefill")
        blocks = torch.zeros(10, 4, dtype=torch.int32)
        state = arena.prepare(20, [7, 5], [7, 5], blocks, torch.arange(12))
        meta = SimpleNamespace()
        state.attach(meta)
        self.assertEqual(state.plan.query_lens, (20,))
        self.assertEqual(meta.seq_lens.tolist(), [20])
        self.assertEqual(meta.query_lens_cpu.tolist(), [20])
        mask = arena.prefill_mask_cpu
        self.assertEqual(mask[6, 0], 0)
        self.assertTrue(torch.isneginf(mask[7, 0]))
        self.assertTrue(torch.isneginf(mask[12, 7]))
        self.assertTrue(torch.isneginf(mask[0, 1]))
        self.assertEqual(mask[19, 12], 0)
        self.assertFalse(hasattr(meta, "token_graph_state"))
        self.assertIs(meta.native_graph_state, state)
        self.assertEqual(meta.slot_mapping.tolist(), list(range(12)) + [-1] * 8)

    def test_decode_reuses_bucket_across_request_counts(self):
        arena = tg.NativeGraphArena(20, 10, 4, 512, "cpu", "decode")
        blocks = torch.arange(40, dtype=torch.int32).reshape(10, 4)
        state = arena.prepare(20, [1] * 8, [64] * 8, blocks, torch.arange(8))
        ptr = state.arena.context.data_ptr()
        again = arena.prepare(20, [1] * 10, [130] * 10, blocks, torch.arange(10))
        self.assertIs(state, again)
        self.assertEqual(again.plan.query_lens, (1,) * 20)
        self.assertEqual(again.arena.context.data_ptr(), ptr)
        self.assertEqual(again.plan.context_lens, (130,) * 10 + (1,) * 10)

    def test_prefill_updates_fixed_shape_device_lengths_in_place(self):
        arena = tg.NativeGraphArena(20, 10, 4, 512, "cpu", "prefill")
        blocks = torch.zeros(10, 4, dtype=torch.int32)
        q = [3] * 4 + [2] * 4
        state = arena.prepare(20, q, q, blocks, torch.arange(20))
        meta = SimpleNamespace()
        state.attach(meta)
        ptr, shape = meta.seq_lens.data_ptr(), meta.seq_lens.shape
        mask_ptr = meta.attn_mask.data_ptr()
        q = [2] * 10
        again = arena.prepare(20, q, q, blocks, torch.arange(20))
        again.attach(meta)
        self.assertIs(state, again)
        self.assertEqual(meta.seq_lens.data_ptr(), ptr)
        self.assertEqual(meta.seq_lens.shape, shape)
        self.assertEqual(meta.attn_mask.data_ptr(), mask_ptr)
        self.assertEqual(meta.seq_lens.tolist(), [20])
        self.assertTrue(torch.isneginf(arena.prefill_mask_cpu[2, 0]))
        self.assertEqual(list(arena.states), [20])


if __name__ == "__main__":
    unittest.main()
