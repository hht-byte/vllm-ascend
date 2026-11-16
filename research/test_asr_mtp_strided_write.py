"""Packed QKV row-write correctness; CPU checks are not NPU acceptance."""

import importlib.util
import unittest
from types import SimpleNamespace


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Torch")
class StridedRowWriteTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.t = torch
        torch.manual_seed(113)
        torch.set_num_threads(1)

    def _check_fused_value_write(self, *, packed, contiguous):
        from asr_mtp_fused import prepare_fused_heads
        from asr_mtp_graph import FixedMTPProgram
        from asr_mtp_heads import BatchedMTPHeads, project_norm_rope
        from asr_mtp_row_copy import _geometry
        from asr_mtp_token_graph import PackedMTPProgram

        torch = self.t
        width, dim = (256, 128) if packed else (128, 64)
        leading = (9,) if packed else (1, 8)
        cfg = SimpleNamespace(
            hidden_size=width,
            intermediate_size=2 * width,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=dim,
            attention_bias=False,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            rope_theta=10000,
            rope_scaling=None,
        )
        model = BatchedMTPHeads(cfg, 2).eval().requires_grad_(False)
        prepare_fused_heads(model)
        hidden = torch.randn(*leading, width)
        anchors = torch.zeros(leading, dtype=torch.long)
        inverse = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        with torch.inference_mode():
            _, _, value = project_norm_rope(
                model, model.blocks[0].decoder.self_attn, hidden, anchors, 1,
                inverse, 32,
            )
        if not packed:
            value = value.transpose(1, 2)
        # V remains a view of the packed QKV projection. Neither the head reshape
        # nor the fixed program's transpose removes the intervening Q/K rows.
        self.assertFalse(value.reshape(-1, dim).is_contiguous())
        if contiguous:
            value = value.contiguous()
        source_rows = value.reshape(-1, dim)
        rows = source_rows.shape[0]
        shape = (*value.shape[:-1], 1)
        indices = (2 * torch.arange(rows)).reshape(shape)
        sinks = (2 * rows + torch.arange(rows)).reshape(shape)
        valid = (torch.arange(rows) % 2 == 0).reshape(shape)
        cache = torch.full((3 * rows, dim), -19.0)
        expected = cache.clone()
        for row in range(rows):
            destination = 2 * row if row % 2 == 0 else 2 * rows + row
            expected[destination].copy_(source_rows[row])

        def validating_device_write(destination, addresses, updates):
            self.assertTrue(updates.is_contiguous(), "strided packed V reached device writer")
            _geometry(destination, addresses, updates)
            if contiguous:
                self.assertEqual(updates.data_ptr(), source_rows.data_ptr())
            destination.index_copy_(0, addresses.reshape(-1), updates)

        program_type = PackedMTPProgram if packed else FixedMTPProgram
        program = program_type.__new__(program_type)
        program.row_width = dim
        program.cache_rows = cache
        program.npu_row_update = validating_device_write
        program._write(indices, value, valid, sinks)
        torch.testing.assert_close(cache, expected, rtol=0, atol=0)

    def test_fixed_packs_strided_fused_value_rows_before_device_write(self):
        self._check_fused_value_write(packed=False, contiguous=False)

    def test_packed_packs_strided_fused_value_rows_before_device_write(self):
        self._check_fused_value_write(packed=True, contiguous=False)

    def test_fixed_contiguous_rows_retain_source_storage(self):
        self._check_fused_value_write(packed=False, contiguous=True)

    def test_packed_contiguous_rows_retain_source_storage(self):
        self._check_fused_value_write(packed=True, contiguous=True)


if __name__ == "__main__":
    unittest.main()
