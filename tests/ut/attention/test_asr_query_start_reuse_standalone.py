# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the installed builder body with CPU tensors, without importing NPU/vLLM."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch


def load_build():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/attention_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendAttentionMetadataBuilder")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "build")
    fn.decorator_list = []
    namespace = {
        "torch": torch,
        "AscendMetadata": NS,
        "CrossAttentionSpec": type("CrossAttentionSpec", (), {}),
        "split_decodes_and_prefills": lambda metadata, **kwargs: (metadata.num_reqs, 0, metadata.num_actual_tokens, 0),
    }
    exec(compile("from __future__ import annotations\n" + ast.unparse(fn), str(source), "exec"), namespace)
    return namespace["build"]


def common(qsl, *, capacity=None):
    n = len(qsl) - 1
    rows = n if capacity is None else capacity
    gpu = torch.tensor(qsl + [999, 999], dtype=torch.int32)
    return NS(
        num_reqs=n,
        num_actual_tokens=qsl[-1],
        query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
        query_start_loc=gpu,
        _seq_lens_cpu=torch.arange(rows, dtype=torch.int32) + 20,
        seq_lens_cpu=None,
        seq_lens=torch.arange(rows, dtype=torch.int32) + 20,
        block_table_tensor=torch.arange(rows * 3, dtype=torch.int32).view(rows, 3),
        slot_mapping=torch.arange(qsl[-1], dtype=torch.int64),
        attn_state="mixed",
        causal=True,
        max_query_len=max(b - a for a, b in zip(qsl, qsl[1:])),
        kvcomp_metadata=None,
    )


class QueryStartReuseTests(unittest.TestCase):
    def setUp(self):
        self.build = load_build()
        self.builder = NS(
            decode_threshold=6,
            speculative_config=None,
            kv_cache_spec=object(),
            attn_mask_builder=NS(get_attention_mask=lambda *args: None),
            model_config=NS(runner_type="generate"),
            device=torch.device("cpu"),
        )

    def test_reuse_skips_host_upload_and_preserves_dummy_padding(self):
        cm = common([0, 3, 5, 8])
        with patch.object(torch.Tensor, "pin_memory", side_effect=AssertionError("redundant qsl upload")):
            result = self.build(self.builder, 0, cm, reuse_query_start_loc=True)
        self.assertEqual(result.query_start_loc.data_ptr(), cm.query_start_loc.data_ptr())
        self.assertEqual(result.query_start_loc.tolist(), [0, 3, 5, 8])
        self.assertEqual(result.actual_seq_lengths_q, [3, 5, 8])
        self.assertEqual(result.seq_lens_list, [20, 21, 22])

    def test_original_and_fast_build_keep_original_snapshot(self):
        for kwargs in ({}, {"fast_build": True}, {"reuse_query_start_loc": False}):
            cm = common([0, 3, 5])
            calls = []

            def snapshot(tensor, *args, calls=calls, **kwargs):
                calls.append(1)
                return tensor.clone()

            with patch.object(torch.Tensor, "pin_memory", snapshot):
                result = self.build(self.builder, 0, cm, **kwargs)
            self.assertEqual(calls, [1])
            self.assertNotEqual(result.query_start_loc.data_ptr(), cm.query_start_loc.data_ptr())
            torch.testing.assert_close(result.query_start_loc, cm.query_start_loc_cpu)

    def test_shrinking_and_mixed_batches_slice_current_storage(self):
        for qsl in ([0, 59, 63, 64], [0, 4], [0, 3, 8, 11, 16]):
            cm = common(list(qsl))
            with patch.object(torch.Tensor, "pin_memory", side_effect=AssertionError("unexpected pin")):
                result = self.build(self.builder, 0, cm, reuse_query_start_loc=True)
            self.assertEqual(result.query_start_loc.shape, (len(qsl),))
            self.assertEqual(result.query_start_loc.tolist(), list(qsl))
            self.assertEqual(result.actual_seq_lengths_q, list(qsl[1:]))

    def test_full_request_capacity_still_pads_kv_and_block_rows(self):
        cm = common([0, 3, 5, 8], capacity=2)
        with patch.object(torch.Tensor, "pin_memory", side_effect=AssertionError("unexpected pin")):
            result = self.build(self.builder, 0, cm, reuse_query_start_loc=True)
        self.assertEqual(result.seq_lens_list, [20, 21, 1])
        self.assertEqual(result.block_tables.tolist(), [[0, 1, 2], [3, 4, 5], [0, 0, 0]])
        self.assertEqual(result.query_start_loc.tolist(), [0, 3, 5, 8])


if __name__ == "__main__":
    unittest.main()
