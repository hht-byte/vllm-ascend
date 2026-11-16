# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU authoritative slots must agree with device lookup across batch changes."""

import unittest
from unittest.mock import patch

import numpy as np
import torch
from test_asr_input_prepare_graph_standalone import CPUCapture, load, npu_available, runner


class CPUSlotsTests(unittest.TestCase):
    def make(self, enabled=True):
        r = runner()
        r.vllm_config.additional_config["asr_input_prepare_graph"]["cpu_slot_mapping"] = enabled
        table = r.input_batch.block_table.block_tables[0]
        table.block_table.np = table.block_table.gpu.numpy().copy()
        graph = load().InputPrepareGraph(r, (8, 16, 32), backend=CPUCapture())
        graph.capture()
        return r, table, graph

    def test_reordered_blocks_shrinking_and_dummy_padding_are_packed_without_slot_kernel(self):
        r, table, g = self.make()
        self.assertTrue(getattr(g, "packs_block_tables", False), "CPU slot packing not implemented")
        with patch.object(
            r.input_batch.block_table, "compute_slot_mapping", side_effect=AssertionError("device slot kernel")
        ):
            for counts, lengths, blocks in [
                ([3, 11], [6, 2], [9, 3]),
                ([7], [2], [21]),
                ([1, 5, 9], [2, 1, 2], [7, 2, 11]),
            ]:
                n, t = len(counts), sum(lengths)
                table.block_table.np[:n] = np.array(blocks)[:, None] + np.arange(16)
                pos = np.concatenate([np.arange(c, c + q) for c, q in zip(counts, lengths)])
                qsl = np.r_[0, np.cumsum(lengths)]
                g.stage(g.select(t), counts, np.array(counts) + lengths, pos, qsl, [False] * n, [])
                g.finish(16, n + 1)
                expected = [
                    int(table.block_table.np[i, p // 4]) * 4 + p % 4
                    for i, (c, q) in enumerate(zip(counts, lengths))
                    for p in range(c, c + q)
                ]
                self.assertEqual(table.slot_mapping.gpu[:32].tolist(), expected + [-1] * (32 - t))
                torch.testing.assert_close(table.block_table.gpu[:n], torch.from_numpy(table.block_table.np[:n]))
                self.assertEqual(table.block_table.gpu[n:].count_nonzero().item(), 0)
                self.assertEqual(r.query_start_loc.gpu[: n + 2].tolist(), qsl.tolist() + [16])

    def test_option_off_preserves_device_lookup(self):
        r, _, g = self.make(False)
        self.assertFalse(getattr(g, "packs_block_tables", False))
        with patch.object(
            r.input_batch.block_table, "compute_slot_mapping", wraps=r.input_batch.block_table.compute_slot_mapping
        ) as compute:
            g.replay(g.select(2), [3], [5], [3, 4], [0, 2], [False], [])
        self.assertEqual(compute.call_count, 1)

    def test_multiple_groups_use_their_own_block_size_and_padded_tail(self):
        from copy import deepcopy

        r = runner()
        first = r.input_batch.block_table.block_tables[0]
        second = deepcopy(first)
        second.block_size = 8
        for i, table in enumerate((first, second)):
            table.block_table.np = table.block_table.gpu.numpy().copy() + i * 100
        r.input_batch.block_table.block_tables.append(second)
        g = load().InputPrepareGraph(r, (8, 16, 32), backend=CPUCapture())
        g.capture()
        g.stage(g.select(3), [7], [10], [7, 8, 9], [0, 3], [False], [])
        g.finish(8, 2)
        for table in (first, second):
            bs = table.block_size
            expected = [int(table.block_table.np[0, p // bs]) * bs + p % bs for p in (7, 8, 9)]
            self.assertEqual(table.slot_mapping.gpu[:32].tolist(), expected + [-1] * 29)

    def test_runner_keeps_block_upload_on_graph_misses_and_option_off(self):
        import textwrap
        from pathlib import Path
        from types import SimpleNamespace as NS
        from unittest.mock import Mock

        source = (Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py").read_text(
            encoding="utf-8"
        )
        begin = source.index("        # OPTIMIZATION: Start copying the block table first.")
        end = source.index("        req_indices =", begin)
        code = compile(textwrap.dedent(source[begin:end]), "<actual block upload ownership>", "exec")
        for bucket, owner, calls in [
            (None, None, 1),
            (None, NS(packs_block_tables=True), 1),
            (8, NS(packs_block_tables=False), 1),
            (8, NS(packs_block_tables=True), 0),
        ]:
            commit = Mock()
            r = NS(_input_prepare_graph=owner, input_batch=NS(block_table=NS(commit_block_table=commit)))
            exec(code, {"self": r, "input_prepare_bucket": bucket, "num_reqs": 2})
            self.assertEqual(commit.call_count, calls)


@unittest.skipUnless(npu_available(), "requires Ascend NPU")
class NPUSlotsTests(unittest.TestCase):
    def test_graph_replay_new_blocks_reorder_shrink_and_full_dummy(self):
        r = runner()
        r.device, r.pin_memory = "npu", True
        for name in ("positions", "seq_lens", "num_computed_tokens"):
            setattr(r, name, getattr(r, name).to("npu"))
        for name in ("query_start_loc", "num_accepted_tokens", "discard_request_mask", "discard_request_indices"):
            getattr(r, name).gpu = getattr(r, name).gpu.to("npu")
        table = r.input_batch.block_table.block_tables[0]
        table.block_table.np = table.block_table.gpu.numpy().copy()
        table.block_table.gpu = table.block_table.gpu.to("npu")
        table.slot_mapping.gpu = table.slot_mapping.gpu.to("npu")
        g = load().InputPrepareGraph(r, (8, 16, 32))
        g.capture()
        with patch.object(
            r.input_batch.block_table, "compute_slot_mapping", side_effect=AssertionError("unexpected slot kernel")
        ):
            for counts, lengths in [([3, 11], [6, 2]), ([7], [2]), ([1, 5, 9], [2, 1, 2])]:
                n, t = len(counts), sum(lengths)
                table.block_table.np[:n] += 1
                pos = np.concatenate([np.arange(c, c + q) for c, q in zip(counts, lengths)])
                qsl = np.r_[0, np.cumsum(lengths)]
                g.stage(g.select(t), counts, np.array(counts) + lengths, pos, qsl, [False] * n, [])
                g.finish(16, n + 1)
                expected = [
                    int(table.block_table.np[i, p // 4]) * 4 + p % 4
                    for i, (c, q) in enumerate(zip(counts, lengths))
                    for p in range(c, c + q)
                ]
                self.assertEqual(table.slot_mapping.gpu[:32].cpu().tolist(), expected + [-1] * (32 - t))
                self.assertEqual(r.query_start_loc.gpu[: n + 2].cpu().tolist(), qsl.tolist() + [16])


if __name__ == "__main__":
    unittest.main()
