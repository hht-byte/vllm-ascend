# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run directly with Python; no vLLM installation or NPU is required."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/token_graph.py"
SPEC = importlib.util.spec_from_file_location("token_graph_under_test", SOURCE)
tg = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tg
SPEC.loader.exec_module(tg)

CASES = (
    (20, [1, 1, 1, 1, 1, 5]),
    (80, [1, 1, 1, 1, 5, 70]),
    (20, [1, 1, 1, 1, 7, 7]),
    (192, [70, 40, 70]),
    (20, [7, 7, 1, 1, 1, 1, 1, 1]),
    (20, [7, 5, 1, 1, 1, 1, 1, 1, 1, 1]),
)


def arena(layout="fixed", mode="task_update"):
    return tg.TokenGraphArena(192, 20, 4, 512, "cpu", tg.TokenGraphConfig(True, mode, layout))


def prepare(storage, bucket, qlens):
    blocks = torch.arange(80, dtype=torch.int32).reshape(20, 4)
    slots = torch.arange(sum(qlens), dtype=torch.int32) + 128
    return storage.prepare(bucket, qlens, [q + 128 for q in qlens], blocks, slots)


class FakeBackend:
    def __init__(self):
        self.npu = self
        self.capturing = False
        self.calls = []

    def is_current_stream_capturing(self):
        return self.capturing

    def current_stream(self):
        return self

    def graph_task_group_begin(self, stream):
        self.calls.append("group_begin")

    def graph_task_group_end(self, stream):
        self.calls.append("group_end")
        return "handle"

    def graph_task_update_begin(self, stream, handle):
        assert handle == "handle"
        self.calls.append("update_begin")

    def graph_task_update_end(self, stream):
        self.calls.append("update_end")

    def _npu_paged_attention_splitfuse_v2(self, **kwargs):
        self.calls.append((kwargs["seq_len"].tolist(), kwargs["block_table"].shape))


class TestTokenGraph(unittest.TestCase):
    def test_all_user_cases_preserve_requests_and_sum(self):
        for bucket, q in CASES:
            for layout in ("fixed", "active"):
                with self.subTest(q=q, layout=layout):
                    plan = tg.TokenGraphPlan.create(bucket, q, [x + 10 for x in q], 20, 512, layout)
                    self.assertEqual(plan.query_lens[: len(q)], tuple(q))
                    self.assertEqual(sum(plan.query_lens), bucket)
                    self.assertEqual(plan.query_start_loc[-1], bucket)
                    self.assertEqual(plan.actual_tokens, sum(q))
                    self.assertEqual(plan.actual_reqs, len(q))
                    self.assertTrue(all(c >= n for n, c in zip(plan.query_lens, plan.context_lens)))

    def test_same_bucket_same_state_and_tensor_addresses(self):
        storage = arena()
        state = prepare(storage, 20, CASES[4][1])
        pointers = {k: v.data_ptr() for k, v in state.metadata_kwargs().items()}
        shapes = {k: v.shape for k, v in state.metadata_kwargs().items()}
        for q in (CASES[5][1], CASES[0][1], CASES[2][1], CASES[4][1]):
            self.assertIs(prepare(storage, 20, q), state)
            self.assertEqual(pointers, {k: v.data_ptr() for k, v in state.metadata_kwargs().items()})
            self.assertEqual(shapes, {k: v.shape for k, v in state.metadata_kwargs().items()})
        self.assertEqual(len(storage.states), 1)

    def test_padding_slots_cleared_after_larger_batch(self):
        storage = arena()
        prepare(storage, 20, [1] * 20)
        state = prepare(storage, 20, [1, 1, 1, 1, 1, 5])
        self.assertEqual(storage.slots[:10].tolist(), list(range(128, 138)))
        self.assertEqual(storage.slots[10:20].tolist(), [-1] * 10)
        self.assertEqual(state.plan.query_lens[6], 10)
        self.assertTrue(torch.all(storage.blocks[6:20] == 0))

    def test_no_dummy_when_exact_and_full(self):
        plan = tg.TokenGraphPlan.create(20, [1] * 20, [1] * 20, 20, 512)
        self.assertEqual(plan.query_lens, (1,) * 20)

    def test_max_reqs_plus_one_dummy(self):
        plan = tg.TokenGraphPlan.create(80, [1] * 20, [2] * 20, 20, 512)
        self.assertEqual(len(plan.query_lens), 21)
        self.assertEqual(plan.query_lens[-1], 60)
        self.assertEqual(plan.context_lens[-1], 60)

    def test_bad_lengths_rejected(self):
        for q, c, bucket in (
            ([0], [1], 20),
            ([2], [1], 20),
            ([21], [21], 20),
            ([1], [513], 20),
            ([1, 1], [1], 20),
            ([], [], 20),
        ):
            with self.subTest(q=q, c=c), self.assertRaises(ValueError):
                tg.TokenGraphPlan.create(bucket, q, c, 20, 512)

    def test_descriptor_change_requires_task_update(self):
        with self.assertRaises(ValueError):
            tg.TokenGraphConfig.from_vllm(
                SimpleNamespace(
                    additional_config={
                        tg.CONFIG_KEY: {"enabled": True, "request_layout": "active", "update_mode": "inplace"}
                    }
                )
            )

    def test_unknown_config_rejected(self):
        with self.assertRaises(ValueError):
            tg.TokenGraphConfig.from_vllm(SimpleNamespace(additional_config={tg.CONFIG_KEY: {"enable": True}}))

    def test_scheduler_count_not_mutated_by_attach(self):
        storage = arena()
        state = prepare(storage, 20, [5, 5])
        metadata = SimpleNamespace()
        state.attach(metadata)
        self.assertEqual(metadata.num_actual_tokens, 20)
        self.assertEqual(state.plan.actual_tokens, 10)
        self.assertEqual(state.plan.actual_reqs, 2)
        self.assertEqual(metadata.slot_mapping.numel(), 20)
        self.assertEqual(metadata.slot_mapping.dtype, torch.int32)

    def test_active_update_uses_new_views_and_one_capture(self):
        storage = arena("active")
        state = prepare(storage, 20, CASES[4][1])
        backend = FakeBackend()
        backend.capturing = True
        query = torch.zeros(20, 2, 16)
        state.attention(backend, "layer0", query, query, query, query, query, 2, 2, 0.25)
        backend.capturing = False
        for q in (CASES[5][1], CASES[4][1]):
            prepare(storage, 20, q)
            state.update(backend)
            self.assertEqual(backend.calls[-2][0], q)
            self.assertEqual(backend.calls[-2][1][0], len(q))
        self.assertEqual(backend.calls.count("group_begin"), 1)
        self.assertEqual(backend.calls.count("update_begin"), 2)

    def test_inplace_does_not_issue_task_update(self):
        storage = arena(mode="inplace")
        state = prepare(storage, 20, CASES[4][1])
        backend = FakeBackend()
        state.update(backend)
        self.assertEqual(backend.calls, [])

    def test_other_bucket_shares_arena_not_task_handles(self):
        storage = arena()
        a = prepare(storage, 20, CASES[4][1])
        b = prepare(storage, 192, [70, 40, 70])
        self.assertIs(a.arena, b.arena)
        self.assertIsNot(a.tasks, b.tasks)
        self.assertEqual(len(storage.states), 2)

    def test_causal_prefix_reference(self):
        # Verify the plan preserves the independent per-request causal domains.
        q, contexts = [1, 1, 1, 1, 5, 70], [5, 20, 128, 129, 35, 100]
        plan = tg.TokenGraphPlan.create(80, q, contexts, 20, 512)
        for i, (n, c) in enumerate(zip(q, contexts)):
            begin, end = plan.query_start_loc[i : i + 2]
            self.assertEqual(end - begin, n)
            visible = list(range(c - n + 1, c + 1))
            self.assertEqual(visible[-1], c)
            self.assertEqual(len(visible), n)


if __name__ == "__main__":
    unittest.main()
