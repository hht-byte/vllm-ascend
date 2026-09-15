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

    def test_enabled_defaults_to_token_inplace(self):
        config = tg.TokenGraphConfig.from_vllm(SimpleNamespace(additional_config={tg.CONFIG_KEY: {"enabled": True}}))
        self.assertEqual(config.update_mode, "inplace")
        self.assertEqual(config.request_layout, "token")

    def test_failed_requestwise_inplace_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "token"):
            tg.TokenGraphConfig.from_vllm(
                SimpleNamespace(
                    additional_config={
                        tg.CONFIG_KEY: {"enabled": True, "request_layout": "fixed", "update_mode": "inplace"}
                    }
                )
            )

    def test_token_rows_preserve_causal_context_and_padding(self):
        plan = tg.TokenGraphPlan.create(8, [3, 2], [130, 5], 20, 512, "token")
        self.assertEqual(plan.query_lens, (1,) * 8)
        self.assertEqual(plan.context_lens, (128, 129, 130, 4, 5, 1, 1, 1))
        self.assertEqual(plan.row_requests, (0, 0, 0, 1, 1, -1, -1, -1))
        self.assertEqual(plan.query_start_loc, tuple(range(9)))
        self.assertEqual((plan.actual_tokens, plan.actual_reqs), (5, 2))

    def test_token_bucket_switches_keep_shapes_addresses_and_unit_qlens(self):
        storage = arena("token", "inplace")
        pointers = None
        for bucket, q in (*CASES, CASES[0]):
            state = prepare(storage, bucket, q)
            self.assertEqual(state.metadata_kwargs()["seq_len"].tolist(), [1] * bucket)
            self.assertEqual(state.metadata_kwargs()["block_table"].shape, (bucket, 4))
            current = {k: v.data_ptr() for k, v in state.metadata_kwargs().items()}
            if pointers is None:
                pointers = current
            self.assertEqual(current, pointers)
            for token, owner in enumerate(state.plan.row_requests):
                expected = [0] * 4 if owner < 0 else list(range(owner * 4, owner * 4 + 4))
                self.assertEqual(storage.blocks[token].tolist(), expected)
            self.assertTrue(torch.all(storage.slots[sum(q) : bucket] == -1))

    def test_token_rows_match_multiquery_causal_attention(self):
        generator = torch.Generator().manual_seed(310)
        keys = torch.randn(80, 64, 4, generator=generator)
        values = torch.randn(80, 64, 4, generator=generator)
        blocks = torch.arange(80, dtype=torch.int32).reshape(20, 4)
        storage = arena("token", "inplace")
        for bucket, q in CASES:
            contexts = [n + 7 for n in q]
            queries = torch.randn(sum(q), 4, generator=generator)
            state = storage.prepare(bucket, q, contexts, blocks, torch.arange(sum(q)))
            expected = []
            offset = 0
            for row, (length, context) in enumerate(zip(q, contexts)):
                k = keys[blocks[row].long()].flatten(0, 1)[:context]
                v = values[blocks[row].long()].flatten(0, 1)[:context]
                scores = queries[offset : offset + length] @ k.T / 2
                visible = torch.arange(context)[None, :] <= torch.arange(context - length, context)[:, None]
                expected.append(scores.masked_fill(~visible, float("-inf")).softmax(-1) @ v)
                offset += length
            actual = []
            for token in range(sum(q)):
                c = state.plan.context_lens[token]
                k = keys[storage.blocks[token].long()].flatten(0, 1)[:c]
                v = values[storage.blocks[token].long()].flatten(0, 1)[:c]
                actual.append((queries[token] @ k.T / 2).softmax(-1) @ v)
            torch.testing.assert_close(torch.stack(actual), torch.cat(expected), rtol=1e-5, atol=1e-6)

    def test_token_capture_views_read_new_metadata_without_task_update(self):
        storage = arena("token", "inplace")
        state = prepare(storage, 20, CASES[4][1])
        metadata = SimpleNamespace()
        state.attach(metadata)
        slot_view, starts_view = metadata.slot_mapping, metadata.query_start_loc
        backend = FakeBackend()
        backend.capturing = True
        query = torch.zeros(20, 2, 16)
        state.attention(backend, "layer0", query, query, query, query, query, 2, 2, 0.25)
        backend.capturing = False
        captured = state.tasks["layer0"].kwargs
        for q in (CASES[5][1], CASES[0][1], CASES[4][1]):
            prepare(storage, 20, q)
            state.update(backend)
            self.assertEqual(captured["seq_len"].tolist(), [1] * 20)
            self.assertEqual(captured["context_lens"].tolist(), list(state.plan.context_lens))
            self.assertEqual(captured["block_table"].tolist(), storage.blocks[:20].tolist())
            self.assertEqual(slot_view.tolist(), storage.slots[:20].tolist())
            self.assertEqual(starts_view.tolist(), list(range(21)))
        self.assertEqual(len(backend.calls), 1)  # Capture only; no task updates.

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
