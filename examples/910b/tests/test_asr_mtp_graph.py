"""Online auxiliary graph contracts; CPU parity is not NPU acceptance."""

import importlib.util
import unittest
from copy import deepcopy
from types import SimpleNamespace


class TinyMTPFixture:
    def setUp(self):
        import torch
        from asr_mtp_heads import BatchedMTPHeads, CacheArena

        self.t = torch
        torch.manual_seed(1729)
        torch.set_num_threads(1)
        self.cfg = SimpleNamespace(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            attention_bias=False,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            rope_theta=10000,
            rope_scaling=None,
        )
        self.eager = BatchedMTPHeads(self.cfg, 3).eval()
        self.fixed = deepcopy(self.eager)
        self.embed = torch.nn.Embedding(32, 8).eval()
        self.logits = torch.nn.Linear(8, 32).eval()
        self.old = CacheArena(self.cfg, 3, max_context=16, slots=3, device="cpu", dtype=torch.float32)
        try:
            from asr_mtp_graph import FixedMTPProgram
        except ImportError:
            self.fail("fixed MTP graph tensor program is not implemented")
        self.new = CacheArena(
            self.cfg,
            3,
            max_context=16,
            slots=3,
            device="cpu",
            dtype=torch.float32,
            scratch_elements=2 * 8 * 8,
        )
        # Unwritten padding may contain arbitrary allocator data, including NaN.
        self.old.storage.fill_(float("nan"))
        self.new.storage.copy_(self.old.storage)
        self.program = FixedMTPProgram(
            self.fixed,
            self.new,
            self.embed,
            self.logits,
            batch_size=2,
            query_size=8,
            context_size=16,
        )
        self.a = [self.old.acquire(), self.old.acquire()]
        self.b = [self.new.acquire(), self.new.acquire()]
        self.ids = [list(range(3, 10)), list(range(10, 13))]
        for a, b, row in zip(self.a, self.b, self.ids):
            a.length = b.length = len(row)
            hidden = torch.randn(len(row), 8)
            a.target[: len(row)].copy_(hidden)
            b.target[: len(row)].copy_(hidden)

    def compare(self, a, b, rows, limits):
        with self.t.inference_mode():
            expected = self.eager.propose(a, rows, self.embed, self.logits, limits)
            self.program.stage(b, rows, limits)
            actual = self.program.run().clone()
            self.program.commit(b, limits)
        self.t.testing.assert_close(actual, expected)
        for first, second in zip(a, b):
            for old, new in zip(first.heads, second.heads):
                self.assertEqual(old.valid, new.valid)
                n = old.valid
                self.t.testing.assert_close(old.hidden[:n], new.hidden[:n], atol=2e-6, rtol=2e-5)
                self.t.testing.assert_close(old.key[:, :n], new.key[:, :n], atol=2e-6, rtol=2e-5)
                self.t.testing.assert_close(old.value[:, :n], new.value[:, :n], atol=2e-6, rtol=2e-5)
        return actual


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Torch")
class FixedMTPTests(TinyMTPFixture, unittest.TestCase):
    def test_mixed_lengths_and_budgets_match_eager_and_do_not_write_other_slots(self):
        untouched = self.new.storage[2].clone()
        self.compare(self.a, self.b, [row + [4] for row in self.ids], [3, 1])
        self.t.testing.assert_close(self.new.storage[2], untouched, equal_nan=True)
        self.assertEqual(self.fixed.processed_positions, self.eager.processed_positions)

    def test_rejection_and_reordered_rows_reuse_only_confirmed_prefix(self):
        draft = self.compare(self.a, self.b, [row + [4] for row in self.ids], [3, 3])
        for i, (a, b, row) in enumerate(zip(self.a, self.b, self.ids)):
            # Confirm previous drive + first auxiliary token, then a new drive.
            row.extend([4, int(draft[i, 0])])
            old_end = a.length
            a.length = b.length = len(row)
            hidden = self.t.randn(2, 8)
            a.target[old_end : a.length] = b.target[old_end : b.length] = hidden
            a.crop_heads(a.length)
            b.crop_heads(b.length)
        self.compare(self.a[::-1], self.b[::-1], [row + [5] for row in self.ids[::-1]], [2, 3])

    def test_recomputed_prefix_and_released_slot_never_reuse_stale_head_rows(self):
        self.compare(self.a, self.b, [row + [4] for row in self.ids], [3, 3])
        a, b = self.a[0], self.b[0]
        a.crop_heads(3)
        b.crop_heads(3)
        hidden = self.t.randn(4, 8)
        a.target[3:7] = b.target[3:7] = hidden
        self.old.release(self.a[1])
        self.new.release(self.b[1])
        self.a[1], self.b[1] = self.old.acquire(), self.new.acquire()
        a, b = self.a[1], self.b[1]
        a.length = b.length = 5
        hidden = self.t.randn(5, 8)
        a.target[:5] = b.target[:5] = hidden
        self.compare(self.a, self.b, [self.ids[0] + [6], [2, 3, 4, 5, 6, 7]], [3, 3])

    def test_invalid_input_fails_before_mutating_live_cache(self):
        before = self.new.storage.clone()
        with self.assertRaises(ValueError):
            self.program.stage(self.b, [self.ids[0], self.ids[1] + [4]], [3, 3])
        self.t.testing.assert_close(self.new.storage, before, equal_nan=True)

    def test_padded_queries_at_context_end_never_overwrite_the_last_live_position(self):
        for a, b in zip(self.a, self.b):
            a.length = b.length = 16
            hidden = self.t.randn(16, 8)
            a.target[:] = b.target[:] = hidden
        # A graph with q=8 must reject a fresh q=16 before any cache writes.
        with self.assertRaises(ValueError):
            self.program.stage(self.b, [[3] * 17] * 2, [3, 3])
        with self.t.inference_mode():
            self.eager.propose(self.a, [[3] * 17] * 2, self.embed, self.logits, [3, 3])
        for a, b in zip(self.a, self.b):
            for old, new in zip(a.heads, b.heads):
                new.hidden.copy_(old.hidden)
                new.key.copy_(old.key)
                new.value.copy_(old.value)
                new.valid = old.valid
            a.crop_heads(16)
            b.crop_heads(16)
        self.compare(self.a, self.b, [[3] * 17] * 2, [3, 3])


class CPUCapture:
    """Hardware seam only; replay still executes the real fixed tensor program."""

    def capture(self, program):
        program.run()
        return program.run


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Torch")
class GraphDispatchTests(TinyMTPFixture, unittest.TestCase):
    def test_unaccounted_310p_worker_and_explicit_eager_are_rejected(self):
        from asr_mtp_graph import validate_graph_runner

        runner = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False))
        validate_graph_runner(runner)
        runner.model_config.enforce_eager = True
        with self.assertRaisesRegex(ValueError, "enforce_eager"):
            validate_graph_runner(runner)
        runner_type = type(
            "NPUModelRunner310",
            (),
            {"__module__": "vllm_ascend._310p.model_runner_310p"},
        )
        runner = runner_type()
        runner.model_config = SimpleNamespace(enforce_eager=False)
        with self.assertRaisesRegex(ValueError, "310P"):
            validate_graph_runner(runner)

    def make_graphs(self, *, context=16):
        from asr_mtp_graph import MTPGraphs, graph_options

        options = graph_options(
            {
                "graph_mode": "full",
                "graph_query_sizes": [8],
                "graph_max_batch_size": 2,
                "graph_context_tokens": context,
            },
            context=16,
            slots=3,
            max_batch=3,
        )
        graphs = MTPGraphs(options, self.fixed, self.new, self.embed, self.logits)
        graphs.prepare(CPUCapture())
        return graphs

    def test_context_miss_reports_batch_envelope_without_mutating_cache(self):
        for state in self.b:
            self.new.release(state)
        graphs = self.make_graphs(context=8)
        states = [self.new.acquire(), self.new.acquire()]
        # Largest context and largest active fresh query come from different
        # rows. Inactive heads on the second row must not inflate the query.
        for state, length, starts in zip(states, [12, 15], [[8, 7, 6], [12, 0, 0]]):
            state.length = length
            for head, start in zip(state.heads, starts):
                head.valid = start
        before = self.new.storage.clone()
        for count in (1, 2):
            with self.assertLogs("asr_mtp_graph", level="INFO") as caught:
                self.assertIsNone(graphs.propose(states, [[3] * 13, [4] * 16], [3, 1]))
            fields = caught.records[0].getMessage().split()
            self.assertEqual(fields[0], "ASR_MTP_CONTEXT_MISS")
            self.assertEqual(
                dict(part.split("=", 1) for part in fields[1:]),
                {
                    "batch_size": "2",
                    "max_context": "15",
                    "context_limit": "8",
                    "max_fresh_query": "6",
                    "query_limit": "8",
                    "miss_count": str(count),
                },
            )
        self.t.testing.assert_close(self.new.storage, before, equal_nan=True)
        self.assertEqual([h.valid for h in states[0].heads], [8, 7, 6])
        self.assertEqual([h.valid for h in states[1].heads], [12, 0, 0])
        self.assertEqual(graphs.stats["miss_context"], 2)
        self.assertEqual(graphs.stats["replays"], 0)
        self.assertEqual(graphs.stats["miss_query"], 0)

    def test_prepared_replay_matches_eager_with_reordered_arena_slots(self):
        # Capture is startup-only and refuses to overwrite leased live slots.
        for state in self.b:
            self.new.release(state)
        saved = [(s.length, s.target.clone()) for s in self.b]
        graphs = self.make_graphs()
        self.b = [self.new.acquire(), self.new.acquire()]
        for state, (length, hidden) in zip(self.b, saved):
            state.length = length
            state.target.copy_(hidden)
        expected = self.eager.propose(
            self.a[::-1],
            [r + [4] for r in self.ids[::-1]],
            self.embed,
            self.logits,
            [3, 1],
        )
        actual = graphs.propose(self.b[::-1], [r + [4] for r in self.ids[::-1]], [3, 1])
        self.t.testing.assert_close(actual, expected)
        self.assertEqual(graphs.stats["replays"], 1)

    def test_uncovered_shape_returns_none_without_changing_cache_or_capturing_online(
        self,
    ):
        for state in self.b:
            self.new.release(state)
        graphs = self.make_graphs()
        self.b = [self.new.acquire(), self.new.acquire()]
        for state in self.b:
            state.length = 9  # q=9 exceeds the captured q=8 bucket.
        before = self.new.storage.clone()
        self.assertIsNone(graphs.propose(self.b, [[3] * 10] * 2, [3, 3]))
        self.t.testing.assert_close(self.new.storage, before, equal_nan=True)
        self.assertEqual(graphs.stats["miss_query"], 1)

    def test_reduced_maximum_head_budget_uses_eager_to_avoid_wasted_lm_heads(self):
        for state in self.b:
            self.new.release(state)
        graphs = self.make_graphs()
        self.b = [self.new.acquire(), self.new.acquire()]
        for state, tokens in zip(self.b, self.ids):
            state.length = len(tokens)
        self.assertIsNone(graphs.propose(self.b, [r + [4] for r in self.ids], [1, 2]))
        self.assertEqual(graphs.stats["miss_budget"], 1)

    def test_capture_cannot_run_while_requests_own_cache_slots(self):
        with self.assertRaisesRegex(RuntimeError, "live"):
            self.make_graphs()

    def test_worker_proposals_use_prepared_graph_and_preserve_target_drive_boundary(
        self,
    ):
        import numpy as np
        from asr_mtp_worker import MTPWorker

        for state in self.b:
            self.new.release(state)
        graphs = self.make_graphs()
        params = SimpleNamespace(
            extra_args={"ascend_mtp": True},
            max_tokens=20,
            temperature=0.0,
            n=1,
            min_tokens=0,
        )
        rows = [r + [4] for r in self.ids]
        batch = SimpleNamespace(
            req_ids=["a", "b"],
            max_model_len=32,
            num_tokens_no_spec=np.array([len(r) for r in rows]),
            num_computed_tokens_cpu=np.array([0, 0]),
            token_ids_cpu=np.zeros((2, 32), dtype=np.int32),
            spec_decode_unsupported_reqs=set(),
        )
        for i, row in enumerate(rows):
            batch.token_ids_cpu[i, : len(row)] = row
        hidden = self.t.cat([s.target[: s.length] for s in self.a])
        runner = SimpleNamespace(
            input_batch=batch,
            query_start_loc=SimpleNamespace(np=np.array([0, 7, 10])),
            requests={key: SimpleNamespace(sampling_params=params, output_token_ids=[4]) for key in batch.req_ids},
            speculative_config=SimpleNamespace(num_speculative_tokens=3),
        )
        worker = MTPWorker(runner, self.fixed, self.new, self.embed, self.logits, {31}, graphs=graphs)
        actual = worker.propose([[4], [4]], SimpleNamespace(num_scheduled_tokens={"a": 7, "b": 3}), hidden)
        expected = self.eager.propose(self.a, rows, self.embed, self.logits, [3, 3]).tolist()
        self.assertEqual(actual, [r[: r.index(31) + 1] if 31 in r else r for r in expected])
        self.assertEqual(graphs.stats["replays"], 1)
        self.assertEqual(worker.states["a"].length, 7)
        self.assertEqual(worker.states["b"].length, 3)
        self.assertEqual(worker.stats["target_forwards"], 0)

    def test_options_are_bounded_and_default_off_does_not_allocate_graph_storage(self):
        from asr_mtp_graph import graph_options

        self.assertIsNone(graph_options({}, context=512, slots=23, max_batch=20))
        for settings in (
            {"graph_mode": "bad"},
            {"graph_mode": True},
            {"graph_mode": "full", "graph_query_sizes": [0]},
            {"graph_mode": "full", "graph_context_tokens": 1024},
            {"graph_mode": "full", "graph_max_batch_size": True},
            {
                "graph_mode": "full",
                "graph_max_batch_size": 100,
                "graph_query_sizes": [1, 2, 4, 8, 16],
            },
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                graph_options(settings, context=512, slots=100, max_batch=100)


def npu_available():
    if importlib.util.find_spec("torch_npu") is None:
        return False
    import torch
    import torch_npu  # noqa: F401

    return torch.npu.is_available()


@unittest.skipUnless(npu_available(), "requires torch-npu and an available NPU")
class ActualNPUReplayTests(unittest.TestCase):
    def test_changed_inputs_rejection_and_graph_bucket_switch_match_eager(self):
        import torch
        from asr_mtp_graph import MTPGraphs, graph_options
        from asr_mtp_heads import BatchedMTPHeads, CacheArena

        torch.manual_seed(137)
        cfg = SimpleNamespace(
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            attention_bias=False,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            rope_theta=10000,
            rope_scaling=None,
        )
        eager = BatchedMTPHeads(cfg, 2).to("npu", torch.float16).eval()
        fixed = deepcopy(eager)
        embed = torch.nn.Embedding(64, 128).to("npu", torch.float16).eval()
        logits = torch.nn.Linear(128, 64).to("npu", torch.float16).eval()
        options = graph_options(
            {
                "graph_mode": "full",
                "graph_query_sizes": [8, 16],
                "graph_max_batch_size": 2,
            },
            context=32,
            slots=2,
            max_batch=2,
        )
        old = CacheArena(cfg, 2, max_context=32, slots=2, device="npu", dtype=torch.float16)
        new = CacheArena(
            cfg,
            2,
            max_context=32,
            slots=2,
            device="npu",
            dtype=torch.float16,
            scratch_elements=options.scratch_elements(cfg),
        )
        graphs = MTPGraphs(options, fixed, new, embed, logits)
        graphs.prepare()  # Actual NPUGraph capture, no CPU capture adapter.
        a, b = [old.acquire(), old.acquire()], [new.acquire(), new.acquire()]
        ids = [list(range(3, 14)), list(range(20, 27))]
        with torch.inference_mode():
            for first, second, row in zip(a, b, ids):
                first.length = second.length = len(row)
                hidden = torch.randn(len(row), 128, device="npu", dtype=torch.float16)
                first.target[: len(row)] = second.target[: len(row)] = hidden
            for step in range(3):
                order = [0, 1] if step % 2 == 0 else [1, 0]
                left, right = [a[i] for i in order], [b[i] for i in order]
                rows = [ids[i] + [4 + step] for i in order]
                expected = eager.propose(left, rows, embed, logits, [2, 2])
                actual = graphs.propose(right, rows, [2, 2]).clone()
                torch.testing.assert_close(actual, expected)
                for first, second in zip(a, b):
                    for h1, h2 in zip(first.heads, second.heads):
                        self.assertEqual(h1.valid, h2.valid)
                        torch.testing.assert_close(
                            h1.hidden[: h1.valid],
                            h2.hidden[: h2.valid],
                            atol=2e-2,
                            rtol=2e-2,
                        )
                for i, (first, second) in enumerate(zip(a, b)):
                    # Accept the drive; reject all auxiliary tokens.
                    ids[i].append(4 + step)
                    hidden = torch.randn(1, 128, device="npu", dtype=torch.float16)
                    first.target[first.length : first.length + 1] = hidden
                    second.target[second.length : second.length + 1] = hidden
                    first.length += 1
                    second.length += 1
                    first.crop_heads(first.length)
                    second.crop_heads(second.length)
        self.assertEqual(graphs.stats["replays"], 3)


if __name__ == "__main__":
    unittest.main()
