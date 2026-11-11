"""Phase depth correctness; CPU replay is not NPU graph acceptance."""

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from asr_mtp_heads import BatchedMTPHeads, CacheArena
from asr_mtp_token_graph import TokenMTPGraphs
from asr_mtp_worker import MTPWorker, stop_ids


class CPUCapture:
    def capture(self, program):
        program.run()
        return program.run


class PhaseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(612)
        self.cfg = SimpleNamespace(
            hidden_size=12,
            intermediate_size=24,
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=4,
            attention_bias=False,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            rope_theta=10000,
            rope_scaling=None,
        )
        self.heads = BatchedMTPHeads(self.cfg, 5).eval()
        self.embed = torch.nn.Embedding(64, 12).eval()
        self.logits = torch.nn.Linear(12, 64).eval()

    def arena(self):
        return CacheArena(
            self.cfg,
            5,
            max_context=17,
            slots=4,
            device="cpu",
            dtype=torch.float32,
            scratch_elements=33 * 12,
        )

    def worker(self, *, graph=True, acceptance=False):
        arena = self.arena()
        options = SimpleNamespace(max_batch=3, queries=(), tokens=(8, 32), context=17, head_counts=(3, 5))
        graphs = TokenMTPGraphs(options, self.heads, arena, self.embed, self.logits) if graph else None
        if graphs:
            graphs.prepare(CPUCapture())
        ids = [[3, 4, 5], [6, 7], [8, 9, 10, 11]]
        rows = [row + [4] for row in ids]
        batch = SimpleNamespace(
            req_ids=["a", "b", "c"],
            max_model_len=32,
            num_tokens_no_spec=np.array([len(r) for r in rows]),
            num_computed_tokens_cpu=np.zeros(3, dtype=np.int32),
            token_ids_cpu=np.zeros((3, 32), dtype=np.int32),
            spec_decode_unsupported_reqs=set(),
        )
        requests = {}
        for i, (key, row) in enumerate(zip(batch.req_ids, rows)):
            batch.token_ids_cpu[i, : len(row)] = row
            params = SimpleNamespace(
                extra_args={"ascend_mtp": True},
                max_tokens=20,
                temperature=0.0,
                n=1,
                min_tokens=0,
            )
            requests[key] = SimpleNamespace(sampling_params=params, output_token_ids=[4])
        runner = SimpleNamespace(
            input_batch=batch,
            query_start_loc=SimpleNamespace(np=np.cumsum([0] + [len(r) for r in ids])),
            speculative_config=SimpleNamespace(num_speculative_tokens=5),
            requests=requests,
        )
        worker = MTPWorker(
            runner,
            self.heads,
            arena,
            self.embed,
            self.logits,
            set(),
            graphs=graphs,
            acceptance_stats=acceptance,
        )
        scheduler = SimpleNamespace(num_scheduled_tokens=dict(zip(batch.req_ids, map(len, ids))))
        return worker, graphs, scheduler, torch.randn(sum(map(len, ids)), 12)

    def advance(self, worker, scheduler):
        batch = worker.runner.input_batch
        old = batch.num_tokens_no_spec.copy()
        batch.num_computed_tokens_cpu[:] = old - 1
        batch.num_tokens_no_spec[:] += 1
        for i, key in enumerate(batch.req_ids):
            batch.token_ids_cpu[i, old[i]] = 7
            worker.runner.requests[key].output_token_ids.append(7)
        worker.runner.query_start_loc.np = np.array([0, 2, 4, 6])
        scheduler.num_scheduled_tokens = dict.fromkeys(batch.req_ids, 2)
        return torch.randn(6, 12)

    def test_defaults_offer_five_cold_then_three_warm_without_extra_d2h(self):
        worker, graphs, scheduler, hidden = self.worker()
        self.assertEqual(list(map(len, worker.propose([[4]] * 3, scheduler, hidden))), [5] * 3)
        hidden = self.advance(worker, scheduler)
        calls, converted = [], []
        hooks = [
            b.fusion.register_forward_pre_hook(lambda _, args, d=d: calls.append(d))
            for d, b in enumerate(self.heads.blocks)
        ]
        original = torch.Tensor.tolist
        try:
            with patch.object(
                torch.Tensor,
                "tolist",
                lambda t: (converted.append(t.shape), original(t))[1],
            ):
                result = worker.propose([[7]] * 3, scheduler, hidden)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(list(map(len, result)), [3] * 3)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(len(converted), 1)
        self.assertEqual(worker.stats["eager_head_batches"], 0)
        self.assertEqual(graphs.stats["miss_budget"], 0)
        for state in worker.states.values():
            self.assertEqual([h.valid for h in state.heads[:3]], [state.length] * 3)
            self.assertLess(state.heads[3].valid, state.length)
            self.assertLess(state.heads[4].valid, state.length)

    def test_eager_uses_same_phase_limits_without_metrics(self):
        worker, _, scheduler, hidden = self.worker(graph=False)
        self.assertEqual(list(map(len, worker.propose([[4]] * 3, scheduler, hidden))), [5] * 3)
        hidden = self.advance(worker, scheduler)
        self.assertEqual(list(map(len, worker.propose([[7]] * 3, scheduler, hidden))), [3] * 3)

    def test_mixed_phase_groups_do_not_execute_warm_tail_and_keep_order(self):
        worker, graphs, scheduler, hidden = self.worker()
        # b offers its legacy draft first, while retaining target hidden history.
        worker.runner.requests["b"].sampling_params.extra_args["ascend_external_draft"] = [9]
        worker.propose([[4]] * 3, scheduler, hidden)
        hidden = self.advance(worker, scheduler)
        converted = []
        original = torch.Tensor.tolist
        with patch.object(
            torch.Tensor,
            "tolist",
            lambda t: (converted.append(t.shape), original(t))[1],
        ):
            result = worker.propose([[7]] * 3, scheduler, hidden)
        self.assertEqual(list(map(len, result)), [3, 5, 3])
        self.assertEqual(len(converted), 1)
        self.assertEqual(worker.stats["eager_head_batches"], 0)
        self.assertEqual(graphs.stats["miss_budget"], 0)

    def test_legacy_and_prefill_do_not_consume_cold_cleanup_resets_it(self):
        worker, _, scheduler, hidden = self.worker(graph=False)
        worker.propose([[]] * 3, scheduler, hidden)
        for req in worker.runner.requests.values():
            req.sampling_params.extra_args["ascend_external_draft"] = [9, 10]
        self.assertEqual(worker.propose([[4]] * 3, scheduler, hidden), [[9, 10]] * 3)
        self.assertEqual(list(map(len, worker.propose([[4]] * 3, scheduler, hidden))), [5] * 3)
        worker.finish(["a", "b", "c"])
        for req in worker.runner.requests.values():
            req.sampling_params.extra_args.pop("ascend_external_draft")
        self.assertEqual(list(map(len, worker.propose([[4]] * 3, scheduler, hidden))), [5] * 3)

    def test_depth_graph_matches_eager_after_rejection_and_short_budget(self):
        reference = deepcopy(self.heads)
        old, new = self.arena(), self.arena()
        graphs = TokenMTPGraphs(
            SimpleNamespace(max_batch=3, tokens=(8, 32), context=17, head_counts=(3, 5)),
            self.heads,
            new,
            self.embed,
            self.logits,
        )
        graphs.prepare(CPUCapture())
        a, b = old.acquire(), new.acquire()
        ids = [3, 4, 5, 6, 7]
        a.length = b.length = len(ids)
        a.target[:5] = b.target[:5] = torch.randn(5, 12)
        for limits in ([5], [3], [1]):
            if limits != [5]:
                a.crop_heads(4)
                b.crop_heads(4)
            expected = reference.propose([a], [ids + [9]], self.embed, self.logits, limits)
            actual = graphs.propose([b], [ids + [9]], limits)
            self.assertIsNotNone(actual, "phase/short budgets must remain graphed")
            torch.testing.assert_close(actual[:1, : limits[0]], expected[:, : limits[0]])
            for left, right in zip(a.heads, b.heads):
                self.assertEqual(left.valid, right.valid)
                torch.testing.assert_close(
                    left.key[:, : left.valid],
                    right.key[:, : right.valid],
                    atol=3e-6,
                    rtol=3e-5,
                )
        old.release(a)
        new.release(b)

    def test_ignore_eos_keeps_explicit_stop_ids(self):
        p = SimpleNamespace(ignore_eos=True, eos_token_id=63, stop_token_ids=[])
        self.assertEqual(stop_ids(p, {63}), set())
        p.stop_token_ids = [63]
        self.assertEqual(stop_ids(p, {63}), {63})

    def test_phase_depths_are_validated_against_loaded_heads(self):
        worker, _, _, _ = self.worker(graph=False)
        args = (worker.runner, self.heads, worker.arena, self.embed, self.logits, set())
        for name in ("cold_heads", "warm_heads"):
            for value in (True, 0, -1, 6, "3"):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    MTPWorker(*args, **{name: value})

    def test_phase_graph_count_limit_accounts_for_both_depths(self):
        from asr_mtp_graph import graph_options

        settings = {"graph_mode": "full", "graph_token_sizes": list(range(1, 34))}
        with self.assertRaisesRegex(ValueError, "graph count"):
            graph_options(settings, context=64, slots=4, max_batch=4, head_counts=(3, 5))
        settings["graph_token_sizes"] = list(range(1, 32))
        options = graph_options(settings, context=64, slots=4, max_batch=4, head_counts=(5, 3, 5))
        self.assertEqual(options.head_counts, (3, 5))

    def test_rectangular_graph_also_executes_only_three_warm_heads(self):
        from asr_mtp_graph import MTPGraphs

        arena, reference_arena = self.arena(), self.arena()
        reference = deepcopy(self.heads)
        graphs = MTPGraphs(
            SimpleNamespace(max_batch=1, queries=(8,), context=17, head_counts=(3, 5)),
            self.heads,
            arena,
            self.embed,
            self.logits,
        )
        graphs.prepare(CPUCapture())
        a, b = reference_arena.acquire(), arena.acquire()
        a.length = b.length = 5
        a.target[:5] = b.target[:5] = torch.randn(5, 12)
        calls = []
        hooks = [
            block.fusion.register_forward_pre_hook(lambda _, args, d=d: calls.append(d))
            for d, block in enumerate(self.heads.blocks)
        ]
        try:
            result = graphs.propose([b], [[3, 4, 5, 6, 7, 8]], [3])
        finally:
            for hook in hooks:
                hook.remove()
        expected = reference.propose([a], [[3, 4, 5, 6, 7, 8]], self.embed, self.logits, [3])
        torch.testing.assert_close(result[:, :3], expected[:, :3])
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual([h.valid for h in b.heads], [5, 5, 5, 0, 0])
        arena.release(b)
        reference_arena.release(a)

    def test_paged_depth_capture_stages_only_executed_fia_heads(self):
        from asr_mtp_paged_cache import PagedCacheArena

        arena = PagedCacheArena(
            self.cfg,
            5,
            max_context=17,
            slots=4,
            block_size=4,
            dummy_tokens=33,
            device="cpu",
            dtype=torch.float32,
            scratch_elements=33 * 12,
        )
        graphs = TokenMTPGraphs(
            SimpleNamespace(max_batch=3, tokens=(8, 32), context=17, head_counts=(3, 5)),
            self.heads,
            arena,
            self.embed,
            self.logits,
        )
        graphs.prepare(CPUCapture())
        program = graphs.entries[3, 8][0]
        self.assertEqual(set(program.attention.lengths), {0, 1, 2})
        self.assertEqual(program.attention.heads, 3)

    def test_explicit_five_five_keeps_warm_depth_five(self):
        worker, _, scheduler, hidden = self.worker(graph=False)
        worker.warm_heads = 5
        worker.propose([[4]] * 3, scheduler, hidden)
        hidden = self.advance(worker, scheduler)
        self.assertEqual(list(map(len, worker.propose([[7]] * 3, scheduler, hidden))), [5] * 3)


if __name__ == "__main__":
    unittest.main()
