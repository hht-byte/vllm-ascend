"""Packed MTP tensor parity and token-only graph dispatch contracts."""

import importlib.util
import unittest
from copy import deepcopy
from types import SimpleNamespace


class PackedFixture:
    def setUp(self):
        import torch
        from asr_mtp_heads import BatchedMTPHeads, CacheArena

        self.t = torch
        torch.set_num_threads(1)
        torch.manual_seed(618)
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
        self.eager = BatchedMTPHeads(self.cfg, 3).eval()
        self.fixed = deepcopy(self.eager)
        self.embed = torch.nn.Embedding(64, 12).eval()
        self.logits = torch.nn.Linear(12, 64).eval()
        self.old = CacheArena(self.cfg, 3, max_context=17, slots=4, device="cpu", dtype=torch.float32)
        self.new = CacheArena(
            self.cfg,
            3,
            max_context=17,
            slots=4,
            device="cpu",
            dtype=torch.float32,
            scratch_elements=33 * 12 + 1,
        )
        self.old.storage.fill_(float("nan"))
        self.new.graph_storage.fill_(float("nan"))
        try:
            from asr_mtp_token_graph import PackedMTPProgram
        except ImportError as exc:
            self.fail(f"packed token MTP graph implementation missing: {exc}")
        self.program = PackedMTPProgram(
            self.fixed,
            self.new,
            self.embed,
            self.logits,
            max_batch=3,
            token_size=32,
            context_size=17,
        )
        self.a = [self.old.acquire() for _ in range(3)]
        self.b = [self.new.acquire() for _ in range(3)]
        self.ids = [[3, 4, 5, 6, 7, 8, 9], [10, 11, 12], [20, 21, 22, 23, 24]]
        for a, b, row in zip(self.a, self.b, self.ids):
            a.length = b.length = len(row)
            hidden = torch.randn(len(row), 12)
            a.target[: len(row)] = b.target[: len(row)] = hidden

    def compare(self, a, b, rows, limits):
        expected = self.eager.propose(a, rows, self.embed, self.logits, limits)
        self.program.stage(b, rows, limits)
        actual = self.program.run()[: len(b)].clone()
        self.program.commit(b, limits)
        self.t.testing.assert_close(actual, expected)
        for left, right in zip(a, b):
            for depth, (x, y) in enumerate(zip(left.heads, right.heads)):
                self.assertEqual(x.valid, y.valid)
                n = x.valid
                if depth < 2:
                    self.t.testing.assert_close(x.hidden[:n], y.hidden[:n], atol=3e-6, rtol=3e-5)
                self.t.testing.assert_close(x.key[:, :n], y.key[:, :n], atol=3e-6, rtol=3e-5)
                self.t.testing.assert_close(x.value[:, :n], y.value[:, :n], atol=3e-6, rtol=3e-5)
        return actual

    def append(self, count=1):
        for left, right, row in zip(self.a, self.b, self.ids):
            old = len(row)
            row.extend([31] * count)
            hidden = self.t.randn(count, 12)
            left.target[old : old + count] = right.target[old : old + count] = hidden
            left.length = right.length = len(row)
            left.crop_heads(len(row))
            right.crop_heads(len(row))


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Torch")
class PackedProgramTests(PackedFixture, unittest.TestCase):
    def test_exact_logical_bucket_has_positive_dummy_and_full_kv_segment(self):
        from asr_mtp_token_graph import PackedMTPProgram

        program = PackedMTPProgram(
            self.fixed,
            self.new,
            self.embed,
            self.logits,
            max_batch=3,
            token_size=8,
            context_size=17,
        )
        row = list(range(3, 11))
        for state in (self.a[0], self.b[0]):
            state.length = 8
        hidden = self.t.randn(8, 12)
        self.a[0].target[:8] = self.b[0].target[:8] = hidden
        program.stage(self.b[:1], [row + [4]], [3])
        self.assertEqual(program.tokens, 8)
        self.assertEqual(getattr(program, "query_capacity", None), 9)
        self.assertEqual(program.kv_capacity, 60)
        for qends, kvends in program.attention.lengths.values():
            self.assertEqual(qends, (8, 9))
            self.assertEqual(kvends, (8, 60))
        expected = self.eager.propose(self.a[:1], [row + [4]], self.embed, self.logits, [3])
        self.t.testing.assert_close(program.run()[:1], expected)
        program.commit(self.b[:1], [3])
        for left, right in zip(self.a[0].heads, self.b[0].heads):
            self.t.testing.assert_close(left.key[:, :8], right.key[:, :8])
            self.t.testing.assert_close(left.value[:, :8], right.value[:, :8])

    def test_partial_bucket_dummy_covers_every_physical_kv_position(self):
        self.program.stage(self.b, [row + [4] for row in self.ids], [3, 3, 3])
        for qends, kvends in self.program.attention.lengths.values():
            self.assertEqual(kvends[-1], self.program.kv_capacity)
            self.assertEqual(qends, (7, 10, 15, 33))
            self.assertEqual(kvends, (7, 10, 15, 84))
        self.compare(self.a, self.b, [row + [4] for row in self.ids], [3, 3, 3])

    def test_total_capacity_is_maximum_per_head_sum_not_sum_of_request_maxima(self):
        from asr_mtp_token_graph import required_tokens

        states = [
            SimpleNamespace(length=10, heads=[SimpleNamespace(valid=v) for v in (9, 8, 7)]),
            SimpleNamespace(length=11, heads=[SimpleNamespace(valid=v) for v in (8, 10, 9)]),
        ]
        # Active query totals by head are 4, 3, 3. The per-request maxima
        # would give 6, and summing all heads would incorrectly give 10.
        self.assertEqual(required_tokens(states, [3, 2], 3), 4)

    def test_logits_are_computed_only_for_request_slots_not_every_token(self):
        rows = []
        hook = self.logits.register_forward_pre_hook(lambda _module, args: rows.append(args[0].shape[0]))
        try:
            self.program.stage(self.b, [r + [4] for r in self.ids], [3, 3, 3])
            self.program.run()
        finally:
            hook.remove()
        self.assertEqual(rows, [3, 3, 3])

    def test_mixed_query_lengths_share_one_packed_program_without_cross_attention(self):
        untouched = self.new.storage[3].clone()
        self.compare(self.a, self.b, [row + [4] for row in self.ids], [3, 1, 2])
        self.t.testing.assert_close(self.new.storage[3], untouched, equal_nan=True)
        self.assertTrue(self.t.isnan(self.new.graph_storage[-1]))
        for state in self.b:
            self.assertTrue(self.t.isnan(state.heads[-1].hidden).all())

    def test_rejection_and_changed_budget_reuse_only_confirmed_history(self):
        self.compare(self.a, self.b, [r + [4] for r in self.ids], [3, 1, 2])
        self.append()
        self.compare(self.a[::-1], self.b[::-1], [r + [5] for r in self.ids[::-1]], [3, 3, 3])
        self.append(2)
        # Short budgets can use the ordinary eager fallback and later re-enter
        # the same packed graph without losing non-final hidden states.
        rows = [r + [6] for r in self.ids]
        expected = self.eager.propose(self.a, rows, self.embed, self.logits, [1, 2, 1])
        actual = self.fixed.propose(self.b, rows, self.embed, self.logits, [1, 2, 1])
        self.t.testing.assert_close(actual, expected)
        self.append()
        self.compare(self.a, self.b, [r + [7] for r in self.ids], [3, 3, 3])

    def test_actual_request_count_can_change_without_changing_token_graph(self):
        self.compare(self.a[:1], self.b[:1], [self.ids[0] + [4]], [3])
        self.compare(self.a[1:], self.b[1:], [r + [5] for r in self.ids[1:]], [3, 3])

    def test_duplicate_slots_and_overflow_fail_before_cache_changes(self):
        from asr_mtp_token_graph import PackedMTPProgram

        before = self.new.graph_storage.clone()
        for states, rows, limits in (
            ([self.b[0], self.b[0]], [self.ids[0] + [4]] * 2, [3, 3]),
            (self.b, [r for r in self.ids], [3, 3, 3]),
        ):
            with self.assertRaises(ValueError):
                self.program.stage(states, rows, limits)
        self.t.testing.assert_close(self.new.graph_storage, before, equal_nan=True)
        too_small = PackedMTPProgram(
            self.fixed,
            self.new,
            self.embed,
            self.logits,
            max_batch=3,
            token_size=8,
            context_size=17,
        )
        with self.assertRaises(ValueError):
            too_small.stage(self.b, [r + [4] for r in self.ids], [3, 3, 3])
        self.t.testing.assert_close(self.new.graph_storage, before, equal_nan=True)

    def test_released_slot_is_reinitialized_before_packed_reuse(self):
        self.compare(self.a, self.b, [r + [4] for r in self.ids], [3, 3, 3])
        self.old.release(self.a[1])
        self.new.release(self.b[1])
        self.a[1], self.b[1] = self.old.acquire(), self.new.acquire()
        self.ids[1] = [40, 41]
        for state in (self.a[1], self.b[1]):
            state.length = 2
        hidden = self.t.randn(2, 12)
        self.a[1].target[:2] = self.b[1].target[:2] = hidden
        self.append()
        self.compare(self.a, self.b, [r + [6] for r in self.ids], [3, 3, 3])


class CPUCapture:
    def __init__(self):
        self.count = 0
        self.replays = []

    def capture(self, program):
        self.count += 1
        program.run()

        def replay():
            self.replays.append(program.tokens)
            return program.run()

        return replay


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Torch")
class TokenDispatchTests(PackedFixture, unittest.TestCase):
    def manager(self):
        from asr_mtp_token_graph import TokenMTPGraphs

        options = SimpleNamespace(max_batch=3, context=17, tokens=(8, 16, 32))
        graphs = TokenMTPGraphs(options, self.fixed, self.new, self.embed, self.logits)
        # prepare is a startup operation, before live slot leases.
        for state in self.b:
            self.new.release(state)
        capture = CPUCapture()
        graphs.prepare(capture)
        self.b = [self.new.acquire() for _ in range(3)]
        for left, right, row in zip(self.a, self.b, self.ids):
            right.length = len(row)
            right.target[: len(row)].copy_(left.target[: len(row)])
        return graphs, capture

    def test_keys_are_total_token_buckets_and_mixed_requests_use_one_replay(self):
        graphs, capture = self.manager()
        self.assertEqual(set(graphs.entries), {8, 16, 32})
        actual = graphs.propose(self.b, [r + [4] for r in self.ids], [3, 3, 3])
        expected = self.eager.propose(self.a, [r + [4] for r in self.ids], self.embed, self.logits, [3, 3, 3])
        self.t.testing.assert_close(actual, expected)
        self.assertEqual(graphs.stats["replays"], 1)
        self.assertEqual(graphs.stats["requests"], 3)
        self.assertEqual(capture.count, 3)
        self.assertEqual(capture.replays, [16])

    def test_full_logical_bucket_does_not_move_to_the_next_graph(self):
        graphs, capture = self.manager()
        row = list(range(3, 11))
        hidden = self.t.randn(8, 12)
        for state in (self.a[0], self.b[0]):
            state.length = 8
            state.target[:8] = hidden
        actual = graphs.propose(self.b[:1], [row + [4]], [3])
        expected = self.eager.propose(self.a[:1], [row + [4]], self.embed, self.logits, [3])
        self.t.testing.assert_close(actual, expected)
        self.assertEqual(capture.replays, [8])
        self.assertEqual(graphs.entries[8][0].tokens, 8)
        self.assertEqual(getattr(graphs.entries[8][0], "query_capacity", None), 9)

    def test_context_and_reduced_budget_fallback_never_capture_online(self):
        graphs, capture = self.manager()
        before = self.new.graph_storage.clone()
        self.assertIsNone(graphs.propose(self.b, [r + [4] for r in self.ids], [1, 2, 1]))
        self.b[0].length = 18
        with self.assertLogs("asr_mtp_token_graph", level="INFO") as records:
            self.assertIsNone(graphs.propose(self.b, [r + [4] for r in self.ids], [3, 3, 3]))
        self.assertIn("max_fresh_query=18", records.output[0])
        self.assertIn("token_limit=32", records.output[0])
        self.assertEqual(capture.count, 3)
        self.t.testing.assert_close(self.new.graph_storage, before, equal_nan=True)


def npu_available():
    if importlib.util.find_spec("torch_npu") is None:
        return False
    import torch
    import torch_npu  # noqa: F401

    return torch.npu.is_available()


@unittest.skipUnless(npu_available(), "requires torch-npu and an available NPU")
class ActualPackedNPUReplayTests(unittest.TestCase):
    def test_token_bucket_replay_changes_request_boundaries_and_rejection_prefixes(
        self,
    ):
        import torch
        from asr_mtp_heads import BatchedMTPHeads, CacheArena
        from asr_mtp_token_graph import TokenMTPGraphs

        torch.manual_seed(137)
        cfg = SimpleNamespace(
            hidden_size=256,
            intermediate_size=512,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            attention_bias=False,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            rope_theta=10000,
            rope_scaling=None,
        )
        eager = BatchedMTPHeads(cfg, 2).to("npu", torch.float16).eval()
        fixed = deepcopy(eager)
        embed = torch.nn.Embedding(64, 256).to("npu", torch.float16).eval()
        logits = torch.nn.Linear(256, 64).to("npu", torch.float16).eval()
        old = CacheArena(cfg, 2, max_context=64, slots=2, device="npu", dtype=torch.float16)
        new = CacheArena(
            cfg,
            2,
            max_context=64,
            slots=2,
            device="npu",
            dtype=torch.float16,
            scratch_elements=33 * 256,
        )
        options = SimpleNamespace(max_batch=2, context=64, tokens=(8, 32))
        graphs = TokenMTPGraphs(options, fixed, new, embed, logits)
        graphs.prepare()  # Real NPUGraph and FIA task-update path.
        a, b = [old.acquire(), old.acquire()], [new.acquire(), new.acquire()]
        ids = [list(range(3, 14)), list(range(20, 27))]
        with torch.inference_mode():
            for left, right, row in zip(a, b, ids):
                left.length = right.length = len(row)
                hidden = torch.randn(len(row), 256, device="npu", dtype=torch.float16)
                left.target[: len(row)] = right.target[: len(row)] = hidden
            for step in range(3):
                order = [0, 1] if step % 2 == 0 else [1, 0]
                left, right = [a[i] for i in order], [b[i] for i in order]
                rows = [ids[i] + [4 + step] for i in order]
                expected = eager.propose(left, rows, embed, logits, [2, 2])
                actual = graphs.propose(right, rows, [2, 2]).clone()
                torch.npu.synchronize()
                torch.testing.assert_close(actual, expected)
                for x, y in zip(a, b):
                    for d, (hx, hy) in enumerate(zip(x.heads, y.heads)):
                        self.assertEqual(hx.valid, hy.valid)
                        n = hx.valid
                        if d == 0:
                            torch.testing.assert_close(hx.hidden[:n], hy.hidden[:n], atol=3e-3, rtol=3e-3)
                        torch.testing.assert_close(hx.key[:, :n], hy.key[:, :n], atol=3e-3, rtol=3e-3)
                        torch.testing.assert_close(hx.value[:, :n], hy.value[:, :n], atol=3e-3, rtol=3e-3)
                if step < 2:
                    for i, (x, y) in enumerate(zip(a, b)):
                        old_end = x.length
                        ids[i].extend([4 + step, int(expected[order.index(i), 0])])
                        x.length = y.length = len(ids[i])
                        hidden = torch.randn(2, 256, device="npu", dtype=torch.float16)
                        x.target[old_end : x.length] = y.target[old_end : y.length] = hidden
                        x.crop_heads(x.length)
                        y.crop_heads(y.length)
                    if step == 1:
                        for state in (*a, *b):
                            state.crop_heads(state.length - 6)
        self.assertEqual(graphs.stats["replays"], 3)
        self.assertEqual(set(graphs.entries), {8, 32})


if __name__ == "__main__":
    unittest.main()
