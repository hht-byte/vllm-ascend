"""Fixed auxiliary PA storage, direct reads and rejection/fallback contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_asr_mtp_token_attention import FakeRuntime
from test_asr_mtp_token_graph import PackedFixture


class PagedArenaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = SimpleNamespace(hidden_size=12, num_key_value_heads=2, head_dim=4)

    def arena(self, **kwargs):
        from asr_mtp_paged_cache import PagedCacheArena

        return PagedCacheArena(
            self.cfg,
            3,
            max_context=9,
            slots=2,
            block_size=4,
            dummy_tokens=17,
            device="cpu",
            dtype=torch.float32,
            scratch_elements=100,
            **kwargs,
        )

    def test_views_share_the_only_kv_storage_and_account_rounding_and_scratch(self):
        arena = self.arena()
        state = arena.acquire()
        self.assertEqual(arena.storage.shape, (2, 9 * 4 * 12))
        self.assertEqual(arena.kv.shape, (3, 2, 11, 4, 2, 4))
        state.heads[1].key[:, 5].fill_(7)
        torch.testing.assert_close(arena.kv[1, 0, 1, 1], torch.full((2, 4), 7.0))
        self.assertEqual(
            state.heads[1].key.untyped_storage().data_ptr(),
            arena.kv.untyped_storage().data_ptr(),
        )
        expected = (2 * 9 * 4 * 12 + 3 * 2 * 11 * 4 * 2 * 4) * 4
        self.assertEqual(arena.cache_bytes, expected)
        self.assertEqual(arena.graph_storage.numel() - arena.scratch_offset, 100)

    def test_joint_write_and_released_slot_never_change_another_live_slot(self):
        arena = self.arena()
        first, second = arena.acquire(), arena.acquire()
        arena.kv.fill_(19)
        key, value = torch.randn(3, 2, 4), torch.randn(3, 2, 4)
        slots = torch.tensor([0, 8, arena.dummy_base + 2], dtype=torch.int32)
        arena.write_kv(0, key, value, slots)
        torch.testing.assert_close(first.heads[0].key[:, 0], key[0])
        torch.testing.assert_close(first.heads[0].value[:, 8], value[1])
        self.assertTrue((second.heads[0].key == 19).all())
        first.heads[0].valid = 8
        arena.release(first)
        reused = arena.acquire()
        self.assertEqual(reused.slot, first.slot)
        self.assertEqual([h.valid for h in reused.heads], [0, 0, 0])
        self.assertTrue((reused.heads[0].key[:, 0] == key[0]).all())

    def test_cache_budget_formula_matches_all_allocated_live_cache_bytes(self):
        arena = self.arena()
        from asr_mtp_paged_cache import PagedCacheArena

        per_slot, dummy = PagedCacheArena.budget_elements(self.cfg, 3, 9, 4, 17)
        self.assertEqual((2 * per_slot + dummy) * 4, arena.cache_bytes)

    def test_nd_scatterpa_api_is_selected_once_and_uses_norm_layout(self):
        calls = []

        def scatter(key, value, key_cache, value_cache, slot_mapping, *, cache_mode):
            calls.append((cache_mode, slot_mapping.dtype))
            key_cache.view(-1, 2, 4).index_copy_(0, slot_mapping.long(), key)
            value_cache.view(-1, 2, 4).index_copy_(0, slot_mapping.long(), value)

        arena = self.arena(npu_api=SimpleNamespace(npu_scatter_pa_kv_cache=scatter))
        key, value = torch.randn(2, 2, 4), torch.randn(2, 2, 4)
        arena.write_kv(0, key, value, torch.tensor([1, 12], dtype=torch.int32))
        self.assertEqual(arena.write_backend, "scatter_pa")
        self.assertEqual(calls, [("Norm", torch.int32)])
        torch.testing.assert_close(arena.kv[0, 0].view(-1, 2, 4)[[1, 12]], key)

    def test_legacy_nz_only_api_uses_one_combined_scatter_without_probing(self):
        calls = []

        def nz_only(key, value, key_cache, value_cache, slot_mapping):
            raise AssertionError("NZ-only op must not be probed")

        def scatter(cache, indices, updates):
            calls.append(len(indices))
            cache.index_copy_(0, indices[:, 0], updates)

        api = SimpleNamespace(npu_scatter_pa_kv_cache=nz_only, npu_scatter_nd_update_=scatter)
        arena = self.arena(npu_api=api)
        arena.write_kv(
            1,
            torch.ones(2, 2, 4),
            torch.full((2, 2, 4), 3.0),
            torch.tensor([0, 12], dtype=torch.int32),
        )
        self.assertEqual(calls, [4])
        self.assertEqual(arena.write_backend, "scatter_nd")
        with self.assertRaisesRegex(ValueError, "ND"):
            self.arena(npu_api=api, write_backend="scatter_pa")


class PagedProgramTests(PackedFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        from asr_mtp_paged_cache import PagedCacheArena
        from asr_mtp_token_graph import PackedMTPProgram

        self.new = PagedCacheArena(
            self.cfg,
            3,
            max_context=17,
            slots=4,
            block_size=4,
            dummy_tokens=33,
            device="cpu",
            dtype=torch.float32,
            scratch_elements=33 * 12,
        )
        self.new.graph_storage.fill_(float("nan"))
        self.program = PackedMTPProgram(
            self.fixed,
            self.new,
            self.embed,
            self.logits,
            max_batch=3,
            token_size=32,
            context_size=17,
        )
        self.b = [self.new.acquire() for _ in range(3)]
        for a, b in zip(self.a, self.b):
            b.length = a.length
            b.target[: a.length] = a.target[: a.length]

    def test_mixed_prefixes_budgets_and_rejection_match_contiguous_reference(self):
        with torch.inference_mode():
            self.compare(self.a, self.b, [r + [4] for r in self.ids], [3, 1, 2])
            self.append(2)
            self.compare(self.a[::-1], self.b[::-1], [r + [5] for r in self.ids[::-1]], [3, 2, 1])
            self.append(1)
            self.compare(self.a, self.b, [r + [6] for r in self.ids], [3, 3, 3])

    def test_eager_fallback_and_paged_replay_share_cache_after_rejection(self):
        with torch.inference_mode():
            self.compare(self.a, self.b, [r + [4] for r in self.ids], [3, 3, 3])
            self.append(1)
            rows = [r + [5] for r in self.ids]
            expected = self.eager.propose(self.a, rows, self.embed, self.logits, [3] * 3)
            actual = self.fixed.propose(self.b, rows, self.embed, self.logits, [3] * 3)
            torch.testing.assert_close(actual, expected)
            self.append(1)
            self.compare(self.a, self.b, [r + [6] for r in self.ids], [3] * 3)

    def test_worker_mixes_paged_replays_and_eager_context_fallback(self):
        from test_asr_mtp_runtime import RuntimeHandoffTests

        worker, graphs, scheduler, hidden, rows = RuntimeHandoffTests.make_worker(
            self,
            max_batch=1,
            context=6,
        )
        actual = worker.propose([[4]] * 3, scheduler, hidden)
        expected = RuntimeHandoffTests.expected(self, rows)
        self.assertEqual(actual, expected)
        self.assertEqual(worker.stats["eager_head_batches"], 1)
        self.assertEqual(graphs.stats["replays"], 2)

    def test_native_fused_paged_program_preserves_rejection_cache(self):
        from asr_mtp_fused import prepare_fused_heads

        prepare_fused_heads(self.fixed.requires_grad_(False))
        self.test_mixed_prefixes_budgets_and_rejection_match_contiguous_reference()

    def test_precomputed_rope_clamps_padding_when_token_bucket_exceeds_context(self):
        from asr_mtp_fused import prepare_fused_heads
        from asr_mtp_paged_cache import PagedCacheArena
        from asr_mtp_qkv_rope import prepare_qkv_rope
        from asr_mtp_token_graph import PackedMTPProgram

        self.new = PagedCacheArena(
            self.cfg,
            3,
            max_context=17,
            slots=4,
            block_size=4,
            dummy_tokens=1025,
            device="cpu",
            dtype=torch.float32,
            scratch_elements=1025 * self.cfg.hidden_size,
        )
        self.program = PackedMTPProgram(
            self.fixed,
            self.new,
            self.embed,
            self.logits,
            max_batch=3,
            token_size=1024,
            context_size=17,
        )
        self.b = [self.new.acquire() for _ in self.a]
        for left, right in zip(self.a, self.b):
            right.length = left.length
            right.target[: left.length] = left.target[: left.length]
        prepare_fused_heads(self.fixed.requires_grad_(False))
        self.fixed.qkv_rope = prepare_qkv_rope(self.fixed, max_context=17, backend="torch")
        self.assertEqual(self.fixed.qkv_rope.position_capacity, 21)
        with torch.inference_mode():
            self.compare(self.a, self.b, [r + [4] for r in self.ids], [3, 2, 1])

    def test_device_program_only_reads_hidden_and_jointly_writes_kv_once_per_head(self):
        reads, writes = [], []
        original_read = self.program._read
        original_write = self.new.write_kv

        def read(indices):
            reads.append(indices.shape)
            return original_read(indices)

        def write(*args):
            writes.append(args[0])
            return original_write(*args)

        # CPU attention materializes reference KV; the device program must not.

        class ReferenceAttention:
            def __call__(self, head, q, k, v, scale):
                return torch.zeros_like(q)

            def stage(self, *args):
                pass

            def set_tables(self, *args):
                pass

        self.program.attention = ReferenceAttention()
        self.program.stage(self.b, [r + [4] for r in self.ids], [3] * 3)
        with (
            torch.inference_mode(),
            patch.object(self.program, "_read", read),
            patch.object(self.new, "write_kv", write),
        ):
            self.program.run()
        self.assertEqual(len(reads), 3)
        self.assertTrue(all(len(shape) == 2 for shape in reads))
        self.assertEqual(writes, [0, 1, 2])

    def test_invalid_shape_fails_before_upload_or_cache_write(self):
        before = self.new.kv.clone()
        with self.assertRaises(ValueError):
            self.program.stage([self.b[0], self.b[0]], [self.ids[0] + [4]] * 2, [3, 3])
        torch.testing.assert_close(before, self.new.kv)

    def test_context_edge_padding_preserves_other_pages_and_rounded_tail(self):
        self.new.kv[:, :, : self.new.dummy_base // self.new.block_size].fill_(19)
        row = list(range(3, 20))
        self.a[0].length = self.b[0].length = len(row)
        hidden = torch.randn(17, self.cfg.hidden_size)
        self.a[0].target[:17] = self.b[0].target[:17] = hidden
        with torch.inference_mode():
            self.compare(self.a[:1], self.b[:1], [row + [4]], [3])
        for kind in range(2):
            cells = self.new.kv[:, kind].reshape(3, self.new.token_capacity, -1)
            self.assertTrue((cells[:, 17 : self.new.dummy_base] == 19).all())


class PagedAttentionTests(unittest.TestCase):
    def test_pa_uses_per_sequence_lengths_and_current_block_table_without_kv_gather(
        self,
    ):
        from asr_mtp_paged_attention import PagedAttention

        runtime = FakeRuntime()
        calls = []

        class API:
            npu_fused_infer_attention_score = SimpleNamespace(out=lambda **kw: calls.append(kw))

            def _npu_fused_infer_attention_score_get_max_workspace(self, **kw):
                return torch.empty(1)

        attention = PagedAttention(
            2,
            1,
            128,
            device="cpu",
            dtype=torch.float32,
            heads=1,
            max_batch=3,
            block_size=4,
            num_blocks=8,
            max_blocks=3,
            npu_api=API(),
            npu_runtime=runtime,
        )
        attention.set_tables([[[1, 2, 0], [4, 5, 0], [6, 7, 0]]])
        attention.stage(0, [2, 3, 5], [4, 6, 8])
        q, k = torch.zeros(5, 2, 128), torch.zeros(8, 4, 128)
        attention(0, q, k, k, 0.25)
        attention.begin_capture()
        attention(0, q, k, k, 0.25)
        attention.end_capture()
        self.assertEqual(calls[-1]["actual_seq_lengths_kv"], (4, 2, 2))
        self.assertEqual(calls[-1]["block_table"].shape, (3, 3))
        self.assertIs(calls[-1]["key"], k)
        attention.set_tables([[[5, 4, 0], [6, 7, 0]]])
        attention.stage(0, [3, 5], [6, 8])
        attention.update()
        self.assertEqual(calls[-1]["actual_seq_lengths_kv"], (6, 2))
        self.assertEqual(calls[-1]["block_table"].shape, (2, 3))
        self.assertEqual(calls[-1]["block_table"][0, :2].tolist(), [5, 4])
        self.assertEqual(runtime.events.count("event_record"), 1)
        with patch("torch.tensor", side_effect=AssertionError("redundant table upload")):
            attention.set_tables([[[5, 4, 0], [6, 7, 0]]])
        before = len(calls)
        attention.update()
        self.assertEqual(len(calls), before)
        self.assertEqual(runtime.events.count("event_record"), 2)
