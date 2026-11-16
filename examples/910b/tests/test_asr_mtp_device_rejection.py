"""Device rejection metadata and optimistic causal MTP equivalence."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_asr_mtp_paged import PagedProgramTests as _PagedFixture


class DeviceRejectionTests(_PagedFixture):
    # Only use the parent fixture, not its inherited test cases.
    def test_device_zero_partial_full_acceptance_matches_confirmed_recompute(self):
        from asr_mtp_device_rejection import RejectionPrelude

        # Establish genuine auxiliary history, then verify a three-token draft.
        old_drafts = self.compare(self.a, self.b, [r + [4] for r in self.ids], [3] * 3)
        previous = [len(r) + 1 for r in self.ids]
        counts = [4] * 3
        samples = torch.tensor([[31, -1, -1, -1], [int(old_drafts[1, 0]), 33, -1, -1],
                                [*old_drafts[2].tolist(), 37]], dtype=torch.int32)
        hidden = torch.randn(12, 12)
        optimistic, rows = [], []
        for b, c, tokens in zip(self.b, previous, self.ids):
            heads = [replace(h, valid=min(h.valid, max(0, c-d)))
                     for d, h in enumerate(b.heads, 1)]
            optimistic.append(replace(b, length=c+3, heads=heads))
            rows.append(tokens + [4, 0, 0, 0, 0])
        self.program.stage(optimistic, rows, [3] * 3)
        prelude = RejectionPrelude(self.program, sample_width=4, vocab_size=64)
        prelude.stage(previous, counts, [20]*3, [20]*3, [0]*3, [set()]*3,
                      samples, hidden)
        with unittest.mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('D2H')):
            prelude.run()
        actual = self.program.run()[:3].clone()
        expected_rows, limits = [], [3] * 3
        for i, (a, c, old) in enumerate(zip(self.a, previous, self.ids)):
            new = [t for t in samples[i].tolist() if t >= 0]
            a.length = c + len(new) - 1
            a.target[c-1:a.length] = hidden[i*4:i*4+len(new)]
            a.crop_heads(a.length)
            expected_rows.append(old + [4] + new)
        expected = self.eager.propose(self.a, expected_rows, self.embed, self.logits, limits)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(self.program.lengths[:3].tolist(), [8, 5, 9])
        for a, b in zip(self.a, self.b):
            for x, y in zip(a.heads, b.heads):
                torch.testing.assert_close(x.key[:, :a.length], y.key[:, :a.length],
                                           atol=3e-6, rtol=3e-5)

    def test_device_eos_budget_and_combined_output(self):
        from asr_mtp_device_rejection import RejectionPrelude

        rows = [r + [4] for r in self.ids]
        self.program.stage(self.b, rows, [3, 3, 3])
        prelude = RejectionPrelude(self.program, sample_width=4, vocab_size=64)
        samples = torch.tensor([[5, -1, -1, -1], [9, -1, -1, -1],
                                [10, -1, -1, -1]], dtype=torch.int32)
        prelude.stage([len(r) for r in self.ids], [1]*3, [10, 2, 10],
                      [10, 10, 2], [0, 0, 0], [{5}, set(), set()],
                      samples, torch.randn(3, 12))
        prelude.run()
        self.assertEqual(self.program.limits[:3].tolist(), [0, 1, 1])
        self.program.run()
        combined = prelude.readback(3)
        self.assertEqual(combined[:, :4].tolist(), samples.tolist())
        self.assertEqual(combined[0, 4:].tolist(), [0, 0, 0])
        self.assertEqual(combined[1:, 5:].tolist(), [[0, 0], [0, 0]])

    def make_worker(self):
        import numpy as np
        from asr_mtp_graph import GraphOptions
        from asr_mtp_token_graph import TokenMTPGraphs
        from asr_mtp_worker import MTPWorker
        options = GraphOptions(3, (), 17, tokens=(32,), head_counts=(3,))
        graphs = TokenMTPGraphs(options, self.fixed, self.new, self.embed, self.logits)
        graphs.ready = True
        graphs.entries[3, 32] = (self.program, lambda prelude=None: (
            prelude() if prelude else None, self.program.run()))
        from asr_mtp_device_rejection import RejectionPrelude
        self.program.rejection = RejectionPrelude(self.program, sample_width=4, vocab_size=64)
        ids = ['stream_a_0', 'stream_b_0', 'stream_c_0']
        params = SimpleNamespace(extra_args={'ascend_mtp': True}, temperature=0.0,
                                 n=1, max_tokens=30, min_tokens=0, logprobs=None)
        batch = SimpleNamespace(req_ids=ids, num_tokens_no_spec=np.array([len(r)+1 for r in self.ids]),
            num_computed_tokens_cpu=np.array([len(r) for r in self.ids]), max_model_len=32,
            token_ids_cpu=np.zeros((3, 32), dtype=np.int32), spec_decode_unsupported_reqs=set(),
            vocab_size=64)
        for i, row in enumerate(self.ids):
            batch.token_ids_cpu[i, :len(row)+1] = row+[4]
        runner = SimpleNamespace(input_batch=batch, num_discarded_requests=0,
            use_async_scheduling=False, routed_experts_initialized=False,
            requests={r: SimpleNamespace(sampling_params=params, output_token_ids=[4]) for r in ids},
            speculative_config=SimpleNamespace(num_speculative_tokens=3),
            query_start_loc=SimpleNamespace(np=np.array([0, 4, 8, 12])))
        worker = MTPWorker(runner, self.fixed, self.new, self.embed, self.logits, set(),
                           graphs=graphs, device_rejection=True, acceptance_stats=True)
        worker.states = dict(zip(ids, self.b))
        worker.proposed.update(ids)
        with torch.inference_mode():
            drafts = self.compare(self.a, self.b, [r+[4] for r in self.ids], [3]*3).tolist()
        scheduler = SimpleNamespace(num_scheduled_tokens=dict.fromkeys(ids, 4),
             scheduled_spec_decode_tokens=dict(zip(ids, drafts)), num_invalid_spec_tokens={})
        for r, row in zip(ids, drafts):
            worker.acceptance.offer(r, row, worker.states[r].length, fresh_positions=0)
        samples = torch.tensor([[31, -1, -1, -1], [drafts[1][0], 33, -1, -1],
                                [*drafts[2], 37]], dtype=torch.int32)
        return worker, scheduler, SimpleNamespace(sampled_token_ids=samples, logprobs_tensors=None)

    def test_worker_launches_before_readback_then_consumes_without_second_proposal(self):
        worker, scheduler, sampler = self.make_worker()
        hidden = torch.randn(12, 12)
        events = []
        real_run, real_readback = self.program.run, self.program.rejection.readback
        worker.graphs.entries[3, 32] = (self.program, lambda prelude=None: (
            events.append('launch'), prelude(), real_run()))
        with patch.object(self.program.rejection, 'readback', side_effect=lambda n: (
                events.append('readback'), real_readback(n))[1]):
            cpu = worker.device_handoff.prepare(scheduler, sampler, hidden)
        self.assertEqual(events, ['launch', 'readback'])
        sampled = [[t for t in row if t >= 0] for row in cpu.tolist()]
        for i, (r, row) in enumerate(zip(worker.runner.input_batch.req_ids, sampled)):
            batch = worker.runner.input_batch
            c = int(batch.num_tokens_no_spec[i])
            batch.token_ids_cpu[i, c:c+len(row)] = row
            batch.num_tokens_no_spec[i] += len(row)
            worker.runner.requests[r].output_token_ids.extend(row)
        with patch.object(worker.graphs, 'propose', side_effect=AssertionError('second draft')):
            output = worker.propose(sampled, scheduler, hidden)
        self.assertEqual([len(row) for row in output], [3, 3, 3])
        for r in worker.runner.input_batch.req_ids:
            state = worker.states[r]
            self.assertEqual([h.valid for h in state.heads], [state.length]*3)
        self.assertIsNone(worker.device_handoff.pending)

    def test_disabled_mixed_and_invalid_spec_fallback_do_not_touch_device_or_pending(self):
        worker, scheduler, sampler = self.make_worker()
        with patch.object(self.program, 'stage', side_effect=AssertionError('must preflight')):
            worker.disabled['stream_b_0'] = 'unsupported_sampling'
            self.assertIsNone(worker.device_handoff.prepare(scheduler, sampler, torch.zeros(12, 12)))
            del worker.disabled['stream_b_0']
            scheduler.num_invalid_spec_tokens['stream_a_0'] = 1
            self.assertIsNone(worker.device_handoff.prepare(scheduler, sampler, torch.zeros(12, 12)))
        self.assertIsNone(worker.device_handoff.pending)

    def test_reordered_warm_batch_matches_reference_and_can_resume_normal_graphs(self):
        worker, scheduler, sampler = self.make_worker()
        batch = worker.runner.input_batch
        order = [2, 0, 1]
        batch.req_ids = [batch.req_ids[i] for i in order]
        for name in ('num_tokens_no_spec', 'num_computed_tokens_cpu', 'token_ids_cpu'):
            setattr(batch, name, getattr(batch, name)[order].copy())
        sampler.sampled_token_ids = sampler.sampled_token_ids[order]
        reference = [self.a[i] for i in order]
        hidden = torch.randn(12, 12)
        cpu = worker.device_handoff.prepare(scheduler, sampler, hidden)
        samples = [[v for v in row if v >= 0] for row in cpu.tolist()]
        inputs = []
        for i, (req, state, row) in enumerate(zip(batch.req_ids, reference, samples)):
            c = int(batch.num_tokens_no_spec[i])
            state.length = c+len(row)-1
            state.crop_heads(state.length)
            state.target[c-1:state.length] = hidden[i*4:i*4+len(row)]
            batch.token_ids_cpu[i, c:c+len(row)] = row
            batch.num_tokens_no_spec[i] += len(row)
            worker.runner.requests[req].output_token_ids.extend(row)
            inputs.append(batch.token_ids_cpu[i, :int(batch.num_tokens_no_spec[i])].tolist())
        expected = self.eager.propose(reference, inputs, self.embed, self.logits, [3]*3)
        got = worker.propose(samples, scheduler, hidden)
        self.assertEqual(got, expected.tolist())
        # A subsequent normal (synchronous) proposal must see only true prefixes.
        for i, (req, state) in enumerate(zip(batch.req_ids, reference)):
            actual = worker.states[req]
            self.assertEqual(actual.length, state.length)
            for a, b in zip(actual.heads, state.heads):
                self.assertEqual(a.valid, b.valid)
                torch.testing.assert_close(a.key[:, :a.valid], b.key[:, :b.valid],
                                           atol=3e-6, rtol=3e-5)

    def test_all_terminal_samples_do_not_offer_phantom_drafts(self):
        worker, scheduler, sampler = self.make_worker()
        before = worker.heads.processed_positions
        worker.eos = {63}
        sampler.sampled_token_ids.fill_(-1)
        sampler.sampled_token_ids[:, 0] = 63
        hidden = torch.randn(12, 12)
        cpu = worker.device_handoff.prepare(scheduler, sampler, hidden)
        self.assertEqual(cpu[:, 0].tolist(), [63]*3)
        for i, req in enumerate(worker.runner.input_batch.req_ids):
            batch = worker.runner.input_batch
            c = int(batch.num_tokens_no_spec[i])
            batch.token_ids_cpu[i, c] = 63
            batch.num_tokens_no_spec[i] += 1
            worker.runner.requests[req].output_token_ids.append(63)
        self.assertEqual(worker.propose([[63]]*3, scheduler, hidden), [[], [], []])
        self.assertIsNone(worker.device_handoff.pending)
        planned = worker.device_handoff.stats['optimistic_positions']
        self.assertEqual(worker.heads.processed_positions-before, planned)
        self.assertEqual(worker.stats['head_positions'], planned)
        self.assertEqual(worker.stats['head_batches'], 1)

    def test_executed_head_cost_includes_heads_cut_by_remaining_output_budget(self):
        worker, scheduler, sampler = self.make_worker()
        for req in worker.runner.requests.values():
            req.sampling_params.max_tokens = 6
        before = worker.heads.processed_positions
        hidden = torch.randn(12, 12)
        cpu = worker.device_handoff.prepare(scheduler, sampler, hidden)
        sampled = [[v for v in row if v >= 0] for row in cpu.tolist()]
        for i, (req, row) in enumerate(zip(worker.runner.input_batch.req_ids, sampled)):
            batch = worker.runner.input_batch
            c = int(batch.num_tokens_no_spec[i])
            batch.token_ids_cpu[i, c:c+len(row)] = row
            batch.num_tokens_no_spec[i] += len(row)
            worker.runner.requests[req].output_token_ids.extend(row)
        output = worker.propose(sampled, scheduler, hidden)
        self.assertEqual([len(r) for r in output], [3, 3, 1])
        planned = worker.device_handoff.stats['optimistic_positions']
        self.assertEqual(worker.heads.processed_positions-before, planned)
        self.assertEqual(worker.stats['head_positions'], planned)
        # Acceptance attribution includes all head work spent on this proposal.
        self.assertEqual(sum(v['proposed_fresh_positions'] for v in worker.acceptance.rows.values()),
                         planned)

    def test_wrong_scheduler_and_lifetime_changes_fail_before_reusing_pending(self):
        worker, scheduler, sampler = self.make_worker()
        worker.device_handoff.prepare(scheduler, sampler, torch.randn(12, 12))
        with self.assertRaisesRegex(RuntimeError, 'lifetimes'):
            worker.finish(['stream_a_0'])
        with self.assertRaisesRegex(RuntimeError, 'batch changed'):
            worker.device_handoff.take(object())

    def test_eos_before_min_tokens_does_not_disable_following_heads(self):
        from asr_mtp_device_rejection import RejectionPrelude
        self.program.stage(self.b, [r+[4] for r in self.ids], [3]*3)
        prelude = RejectionPrelude(self.program, sample_width=4, vocab_size=64)
        prelude.stage([len(r) for r in self.ids], [1]*3, [10]*3, [10]*3,
                      [2]*3, [{63}]*3, torch.tensor([[63], [63], [63]], dtype=torch.int32),
                      torch.randn(3, 12))
        prelude.run()
        self.assertEqual(self.program.limits[:3].tolist(), [3]*3)

    def test_shorter_reuse_ignores_stale_sample_columns_without_hot_path_fills(self):
        from asr_mtp_device_rejection import RejectionPrelude
        self.program.stage(self.b, [r+[4] for r in self.ids], [3]*3)
        prelude = RejectionPrelude(self.program, sample_width=4, vocab_size=64)
        prelude.raw.fill_(63)  # stale columns from a previous wider sample block
        with patch.object(torch.Tensor, 'fill_', side_effect=AssertionError('eager fill')), \
             patch.object(torch.Tensor, 'zero_', side_effect=AssertionError('eager zero')):
            prelude.stage([len(r) for r in self.ids], [1]*3, [10]*3, [10]*3,
                          [0]*3, [{63}]*3, torch.tensor([[5], [5], [5]], dtype=torch.int32),
                          torch.randn(3, 12))
        prelude.run()
        self.assertEqual(self.program.limits[:3].tolist(), [3]*3)
        self.assertEqual(self.program.lengths[:3].tolist(), [len(r) for r in self.ids])


# Avoid running inherited parity cases twice; the parent fixture stays reusable.
for _name in dir(_PagedFixture):
    if _name.startswith('test_') and _name not in DeviceRejectionTests.__dict__:
        setattr(DeviceRejectionTests, _name, None)
del _PagedFixture


def _npu_available():
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except ImportError:
        return False


@unittest.skipUnless(_npu_available(), 'requires torch-npu and available NPU')
class DeviceRejectionNPUTests(unittest.TestCase):
    @torch.inference_mode()
    def test_real_prelude_and_fia_replay_matches_eager_after_partial_rejection(self):
        from copy import deepcopy

        from asr_mtp_device_rejection import optimistic_states
        from asr_mtp_graph import GraphOptions
        from asr_mtp_heads import BatchedMTPHeads, CacheArena
        from asr_mtp_paged_cache import PagedCacheArena
        from asr_mtp_token_graph import TokenMTPGraphs
        torch.manual_seed(123)
        cfg = SimpleNamespace(hidden_size=256, intermediate_size=512,
            num_attention_heads=2, num_key_value_heads=1, head_dim=128,
            attention_bias=False, rms_norm_eps=1e-6, hidden_act='silu',
            rope_theta=10000, rope_scaling=None)
        eager = BatchedMTPHeads(cfg, 3).to('npu', torch.float16).eval()
        fixed = deepcopy(eager)
        embed = torch.nn.Embedding(64, 256).to('npu', torch.float16).eval()
        logits = torch.nn.Linear(256, 64).to('npu', torch.float16).eval()
        old = CacheArena(cfg, 3, max_context=32, slots=2, device='npu', dtype=torch.float16)
        new = PagedCacheArena(cfg, 3, max_context=32, slots=2, block_size=128,
            dummy_tokens=33, device='npu', dtype=torch.float16, scratch_elements=33*256)
        graphs = TokenMTPGraphs(GraphOptions(2, (), 32, tokens=(32,), head_counts=(3,)),
                                fixed, new, embed, logits)
        graphs.device_rejection = (4, 64)
        graphs.prepare()
        a, b = [old.acquire() for _ in range(2)], [new.acquire() for _ in range(2)]
        ids = [list(range(3, 10)), list(range(12, 17))]
        for left, right, row in zip(a, b, ids):
            left.length = right.length = len(row)
            hidden = torch.randn(len(row), 256, device='npu', dtype=torch.float16)
            left.target[:len(row)] = right.target[:len(row)] = hidden
        expected = eager.propose(a, [row+[4] for row in ids], embed, logits, [3]*2)
        normal = graphs.propose(b, [row+[4] for row in ids], [3]*2)
        torch.testing.assert_close(normal, expected)
        accepted = [[int(expected[0, 0]), 31], [*expected[1].tolist(), 32]]
        raw = torch.tensor([accepted[0]+[-1]*2, accepted[1]], device='npu', dtype=torch.int32)
        hidden = torch.randn(8, 256, device='npu', dtype=torch.float16)
        c = [len(row)+1 for row in ids]
        upper = optimistic_states(b, c, [4, 4])
        program, replay = graphs.entries[3, 32]
        program.stage(upper, [row+[4]+[0]*4 for row in ids], [3]*2)
        program.rejection.stage(c, [4]*2, [20]*2, [20]*2, [0]*2, [set(), set()], raw, hidden)
        replay(prelude=program.rejection.replay)
        actual = program.rejection.readback(2)[:, 4:].clone()
        rows = []
        for i, (state, count, tokens) in enumerate(zip(a, c, accepted)):
            state.length = count+len(tokens)-1
            state.crop_heads(state.length)
            state.target[count-1:state.length] = hidden[i*4:i*4+len(tokens)]
            rows.append(ids[i]+[4]+tokens)
        expected = eager.propose(a, rows, embed, logits, [3]*2).cpu()
        torch.testing.assert_close(actual, expected)
