"""Cold, partial prefill and warm mixing through the early device handoff."""
import importlib.util
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch
from test_asr_mtp_device_rejection import DeviceRejectionTests as _Fixture
from test_asr_mtp_device_rejection import _npu_available


class DevicePrefillTests(_Fixture):
    def cold_worker(self):
        worker, scheduler, sampler = self.make_worker()
        worker.finish(tuple(worker.states))
        batch = worker.runner.input_batch
        batch.num_computed_tokens_cpu[:] = 0
        counts = batch.num_tokens_no_spec.tolist()
        worker.runner.query_start_loc.np = np.array([0, *np.cumsum(counts)])
        for request in worker.runner.requests.values():
            request.output_token_ids.clear()
        scheduler.num_scheduled_tokens = dict(zip(batch.req_ids, counts))
        scheduler.scheduled_spec_decode_tokens = {}
        sampler.sampled_token_ids = torch.tensor([[11], [12], [13]], dtype=torch.int64)
        return worker, scheduler, sampler

    def bookkeep(self, worker, cpu):
        runner, batch = worker.runner, worker.runner.input_batch
        discarded = (set(runner.discard_request_indices.np[:runner.num_discarded_requests])
                     if runner.num_discarded_requests else set())
        samples = [[] if i in discarded else [int(t) for t in row if 0 <= t < batch.vocab_size]
                   for i, row in enumerate(cpu.tolist())]
        for i, (req, row) in enumerate(zip(batch.req_ids, samples)):
            c = int(batch.num_tokens_no_spec[i])
            batch.token_ids_cpu[i, c:c+len(row)] = row
            batch.num_tokens_no_spec[i] += len(row)
            runner.requests[req].output_token_ids.extend(row)
        return samples

    def reference(self, worker):
        from asr_mtp_heads import CacheArena
        from asr_mtp_worker import MTPWorker
        arena = CacheArena(self.cfg, 3, max_context=17, slots=4,
                           device='cpu', dtype=torch.float32)
        ref = MTPWorker(deepcopy(worker.runner), deepcopy(worker.heads), arena,
                        self.embed, self.logits, worker.eos, cold_heads=worker.cold_heads,
                        warm_heads=worker.warm_heads, request_heads=worker.request_heads)
        ref.proposed = worker.proposed.copy()
        ref.legacy_submitted = worker.legacy_submitted.copy()
        for req, state in worker.states.items():
            old = arena.acquire()
            old.length = state.length
            old.target[:state.length].copy_(state.target[:state.length])
            for a, b in zip(old.heads, state.heads):
                a.valid = b.valid
                a.key[:, :b.valid].copy_(b.key[:, :b.valid])
                a.value[:, :b.valid].copy_(b.value[:, :b.valid])
                a.hidden[:b.valid].copy_(b.hidden[:b.valid])
            ref.states[req] = old
        return ref

    def compare_handoff(self, worker, scheduler, sampler):
        ref = self.reference(worker)
        total = sum(scheduler.num_scheduled_tokens.values())
        hidden = torch.randn(total, self.cfg.hidden_size)
        cpu = worker.device_handoff.prepare(scheduler, sampler, hidden)
        self.assertIsNotNone(cpu, worker.device_handoff.stats)
        self.assertGreater(cpu.shape[1], 1, 'CPU parser must bypass device _to_list wait')
        samples = self.bookkeep(worker, cpu)
        reference_samples = self.bookkeep(ref, sampler.sampled_token_ids)
        self.assertEqual(samples, reference_samples)
        expected = ref.propose(reference_samples, scheduler, hidden)
        actual = worker.propose(samples, scheduler, hidden)
        self.assertEqual(actual, expected)
        for req, right in ref.states.items():
            left = worker.states[req]
            self.assertEqual(left.length, right.length)
            torch.testing.assert_close(left.target[:left.length], right.target[:right.length])
            for a, b in zip(left.heads, right.heads):
                self.assertEqual(a.valid, b.valid)
                torch.testing.assert_close(a.key[:, :a.valid], b.key[:, :b.valid],
                                           atol=4e-6, rtol=4e-5)
        return actual

    def test_full_cold_prefill_with_single_int64_sample_starts_mtp_before_readback(self):
        worker, scheduler, sampler = self.cold_worker()
        self.assertEqual([len(r) for r in self.compare_handoff(worker, scheduler, sampler)], [3]*3)
        self.assertEqual(worker.device_handoff.stats['cold_requests'], 3)

    def test_first_prefill_external_verification_accepts_variable_prefixes(self):
        worker, scheduler, sampler = self.cold_worker()
        batch = worker.runner.input_batch
        for req in batch.req_ids:
            worker.runner.requests[req].sampling_params.extra_args['ascend_external_draft_first_verify'] = True
        drafts = [[21, 22], [23], [24, 25, 26]]
        scheduler.scheduled_spec_decode_tokens = dict(zip(batch.req_ids, drafts))
        counts = [int(c)+len(d) for c, d in zip(batch.num_tokens_no_spec, drafts)]
        scheduler.num_scheduled_tokens = dict(zip(batch.req_ids, counts))
        worker.runner.query_start_loc.np = np.array([0, *np.cumsum(counts)])
        sampler.sampled_token_ids = torch.tensor([[21, 30, -1, -1], [31, -1, -1, -1],
                                                [24, 25, 26, 32]], dtype=torch.int32)
        self.compare_handoff(worker, scheduler, sampler)

    def test_cold_prefill_and_warm_decode_share_one_replay(self):
        worker, scheduler, sampler = self.make_worker()
        req = worker.runner.input_batch.req_ids[1]
        worker.finish([req])
        batch = worker.runner.input_batch
        batch.num_computed_tokens_cpu[1] = 0
        worker.runner.requests[req].output_token_ids.clear()
        scheduler.scheduled_spec_decode_tokens.pop(req)
        counts = [4, int(batch.num_tokens_no_spec[1]), 4]
        scheduler.num_scheduled_tokens = dict(zip(batch.req_ids, counts))
        worker.runner.query_start_loc.np = np.array([0, *np.cumsum(counts)])
        sampler.sampled_token_ids[1] = torch.tensor([31, -1, -1, -1])
        self.compare_handoff(worker, scheduler, sampler)
        self.assertEqual(worker.device_handoff.stats['cold_requests'], 1)
        self.assertEqual(worker.device_handoff.stats['warm_requests'], 2)

    def test_chunked_prefill_saves_hidden_then_initializes_heads_on_last_chunk(self):
        worker, scheduler, sampler = self.cold_worker()
        runner, batch = worker.runner, worker.runner.input_batch
        runner.num_discarded_requests = 3
        runner.discard_request_indices = type(runner.query_start_loc)(np=np.array([0, 1, 2]))
        scheduler.num_scheduled_tokens = dict.fromkeys(batch.req_ids, 2)
        runner.query_start_loc.np = np.array([0, 2, 4, 6])
        self.assertEqual(self.compare_handoff(worker, scheduler, sampler), [[], [], []])
        self.assertEqual(worker.stats['head_batches'], 0)
        self.assertFalse(worker.proposed)
        batch.num_computed_tokens_cpu[:] = 2
        counts = (batch.num_tokens_no_spec-2).tolist()
        scheduler.num_scheduled_tokens = dict(zip(batch.req_ids, counts))
        runner.query_start_loc.np = np.array([0, *np.cumsum(counts)])
        runner.num_discarded_requests = 0
        self.assertEqual([len(r) for r in self.compare_handoff(worker, scheduler, sampler)], [3]*3)

    def test_legacy_rollback_is_offered_before_cold_mtp(self):
        worker, scheduler, sampler = self.cold_worker()
        for request in worker.runner.requests.values():
            request.sampling_params.extra_args['ascend_external_draft'] = [21, 22]
        self.assertEqual(self.compare_handoff(worker, scheduler, sampler), [[21, 22]]*3)
        self.assertEqual(worker.stats['head_batches'], 0)
        self.assertFalse(worker.proposed)
        batch = worker.runner.input_batch
        batch.num_computed_tokens_cpu[:] = batch.num_tokens_no_spec-1
        scheduler.num_scheduled_tokens = dict.fromkeys(batch.req_ids, 3)
        scheduler.scheduled_spec_decode_tokens = {req: [21, 22] for req in batch.req_ids}
        worker.runner.query_start_loc.np = np.array([0, 3, 6, 9])
        sampler.sampled_token_ids = torch.tensor([[21, 22, 30]]*3, dtype=torch.int32)
        self.assertEqual([len(r) for r in self.compare_handoff(worker, scheduler, sampler)], [3]*3)

    def test_capacity_miss_does_not_lease_cold_slots_or_modify_existing_states(self):
        worker, scheduler, sampler = self.cold_worker()
        held = [worker.arena.acquire() for _ in range(2)]
        before = set(worker.arena.leased)
        result = worker.device_handoff.prepare(scheduler, sampler, torch.randn(18, 12))
        self.assertIsNone(result)
        self.assertEqual(worker.arena.leased, before)
        self.assertFalse(worker.states)
        for state in held:
            worker.arena.release(state)

    def test_partial_prefill_and_warm_decode_can_share_a_batch(self):
        worker, scheduler, sampler = self.make_worker()
        runner, batch = worker.runner, worker.runner.input_batch
        req = batch.req_ids[1]
        worker.finish([req])
        batch.num_computed_tokens_cpu[1] = 0
        runner.requests[req].output_token_ids.clear()
        runner.num_discarded_requests = 1
        runner.discard_request_indices = type(runner.query_start_loc)(np=np.array([1]))
        scheduler.scheduled_spec_decode_tokens.pop(req)
        scheduler.num_scheduled_tokens[req] = 2
        runner.query_start_loc.np = np.array([0, 4, 6, 10])
        outputs = self.compare_handoff(worker, scheduler, sampler)
        self.assertEqual([len(row) for row in outputs], [3, 0, 3])
        self.assertEqual(worker.states[req].length, 2)
        self.assertEqual([h.valid for h in worker.states[req].heads], [0]*3)
        self.assertEqual(worker.device_handoff.stats['warm_requests'], 2)
        self.assertEqual(worker.device_handoff.stats['cache_only_requests'], 1)

    def test_staging_error_releases_new_slots_and_preserves_free_order(self):
        worker, scheduler, sampler = self.cold_worker()
        before = list(worker.arena.free)
        with patch.object(self.program.rejection, 'stage', side_effect=ValueError('stage failed')), \
             self.assertRaisesRegex(ValueError, 'stage failed'):
            worker.device_handoff.prepare(scheduler, sampler, torch.randn(18, 12))
        self.assertEqual(worker.arena.free, before)
        self.assertFalse(worker.arena.leased)
        self.assertFalse(worker.states)
        self.assertIsNone(worker.device_handoff.pending)

    def test_capture_miss_is_cpu_only_before_cold_slots_are_acquired(self):
        worker, scheduler, sampler = self.cold_worker()
        worker.graphs.entries[3, 32][0].rejection = type('NoCapacity', (), {'width': 0})()
        before = list(worker.arena.free)
        result = worker.device_handoff.prepare(scheduler, sampler, torch.randn(18, 12))
        self.assertIsNone(result)
        self.assertEqual(worker.arena.free, before)
        self.assertFalse(worker.states)

    def test_cold_eos_and_output_budget_keep_no_phantom_proposals(self):
        worker, scheduler, sampler = self.cold_worker()
        worker.eos = {11, 12, 13}
        self.assertEqual(self.compare_handoff(worker, scheduler, sampler), [[], [], []])
        self.assertFalse(worker.proposed)
        self.assertGreater(worker.stats['head_positions'], 0)

    def test_cold_known_last_output_caches_without_running_heads(self):
        worker, scheduler, sampler = self.cold_worker()
        for request in worker.runner.requests.values():
            request.sampling_params.max_tokens = 1
        self.assertEqual(self.compare_handoff(worker, scheduler, sampler), [[], [], []])
        self.assertEqual(worker.stats['head_batches'], 0)

    def test_mixed_cold_and_warm_respect_separate_head_policies(self):
        worker, scheduler, sampler = self.make_worker()
        worker.warm_heads = 1
        req = worker.runner.input_batch.req_ids[1]
        worker.finish([req])
        batch = worker.runner.input_batch
        batch.num_computed_tokens_cpu[1] = 0
        worker.runner.requests[req].output_token_ids.clear()
        scheduler.scheduled_spec_decode_tokens.pop(req)
        counts = [4, int(batch.num_tokens_no_spec[1]), 4]
        scheduler.num_scheduled_tokens = dict(zip(batch.req_ids, counts))
        worker.runner.query_start_loc.np = np.array([0, *np.cumsum(counts)])
        sampler.sampled_token_ids[1] = torch.tensor([31, -1, -1, -1])
        self.assertEqual([len(r) for r in self.compare_handoff(worker, scheduler, sampler)], [1, 3, 1])


for _name in dir(_Fixture):
    if _name.startswith('test_') and _name not in DevicePrefillTests.__dict__:
        setattr(DevicePrefillTests, _name, None)
del _Fixture


@unittest.skipUnless(importlib.util.find_spec('vllm'), 'requires installed vLLM parser')
class DevicePrefillParserTests(unittest.TestCase):
    def test_handoff_cpu_parser_matches_installed_parser(self):
        from asr_mtp_device_rejection import CPUHandoffParser
        from vllm.v1.sample.rejection_sampler import RejectionSampler
        generator = torch.Generator().manual_seed(32)
        for batch in (1, 4, 8):
            for width in (2, 4, 16):
                tokens = torch.randint(-2, 70, (batch, width), generator=generator)
                for discard in ([], [0], list(range(batch))):
                    expected = RejectionSampler.parse_output(tokens, 64, discard, logprobs_tensors=None)
                    actual = CPUHandoffParser(tokens).parse(tokens, 64, discard,
                        parser=RejectionSampler.parse_output, logprobs_tensors=None)
                    self.assertEqual(actual, expected)

    def test_upstream_cpu_parser_handles_single_sample_and_discarded_prefill_rows(self):
        from vllm.v1.sample.rejection_sampler import RejectionSampler
        # This is the padded CPU block returned to original bookkeeping. Row 1
        # is discarded intermediate prefill, row 2 is a warm partial acceptance.
        cpu = torch.tensor([[11, -1, -1, -1], [-1, -1, -1, -1],
                            [21, 22, 30, -1]], dtype=torch.int64)
        rows, logprobs = RejectionSampler.parse_output(
            cpu, 64, np.array([1]), logprobs_tensors=None)
        self.assertEqual(rows, [[11], [], [21, 22, 30]])
        self.assertIsNone(logprobs)


@unittest.skipUnless(_npu_available(), 'requires torch-npu and available NPU')
class DevicePrefillNPUTests(unittest.TestCase):
    @torch.inference_mode()
    def test_partial_then_completed_prefill_mixed_with_warm_replays(self):
        from types import SimpleNamespace

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
        program, replay = graphs.entries[3, 32]
        # Row 0 is an intermediate prefill; row 1 completes prefill with
        # an ordinary single-column int64 sampler result.
        hidden = torch.randn(8, 256, device='npu', dtype=torch.float16)
        upper = optimistic_states(b, [7, 5], [3, 5], [0, 0])
        program.stage(upper, [ids[0][:4], ids[1]+[0]], [0, 3], allow_inactive=True)
        program.rejection.stage([7, 5], [3, 5], [20]*2, [20]*2, [0]*2, [set(), set()],
            torch.tensor([[30], [31]], device='npu', dtype=torch.int64), hidden,
            starts=[0, 0], sample_counts=[0, 1])
        replay(prelude=program.rejection.replay)
        combined = program.rejection.readback(2).clone()
        self.assertEqual(combined[0, :4].tolist(), [-1]*4)
        for i, length in enumerate((3, 5)):
            a[i].length = b[i].length = length
            begin = 0 if i == 0 else 3
            a[i].target[:length].copy_(hidden[begin:begin+length])
        expected = eager.propose([a[1]], [ids[1]+[31]], embed, logits, [3])
        torch.testing.assert_close(combined[1, 4:], expected.cpu()[0])
        for head in b[1].heads:
            head.valid = 5
        # Row 0 now completes cold prefill; row 1 verifies a warm draft,
        # rejecting at token zero. Both use the same captured head graph.
        hidden = torch.randn(8, 256, device='npu', dtype=torch.float16)
        upper = optimistic_states(b, [7, 6], [4, 4], [3, 5])
        program.stage(upper, [ids[0]+[0], ids[1]+[31]+[0]*4], [3, 3])
        raw = torch.tensor([[32, -1, -1, -1], [33, -1, -1, -1]],
                           device='npu', dtype=torch.int32)
        program.rejection.stage([7, 6], [4, 4], [20]*2, [20]*2, [0]*2, [set(), set()],
                               raw, hidden, starts=[3, 5], sample_counts=[1, 4])
        replay(prelude=program.rejection.replay)
        actual = program.rejection.readback(2)[:, 4:].clone()
        a[0].target[3:7].copy_(hidden[:4])
        a[1].target[5:6].copy_(hidden[4:5])
        for state, length in zip(a, (7, 6)):
            state.length = length
            state.crop_heads(length)
        expected = eager.propose(a, [ids[0]+[32], ids[1]+[31, 33]], embed, logits, [3]*2)
        torch.testing.assert_close(actual, expected.cpu())
