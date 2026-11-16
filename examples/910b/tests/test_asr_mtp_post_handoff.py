"""Preserve handoff semantics while reducing CPU work after the MTP graph."""
import unittest
from unittest.mock import patch

import numpy as np
import test_asr_mtp_device_prefill as fixtures
import torch


class TailOnlyTokens(np.ndarray):
    def __getitem__(self, key):
        if isinstance(key, tuple) and isinstance(key[1], slice) and key[1].start is None:
            raise AssertionError('pending MTP must not materialize the full token history')
        return super().__getitem__(key)


class PostHandoffTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DevicePrefillTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def pending(self):
        worker, scheduler, sampler = self.fixture.make_worker()
        hidden = torch.randn(12, 12)
        cpu = worker.device_handoff.prepare(scheduler, sampler, hidden)
        samples = self.fixture.bookkeep(worker, cpu)
        return worker, scheduler, samples, hidden

    def test_pending_consumption_only_reads_confirmed_tail_and_keeps_exact_drafts(self):
        worker, scheduler, samples, hidden = self.pending()
        expected = [row.copy() for row in worker.device_handoff.pending[2]]
        worker.runner.input_batch.token_ids_cpu = worker.runner.input_batch.token_ids_cpu.view(TailOnlyTokens)
        with patch.object(worker.graphs, 'propose', side_effect=AssertionError('second MTP execution')):
            self.assertEqual(worker.propose(samples, scheduler, hidden), expected)
        self.assertIsNone(worker.device_handoff.pending)

    def test_pending_consumption_still_rejects_inconsistent_bookkeeping_tail(self):
        worker, scheduler, samples, hidden = self.pending()
        batch = worker.runner.input_batch
        batch.token_ids_cpu[0, int(batch.num_tokens_no_spec[0])-1] = 62
        with self.assertRaisesRegex(RuntimeError, 'bookkept'):
            worker.propose(samples, scheduler, hidden)

    def test_readback_submission_does_not_wait_and_reuse_requires_finish(self):
        prelude = self.fixture.make_worker()[0].graphs.entries[3, 32][0].rejection
        self.assertTrue(callable(getattr(prelude, 'submit_readback', None)))
        events = []

        class Event:
            def record(self, *args):
                events.append('record')
            def synchronize(self):
                events.append('wait')

        prelude.event = Event()
        with patch.object(torch, 'npu', create=True) as npu:
            npu.current_stream.return_value = object()
            prelude.combined.fill_(11)
            prelude.submit_readback()
            self.assertEqual(events, ['record'])
            with self.assertRaisesRegex(RuntimeError, 'pending'):
                prelude.submit_readback()
            with self.assertRaisesRegex(RuntimeError, 'pending'):
                prelude.stage([], [], [], [], [], [], torch.empty(0, 1), torch.empty(0, 12))
            cpu = prelude.finish_readback(3)
            self.assertEqual(events, ['record', 'wait'])
            self.assertEqual(cpu.tolist(), [[11]*7]*3)
            with self.assertRaisesRegex(RuntimeError, 'pending'):
                prelude.finish_readback(3)

    def test_default_handoff_does_not_read_clock_and_opt_in_has_stage_spans(self):
        import asr_mtp_device_rejection as module
        worker, scheduler, sampler = self.fixture.make_worker()
        with patch.object(module, 'perf_counter', side_effect=AssertionError('default timing overhead')):
            worker.device_handoff.prepare(scheduler, sampler, torch.zeros(12, 12))
        self.assertIsNone(worker.device_handoff.profile)
        worker.device_handoff.take(scheduler)
        worker.profile_stages = True
        worker.device_handoff.prepare(scheduler, sampler, torch.zeros(12, 12))
        profile = worker.device_handoff.profile
        self.assertEqual(profile['requests'], 3)
        for field in ('prepare_host_ms', 'replay_submit_ms', 'readback_submit_ms',
                      'overlap_cpu_ms', 'readback_wait_ms', 'result_host_ms'):
            self.assertGreaterEqual(profile[field], 0)

    def test_cpu_handoff_parser_preserves_filter_discard_and_fallback(self):
        from asr_mtp_device_rejection import CPUHandoffParser
        tokens = torch.tensor([[4, -1, 9, 64], [3, -2, -1, 8]], dtype=torch.int64)
        fallback = unittest.mock.Mock(return_value=('fallback', 'logprobs'))
        parser = CPUHandoffParser(tokens)
        self.assertEqual(parser.parse(tokens, 64, np.array([0]), parser=fallback,
                                      logprobs_tensors=None), ([[], [3, -2, 8]], None))
        fallback.assert_not_called()
        self.assertEqual(parser.parse(tokens, 64, [], parser=fallback,
                                      logprobs_tensors=object()), ('fallback', 'logprobs'))
        self.assertEqual(parser.parse(tokens.clone(), 64, [], parser=fallback,
                                      logprobs_tensors=None), ('fallback', 'logprobs'))


if __name__ == '__main__':
    unittest.main()
