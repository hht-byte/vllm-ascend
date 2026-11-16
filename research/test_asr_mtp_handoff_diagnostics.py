"""A completely falling-back handoff must still explain its eligibility gate."""
import unittest
from unittest.mock import patch

import test_asr_mtp_device_rejection as fixtures
import torch


class HandoffDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DeviceRejectionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.worker, self.scheduler, self.sampler = self.fixture.make_worker()
        self.handoff = self.worker.device_handoff
        self.hidden = torch.zeros(12, 12)

    def test_zero_successes_reports_batch_capacity_and_periodic_totals_without_device_reads(self):
        from dataclasses import replace
        self.worker.graphs.options = replace(self.worker.graphs.options, max_batch=2)
        with self.assertLogs('asr_mtp_device_rejection', level='INFO') as logs, \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('device read')), \
             patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('device read')), \
             patch.object(self.fixture.program, 'stage', side_effect=AssertionError('stage on miss')):
            for _ in range(108):
                self.assertIsNone(self.handoff.prepare(self.scheduler, self.sampler, self.hidden))
        self.assertEqual(len(logs.records), 2, 'first failure and periodic total, not every step')
        text = '\n'.join(logs.output)
        self.assertIn('reason=batch', text)
        self.assertIn("'num_requests': 3", text)
        self.assertIn("'max_batch': 2", text)
        self.assertIn('attempts=100', text)
        self.assertIn('batches=0', text)
        self.assertEqual(self.handoff.stats['attempts'], 108)
        self.assertEqual(self.handoff.stats['fallback_batches'], 108)
        self.assertIsNone(self.handoff.pending)

    def test_new_failure_categories_are_reported_immediately_with_cpu_gate_values(self):
        from dataclasses import replace
        original_options = self.worker.graphs.options
        with self.assertLogs('asr_mtp_device_rejection', level='INFO') as logs, \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('device read')), \
             patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('device read')):
            self.worker.graphs.options = replace(original_options, context=2)
            self.assertIsNone(self.handoff.prepare(self.scheduler, self.sampler, self.hidden))
            self.worker.graphs.options = original_options
            self.worker.disabled['stream_a_0'] = 'missing_hidden_prefix'
            self.assertIsNone(self.handoff.prepare(self.scheduler, self.sampler, self.hidden))
            self.worker.disabled.clear()
            self.scheduler.num_invalid_spec_tokens['stream_a_0'] = 1
            self.assertIsNone(self.handoff.prepare(self.scheduler, self.sampler, self.hidden))
        self.assertEqual(len(logs.records), 3)
        context, request, geometry = logs.output
        self.assertIn('reason=context', context)
        self.assertIn("'context_limit': 2", context)
        self.assertIn('reason=request', request)
        self.assertIn("'disabled': 'missing_hidden_prefix'", request)
        self.assertIn('reason=geometry', geometry)
        self.assertIn("'invalid_spec_tokens': 1", geometry)
        self.assertIn('stream_a_0', geometry)
        self.assertEqual(self.handoff.stats['attempts'], 3)
        self.assertIsNone(self.handoff.pending)

    def test_success_after_fallback_counts_both_attempts_and_preserves_handoff_output(self):
        self.worker.disabled['stream_a_0'] = 'missing_hidden_prefix'
        with self.assertLogs('asr_mtp_device_rejection', level='INFO'):
            self.assertIsNone(self.handoff.prepare(self.scheduler, self.sampler, self.hidden))
        self.worker.disabled.clear()
        with self.assertLogs('asr_mtp_device_rejection', level='INFO') as logs:
            cpu = self.handoff.prepare(self.scheduler, self.sampler, self.hidden)
        self.assertIsNotNone(cpu)
        torch.testing.assert_close(cpu, self.sampler.sampled_token_ids.to(torch.int64))
        self.assertEqual(self.handoff.stats['attempts'], 2)
        self.assertEqual(self.handoff.stats['batches'], 1)
        self.assertEqual(self.handoff.stats['fallback_batches'], 1)
        self.assertIn('attempts=2', '\n'.join(logs.output))


if __name__ == '__main__':
    unittest.main()
