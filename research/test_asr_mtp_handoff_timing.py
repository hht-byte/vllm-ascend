"""Early handoff attribution and disabled diagnostics boundaries."""
from unittest.mock import patch

import torch

from test_asr_mtp_device_rejection import DeviceRejectionTests as _DeviceRejectionTests


class HandoffTimingTests(_DeviceRejectionTests):
    def test_profile_attributes_early_handoff_without_extra_replay(self):
        worker, scheduler, sampler = self.make_worker()
        worker.profile_stages = True
        worker.device_handoff.prepare(scheduler, sampler, torch.randn(12, 12))
        timing = worker.device_handoff.last_timings
        for name in ('preflight_host_ms', 'program_stage_host_ms',
                     'prelude_stage_host_ms', 'replay_host_ms',
                     'combined_readback_host_ms', 'cpu_conversion_host_ms'):
            self.assertGreaterEqual(timing[name], 0)
        self.assertEqual(timing['batch'], 3)
        self.assertEqual(timing['capacity'], 3)
        self.assertEqual(timing['bucket'], 32)
        self.assertEqual(timing['active_requests'], 3)
        self.assertEqual(worker.device_handoff.stats['batches'], 1)

    def test_disabled_diagnostics_never_read_clock(self):
        worker, scheduler, sampler = self.make_worker()
        with patch('asr_mtp_device_rejection.perf_counter', create=True,
                   side_effect=AssertionError('disabled timer')):
            worker.device_handoff.prepare(scheduler, sampler, torch.randn(12, 12))
        self.assertIsNone(worker.device_handoff.last_timings)

    def test_fallback_only_workload_reports_first_and_periodic_miss(self):
        worker, scheduler, sampler = self.make_worker()
        worker.disabled['stream_b_0'] = 'unsupported_sampling'
        with self.assertLogs('asr_mtp_device_rejection', level='INFO') as logs:
            for _ in range(100):
                self.assertIsNone(worker.device_handoff.prepare(
                    scheduler, sampler, torch.randn(12, 12)))
        reports = [line for line in logs.output if 'ASR_MTP_DEVICE_REJECTION_FALLBACK' in line]
        self.assertEqual(len(reports), 2)
        self.assertIn('fallback_request', reports[-1])
        self.assertEqual(worker.device_handoff.stats['fallbacks'], 100)


for _name in dir(_DeviceRejectionTests):
    if _name.startswith('test_'):
        setattr(HandoffTimingTests, _name, None)
del _DeviceRejectionTests
