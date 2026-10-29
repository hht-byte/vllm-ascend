"""Check the configurable prespec launch point against 16 kHz audio."""

import math
import unittest
from types import SimpleNamespace

import streaming_prespec
from streaming_prespec_helpers import PreSpecModelHelpers


class LaunchTimingTests(unittest.TestCase):
    def test_default_launch_is_at_1_6_seconds(self):
        self.assertEqual(streaming_prespec.AdmissionPolicy().midpoint_samples, 25_600)

    def test_launch_seconds_convert_to_samples_within_chunk(self):
        convert = streaming_prespec.launch_samples_for_window
        self.assertEqual(convert(1.6, 2.0), 25_600)
        self.assertEqual(convert(1.0, 2.0), 16_000)
        for launch_sec in (0.0, -0.1, 2.0, 2.1, math.nan, math.inf, 1e-9):
            with self.subTest(launch_sec=launch_sec):
                with self.assertRaises(ValueError):
                    convert(launch_sec, 2.0)
        with self.assertRaises(ValueError):
            convert(1.6, 1.5)
        with self.assertRaises(ValueError):
            convert(1.6, 1e308)

    def test_model_launches_at_exact_sample_and_deadline_has_400_ms(self):
        captured = []

        class Model(PreSpecModelHelpers):
            pass

        model = Model()
        model.prespec = SimpleNamespace(
            policy=streaming_prespec.AdmissionPolicy(),
            clock=lambda: 1_000_000_000,
            maybe_start=lambda request, load, predicted: (
                captured.append(request)
                or streaming_prespec.LaunchDecision(True, "launched")
            ),
        )

        def attempt(samples):
            return model.try_start_prespec(
                ws_id="ws", seg_id=1, window_id=0, prefix_version=0,
                full_audio_pcm_f32le=b"\x00" * (samples * 4),
                current_audio_end_samples=samples,
                last_formal_audio_end_samples=0,
                chunk_size_samples=32_000, prefix_text="",
                stable_prefix_token_ids=(), rollback_draft_token_ids=(),
                model_context={}, sampling_params=SimpleNamespace(),
                committed=False, audio_queue_empty=True,
            )

        self.assertEqual(attempt(25_599).reason, "before_midpoint")
        self.assertFalse(captured)
        self.assertTrue(attempt(25_600).launched)
        self.assertEqual(captured[0].window_audio_samples, 25_600)
        self.assertEqual(captured[0].visible_audio_end_samples, 25_600)
        self.assertEqual(captured[0].deadline_ns, 1_400_000_000)
        self.assertEqual(attempt(32_000).reason, "formal_chunk_due")
        self.assertEqual(len(captured), 1)


if __name__ == "__main__":
    unittest.main()
