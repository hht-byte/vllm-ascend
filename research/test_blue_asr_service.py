"""Bounded public-audio simulation does not need a model for orchestration tests."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from blue_asr_service import chunk_ends, simulate_cumulative, vllm_engine_options


class PublicASRSimulationTests(unittest.TestCase):
    def test_final_partial_chunk_is_consumed_once(self):
        self.assertEqual(chunk_ends(37, 10, 1.0, 10.0), [10, 20, 30, 37])
        self.assertEqual(chunk_ends(40, 10, 1.0, 10.0), [10, 20, 30, 40])

    def test_limit_caps_input_before_decoding(self):
        seen = []

        def decode(audio, rate):
            seen.append((list(audio), rate))
            return {'text': 'public example', 'token_ids': [1, 2]}

        rows = simulate_cumulative(list(range(50)), 10, decode,
                                   chunk_seconds=1.0, max_audio_seconds=2.5)
        self.assertEqual([len(audio) for audio, _ in seen], [10, 20, 25])
        self.assertEqual([row['audio_end_samples'] for row in rows], [10, 20, 25])
        self.assertTrue(rows[-1]['final'])
        self.assertTrue(all(row['simulation'] == 'cumulative_audio_redecode' for row in rows))
        self.assertTrue(all(row['mtp_enabled'] is False for row in rows))

    def test_bad_timing_is_rejected_before_decoder(self):
        for seconds in [0, -1, float('inf'), float('nan')]:
            with self.assertRaises(ValueError):
                chunk_ends(100, 16000, seconds, 10)
        with self.assertRaises(ValueError):
            chunk_ends(0, 16000, 2, 10)

    def test_decoder_cannot_override_simulation_evidence(self):
        rows = simulate_cumulative([0], 1, lambda *_: {
            'text': 'public', 'token_ids': [2], 'mtp_enabled': True,
            'final': False, 'simulation': 'trained_mtp',
        })
        self.assertFalse(rows[0]['mtp_enabled'])
        self.assertTrue(rows[0]['final'])
        self.assertEqual(rows[0]['simulation'], 'cumulative_audio_redecode')

    def test_cli_rejects_missing_inputs_without_exposing_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            private_path = Path(directory) / 'private-input-name'
            output = Path(directory) / 'attempt'
            result = subprocess.run([
                sys.executable, str(Path(__file__).with_name('blue_asr_service.py')),
                '--model', str(private_path), '--model-revision', 'a' * 40,
                '--audio', str(private_path), '--output-dir', str(output),
            ], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn('private-input-name', result.stderr)
            self.assertFalse(output.exists())

    def test_native_engine_options_bound_audio_and_disable_speculation(self):
        options = vllm_engine_options('local-public-model', 4.0)
        self.assertEqual(options['limit_mm_per_prompt'], {'audio': {'count': 1, 'length': 64000}})
        self.assertEqual(options['max_num_seqs'], 1)
        self.assertEqual(options.get('block_size'), 128)
        self.assertTrue(options['enforce_eager'])
        self.assertFalse(options['async_scheduling'])
        self.assertNotIn('speculative_config', options)


if __name__ == '__main__':
    unittest.main()
