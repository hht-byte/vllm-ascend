"""Real asyncio scheduling checks; these timings make no device-performance claim."""
import asyncio
import importlib.util
import contextlib
import hashlib
import io
import json
import tempfile
from time import perf_counter
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


def cases():
    return [{'case_id': f'clip{clip}-vad{segment}', 'clip_id': f'clip{clip}',
             'reference': {'text': f'text-{clip}-{segment}', 'token_ids': [clip + 1, segment + 10]}}
            for clip in range(4) for segment in range(3)]


def load_module(test):
    path = Path(__file__).with_name('blue_mtp_load.py')
    test.assertTrue(path.exists(), 'closed-loop load client is missing')
    spec = importlib.util.spec_from_file_location('isolated_blue_mtp_load', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LoadContracts(unittest.IsolatedAsyncioTestCase):
    async def test_closed_loop_warmup_three_windows_and_unmeasured_drain(self):
        module = load_module(self)
        active = 0
        peak = 0
        active_by_client = Counter()
        first_cases = {}

        async def transcribe(case, request_id):
            nonlocal active, peak
            client = int(request_id.split('-')[1])
            first_cases.setdefault(client, case['case_id'])
            active += 1
            active_by_client[client] += 1
            self.assertEqual(active_by_client[client], 1)
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.002 + client * 0.0005)
                return {**case['reference'], 'finish_reason': 'stop'}
            finally:
                active -= 1
                active_by_client[client] -= 1

        result = await module.run_load(cases(), transcribe, clients=3,
            ramp_interval_s=0.005, rounds=3, completions_per_round=4, warmup_per_client=2)
        self.assertEqual(active, 0)
        self.assertEqual(peak, 3)
        self.assertEqual(result['maximum_inflight'], 3)
        self.assertEqual(first_cases, {0: 'clip0-vad0', 1: 'clip1-vad0', 2: 'clip2-vad0'})
        self.assertEqual([window['completions'] for window in result['rounds']], [4, 4, 4])
        measured = [sample for sample in result['samples'] if sample['phase'] == 'measured']
        self.assertEqual(len(measured), 12)
        self.assertEqual(Counter(sample['round'] for sample in measured), {1: 4, 2: 4, 3: 4})
        self.assertTrue(all(count >= 2 for count in result['warmup']['completed_per_client']))
        for client in range(3):
            qualified = [sample for sample in result['samples'] if sample['phase'] == 'warmup'
                         and sample['client'] == client
                         and sample['start_s'] >= result['ramp']['end_s']]
            self.assertGreaterEqual(len(qualified), 2)
        self.assertGreater(result['drain']['completions'], 0)
        self.assertEqual(len([s for s in result['samples'] if s['phase'] == 'drain']),
                         result['drain']['completions'])
        for i, window in enumerate(result['rounds']):
            self.assertEqual(window['output_tokens'], 8)
            self.assertGreater(window['duration_s'], 0)
            self.assertAlmostEqual(window['requests_per_second'], 4 / window['duration_s'])
            self.assertAlmostEqual(window['output_tokens_per_second'], 8 / window['duration_s'])
            self.assertGreater(window['latency_ms']['p50'], 0)
            if i:
                self.assertEqual(window['start_s'], result['rounds'][i - 1]['end_s'])
        last = result['rounds'][-1]
        self.assertTrue(all(s['end_s'] >= last['end_s'] for s in result['samples'] if s['phase'] == 'drain'))

    async def test_parity_failure_stops_refills_and_cancels_other_inflight_requests(self):
        module = load_module(self)
        active = 0
        starts = 0
        calls_after_mismatch = 0
        mismatch_emitted = False

        async def transcribe(case, request_id):
            nonlocal active, starts, calls_after_mismatch, mismatch_emitted
            if mismatch_emitted:
                calls_after_mismatch += 1
            active += 1
            starts += 1
            try:
                if request_id.startswith('client-00-'):
                    await asyncio.sleep(0.004)
                    mismatch_emitted = True
                    return {'text': 'wrong', 'token_ids': [999], 'finish_reason': 'stop'}
                await asyncio.sleep(0.2)
                return {**case['reference'], 'finish_reason': 'stop'}
            finally:
                active -= 1

        with self.assertRaises(module.ParityError):
            await module.run_load(cases(), transcribe, clients=3, ramp_interval_s=0,
                rounds=3, completions_per_round=4, warmup_per_client=2)
        self.assertEqual(active, 0)
        self.assertEqual(starts, 3)
        self.assertEqual(calls_after_mismatch, 0)

    async def test_text_and_token_parity_are_both_required_even_during_drain(self):
        module = load_module(self)
        for field, wrong in (('text', 'different'), ('token_ids', [777])):
            async def transcribe(case, request_id):
                await asyncio.sleep(0.001)
                value = {**case['reference'], 'finish_reason': 'stop'}
                value[field] = wrong
                return value
            with self.subTest(field=field), self.assertRaises(module.ParityError):
                await module.run_load(cases(), transcribe, clients=1, ramp_interval_s=0,
                    rounds=1, completions_per_round=1, warmup_per_client=0)

    async def test_invalid_case_reference_and_load_bounds_fail_before_work(self):
        module = load_module(self)
        called = False
        async def transcribe(case, request_id):
            nonlocal called
            called = True
        malformed = [{'case_id': 'a', 'clip_id': 'clip',
                      'reference': {'text': 'hello', 'token_ids': [True]}}]
        for kwargs in ({'clients': 0}, {'rounds': 0}, {'completions_per_round': 0},
                       {'ramp_interval_s': -1}, {'warmup_per_client': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                await module.run_load(cases(), transcribe, **kwargs)
        with self.assertRaises((TypeError, ValueError)):
            await module.run_load(malformed, transcribe)
        self.assertFalse(called)

    async def test_memory_callbacks_bracket_measurement_once_before_unmeasured_drain(self):
        module = load_module(self)
        active = 0
        events = []
        after_boundary = False
        starts_after_boundary = 0
        async def transcribe(case, request_id):
            nonlocal active, starts_after_boundary
            if after_boundary:
                starts_after_boundary += 1
            active += 1
            try:
                await asyncio.sleep(0.003)
                return dict(case['reference'])
            finally:
                active -= 1
        async def before():
            events.append(('before', perf_counter(), active))
            await asyncio.sleep(0.001)
            events.append(('before_done', perf_counter(), active))
            return {'peak_reset': True}
        async def after():
            nonlocal after_boundary
            after_boundary = True
            events.append(('after', perf_counter(), active))
            await asyncio.sleep(0.001)
            return {'peak_bytes': 123}
        result = await module.run_load(cases(), transcribe, clients=3, ramp_interval_s=0,
            rounds=3, completions_per_round=4, warmup_per_client=2,
            before_measurement=before, after_measurement=after)
        self.assertEqual([event[0] for event in events], ['before', 'before_done', 'after'])
        self.assertGreaterEqual(result['rounds'][0]['start_s'], events[1][1])
        self.assertLessEqual(result['rounds'][-1]['end_s'], events[2][1])
        # Other in-flight responses may already have completed when the async
        # status callback runs. The contract is to stop refills at the boundary.
        self.assertEqual(starts_after_boundary, 0)
        self.assertEqual(result['callbacks'], {'before_measurement': {'peak_reset': True},
                                             'after_measurement': {'peak_bytes': 123}})
        self.assertEqual(active, 0)

    async def test_drain_parity_failure_invalidates_completed_measurement(self):
        module = load_module(self)
        async def transcribe(case, request_id):
            if request_id.startswith('client-00-'):
                await asyncio.sleep(0.001)
                return dict(case['reference'])
            await asyncio.sleep(0.01)
            return {'text': 'wrong', 'token_ids': [999]}
        with self.assertRaises(module.ParityError):
            await module.run_load(cases(), transcribe, clients=2, ramp_interval_s=0,
                rounds=1, completions_per_round=1, warmup_per_client=0)

    async def test_failure_during_ramp_prevents_later_clients_from_starting(self):
        module = load_module(self)
        starts = []
        async def transcribe(case, request_id):
            starts.append(request_id)
            await asyncio.sleep(0.001)
            return {'text': 'wrong', 'token_ids': [999]}
        with self.assertRaises(module.ParityError):
            await module.run_load(cases(), transcribe, clients=3, ramp_interval_s=0.05,
                rounds=1, completions_per_round=1, warmup_per_client=0)
        self.assertEqual(starts, ['client-00-000000'])

    async def test_reset_callback_failure_cleans_up_all_client_work(self):
        module = load_module(self)
        active = 0
        async def transcribe(case, request_id):
            nonlocal active
            active += 1
            try:
                await asyncio.sleep(0.002)
                return dict(case['reference'])
            finally:
                active -= 1
        async def before():
            raise RuntimeError('reset failed')
        with self.assertRaisesRegex(RuntimeError, 'reset failed'):
            await module.run_load(cases(), transcribe, clients=3, ramp_interval_s=0,
                rounds=1, completions_per_round=1, warmup_per_client=2,
                before_measurement=before)
        self.assertEqual(active, 0)

    async def test_mutating_case_reference_cannot_relax_frozen_parity(self):
        module = load_module(self)
        async def transcribe(case, request_id):
            case['reference']['token_ids'][:] = [999]
            return dict(case['reference'])
        with self.assertRaises(module.ParityError):
            await module.run_load(cases(), transcribe, clients=1, ramp_interval_s=0,
                rounds=1, completions_per_round=1, warmup_per_client=0)


class StatisticsContracts(unittest.TestCase):
    def test_percentiles_use_linear_interpolation_between_sorted_ranks(self):
        module = load_module(self)
        self.assertEqual(module.percentile([40, 10, 30, 20], 50), 25)
        self.assertAlmostEqual(module.percentile([40, 10, 30, 20], 95), 38.5)
        self.assertEqual(module.percentile([5], 99), 5)
        with self.assertRaises(ValueError):
            module.percentile([], 50)

    def test_manifest_relative_wav_and_separate_reference_hash_are_frozen_before_io(self):
        module = load_module(self)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'cases').mkdir()
            data = b'RIFF-controlled-fixture'
            wav = root / 'cases/a.wav'
            wav.write_bytes(data)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'cases': [{'case_id': 'a', 'clip_id': 'clip0',
                'wav': 'cases/a.wav', 'sha256': hashlib.sha256(data).hexdigest()}]}))
            references = root / 'references.json'
            references.write_text(json.dumps({'a': {'text': 'texto', 'token_ids': [1, 2],
                                                  'finish_reason': 'stop'}}))
            loaded, audio = module.read_inputs(manifest, references)
            self.assertEqual(loaded[0]['reference']['token_ids'], [1, 2])
            self.assertEqual(audio, {'a': data})
            wav.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash'):
                module.read_inputs(manifest, references)


class OutputContracts(unittest.TestCase):
    def test_existing_attempt_result_is_unchanged_and_no_http_work_starts(self):
        module = load_module(self)
        called = False
        async def http(args):
            nonlocal called
            called = True
            return {'schema_version': 1, 'rounds': []}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'existing.json'
            output.write_bytes(b'original attempt bytes\n')
            with patch.object(module, '_http_load', http), contextlib.redirect_stdout(io.StringIO()):
                result = module.main(['--manifest', 'unused.json', '--output', str(output)])
            self.assertEqual(result, 1)
            self.assertFalse(called)
            self.assertEqual(output.read_bytes(), b'original attempt bytes\n')

    def test_failed_new_attempt_is_reserved_before_http_and_preserved_without_raw_error(self):
        module = load_module(self)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'failed.json'
            async def http(args):
                self.assertTrue(output.is_file())
                self.assertEqual(json.loads(output.read_text())['status'], 'running')
                raise RuntimeError('private upstream error body')
            stdout = io.StringIO()
            with patch.object(module, '_http_load', http), contextlib.redirect_stdout(stdout):
                result = module.main(['--manifest', 'unused.json', '--output', str(output)])
            self.assertEqual(result, 1)
            evidence = output.read_text()
            self.assertEqual(json.loads(evidence)['status'], 'failed')
            self.assertNotIn('private upstream', evidence + stdout.getvalue())


if __name__ == '__main__':
    unittest.main()
