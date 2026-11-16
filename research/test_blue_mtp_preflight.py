"""CPU preflight contracts; no device or external service is used."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


def module(test):
    path = Path(__file__).with_name('blue_mtp_preflight.py')
    test.assertTrue(path.exists(), 'isolated preflight client is missing')
    spec = importlib.util.spec_from_file_location('isolated_preflight', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def cases():
    return [{'case_id': f'case{i}', 'clip_id': f'clip{i // 3}'} for i in range(12)]


def response(case):
    index = int(case['case_id'][4:])
    return {'text': f'texto {index}', 'token_ids': [index, 100], 'finish_reason': 'stop'}


class PreflightContracts(unittest.IsolatedAsyncioTestCase):
    async def test_ar_freeze_and_c1_c20_cover_every_case_without_draft_input(self):
        m = module(self)
        active = peak = 0
        diagnostic_calls = []
        async def transcribe(case, request_id, *, initial_draft=None):
            nonlocal active, peak
            diagnostic_calls.append(initial_draft)
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.001)
                return response(case)
            finally:
                active -= 1
        references, audit = await m.collect_references(cases(), transcribe)
        self.assertEqual(set(references), {case['case_id'] for case in cases()})
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['maximum_inflight'], 1)
        result = await m.verify_variant(cases(), references, transcribe, variant='baseline')
        self.assertEqual(result['c1']['requests'], 12)
        self.assertEqual(result['c1']['maximum_inflight'], 1)
        self.assertEqual(result['c20']['requests'], 24)
        self.assertEqual(result['c20']['maximum_inflight'], 20)
        self.assertTrue(result['c1']['token_parity'])
        self.assertTrue(result['c20']['text_parity'])
        self.assertEqual(set(result['c20']['case_counts']), set(references))
        self.assertEqual(active, 0)
        self.assertEqual(peak, 20)
        self.assertTrue(all(draft is None for draft in diagnostic_calls))

    async def test_mismatch_cancels_outstanding_work_and_diagnostic_is_separate(self):
        m = module(self)
        references = {case['case_id']: response(case) for case in cases()}
        active = 0
        async def transcribe(case, request_id, *, initial_draft=None):
            nonlocal active
            active += 1
            try:
                await asyncio.sleep(0.001)
                value = response(case)
                if case['case_id'] == 'case0': value['text'] = 'wrong'
                return value
            finally:
                active -= 1
        with self.assertRaises(m.PreflightError):
            await m.verify_variant(cases(), references, transcribe, variant='candidate')
        self.assertEqual(active, 0)
        offered = []
        async def diagnostic(case, request_id, *, initial_draft=None):
            offered.append(initial_draft)
            await asyncio.sleep(0.001)
            return response(case)
        result = await m.verify_diagnostic(cases(), references, diagnostic)
        self.assertEqual(result['phase'], 'diagnostic')
        self.assertFalse(result['benchmark_eligible'])
        self.assertEqual(offered[0], [0, 100])
        self.assertEqual(offered[12], [1, 100])
        self.assertEqual(result['wrong_first_token']['requests'], 12)

    async def test_truncated_or_unfinished_responses_cannot_become_references_or_parity(self):
        m = module(self)
        for reason in ('length', None):
            async def transcribe(case, request_id):
                await asyncio.sleep(0.001)
                return {**response(case), 'finish_reason': reason}
            with self.subTest(reason=reason), self.assertRaises(m.PreflightError):
                await m.collect_references(cases(), transcribe)


class ArtifactGateContracts(unittest.TestCase):
    def test_atomic_publication_never_overwrites_prior_attempt_or_racing_result(self):
        m = module(self)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            artifact = m.AtomicArtifact(path)
            self.assertFalse(path.exists())
            with self.assertRaises(FileExistsError): m.AtomicArtifact(path)
            artifact.publish({'status': 'passed'})
            self.assertEqual(json.loads(path.read_text()), {'status': 'passed'})
            with self.assertRaises(FileExistsError): m.AtomicArtifact(path)
            race = Path(directory) / 'race.json'
            reserved = m.AtomicArtifact(race)
            race.write_bytes(b'other attempt')
            with self.assertRaises(FileExistsError): reserved.publish({'wrong': True})
            self.assertEqual(race.read_bytes(), b'other attempt')

    def test_gate_requires_all_real_phases_and_exact_immutable_pins(self):
        m = module(self)
        pins = {key: ('a' * 40 if key.endswith('_sha') or key == 'tool_commit' else 'b' * 64)
                for key in m.PIN_FIELDS}
        ar = {'schema_version': 1, 'status': 'passed', 'phase': 'ar', 'passed': True,
              'dataset_sha256': pins['dataset_sha256'], 'reference_sha256': pins['reference_sha256'],
              'case_count': 12}
        suite = {'token_parity': True, 'text_parity': True, 'initial_draft_used': False,
                 'requests': 24, 'case_count': 12, 'maximum_inflight': 20}
        def identity(variant):
            return {'variant_sha': pins[variant + '_sha'], 'overlay_sha256': 'd' * 64,
                    **{key: pins[key] for key in ('tool_commit', 'common_config_sha256', 'dataset_sha256')}}
        def status(variant):
            return {'cleanup_uncertain': False, 'quarantined': False, 'hardware_class': '910B3',
                    'identity': identity(variant),
                    'mtp': {'loaded': True, 'heads': 5, 'weights_sha256': 'f' * 64, 'head_batches': 1}}
        baseline = {'schema_version': 1, 'status': 'passed', 'phase': 'parity',
                    'variant': 'baseline', 'pins': pins, 'c1': {**suite, 'maximum_inflight': 1}, 'c20': suite,
                    'expected_identity': identity('baseline'), 'status_before': status('baseline'),
                    'status_after': {**status('baseline'), 'mtp': {**status('baseline')['mtp'], 'head_batches': 2}},
                    'checkpoint_sha256': 'f' * 64}
        candidate = {**baseline, 'variant': 'candidate', 'expected_identity': identity('candidate'),
                     'status_before': status('candidate'),
                     'status_after': {**status('candidate'), 'mtp': {**status('candidate')['mtp'], 'head_batches': 2}}}
        gate = m.assemble_gate(ar, baseline, candidate, pins)
        self.assertTrue(gate['preflights']['ar']['passed'])
        self.assertTrue(gate['preflights']['candidate']['c20']['token_parity'])
        for change in ('reference', 'diagnostic', 'c20', 'live_config', 'weights', 'no_mtp_work'):
            bad = {**candidate}
            if change == 'reference': bad['pins'] = {**pins, 'reference_sha256': 'c' * 64}
            elif change == 'diagnostic': bad['phase'] = 'diagnostic'
            elif change == 'c20': bad['c20'] = {**suite, 'maximum_inflight': 1}
            elif change == 'live_config':
                bad['status_before'] = {**status('candidate'),
                    'identity': {**identity('candidate'), 'common_config_sha256': 'e' * 64}}
            elif change == 'weights':
                bad['status_after'] = {**status('candidate'),
                    'mtp': {**status('candidate')['mtp'], 'weights_sha256': 'e' * 64, 'head_batches': 2}}
            else: bad['status_after'] = status('candidate')
            with self.subTest(change=change), self.assertRaises(m.PreflightError):
                m.assemble_gate(ar, baseline, bad, pins)


@unittest.skipUnless(sys.platform == 'linux', 'Unix HTTP transport requires the Linux experiment host')
class UnixHTTPContracts(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_unix_http_parses_json_and_rejects_error_bodies(self):
        m = module(self)
        async def server(reader, writer):
            try:
                headers = await reader.readuntil(b'\r\n\r\n')
                path = headers.split(b' ')[1]
                if path == b'/ok':
                    body = b'{"text":"ok","token_ids":[1]}'
                    status = b'200 OK'
                else:
                    body = b'private external error body'
                    status = b'500 Internal Server Error'
                writer.write(b'HTTP/1.1 ' + status + b'\r\nContent-Length: ' + str(len(body)).encode()
                             + b'\r\nConnection: close\r\n\r\n' + body)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
        with tempfile.TemporaryDirectory() as directory:
            socket = str(Path(directory) / 'service.sock')
            listener = await asyncio.start_unix_server(server, socket)
            try:
                client = m.UnixHTTPClient(socket, timeout_s=1, max_response_bytes=1024)
                self.assertEqual((await client.json('GET', '/ok'))['token_ids'], [1])
                with self.assertRaises(m.PreflightError) as caught:
                    await client.json('GET', '/error')
                self.assertNotIn('private', str(caught.exception))
                tiny = m.UnixHTTPClient(socket, timeout_s=1, max_response_bytes=2)
                with self.assertRaises(m.PreflightError): await tiny.json('GET', '/ok')
            finally:
                listener.close()
                await listener.wait_closed()


if __name__ == '__main__': unittest.main()
