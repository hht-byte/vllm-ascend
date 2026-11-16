"""Isolated AR reference, C1/C20 parity, and diagnostic-only draft preflights.

Run on the trusted outer Linux host. Raw references/reports must be placed
outside tracked files and candidate-visible mounts. No benchmark is run here.
Each output is reserved before HTTP and atomically published without clobbering
an existing attempt. Diagnostic initial drafts never enter parity gate inputs.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlencode


PIN_FIELDS = ('baseline_sha', 'candidate_sha', 'tool_commit',
              'common_config_sha256', 'dataset_sha256', 'reference_sha256')


class PreflightError(RuntimeError):
    pass


def require(condition, message='Preflight evidence contract failed'):
    if not condition:
        raise PreflightError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decoded(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate preflight JSON field')
            result[key] = value
        return result
    def constant(value):
        raise PreflightError('Nonfinite preflight JSON value')
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    encoded(value)
    return value


def read_json(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 16 * 1024**2, 'Missing or oversized preflight input')
    return decoded(path.read_bytes())


class AtomicArtifact:
    """Reserve an attempt before work; publish final bytes using no-clobber link."""
    def __init__(self, path):
        self.path = Path(path)
        self.reservation = self.path.with_name(self.path.name + '.reserved')
        self.published = False
        if self.path.exists():
            raise FileExistsError('preflight output already exists')
        with self.reservation.open('xb') as stream:
            stream.write(encoded({'schema_version': 1, 'status': 'reserved'}))
            stream.flush()
            os.fsync(stream.fileno())
        if self.path.exists():
            raise FileExistsError('preflight output already exists')

    def publish(self, value):
        content = encoded(value)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', dir=self.path.parent,
                prefix=self.path.name + '.tmp-', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            # Same-directory hard link is atomic and refuses any existing name.
            os.link(temporary, self.path)
            self.published = True
            self.reservation.unlink()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def fail(self, error):
        if not self.published:
            self.publish({'schema_version': 1, 'status': 'failed', 'error_type': type(error).__name__})


def read_assets(manifest):
    manifest = Path(manifest).resolve()
    value = read_json(manifest)
    cases = value.get('cases') if isinstance(value, Mapping) else value
    require(isinstance(cases, list) and 0 < len(cases) <= 1024, 'Invalid registered case list')
    seen, audio = set(), {}
    for case in cases:
        require(isinstance(case, Mapping), 'Invalid registered case')
        case_id = case.get('case_id')
        require(isinstance(case_id, str) and 0 < len(case_id) <= 128 and case_id not in seen,
                'Invalid or duplicate registered case ID')
        path = (manifest.parent / case.get('wav', case.get('wav_path', ''))).resolve()
        require(path.is_relative_to(manifest.parent) and path.is_file(), 'WAV escaped registered assets')
        require(path.stat().st_size <= 300000, 'Registered WAV exceeds the service bound')
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == case.get('sha256'), 'Registered WAV hash mismatch')
        audio[case_id] = data
        seen.add(case_id)
    return cases, audio


def response(value):
    require(isinstance(value, Mapping) and isinstance(value.get('text'), str), 'Invalid ASR response')
    tokens = value.get('token_ids')
    require(isinstance(tokens, (list, tuple)) and len(tokens) <= 8192
            and all(type(token) is int and token >= 0 for token in tokens), 'Invalid ASR token IDs')
    reason = value.get('finish_reason')
    require(reason == 'stop', 'ASR output was truncated or did not finish normally')
    return {'text': value['text'], 'token_ids': list(tokens), 'finish_reason': reason}


async def _suite(cases, transcribe, *, prefix, concurrency, references=None, diagnostic=False,
                 wrong_first_token=False):
    require(type(concurrency) is int and concurrency >= 1 and cases, 'Invalid preflight concurrency')
    expected = {case['case_id']: response(references[case['case_id']]) for case in cases} if references is not None else None
    # Freeze expected text/tokens independently of mutable case or I/O objects.
    frozen = {key: (value['text'], tuple(value['token_ids']), value['finish_reason'])
              for key, value in (expected or {}).items()}
    repeats = max(1, math.ceil(concurrency / len(cases)))
    jobs = [dict(case) for _ in range(repeats) for case in cases]
    results = [None] * len(jobs)
    next_job = active = peak = 0
    stopped = False

    async def worker():
        nonlocal next_job, active, peak, stopped
        try:
            while not stopped and next_job < len(jobs):
                index = next_job
                next_job += 1
                case = jobs[index]
                request_id = f'{prefix}-{index:05d}'
                active += 1
                peak = max(peak, active)
                try:
                    if diagnostic:
                        draft = list(frozen[case['case_id']][1][:5])
                        require(draft, 'Diagnostic candidate must be nonempty')
                        if wrong_first_token: draft[0] = 0 if draft[0] != 0 else 1
                        raw = await transcribe(case, request_id, initial_draft=draft)
                    else:
                        raw = await transcribe(case, request_id)
                finally:
                    active -= 1
                actual = response(raw)
                if expected is not None:
                    require((actual['text'], tuple(actual['token_ids']), actual['finish_reason']) == frozen[case['case_id']],
                            'ASR text or token parity failed')
                results[index] = {'case_id': case['case_id'], 'request_id': request_id, 'response': actual}
                if diagnostic: results[index]['initial_draft'] = draft
                await asyncio.sleep(0)
        except BaseException:
            stopped = True
            raise
    tasks = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(jobs)))]
    try:
        await asyncio.gather(*tasks)
    finally:
        stopped = True
        for task in tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    require(all(item is not None for item in results) and peak == min(concurrency, len(jobs)),
            'Requested preflight concurrency was not observed')
    return {'passed': True, 'token_parity': expected is not None, 'text_parity': expected is not None,
            'initial_draft_used': diagnostic, 'requests': len(results), 'case_count': len(cases),
            'case_counts': dict(Counter(item['case_id'] for item in results)),
            'maximum_inflight': peak, 'responses': results}


async def collect_references(cases, transcribe):
    audit = await _suite(cases, transcribe, prefix='ar', concurrency=1)
    references = {item['case_id']: item['response'] for item in audit['responses']}
    return references, audit


async def verify_variant(cases, references, transcribe, *, variant):
    require(variant in ('baseline', 'candidate'), 'Invalid parity variant')
    return {f'c{count}': await _suite(cases, transcribe, prefix=f'{variant}-c{count}',
            concurrency=count, references=references) for count in (1, 20)}


async def verify_diagnostic(cases, references, transcribe):
    return {'phase': 'diagnostic', 'benchmark_eligible': False,
            'gold_prefix': await _suite(cases, transcribe, prefix='diagnostic-gold', concurrency=1,
                                        references=references, diagnostic=True),
            'wrong_first_token': await _suite(cases, transcribe, prefix='diagnostic-reject', concurrency=1,
                references=references, diagnostic=True, wrong_first_token=True)}


def checked_pins(pins):
    require(isinstance(pins, Mapping), 'Missing immutable preflight pins')
    for key in PIN_FIELDS:
        length = 40 if key in ('baseline_sha', 'candidate_sha', 'tool_commit') else 64
        require(isinstance(pins.get(key), str) and re.fullmatch('[0-9a-f]{' + str(length) + '}', pins[key]),
                'Invalid immutable preflight pin')
    return {key: pins[key] for key in PIN_FIELDS}


def assemble_gate(ar, baseline, candidate, pins):
    pins = checked_pins(pins)
    require(ar.get('schema_version') == 1 and ar.get('status') == 'passed'
            and ar.get('phase') == 'ar' and ar.get('passed') is True
            and ar.get('dataset_sha256') == pins['dataset_sha256']
            and ar.get('reference_sha256') == pins['reference_sha256'], 'AR reference proof mismatch')
    count = ar.get('case_count')
    require(type(count) is int and count > 0, 'Missing AR case coverage')
    preflights = {'ar': {'passed': True, 'case_count': count}}
    for name, report in (('baseline', baseline), ('candidate', candidate)):
        require(report.get('schema_version') == 1 and report.get('status') == 'passed'
                and report.get('phase') == 'parity' and report.get('variant') == name
                and report.get('pins') == pins, 'Variant parity proof mismatch')
        checkpoint = report.get('checkpoint_sha256')
        require(isinstance(checkpoint, str) and re.fullmatch('[0-9a-f]{64}', checkpoint),
                'Missing parity checkpoint pin')
        require(checkpoint == baseline.get('checkpoint_sha256'), 'Parity checkpoint pins differ')
        expected_identity = report.get('expected_identity')
        require(isinstance(expected_identity, Mapping), 'Missing expected parity engine identity')
        before = check_status(report.get('status_before'), mode='mtp', pins=pins, variant=name,
            expected_identity=expected_identity, checkpoint_sha256=checkpoint)
        after = check_status(report.get('status_after'), mode='mtp', pins=pins, variant=name,
            expected_identity=expected_identity, checkpoint_sha256=checkpoint)
        require(after['mtp']['head_batches'] > before['mtp']['head_batches'], 'Parity executed no MTP head work')
        summaries = {}
        for concurrency in (1, 20):
            suite = report.get(f'c{concurrency}', {})
            require(suite.get('token_parity') is True and suite.get('text_parity') is True
                    and suite.get('initial_draft_used') is False and suite.get('case_count') == count
                    and suite.get('maximum_inflight') == concurrency
                    and type(suite.get('requests')) is int and suite['requests'] >= max(count, concurrency),
                    'Incomplete C1/C20 parity proof')
            summaries[f'c{concurrency}'] = {key: value for key, value in suite.items() if key != 'responses'}
        preflights[name] = summaries
    return {'schema_version': 1, 'status': 'passed', **pins,
            'checkpoint_sha256': baseline['checkpoint_sha256'], 'preflights': preflights}


@contextmanager
def socket_address(path):
    # Keep supplied proc-fd addresses intact; resolving expands long host paths.
    if sys.platform != 'linux' or str(path).startswith('/proc/self/fd/'):
        yield str(path)
        return
    path = Path(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield f'/proc/self/fd/{descriptor}/{path.name}'
    finally:
        os.close(descriptor)


class UnixHTTPClient:
    def __init__(self, socket_path, *, timeout_s=300, max_response_bytes=1024**2):
        require(type(timeout_s) in (int, float) and math.isfinite(timeout_s) and timeout_s > 0,
                'Invalid preflight timeout')
        self.socket_path, self.timeout_s, self.maximum = socket_path, timeout_s, max_response_bytes

    async def json(self, method, path, body=b'', headers=None):
        async def request():
            writer = None
            try:
                with socket_address(self.socket_path) as address:
                    reader, writer = await asyncio.open_unix_connection(address, limit=65536)
                supplied = {'Host': 'localhost', 'Connection': 'close', 'Content-Length': str(len(body)),
                            **(headers or {})}
                data = f'{method} {path} HTTP/1.1\r\n' + ''.join(f'{key}: {value}\r\n' for key, value in supplied.items())
                writer.write(data.encode('ascii') + b'\r\n' + body)
                await writer.drain()
                status = await reader.readline()
                require(status.startswith(b'HTTP/1.1 200 ') or status.startswith(b'HTTP/1.0 200 '),
                        'Preflight HTTP request failed')
                received, size = {}, len(status)
                while True:
                    line = await reader.readline()
                    size += len(line)
                    require(line and size <= 32768, 'Preflight HTTP headers exceeded their bound')
                    if line == b'\r\n': break
                    key, value = line.decode('ascii').split(':', 1)
                    key = key.lower()
                    require(key not in received, 'Duplicate preflight HTTP header')
                    received[key] = value.strip()
                require('transfer-encoding' not in received, 'Unsupported preflight HTTP transfer encoding')
                length = int(received.get('content-length', '-1'))
                require(0 <= length <= self.maximum, 'Preflight HTTP response exceeded its bound')
                return decoded(await reader.readexactly(length))
            finally:
                if writer is not None:
                    writer.close()
                    try: await writer.wait_closed()
                    except (ConnectionError, OSError): pass
        try:
            return await asyncio.wait_for(request(), self.timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception:
            raise PreflightError('Bounded preflight HTTP request failed') from None


def check_status(status, *, mode, pins=None, variant=None, expected_identity=None, checkpoint_sha256=None):
    require(isinstance(status, Mapping) and status.get('cleanup_uncertain') is False
            and status.get('quarantined') is False, 'Service cleanup state is unconfirmed')
    require(status.get('hardware_class') == '910B3', 'Unexpected preflight hardware class')
    mtp = status.get('mtp', {})
    require(mtp.get('loaded') is (mode != 'ar'), 'Preflight engine mode mismatch')
    if mode != 'ar':
        require(mtp.get('heads') == 5 and type(mtp.get('head_batches')) is int
                and mtp['head_batches'] >= 0, 'Missing loaded MTP heads or work counters')
        if checkpoint_sha256 is not None:
            require(re.fullmatch('[0-9a-f]{64}', checkpoint_sha256)
                    and mtp.get('weights_sha256') == checkpoint_sha256, 'Loaded MTP checkpoint mismatch')
    identity = status.get('identity')
    require(isinstance(identity, Mapping), 'Missing preflight engine identity')
    if expected_identity is not None:
        require(identity == expected_identity, 'Preflight engine identity mismatch')
    if variant is not None:
        require(expected_identity is not None, 'Parity requires a frozen exact engine identity')
        require(identity.get('variant_sha') == pins[variant + '_sha'], 'Preflight variant SHA mismatch')
        for key in ('tool_commit', 'common_config_sha256', 'dataset_sha256'):
            require(identity.get(key) == pins[key], 'Live preflight shared configuration pin mismatch')
        require(isinstance(identity.get('overlay_sha256'), str)
                and re.fullmatch('[0-9a-f]{64}', identity['overlay_sha256']), 'Missing exact overlay identity')
    return status


async def run_service(args):
    cases, audio = read_assets(args.manifest)
    dataset_sha = digest(args.manifest)
    client = UnixHTTPClient(args.socket, timeout_s=args.timeout_s)
    pins = checked_pins(read_json(args.pins)) if args.pins else None
    expected_identity = read_json(args.expected_identity) if args.expected_identity else None
    mode = 'ar' if args.command == 'ar' else 'mtp'
    variant = args.variant if args.command == 'parity' else None
    if pins is not None: require(pins['dataset_sha256'] == dataset_sha, 'Preflight dataset pin mismatch')
    references = None
    reference_sha = None
    if args.command != 'ar':
        references = read_json(args.references)
        require(isinstance(references, Mapping), 'Invalid frozen AR references')
        reference_sha = digest(args.references)
        if pins is not None: require(reference_sha == pins['reference_sha256'], 'Preflight reference pin mismatch')

    async def transcribe(case, request_id, *, initial_draft=None):
        headers = {'Content-Type': 'audio/wav'}
        if initial_draft is not None:
            headers['x-initial-draft'] = json.dumps(initial_draft, separators=(',', ':'))
        return await client.json('POST', '/transcribe?' + urlencode({'case_id': case['case_id'],
            'request_id': request_id}), audio[case['case_id']], headers)

    before = check_status(await client.json('GET', '/status'), mode=mode, pins=pins,
        variant=variant, expected_identity=expected_identity, checkpoint_sha256=args.checkpoint_sha256)
    report = {'schema_version': 1, 'status': 'passed', 'phase': args.command,
              'dataset_sha256': dataset_sha, 'case_count': len(cases), 'status_before': before}
    if args.command == 'ar':
        references, audit = await collect_references(cases, transcribe)
        reference_sha = hashlib.sha256(encoded(references)).hexdigest()
        report.update(passed=True, audit=audit)
    elif args.command == 'parity':
        require(pins is not None, 'Parity requires immutable pins')
        report.update(variant=args.variant, pins=pins,
                      expected_identity=expected_identity, checkpoint_sha256=args.checkpoint_sha256,
                      **await verify_variant(cases, references, transcribe, variant=args.variant))
    else:
        report.update(await verify_diagnostic(cases, references, transcribe))
    report['status_after'] = check_status(await client.json('GET', '/status'), mode=mode, pins=pins,
        variant=variant, expected_identity=expected_identity, checkpoint_sha256=args.checkpoint_sha256)
    if args.command == 'parity':
        delta = report['status_after']['mtp']['head_batches'] - before['mtp']['head_batches']
        require(delta > 0, 'Parity executed no MTP head work')
        report['mtp_head_batches_delta'] = delta
    require(digest(args.manifest) == dataset_sha, 'Preflight manifest changed during requests')
    if args.command != 'ar':
        require(digest(args.references) == reference_sha, 'Frozen references changed during requests')
    report['reference_sha256'] = reference_sha
    return report, references


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('ar', 'parity', 'diagnostic'):
        command = commands.add_parser(name)
        for field in ('manifest', 'socket', 'references', 'output'):
            command.add_argument('--' + field, required=True)
        command.add_argument('--pins', required=name == 'parity')
        command.add_argument('--expected-identity', required=name == 'parity')
        command.add_argument('--checkpoint-sha256', required=name == 'parity')
        command.add_argument('--timeout-s', type=float, default=300)
        if name == 'parity': command.add_argument('--variant', choices=('baseline', 'candidate'), required=True)
    gate = commands.add_parser('gate')
    for field in ('ar-report', 'baseline-report', 'candidate-report', 'pins', 'output'):
        gate.add_argument('--' + field, required=True)
    args = parser.parse_args(argv)
    artifacts = []
    try:
        report_artifact = AtomicArtifact(args.output)
        artifacts.append(report_artifact)
        reference_artifact = None
        if args.command == 'ar':
            require(Path(args.references).absolute() != Path(args.output).absolute(), 'Outputs must be distinct')
            reference_artifact = AtomicArtifact(args.references)
            artifacts.append(reference_artifact)
        if args.command == 'gate':
            reports = [read_json(getattr(args, name + '_report')) for name in ('ar', 'baseline', 'candidate')]
            result = assemble_gate(*reports, read_json(args.pins))
            result['report_sha256'] = {name: digest(getattr(args, name + '_report'))
                                      for name in ('ar', 'baseline', 'candidate')}
        else:
            result, references = asyncio.run(run_service(args))
            if reference_artifact is not None: reference_artifact.publish(references)
        report_artifact.publish(result)
    except Exception as error:
        for artifact in artifacts:
            try: artifact.fail(error)
            except (OSError, ValueError): pass
        print(f'Preflight failed ({type(error).__name__}); no parity gate accepted.')
        return 1
    print('Preflight completed; local evidence published without overwriting prior attempts.')
    return 0


if __name__ == '__main__': raise SystemExit(main())
