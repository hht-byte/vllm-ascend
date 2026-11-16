"""Parity-gated closed-loop ASR load with separate ramp, warmup, and drain.

``transcribe(case, request_id)`` is an async callable returning a mapping with
``text`` and ``token_ids``. Cases contain ``case_id``, ``clip_id``, and frozen
``reference: {text, token_ids}`` (flat reference_* fields are also accepted).
Clients cycle all cases, starting at client_index modulo the clip-group count.

Every response, including warmup and drain, must exactly match its frozen AR
reference. Measurement windows are contiguous groups of N validated completions;
request latency includes the whole request even when it began before a boundary.
Percentiles linearly interpolate sorted samples at rank (n - 1) * p / 100.
Raw returned samples belong outside tracked files. CPU tests do not establish
device performance. The optional aiohttp dependency is imported only by the CLI.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter


class ParityError(RuntimeError):
    """A response failed the frozen text/token contract; raw content is omitted."""
    def __init__(self, case_id, request_id, reason):
        self.case_id = case_id
        self.request_id = request_id
        self.reason = reason
        super().__init__('formal response does not match its frozen AR reference')


class LoadTransportError(RuntimeError):
    pass


def percentile(values, p):
    """Return linear interpolation at zero-based sorted rank (n - 1) * p / 100."""
    if not values or not 0 <= p <= 100:
        raise ValueError('percentile requires samples and a percentage from 0 to 100')
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    lower, upper = math.floor(rank), math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _cases(cases):
    records = []
    seen = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise TypeError('each load case must be a mapping')
        case_id = case.get('case_id')
        clip_id = case.get('clip_id', case.get('source_id'))
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError('load case IDs must be unique nonempty strings')
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError('each load case requires a clip_id or source_id')
        reference = case.get('reference')
        if reference is None:
            reference = {'text': case.get('reference_text'),
                         'token_ids': case.get('reference_token_ids')}
        if not isinstance(reference, Mapping) or not isinstance(reference.get('text'), str):
            raise ValueError('each case requires frozen AR reference text')
        tokens = reference.get('token_ids')
        if not isinstance(tokens, (list, tuple)) or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError('each case requires frozen nonnegative AR token IDs')
        # Expected bytes/tokens are separate from the mutable case passed to I/O.
        records.append((dict(case), case_id, clip_id, reference['text'], tuple(tokens)))
        seen.add(case_id)
    if not records:
        raise ValueError('load cases must not be empty')
    groups = {}
    for record in records:
        groups.setdefault(record[2], []).append(record)
    ordered = [record for group in groups.values() for record in group]
    offsets = []
    offset = 0
    for group in groups.values():
        offsets.append(offset)
        offset += len(group)
    return ordered, offsets


def _check_response(response, record, request_id):
    if not isinstance(response, Mapping):
        raise ParityError(record[1], request_id, 'invalid_response')
    tokens = response.get('token_ids')
    if (not isinstance(tokens, (list, tuple))
            or any(type(token) is not int or token < 0 for token in tokens)):
        raise ParityError(record[1], request_id, 'invalid_token_ids')
    if tuple(tokens) != record[4]:
        raise ParityError(record[1], request_id, 'token_ids')
    if response.get('text') != record[3]:
        raise ParityError(record[1], request_id, 'text')
    return len(tokens)


def _window(index, start, end, samples):
    duration = end - start
    if duration <= 0:
        raise RuntimeError('load measurement clock did not advance')
    latencies = [sample['latency_ms'] for sample in samples]
    tokens = sum(sample['output_tokens'] for sample in samples)
    return {'round': index, 'start_s': start, 'end_s': end, 'duration_s': duration,
            'completions': len(samples), 'output_tokens': tokens,
            'case_completions': dict(Counter(sample['case_id'] for sample in samples)),
            'requests_per_second': len(samples) / duration,
            'output_tokens_per_second': tokens / duration,
            'latency_ms': {'min': min(latencies), 'mean': sum(latencies) / len(latencies),
                           'p50': percentile(latencies, 50), 'p95': percentile(latencies, 95),
                           'p99': percentile(latencies, 99), 'max': max(latencies)}}


async def run_load(cases, transcribe, *, clients=20, ramp_interval_s=1, rounds=3,
                   completions_per_round=60, warmup_per_client=2, clock=perf_counter,
                   before_measurement=None, after_measurement=None):
    """Run one request at a time per client, then drain without further refill.

Warmup requires each client to complete ``warmup_per_client`` requests that
*started* after the last client was spawned. A parity/transport failure stops
refilling immediately, cancels other in-flight requests, and awaits cleanup.
No partial result is returned on failure, including failure during drain.
    Async callbacks reset counters after warmup and snapshot the final boundary
    before drain; their returned metadata is included separately from timings.
"""
    for name, value in (('clients', clients), ('rounds', rounds),
                        ('completions_per_round', completions_per_round)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if type(warmup_per_client) is not int or warmup_per_client < 0:
        raise ValueError('warmup_per_client must be a nonnegative integer')
    if (type(ramp_interval_s) not in (int, float) or not math.isfinite(ramp_interval_s)
            or ramp_interval_s < 0):
        raise ValueError('ramp_interval_s must be a finite nonnegative number')
    ordered, offsets = _cases(cases)
    samples, windows, current = [], [], []
    warmup = [0] * clients
    ramp_start = clock()
    ramp_end = None
    window_start = None
    measurement_start = None
    measurement_starting = False
    callbacks = {}
    inflight = peak = 0
    stop_refill = False
    failure = None
    wake = asyncio.Event()
    tasks = []
    spawn_times = []

    async def begin_measurement():
        nonlocal measurement_starting, measurement_start, window_start
        measurement_starting = True
        if before_measurement is not None:
            callbacks['before_measurement'] = await before_measurement()
        measurement_start = window_start = clock()

    async def client_loop(client):
        nonlocal inflight, peak, stop_refill, failure, window_start, measurement_start
        nonlocal measurement_starting
        sequence = 0
        position = offsets[client % len(offsets)]
        try:
            while not stop_refill:
                record = ordered[position % len(ordered)]
                request_id = f'client-{client:02d}-{sequence:06d}'
                start = clock()
                inflight += 1
                peak = max(peak, inflight)
                try:
                    response = await transcribe(record[0], request_id)
                    end = clock()
                finally:
                    inflight -= 1
                output_tokens = _check_response(response, record, request_id)
                sample = {'request_id': request_id, 'client': client, 'case_id': record[1],
                          'clip_id': record[2], 'start_s': start, 'end_s': end,
                          'latency_ms': (end - start) * 1000, 'output_tokens': output_tokens,
                          'phase': 'ramp', 'round': None}
                call_before = call_after = False
                if end < start:
                    raise RuntimeError('load measurement clock moved backwards')
                if stop_refill:
                    sample['phase'] = 'drain'
                elif ramp_end is not None and measurement_start is None:
                    sample['phase'] = 'warmup'
                    if start >= ramp_end:
                        warmup[client] += 1
                    if not measurement_starting and all(count >= warmup_per_client for count in warmup):
                        # Claim the boundary before awaiting reset I/O; other
                        # clients continue warmup until reset has completed.
                        measurement_starting = True
                        call_before = True
                elif measurement_start is not None:
                    sample['phase'] = 'measured'
                    sample['round'] = len(windows) + 1
                    current.append(sample)
                    if len(current) == completions_per_round:
                        windows.append(_window(len(windows) + 1, window_start, end, current))
                        current.clear()
                        window_start = end
                        if len(windows) == rounds:
                            stop_refill = True
                            wake.set()
                            call_after = True
                samples.append(sample)
                if call_before:
                    await begin_measurement()
                if call_after and after_measurement is not None:
                    callbacks['after_measurement'] = await after_measurement()
                position += 1
                sequence += 1
                # Preserve fairness even when a supplied async callable returns
                # immediately rather than yielding to I/O.
                if not stop_refill:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if failure is None:
                failure = error
            stop_refill = True
            wake.set()
            raise

    try:
        for client in range(clients):
            if client and ramp_interval_s:
                remaining = ramp_start + client * ramp_interval_s - clock()
                try:
                    await asyncio.wait_for(wake.wait(), timeout=max(0, remaining))
                except TimeoutError:
                    pass
            if failure is not None:
                raise failure
            spawn_times.append(clock())
            tasks.append(asyncio.create_task(client_loop(client)))
        ramp_end = clock()
        if warmup_per_client == 0:
            await begin_measurement()
        await asyncio.gather(*tasks)
        if failure is not None:
            raise failure
        drained = [sample for sample in samples if sample['phase'] == 'drain']
        measured = [sample for sample in samples if sample['phase'] == 'measured']
        aggregate = _window(0, windows[0]['start_s'], windows[-1]['end_s'], measured)
        aggregate.pop('round')
        aggregate['window_count'] = len(windows)
        return {'schema_version': 1, 'parity': 'exact_text_and_token_ids',
                'parameters': {'clients': clients, 'ramp_interval_s': ramp_interval_s,
                    'rounds': rounds, 'completions_per_round': completions_per_round,
                    'warmup_per_client': warmup_per_client, 'case_count': len(ordered)},
                'percentile_definition': 'linear interpolation at sorted rank (n - 1) * p / 100',
                'ramp': {'start_s': ramp_start, 'end_s': ramp_end, 'client_spawn_s': spawn_times},
                'warmup': {'completed_per_client': warmup, 'end_s': measurement_start},
                'rounds': windows, 'aggregate': aggregate, 'callbacks': callbacks, 'maximum_inflight': peak,
                'drain': {'completions': len(drained), 'end_s': clock()}, 'samples': samples}
    finally:
        stop_refill = True
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def read_inputs(manifest_path, reference_path=None):
    """Read frozen references and WAV bytes before any measured HTTP work."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    cases = manifest['cases'] if isinstance(manifest, Mapping) else manifest
    if reference_path:
        references = json.loads(Path(reference_path).read_text(encoding='utf-8'))
        references = references.get('cases', references) if isinstance(references, Mapping) else references
        if isinstance(references, list):
            references = {item['case_id']: item.get('reference', item) for item in references}
        cases = [{**case, 'reference': references[case['case_id']]} for case in cases]
    _cases(cases)
    audio = {}
    for case in cases:
        path = Path(case.get('wav_path', case.get('wav', '')))
        if not path.is_absolute():
            path = manifest_path.parent / path
        data = path.read_bytes()
        if case.get('sha256') and hashlib.sha256(data).hexdigest() != case['sha256']:
            raise ValueError('load WAV content differs from its frozen manifest hash')
        audio[case['case_id']] = data
    return cases, audio


async def _http_load(args):
    import aiohttp

    manifest_path = args.manifest or str(Path(args.assets_dir) / 'manifest.json')
    cases, audio = read_inputs(manifest_path, args.references)
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    connector = (aiohttp.UnixConnector(path=args.socket, limit=args.clients)
                 if args.socket else aiohttp.TCPConnector(limit=args.clients, limit_per_host=args.clients))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def control(method, endpoint):
            try:
                async with session.request(method, args.url.rstrip('/') + endpoint) as response:
                    if response.status != 200:
                        raise LoadTransportError('load control HTTP request failed')
                    return await response.json()
            except asyncio.CancelledError:
                raise
            except Exception:
                raise LoadTransportError('load control HTTP request failed') from None

        async def transcribe(case, request_id):
            try:
                async with session.post(args.url.rstrip('/') + '/transcribe',
                    params={'case_id': case['case_id'], 'request_id': request_id},
                    data=audio[case['case_id']], headers={'Content-Type': 'audio/wav'}) as response:
                    if response.status != 200:
                        raise LoadTransportError('transcription HTTP request failed')
                    return await response.json()
            except asyncio.CancelledError:
                raise
            except Exception:
                raise LoadTransportError('transcription HTTP request failed') from None

        before = await control('GET', '/status')
        result = await run_load(cases, transcribe, clients=args.clients,
            ramp_interval_s=args.ramp_interval_s, rounds=args.rounds,
            completions_per_round=args.completions_per_round, warmup_per_client=args.warmup_per_client,
            before_measurement=lambda: control('POST', '/reset'),
            after_measurement=lambda: control('GET', '/status'))
        result['server_before'] = before
        result['server_after'] = await control('GET', '/status')
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--manifest')
    inputs.add_argument('--assets-dir', help='directory containing manifest.json and prepared WAVs')
    parser.add_argument('--reference', '--references', dest='references')
    parser.add_argument('--socket', help='Unix-domain socket path; uses aiohttp.UnixConnector')
    parser.add_argument('--url', default='http://127.0.0.1:18080')
    parser.add_argument('--output', required=True, help='raw result path outside tracked files')
    parser.add_argument('--clients', type=int, default=20)
    parser.add_argument('--ramp-interval-s', type=float, default=1)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--completions-per-round', type=int, default=60)
    parser.add_argument('--warmup-per-client', type=int, default=2)
    parser.add_argument('--timeout-s', type=float, default=300)
    args = parser.parse_args(argv)
    try:
        # Reserve this attempt before reading inputs or starting HTTP. Exclusive
        # creation also closes the race between a preflight check and a write.
        output = Path(args.output).open('x', encoding='utf-8')
    except Exception as error:
        print(f'Load output is unavailable ({type(error).__name__}); no requests started.')
        return 1
    def store(value):
        content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        output.seek(0)
        output.write(content)
        output.truncate()
        output.flush()
    with output:
        try:
            store({'schema_version': 1, 'status': 'running'})
            result = asyncio.run(_http_load(args))
            store(result)
        except Exception as error:
            # Preserve the owned failed attempt without exposing upstream error
            # bodies, request contents, or private configuration.
            try:
                store({'schema_version': 1, 'status': 'failed', 'error_type': type(error).__name__})
            except OSError:
                pass
            print(f'Load run failed ({type(error).__name__}); no timing result accepted.')
            return 1
    print(f'Completed {len(result["rounds"])} windows with exact text/token parity.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
