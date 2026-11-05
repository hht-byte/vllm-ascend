"""Prespec request metadata and cancellation with optional worker MTP."""
import asyncio
from copy import deepcopy
from enum import Enum
from types import SimpleNamespace
import unittest

from streaming_prespec_helpers import PreSpecModelHelpers
from streaming_prespec_helpers import ModelShadowRequest
from streaming_prespec import AdmissionPolicy, PreSpecCoordinator


class OutputKind(Enum):
    CUMULATIVE = 0
    DELTA = 1


class Model(PreSpecModelHelpers):
    def __init__(self):
        self.prespec = SimpleNamespace(policy=SimpleNamespace(max_rollback_draft_tokens=5))

    def _pack_stream_request(self, *, sampling_params, draft_token_ids, **kwargs):
        sampling_params.extra_args = {**(getattr(sampling_params, 'extra_args', None) or {}),
                                      'ascend_external_draft': list(draft_token_ids)}
        return {'prompt': kwargs, 'sampling_params': sampling_params}


class ShadowMTPArgsTests(unittest.TestCase):
    def request(self, tokens=(20, 21)):
        return SimpleNamespace(rollback_tokens=tokens, sampling_params=SimpleNamespace(
            max_tokens=128, min_tokens=3, n=1, best_of=None, output_kind=OutputKind.DELTA,
            extra_args={'other': 123, 'ascend_mtp': False,
                        'ascend_external_draft_first_verify': True,
                        'ascend_external_draft_rejoin': 'ngram_1',
                        'ascend_external_draft_ngram_after_first_mismatch': True}),
            audio_pcm_f32le=b'audio', prefix_text='', stable_prefix_tokens=(),
            model_context={}, request_id='prespec-test')

    def test_prespec_opt_in_preserves_legacy_rollback_and_clears_formal_markers(self):
        model, request = Model(), self.request()
        original = deepcopy(request.sampling_params.__dict__)
        params = model._build_stream_request(request, 5)['sampling_params']
        self.assertTrue(params.extra_args['ascend_mtp'])
        self.assertEqual(params.extra_args['ascend_external_draft'], [20, 21])
        self.assertEqual(params.extra_args['other'], 123)
        for key in ('ascend_external_draft_first_verify', 'ascend_external_draft_rejoin',
                    'ascend_external_draft_ngram_after_first_mismatch'):
            self.assertNotIn(key, params.extra_args)
        self.assertEqual(params.max_tokens, 5)
        self.assertEqual(params.min_tokens, 0)
        self.assertEqual(params.output_kind, OutputKind.CUMULATIVE)
        self.assertEqual(request.sampling_params.__dict__, original)

    def test_prespec_can_opt_out_without_inheriting_formal_mtp(self):
        model, request = Model(), self.request()
        model.prespec_mtp_enabled = False
        request.sampling_params.extra_args['ascend_mtp'] = True
        params = model._build_stream_request(request, 5)['sampling_params']
        self.assertIs(params.extra_args['ascend_mtp'], False)
        self.assertNotIn('ascend_external_draft_first_verify', params.extra_args)

    def test_draftless_prespec_and_boolean_validation(self):
        model = Model()
        params = model._build_stream_request(self.request(()), 5)['sampling_params']
        self.assertTrue(params.extra_args['ascend_mtp'])
        self.assertEqual(params.extra_args['ascend_external_draft'], [])
        model.prespec_mtp_enabled = 'false'
        with self.assertRaises(TypeError):
            model._build_stream_request(self.request(), 5)


class WaitingEngine:
    def __init__(self):
        self.calls, self.aborted, self.closed = [], [], []
        self.published = asyncio.Event()

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        try:
            yield SimpleNamespace(finished=False, outputs=[SimpleNamespace(
                token_ids=[7, 20, 99, 30], finish_reason=None)])
            self.published.set()
            await asyncio.Future()
        finally:
            self.closed.append(kwargs['request_id'])

    async def abort(self, req_id):
        self.aborted.append(req_id)


class ShadowMTPCancelTests(unittest.IsolatedAsyncioTestCase):
    async def check_partial(self, timeout):
        model, engine = Model(), WaitingEngine()
        model.prespec_engine = engine
        model.prespec = PreSpecCoordinator(model._run_prespec, AdmissionPolicy(
            admission_mode='force', midpoint_samples=16000,
            max_rollback_draft_tokens=5, max_candidate_tokens=5), shadow_only=False)
        settings = ShadowMTPArgsTests().request().sampling_params
        request = ModelShadowRequest(
            session_id='ws', audio_epoch=1, window_id=1, base_prefix_version=2,
            visible_audio_end_samples=48000, window_audio_samples=16000,
            audio_pcm_f32le=b'audio', stable_prefix_tokens=(), rollback_tokens=(20, 21),
            deadline_ns=model.prespec.clock() + (50_000_000 if timeout else 1_000_000_000),
            prefix_text='', model_context={}, sampling_params=settings)
        try:
            self.assertTrue(model.prespec.maybe_start(request, None, 0).launched)
            await asyncio.wait_for(engine.published.wait(), 1)
            if timeout:
                task = model.prespec._tasks[('ws', 1, 1)]
                with self.assertRaises(TimeoutError):
                    await task
            candidate = model.prespec.begin_authoritative_chunk(
                'ws', audio_epoch=1, window_id=1, prefix_version=2, full_audio_end_samples=64000)
            self.assertTrue(candidate.is_partial)
            self.assertEqual(candidate.token_ids, (7, 20, 99, 30))
            await model.prespec.close_session('ws')
            self.assertEqual(engine.aborted, [request.request_id])
            self.assertEqual(engine.closed, [request.request_id])
            self.assertTrue(engine.calls[0]['sampling_params'].extra_args['ascend_mtp'])
        finally:
            await model.prespec.close_session('ws')

    async def test_formal_cancel_retains_verified_partial_with_mtp_enabled(self):
        await self.check_partial(False)

    async def test_deadline_retains_verified_partial_with_mtp_enabled(self):
        await self.check_partial(True)


if __name__ == '__main__':
    unittest.main()
