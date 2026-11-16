import ast
import asyncio
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


class ServerConfigTests(unittest.TestCase):
    def test_full_mtp_uses_matching_target_graphs_and_fixed_greedy_backend(self):
        from blue_mtp_server import engine_options
        opts=engine_options({'mode':'mtp','graph_mode':'full'})
        self.assertFalse(opts['enforce_eager'])
        self.assertEqual(opts['compilation_config']['cudagraph_mode'],'FULL')
        self.assertEqual(opts['speculative_config']['num_speculative_tokens'],5)
        self.assertFalse(opts['async_scheduling'])
        self.assertFalse(opts['enable_prefix_caching'])
        self.assertEqual(opts['block_size'],128)
        settings=opts['additional_config']['asr_mtp']
        self.assertEqual(settings['graph_max_batch_size'],20)
        self.assertTrue(settings['device_rejection'])

    def test_ar_has_no_auxiliary_model_and_eager_smoke_is_explicit(self):
        from blue_mtp_server import engine_options
        ar=engine_options({'mode':'ar','graph_mode':'off'})
        self.assertTrue(ar['enforce_eager'])
        self.assertNotIn('speculative_config',ar)
        mtp=engine_options({'mode':'mtp','graph_mode':'off'})
        self.assertEqual(mtp['additional_config']['asr_mtp']['graph_mode'],'off')
        self.assertFalse(mtp['additional_config']['asr_mtp']['device_rejection'])
        with self.assertRaises(ValueError):engine_options({'mode':'UNKNOWN'})

    def test_oracle_initial_draft_is_diagnostic_only_never_performance_input(self):
        from blue_mtp_server import request_extra
        self.assertEqual(request_extra('mtp',None,False),{'ascend_mtp':True})
        self.assertIsNone(request_extra('ar',None,False))
        with self.assertRaises(ValueError):request_extra('mtp',[1,2],False)
        extra=request_extra('mtp',[1,2],True)
        self.assertTrue(extra['ascend_external_draft_first_verify'])
        self.assertEqual(extra['ascend_external_draft'],[1,2])


class RequestLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def helpers(self):
        import blue_mtp_server as module
        self.assertTrue(hasattr(module, 'reserve_request'), 'atomic request guard is missing')
        self.assertTrue(hasattr(module, 'finalize_request'), 'bounded cleanup guard is missing')
        return module

    def state(self, engine=None):
        return {'engine': engine, 'active': set(), 'served': 0, 'prompt': 'prompt',
                'cleanup_uncertain': False, 'quarantined': False}

    def handler(self, module, state):
        source = Path(module.__file__).read_text(encoding='utf-8')
        app = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                   and n.name == 'application')
        handler = next(n for n in app.body if isinstance(n, ast.AsyncFunctionDef)
                       and n.name == 'transcribe')
        handler.decorator_list = []
        class HTTPException(Exception):
            def __init__(self, status_code, detail):
                self.status_code, self.detail = status_code, detail
        raw = b'controlled-wave-fixture'
        namespace = {'state': state, 'config': {'mode': 'mtp'}, 'Request': object,
            'cases': {'case': {'sha256': hashlib.sha256(raw).hexdigest()}},
            'hashlib': hashlib, 'io': io, 'json': json,
            'sf': SimpleNamespace(read=lambda *a, **k: (SimpleNamespace(ndim=1), 16000)),
            'np': SimpleNamespace(isfinite=lambda _: SimpleNamespace(all=lambda: True)),
            'SamplingParams': lambda **kwargs: SimpleNamespace(**kwargs),
            'HTTPException': HTTPException, 'request_extra': module.request_extra,
            'reserve_request': module.reserve_request, 'finalize_request': module.finalize_request,
            'RequestUnavailable': module.RequestUnavailable}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[handler], type_ignores=[])),
                     module.__file__, 'exec'), namespace)
        class Request:
            query_params = {'case_id': 'case', 'request_id': 'same'}
            headers = {}
            async def body(self):
                await asyncio.sleep(0.001)
                return raw
        return namespace['transcribe'], Request, HTTPException

    async def test_actual_handler_rechecks_duplicate_id_after_concurrent_body_reads(self):
        module = self.helpers()
        class Engine:
            def __init__(self): self.started = []
            async def generate(self, payload, params, request_id):
                self.started.append(request_id)
                await asyncio.sleep(0.003)
                yield SimpleNamespace(finished=True, outputs=[SimpleNamespace(
                    text='ok', token_ids=[1], finish_reason='stop')])
            async def abort(self, request_id):
                raise AssertionError('completed request must not be aborted')
        engine = Engine()
        state = self.state(engine)
        handler, Request, HTTPException = self.handler(module, state)
        results = await asyncio.gather(handler(Request()), handler(Request()), return_exceptions=True)
        self.assertEqual(engine.started, ['same'])
        self.assertEqual(sum(isinstance(value, dict) for value in results), 1)
        rejected = [value for value in results if isinstance(value, HTTPException)]
        self.assertEqual(len(rejected), 1)
        self.assertIn(rejected[0].status_code, (400, 409))
        self.assertFalse(state['active'])
        self.assertEqual(state['served'], 1)

    async def test_abort_error_releases_active_and_quarantines_future_generation(self):
        module = self.helpers()
        class Engine:
            async def abort(self, request_id):
                await asyncio.sleep(0.001)
                raise RuntimeError('private engine error body')
        state = self.state(Engine())
        module.reserve_request(state, 'a')
        with self.assertRaises(module.CleanupUncertainError) as caught:
            await module.finalize_request(state, 'a', finished=False)
        self.assertFalse(state['active'])
        self.assertTrue(state['cleanup_uncertain'])
        self.assertTrue(state['quarantined'])
        self.assertNotIn('private engine', str(caught.exception))
        with self.assertRaises(module.RequestUnavailable):
            module.reserve_request(state, 'new')

    async def test_cancelled_abort_marks_uncertain_and_still_releases_active(self):
        module = self.helpers()
        class Engine:
            async def abort(self, request_id):
                await asyncio.sleep(1)
        state = self.state(Engine())
        module.reserve_request(state, 'a')
        task = asyncio.create_task(module.finalize_request(state, 'a', finished=False))
        await asyncio.sleep(0.001)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertFalse(state['active'])
        self.assertTrue(state['cleanup_uncertain'])
        self.assertTrue(state['quarantined'])

    async def test_actual_handler_uses_cleanup_guard_when_generation_and_abort_fail(self):
        module = self.helpers()
        class Engine:
            async def generate(self, payload, params, request_id):
                raise RuntimeError('generation failed')
                yield
            async def abort(self, request_id):
                await asyncio.sleep(0.001)
                raise RuntimeError('abort failed')
        state = self.state(Engine())
        handler, Request, _ = self.handler(module, state)
        with self.assertRaises(module.CleanupUncertainError):
            await handler(Request())
        self.assertFalse(state['active'])
        self.assertTrue(state['cleanup_uncertain'])
        self.assertTrue(state['quarantined'])
