"""Opt-in runtime pinning and worker registration checks without Torch or NPU."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def make_vendor(assets):
    root = assets / 'runtime-deps'
    files = {'python/batch_invariant_ops/__init__.py': b'# fixture only\n',
             'opp/op_api/lib/libfixture.so': b'not an executable fixture'}
    hashes = {}
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    (root / 'tree.json').write_bytes(canonical(hashes))
    config = {'batch_invariant': True, 'target_graph_mode': 'piecewise', 'graph_mode': 'full',
              'batch_invariant_vendor_sha256': hashlib.sha256(canonical(hashes)).hexdigest()}
    return root, config


class RuntimePinTests(unittest.TestCase):
    def server(self):
        import blue_mtp_server
        self.assertTrue(hasattr(blue_mtp_server, 'configure_runtime'), 'runtime dependency pinning is missing')
        return blue_mtp_server

    def test_verified_tree_is_required_before_any_environment_or_path_change(self):
        server = self.server()
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            root, config = make_vendor(assets)
            previous = {'ASCEND_CUSTOM_OPP_PATH': '/vendor/a:/vendor/b', 'LD_LIBRARY_PATH': '/vendor/lib'}
            with patch.dict(os.environ, previous, clear=True), patch.object(sys, 'path', ['original']):
                server.configure_runtime(config, assets)
                self.assertEqual(os.environ['VLLM_BATCH_INVARIANT'], '1')
                self.assertEqual(os.environ['LLM_OPT_BATCH_INVARIANT_MATMUL_BACKEND'], 'ascendc')
                self.assertEqual(os.environ['ASCEND_CUSTOM_OPP_PATH'], '/vendor/a:/vendor/b:' + str(root / 'opp'))
                self.assertEqual(os.environ['LD_LIBRARY_PATH'], str(root / 'opp/op_api/lib') + ':/vendor/lib')
                self.assertEqual(sys.path, [str(root / 'python'), 'original'])

    def test_hash_missing_extra_and_manifest_pin_fail_without_mutating_environment(self):
        server = self.server()
        for kind in ('bytes', 'missing', 'extra', 'pin'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                assets = Path(directory)
                root, config = make_vendor(assets)
                if kind == 'bytes': (root / 'python/batch_invariant_ops/__init__.py').write_bytes(b'mutated')
                elif kind == 'missing': (root / 'opp/op_api/lib/libfixture.so').unlink()
                elif kind == 'extra': (root / 'python/extra.py').write_bytes(b'extra')
                else: config['batch_invariant_vendor_sha256'] = '0' * 64
                with patch.dict(os.environ, {'KEPT': 'same'}, clear=True), patch.object(sys, 'path', ['original']):
                    with self.assertRaises((ValueError, RuntimeError)):
                        server.configure_runtime(config, assets)
                    self.assertEqual(dict(os.environ), {'KEPT': 'same'})
                    self.assertEqual(sys.path, ['original'])

    def test_explicit_triton_selector_is_fixed_and_invalid_values_never_change_process_state(self):
        server = self.server()
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            _, config = make_vendor(assets)
            with patch.dict(os.environ, {}, clear=True), patch.object(sys, 'path', ['original']):
                server.configure_runtime({**config, 'batch_invariant_matmul_backend': 'triton'}, assets)
                self.assertEqual(os.environ['LLM_OPT_BATCH_INVARIANT_MATMUL_BACKEND'], 'triton')
            for invalid in ('unknown', None, True, {}, []):
                with self.subTest(invalid=invalid), patch.dict(os.environ, {'KEPT': 'same'}, clear=True), \
                        patch.object(sys, 'path', ['original']):
                    with self.assertRaises(ValueError):
                        server.configure_runtime({**config, 'batch_invariant_matmul_backend': invalid}, assets)
                    self.assertEqual(dict(os.environ), {'KEPT': 'same'})
                    self.assertEqual(sys.path, ['original'])

    def test_dependency_symlink_is_rejected_before_environment_changes(self):
        server = self.server()
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            root, config = make_vendor(assets)
            external = assets / 'external.py'
            external.write_bytes(b'external')
            link = root / 'python/linked.py'
            try: link.symlink_to(external)
            except OSError: self.skipTest('symlink creation is unavailable in this test environment')
            with patch.dict(os.environ, {'KEPT': 'same'}, clear=True), patch.object(sys, 'path', ['original']):
                with self.assertRaises((ValueError, RuntimeError)):
                    server.configure_runtime(config, assets)
                self.assertEqual(dict(os.environ), {'KEPT': 'same'})
                self.assertEqual(sys.path, ['original'])

    def test_default_is_inert_and_explicit_target_piecewise_keeps_mtp_full(self):
        server = self.server()
        with patch.dict(os.environ, {'KEPT': 'same'}, clear=True), patch.object(sys, 'path', ['original']):
            server.configure_runtime({}, Path('/assets'))
            self.assertEqual(dict(os.environ), {'KEPT': 'same'})
            self.assertEqual(sys.path, ['original'])
        options = server.engine_options({'graph_mode': 'full', 'target_graph_mode': 'piecewise',
                                         'batch_invariant': True})
        self.assertEqual(options['compilation_config']['cudagraph_mode'], 'PIECEWISE')
        self.assertFalse(options['enforce_eager'])
        self.assertEqual(options['additional_config']['asr_mtp']['graph_mode'], 'full')
        self.assertEqual(server.engine_options({})['compilation_config']['cudagraph_mode'], 'FULL')
        self.assertTrue(server.engine_options({'graph_mode': 'off'})['enforce_eager'])
        for config in ({'batch_invariant': True}, {'graph_mode': 'full', 'target_graph_mode': 'off'},
                       {'batch_invariant': 'yes'}, {'target_graph_mode': 'unknown'}):
            with self.subTest(config=config), self.assertRaises(ValueError): server.engine_options(config)


def package_modules(*leaves):
    result = {}
    for leaf in leaves:
        pieces = leaf.__name__.split('.')
        for count in range(1, len(pieces)):
            name = '.'.join(pieces[:count])
            if name not in result:
                result[name] = ModuleType(name)
                result[name].__path__ = []
        result[leaf.__name__] = leaf
    for name, module in result.items():
        parent, _, attribute = name.rpartition('.')
        if parent in result: setattr(result[parent], attribute, module)
    return result


class WorkerRegistrationTests(unittest.TestCase):
    def fixture(self, *, requested=True, ascendc=True, triton=True, actual_backend='ascendc'):
        import blue_worker_probe as probe
        self.assertTrue(hasattr(probe, 'batch_invariant_status'), 'worker batch state is missing')
        events = []
        worker = ModuleType('fixture_worker')
        batch = ModuleType('vllm_ascend.batch_invariant')
        batch.HAS_ASCENDC_BATCH_INVARIANT, batch.HAS_TRITON = ascendc, triton
        batch._batch_invariant_LIB = None
        batch.BLUE_BATCH_INVARIANT_MATMUL_BACKEND = None
        def original():
            events.append('stock')
            batch._batch_invariant_LIB = object()
            batch.BLUE_BATCH_INVARIANT_MATMUL_BACKEND = actual_backend
            return 'initialized'
        worker.init_batch_invariance = original
        envs = ModuleType('vllm.envs')
        envs.VLLM_BATCH_INVARIANT = requested
        npu = ModuleType('torch_npu')
        torch = ModuleType('torch')
        ops = object()
        torch.ops = SimpleNamespace(batch_invariant_ops=ops)
        bridge = ModuleType('vllm_ascend.patch.worker.blue_batch_invariant')
        def install_fia_bridge(actual_npu, actual_ops):
            self.assertIs(actual_npu, npu)
            self.assertIs(actual_ops, ops)
            events.append('bridge')
            npu.bridge_ready = True
        bridge.install_fia_bridge = install_fia_bridge
        bridge.fia_bridge_installed = lambda actual: bool(getattr(actual, 'bridge_ready', False))
        class Worker: pass
        Worker.__module__ = worker.__name__
        return probe, Worker, worker, events, package_modules(worker, batch, envs, npu, torch, bridge)

    def test_worker_bridge_runs_after_stock_registration_once_only_when_requested(self):
        probe, Worker, worker, events, modules = self.fixture()
        with patch.dict(sys.modules, modules):
            probe.install(Worker)
            wrapped = worker.init_batch_invariance
            probe.install(Worker)
            self.assertIs(worker.init_batch_invariance, wrapped)
            self.assertEqual(events, [])
            self.assertEqual(wrapped(), 'initialized')
            self.assertEqual(events, ['stock', 'bridge'])
            self.assertEqual(probe.batch_invariant_status(), {'requested': True, 'registered': True,
                'has_ascendc': True, 'has_triton': True, 'bridge_registered': True,
                'matmul_backend': 'ascendc'})
        probe, Worker, worker, events, modules = self.fixture(requested=False)
        with patch.dict(sys.modules, modules):
            probe.install(Worker)
            worker.init_batch_invariance()
            self.assertEqual(events, ['stock'])

    def test_partial_operator_registration_is_not_accepted_as_full_mode(self):
        for ascendc, triton in ((False, True), (True, False)):
            probe, Worker, worker, events, modules = self.fixture(ascendc=ascendc, triton=triton)
            with self.subTest(ascendc=ascendc, triton=triton), patch.dict(sys.modules, modules):
                probe.install(Worker)
                with self.assertRaises(RuntimeError): worker.init_batch_invariance()
                self.assertEqual(events, ['stock'])

    def test_post_init_reports_and_checks_actual_backend_before_fia_bridge(self):
        probe, Worker, worker, events, modules = self.fixture(actual_backend='triton')
        with patch.dict(sys.modules, modules), patch.dict(os.environ,
                {'LLM_OPT_BATCH_INVARIANT_MATMUL_BACKEND': 'triton'}):
            self.assertIsNone(probe.batch_invariant_status()['matmul_backend'])
            probe.install(Worker)
            worker.init_batch_invariance()
            self.assertEqual(events, ['stock', 'bridge'])
            self.assertEqual(probe.batch_invariant_status()['matmul_backend'], 'triton')
        for actual in ('ascendc', None):
            probe, Worker, worker, events, modules = self.fixture(actual_backend=actual)
            with self.subTest(actual=actual), patch.dict(sys.modules, modules), patch.dict(os.environ,
                    {'LLM_OPT_BATCH_INVARIANT_MATMUL_BACKEND': 'triton'}):
                probe.install(Worker)
                with self.assertRaises(RuntimeError): worker.init_batch_invariance()
                self.assertEqual(events, ['stock'])


class ObservationTests(unittest.IsolatedAsyncioTestCase):
    def test_full_status_must_match_registered_common_backend(self):
        import blue_mtp_server as server
        status = {'batch_invariant': {'requested': True, 'registered': True, 'has_ascendc': True,
                                     'has_triton': True, 'bridge_registered': True, 'matmul_backend': 'ascendc'}}
        server.validate_batch_invariant_status({'batch_invariant': True}, status)
        with self.assertRaises(RuntimeError):
            server.validate_batch_invariant_status({'batch_invariant': True,
                'batch_invariant_matmul_backend': 'triton'}, status)
        status['batch_invariant']['matmul_backend'] = 'triton'
        server.validate_batch_invariant_status({'batch_invariant': True,
            'batch_invariant_matmul_backend': 'triton'}, status)

    async def test_actual_status_handler_refuses_partial_batch_mode(self):
        import blue_mtp_server as server
        self.assertTrue(hasattr(server, 'validate_batch_invariant_status'), 'status mode guard is missing')
        tree = ast.parse(Path(server.__file__).read_text(encoding='utf-8'))
        app = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'application')
        observe = next(n for n in app.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'observe')
        class Engine:
            async def collective_rpc(self, *args, **kwargs):
                return [{'batch_invariant': {'requested': True, 'registered': True,
                    'has_ascendc': False, 'has_triton': True, 'bridge_registered': False}}]
        namespace = {'state': {'engine': Engine(), 'active': set(), 'served': 0,
            'cleanup_uncertain': False, 'quarantined': False}, 'config': {'batch_invariant': True},
            'validate_batch_invariant_status': server.validate_batch_invariant_status}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[observe], type_ignores=[])),
                     server.__file__, 'exec'), namespace)
        with self.assertRaises(RuntimeError): await namespace['observe']()
