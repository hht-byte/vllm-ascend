"""Bounded external-draft transport and real v0.23 dispatch contracts on CPU."""
import ast
import builtins
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
SNAPSHOT = Path(os.environ.get('BLUE_ENGINE_SNAPSHOT', str(
    ROOT.parents[2] / 'asr-mtp-e2e-20261116' / 'engine-snapshot')))


class TokenTable:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    def __getitem__(self, key):
        row, columns = key
        return self.rows[row][columns]

    def __setitem__(self, key, value):
        raise AssertionError('proposer must not write already-bookkept tokens')


class FixtureIsolationTests(unittest.TestCase):
    def test_fixture_never_imports_installed_engine_or_torch_packages(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            root = name.split('.')[0]
            if root in ('torch', 'vllm') or (root == 'vllm_ascend' and name not in sys.modules):
                raise AssertionError('fixture tried to import an installed engine dependency')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded):
            external, _ = load_modules()
            self.assertEqual(external.first_mismatch((1, 2), (1, 3)), 1)


def engine_modules(*leaves):
    """Supply complete package parents so imports cannot execute an installed engine."""
    modules = {}
    for leaf in leaves:
        pieces = leaf.__name__.split('.')
        for count in range(1, len(pieces)):
            name = '.'.join(pieces[:count])
            if name not in modules:
                package = ModuleType(name)
                package.__path__ = []
                modules[name] = package
        modules[leaf.__name__] = leaf
    for name, value in modules.items():
        parent, _, attribute = name.rpartition('.')
        if parent in modules:
            setattr(modules[parent], attribute, value)
    return modules


def load_modules():
    name = 'vllm_ascend.spec_decode.ngram_proposer'
    ngram = ModuleType(name)
    class AscendNgramProposer:
        def __init__(self, *args):
            raise AssertionError('external proposer must not allocate ngram machinery')
    ngram.AscendNgramProposer = AscendNgramProposer
    external_name = 'vllm_ascend.spec_decode.external_draft_proposer'
    spec = importlib.util.spec_from_file_location(external_name, ROOT / 'blue_external_draft.py')
    external = importlib.util.module_from_spec(spec)
    rejoin_spec = importlib.util.spec_from_file_location(
        'vllm_ascend.spec_decode.external_draft_rejoin', ROOT / 'external_draft_rejoin.py')
    rejoin = importlib.util.module_from_spec(rejoin_spec)
    rejoin_spec.loader.exec_module(rejoin)
    with patch.dict(sys.modules, engine_modules(ngram, rejoin, external)):
        spec.loader.exec_module(external)
    return external, ngram


def batch(ids=('a',), outputs=((9,),), *, first=True, candidate=(1, 2, 3, 4), k=4):
    rows = [[101, 102, *tokens, *([0] * 16)] for tokens in outputs]
    result = SimpleNamespace(req_ids=list(ids), num_tokens_no_spec=[2 + len(t) for t in outputs],
                             num_prompt_tokens=[2] * len(ids), max_model_len=20,
                             token_ids_cpu=TokenTable(rows), spec_decode_unsupported_reqs=set(),
                             vocab_size=200, _external_draft_kmax=k)
    result._external_draft_tokens = {req: tuple(candidate) for req in ids}
    result._external_draft_first_verify = set(ids) if first else set()
    result._external_draft_policies = {req: 'ngram_1' for req in ids}
    result._external_draft_last_proposal = {}
    result._external_draft_aligned_blocked = set()
    result._external_draft_first_mismatch_ngram = set(ids)
    return result


class ExternalDraftTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / 'blue_external_draft.py').exists(), 'bounded proposer is missing')
        self.external, self.ngram = load_modules()

    def proposer(self, input_batch, *, asynchronous=False):
        runner = SimpleNamespace(input_batch=input_batch, use_async_scheduling=asynchronous)
        config = SimpleNamespace(speculative_config=SimpleNamespace(num_speculative_tokens=4))
        return self.external.AscendExternalDraftProposer(config, runner)

    def test_upstream_config_only_bootstrap_is_inert_until_ascend_rebinds_runner(self):
        config=SimpleNamespace(speculative_config=SimpleNamespace(num_speculative_tokens=4))
        provisional=self.external.AscendExternalDraftProposer(config)
        self.assertIsNone(provisional.runner)
        with self.assertRaises(RuntimeError):provisional.propose([[9]])
        live=self.proposer(batch(first=False))
        self.assertEqual(live.propose([[9]]),[[1,2,3,4]])

    def test_legacy_draft_is_one_shot_without_reappending_bookkept_tokens(self):
        value = batch(first=False)
        proposer = self.proposer(value)
        self.assertEqual(proposer.propose([[9]]), [[1, 2, 3, 4]])
        self.assertEqual(value.num_tokens_no_spec, [3])
        self.assertEqual(value.token_ids_cpu.rows[0][:4], [101, 102, 9, 0])
        self.assertEqual(proposer.propose([[9]]), [[]])

    def test_first_verify_uses_only_confirmed_prefix_then_rejoins_after_mismatch(self):
        value = batch()
        proposer = self.proposer(value)
        self.assertEqual(proposer.propose([[9]]), [[2, 3, 4]])
        value.token_ids_cpu.rows[0][3] = 3
        value.num_tokens_no_spec[0] = 4
        self.assertEqual(proposer.propose([[3]]), [[4]])

    def test_full_reordered_batch_maps_each_sample_row_to_its_current_request(self):
        value = batch(ids=('b', 'a'), outputs=((7,), (1, 2)), candidate=(1, 2, 3, 4))
        value._external_draft_tokens['b'] = (7, 8, 9)
        self.assertEqual(self.proposer(value).propose([[7], [2]]), [[8, 9], [3, 4]])

    def test_empty_and_unsupported_rows_do_not_consume_legacy_candidates(self):
        value = batch(ids=('b', 'a'), outputs=((7,), (9,)), first=False)
        value.spec_decode_unsupported_reqs.add('a')
        self.assertEqual(self.proposer(value).propose([[], [9]]), [[], []])
        self.assertEqual(set(value._external_draft_tokens), {'a', 'b'})

    def test_context_capacity_is_based_on_current_bookkept_length(self):
        value = batch(first=False)
        value.max_model_len = 5
        self.assertEqual(self.proposer(value).propose([[9]]), [[1, 2]])

    def test_async_and_non_cpu_or_partial_rows_fail_before_consuming_state(self):
        value = batch(first=False)
        with self.assertRaisesRegex(ValueError, 'async'):
            self.proposer(value, asynchronous=True)
        proposer = self.proposer(value)
        for sampled in ((), [], [[True]], [[-1]], [[9], [1]]):
            with self.subTest(sampled=sampled), self.assertRaises((TypeError, ValueError)):
                proposer.propose(sampled)
        self.assertEqual(value._external_draft_tokens, {'a': (1, 2, 3, 4)})

    def test_actual_runner_typed_ngram_branch_accepts_external_subclass(self):
        runner_path = SNAPSHOT / 'vllm_ascend/worker/model_runner_v1.py'
        if not runner_path.exists():
            self.skipTest('pinned engine snapshot is unavailable')
        tree = ast.parse(runner_path.read_text(encoding='utf-8'))
        runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NPUModelRunner')
        method = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == 'propose_draft_token_ids')
        typed = next(n for n in ast.walk(method) if isinstance(n, ast.If) and isinstance(n.test, ast.Call)
                     and isinstance(n.test.func, ast.Name) and n.test.func.id == 'isinstance'
                     and isinstance(n.test.args[1], ast.Name) and n.test.args[1].id == 'AscendNgramProposer')
        standalone = ast.FunctionDef(name='dispatch', args=ast.arguments(posonlyargs=[],
            args=[ast.arg(arg='self'), ast.arg(arg='valid_sampled_token_ids')], kwonlyargs=[],
            kw_defaults=[], defaults=[]), body=[ast.If(test=typed.test, body=typed.body, orelse=[]),
            ast.Return(value=ast.Name(id='draft_token_ids', ctx=ast.Load()))], decorator_list=[])
        namespace = {'AscendNgramProposer': self.ngram.AscendNgramProposer, 'vllm_version_is': lambda _: True}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[standalone], type_ignores=[])), str(runner_path), 'exec'), namespace)
        value = batch(first=False)
        self.assertEqual(namespace['dispatch'](SimpleNamespace(drafter=self.proposer(value)), [[9]]), [[1, 2, 3, 4]])


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / 'blue_external_draft_transport.py').exists(), 'bounded transport is missing')
        self.external, _ = load_modules()
        class InputBatch:
            def __init__(self, num_speculative_tokens=4):
                self.vocab_size = 200
                self.added = []
                self.removed = []
            def add_request(self, request):
                if getattr(request, 'fail', False):
                    raise RuntimeError('allocation failed')
                self.added.append(request.req_id)
                return 3
            def remove_request(self, req_id):
                self.removed.append(req_id)
                return 3
        module = ModuleType('vllm_ascend.worker.npu_input_batch')
        module.NPUInputBatch = InputBatch
        name = 'blue_transport_isolated'
        spec = importlib.util.spec_from_file_location(name, ROOT / 'blue_external_draft_transport.py')
        self.transport = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, engine_modules(module, self.external)):
            spec.loader.exec_module(self.transport)
        self.Batch = InputBatch

    def request(self, extra, **kwargs):
        return SimpleNamespace(req_id='a', sampling_params=SimpleNamespace(extra_args=extra), **kwargs)

    def test_invalid_metadata_is_rejected_before_batch_side_effects(self):
        cases = ({'ascend_external_draft': [True]}, {'ascend_external_draft': [200]},
                 {'ascend_external_draft': [1, 2, 3, 4, 5]},
                 {'ascend_external_draft': [1], 'ascend_external_draft_first_verify': 'yes'},
                 {'ascend_external_draft': [], 'ascend_external_draft_first_verify': True},
                 {'ascend_external_draft': [1], 'ascend_external_draft_first_verify': True,
                  'ascend_external_draft_rejoin': 'ngram_4'})
        for extra in cases:
            with self.subTest(extra=extra):
                value = self.Batch()
                with self.assertRaises((TypeError, ValueError)):
                    value.add_request(self.request(extra))
                self.assertEqual(value.added, [])

    def test_add_copies_candidate_and_remove_cleans_all_metadata_for_reused_id(self):
        value = self.Batch()
        tokens = [1, 2, 3]
        request = self.request({'ascend_external_draft': tokens, 'ascend_external_draft_first_verify': True,
                                'ascend_external_draft_rejoin': 'ngram_2'})
        self.assertEqual(value.add_request(request), 3)
        tokens.append(4)
        self.assertEqual(value._external_draft_tokens['a'], (1, 2, 3))
        value._external_draft_last_proposal['a'] = ('aligned', (2, 3))
        self.assertEqual(value.remove_request('a'), 3)
        value.add_request(self.request({}))
        for name in self.external.REQUEST_STATE_ATTRS:
            self.assertNotIn('a', getattr(value, name))

    def test_failed_add_does_not_register_candidate(self):
        value = self.Batch()
        with self.assertRaisesRegex(RuntimeError, 'allocation'):
            value.add_request(self.request({'ascend_external_draft': [1]}, fail=True))
        self.assertFalse(getattr(value, '_external_draft_tokens', {}))

    def test_install_is_idempotent_and_does_not_duplicate_adds(self):
        self.transport.install(self.Batch)
        value = self.Batch()
        value.add_request(self.request({'ascend_external_draft': [1]}))
        self.assertEqual(value.added, ['a'])

    def test_reinitialized_input_batch_keeps_its_own_speculative_bound(self):
        value = self.Batch(num_speculative_tokens=1)
        value.add_request(self.request({'ascend_external_draft': [1]}))
        other = self.Batch(num_speculative_tokens=2)
        other.add_request(self.request({'ascend_external_draft': [1, 2]}))
        with self.assertRaises(ValueError):
            value.add_request(self.request({'ascend_external_draft': [1, 2]}))

    def test_existing_first_verify_hook_layers_without_changing_batch_result(self):
        module = ModuleType('vllm_ascend.worker.npu_input_batch')
        module.NPUInputBatch = self.Batch
        namespace = {'__name__': 'existing_first_verify_isolated'}
        with patch.dict(sys.modules, engine_modules(module, self.external)):
            exec((ROOT / 'patch_external_draft_first_verify.py').read_text(encoding='utf-8'), namespace)
        value = self.Batch()
        request = self.request({'ascend_external_draft': [1, 2, 3],
            'ascend_external_draft_first_verify': True,
            'ascend_external_draft_ngram_after_first_mismatch': True})
        self.assertEqual(value.add_request(request), 3)
        self.assertEqual(value.added, ['a'])
        self.assertIn('a', value._external_draft_first_verify)
        self.assertIn('a', value._external_draft_first_mismatch_ngram)
        value.remove_request('a')
        self.assertEqual(value.removed, ['a'])
        for name in self.external.REQUEST_STATE_ATTRS:
            self.assertNotIn('a', getattr(value, name))

    def prepare_row(self, value, outputs=(9,)):
        row = batch(outputs=(outputs,))
        for name in ('req_ids', 'num_tokens_no_spec', 'num_prompt_tokens', 'max_model_len',
                     'token_ids_cpu', 'spec_decode_unsupported_reqs'):
            setattr(value, name, getattr(row, name))
        return self.external.AscendExternalDraftProposer(
            SimpleNamespace(speculative_config=SimpleNamespace(num_speculative_tokens=4)),
            SimpleNamespace(input_batch=value, use_async_scheduling=False))

    def layer_first_verify(self):
        module = ModuleType('vllm_ascend.worker.npu_input_batch')
        module.NPUInputBatch = self.Batch
        with patch.dict(sys.modules, engine_modules(module, self.external)):
            exec((ROOT / 'patch_external_draft_first_verify.py').read_text(encoding='utf-8'),
                 {'__name__': 'existing_first_verify_lifetime'})

    def test_row_eviction_and_failed_readds_do_not_repeat_consumed_legacy_draft(self):
        self.layer_first_verify()
        value = self.Batch()
        request = self.request({'ascend_external_draft': [1, 2, 3]})
        value.add_request(request)
        proposer = self.prepare_row(value)
        self.assertEqual(proposer.propose([[9]]), [[1, 2, 3]])
        value.remove_request('a')
        for _ in range(2):
            request.fail = True
            with self.assertRaisesRegex(RuntimeError, 'allocation'):
                value.add_request(request)
            self.assertFalse(value._external_draft_request_states)
        request.fail = False
        value.add_request(request)
        self.assertEqual(proposer.propose([[9]]), [[]])
        value.remove_request('a')
        value.add_request(self.request({'ascend_external_draft': [4, 5]}))
        self.assertEqual(proposer.propose([[9]]), [[4, 5]])

    def test_first_verify_retry_history_survives_outer_hook_row_evictions(self):
        self.layer_first_verify()
        value = self.Batch()
        request = self.request({'ascend_external_draft': [1, 2, 3, 4],
                                'ascend_external_draft_first_verify': True})
        value.add_request(request)
        proposer = self.prepare_row(value)
        self.assertEqual(proposer.propose([[9]]), [[2, 3, 4]])
        value.remove_request('a')
        value.add_request(request)
        self.prepare_row(value, outputs=(9, 3))
        self.assertEqual(proposer.propose([[3]]), [[]])
        value.remove_request('a')
        value.add_request(request)
        self.assertEqual(proposer.propose([[3]]), [[]])

    def test_changed_metadata_on_same_cached_request_is_rejected_before_readd(self):
        value = self.Batch()
        request = self.request({'ascend_external_draft': [1]})
        value.add_request(request)
        value.remove_request('a')
        request.sampling_params.extra_args['ascend_external_draft'] = [2]
        with self.assertRaisesRegex(ValueError, 'lifetime'):
            value.add_request(request)
        self.assertEqual(value.added, ['a'])

    def test_request_lifetime_survives_move_into_replacement_input_batch(self):
        self.layer_first_verify()
        request = self.request({'ascend_external_draft': [1, 2, 3, 4],
                                'ascend_external_draft_first_verify': True})
        first_batch = self.Batch()
        first_batch.add_request(request)
        proposer = self.prepare_row(first_batch)
        self.assertEqual(proposer.propose([[9]]), [[2, 3, 4]])
        first_batch.remove_request('a')
        replacement = self.Batch()
        replacement.add_request(request)
        next_proposer = self.prepare_row(replacement, outputs=(9, 3))
        self.assertEqual(next_proposer.propose([[3]]), [[]])
        self.assertFalse(first_batch._external_draft_request_states)
        replacement.remove_request('a')
        self.assertFalse(replacement._external_draft_request_states)


if __name__ == '__main__':
    unittest.main()
