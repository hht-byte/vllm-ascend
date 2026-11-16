"""Test worker registration/dispatch without importing vLLM/NPU kernels."""
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

CAPTURE_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 32,
                 40, 48, 64, 80, 96, 128, 144, 160, 180, 200, 256, 300,
                 384, 512, 768, 1024]


def graph_inputs(support='ALWAYS'):
    class Builder:
        @staticmethod
        def get_cudagraph_support(config, cache_spec):
            return SimpleNamespace(name=support)

    class Backend:
        @staticmethod
        def get_builder_cls():
            return Builder

    return [{Backend}], [SimpleNamespace(kv_cache_spec=object())]


class WorkerPatchTests(unittest.TestCase):
    def test_device_rejection_enabled_with_old_hook_fails_before_loading_weights(self):
        from asr_mtp_worker import load_worker
        runner = SimpleNamespace(
            vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(),
                cache_config=SimpleNamespace(enable_prefix_caching=False), quant_config=None),
            use_async_scheduling=False,
            _bookkeeping_sync=lambda *args: None)
        with self.assertRaisesRegex(RuntimeError, 'patch_external_draft_mtp.py'):
            load_worker(runner, None, {'device_rejection': True, 'checkpoint': 'unused.pt'})

    def test_installed_bookkeeping_hook_satisfies_device_handoff_startup_contract(self):
        from asr_mtp_worker import validate_device_handoff_hook
        runner, _ = self.graph_runner()
        validate_device_handoff_hook(runner, enabled=True)
        # Default OFF must remain usable with the old hook or a custom runner.
        validate_device_handoff_hook(SimpleNamespace(), enabled=False)

    def test_graph_capture_precedes_original_profile_and_is_reserved_once(self):
        runner, namespace = self.graph_runner()
        events = []
        graphs = SimpleNamespace(ready=False, reserved_bytes=200, allocated_bytes=160)

        def prepare():
            events.append('capture')
            graphs.ready = True

        graphs.prepare = prepare
        runner.drafter.online = SimpleNamespace(graphs=graphs)

        class Worker:
            def determine_available_memory(self):
                events.append('profile')
                if not graphs.ready:
                    raise AssertionError('graphs must be resident before target activation profiling')
                self.available_kv_cache_memory_bytes = 1000
                return 1000

        namespace['install_worker_memory_hook'](Worker)
        worker = Worker()
        worker.model_runner = runner
        worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=None)
        self.assertEqual(worker.determine_available_memory(), 800)
        self.assertEqual(worker.available_kv_cache_memory_bytes, 800)
        self.assertEqual(events, ['capture', 'profile'])
        self.assertEqual(worker.determine_available_memory(), 800)
        self.assertEqual(events, ['capture', 'profile', 'profile'])

    def test_remaining_pool_reservation_still_rejects_insufficient_budget(self):
        runner, namespace = self.graph_runner()
        graphs = SimpleNamespace(ready=True, reserved_bytes=200, allocated_bytes=160)
        runner.drafter.online = SimpleNamespace(graphs=graphs)

        class Worker:
            def determine_available_memory(self):
                graphs.ready = True
                return 30

        namespace['install_worker_memory_hook'](Worker)
        worker = Worker()
        worker.model_runner = runner
        worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=None)
        with self.assertRaisesRegex(ValueError, 'graph'):
            worker.determine_available_memory()

    def test_auxiliary_graph_reservation_reduces_only_automatic_kv_budget(self):
        runner, namespace = self.graph_runner()
        graphs = SimpleNamespace(ready=True, reserved_bytes=200)
        runner.drafter.online = SimpleNamespace(graphs=graphs)
        class Worker:
            def determine_available_memory(self):
                self.available_kv_cache_memory_bytes = self.base_budget
                return self.base_budget
        install_memory = namespace.get('install_worker_memory_hook')
        self.assertIsNotNone(install_memory, 'startup graph-pool memory accounting is missing')
        install_memory(Worker)
        install_memory(Worker)
        worker = Worker()
        worker.model_runner, worker.base_budget = runner, 1000
        worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=None)
        self.assertEqual(worker.determine_available_memory(), 800)
        self.assertEqual(worker.available_kv_cache_memory_bytes, 800)
        worker.cache_config.kv_cache_memory_bytes = 1000
        self.assertEqual(worker.determine_available_memory(), 1000)
        worker.cache_config.kv_cache_memory_bytes = None
        worker.base_budget = 100
        with self.assertRaisesRegex(ValueError, 'graph'):
            worker.determine_available_memory()
        runner.drafter.online.graphs = None
        self.assertEqual(worker.determine_available_memory(), 100)

    def load(self):
        external = ModuleType('vllm_ascend.spec_decode.external_draft_proposer')
        worker = ModuleType('vllm_ascend.spec_decode.asr_mtp_worker')
        runner_module = ModuleType('vllm_ascend.worker.model_runner_v1')
        class Base:
            def __init__(self, config, runner):
                self.runner = runner
            def propose(self, tokens):
                return [[8] if row else [] for row in tokens]
        class Runner:
            def _bookkeeping_sync(self, scheduler_output, sampler_output, logits, hidden_states,
                                  num_scheduled_tokens, spec_decode_metadata):
                self.seen_sampler = sampler_output
                self.order.append('bookkeeping')
                if hasattr(self, 'on_bookkeeping'):
                    self.on_bookkeeping()
                return 'bookkept'
            def _get_drafter(self):
                return 'original'
            def _update_states(self, scheduler_output):
                return 'deferred'
            def _check_and_update_cudagraph_mode(self, attention_backends, kv_cache_groups):
                # Model upstream resolution: round total-token buckets to K+1,
                # then initialize the dispatcher with the setup query length.
                self.setup_inputs = (attention_backends, kv_cache_groups)
                self.setup_query_len = multiple = self.uniform_decode_query_len
                config = self.compilation_config
                if multiple > 1:
                    config.cudagraph_capture_sizes = sorted({
                        (size + multiple - 1) // multiple * multiple
                        for size in config.cudagraph_capture_sizes
                        if (size + multiple - 1) // multiple * multiple <= config.max_cudagraph_capture_size
                    })
                if getattr(self, 'fail_graph_setup', False):
                    raise RuntimeError('graph setup failed')
                return 'graphs_ready'
            def propose_draft_token_ids(self, valid_sampled_token_ids, sampling_metadata,
                                       scheduler_output, spec_decode_metadata,
                                       spec_decode_common_attn_metadata, positions,
                                       num_scheduled_tokens, hidden_states,
                                       aux_hidden_states=None, sample_hidden_states=None,
                                       target_model_batch_desc=None):
                return 'original_dispatch'
        external.AscendExternalDraftProposer = Base
        worker.load_worker = lambda *a: None
        worker.request_enabled = lambda params: (params.extra_args or {}).get('ascend_mtp', False)
        runner_module.NPUModelRunner = Runner
        modules = {external.__name__: external, worker.__name__: worker,
                   runner_module.__name__: runner_module}
        namespace = {'__name__': 'isolated_mtp_patch'}
        source = Path(__file__).with_name('patch_external_draft_mtp.py')
        if not source.exists():
            # Delivery tests live in examples/910b/tests after patch install.
            source = Path(__file__).resolve().parents[3] / 'vllm_ascend/patch/worker/patch_external_draft_mtp.py'
        with patch.dict(sys.modules, modules):
            exec(source.read_text(encoding='utf-8'), namespace)  # noqa: S102
        return Runner, namespace

    def graph_runner(self, *, mtp=True, mode='FULL'):
        Runner, namespace = self.load()
        runner = Runner()
        runner.vllm_config = SimpleNamespace(additional_config={'asr_mtp': {}} if mtp else {})
        runner.speculative_config = SimpleNamespace(
            method='custom_class', model=namespace['EXTERNAL_CLASS'], num_speculative_tokens=15)
        runner.drafter = runner._get_drafter()
        runner.uniform_decode_query_len = 16
        runner.compilation_config = SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name=mode), cudagraph_capture_sizes=CAPTURE_SIZES.copy(),
            max_cudagraph_capture_size=1024)
        return runner, namespace

    def test_device_rejection_precedes_bookkeeping_and_preserves_original_sampler(self):
        runner, _ = self.graph_runner()
        runner.order = []
        raw, cpu = object(), object()
        sampler = SimpleNamespace(sampled_token_ids=raw, logprobs_tensors=None)
        handoff = SimpleNamespace(prepare=lambda *a: (runner.order.append('mtp'), cpu)[1])
        runner.drafter.online = SimpleNamespace(device_handoff=handoff)
        result = runner._bookkeeping_sync('schedule', sampler, None, 'hidden', 4, 'spec')
        self.assertEqual(result, 'bookkept')
        self.assertEqual(runner.order, ['mtp', 'bookkeeping'])
        self.assertIs(runner.seen_sampler.sampled_token_ids, cpu)
        self.assertIs(sampler.sampled_token_ids, raw)

    def test_device_rejection_fallback_retains_original_bookkeeping_arguments(self):
        runner, _ = self.graph_runner()
        runner.order = []
        sampler = SimpleNamespace(sampled_token_ids=object(), logprobs_tensors=None)
        runner.drafter.online = SimpleNamespace(device_handoff=SimpleNamespace(prepare=lambda *a: None))
        runner._bookkeeping_sync('schedule', sampler, None, 'hidden', 4, 'spec')
        self.assertEqual(runner.order, ['bookkeeping'])
        self.assertIs(runner.seen_sampler, sampler)

    def test_handoff_timing_covers_bookkeeping_late_propose_and_turnaround(self):
        runner, namespace = self.graph_runner()
        runner.order = []
        handoff = SimpleNamespace(profile={'batch': 1}, prepare=lambda *a: object())
        runner.drafter.online = SimpleNamespace(device_handoff=handoff, profile_stages=True,
                                               finish=lambda *a: None)
        runner._bookkeeping_sync('schedule', SimpleNamespace(sampled_token_ids=object()), None, 'hidden', 4, 'spec')
        self.assertIn('bookkeeping_host_ms', handoff.profile)
        runner.drafter.propose_engine = lambda *a: [[]]
        runner.propose_draft_token_ids([], None, 'schedule', None, None, None, 4, 'hidden')
        self.assertIn('late_propose_host_ms', handoff.profile)
        with self.assertLogs(namespace['logger'].name, level='INFO') as logs:
            runner._update_states(SimpleNamespace(finished_req_ids=[]))
        text = '\n'.join(logs.output)
        self.assertIn('ASR_MTP_HANDOFF_STAGE', text)
        self.assertIn('turnaround_host_ms', text)
        self.assertIsNone(handoff.profile)

    def test_handoff_cpu_parser_is_scoped_and_restored_on_success_or_error(self):
        runner, _ = self.graph_runner()
        runner.order = []
        previous, parser = object(), object()
        handoff = SimpleNamespace(prepare=lambda *a: object(), cpu_parser=parser, pending=object())
        finished = []
        runner.input_batch = SimpleNamespace(req_ids=['r'])
        runner.drafter.online = SimpleNamespace(device_handoff=handoff,
                                               finish=lambda ids: finished.extend(ids))
        runner._rejection_output_buffer = previous
        runner.on_bookkeeping = lambda: self.assertIs(runner._rejection_output_buffer, parser)
        runner._bookkeeping_sync('s', SimpleNamespace(sampled_token_ids=object()), None, None, 1, None)
        self.assertIs(runner._rejection_output_buffer, previous)
        def fail():
            self.assertIs(runner._rejection_output_buffer, parser)
            raise RuntimeError('bookkeeping failed')
        runner.on_bookkeeping = fail
        with self.assertRaisesRegex(RuntimeError, 'bookkeeping failed'):
            runner._bookkeeping_sync('s', SimpleNamespace(sampled_token_ids=object()), None, None, 1, None)
        self.assertIs(runner._rejection_output_buffer, previous)
        self.assertIsNone(handoff.pending)
        self.assertEqual(finished, ['r'])

    def test_non_spec_prefill_handoff_runs_before_bookkeeping(self):
        runner, _ = self.graph_runner()
        runner.order = []
        raw, cpu = object(), object()
        sampler = SimpleNamespace(sampled_token_ids=raw, logprobs_tensors=None)
        handoff = SimpleNamespace(prepare=lambda *a: (runner.order.append('mtp'), cpu)[1])
        runner.drafter.online = SimpleNamespace(device_handoff=handoff)
        runner._bookkeeping_sync('schedule', sampler, None, 'hidden', 59, None)
        self.assertEqual(runner.order, ['mtp', 'bookkeeping'])
        self.assertIs(runner.seen_sampler.sampled_token_ids, cpu)
        self.assertIs(sampler.sampled_token_ids, raw)

    def test_full_graph_preserves_small_buckets_and_real_draft_budget(self):
        runner, _ = self.graph_runner()
        backends, groups = graph_inputs()
        self.assertEqual(runner._check_and_update_cudagraph_mode(backends, groups), 'graphs_ready')
        self.assertEqual(runner.compilation_config.cudagraph_capture_sizes, CAPTURE_SIZES)
        self.assertEqual(runner.setup_query_len, 1)
        self.assertEqual(runner.setup_inputs, (backends, groups))
        self.assertEqual(runner.uniform_decode_query_len, 16)
        self.assertEqual(runner.speculative_config.num_speculative_tokens, 15)

    def test_graph_setup_failure_restores_runtime_query_length(self):
        runner, _ = self.graph_runner()
        runner.fail_graph_setup = True
        with self.assertRaisesRegex(RuntimeError, 'graph setup failed'):
            runner._check_and_update_cudagraph_mode([], [])
        self.assertEqual(runner.uniform_decode_query_len, 16)
        self.assertEqual(runner.speculative_config.num_speculative_tokens, 15)
        self.assertEqual(runner.compilation_config.cudagraph_capture_sizes, CAPTURE_SIZES)

    def test_disabled_mtp_and_other_graph_modes_keep_upstream_resolution(self):
        for mtp, mode in [(False, 'FULL'), (True, 'NONE'), (True, 'PIECEWISE'),
                          (True, 'FULL_DECODE_ONLY'), (True, 'FULL_AND_PIECEWISE')]:
            with self.subTest(mtp=mtp, mode=mode):
                runner, _ = self.graph_runner(mtp=mtp, mode=mode)
                self.assertEqual(runner._check_and_update_cudagraph_mode([], []), 'graphs_ready')
                self.assertEqual(runner.setup_query_len, 16)
                self.assertEqual(len(runner.compilation_config.cudagraph_capture_sizes), 17)
                self.assertEqual(runner.uniform_decode_query_len, 16)

    def test_backend_downgrades_keep_real_query_length_and_uniform_buckets(self):
        for support in ('UNIFORM_BATCH', 'UNIFORM_SINGLE_TOKEN_DECODE', 'NEVER'):
            with self.subTest(support=support):
                runner, _ = self.graph_runner()
                runner._check_and_update_cudagraph_mode(*graph_inputs(support))
                # Upstream may downgrade requested FULL to a decode-only mode;
                # it must see the real K+1 for safety checks and bucket rounding.
                self.assertEqual(runner.setup_query_len, 16)
                self.assertEqual(len(runner.compilation_config.cudagraph_capture_sizes), 17)
                self.assertEqual(runner.uniform_decode_query_len, 16)

    def test_graph_hook_install_is_idempotent(self):
        runner, namespace = self.graph_runner()
        installed = type(runner)._check_and_update_cudagraph_mode
        namespace['install']()
        self.assertIs(type(runner)._check_and_update_cudagraph_mode, installed)

    def test_disabled_engine_is_unchanged_and_install_is_idempotent(self):
        Runner, namespace = self.load()
        instance = Runner()
        instance.vllm_config = SimpleNamespace(additional_config={})
        self.assertEqual(instance._get_drafter(), 'original')
        instance.drafter = None
        self.assertEqual(instance.propose_draft_token_ids(*([None] * 8)), 'original_dispatch')
        installed = Runner.propose_draft_token_ids
        namespace['install']()
        self.assertIs(Runner.propose_draft_token_ids, installed)

    def test_mixed_shadow_and_formal_dispatch_and_finished_request_cleanup(self):
        Runner, namespace = self.load()
        instance = Runner()
        instance.vllm_config = SimpleNamespace(additional_config={'asr_mtp': {'checkpoint': '/local.pt'}})
        instance.speculative_config = SimpleNamespace(method='custom_class', model=namespace['EXTERNAL_CLASS'])
        drafter = instance._get_drafter()
        instance.drafter = drafter
        params = lambda enabled: SimpleNamespace(extra_args={'ascend_mtp': enabled})
        instance.input_batch = SimpleNamespace(req_ids=['formal', 'shadow'])
        instance.requests = {'formal': SimpleNamespace(sampling_params=params(True)),
                             'shadow': SimpleNamespace(sampling_params=params(False))}
        finished = []
        drafter.online = SimpleNamespace(propose=lambda *a: [[2, 3], []],
                                          finish=lambda ids: finished.extend(ids))
        output = instance.propose_draft_token_ids([[1], [7]], None, None, None, None, None, 10, object())
        self.assertEqual(output, [[2, 3], [8]])
        self.assertEqual(instance._update_states(SimpleNamespace(finished_req_ids={'formal'})), 'deferred')
        self.assertEqual(finished, ['formal'])


if __name__ == '__main__':
    unittest.main()
