"""Test worker registration/dispatch without importing vLLM/NPU kernels."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class WorkerPatchTests(unittest.TestCase):
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
            def _get_drafter(self):
                return 'original'
            def _update_states(self, scheduler_output):
                return 'deferred'
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
        with patch.dict(sys.modules, modules):
            exec(Path(__file__).with_name('patch_external_draft_mtp.py').read_text(encoding='utf-8'), namespace)
        return Runner, namespace

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
