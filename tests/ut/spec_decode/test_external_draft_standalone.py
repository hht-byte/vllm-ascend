"""CPU behavioral checks; run directly to avoid the NPU-wide UT conftest.

Only vLLM's constructor and the input batch's inherited lifecycle boundary
are substituted. The Ascend ngram base, external proposer, patch installers,
and CPU Torch token storage are real production code.
"""

import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
CLASS_PATH = "vllm_ascend.spec_decode.external_draft_proposer.AscendExternalDraftProposer"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def runtime():
    class NgramProposer:
        def __init__(self, config):
            # Match v0.23's constructor boundary: custom_class leaves these
            # fields unset, so external drafts must not initialize ngram JIT.
            assert config.speculative_config.prompt_lookup_min is not None
            assert config.speculative_config.prompt_lookup_max is not None
            self.k = config.speculative_config.num_speculative_tokens
            self.propose([[]])

    class Batch:
        def __init__(self):
            self.req_ids = []
            self.requests = {}
            self.spec_decode_unsupported_reqs = set()
            self.max_model_len = 24
            self.token_tensor = torch.full((4, self.max_model_len), 777, dtype=torch.int32)
            self.token_ids_cpu = self.token_tensor.numpy()
            self.num_tokens_no_spec = [3] * 4
            self.num_prompt_tokens = [3] * 4
            self.add_calls = 0
            self.remove_calls = 0

        def add_request(self, request):
            self.add_calls += 1
            self.requests[request.req_id] = request
            self.req_ids.append(request.req_id)
            return len(self.req_ids) - 1

        def remove_request(self, req_id):
            self.remove_calls += 1
            self.requests.pop(req_id, None)
            if req_id in self.req_ids:
                self.req_ids.remove(req_id)
            return 7

    class Runner:
        def __init__(self, method="custom_class", model=CLASS_PATH):
            self.input_batch = Batch()
            self.speculative_config = types.SimpleNamespace(
                method=method,
                model=model,
                num_speculative_tokens=15,
                prompt_lookup_min=None,
                prompt_lookup_max=None,
            )
            self.vllm_config = types.SimpleNamespace(
                speculative_config=self.speculative_config,
                model_config=types.SimpleNamespace(max_model_len=self.input_batch.max_model_len),
            )

        def _get_drafter(self):
            return "original-drafter"

    modules = {}
    for name in (
        "vllm",
        "vllm.v1",
        "vllm.v1.spec_decode",
        "vllm_ascend",
        "vllm_ascend.spec_decode",
        "vllm_ascend.patch",
        "vllm_ascend.patch.worker",
        "vllm_ascend.worker",
    ):
        modules[name] = types.ModuleType(name)
        modules[name].__path__ = []
    upstream = types.ModuleType("vllm.v1.spec_decode.ngram_proposer")
    upstream.NgramProposer = NgramProposer
    modules[upstream.__name__] = upstream
    utils = types.ModuleType("vllm_ascend.utils")
    utils.vllm_version_is = lambda version: version == "0.23.0"
    modules[utils.__name__] = utils
    batch_module = types.ModuleType("vllm_ascend.worker.npu_input_batch")
    batch_module.NPUInputBatch = Batch

    with patch.dict(sys.modules, modules):
        load_file("vllm_ascend.spec_decode.ngram_proposer", ROOT / "vllm_ascend/spec_decode/ngram_proposer.py")
        rejoin = load_file(
            "vllm_ascend.spec_decode.external_draft_rejoin", ROOT / "vllm_ascend/spec_decode/external_draft_rejoin.py"
        )
        proposer = load_file(
            "vllm_ascend.spec_decode.external_draft_proposer",
            ROOT / "vllm_ascend/spec_decode/external_draft_proposer.py",
        )
        base = load_file(
            "vllm_ascend.patch.worker.patch_external_draft", ROOT / "vllm_ascend/patch/worker/patch_external_draft.py"
        )
        first = load_file(
            "vllm_ascend.patch.worker.patch_external_draft_first_verify",
            ROOT / "vllm_ascend/patch/worker/patch_external_draft_first_verify.py",
        )
        # Importing the installers above must not need an initialized runner
        # or input batch. Supply the external batch only for explicit install.
        sys.modules[batch_module.__name__] = batch_module
        yield types.SimpleNamespace(
            Runner=Runner, Batch=Batch, base=base, first=first, proposer=proposer, rejoin=rejoin
        )


def request(req_id="a", draft=(1, 2, 3, 4, 5), *, first=True, candidate=True, policy="ngram_1"):
    extra = {"ascend_external_draft": list(draft)}
    if first:
        extra["ascend_external_draft_first_verify"] = True
        extra["ascend_external_draft_rejoin"] = policy
        if candidate:
            extra["ascend_external_draft_ngram_after_first_mismatch"] = True
    return types.SimpleNamespace(req_id=req_id, sampling_params=types.SimpleNamespace(extra_args=extra))


def install(env):
    env.base.install(env.Runner)
    env.first.install(env.Runner)


def confirm(batch, ids):
    batch.token_ids_cpu[0, 3 : 3 + len(ids)] = ids
    batch.num_tokens_no_spec[0] = 3 + len(ids)


class ExternalDraftTests(unittest.TestCase):
    def setUp(self):
        for path in (
            "spec_decode/external_draft_proposer.py",
            "spec_decode/external_draft_rejoin.py",
            "patch/worker/patch_external_draft.py",
            "patch/worker/patch_external_draft_first_verify.py",
        ):
            self.assertTrue((ROOT / "vllm_ascend" / path).exists(), f"missing external-draft implementation: {path}")

    def test_import_is_inert_and_install_is_idempotent(self):
        with runtime() as env:
            runner = env.Runner()
            self.assertEqual(runner._get_drafter(), "original-drafter")
            original_add = env.Batch.add_request
            install(env)
            installed = (env.Runner._get_drafter, env.Batch.add_request, env.Batch.remove_request)
            self.assertIsNot(env.Batch.add_request, original_add)
            install(env)
            self.assertEqual(installed, (env.Runner._get_drafter, env.Batch.add_request, env.Batch.remove_request))
            self.assertIsInstance(runner._get_drafter(), env.proposer.AscendExternalDraftProposer)
            self.assertEqual(env.Runner("ngram")._get_drafter(), "original-drafter")
            self.assertEqual(env.Runner(model="other.Custom")._get_drafter(), "original-drafter")

    def test_upstream_custom_factory_bootstrap_precedes_runner_bound_drafter(self):
        with runtime() as env:
            runner = env.Runner()
            try:
                bootstrap = env.proposer.AscendExternalDraftProposer(runner.vllm_config)
            except TypeError as exc:
                self.fail(f"upstream custom_class factory constructs with config only: {exc}")
            self.assertIsNone(bootstrap.runner)
            self.assertEqual(bootstrap.k, 15)
            self.assertEqual(bootstrap.max_model_len, runner.input_batch.max_model_len)
            self.assertIsNone(bootstrap.load_model())
            self.assertIsNone(bootstrap.dummy_run(num_tokens=1, num_reqs=1))
            install(env)
            operational = runner._get_drafter()
            self.assertIsNot(operational, bootstrap)
            self.assertIs(operational.runner, runner)
            runner.input_batch.add_request(request(first=False, draft=(5, 6)))
            self.assertEqual(operational.propose([[9]]), [[5, 6]])

    def test_custom_class_initializes_without_ngram_lookup_configuration(self):
        with runtime() as env:
            install(env)
            for lookup in (None, 0):
                with self.subTest(lookup=lookup):
                    runner = env.Runner()
                    runner.speculative_config.prompt_lookup_min = lookup
                    runner.speculative_config.prompt_lookup_max = lookup
                    batch = runner.input_batch
                    # The runner creates its drafter before its input batch.
                    del runner.input_batch
                    proposer = runner._get_drafter()
                    runner.input_batch = batch
                    batch.add_request(request(first=False, draft=tuple(range(20))))
                    self.assertEqual(proposer.propose([[9]]), [list(range(15))])
                    self.assertIsNone(proposer.load_model())
                    self.assertIsNone(proposer.dummy_run(num_tokens=1, num_reqs=1))
                    self.assertEqual(runner.speculative_config.prompt_lookup_min, lookup)
                    self.assertEqual(runner.speculative_config.prompt_lookup_max, lookup)

    def test_custom_class_rejects_nonpositive_speculative_budget(self):
        with runtime() as env:
            install(env)
            for budget in (0, -1, True):
                with self.subTest(budget=budget):
                    runner = env.Runner()
                    runner.speculative_config.prompt_lookup_min = 0
                    runner.speculative_config.prompt_lookup_max = 0
                    runner.speculative_config.num_speculative_tokens = budget
                    with self.assertRaisesRegex(ValueError, "positive"):
                        runner._get_drafter()

    def test_actual_mtp_registration_layers_over_actual_external_proposer(self):
        with runtime() as env:
            # These signatures are the external v0.23 runner boundary checked
            # by the real MTP installer. No inference methods are executed.
            class Runner(env.Runner):
                def propose_draft_token_ids(
                    self,
                    valid_sampled_token_ids,
                    sampling_metadata,
                    scheduler_output,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                    positions,
                    num_scheduled_tokens,
                    hidden_states,
                    aux_hidden_states=None,
                    sample_hidden_states=None,
                    target_model_batch_desc=None,
                ):
                    raise AssertionError("registration must not run inference")

                def _bookkeeping_sync(
                    self,
                    scheduler_output,
                    sampler_output,
                    logits,
                    hidden_states,
                    num_scheduled_tokens,
                    spec_decode_metadata,
                ):
                    raise AssertionError("registration must not run bookkeeping")

                def _update_states(self, scheduler_output):
                    raise AssertionError("registration must not update requests")

                def _check_and_update_cudagraph_mode(self, attention_backends, kv_cache_groups):
                    raise AssertionError("registration must not prepare graphs")

            env.base.install(Runner)
            env.first.install()
            sys.modules["vllm_ascend.spec_decode"].__path__ = [str(ROOT / "vllm_ascend/spec_decode")]
            runner_module = types.ModuleType("vllm_ascend.worker.model_runner_v1")
            runner_module.NPUModelRunner = Runner
            sys.modules[runner_module.__name__] = runner_module
            native_before = "torch_npu" in sys.modules
            mtp = load_file(
                "vllm_ascend.patch.worker.patch_external_draft_mtp",
                ROOT / "vllm_ascend/patch/worker/patch_external_draft_mtp.py",
            )
            runner = Runner()
            # As during runner initialization, no batch exists yet.
            del runner.input_batch
            runner.vllm_config.additional_config = {}
            self.assertIs(type(runner._get_drafter()), env.proposer.AscendExternalDraftProposer)
            runner.vllm_config.additional_config = {"asr_mtp": {"checkpoint": "unused-at-registration.pt"}}
            proposer = runner._get_drafter()
            self.assertIs(type(proposer), mtp.AscendExternalDraftMTPProposer)
            self.assertIsInstance(proposer, env.proposer.AscendExternalDraftProposer)
            self.assertIsNone(proposer.online)
            self.assertEqual("torch_npu" in sys.modules, native_before)

    def test_transport_copies_tokens_and_cleans_all_request_state(self):
        with runtime() as env:
            install(env)
            batch = env.Batch()
            req = request(policy="ngram_2")
            self.assertEqual(batch.add_request(req), 0)
            req.sampling_params.extra_args["ascend_external_draft"][0] = 90
            self.assertEqual(batch._external_draft_tokens["a"], [1, 2, 3, 4, 5])
            self.assertEqual(batch._external_draft_policies, {"a": "ngram_2"})
            self.assertEqual(batch._external_draft_first_mismatch_ngram, {"a"})
            batch._external_draft_last_proposal["a"] = ("aligned", (2, 3))
            batch._external_draft_aligned_blocked.add("a")
            self.assertEqual(batch.remove_request("a"), 7)
            for name in (
                "_external_draft_tokens",
                "_external_draft_first_verify",
                "_external_draft_policies",
                "_external_draft_last_proposal",
                "_external_draft_aligned_blocked",
                "_external_draft_first_mismatch_ngram",
            ):
                self.assertFalse(getattr(batch, name))
            self.assertEqual((batch.add_calls, batch.remove_calls), (1, 1))

    def test_invalid_payloads_fail_before_batch_mutation(self):
        invalid = [
            {"ascend_external_draft": [-1]},
            {"ascend_external_draft": [True]},
            {"ascend_external_draft": "1,2"},
            {"ascend_external_draft": [1], "ascend_external_draft_first_verify": "false"},
            {"ascend_external_draft": [1], "ascend_external_draft_first_verify": 1},
            {"ascend_external_draft": [1], "ascend_external_draft_first_verify": None},
            {"ascend_external_draft": [], "ascend_external_draft_first_verify": True},
            {
                "ascend_external_draft": [1],
                "ascend_external_draft_first_verify": True,
                "ascend_external_draft_rejoin": "ngram_4",
            },
            {"ascend_external_draft": [1], "ascend_external_draft_ngram_after_first_mismatch": True},
            {
                "ascend_external_draft": [1],
                "ascend_external_draft_first_verify": True,
                "ascend_external_draft_ngram_after_first_mismatch": "yes",
            },
            {
                "ascend_external_draft": [1],
                "ascend_external_draft_first_verify": True,
                "ascend_external_draft_rejoin": "aligned",
                "ascend_external_draft_ngram_after_first_mismatch": True,
            },
        ]
        with runtime() as env:
            install(env)
            for extra in invalid:
                with self.subTest(extra=extra):
                    batch = env.Batch()
                    req = request()
                    req.sampling_params.extra_args = extra
                    with self.assertRaises((ValueError, TypeError)):
                        batch.add_request(req)
                    self.assertEqual(batch.add_calls, 0)
                    self.assertEqual(batch.requests, {})

    def test_legacy_draft_is_one_shot_and_preserves_sampled_tokens(self):
        with runtime() as env:
            install(env)
            runner = env.Runner()
            runner.input_batch.add_request(request(first=False))
            proposer = runner._get_drafter()
            confirm(runner.input_batch, [9])
            before = runner.input_batch.token_tensor.clone()
            self.assertEqual(proposer.propose([[9]]), [[1, 2, 3, 4, 5]])
            self.assertEqual(runner.input_batch.token_tensor[0, 3].item(), 9)
            self.assertTrue(torch.equal(runner.input_batch.token_tensor, before))
            confirm(runner.input_batch, [9, 8])
            self.assertEqual(proposer.propose([[8]]), [[]])

    def test_legacy_uses_already_bookkept_end_for_context_budget(self):
        with runtime() as env:
            install(env)
            runner = env.Runner()
            batch = runner.input_batch
            batch.add_request(request(first=False))
            proposer = runner._get_drafter()
            batch.max_model_len = 6
            confirm(batch, [9, 8])
            before = batch.token_tensor.clone()
            # The confirmed prefix already includes both sampled IDs and
            # leaves one slot. It must not subtract the sampled length again.
            self.assertEqual(proposer.propose([[9, 8]]), [[1]])
            self.assertTrue(torch.equal(batch.token_tensor, before))
            self.assertEqual(batch.token_tensor[0, 5].item(), 777)
            self.assertEqual(proposer.propose([[9, 8]]), [[]])

    def test_legacy_consumption_survives_batch_eviction_of_same_request(self):
        with runtime() as env:
            install(env)
            runner = env.Runner()
            batch = runner.input_batch
            req = request(first=False)
            batch.add_request(req)
            proposer = runner._get_drafter()
            confirm(batch, [9])
            self.assertEqual(proposer.propose([[9]]), [[1, 2, 3, 4, 5]])
            self.assertEqual(proposer.propose([[9]]), [[]])
            batch.remove_request("a")
            for state in vars(batch).values():
                if isinstance(state, dict):
                    self.assertNotIn("a", state)
            # GPUInputBatch evicts unscheduled rows while runner.requests
            # retains and later re-adds this exact CachedRequestState owner.
            batch.add_request(req)
            confirm(batch, [9, 8])
            self.assertEqual(proposer.propose([[8]]), [[]])
            batch.remove_request("a")
            # A new request object with a reused ID owns a new lifetime.
            batch.add_request(request(first=False))
            confirm(batch, [9])
            self.assertEqual(proposer.propose([[9]]), [[1, 2, 3, 4, 5]])

    def test_unconsumed_legacy_offer_survives_batch_eviction(self):
        with runtime() as env:
            install(env)
            runner = env.Runner()
            batch = runner.input_batch
            req = request(first=False)
            batch.add_request(req)
            batch.remove_request("a")
            batch.add_request(req)
            confirm(batch, [9])
            proposer = runner._get_drafter()
            self.assertEqual(proposer.propose([[9]]), [[1, 2, 3, 4, 5]])
            self.assertEqual(proposer.propose([[9]]), [[]])

    def test_first_verify_phase_survives_batch_eviction_of_same_request(self):
        with runtime() as env:
            install(env)
            runner = env.Runner()
            batch = runner.input_batch
            req = request(candidate=False)
            batch.add_request(req)
            proposer = runner._get_drafter()
            confirm(batch, [9])
            self.assertEqual(proposer.propose([[9]]), [[2, 3, 4, 5]])
            confirm(batch, [9, 3])
            self.assertEqual(proposer.propose([[3]]), [[]])
            batch.remove_request("a")
            for state in vars(batch).values():
                if isinstance(state, (dict, set)):
                    self.assertNotIn("a", state)
            batch.add_request(req)
            confirm(batch, [9, 3, 4])
            self.assertEqual(proposer.propose([[4]]), [[]])
            self.assertEqual(batch._external_draft_last_proposal["a"], ("aligned", (2, 3, 4, 5)))
            batch.remove_request("a")
            batch.add_request(request(candidate=False))
            confirm(batch, [9])
            self.assertEqual(proposer.propose([[9]]), [[2, 3, 4, 5]])

    def test_first_verify_reads_confirmed_prefix_without_rewriting_tail(self):
        with runtime() as env:
            install(env)
            runner = env.Runner()
            batch = runner.input_batch
            batch.add_request(request())
            confirm(batch, [1, 2])
            before = batch.token_tensor.clone()
            self.assertEqual(runner._get_drafter().propose([[1, 2]]), [[3, 4, 5]])
            self.assertTrue(torch.equal(batch.token_tensor, before))
            self.assertEqual(batch._external_draft_tokens["a"], [1, 2, 3, 4, 5])

    def test_candidate_first_mismatch_gets_one_aligned_retry_then_ngram(self):
        with runtime() as env:
            install(env)
            runner = env.Runner()
            batch = runner.input_batch
            batch.add_request(request(draft=(1, 2, 3, 4, 3, 4)))
            proposer = runner._get_drafter()
            confirm(batch, [9])
            self.assertEqual(proposer.propose([[9]]), [[2, 3, 4, 3, 4]])
            confirm(batch, [9, 3])
            self.assertEqual(proposer.propose([[3]]), [[4]])
            confirm(batch, [9, 3, 8])
            self.assertEqual(proposer.propose([[8]]), [[]])
            confirm(batch, [9, 3, 8, 3])
            self.assertEqual(proposer.propose([[3]]), [[4]])

    def test_partial_aligned_acceptance_uses_confirmed_correction_anchor(self):
        with runtime() as env:
            install(env)
            runner = env.Runner()
            batch = runner.input_batch
            batch.add_request(request(draft=(1, 2, 3, 4, 5, 6)))
            proposer = runner._get_drafter()
            confirm(batch, [9])
            self.assertEqual(proposer.propose([[9]]), [[2, 3, 4, 5, 6]])
            confirm(batch, [9, 2, 5])
            self.assertEqual(proposer.propose([[2, 5]]), [[6]])

    def test_rollback_first_mismatch_stops_after_failed_aligned_retry(self):
        with runtime() as env:
            install(env)
            runner = env.Runner()
            batch = runner.input_batch
            batch.add_request(request(candidate=False))
            proposer = runner._get_drafter()
            confirm(batch, [9])
            self.assertEqual(proposer.propose([[9]]), [[2, 3, 4, 5]])
            confirm(batch, [9, 3])
            self.assertEqual(proposer.propose([[3]]), [[]])
            self.assertEqual(batch._external_draft_first_mismatch_ngram, set())

    def test_skip_paths_and_remaining_context_bound(self):
        with runtime() as env:
            install(env)
            runner = env.Runner()
            batch = runner.input_batch
            batch.add_request(request())
            proposer = runner._get_drafter()
            confirm(batch, [1])
            batch.max_model_len = 6
            self.assertEqual(proposer.propose([[1]]), [[2, 3]])
            self.assertEqual(proposer.propose([[]]), [[]])
            batch.spec_decode_unsupported_reqs.add("a")
            self.assertEqual(proposer.propose([[1]]), [[]])
            batch.spec_decode_unsupported_reqs.clear()
            batch.num_tokens_no_spec[0] = batch.max_model_len
            self.assertEqual(proposer.propose([[1]]), [[]])
            self.assertIn("a", batch._external_draft_tokens)

    def test_rejoin_policies_and_rightmost_anchor(self):
        with runtime() as env:
            next_draft = env.rejoin.next_draft
            self.assertEqual(next_draft((1, 2, 3), (), max_draft_tokens=2), (1, 2))
            self.assertEqual(next_draft((1, 2, 3), (9,), policy="ngram_3", max_draft_tokens=3), (2, 3))
            self.assertEqual(next_draft((1, 2, 3, 4), (1, 9), policy="aligned", max_draft_tokens=2), (3, 4))
            self.assertEqual(
                next_draft((10, 20, 30, 40, 20, 30, 50), (10, 99, 20, 30), policy="ngram_2", max_draft_tokens=3), (50,)
            )
            self.assertEqual(next_draft((1, 2, 3, 4, 5), (1, 9, 2, 3, 4), policy="ngram_3", max_draft_tokens=3), (5,))
            self.assertEqual(next_draft((1, 2, 3), (1, 9), max_draft_tokens=3), ())
            with self.assertRaises(ValueError):
                next_draft((1,), (), policy="ngram_4", max_draft_tokens=1)
            with self.assertRaises(ValueError):
                next_draft((1,), (), max_draft_tokens=0)


if __name__ == "__main__":
    unittest.main()
