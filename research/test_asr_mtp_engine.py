"""CPU integration checks for the online worker drafter, not a replay script."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(os.environ.get('SPEC_ASR_TEST_ROOT',
                          Path(__file__).resolve().parents[1] / '.tmp/asr-mtp-reference'))
AVAILABLE = (importlib.util.find_spec('torch') is not None and
             importlib.util.find_spec('transformers') is not None and
             (ROOT / 'spec_asr/mtp.py').is_file())


@unittest.skipUnless(AVAILABLE, 'requires CPU Torch, transformers and supplied MTP source')
class EngineHeadsTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding
        sys.path.insert(0, str(ROOT))
        from spec_asr.mtp import SequentialMTP
        self.t = torch
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.config = Qwen3Config(vocab_size=32, hidden_size=8, intermediate_size=16,
                                 num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                                 num_hidden_layers=2, rope_theta=10000)
        self.config._attn_implementation = 'eager'
        text = SimpleNamespace(config=self.config, layers=[Qwen3DecoderLayer(self.config, 1)],
                               norm=torch.nn.RMSNorm(8, eps=self.config.rms_norm_eps))
        self.reference = SequentialMTP(text, 3).eval()
        rotary = Qwen3RotaryEmbedding(self.config)
        self.rotary = lambda hidden, positions: rotary(hidden, positions[0])
        self.embed, self.head = torch.nn.Embedding(32, 8), torch.nn.Linear(8, 32)

    def test_batched_online_heads_match_original_shifted_mtp_after_rejection(self):
        from asr_mtp_heads import BatchedMTPHeads, CacheArena
        t = self.t
        model = BatchedMTPHeads(self.config, 3).eval()
        model.load_state_dict(self.reference.state_dict(), strict=True)
        arena = CacheArena(self.config, 3, max_context=32, slots=2, device='cpu', dtype=t.float32)
        states = [arena.acquire(), arena.acquire()]
        ids = [list(range(3, 15)), list(range(5, 12))]
        for state, row in zip(states, ids):
            state.target[:len(row)] = t.randn(len(row), 8)
            state.length = len(row)
        for round_index in range(3):
            inputs = [row + [4] for row in ids]
            with t.inference_mode():
                actual = model.propose(states, inputs, self.embed, self.head, [3, 1])
                for state, row, output, count in zip(states, inputs, actual, [3, 1]):
                    expected = self.reference.propose(
                        state.target[:state.length].unsqueeze(0), t.tensor([row[:-1]]),
                        t.tensor([[row[-1]]]), self.embed, self.head, self.rotary, max_heads=count)
                    t.testing.assert_close(output[:count], expected[0, 1:])
            # Only the known target drive token and one auxiliary token survive.
            for state, row, output in zip(states, ids, actual):
                row.extend([4, int(output[0])])
                state.target[state.length:len(row)] = t.randn(2, 8)
                state.length = len(row)
                state.crop_heads(state.length)
        self.assertGreater(model.reused_positions, 0)

    def test_arena_is_fixed_capacity_and_slots_are_reusable(self):
        from asr_mtp_heads import CacheArena
        arena = CacheArena(self.config, 3, max_context=16, slots=1,
                           device='cpu', dtype=self.t.float32)
        first = arena.acquire()
        self.assertIsNone(arena.acquire())
        arena.release(first)
        second = arena.acquire()
        self.assertEqual(second.length, 0)
        self.assertEqual([h.valid for h in second.heads], [0, 0, 0])
        self.assertEqual(first.target.data_ptr(), second.target.data_ptr())

    def make_worker(self):
        import numpy as np
        from asr_mtp_heads import BatchedMTPHeads, CacheArena
        from asr_mtp_worker import MTPWorker
        t = self.t
        params = SimpleNamespace(extra_args={'ascend_mtp': True}, max_tokens=20,
                                 temperature=0.0, n=1)
        batch = SimpleNamespace(req_ids=['a'], num_tokens_no_spec=np.array([5]),
                                num_computed_tokens_cpu=np.array([0]), num_prompt_tokens=np.array([4]),
                                token_ids_cpu=np.zeros((1, 32), dtype=np.int32), max_model_len=32,
                                spec_decode_unsupported_reqs=set())
        batch.token_ids_cpu[0, :5] = [3, 4, 5, 6, 7]
        runner = SimpleNamespace(input_batch=batch, query_start_loc=SimpleNamespace(np=np.array([0, 7])),
                                 requests={'a': SimpleNamespace(sampling_params=params, output_token_ids=[7])},
                                 speculative_config=SimpleNamespace(num_speculative_tokens=3))
        model = BatchedMTPHeads(self.config, 3).eval()
        model.load_state_dict(self.reference.state_dict())
        arena = CacheArena(self.config, 3, max_context=32, slots=1, device='cpu', dtype=t.float32)
        worker = MTPWorker(runner, model, arena, self.embed, self.head, {31})
        return worker, runner

    def test_initial_external_mismatch_uses_only_confirmed_prefill_hidden(self):
        worker, runner = self.make_worker()
        t = self.t
        hidden = t.randn(7, 8)  # prompt(4) plus rejected external draft(3)
        step = SimpleNamespace(num_scheduled_tokens={'a': 7})
        with t.inference_mode():
            actual = worker.propose([[7]], step, hidden)
            expected = self.reference.propose(hidden[:4].unsqueeze(0), t.tensor([[3, 4, 5, 6]]),
                t.tensor([[7]]), self.embed, self.head, self.rotary, max_heads=3)
        self.assertEqual(actual[0], expected[0, 1:].tolist())
        self.assertEqual(worker.states['a'].length, 4)
        t.testing.assert_close(worker.states['a'].target[:4], hidden[:4])
        self.assertEqual(worker.stats['target_forwards'], 0)

    def test_shadow_verifies_legacy_rollback_once_then_uses_repaired_state_for_mtp(self):
        for accepted in (0, 1, 2):
            with self.subTest(accepted=accepted):
                worker, runner = self.make_worker()
                params = runner.requests['a'].sampling_params
                params.extra_args['ascend_external_draft'] = [20, 21]
                first_hidden = self.t.randn(4, 8)
                with self.t.inference_mode():
                    initial = worker.propose([[7]], SimpleNamespace(num_scheduled_tokens={'a': 4}), first_hidden)
                self.assertEqual(initial, [[20, 21]])
                self.assertEqual(worker.stats['proposal_batches'], 0)
                self.assertEqual(worker.states['a'].length, 4)
                sampled = [20, 21][:accepted] + [9]
                full = [3, 4, 5, 6, 7] + sampled
                runner.input_batch.num_computed_tokens_cpu[0] = 4
                runner.input_batch.num_tokens_no_spec[0] = len(full)
                runner.input_batch.token_ids_cpu[0, :len(full)] = full
                runner.requests['a'].output_token_ids.extend(sampled)
                hidden = self.t.randn(3, 8)
                with self.t.inference_mode():
                    result = worker.propose([sampled], SimpleNamespace(num_scheduled_tokens={'a': 3}), hidden)
                    state = worker.states['a']
                    expected = self.reference.propose(
                        state.target[:state.length].unsqueeze(0), self.t.tensor([full[:-1]]),
                        self.t.tensor([[9]]), self.embed, self.head, self.rotary, max_heads=3)
                self.assertEqual(result[0], expected[0, 1:].tolist())
                self.assertEqual(worker.stats['proposal_batches'], 1)
                self.assertEqual(state.length, 5 + accepted)
                worker.finish({'a'})
                self.assertNotIn('a', worker.legacy_submitted)

    def test_legacy_rollback_survives_auxiliary_context_cap_fallback(self):
        worker, runner = self.make_worker()
        runner.requests['a'].sampling_params.extra_args['ascend_external_draft'] = [20, 21]
        runner.input_batch.token_ids_cpu = self.t.zeros(1, 64, dtype=self.t.long).numpy()
        runner.input_batch.max_model_len = 64
        runner.input_batch.num_tokens_no_spec[0] = 34
        with self.t.inference_mode():
            result = worker.propose([[7]], SimpleNamespace(num_scheduled_tokens={'a': 33}),
                                   self.t.randn(33, 8))
        self.assertEqual(result, [[20, 21]])
        self.assertEqual(worker.disabled['a'], 'context_cap')
        with self.t.inference_mode():
            self.assertEqual(worker.propose([[7]], SimpleNamespace(num_scheduled_tokens={'a': 1}),
                                            self.t.randn(1, 8)), [[]])
        worker.finish({'a'})
        self.assertFalse(worker.legacy_submitted)

    def test_legacy_stage_waits_for_a_drive_after_chunked_prefill(self):
        worker, runner = self.make_worker()
        runner.requests['a'].sampling_params.extra_args['ascend_external_draft'] = [20, 21]
        runner.requests['a'].output_token_ids = []
        runner.input_batch.num_tokens_no_spec[0] = 4
        first, second = self.t.randn(2, 8), self.t.randn(2, 8)
        with self.t.inference_mode():
            self.assertEqual(worker.propose([[]], SimpleNamespace(num_scheduled_tokens={'a': 2}), first), [[]])
            self.assertNotIn('a', worker.legacy_submitted)
            runner.input_batch.num_computed_tokens_cpu[0] = 2
            runner.input_batch.num_tokens_no_spec[0] = 5
            runner.requests['a'].output_token_ids = [7]
            self.assertEqual(worker.propose([[7]], SimpleNamespace(num_scheduled_tokens={'a': 2}), second), [[20, 21]])
        self.assertEqual(worker.stats['proposal_batches'], 0)
        self.t.testing.assert_close(worker.states['a'].target[:4], self.t.cat((first, second)))

    def test_reordered_batch_partial_acceptance_and_finished_cleanup(self):
        worker, runner = self.make_worker()
        t = self.t
        step = SimpleNamespace(num_scheduled_tokens={'a': 7})
        with t.inference_mode():
            draft = worker.propose([[7]], step, t.randn(7, 8))[0]
            batch = runner.input_batch
            # Next query is previous drive + three drafts; accept one draft,
            # then sample correction 9. Its hidden has NOT been computed yet.
            batch.num_computed_tokens_cpu[0] = 4
            batch.num_tokens_no_spec[0] = 7
            batch.token_ids_cpu[0, :7] = [3, 4, 5, 6, 7, draft[0], 9]
            runner.requests['a'].output_token_ids += [draft[0], 9]
            step.num_scheduled_tokens['a'] = 4
            worker.propose([[draft[0], 9]], step, t.randn(4, 8))
        self.assertEqual(worker.states['a'].length, 6)
        worker.finish({'a'})
        self.assertFalse(worker.states)
        self.assertEqual(len(worker.arena.free), 1)

    def test_missing_hidden_prefix_falls_back_and_never_uses_stale_slot(self):
        worker, runner = self.make_worker()
        runner.input_batch.num_computed_tokens_cpu[0] = 2
        with self.t.inference_mode():
            self.assertEqual(worker.propose([[7]], SimpleNamespace(num_scheduled_tokens={'a': 5}),
                                            self.t.randn(5, 8)), [[]])
        self.assertFalse(worker.states)
        self.assertEqual(worker.disabled['a'], 'missing_hidden_prefix')
        worker.finish({'a'})
        self.assertFalse(worker.disabled)

    def test_profile_uses_preallocated_slots_and_restores_counters(self):
        worker, _ = self.make_worker()
        worker.profile()
        self.assertFalse(worker.arena.leased)
        self.assertEqual(worker.heads.processed_positions, 0)
        self.assertFalse(worker.states)

    def test_request_marker_defaults_to_formal_first_verify_but_can_opt_out(self):
        from asr_mtp_worker import request_enabled
        self.assertFalse(request_enabled(SimpleNamespace(extra_args={})))
        self.assertTrue(request_enabled(SimpleNamespace(extra_args={
            'ascend_external_draft_first_verify': True})))
        self.assertFalse(request_enabled(SimpleNamespace(extra_args={
            'ascend_external_draft_first_verify': True, 'ascend_mtp': False})))

    def test_query_buckets_separate_full_prefill_from_incremental_heads(self):
        from asr_mtp_worker import query_buckets
        new = SimpleNamespace(length=60, heads=[SimpleNamespace(valid=0)] * 3)
        old = SimpleNamespace(length=100, heads=[SimpleNamespace(valid=99),
                             SimpleNamespace(valid=98), SimpleNamespace(valid=97)])
        groups = query_buckets([new, old], [3, 3])
        self.assertEqual(sorted(groups.values()), [[0], [1]])

    def test_remaining_output_eos_and_context_caps_bound_head_work(self):
        for setting in ('one_left', 'none_left', 'eos', 'context'):
            with self.subTest(setting=setting):
                worker, runner = self.make_worker()
                sampled, count = [7], 7
                params = runner.requests['a'].sampling_params
                if setting == 'one_left':
                    params.max_tokens = 2
                elif setting == 'none_left':
                    params.max_tokens = 1
                elif setting == 'eos':
                    sampled = [31]
                else:
                    runner.input_batch.num_tokens_no_spec[0] = 34
                    count = 36
                with self.t.inference_mode():
                    result = worker.propose([sampled], SimpleNamespace(num_scheduled_tokens={'a': count}),
                                            self.t.randn(count, 8))
                self.assertEqual(len(result[0]), 1 if setting == 'one_left' else 0)
                if setting == 'one_left':
                    self.assertEqual(worker.heads.processed_positions, 4)
                if setting == 'context':
                    self.assertEqual(worker.disabled['a'], 'context_cap')

    def test_accepted_eos_followed_by_bonus_does_not_launch_unused_heads(self):
        worker, runner = self.make_worker()
        runner.input_batch.num_tokens_no_spec[0] = 6
        runner.input_batch.token_ids_cpu[0, :6] = [3, 4, 5, 6, 31, 7]
        runner.requests['a'].output_token_ids = [31, 7]
        with self.t.inference_mode():
            result = worker.propose([[31, 7]], SimpleNamespace(num_scheduled_tokens={'a': 7}),
                                    self.t.randn(7, 8))
        self.assertEqual(result, [[]])
        self.assertEqual(worker.stats['proposal_batches'], 0)
        self.assertFalse(worker.arena.leased)

    def test_ignore_eos_and_min_tokens_preserve_nonterminal_mtp(self):
        for setting in ('ignored', 'minimum', 'explicit_stop'):
            with self.subTest(setting=setting):
                worker, runner = self.make_worker()
                params = runner.requests['a'].sampling_params
                params.eos_token_id = None
                if setting == 'ignored':
                    params.ignore_eos = True
                elif setting == 'minimum':
                    params.eos_token_id, params.min_tokens = 31, 3
                else:
                    params.ignore_eos, params.stop_token_ids = True, [31]
                runner.input_batch.num_tokens_no_spec[0] = 6
                runner.input_batch.token_ids_cpu[0, :6] = [3, 4, 5, 6, 31, 7]
                runner.requests['a'].output_token_ids = [31, 7]
                with self.t.inference_mode():
                    result = worker.propose([[31, 7]], SimpleNamespace(num_scheduled_tokens={'a': 7}),
                                            self.t.randn(7, 8))
                self.assertEqual(bool(result[0]), setting != 'explicit_stop')

    def test_request_cache_survives_real_row_reordering_without_hidden_cross_talk(self):
        import numpy as np
        from asr_mtp_heads import CacheArena
        worker, runner = self.make_worker()
        t, batch = self.t, runner.input_batch
        worker.arena = CacheArena(self.config, 3, max_context=32, slots=2,
                                 device='cpu', dtype=t.float32)
        batch.req_ids = ['a', 'b']
        runner.requests['b'] = SimpleNamespace(sampling_params=runner.requests['a'].sampling_params,
                                                output_token_ids=[14])
        batch.num_computed_tokens_cpu = np.array([0, 0])
        batch.num_tokens_no_spec = np.array([5, 5])
        batch.token_ids_cpu = np.zeros((2, 32), dtype=np.int32)
        batch.token_ids_cpu[0, :5] = [3, 4, 5, 6, 7]
        batch.token_ids_cpu[1, :5] = [10, 11, 12, 13, 14]
        runner.query_start_loc.np = np.array([0, 7, 14])
        first = t.randn(14, 8)
        with t.inference_mode():
            proposals = worker.propose([[7], [14]], SimpleNamespace(num_scheduled_tokens={'a': 7, 'b': 7}), first)
            batch.req_ids = ['b', 'a']
            batch.num_computed_tokens_cpu[:] = 4
            batch.num_tokens_no_spec[:] = 7
            batch.token_ids_cpu[0, :7] = [10, 11, 12, 13, 14, proposals[1][0], 9]
            batch.token_ids_cpu[1, :7] = [3, 4, 5, 6, 7, proposals[0][0], 8]
            runner.query_start_loc.np = np.array([0, 4, 8])
            second = t.randn(8, 8)
            worker.propose([[proposals[1][0], 9], [proposals[0][0], 8]],
                           SimpleNamespace(num_scheduled_tokens={'a': 4, 'b': 4}), second)
        t.testing.assert_close(worker.states['a'].target[:4], first[:4])
        t.testing.assert_close(worker.states['b'].target[:4], first[7:11])
        t.testing.assert_close(worker.states['a'].target[4:6], second[4:6])
        t.testing.assert_close(worker.states['b'].target[4:6], second[:2])

    def test_loads_only_auxiliary_checkpoint_and_rejects_wrong_target(self):
        import tempfile
        from asr_mtp_worker import load_worker, base_signature
        t = self.t
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{}', encoding='utf-8')
            checkpoint = root / 'mtp.pt'
            t.save({'metadata': {'base_signature': base_signature(root), 'num_heads': 3},
                    'mtp': self.reference.state_dict()}, checkpoint)
            hf = SimpleNamespace(model_type='qwen3_asr', thinker_config=SimpleNamespace(text_config=self.config))
            config = SimpleNamespace(parallel_config=SimpleNamespace(),
                                     cache_config=SimpleNamespace(enable_prefix_caching=False),
                                     quant_config=None, lora_config=None)
            runner = SimpleNamespace(vllm_config=config, model_config=SimpleNamespace(hf_config=hf, model=str(root)),
                                     use_async_scheduling=False, device='cpu', dtype=t.float32,
                                     speculative_config=SimpleNamespace(num_speculative_tokens=3))
            target = SimpleNamespace(language_model=SimpleNamespace(embed_input_ids=self.embed,
                                                                      compute_logits=self.head))
            settings = {'checkpoint': str(checkpoint), 'num_heads': 2, 'max_context_tokens': 32,
                        'cache_mib': 1, 'max_requests': 2}
            worker = load_worker(runner, target, settings)
            self.assertEqual(worker.heads.num_heads, 2)
            self.assertEqual(worker.arena.storage.shape[0], 2)
            self.assertIs(worker.embedding, self.embed)
            (root / 'config.json').write_text('{"changed":true}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'signature'):
                load_worker(runner, target, settings)
            runner.use_async_scheduling = True
            with self.assertRaisesRegex(ValueError, 'async_scheduling'):
                load_worker(runner, target, settings)

    def test_worker_loop_uses_target_drive_without_an_extra_correction_forward(self):
        import numpy as np
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
        from asr_mtp_heads import BatchedMTPHeads, CacheArena
        from asr_mtp_worker import MTPWorker
        t = self.t
        target = Qwen3Model(self.config).eval()
        prompt = [3, 4, 5, 6]
        with t.inference_mode():
            state = target(input_ids=t.tensor([prompt]), use_cache=True)
            greedy = []
            for _ in range(12):
                token = self.head(state.last_hidden_state[:, -1]).argmax(-1, keepdim=True)
                greedy.append(int(token.item()))
                state = target(input_ids=token, past_key_values=state.past_key_values, use_cache=True)
            for source in ('rollback', 'prespec', 'shadow'):
                for accepted_initial in (0, 1, 3):
                    with self.subTest(source=source, initial_accepted=accepted_initial):
                        batch = SimpleNamespace(req_ids=['a'], num_computed_tokens_cpu=np.array([0]),
                            num_tokens_no_spec=np.array([4]), token_ids_cpu=np.zeros((1, 64), dtype=np.int32),
                            max_model_len=64, spec_decode_unsupported_reqs=set())
                        params = SimpleNamespace(extra_args={'ascend_external_draft_first_verify': True,
                            'source': source}, max_tokens=12, temperature=0.0, n=1)
                        runner = SimpleNamespace(input_batch=batch, requests={'a': SimpleNamespace(
                            sampling_params=params, output_token_ids=[])}, query_start_loc=SimpleNamespace(np=np.array([0, 7])),
                            speculative_config=SimpleNamespace(num_speculative_tokens=3))
                        heads = BatchedMTPHeads(self.config, 3).eval()
                        heads.load_state_dict(self.reference.state_dict())
                        arena = CacheArena(self.config, 3, max_context=64, slots=1,
                                           device='cpu', dtype=t.float32)
                        online = MTPWorker(runner, heads, arena, target.embed_tokens, self.head, {99})
                        external = greedy[1:4] if source == 'shadow' else greedy[:3]
                        if accepted_initial < 3:
                            external[accepted_initial] = (external[accepted_initial] + 1) % 32
                        full = prompt.copy()
                        if source == 'shadow':
                            params.extra_args = {'ascend_mtp': True, 'ascend_external_draft': external}
                            query, drafted = prompt, []
                        else:
                            query, drafted = prompt + external, external
                        past, start = None, 0
                        calls = 0
                        while len(full) - len(prompt) < 12:
                            output = target(input_ids=t.tensor([query]), past_key_values=past, use_cache=True)
                            calls += 1
                            # First verification samples at prompt[-1]; later queries
                            # begin with the previously sampled drive token.
                            offset = len(prompt) - 1 if past is None else 0
                            expected = self.head(output.last_hidden_state[:, offset:]).argmax(-1)[0].tolist()
                            accepted = 0
                            for proposed, actual in zip(drafted, expected):
                                if proposed != actual:
                                    break
                                accepted += 1
                            sampled = drafted[:accepted] + [expected[accepted]]
                            full.extend(sampled)
                            runner.requests['a'].output_token_ids.extend(sampled)
                            batch.num_tokens_no_spec[0] = len(full)
                            batch.token_ids_cpu[0, :len(full)] = full
                            batch.num_computed_tokens_cpu[0] = start
                            step = SimpleNamespace(num_scheduled_tokens={'a': len(query)})
                            drafted = online.propose([sampled], step, output.last_hidden_state[0])[0]
                            past = output.past_key_values
                            past.crop(len(full) - 1)
                            start = len(full) - 1
                            query = [sampled[-1]] + drafted
                        self.assertEqual(full[len(prompt):len(prompt) + 12], greedy)
                        self.assertEqual(online.stats['target_forwards'], 0)
                        self.assertGreater(online.stats['proposal_batches'], 0)
                        self.assertLessEqual(calls, 12)
                        online.finish({'a'})
                        self.assertFalse(online.legacy_submitted)


if __name__ == '__main__':
    unittest.main()
