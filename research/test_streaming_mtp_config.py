"""Service master switch must control engine loading as well as request flags."""
import argparse
import unittest
from copy import deepcopy

from streaming_mtp_config import add_mtp_args, mtp_engine_config


class MTPServiceConfigTests(unittest.TestCase):
    def test_graph_switch_reaches_engine_without_changing_master_or_base(self):
        parser = argparse.ArgumentParser()
        add_mtp_args(parser)
        args = parser.parse_args(['--mtp', '--mtp-checkpoint', '/heads.pt',
                                  '--mtp-graph-mode', 'full'])
        base = self.base()
        saved = deepcopy(base)
        result = mtp_engine_config(base, mtp_enabled=args.mtp_enabled,
            mtp_checkpoint=args.mtp_checkpoint, mtp_graph_mode=args.mtp_graph_mode)
        self.assertEqual(result['additional_config']['asr_mtp']['graph_mode'], 'full')
        self.assertEqual(base, saved)
        self.assertNotIn('mtp_graph_mode', result)
        result = mtp_engine_config(result, mtp_enabled=True, mtp_graph_mode='off')
        self.assertEqual(result['additional_config']['asr_mtp']['graph_mode'], 'off')
        disabled = mtp_engine_config(result, mtp_enabled=False, mtp_graph_mode='full')
        self.assertNotIn('asr_mtp', disabled['additional_config'])

    def base(self):
        return dict(
            model='/target', max_num_seqs=20,
            speculative_config=dict(
                method='custom_class', num_speculative_tokens=15,
                model='vllm_ascend.spec_decode.external_draft_proposer.AscendExternalDraftProposer'),
            additional_config={'other_extension': {'keep': 1}},
        )

    def test_cli_defaults_and_independent_shadow_flag(self):
        parser = argparse.ArgumentParser()
        add_mtp_args(parser)
        for argv, expected in (([], (False, True)), (['--mtp'], (True, True)),
                               (['--mtp', '--no-prespec-mtp'], (True, False)),
                               (['--no-mtp', '--prespec-mtp'], (False, True))):
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                self.assertEqual((args.mtp_enabled, args.prespec_mtp_enabled), expected)

    def test_off_removes_only_mtp_engine_config_without_mutating_input(self):
        base = self.base()
        base['additional_config']['asr_mtp'] = {'checkpoint': '/old.pt'}
        original = deepcopy(base)
        result = mtp_engine_config(base, mtp_enabled=False)
        self.assertNotIn('asr_mtp', result['additional_config'])
        self.assertEqual(result['additional_config']['other_extension'], {'keep': 1})
        self.assertEqual(result['speculative_config'], base['speculative_config'])
        self.assertEqual(base, original)
        for name in ('mtp_enabled', 'prespec_mtp_enabled', 'mtp_checkpoint'):
            self.assertNotIn(name, result)

    def test_on_adds_checkpoint_and_sync_settings_without_mutating_input(self):
        base, parser = self.base(), argparse.ArgumentParser()
        original = deepcopy(base)
        add_mtp_args(parser)
        args = parser.parse_args(['--mtp', '--mtp-checkpoint', '/heads.pt'])
        result = mtp_engine_config(
            base, mtp_enabled=args.mtp_enabled, mtp_checkpoint=args.mtp_checkpoint)
        self.assertEqual(result['additional_config']['asr_mtp']['checkpoint'], '/heads.pt')
        self.assertFalse(result['async_scheduling'])
        self.assertFalse(result['enable_prefix_caching'])
        self.assertEqual(base, original)

    def test_explicit_engine_head_settings_are_preserved(self):
        base = self.base()
        base['additional_config']['asr_mtp'] = {
            'checkpoint': '/old.pt', 'num_heads': 3, 'cache_mib': 128,
            'max_requests': 20, 'max_context_tokens': 256,
            'verify_base_signature': True,
        }
        result = mtp_engine_config(base, mtp_enabled=True)
        self.assertEqual(result['additional_config']['asr_mtp'],
                         base['additional_config']['asr_mtp'])
        updated = mtp_engine_config(base, mtp_enabled=True, mtp_checkpoint='/new.pt')
        self.assertEqual(updated['additional_config']['asr_mtp']['checkpoint'], '/new.pt')
        self.assertEqual(updated['additional_config']['asr_mtp']['num_heads'], 3)

    def test_on_requires_checkpoint_and_external_draft_backend(self):
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            mtp_engine_config(self.base(), mtp_enabled=True)
        for spec in (None, {'method': 'mtp'}, {'method': 'custom_class', 'model': 'other'}):
            base = self.base()
            base['speculative_config'] = spec
            with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, 'external_draft'):
                mtp_engine_config(base, mtp_enabled=True, mtp_checkpoint='/heads.pt')

    def test_off_is_valid_without_engine_speculation_or_checkpoint(self):
        result = mtp_engine_config({'model': '/target'}, mtp_enabled=False)
        self.assertEqual(result, {'model': '/target'})

    def test_invalid_master_and_incompatible_engine_settings_fail_at_startup(self):
        for value in ('false', 0, 1, None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                mtp_engine_config(self.base(), mtp_enabled=value)
        for key in ('async_scheduling', 'enable_prefix_caching'):
            base = self.base()
            base[key] = True
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                mtp_engine_config(base, mtp_enabled=True, mtp_checkpoint='/heads.pt')


if __name__ == '__main__':
    unittest.main()
