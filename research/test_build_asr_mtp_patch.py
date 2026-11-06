import ast
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class PatchExportTests(unittest.TestCase):
    def test_registry_removes_eager_mtp_import_and_is_idempotent(self):
        from build_asr_mtp_patch import registry
        source = ('if vllm_version_is("0.23.0"):\n'
                  '    import vllm_ascend.patch.worker.patch_external_draft  # noqa\n'
                  '    import vllm_ascend.patch.worker.patch_external_draft_first_verify  # noqa\n')
        result = registry(source + '    import vllm_ascend.patch.worker.patch_external_draft_mtp  # noqa\n')
        self.assertEqual(result, source)
        self.assertNotIn('patch_external_draft_mtp', result)
        self.assertEqual(registry(result), result)
        ast.parse(result)
        with self.assertRaisesRegex(ValueError, 'first-verify'):
            registry('')

    def test_export_contains_three_modules_and_late_worker_import(self):
        from build_asr_mtp_patch import make_patch
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            entry = root / 'vllm_ascend/patch/worker/__init__.py'
            entry.parent.mkdir(parents=True)
            entry.write_text('if True:\n    import vllm_ascend.patch.worker.patch_external_draft_first_verify  # noqa\n', encoding='utf-8')
            runner = root / 'vllm_ascend/worker/model_runner_v1.py'
            runner.parent.mkdir(parents=True)
            worker = root / 'vllm_ascend/worker/worker.py'
            worker.write_text(
                'from vllm_ascend.utils import vllm_version_is\n'
                'from vllm_ascend.worker.model_runner_v1 import NPUModelRunner\n',
                encoding='utf-8')
            fixture = Path(__file__).with_name('test_patch_external_draft_mtp.py').read_text(encoding='utf-8')
            tree = ast.parse(fixture)
            cls = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == 'Runner')
            cls.body.append(ast.parse('def _bookkeeping_sync(self):\n self.input_batch.num_tokens_no_spec[req_idx] = end_idx').body[0])
            cls.name = 'NPUModelRunner'
            runner.write_text(ast.unparse(cls), encoding='utf-8')
            # Export checks the real call order, so include a representative sample method.
            with runner.open('a', encoding='utf-8') as stream:
                stream.write('\n    def sample_tokens(self):\n        self._bookkeeping_sync()\n        propose_draft_token_ids(valid_sampled_token_ids)\n')
            patch = make_patch(root)
            self.assertEqual(patch.count('--- /dev/null'), 3)
            self.assertNotIn('+++ b/vllm_ascend/worker/model_runner_v1.py', patch)
            self.assertIn('+++ b/vllm_ascend/worker/worker.py', patch)
            self.assertNotIn('+++ b/vllm_ascend/patch/worker/__init__.py', patch)

    def test_late_worker_import_is_version_gated_and_idempotent(self):
        from build_asr_mtp_patch import worker_entry
        source = ('from vllm_ascend.utils import vllm_version_is\n'
                  'from vllm_ascend.worker.model_runner_v1 import NPUModelRunner\n'
                  'class NPUWorker: pass\n')
        result = worker_entry(source)
        self.assertEqual(worker_entry(result), result)
        self.assertLess(result.index('from vllm_ascend.worker.model_runner_v1'),
                        result.index('import vllm_ascend.patch.worker.patch_external_draft_mtp'))
        self.assertIn('if vllm_version_is("0.23.0"):', result)
        ast.parse(result)
        with self.assertRaisesRegex(ValueError, 'worker'):
            worker_entry('')
        with self.assertRaisesRegex(ValueError, 'vllm_version_is'):
            worker_entry('from vllm_ascend.worker.model_runner_v1 import NPUModelRunner\n')

    def test_real_import_order_no_longer_reads_partially_initialized_runner(self):
        from build_asr_mtp_patch import registry, worker_entry
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            package = root / 'vllm_ascend'
            for relative in ('', 'patch', 'patch/worker', 'worker', 'spec_decode'):
                folder = package / relative
                folder.mkdir(parents=True, exist_ok=True)
                (folder / '__init__.py').write_text('', encoding='utf-8')
            (package / 'utils.py').write_text(
                'def vllm_version_is(version): return version == "0.23.0"\n', encoding='utf-8')
            entry = ('if True:\n'
                     '    import vllm_ascend.patch.worker.patch_external_draft_first_verify  # noqa\n'
                     '    import vllm_ascend.patch.worker.patch_external_draft_mtp  # noqa\n')
            (package / 'patch/worker/__init__.py').write_text(registry(entry), encoding='utf-8')
            (package / 'patch/worker/patch_external_draft_first_verify.py').write_text('', encoding='utf-8')
            (package / 'patch/worker/patch_external_draft_mtp.py').write_text(
                Path(__file__).with_name('patch_external_draft_mtp.py').read_text(encoding='utf-8'),
                encoding='utf-8')
            fixture = ast.parse(Path(__file__).with_name('test_patch_external_draft_mtp.py').read_text(encoding='utf-8'))
            cls = next(node for node in ast.walk(fixture)
                       if isinstance(node, ast.ClassDef) and node.name == 'Runner')
            cls.name = 'NPUModelRunner'
            runner_source = ('from vllm_ascend.patch.worker import patch_external_draft_first_verify\n'
                             '# The class is defined only after the patch package has loaded.\n'
                             + ast.unparse(cls) + '\n')
            (package / 'worker/model_runner_v1.py').write_text(runner_source, encoding='utf-8')
            (package / 'worker/worker.py').write_text(worker_entry(
                'from vllm_ascend.utils import vllm_version_is\n'
                'from vllm_ascend.worker.model_runner_v1 import NPUModelRunner\n'), encoding='utf-8')
            external = ModuleType('vllm_ascend.spec_decode.external_draft_proposer')
            class Base:
                def __init__(self, config, runner):
                    self.runner = runner
            external.AscendExternalDraftProposer = Base
            mtp = ModuleType('vllm_ascend.spec_decode.asr_mtp_worker')
            mtp.load_worker = lambda *args: None
            mtp.request_enabled = lambda params: False
            # Use actual import machinery, not exec() with an already-defined runner.
            for supported in (True, False):
                with self.subTest(supported=supported), patch.dict(sys.modules):
                    for name in tuple(sys.modules):
                        if name == 'vllm_ascend' or name.startswith('vllm_ascend.'):
                            del sys.modules[name]
                    utils = ModuleType('vllm_ascend.utils')
                    utils.vllm_version_is = lambda version: supported
                    sys.modules.update({external.__name__: external, mtp.__name__: mtp,
                                        utils.__name__: utils})
                    with patch.object(sys, 'path', [str(root), *sys.path]):
                        importlib.invalidate_caches()
                        module = importlib.import_module('vllm_ascend.worker.worker')
                        self.assertEqual(getattr(module.NPUModelRunner, '_asr_mtp_patch_installed', False), supported)
                        self.assertEqual('vllm_ascend.patch.worker.patch_external_draft_mtp' in sys.modules,
                                         supported)
                        runner = module.NPUModelRunner()
                        runner.vllm_config = SimpleNamespace(additional_config={})
                        self.assertEqual(runner._get_drafter(), 'original')
                        if supported:
                            registration = sys.modules['vllm_ascend.patch.worker.patch_external_draft_mtp']
                            runner.vllm_config.additional_config['asr_mtp'] = {'checkpoint': '/heads.pt'}
                            runner.speculative_config = SimpleNamespace(
                                method='custom_class', model=registration.EXTERNAL_CLASS)
                            self.assertIsInstance(runner._get_drafter(), registration.AscendExternalDraftMTPProposer)

    def test_real_runner_contract_is_supported(self):
        from build_asr_mtp_patch import check_runner
        path = Path('E:/26B-multimodal/opencode-project/asr-optimize/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')
        if not path.exists():
            self.skipTest('local engine reference unavailable')
        check_runner(path.read_text(encoding='utf-8'))

    def test_export_rejects_missing_or_changed_graph_setup_api(self):
        from build_asr_mtp_patch import check_runner
        fixture = ast.parse(Path(__file__).with_name('test_patch_external_draft_mtp.py').read_text(encoding='utf-8'))
        cls = next(node for node in ast.walk(fixture)
                   if isinstance(node, ast.ClassDef) and node.name == 'Runner')
        cls.name = 'NPUModelRunner'
        cls.body.extend(ast.parse(
            'def _bookkeeping_sync(self):\n self.input_batch.num_tokens_no_spec[req_idx] = end_idx\n'
            'def sample_tokens(self):\n self._bookkeeping_sync()\n propose_draft_token_ids(tokens)\n'
        ).body)
        setup = next(node for node in cls.body if node.name == '_check_and_update_cudagraph_mode')
        setup.args.args.append(ast.arg(arg='new_required_argument'))
        with self.assertRaisesRegex(ValueError, 'graph setup API'):
            check_runner(ast.unparse(cls))
        cls.body.remove(setup)
        with self.assertRaisesRegex(ValueError, 'graph setup API'):
            check_runner(ast.unparse(cls))

    def test_exported_release_modules_are_current(self):
        directory = Path(__file__).resolve().parent
        lines = (directory / 'asr_mtp_v023.patch').read_text(encoding='utf-8').splitlines(keepends=True)
        self.assertNotIn(' \n', lines)
        for name in ('asr_mtp_heads.py', 'asr_mtp_worker.py', 'patch_external_draft_mtp.py'):
            start = next(i for i, line in enumerate(lines) if line.startswith('+++ b/') and line.rstrip().endswith('/' + name))
            section = []
            for line in lines[start + 1:]:
                if line.startswith('--- '):
                    break
                if line.startswith('+'):
                    section.append(line[1:])
            self.assertEqual(''.join(section), (directory / name).read_text(encoding='utf-8'), name)


if __name__ == '__main__':
    unittest.main()
