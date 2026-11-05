import ast
from pathlib import Path
import tempfile
import unittest


class PatchExportTests(unittest.TestCase):
    def test_registry_import_is_after_first_verify_and_idempotent(self):
        from build_asr_mtp_patch import registry
        source = ('if vllm_version_is("0.23.0"):\n'
                  '    import vllm_ascend.patch.worker.patch_external_draft  # noqa\n'
                  '    import vllm_ascend.patch.worker.patch_external_draft_first_verify  # noqa\n')
        result = registry(source)
        self.assertTrue(result.endswith('    import vllm_ascend.patch.worker.patch_external_draft_mtp  # noqa\n'))
        self.assertEqual(registry(result), result)
        ast.parse(result)
        with self.assertRaisesRegex(ValueError, 'first-verify'):
            registry('')

    def test_export_contains_only_three_modules_and_registry_import(self):
        from build_asr_mtp_patch import make_patch
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            entry = root / 'vllm_ascend/patch/worker/__init__.py'
            entry.parent.mkdir(parents=True)
            entry.write_text('if True:\n    import vllm_ascend.patch.worker.patch_external_draft_first_verify  # noqa\n', encoding='utf-8')
            runner = root / 'vllm_ascend/worker/model_runner_v1.py'
            runner.parent.mkdir(parents=True)
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
            self.assertIn('+++ b/vllm_ascend/patch/worker/__init__.py', patch)

    def test_real_runner_contract_is_supported(self):
        from build_asr_mtp_patch import check_runner
        path = Path('E:/26B-multimodal/opencode-project/asr-optimize/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')
        if not path.exists():
            self.skipTest('local engine reference unavailable')
        check_runner(path.read_text(encoding='utf-8'))

    def test_exported_release_modules_are_current(self):
        directory = Path(__file__).resolve().parent
        lines = (directory / 'asr_mtp_v023.patch').read_text(encoding='utf-8').splitlines(keepends=True)
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
