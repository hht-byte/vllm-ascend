"""Execute transformed pinned dispatch/worker registration without NPU imports."""
import ast
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_blue_external_draft import ROOT, SNAPSHOT, batch, engine_modules, load_modules


class TransportPortTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'blue_external_draft_port.py'
        self.assertTrue(path.exists(), 'pristine-engine transport port is missing')
        spec = importlib.util.spec_from_file_location('blue_external_port', path)
        self.port = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.port)
        if not (SNAPSHOT / 'vllm_ascend/spec_decode/__init__.py').exists():
            self.skipTest('pinned engine snapshot is unavailable')

    def test_actual_dispatch_preserves_ngram_and_restricts_custom_class(self):
        source = (SNAPSHOT / 'vllm_ascend/spec_decode/__init__.py').read_text(encoding='utf-8')
        transformed = self.port.dispatch(source)
        self.assertEqual(self.port.dispatch(transformed), transformed)
        function = next(n for n in ast.parse(transformed).body if isinstance(n, ast.FunctionDef)
                        and n.name == 'get_spec_decode_method')
        external, ngram = load_modules()
        def original_ngram(config, runner):
            return ('original-ngram', config, runner)
        namespace = {'AscendNgramProposer': original_ngram}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'actual-dispatch', 'exec'), namespace)
        config = SimpleNamespace(speculative_config=SimpleNamespace(
            num_speculative_tokens=4, model=external.EXTERNAL_CLASS))
        runner = SimpleNamespace(input_batch=batch(), use_async_scheduling=False)
        get = namespace['get_spec_decode_method']
        self.assertEqual(get('ngram', config, None, runner), ('original-ngram', config, runner))
        with patch.dict(sys.modules, engine_modules(external)):
            proposer = get('custom_class', config, None, runner)
            self.assertIsInstance(proposer, ngram.AscendNgramProposer)
            self.assertEqual(proposer.propose([[9]]), [[2, 3, 4]])
            config.speculative_config.model = 'untrusted.module.ArbitraryCode'
            with self.assertRaisesRegex(ValueError, 'external'):
                get('custom_class', config, None, runner)

    def test_worker_registration_imports_base_before_first_verify_only_on_v023(self):
        source = (SNAPSHOT / 'vllm_ascend/patch/worker/__init__.py').read_text(encoding='utf-8')
        transformed = self.port.worker_registry(source)
        self.assertEqual(self.port.worker_registry(transformed), transformed)
        # Run the added registration block via real Python package imports.
        old_lines = len(source.splitlines())
        tree = ast.parse(transformed)
        additions = [n for n in tree.body if n.lineno > old_lines]
        code = ast.unparse(ast.Module(body=additions, type_ignores=[]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'vllm_ascend/patch/worker'
            package.mkdir(parents=True)
            for folder in (root / 'vllm_ascend', root / 'vllm_ascend/patch', package):
                (folder / '__init__.py').write_text('', encoding='utf-8')
            (package / 'patch_external_draft.py').write_text(
                "from pathlib import Path\nPath('events').write_text('base\\n')\n", encoding='utf-8')
            (package / 'patch_external_draft_first_verify.py').write_text(
                "from pathlib import Path\np = Path('events')\np.write_text(p.read_text() + 'first\\n')\n", encoding='utf-8')
            for enabled, expected in ((True, 'base\nfirst\n'), (False, '')):
                (root / 'events').unlink(missing_ok=True)
                run = subprocess.run([sys.executable, '-c',
                    f'vllm_version_is = lambda version: {enabled!r}\n' + code], cwd=root,
                    text=True, capture_output=True, check=False)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual((root / 'events').read_text() if (root / 'events').exists() else '', expected)

    def test_transform_rejects_unknown_dispatch_and_input_batch_contracts(self):
        with self.assertRaisesRegex(ValueError, 'dispatch'):
            self.port.dispatch('def get_spec_decode_method(method):\n    return None\n')
        actual = (SNAPSHOT / 'vllm_ascend/worker/npu_input_batch.py').read_text(encoding='utf-8')
        self.port.check_input_batch(actual)
        altered = actual.replace('num_speculative_tokens: int = 0,', '')
        with self.assertRaisesRegex(ValueError, 'InputBatch'):
            self.port.check_input_batch(altered)


if __name__ == '__main__':
    unittest.main()
