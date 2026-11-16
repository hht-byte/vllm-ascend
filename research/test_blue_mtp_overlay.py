"""Source/path boundaries for the isolated engine overlay builder."""
import tempfile
import unittest
from pathlib import Path
import subprocess


class OverlayBoundaryTests(unittest.TestCase):
    def test_checkpoint_loader_is_restricted_and_unknown_source_rejected(self):
        from build_blue_mtp_overlay import restricted_worker
        source = "saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)\n"
        self.assertIn('weights_only=True', restricted_worker(source))
        with self.assertRaises(ValueError):
            restricted_worker('saved = unsafe_load(path)\n')

    def test_cached_request_state_requires_dynamic_attributes(self):
        from build_blue_mtp_overlay import check_cached_state
        check_cached_state('@dataclass\nclass CachedRequestState:\n    req_id: str\n')
        for decorator in ('@dataclass(slots=True)', '@dataclass(frozen=True)'):
            with self.assertRaises(ValueError):
                check_cached_state(decorator+'\nclass CachedRequestState:\n    req_id: str\n')

    def test_output_never_replaces_source_or_prior_attempt(self):
        from build_blue_mtp_overlay import check_output
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root/'engine'; package.mkdir()
            (package/'__init__.py').write_text('ORIGINAL', encoding='utf-8')
            for path in (package, package/'nested'):
                with self.assertRaises(ValueError):
                    check_output(path, [package])
            prior = root/'attempt'; prior.mkdir()
            with self.assertRaises(ValueError):
                check_output(prior, [package])
            check_output(root/'fresh-attempt', [package])
            self.assertEqual((package/'__init__.py').read_text(), 'ORIGINAL')

    def test_manifest_pins_unmodified_copied_python_and_native_modules(self):
        from build_blue_mtp_overlay import package_hashes
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'vllm').mkdir();(p/'vllm_ascend').mkdir()
            (p/'vllm/__init__.py').write_text('BASE')
            (p/'vllm_ascend/_C.so').write_bytes(b'COMPILED')
            (p/'overlay.json').write_text('NOT AN INPUT')
            self.assertEqual(set(package_hashes(p)),{'vllm/__init__.py','vllm_ascend/_C.so'})

    def test_dirty_engine_cannot_claim_its_clean_commit(self):
        from build_blue_mtp_overlay import source_identity
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            def git(*args):
                return subprocess.run(['git','-c',f'safe.directory={p.as_posix()}',
                    '-c','user.name=Test','-c','user.email=test@localhost',*args],cwd=p,
                    check=True,capture_output=True,text=True).stdout.strip()
            git('init','--initial-branch=test');(p/'engine.py').write_text('clean')
            git('add','engine.py');git('commit','-m','fixture')
            self.assertEqual(source_identity(p),git('rev-parse','HEAD'))
            (p/'engine.py').write_text('changed')
            with self.assertRaises(ValueError):source_identity(p)
