"""Isolation launcher input boundaries; real namespace probe is separate."""
import tempfile
import unittest
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


class IsolationBoundaryTests(unittest.TestCase):
    def test_environment_drops_credentials_and_keeps_only_runtime_inputs(self):
        from blue_isolated_run import child_environment
        env = child_environment({'PATH':'/usr/bin','LD_LIBRARY_PATH':'/usr/lib',
            'HF_TOKEN':'PRIVATE','GITHUB_TOKEN':'PRIVATE','SSH_AUTH_SOCK':'PRIVATE'}, 6)
        self.assertNotIn('HF_TOKEN', env)
        self.assertNotIn('GITHUB_TOKEN', env)
        self.assertNotIn('SSH_AUTH_SOCK', env)
        self.assertEqual(env['ASCEND_RT_VISIBLE_DEVICES'], '0')
        self.assertEqual(env['LLM_OPT_PHYSICAL_NPU_ID'], '6')
        self.assertEqual(env['PYTHONPATH'], '/overlay:/tools')
        self.assertEqual(env['VLLM_HOST_IP'], '127.0.0.1')

    def test_registered_vendor_python_paths_survive_without_host_project_paths(self):
        from blue_isolated_run import child_environment
        env = child_environment({'PYTHONPATH':'/usr/local/Ascend/cann/python/site-packages:/home/private-code'},6)
        self.assertEqual(env['PYTHONPATH'], '/overlay:/tools:/usr/local/Ascend/cann/python/site-packages')

    def test_existing_root_or_missing_mounts_fail_before_launch(self):
        from blue_isolated_run import validate
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            cfg = {k:str(p/k) for k in ('sandbox_root','overlay','tools','model','assets','results')}
            cfg.update(physical_device=6, command=['python3','/tools/probe.py'])
            for k in ('overlay','tools','model','assets','results'):
                Path(cfg[k]).mkdir()
            validate(cfg)
            Path(cfg['sandbox_root']).mkdir()
            with self.assertRaises(ValueError): validate(cfg)
            cfg['sandbox_root'] = str(p/'new-root')
            cfg['model'] = str(p/'missing')
            with self.assertRaises(ValueError): validate(cfg)

    def test_writable_results_cannot_alias_or_contain_readonly_inputs(self):
        from blue_isolated_run import validate
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            cfg = {k:str(p/k) for k in ('sandbox_root','overlay','tools','model','assets','results')}
            cfg.update(physical_device=6, command=['python3','/tools/probe.py'])
            for k in ('overlay','tools','model','assets','results'):
                Path(cfg[k]).mkdir()
            for key in ('overlay','tools','model','assets'):
                source = Path(cfg[key])
                nested = source/'writable'
                nested.mkdir()
                for results in (source, nested, source.parent):
                    with self.subTest(input=key, results=results):
                        attempt = dict(cfg, results=str(results))
                        with self.assertRaisesRegex(ValueError, 'results.*read-only'):
                            validate(attempt)
            self.assertFalse(Path(cfg['sandbox_root']).exists())

    def test_writable_results_cannot_contain_checkpoint_mount(self):
        from blue_isolated_run import validate
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            cfg = {k:str(p/k) for k in ('sandbox_root','overlay','tools','model','assets','results')}
            cfg.update(physical_device=6, command=['python3','/tools/probe.py'])
            for k in ('overlay','tools','model','assets','results'):
                Path(cfg[k]).mkdir()
            checkpoint = Path(cfg['results'])/'mtp.pt'
            checkpoint.write_bytes(b'fixture')
            with self.assertRaisesRegex(ValueError, 'results.*read-only'):
                validate(dict(cfg, checkpoint=str(checkpoint)))
            checkpoint.unlink()
            checkpoint = p/'weights.pt'
            checkpoint.write_bytes(b'fixture')
            validate(dict(cfg, checkpoint=str(checkpoint)))

    def test_namespace_command_requires_parent_death_child_kill(self):
        import blue_isolated_run
        factory = getattr(blue_isolated_run, 'namespace_command', None)
        self.assertIsNotNone(factory, 'namespace launch must bind child cleanup to parent lifetime')
        command = factory(['python3', '/tools/probe.py'])
        self.assertIn('--kill-child=SIGKILL', command)
        self.assertEqual(command[-2:], ['python3', '/tools/probe.py'])

    @unittest.skipUnless(sys.platform == 'linux', 'requires Linux namespace support')
    def test_namespace_child_stops_when_launcher_parent_is_killed(self):
        from blue_isolated_run import namespace_command
        env = dict(os.environ, LC_ALL='C')
        try:
            capability = subprocess.run(namespace_command([sys.executable, '-c', 'pass']),
                capture_output=True, text=True, timeout=5, env=env)
        except FileNotFoundError:
            self.skipTest('Linux namespace tool unavailable')
        if capability.returncode and any(reason in capability.stderr for reason in (
                'Operation not permitted', 'Permission denied')):
            self.skipTest('container does not permit namespace creation')
        self.assertEqual(capability.returncode, 0, capability.stderr)

        def running(pid):
            try:
                return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
            except FileNotFoundError:
                return False

        with tempfile.TemporaryDirectory() as temp:
            ready = Path(temp)/'ready'
            payload = ('from pathlib import Path; import time; '
                       f'Path({str(ready)!r}).write_text("ready"); time.sleep(30)')
            parent = subprocess.Popen(namespace_command([sys.executable, '-c', payload]),
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
            child = None
            try:
                deadline = time.monotonic()+3
                while not ready.exists() and parent.poll() is None and time.monotonic() < deadline:
                    children = Path(f'/proc/{parent.pid}/task/{parent.pid}/children').read_text().split()
                    if len(children) == 1:
                        child = int(children[0])
                    time.sleep(0.02)
                self.assertTrue(ready.exists(), 'namespace child did not become ready')
                children = Path(f'/proc/{parent.pid}/task/{parent.pid}/children').read_text().split()
                self.assertEqual(len(children), 1)
                child = int(children[0])
                self.assertTrue(running(child))
                parent.kill()
                parent.wait(timeout=3)
                deadline = time.monotonic()+3
                while running(child) and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertFalse(running(child), 'orphaned namespace child survived launcher death')
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait(timeout=3)
                if child is not None and running(child):
                    os.kill(child, signal.SIGKILL)
                if parent.stderr is not None:
                    parent.stderr.close()
