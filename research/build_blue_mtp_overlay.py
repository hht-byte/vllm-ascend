"""Build a fresh local engine overlay; never patch the installed packages."""
import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from blue_external_draft_port import check_input_batch, dispatch, worker_registry
from blue_scheduler_port import base_scheduler, recompute_scheduler
from build_asr_mtp_patch import check_runner, worker_entry, worker_memory_entry

MODULES = (
    'asr_mtp_heads', 'asr_mtp_fused', 'asr_mtp_qkv_rope', 'asr_mtp_row_copy',
    'asr_mtp_row_copy_kernel', 'asr_mtp_graph', 'asr_mtp_staging',
    'asr_mtp_device_rejection', 'asr_mtp_token_attention', 'asr_mtp_token_graph',
    'asr_mtp_paged_cache', 'asr_mtp_paged_attention', 'asr_mtp_worker', 'asr_mtp_metrics',
)


def restricted_worker(source):
    old = "map_location='cpu', mmap=True, weights_only=False"
    if source.count(old) != 1:
        raise ValueError('unsupported checkpoint loader source')
    return source.replace(old, "map_location='cpu', mmap=True, weights_only=True", 1)


def check_cached_state(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'CachedRequestState')
    if any(isinstance(n, (ast.Assign, ast.AnnAssign)) and
           '__slots__' in ast.unparse(n) for n in cls.body):
        raise ValueError('request state must permit request-owned metadata')
    for decorator in cls.decorator_list:
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                if keyword.arg in ('frozen', 'slots') and ast.literal_eval(keyword.value):
                    raise ValueError('request state must permit request-owned metadata')


def check_output(output, sources):
    output = Path(output).resolve()
    if output.exists() or any(output.is_relative_to(Path(p).resolve()) for p in sources):
        raise ValueError('overlay output must be a fresh directory outside source inputs')


def source_identity(root):
    env=dict(os.environ,GIT_NO_REPLACE_OBJECTS='1',GIT_OPTIONAL_LOCKS='0')
    argv=['git','-c',f'safe.directory={Path(root).resolve().as_posix()}','-C',str(root)]
    def git(*args):
        result=subprocess.run([*argv,*args],capture_output=True,text=True,timeout=20,env=env)
        if result.returncode:raise ValueError('engine source identity unavailable')
        return result.stdout.strip()
    sha=git('rev-parse','HEAD')
    if git('status','--porcelain','--untracked-files=no'):
        raise ValueError('engine source has uncommitted tracked changes')
    return sha


def package_hashes(root):
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for name in ('vllm','vllm_ascend') for p in sorted((Path(root)/name).rglob('*')) if p.is_file()}


def build(vllm_root, ascend_root, mtp_source, output):
    vllm_root, ascend_root, mtp_source, output = map(Path,
        (vllm_root, ascend_root, mtp_source, output))
    tooling = Path(__file__).resolve().parent
    check_output(output, (vllm_root, ascend_root, mtp_source, tooling))
    identities={'vllm':source_identity(vllm_root),'vllm_ascend':source_identity(ascend_root)}
    originals = {}
    changes = {}

    def read(name):
        root = vllm_root if name.startswith('vllm/') else ascend_root
        text = (root/name).read_text(encoding='utf-8').replace('\r\n', '\n')
        originals[name] = text
        return text

    check_runner(read('vllm_ascend/worker/model_runner_v1.py'))
    check_input_batch(read('vllm_ascend/worker/npu_input_batch.py'))
    check_cached_state(read('vllm/v1/worker/gpu_input_batch.py'))
    changes['vllm/v1/core/sched/scheduler.py'] = base_scheduler(read('vllm/v1/core/sched/scheduler.py'))
    changes['vllm_ascend/core/recompute_scheduler.py'] = recompute_scheduler(read('vllm_ascend/core/recompute_scheduler.py'))
    changes['vllm_ascend/spec_decode/__init__.py'] = dispatch(read('vllm_ascend/spec_decode/__init__.py'))
    changes['vllm_ascend/patch/worker/__init__.py'] = worker_registry(read('vllm_ascend/patch/worker/__init__.py'))
    changes['vllm_ascend/worker/worker.py'] = worker_memory_entry(worker_entry(read('vllm_ascend/worker/worker.py')))+(
        '\nfrom vllm_ascend.patch.worker.blue_worker_probe import install as _install_blue_probe\n'
        '_install_blue_probe(NPUWorker)\n')
    for name in MODULES:
        text = (mtp_source/(name+'.py')).read_text(encoding='utf-8').replace('\r\n', '\n')
        changes['vllm_ascend/spec_decode/'+name+'.py'] = restricted_worker(text) if name == 'asr_mtp_worker' else text
    for source, destination in (
        (tooling/'blue_external_draft.py', 'vllm_ascend/spec_decode/external_draft_proposer.py'),
        (tooling/'blue_external_draft_transport.py', 'vllm_ascend/patch/worker/patch_external_draft.py'),
        (tooling/'external_draft_rejoin.py', 'vllm_ascend/spec_decode/external_draft_rejoin.py'),
        (tooling/'patch_external_draft_first_verify.py', 'vllm_ascend/patch/worker/patch_external_draft_first_verify.py'),
        (mtp_source/'patch_external_draft_mtp.py', 'vllm_ascend/patch/worker/patch_external_draft_mtp.py'),
        (tooling/'blue_worker_probe.py', 'vllm_ascend/patch/worker/blue_worker_probe.py'),
        (tooling/'blue_batch_invariant.py', 'vllm_ascend/patch/worker/blue_batch_invariant.py'),
    ):
        changes[destination] = source.read_text(encoding='utf-8').replace('\r\n', '\n')
    for name, text in changes.items():
        ast.parse(text, filename=name)
    # All bounded source checks complete before any output or copied byte exists.
    output.mkdir(parents=True)
    for root, name in ((vllm_root, 'vllm'), (ascend_root, 'vllm_ascend')):
        shutil.copytree(root/name, output/name, ignore=shutil.ignore_patterns(
            '__pycache__', '*.pyc', '.git', '*.log'))
    patch = []
    for name, text in changes.items():
        target = output/name
        old = originals.get(name)
        if old is None:
            old = target.read_text(encoding='utf-8').replace('\r\n', '\n') if target.exists() else ''
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode('utf-8'))
        patch.extend(difflib.unified_diff(old.splitlines(True), text.splitlines(True),
            fromfile='a/'+name if old else '/dev/null', tofile='b/'+name))
    (output/'engine.patch').write_bytes(''.join(patch).encode('utf-8'))
    if identities!={'vllm':source_identity(vllm_root),'vllm_ascend':source_identity(ascend_root)}:
        raise ValueError('engine source identity changed during copy')
    manifest = {'schema_version':1, 'runtime_sources':identities,
        'shared_adjustments':['bounded_external_draft_transport','first_verify_scheduler_port',
                              'restricted_checkpoint_loading'],
        'changed_files':sorted(changes),'files':package_hashes(output)}
    (output/'overlay.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('vllm-root','ascend-root','mtp-source','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build(args.vllm_root, args.ascend_root, args.mtp_source, args.output)
    except Exception as exc:
        print(json.dumps({'status':'failed','error_type':type(exc).__name__}))
        return 1
    print(json.dumps({'status':'prepared','changed_files':len(result['changed_files']),'pinned_files':len(result['files'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
