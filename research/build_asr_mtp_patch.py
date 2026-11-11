"""Export online MTP modules plus a late worker import; never edit the engine.

Apply on top of external_draft_first_verify_v023.patch in the user's engine
checkout. Source anchors/API and post-bookkeeping call order are checked first.
"""
import argparse
import ast
import difflib
from pathlib import Path


def registry(source):
    obsolete = '    import vllm_ascend.patch.worker.patch_external_draft_mtp  # noqa\n'
    anchor = '    import vllm_ascend.patch.worker.patch_external_draft_first_verify  # noqa\n'
    if source.count(anchor) != 1:
        raise ValueError('apply the existing first-verify worker patch before the MTP extension')
    if source.count(obsolete) > 1:
        raise ValueError('ambiguous old MTP worker registry imports')
    # This package is imported by model_runner_v1 before its class definition.
    # Undo the old registration location when upgrading an installed patch.
    return source.replace(obsolete, '', 1)


def worker_entry(source):
    anchor = 'from vllm_ascend.worker.model_runner_v1 import NPUModelRunner\n'
    block = ('\n# Install optional ASR MTP hooks only after the runner is fully imported.\n'
             'if vllm_version_is("0.23.0"):\n'
             '    import vllm_ascend.patch.worker.patch_external_draft_mtp  # noqa\n')
    if source.count(anchor) != 1:
        raise ValueError('unsupported worker entry: expected one NPUModelRunner import')
    imports_version = any(
        isinstance(node, ast.ImportFrom) and node.module == 'vllm_ascend.utils'
        and any((alias.asname or alias.name) == 'vllm_version_is' for alias in node.names)
        for node in ast.walk(ast.parse(source))
    )
    if not imports_version:
        raise ValueError('unsupported worker entry: missing vllm_version_is import')
    if source.count(block) == 1 and source.index(block) > source.index(anchor):
        return source
    if 'import vllm_ascend.patch.worker.patch_external_draft_mtp' in source:
        raise ValueError('unsupported worker entry: ambiguous MTP registration')
    return source.replace(anchor, anchor + block, 1)


def check_runner(source):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'NPUModelRunner')
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    required = ('_get_drafter', '_update_states', 'propose_draft_token_ids', '_bookkeeping_sync', 'sample_tokens')
    if any(name not in methods for name in required):
        raise ValueError('unsupported runner: missing MTP hook or bookkeeping method')
    graph_setup = methods.get('_check_and_update_cudagraph_mode')
    if (graph_setup is None or tuple(arg.arg for arg in graph_setup.args.args) !=
            ('self', 'attention_backends', 'kv_cache_groups') or graph_setup.args.kwonlyargs or
            graph_setup.args.vararg or graph_setup.args.kwarg):
        raise ValueError('unsupported runner graph setup API; inspect the current worker before applying')
    actual = tuple(arg.arg for arg in methods['propose_draft_token_ids'].args.args)
    expected = ('self', 'valid_sampled_token_ids', 'sampling_metadata', 'scheduler_output',
                'spec_decode_metadata', 'spec_decode_common_attn_metadata', 'positions',
                'num_scheduled_tokens', 'hidden_states', 'aux_hidden_states',
                'sample_hidden_states', 'target_model_batch_desc')
    if actual != expected:
        raise ValueError('unsupported runner proposal API; inspect the current worker before applying')
    sample = methods['sample_tokens']
    bookkeeping = [node.lineno for node in ast.walk(sample) if isinstance(node, ast.Call) and
                   isinstance(node.func, ast.Attribute) and node.func.attr == '_bookkeeping_sync']
    proposal = [node.lineno for node in ast.walk(sample) if isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and node.func.id == 'propose_draft_token_ids']
    body = ast.unparse(methods['_bookkeeping_sync'])
    if (not bookkeeping or not proposal or min(proposal) <= max(bookkeeping) or
            'self.input_batch.num_tokens_no_spec[req_idx] = end_idx' not in body):
        raise ValueError('unsupported runner: MTP requires proposal after confirmed-token bookkeeping')


def worker_memory_entry(source):
    block = ('\n# Account for optional auxiliary graph pools after the worker class is defined.\n'
             'if vllm_version_is("0.23.0"):\n'
             '    from vllm_ascend.patch.worker.patch_external_draft_mtp import install_worker_memory_hook\n'
             '    install_worker_memory_hook(NPUWorker)\n')
    tree = ast.parse(source)
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NPUWorker'), None)
    method = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and
                   n.name == 'determine_available_memory'), None) if cls else None
    if method is None or tuple(a.arg for a in method.args.args) != ('self',):
        raise ValueError('unsupported worker memory API; inspect before applying MTP graphs')
    if source.count(block) == 1:
        return source
    if 'install_worker_memory_hook' in source:
        raise ValueError('ambiguous worker memory hook registration')
    return source.rstrip() + '\n' + block


def make_patch(root):
    root = Path(root)
    check_runner((root / 'vllm_ascend/worker/model_runner_v1.py').read_text(encoding='utf-8'))
    directory = Path(__file__).resolve().parent
    paths = {
        'vllm_ascend/patch/worker/__init__.py': None,
        'vllm_ascend/worker/worker.py': None,
        'vllm_ascend/spec_decode/asr_mtp_heads.py': directory / 'asr_mtp_heads.py',
        'vllm_ascend/spec_decode/asr_mtp_fused.py': directory / 'asr_mtp_fused.py',
        'vllm_ascend/spec_decode/asr_mtp_qkv_rope.py': directory / 'asr_mtp_qkv_rope.py',
        'vllm_ascend/spec_decode/asr_mtp_row_copy.py': directory / 'asr_mtp_row_copy.py',
        'vllm_ascend/spec_decode/asr_mtp_row_copy_kernel.py': directory / 'asr_mtp_row_copy_kernel.py',
        'vllm_ascend/spec_decode/asr_mtp_graph.py': directory / 'asr_mtp_graph.py',
        'vllm_ascend/spec_decode/asr_mtp_token_attention.py': directory / 'asr_mtp_token_attention.py',
        'vllm_ascend/spec_decode/asr_mtp_token_graph.py': directory / 'asr_mtp_token_graph.py',
        'vllm_ascend/spec_decode/asr_mtp_paged_cache.py': directory / 'asr_mtp_paged_cache.py',
        'vllm_ascend/spec_decode/asr_mtp_paged_attention.py': directory / 'asr_mtp_paged_attention.py',
        'vllm_ascend/spec_decode/asr_mtp_worker.py': directory / 'asr_mtp_worker.py',
        'vllm_ascend/patch/worker/patch_external_draft_mtp.py': directory / 'patch_external_draft_mtp.py',
    }
    chunks = []
    for name, source in paths.items():
        target = root / name
        old = target.read_text(encoding='utf-8').replace('\r\n', '\n') if target.exists() else ''
        if source is None:
            new = (registry(old) if name.endswith('/__init__.py') else
                   worker_memory_entry(worker_entry(old)))
        else:
            new = source.read_text(encoding='utf-8').replace('\r\n', '\n')
        diff = difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile='a/' + name if target.exists() else '/dev/null', tofile='b/' + name)
        # git apply accepts empty context lines; avoid trailing spaces in a
        # committed patch artifact for unchanged blank source lines.
        chunks.extend('\n' if line == ' \n' else line for line in diff)
    return ''.join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ascend-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = make_patch(args.ascend_root)
    args.out.write_bytes(result.encode('utf-8'))
    print(f'wrote {args.out}: {len(result.splitlines())} lines; engine unchanged')


if __name__ == '__main__':
    main()
