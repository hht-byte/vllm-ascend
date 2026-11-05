"""Export online MTP modules plus one worker-registry import; never edit the engine.

Apply on top of external_draft_first_verify_v023.patch in the user's engine
checkout. Source anchors/API and post-bookkeeping call order are checked first.
"""
import argparse
import ast
import difflib
from pathlib import Path


def registry(source):
    added = '    import vllm_ascend.patch.worker.patch_external_draft_mtp  # noqa\n'
    anchor = '    import vllm_ascend.patch.worker.patch_external_draft_first_verify  # noqa\n'
    if source.count(added) == 1:
        return source
    if source.count(anchor) != 1:
        raise ValueError('apply the existing first-verify worker patch before the MTP extension')
    return source.replace(anchor, anchor + added, 1)


def check_runner(source):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'NPUModelRunner')
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    required = ('_get_drafter', '_update_states', 'propose_draft_token_ids', '_bookkeeping_sync', 'sample_tokens')
    if any(name not in methods for name in required):
        raise ValueError('unsupported runner: missing MTP hook or bookkeeping method')
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


def make_patch(root):
    root = Path(root)
    check_runner((root / 'vllm_ascend/worker/model_runner_v1.py').read_text(encoding='utf-8'))
    directory = Path(__file__).resolve().parent
    paths = {
        'vllm_ascend/patch/worker/__init__.py': None,
        'vllm_ascend/spec_decode/asr_mtp_heads.py': directory / 'asr_mtp_heads.py',
        'vllm_ascend/spec_decode/asr_mtp_worker.py': directory / 'asr_mtp_worker.py',
        'vllm_ascend/patch/worker/patch_external_draft_mtp.py': directory / 'patch_external_draft_mtp.py',
    }
    chunks = []
    for name, source in paths.items():
        target = root / name
        old = target.read_text(encoding='utf-8').replace('\r\n', '\n') if target.exists() else ''
        new = registry(old) if source is None else source.read_text(encoding='utf-8').replace('\r\n', '\n')
        chunks.extend(difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile='a/' + name if target.exists() else '/dev/null', tofile='b/' + name))
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
