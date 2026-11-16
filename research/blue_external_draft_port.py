"""Read-only source transforms for pristine vLLM-Ascend v0.23 transport.

The caller writes the returned source into its isolated engine overlay. These
helpers never import or mutate an installed engine and reject unknown APIs.
"""
import ast


DISPATCH_BLOCK = '''    if method == "custom_class":
        from vllm_ascend.spec_decode.external_draft_proposer import (
            EXTERNAL_CLASS, AscendExternalDraftProposer,
        )
        if vllm_config.speculative_config.model != EXTERNAL_CLASS:
            raise ValueError("Ascend custom_class supports only the external draft proposer")
        return AscendExternalDraftProposer(vllm_config, runner)
'''
REGISTRY_BLOCK = '''
# Install synchronous external draft transport before first-verify metadata.
if vllm_version_is("0.23.0"):
    import vllm_ascend.patch.worker.patch_external_draft  # noqa
    import vllm_ascend.patch.worker.patch_external_draft_first_verify  # noqa
'''


def dispatch(source):
    """Add the exact external class to Ascend's existing typed proposer path."""
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'get_spec_decode_method']
    if (len(functions) != 1 or tuple(arg.arg for arg in functions[0].args.args) !=
            ('method', 'vllm_config', 'device', 'runner') or functions[0].args.kwonlyargs
            or functions[0].args.vararg or functions[0].args.kwarg):
        raise ValueError('unsupported Ascend speculative dispatch API')
    if source.count(DISPATCH_BLOCK) == 1:
        return source
    if 'custom_class' in ast.unparse(functions[0]):
        raise ValueError('ambiguous existing Ascend custom_class dispatch')
    marker = '    if method == "ngram":\n        return AscendNgramProposer(vllm_config, runner)\n'
    if source.count(marker) != 1:
        raise ValueError('unsupported Ascend ngram dispatch anchor')
    return source.replace(marker, DISPATCH_BLOCK + marker, 1)


def worker_registry(source):
    """Register only version-compatible hooks before model runner import."""
    tree = ast.parse(source)
    imports_version = any(isinstance(node, ast.ImportFrom) and node.module == 'vllm_ascend.utils'
        and any(alias.name == 'vllm_version_is' and alias.asname is None for alias in node.names)
        for node in tree.body)
    if not imports_version:
        raise ValueError('unsupported worker registry: missing vllm_version_is import')
    if source.count(REGISTRY_BLOCK) == 1:
        return source
    if 'import vllm_ascend.patch.worker.patch_external_draft' in source:
        raise ValueError('ambiguous existing external draft worker registry')
    return source.rstrip() + '\n' + REGISTRY_BLOCK


def check_input_batch(source):
    """Ensure each initial or rebuilt input batch exposes bounded token state."""
    tree = ast.parse(source)
    cls = next((node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == 'NPUInputBatch'), None)
    constructor = next((node for node in cls.body if isinstance(node, ast.FunctionDef)
                        and node.name == '__init__'), None) if cls else None
    required = {'self', 'num_speculative_tokens', 'vocab_size', 'max_model_len'}
    if (cls is None or not any(isinstance(base, ast.Name) and base.id == 'InputBatch'
                              for base in cls.bases) or constructor is None
            or not required.issubset({arg.arg for arg in constructor.args.args})):
        raise ValueError('unsupported NPUInputBatch constructor or inheritance contract')
    assignments = {ast.unparse(target) for node in ast.walk(constructor)
                   if isinstance(node, ast.Assign) for target in node.targets}
    if not {'self.vocab_size', 'self.max_model_len'}.issubset(assignments):
        raise ValueError('unsupported NPUInputBatch token-bound state')
