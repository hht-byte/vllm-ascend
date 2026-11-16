"""Bridge the separately named AscendC FIA operators to TorchNPU's out API.

Install as ``vllm_ascend.patch.worker.blue_batch_invariant``. The trusted worker
hook calls ``install_fia_bridge(torch_npu, torch.ops.batch_invariant_ops)`` AFTER
the stock ``init_batch_invariance`` returns and BEFORE model/arena construction.
Only the caller opts into batch invariance and verifies the AscendC registration;
this module neither enables a mode nor imports Torch or an extension by itself.

ABI: the official batch_invariant_ops 2.0.0 archive defines default, workspace,
named out and output-inference as four distinct operator names. Only these 2.0
names are accepted; no 1.0 name detection or fallback occurs. The out schema takes
``attention_out`` and ``softmax_lse`` aliases rather than ``out=[...]``. The
workspace is opaque and must be prepared before graph capture. No tensor is
allocated, copied, cast or reshaped here, and no original FIA fallback exists.
The extension decides output layout and binary/kernel compatibility. Its CANN
package reports compiler 9.1.0; compatibility with the exact runtime, graph task
capture/rebinding and token parity require actual isolated preflight evidence.
This bridge does not establish support for target-model FULL graph mode.

``make_batch_invariant_matmul_patch`` generates a bounded stock-source change
for an explicit, recorded mm/matmul backend selection. It never catches a
failed kernel or changes registration in response to a runtime failure.
"""

import ast
import hashlib


_PREFIX = "npu_fused_infer_attention_score"
_WORKSPACE_ATTRIBUTE = "_npu_fused_infer_attention_score_get_max_workspace"
_OP_NAMES = (
    _PREFIX + "_batch_invariant",
    _PREFIX + "_batch_invariant_out",
    "_" + _PREFIX + "_batch_invariant_get_max_workspace",
    "_" + _PREFIX + "_batch_invariant_infer_output",
)
_STOCK_ENABLE_SHA256 = "6b484ec54ca07ea9557c12d19784ffac784b3fa069a523c72b8c3ee961acbc55"
_BACKEND_VARIABLE = "LLM_OPT_BATCH_INVARIANT_MATMUL_BACKEND"
_BACKEND_MARKER = "BLUE_BATCH_INVARIANT_MATMUL_BACKEND"


def make_batch_invariant_matmul_patch(source):
    """Return patched frozen Ascend 1cdb8c4 source; reject drift or reapplication.

    The exact initializer body is bound by its normalized source SHA256, making
    this independent of Python-version changes in AST serialization. Parsing
    and unique module/function anchors are checked before and after rewriting.
    The AscendC branch selects its two matmul implementations explicitly and
    uses the exact default-sum ABI adapter below. Other operator registrations,
    including the existing Triton-only branch, remain intact.
    """
    source = source.replace("\r\n", "\n")
    tree = ast.parse(source)
    functions = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == "enable_batch_invariant_mode"]
    if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef):
        raise ValueError("Unknown stock batch-invariant initializer")
    original = ast.get_source_segment(source, functions[0])
    if hashlib.sha256(original.encode("utf-8")).hexdigest() != _STOCK_ENABLE_SHA256:
        raise ValueError("Stock batch-invariant initializer changed")
    anchor = "_batch_invariant_LIB = None\n"
    imports_os = any(isinstance(node, ast.Import)
                     and any(alias.name == "os" and alias.asname is None
                             for alias in node.names) for node in tree.body)
    if (source.count(anchor) != 1 or not imports_os
            or _BACKEND_VARIABLE in source or _BACKEND_MARKER in source):
        raise ValueError("Unknown stock batch-invariant module contract")
    changed = original.replace(
        "    global _batch_invariant_LIB\n",
        "    global _batch_invariant_LIB, BLUE_BATCH_INVARIANT_MATMUL_BACKEND\n"
        "    blue_matmul_backend = os.environ.get(\n"
        "        'LLM_OPT_BATCH_INVARIANT_MATMUL_BACKEND', 'ascendc')\n"
        "    if blue_matmul_backend not in ('ascendc', 'triton'):\n"
        "        raise ValueError('Unknown batch-invariant matmul backend')\n"
        "    if blue_matmul_backend == 'triton' and not HAS_TRITON:\n"
        "        raise ValueError('Requested batch-invariant Triton is unavailable')\n"
        "    from vllm_ascend.patch.worker.blue_batch_invariant import defaultSumCallable\n"
        "    blue_default_sum = defaultSumCallable(torch)\n",
        1,
    )
    for name, triton_function, ascendc_function in (
        ("mm", "mm_batch_invariant", "npu_mm_batch_invariant"),
        ("matmul", "matmul_batch_invariant", "npu_matmul_batch_invariant"),
    ):
        old = (f'        _batch_invariant_LIB.impl("aten::{name}", '
               f'torch.ops.batch_invariant_ops.{ascendc_function}, "NPU")')
        new = (f'        _batch_invariant_LIB.impl("aten::{name}",\n'
               f'            {triton_function} if blue_matmul_backend == "triton"\n'
               f'            else torch.ops.batch_invariant_ops.{ascendc_function}, "NPU")')
        if changed.count(old) != 1:
            raise ValueError("Unknown stock matmul registration anchor")
        changed = changed.replace(old, new, 1)
    old_sum = ('        _batch_invariant_LIB.impl("aten::sum", '
               'torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant, "NPU")')
    if changed.count(old_sum) != 1:
        raise ValueError("Unknown stock default-sum registration anchor")
    changed = changed.replace(
        old_sum, '        _batch_invariant_LIB.impl("aten::sum", blue_default_sum, "NPU")', 1)
    changed += (
        "\n\n    BLUE_BATCH_INVARIANT_MATMUL_BACKEND = (\n"
        "        blue_matmul_backend if HAS_ASCENDC_BATCH_INVARIANT\n"
        "        else 'triton' if HAS_TRITON else None)"
    )
    source = source.replace(original, changed, 1).replace(
        anchor, anchor + _BACKEND_MARKER + " = None\n", 1)
    ast.parse(source)
    return source


def defaultSumCallable(torch):
    """Adapt aten.sum.default(x, *, dtype=None) using its untouched dim overload.

    Capture dim_IntList before registration. It receives explicit all-dimension
    reduction axes (axis zero for a scalar), keepdim=False and the caller's
    dtype. No patched default/global sum, host transfer or output guess is used.
    """
    aten = getattr(getattr(torch, "ops", None), "aten", None)
    operation = getattr(getattr(aten, "sum", None), "dim_IntList", None)
    if not callable(operation):
        raise ValueError("Untouched aten.sum.dim_IntList overload is unavailable")

    def default_sum(x, *, dtype=None):
        dimensions = list(range(x.dim())) or [0]
        return operation(x, dimensions, False, dtype=dtype)

    return default_sum


class _FiaBridge:
    def __init__(self, batch_ops, operations):
        self._batch_ops = batch_ops
        self.default, self._out, self.get_max_workspace, self.infer_output = operations

    def __call__(self, *args, **kwargs):
        return self.default(*args, **kwargs)

    def out(self, *args, out, **kwargs):
        if not isinstance(out, (list, tuple)) or len(out) != 2:
            raise TypeError("FIA out requires two output tensors")
        if "attention_out" in kwargs or "softmax_lse" in kwargs:
            raise TypeError("FIA out aliases must be supplied through out")
        return self._out(
            *args, **kwargs, attention_out=out[0], softmax_lse=out[1]
        )


def fia_bridge_installed(torch_npu):
    """Return whether both TorchNPU entry points still use the same bridge."""
    current = getattr(torch_npu, _PREFIX, None)
    return (
        isinstance(current, _FiaBridge)
        and getattr(torch_npu, _WORKSPACE_ATTRIBUTE, None)
        is current.get_max_workspace
    )


def install_fia_bridge(torch_npu, batch_ops):
    """Install the four verified extension entry points, once per namespace.

    ``batch_ops`` must be ``torch.ops.batch_invariant_ops``, whose operators use
    the schemas documented above. All required callables are checked before
    changing TorchNPU. An existing bridge with a different namespace or damaged
    workspace wiring fails closed instead of silently changing a live graph.
    """
    operations = tuple(getattr(batch_ops, name, None) for name in _OP_NAMES)
    if not all(callable(operation) for operation in operations):
        raise ValueError("AscendC FIA bridge requires all four 2.0 extension operators")
    current = getattr(torch_npu, _PREFIX, None)
    if isinstance(current, _FiaBridge):
        if current._batch_ops is not batch_ops or not fia_bridge_installed(torch_npu):
            raise ValueError("AscendC FIA bridge is already bound differently")
        return current
    if not callable(current):
        raise ValueError("TorchNPU FIA entry point is unavailable")
    bridge = _FiaBridge(batch_ops, operations)
    setattr(torch_npu, _PREFIX, bridge)
    setattr(torch_npu, _WORKSPACE_ATTRIBUTE, bridge.get_max_workspace)
    return bridge
