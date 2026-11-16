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
"""


_PREFIX = "npu_fused_infer_attention_score"
_WORKSPACE_ATTRIBUTE = "_npu_fused_infer_attention_score_get_max_workspace"
_OP_NAMES = (
    _PREFIX + "_batch_invariant",
    _PREFIX + "_batch_invariant_out",
    "_" + _PREFIX + "_batch_invariant_get_max_workspace",
    "_" + _PREFIX + "_batch_invariant_infer_output",
)


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
