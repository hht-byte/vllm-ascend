# SPDX-License-Identifier: Apache-2.0
"""Local, frozen MTP-head fusion without changing checkpoint parameter names.

Call ``prepare_fused_heads`` once, after strict checkpoint loading, final
device/dtype placement, and ``eval().requires_grad_(False)``. Packed buffers
share storage with the original projection Parameters and are nonpersistent.
Default ``load_state_dict(assign=False)`` continues to update those views;
moving the model or replacing Parameters with ``assign=True`` afterwards is
outside this startup contract. Prepare again from an unprepared model instead.

Only NPU preparation imports torch_npu. CPU uses real Torch reference math,
including packed projections, so checkpoint/aliasing tests need no NPU runtime.
Native operations can round differently; device capture and ASR acceptance
must still be checked on the service's torch-npu/CANN version.
"""

import importlib

import torch
from torch import nn
from torch.nn import functional as F

_NATIVE_APIS = ("npu_rms_norm", "npu_add_rms_norm", "npu_swiglu")
_NORM_PATHS = (
    "hidden_norm",
    "token_norm",
    "decoder.input_layernorm",
    "decoder.self_attn.q_norm",
    "decoder.self_attn.k_norm",
    "decoder.post_attention_layernorm",
    "output_norm",
)


class _FusedRMSNorm(nn.Module):
    """Retain the checkpoint's RMSNorm weight; cache native dispatch at load."""

    def __init__(self, original, native_ops):
        super().__init__()
        self.normalized_shape = original.normalized_shape
        self.eps = original.eps
        self.elementwise_affine = original.elementwise_affine
        self.register_parameter("weight", original.weight)
        self._mtp_native_ops = native_ops
        self.train(original.training)

    def forward(self, value):
        if self._mtp_native_ops is None:
            return F.rms_norm(value, self.normalized_shape, self.weight, self.eps)
        epsilon = self.eps if self.eps is not None else torch.finfo(value.dtype).eps
        return self._mtp_native_ops.npu_rms_norm(value, self.weight, epsilon)[0]


def _projection_group(projections):
    """Check everything before replacing any storage in the model."""
    first = projections[0]
    if not all(isinstance(module, nn.Linear) for module in projections):
        raise ValueError("MTP fusion requires Linear projection modules")
    if any(
        module.in_features != first.in_features
        or module.weight.device != first.weight.device
        or module.weight.dtype != first.weight.dtype
        or (module.bias is None) != (first.bias is None)
        for module in projections
    ):
        raise ValueError("MTP fused projection geometry/device/dtype/bias differs")


def _pack(module, prefix, projections):
    # Frozen Parameters remain registered under their original names. Their
    # data becomes a packed-storage view; no duplicate resident weights remain.
    weight = torch.cat(
        [projection.weight.detach() for projection in projections], dim=0
    )
    module.register_buffer(f"{prefix}_weight", weight, persistent=False)
    offset = 0
    for projection in projections:
        stop = offset + projection.out_features
        projection.weight.data = weight[offset:stop]
        offset = stop
    bias = None
    if projections[0].bias is not None:
        bias = torch.cat(
            [projection.bias.detach() for projection in projections], dim=0
        )
        offset = 0
        for projection in projections:
            stop = offset + projection.out_features
            projection.bias.data = bias[offset:stop]
            offset = stop
    module.register_buffer(f"{prefix}_bias", bias, persistent=False)


@torch.no_grad()
def prepare_fused_heads(model, *, enabled=True, native_ops=None):
    """Pack frozen auxiliary heads once and install seven local Norms/head.

    ``enabled=False`` leaves a fresh model completely unchanged. It does not
    undo a model already prepared; use the startup switch on a fresh load.
    ``native_ops`` is an optional API-compatible test backend. Production
    dispatch is resolved and validated here, never during graph capture.
    """
    if type(enabled) is not bool:
        raise TypeError("MTP native fusion enabled must be a bool")
    if not enabled:
        return False
    if getattr(model, "_mtp_fused_prepared", False):
        return True
    if model.training or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise ValueError("MTP native fusion requires eval mode and frozen Parameters")
    blocks = list(model.blocks)
    if not blocks:
        raise ValueError("MTP native fusion requires auxiliary heads")
    parameters = list(model.parameters())
    if native_ops is None and any(
        parameter.device.type == "npu" for parameter in parameters
    ):
        native_ops = importlib.import_module("torch_npu")
    if native_ops is not None and any(
        not callable(getattr(native_ops, name, None)) for name in _NATIVE_APIS
    ):
        raise ValueError(
            "MTP native fusion requires rms_norm, add_rms_norm and swiglu APIs"
        )
    for block in blocks:
        attention, mlp = block.decoder.self_attn, block.decoder.mlp
        _projection_group((attention.q_proj, attention.k_proj, attention.v_proj))
        _projection_group((mlp.gate_proj, mlp.up_proj))
        if mlp.gate_proj.out_features != mlp.up_proj.out_features:
            raise ValueError("MTP fused gate/up widths must match")
        for path in _NORM_PATHS:
            norm = block.get_submodule(path)
            if not isinstance(norm, nn.RMSNorm) or norm.weight is None:
                raise ValueError("MTP native fusion requires affine RMSNorm modules")
    for block in blocks:
        attention, mlp = block.decoder.self_attn, block.decoder.mlp
        projections = (attention.q_proj, attention.k_proj, attention.v_proj)
        attention._mtp_qkv_sizes = tuple(
            projection.out_features for projection in projections
        )
        _pack(attention, "_mtp_qkv", projections)
        _pack(mlp, "_mtp_gate_up", (mlp.gate_proj, mlp.up_proj))
        mlp._mtp_native_ops = native_ops
        for path in _NORM_PATHS:
            parent_path, _, name = path.rpartition(".")
            parent = block.get_submodule(parent_path) if parent_path else block
            setattr(parent, name, _FusedRMSNorm(getattr(parent, name), native_ops))
    model._mtp_fused_prepared = True
    return True


def project_qkv(attn, hidden):
    """One packed projection after preparation; original three calls otherwise."""
    weight = getattr(attn, "_mtp_qkv_weight", None)
    if weight is None:
        return attn.q_proj(hidden), attn.k_proj(hidden), attn.v_proj(hidden)
    return F.linear(hidden, weight, attn._mtp_qkv_bias).split(
        attn._mtp_qkv_sizes, dim=-1
    )


def add_rms_norm(norm, left, right):
    """Return both normalized output and the residual used by the next add."""
    native_ops = getattr(norm, "_mtp_native_ops", None)
    if native_ops is None:
        residual = left + right
        return norm(residual), residual
    epsilon = norm.eps if norm.eps is not None else torch.finfo(left.dtype).eps
    normalized, _, residual = native_ops.npu_add_rms_norm(
        left, right, norm.weight, epsilon
    )
    return normalized, residual


def mlp_forward(mlp, hidden):
    """Pack gate/up once, then use native SwiGLU on the combined activation."""
    weight = getattr(mlp, "_mtp_gate_up_weight", None)
    if weight is None:
        return mlp.down_proj(F.silu(mlp.gate_proj(hidden)) * mlp.up_proj(hidden))
    packed = F.linear(hidden, weight, mlp._mtp_gate_up_bias)
    if mlp._mtp_native_ops is None:
        gate, up = packed.chunk(2, dim=-1)
        activated = F.silu(gate) * up
    else:
        activated = mlp._mtp_native_ops.npu_swiglu(packed)
    return mlp.down_proj(activated)
