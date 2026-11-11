# SPDX-License-Identifier: Apache-2.0
"""Startup-only QKV/RoPE dispatch for frozen auxiliary MTP heads.

Prepare after ``prepare_fused_heads`` and final device/dtype placement. This
helper reuses its packed projection storage; it never packs or changes model
parameters. Moving the model or assigning new Parameters after preparation is
unsupported. The returned object may be installed as ``heads.qkv_rope``.

``auto`` chooses native half-style RoPE when available, preserving the existing
separate Q/K normalization and its output rounding. ``torch`` retains the old
normalization and multiply/add order. ``triton`` is an explicit precision
tradeoff: the existing Ascend MRoPE kernel computes normalization and rotation
in FP32 before its final cast. It is never selected automatically.

Position tables are calculated once on the final device. Runtime only selects
device rows; it performs no trigonometry, scalar extraction, or CPU bounds
check. Callers must clamp invalid padding anchors to [0, max_context-1] before
adding head depth; valid anchors must obey the same context bound.
"""

import importlib
import inspect

import torch
from torch.nn import functional as F

_JOINT_ARGUMENTS = (
    "qkv", "q_weight", "k_weight", "cos_sin", "num_q_heads",
    "num_kv_heads", "head_size", "eps", "mrope_section",
    "is_interleaved", "rope_dim", "q_bias", "k_bias", "has_gate",
)


def _joint_operator():
    # Engine startup imports vllm_ascend.ops when Triton is supported. Do not
    # trigger worker imports or register global operators from this helper.
    return getattr(torch.ops.vllm, "triton_split_qkv_rmsnorm_mrope", None)


def _validate_joint_signature(operator):
    schemas = getattr(operator, "_schemas", None)
    if schemas:
        if not any(
            tuple(argument.name for argument in schema.arguments) == _JOINT_ARGUMENTS
            for schema in schemas.values()
        ):
            raise ValueError("MTP Triton QKV/RoPE operator signature differs")
        return
    try:
        signature = inspect.signature(operator)
    except (TypeError, ValueError) as exc:
        raise ValueError("MTP Triton QKV/RoPE operator signature unavailable") from exc
    try:
        signature.bind(**dict.fromkeys(_JOINT_ARGUMENTS))
    except TypeError as exc:
        raise ValueError("MTP Triton QKV/RoPE operator signature differs") from exc


def _power_of_two(value):
    return value > 0 and value & (value - 1) == 0


class QKVRope:
    """Fixed-device projection, Q/K normalization, and half-style rotation."""

    def __init__(self, heads, max_context, backend, native_ops, fused_op):
        cfg = heads.config
        self.backend = backend
        self.num_q_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.hidden_size = cfg.hidden_size
        self.position_capacity = max_context + heads.num_heads + 1
        self._native = native_ops
        self._joint = fused_op
        self._attentions = {id(block.decoder.self_attn) for block in heads.blocks}
        first = heads.blocks[0].decoder.self_attn._mtp_qkv_weight
        self.device, self.dtype = first.device, first.dtype
        inverse = 1.0 / (cfg.rope_theta ** (torch.arange(
            0, self.head_dim, 2, device=self.device, dtype=torch.float32
        ) / self.head_dim))
        frequencies = torch.arange(
            self.position_capacity, device=self.device, dtype=torch.float32
        )[:, None] * inverse
        if backend == "triton":
            # Existing op expects [3,T,D] with each row [cos_half,sin_half].
            # All axes share the same position for MTP's ordinary RoPE.
            cache = torch.cat((frequencies.cos(), frequencies.sin()), -1)
            self.table = cache.unsqueeze(0).repeat(3, 1, 1).contiguous()
        else:
            angles = torch.cat((frequencies, frequencies), -1)
            # Separate coefficient planes keep both selected [T,D] tensors
            # contiguous views after one index_select. Native RoPE accepts
            # noncontiguous inputs too, but this avoids backend stride handling
            # or an implicit contiguous conversion on every head invocation.
            self.table = torch.stack(
                (angles.cos().to(self.dtype), angles.sin().to(self.dtype)), 0
            ).contiguous()

    @torch.inference_mode()
    def __call__(self, attn, normed, positions):
        if id(attn) not in self._attentions:
            raise ValueError("MTP QKV/RoPE attention was not prepared")
        if normed.shape[-1] != self.hidden_size:
            raise ValueError("MTP QKV/RoPE hidden shape differs")
        leading = normed.shape[:-1]
        if positions.shape != leading:
            raise ValueError("MTP QKV/RoPE position shape must match token shape")
        if positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("MTP QKV/RoPE positions must have integer dtype")
        if (positions.device != self.device or normed.device != self.device
                or normed.dtype != self.dtype):
            raise ValueError("MTP QKV/RoPE device/dtype differs from startup placement")
        flat_positions = positions.reshape(-1)
        # Projection bias is applied once here. The joint kernel's q_bias and
        # k_bias mean post-normalization bias and must remain None.
        packed = F.linear(normed, attn._mtp_qkv_weight, attn._mtp_qkv_bias)
        if self.backend == "triton":
            selected = self.table.index_select(1, flat_positions)
            q, k, v, _ = self._joint(
                qkv=packed.reshape(-1, packed.shape[-1]),
                q_weight=attn.q_norm.weight,
                k_weight=attn.k_norm.weight,
                cos_sin=selected,
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                eps=attn.q_norm.eps,
                mrope_section=[self.head_dim // 2, 0, 0],
                is_interleaved=False,
                rope_dim=self.head_dim,
                q_bias=None,
                k_bias=None,
                has_gate=False,
            )
            return tuple(value.reshape(*leading, count, self.head_dim)
                         for value, count in ((q, self.num_q_heads),
                                              (k, self.num_kv_heads),
                                              (v, self.num_kv_heads)))
        q, k, v = packed.split(attn._mtp_qkv_sizes, -1)
        q = attn.q_norm(q.reshape(*leading, self.num_q_heads, self.head_dim))
        k = attn.k_norm(k.reshape(*leading, self.num_kv_heads, self.head_dim))
        v = v.reshape(*leading, self.num_kv_heads, self.head_dim)
        selected = self.table.index_select(1, flat_positions)
        cos, sin = selected.unbind(0)
        if self.backend == "native":
            cos, sin = (part.reshape(1, -1, 1, self.head_dim) for part in (cos, sin))
            rotated = []
            for value, count in ((q, self.num_q_heads), (k, self.num_kv_heads)):
                output = self._native.npu_rotary_mul(
                    value.reshape(1, -1, count, self.head_dim), cos, sin,
                    rotary_mode="half",
                )
                rotated.append(output.reshape(*leading, count, self.head_dim))
            return rotated[0], rotated[1], v
        cos, sin = (part.reshape(*leading, 1, self.head_dim) for part in (cos, sin))
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin), v

    @staticmethod
    def _rotate(value, cos, sin):
        left, right = value.chunk(2, dim=-1)
        return value * cos + torch.cat((-right, left), -1) * sin


@torch.no_grad()
def prepare_qkv_rope(heads, *, max_context, backend="auto", native_ops=None,
                     fused_op=None):
    """Return a prepared callable without registering state or repacking weights.

    Explicit native/Triton requests fail during startup when unsupported;
    ``auto`` may fall back to Torch if native RoPE is unavailable. Optional
    API-compatible backends allow real-math CPU contract tests.
    """
    if backend not in ("auto", "torch", "native", "triton"):
        raise ValueError("MTP QKV/RoPE backend must be auto, torch, native or triton")
    if type(max_context) is not int or max_context < 1:
        raise ValueError("MTP QKV/RoPE max_context must be a positive integer")
    if heads.training or any(parameter.requires_grad for parameter in heads.parameters()):
        raise ValueError("MTP QKV/RoPE preparation requires frozen eval heads")
    if not heads.blocks or any(
        getattr(block.decoder.self_attn, "_mtp_qkv_weight", None) is None
        for block in heads.blocks
    ):
        raise ValueError("prepare_fused_heads must run before MTP QKV/RoPE preparation")
    cfg = heads.config
    first = heads.blocks[0].decoder.self_attn._mtp_qkv_weight
    if native_ops is None and first.device.type == "npu" and backend in ("auto", "native"):
        try:
            native_ops = importlib.import_module("torch_npu")
        except ImportError:
            native_ops = None
    native_available = callable(getattr(native_ops, "npu_rotary_mul", None))
    if backend == "auto":
        backend = "native" if native_available else "torch"
    if backend == "native" and not native_available:
        raise ValueError("MTP native RoPE requires npu_rotary_mul")
    if backend == "triton":
        if (cfg.head_dim not in (128, 256)
                or first.dtype not in (torch.float16, torch.bfloat16)
                or not _power_of_two(cfg.num_attention_heads * cfg.head_dim)
                or not _power_of_two(cfg.num_key_value_heads * cfg.head_dim)):
            raise ValueError("unsupported MTP Triton QKV/RoPE geometry or dtype")
        if any(block.decoder.self_attn.q_norm.eps is None
               or block.decoder.self_attn.q_norm.eps != block.decoder.self_attn.k_norm.eps
               for block in heads.blocks):
            raise ValueError("MTP Triton QKV/RoPE requires equal explicit Q/K norm epsilon")
        fused_op = fused_op if fused_op is not None else _joint_operator()
        if not callable(fused_op):
            raise ValueError("MTP Triton split QKV RMSNorm MRoPE operator is unavailable")
        _validate_joint_signature(fused_op)
    return QKVRope(heads, max_context, backend, native_ops, fused_op)
