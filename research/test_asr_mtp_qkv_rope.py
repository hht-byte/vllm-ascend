"""QKV/RoPE startup dispatch and real math; CPU is not NPU acceptance."""

import importlib.util
import unittest
from copy import deepcopy
from types import SimpleNamespace


def helper_api(test):
    try:
        import asr_mtp_qkv_rope
    except ImportError:
        test.fail("MTP QKV/RoPE preparation is not implemented")
    return asr_mtp_qkv_rope


def rotate_half(tensor):
    left, right = tensor.chunk(2, -1)
    import torch

    return torch.cat((-right, left), -1)


class NativeMath:
    def __init__(self):
        self.calls = []
        self.coefficients_contiguous = []

    def npu_rotary_mul(self, value, cos, sin, *, rotary_mode="half"):
        self.calls.append((tuple(value.shape), tuple(cos.shape), rotary_mode))
        self.coefficients_contiguous.append((cos.is_contiguous(), sin.is_contiguous()))
        return value * cos + rotate_half(value) * sin


class FusedMath:
    """Mirror the joint kernel's FP32 norm + RoPE and final cast."""

    def __init__(self):
        self.calls = []

    def __call__(self, qkv, q_weight, k_weight, cos_sin, num_q_heads,
                 num_kv_heads, head_size, eps, mrope_section,
                 is_interleaved=False, rope_dim=None, q_bias=None,
                 k_bias=None, has_gate=False):
        import torch

        self.calls.append((qkv.shape, cos_sin.shape, mrope_section,
                           is_interleaved, rope_dim, q_bias, k_bias, has_gate))
        qsize, kvsize = num_q_heads * head_size, num_kv_heads * head_size
        q, k, v = qkv.split((qsize, kvsize, kvsize), -1)
        half_cos, half_sin = cos_sin[0].chunk(2, -1)
        cos = torch.cat((half_cos, half_cos), -1).float()[:, None]
        sin = torch.cat((half_sin, half_sin), -1).float()[:, None]
        result = []
        for value, weight, count in ((q, q_weight, num_q_heads),
                                     (k, k_weight, num_kv_heads)):
            value = value.view(-1, count, head_size).float()
            value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
            value = value * weight.float()
            result.append((value * cos + rotate_half(value) * sin).to(qkv.dtype))
        return result[0].flatten(1), result[1].flatten(1), v, qkv.new_empty((0,))


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Torch")
class QKVRopeTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.t = torch
        torch.set_num_threads(1)
        torch.manual_seed(214)

    def model(self, dtype=None, *, head_dim=4, bias=True):
        from asr_mtp_fused import prepare_fused_heads
        from asr_mtp_heads import BatchedMTPHeads

        cfg = SimpleNamespace(hidden_size=2 * head_dim,
                              intermediate_size=4 * head_dim,
                              num_attention_heads=2, num_key_value_heads=1,
                              head_dim=head_dim, attention_bias=bias,
                              rms_norm_eps=1e-5, hidden_act="silu",
                              rope_theta=10000, rope_scaling=None)
        model = BatchedMTPHeads(cfg, 2).to(dtype=dtype or self.t.float32)
        model.eval().requires_grad_(False)
        prepare_fused_heads(model)
        return model

    def expected(self, model, attn, hidden, positions):
        from asr_mtp_fused import project_qkv

        cfg = model.config
        q, k, v = project_qkv(attn, hidden)
        q = attn.q_norm(q.view(*hidden.shape[:-1], cfg.num_attention_heads, cfg.head_dim))
        k = attn.k_norm(k.view(*hidden.shape[:-1], cfg.num_key_value_heads, cfg.head_dim))
        v = v.view(*hidden.shape[:-1], cfg.num_key_value_heads, cfg.head_dim)
        inverse = 1.0 / (cfg.rope_theta ** (self.t.arange(
            0, cfg.head_dim, 2, device=hidden.device, dtype=self.t.float32) / cfg.head_dim))
        frequency = positions.float().unsqueeze(-1) * inverse
        angle = self.t.cat((frequency, frequency), -1)
        cos, sin = angle.cos().to(hidden.dtype).unsqueeze(-2), angle.sin().to(hidden.dtype).unsqueeze(-2)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin, v

    def test_torch_matches_existing_bias_gqa_and_norm_rounding_for_all_layouts(self):
        api = helper_api(self)
        for dtype in (self.t.float32, self.t.float16, self.t.bfloat16):
            for bias in (False, True):
                with self.subTest(dtype=dtype, bias=bias):
                    model = self.model(dtype, bias=bias)
                    before = deepcopy(model.state_dict())
                    helper = api.prepare_qkv_rope(model, max_context=17, backend="torch")
                    self.assertEqual(helper.backend, "torch")
                    for shape in ((5, 8), (2, 3, 8)):
                        value = self.t.randn(shape, dtype=dtype)
                        positions = self.t.arange(value.numel() // 8).reshape(shape[:-1]) + 2
                        attn = model.blocks[0].decoder.self_attn
                        for actual, expected in zip(helper(attn, value, positions), self.expected(model, attn, value, positions)):
                            self.t.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(set(before), set(model.state_dict()))
                    for name, value in model.state_dict().items():
                        self.t.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_native_uses_half_rope_with_flattened_4d_inputs_and_preserves_math(self):
        api = helper_api(self)
        for dtype in (self.t.float16, self.t.bfloat16):
            model = self.model(dtype)
            native = NativeMath()
            helper = api.prepare_qkv_rope(model, max_context=8, backend="native", native_ops=native)
            value = self.t.randn((2, 3, 8), dtype=dtype)
            positions = self.t.tensor([[0, 1, 2], [6, 7, 9]])
            attn = model.blocks[1].decoder.self_attn
            for actual, expected in zip(helper(attn, value, positions), self.expected(model, attn, value, positions)):
                self.t.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(native.calls, [((1, 6, 2, 4), (1, 6, 1, 4), "half"),
                                            ((1, 6, 1, 4), (1, 6, 1, 4), "half")])

    def test_auto_keeps_joint_precision_change_opt_in_and_falls_back_without_api(self):
        api = helper_api(self)
        model = self.model()
        self.assertEqual(api.prepare_qkv_rope(model, max_context=8).backend, "torch")
        joint = FusedMath()
        self.assertEqual(api.prepare_qkv_rope(model, max_context=8, fused_op=joint).backend, "torch")
        native = NativeMath()
        self.assertEqual(api.prepare_qkv_rope(model, max_context=8, native_ops=native, fused_op=joint).backend, "native")
        self.assertEqual(joint.calls, [])

    def test_native_coefficients_are_contiguous_without_runtime_clone(self):
        api = helper_api(self)
        from torch.utils._python_dispatch import TorchDispatchMode

        model = self.model(self.t.float16)
        native = NativeMath()
        helper = api.prepare_qkv_rope(model, max_context=8, backend="native", native_ops=native)
        calls = []

        class Observed(TorchDispatchMode):
            def __torch_dispatch__(mode, func, types, args=(), kwargs=None):
                calls.append(str(func))
                return func(*args, **(kwargs or {}))

        with Observed():
            helper(model.blocks[0].decoder.self_attn,
                   self.t.randn(2, 3, 8, dtype=self.t.float16),
                   self.t.tensor([[0, 2, 3], [1, 4, 5]]))
        self.assertEqual(native.coefficients_contiguous, [(True, True), (True, True)])
        self.assertEqual(helper.table.shape, (2, helper.position_capacity, 4))
        self.assertEqual(calls.count("aten.index_select.default"), 1)
        self.assertNotIn("aten.clone.default", calls)

    def test_joint_kernel_receives_packed_projection_bias_once_and_fp32_norm_contract(self):
        api = helper_api(self)
        for dtype in (self.t.float16, self.t.bfloat16):
            model = self.model(dtype, head_dim=128)
            joint = FusedMath()
            helper = api.prepare_qkv_rope(model, max_context=8, backend="triton", fused_op=joint)
            value = self.t.randn((2, 3, 256), dtype=dtype)
            positions = self.t.tensor([[0, 1, 2], [6, 7, 9]])
            attn = model.blocks[0].decoder.self_attn
            actual = helper(attn, value, positions)
            # The independent float reference retains projection rounding, but
            # has one final cast after norm and rotary like the Triton kernel.
            qkv = self.t.nn.functional.linear(value, attn._mtp_qkv_weight, attn._mtp_qkv_bias).flatten(0, 1)
            inv = 1.0 / (10000 ** (self.t.arange(0, 128, 2).float() / 128))
            frequencies = positions.flatten().float()[:, None] * inv
            cache = self.t.cat((frequencies.cos(), frequencies.sin()), -1)
            expected = FusedMath()(qkv, attn.q_norm.weight, attn.k_norm.weight,
                                   cache.unsqueeze(0).expand(3, -1, -1),
                                   2, 1, 128, 1e-5, [64, 0, 0], rope_dim=128)
            for result, wanted, count in zip(actual, expected[:3], (2, 1, 1)):
                self.t.testing.assert_close(result, wanted.view(2, 3, count, 128), rtol=0, atol=0)
            self.assertEqual(joint.calls[0], (self.t.Size((6, 512)), self.t.Size((3, 6, 128)), [64, 0, 0], False, 128, None, None, False))

    def test_position_shape_dtype_and_precomputed_bounds(self):
        api = helper_api(self)
        model = self.model()
        helper = api.prepare_qkv_rope(model, max_context=8, backend="torch")
        attn = model.blocks[0].decoder.self_attn
        value = self.t.randn(1, 8)
        # Supports the highest clamped padding anchor plus the deepest head.
        helper(attn, value, self.t.tensor([8 - 1 + model.num_heads]))
        for positions in (self.t.tensor([-1]), self.t.tensor([helper.position_capacity])):
            with self.assertRaises((IndexError, RuntimeError)):
                helper(attn, value, positions)
        with self.assertRaisesRegex(ValueError, "shape"):
            helper(attn, value, self.t.zeros(1, 1, dtype=self.t.long))
        with self.assertRaisesRegex((TypeError, ValueError), "int"):
            helper(attn, value, self.t.zeros(1))

    def test_projection_once_and_precomputed_rope_has_no_runtime_trigonometry(self):
        api = helper_api(self)
        from torch.utils._python_dispatch import TorchDispatchMode

        model = self.model()
        helper = api.prepare_qkv_rope(model, max_context=8, backend="torch")
        calls = []

        class Observed(TorchDispatchMode):
            def __torch_dispatch__(mode, func, types, args=(), kwargs=None):
                calls.append(str(func))
                return func(*args, **(kwargs or {}))

        with Observed():
            helper(model.blocks[0].decoder.self_attn, self.t.randn(2, 8), self.t.tensor([0, 2]))
        self.assertEqual(sum(name in ("aten.linear.default", "aten.addmm.default", "aten.mm.default") for name in calls), 1)
        self.assertFalse(any(name.startswith(("aten.sin.", "aten.cos.", "aten._local_scalar_dense.")) for name in calls))

    def test_startup_fails_for_unprepared_heads_missing_api_or_unsupported_joint_geometry(self):
        api = helper_api(self)
        model = self.model()
        with self.assertRaisesRegex(ValueError, "backend"):
            api.prepare_qkv_rope(model, max_context=8, backend="unknown")
        with self.assertRaisesRegex(ValueError, "rotary"):
            api.prepare_qkv_rope(model, max_context=8, backend="native", native_ops=SimpleNamespace())
        with self.assertRaisesRegex(ValueError, "geometry"):
            api.prepare_qkv_rope(model, max_context=8, backend="triton", fused_op=FusedMath())
        del model.blocks[0].decoder.self_attn._mtp_qkv_weight
        with self.assertRaisesRegex(ValueError, "prepare_fused_heads"):
            api.prepare_qkv_rope(model, max_context=8)

    def test_joint_startup_rejects_api_drift_and_different_norm_epsilons(self):
        api = helper_api(self)
        model = self.model(self.t.float16, head_dim=128)
        with self.assertRaisesRegex(ValueError, "signature"):
            api.prepare_qkv_rope(model, max_context=8, backend="triton",
                                 fused_op=lambda qkv: qkv)
        attn = model.blocks[1].decoder.self_attn
        attn.k_norm.eps *= 2
        with self.assertRaisesRegex(ValueError, "epsilon"):
            api.prepare_qkv_rope(model, max_context=8, backend="triton",
                                 fused_op=FusedMath())


def npu_available():
    if importlib.util.find_spec("torch_npu") is None:
        return False
    import torch
    import torch_npu  # noqa: F401

    return torch.npu.is_available()


@unittest.skipUnless(npu_available(), "requires torch-npu and an available NPU")
class ActualQKVRopeTests(unittest.TestCase):
    def test_native_graph_replay_uses_changed_positions_and_values(self):
        import torch
        from asr_mtp_fused import prepare_fused_heads
        from asr_mtp_heads import BatchedMTPHeads

        api = helper_api(self)
        cfg = SimpleNamespace(hidden_size=256, intermediate_size=512,
                              num_attention_heads=2, num_key_value_heads=1,
                              head_dim=128, attention_bias=True, rms_norm_eps=1e-5,
                              hidden_act="silu", rope_theta=10000, rope_scaling=None)
        model = BatchedMTPHeads(cfg, 2).to("npu", torch.float16).eval().requires_grad_(False)
        prepare_fused_heads(model)
        native = api.prepare_qkv_rope(model, max_context=32, backend="native")
        reference = api.prepare_qkv_rope(model, max_context=32, backend="torch")
        hidden = torch.randn(8, 256, device="npu", dtype=torch.float16)
        positions = torch.arange(8, device="npu")
        attn = model.blocks[0].decoder.self_attn
        with torch.inference_mode():
            native(attn, hidden, positions)
            native(attn, hidden, positions)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs = native(attn, hidden, positions)
            for offset in (3, 9):
                hidden.normal_()
                positions.copy_(torch.arange(8, device="npu") + offset)
                graph.replay()
                for actual, expected in zip(outputs, reference(attn, hidden, positions)):
                    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_joint_graph_replay_compares_real_triton_kernel_to_torch_reference(self):
        import torch
        from asr_mtp_fused import prepare_fused_heads
        from asr_mtp_heads import BatchedMTPHeads

        if importlib.util.find_spec("vllm_ascend") is None:
            self.skipTest("requires the installed vllm-ascend Triton operator")
        import vllm_ascend.ops  # noqa: F401

        api = helper_api(self)
        cfg = SimpleNamespace(hidden_size=256, intermediate_size=512,
                              num_attention_heads=2, num_key_value_heads=1,
                              head_dim=128, attention_bias=True, rms_norm_eps=1e-5,
                              hidden_act="silu", rope_theta=10000, rope_scaling=None)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = BatchedMTPHeads(cfg, 2).to("npu", dtype).eval().requires_grad_(False)
                prepare_fused_heads(model)
                joint = api.prepare_qkv_rope(model, max_context=32, backend="triton")
                reference = api.prepare_qkv_rope(model, max_context=32, backend="torch")
                hidden = torch.randn(8, 256, device="npu", dtype=dtype)
                positions = torch.arange(8, device="npu")
                attn = model.blocks[0].decoder.self_attn
                with torch.inference_mode():
                    joint(attn, hidden, positions)
                    joint(attn, hidden, positions)
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        outputs = joint(attn, hidden, positions)
                    for offset in (3, 9):
                        hidden.normal_()
                        positions.copy_(torch.arange(8, device="npu") + offset)
                        graph.replay()
                        for actual, expected in zip(outputs, reference(attn, hidden, positions)):
                            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
