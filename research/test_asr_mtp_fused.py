"""Frozen-head packing and local native dispatch; CPU is not NPU acceptance."""

import importlib.util
import unittest
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace


def fused_api(test):
    try:
        import asr_mtp_fused
    except ImportError:
        test.fail("native MTP head fusion is not implemented")
    return asr_mtp_fused


class ReferenceNativeOps:
    """Execute real Torch math while exposing the native call boundary."""

    def __init__(self, torch):
        self.t = torch
        self.calls = Counter()
        self.epsilons = []

    def npu_rms_norm(self, value, weight, epsilon):
        self.calls["rms_norm"] += 1
        self.epsilons.append(epsilon)
        output = self.t.nn.functional.rms_norm(value, weight.shape, weight, epsilon)
        return output, self.t.rsqrt(
            value.float().square().mean(-1, keepdim=True) + epsilon
        )

    def npu_add_rms_norm(self, left, right, weight, epsilon):
        self.calls["add_rms_norm"] += 1
        self.epsilons.append(epsilon)
        residual = left + right
        output = self.t.nn.functional.rms_norm(residual, weight.shape, weight, epsilon)
        rstd = self.t.rsqrt(residual.float().square().mean(-1, keepdim=True) + epsilon)
        return output, rstd, residual

    def npu_swiglu(self, packed):
        self.calls["swiglu"] += 1
        gate, up = packed.chunk(2, dim=-1)
        return self.t.nn.functional.silu(gate) * up


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Torch")
class FusedHeadsTests(unittest.TestCase):
    def setUp(self):
        import torch
        from asr_mtp_heads import BatchedMTPHeads

        self.t = torch
        torch.set_num_threads(1)
        torch.manual_seed(412)
        self.config = SimpleNamespace(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            attention_bias=True,
            rms_norm_eps=3e-5,
            hidden_act="silu",
            rope_theta=10000,
            rope_scaling=None,
        )
        self.model = BatchedMTPHeads(self.config, 2).eval().requires_grad_(False)

    def test_unprepared_helpers_preserve_original_projections_mlp_and_residual(self):
        api = fused_api(self)
        block = self.model.blocks[0]
        attention, mlp = block.decoder.self_attn, block.decoder.mlp
        value, residual = self.t.randn(2, 3, 8), self.t.randn(2, 3, 8)
        for result, expected in zip(
            api.project_qkv(attention, value),
            (attention.q_proj(value), attention.k_proj(value), attention.v_proj(value)),
        ):
            self.t.testing.assert_close(result, expected)
        expected_mlp = mlp.down_proj(
            self.t.nn.functional.silu(mlp.gate_proj(value)) * mlp.up_proj(value)
        )
        self.t.testing.assert_close(api.mlp_forward(mlp, value), expected_mlp)
        normalized, combined = api.add_rms_norm(block.output_norm, value, residual)
        self.t.testing.assert_close(combined, value + residual)
        self.t.testing.assert_close(normalized, block.output_norm(value + residual))

    def test_qkv_and_gate_up_packing_preserve_bias_gqa_and_multidimensional_math(self):
        api = fused_api(self)
        for bias in (False, True):
            with self.subTest(bias=bias):
                from asr_mtp_heads import BatchedMTPHeads

                config = SimpleNamespace(
                    **{**vars(self.config), "attention_bias": bias}
                )
                model = BatchedMTPHeads(config, 2).eval().requires_grad_(False)
                expected = deepcopy(model)
                self.assertTrue(api.prepare_fused_heads(model))
                for shape in ((5, 8), (2, 3, 8)):
                    value = self.t.randn(shape)
                    for block, original in zip(model.blocks, expected.blocks):
                        attention = block.decoder.self_attn
                        old = original.decoder.self_attn
                        for actual, wanted in zip(
                            api.project_qkv(attention, value),
                            (old.q_proj(value), old.k_proj(value), old.v_proj(value)),
                        ):
                            self.t.testing.assert_close(actual, wanted)
                        old_mlp = original.decoder.mlp
                        wanted = old_mlp.down_proj(
                            self.t.nn.functional.silu(old_mlp.gate_proj(value))
                            * old_mlp.up_proj(value)
                        )
                        self.t.testing.assert_close(
                            api.mlp_forward(block.decoder.mlp, value), wanted
                        )

    def test_packed_helpers_reduce_real_projection_matrix_operations(self):
        api = fused_api(self)
        from torch.utils._python_dispatch import TorchDispatchMode

        calls = []

        class MatrixCalls(TorchDispatchMode):
            def __torch_dispatch__(mode, func, types, args=(), kwargs=None):
                if func in (self.t.ops.aten.mm.default, self.t.ops.aten.addmm.default):
                    calls.append(func)
                return func(*args, **(kwargs or {}))

        api.prepare_fused_heads(self.model)
        block = self.model.blocks[0]
        value = self.t.randn(4, 8)
        with MatrixCalls():
            api.project_qkv(block.decoder.self_attn, value)
        self.assertEqual(len(calls), 1)
        calls.clear()
        with MatrixCalls():
            api.mlp_forward(block.decoder.mlp, value)
        self.assertEqual(len(calls), 2)

    def test_original_checkpoint_keys_and_storage_are_shared_after_packing(self):
        api = fused_api(self)
        before = deepcopy(self.model.state_dict())
        keys = set(before)
        api.prepare_fused_heads(self.model)
        self.assertEqual(set(self.model.state_dict()), keys)
        for name, value in self.model.state_dict().items():
            self.t.testing.assert_close(value, before[name])
        attention = self.model.blocks[0].decoder.self_attn
        q, k, v = (
            module.weight
            for module in (attention.q_proj, attention.k_proj, attention.v_proj)
        )
        self.assertEqual(q.untyped_storage().data_ptr(), k.untyped_storage().data_ptr())
        self.assertEqual(q.untyped_storage().data_ptr(), v.untyped_storage().data_ptr())
        biases = [
            module.bias
            for module in (attention.q_proj, attention.k_proj, attention.v_proj)
        ]
        self.assertEqual(len({x.untyped_storage().data_ptr() for x in biases}), 1)
        mlp = self.model.blocks[0].decoder.mlp
        self.assertEqual(
            mlp.gate_proj.weight.untyped_storage().data_ptr(),
            mlp.up_proj.weight.untyped_storage().data_ptr(),
        )
        original_parameters = {id(p) for p in self.model.parameters()}
        for module in (attention, mlp):
            self.assertTrue(list(module.buffers(recurse=False)))
            for buffer in module.buffers(recurse=False):
                self.assertTrue(
                    any(
                        buffer.untyped_storage().data_ptr()
                        == p.untyped_storage().data_ptr()
                        for p in module.parameters()
                    )
                )
        self.assertEqual(len(original_parameters), len(list(self.model.parameters())))

    def test_strict_checkpoint_roundtrip_and_default_reload_update_packed_views(self):
        api = fused_api(self)
        from asr_mtp_heads import BatchedMTPHeads

        api.prepare_fused_heads(self.model)
        saved = deepcopy(self.model.state_dict())
        name = "blocks.0.decoder.self_attn.q_proj.weight"
        saved[name].add_(0.25)
        self.model.load_state_dict(saved, strict=True)
        fresh = BatchedMTPHeads(self.config, 2).eval().requires_grad_(False)
        fresh.load_state_dict(saved, strict=True)
        value = self.t.randn(3, 8)
        actual = api.project_qkv(self.model.blocks[0].decoder.self_attn, value)
        wanted = fresh.blocks[0].decoder.self_attn
        for result, expected in zip(
            actual, (wanted.q_proj(value), wanted.k_proj(value), wanted.v_proj(value))
        ):
            self.t.testing.assert_close(result, expected)

    def test_native_dispatch_preserves_norm_parameters_residual_and_swiglu_order(self):
        api = fused_api(self)
        native = ReferenceNativeOps(self.t)
        expected = deepcopy(self.model)
        api.prepare_fused_heads(self.model, native_ops=native)
        for block, original in zip(self.model.blocks, expected.blocks):
            names = (
                "hidden_norm",
                "token_norm",
                "decoder.input_layernorm",
                "decoder.self_attn.q_norm",
                "decoder.self_attn.k_norm",
                "decoder.post_attention_layernorm",
                "output_norm",
            )
            for name in names:
                norm, old = block.get_submodule(name), original.get_submodule(name)
                value = self.t.randn(2, 3, *old.normalized_shape)
                self.t.testing.assert_close(norm(value), old(value))
                self.assertEqual(norm.eps, old.eps)
                self.assertEqual(norm.normalized_shape, old.normalized_shape)
            value, residual = self.t.randn(2, 3, 8), self.t.randn(2, 3, 8)
            output, combined = api.add_rms_norm(block.output_norm, value, residual)
            self.t.testing.assert_close(combined, value + residual)
            self.t.testing.assert_close(output, original.output_norm(value + residual))
            old_mlp = original.decoder.mlp
            wanted = old_mlp.down_proj(
                self.t.nn.functional.silu(old_mlp.gate_proj(value))
                * old_mlp.up_proj(value)
            )
            self.t.testing.assert_close(
                api.mlp_forward(block.decoder.mlp, value), wanted
            )
        self.assertEqual(native.calls, Counter(rms_norm=14, add_rms_norm=2, swiglu=2))
        self.assertEqual(set(native.epsilons), {self.config.rms_norm_eps})

    def test_disabled_preparation_is_unchanged_and_enabled_is_idempotent(self):
        api = fused_api(self)
        modules = {name: id(module) for name, module in self.model.named_modules()}
        parameters = {name: id(p) for name, p in self.model.named_parameters()}
        self.assertFalse(api.prepare_fused_heads(self.model, enabled=False))
        self.assertEqual(
            modules, {name: id(module) for name, module in self.model.named_modules()}
        )
        self.assertEqual(
            parameters, {name: id(p) for name, p in self.model.named_parameters()}
        )
        api.prepare_fused_heads(self.model)
        packed = {name: p.data_ptr() for name, p in self.model.named_parameters()}
        self.assertTrue(api.prepare_fused_heads(self.model))
        self.assertEqual(
            packed, {name: p.data_ptr() for name, p in self.model.named_parameters()}
        )

    def test_training_or_incomplete_native_api_fails_before_mutating_modules(self):
        api = fused_api(self)
        self.model.requires_grad_(True)
        modules = {name: id(module) for name, module in self.model.named_modules()}
        with self.assertRaisesRegex(ValueError, "frozen"):
            api.prepare_fused_heads(self.model)
        self.assertEqual(
            modules, {name: id(module) for name, module in self.model.named_modules()}
        )
        self.model.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "native"):
            api.prepare_fused_heads(
                self.model, native_ops=SimpleNamespace(npu_rms_norm=lambda: None)
            )
        self.assertEqual(
            modules, {name: id(module) for name, module in self.model.named_modules()}
        )


def npu_available():
    if importlib.util.find_spec("torch_npu") is None:
        return False
    import torch
    import torch_npu  # noqa: F401

    return torch.npu.is_available()


@unittest.skipUnless(npu_available(), "requires torch-npu and an available NPU")
class ActualNativeFusionTests(unittest.TestCase):
    def test_native_projection_norm_mlp_capture_and_changed_inputs(self):
        import torch
        from asr_mtp_heads import BatchedMTPHeads

        api = fused_api(self)
        cfg = SimpleNamespace(
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            attention_bias=True,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            rope_theta=10000,
            rope_scaling=None,
        )
        original = (
            BatchedMTPHeads(cfg, 1)
            .to("npu", torch.float16)
            .eval()
            .requires_grad_(False)
        )
        model = deepcopy(original)
        api.prepare_fused_heads(model)
        value = torch.randn(8, 128, device="npu", dtype=torch.float16)
        residual = torch.randn_like(value)
        output = torch.empty_like(value)

        def body():
            block = model.blocks[0]
            q, _, _ = api.project_qkv(block.decoder.self_attn, value)
            hidden = block.decoder.self_attn.q_norm(q.view(8, 2, 64)).reshape(8, 128)
            normalized, combined = api.add_rms_norm(block.output_norm, hidden, residual)
            output.copy_(api.mlp_forward(block.decoder.mlp, normalized) + combined)

        with torch.inference_mode():
            body()
            body()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                body()
            for _ in range(2):
                value.normal_()
                residual.normal_()
                graph.replay()
                block = original.blocks[0]
                q = block.decoder.self_attn.q_proj(value)
                hidden = block.decoder.self_attn.q_norm(q.view(8, 2, 64)).reshape(
                    8, 128
                )
                combined = hidden + residual
                wanted = block.decoder.mlp(block.output_norm(combined)) + combined
                torch.testing.assert_close(output, wanted, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
