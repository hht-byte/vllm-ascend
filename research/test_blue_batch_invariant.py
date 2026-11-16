"""CPU contract tests for the isolated AscendC FIA compatibility bridge."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).with_name("blue_batch_invariant.py")


class FakeOps:
    """Model the four separately named extension schemas, not TorchNPU's out ABI."""

    def __init__(self):
        self.calls = []
        self.result = (object(), object())
        self.workspace = object()

    def npu_fused_infer_attention_score_batch_invariant(self, *args, **kwargs):
        self.calls.append(("default", args, kwargs))
        return self.result

    def npu_fused_infer_attention_score_out_batch_invariant(
        self, query, key, value, *, workspace=None, attention_out, softmax_lse,
        **kwargs,
    ):
        self.calls.append(("out", (query, key, value), {
            **kwargs, "workspace": workspace, "attention_out": attention_out,
            "softmax_lse": softmax_lse,
        }))
        attention_out[:] = ["attention-written"]
        softmax_lse[:] = ["lse-written"]
        return attention_out, softmax_lse

    def npu_fused_infer_attention_score_get_max_workspace_batch_invariant(
        self, *args, **kwargs,
    ):
        self.calls.append(("workspace", args, kwargs))
        return self.workspace

    def npu_fused_infer_attention_score_infer_output_batch_invariant(
        self, *args, **kwargs,
    ):
        self.calls.append(("infer_output", args, kwargs))
        return self.result


class FiaBridgeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SOURCE.is_file(), "FIA compatibility helper is missing")
        spec = importlib.util.spec_from_file_location("blue_batch_invariant", SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.original = lambda *args, **kwargs: "original"
        self.old_workspace = lambda *args, **kwargs: "old-workspace"
        self.unrelated = object()
        self.npu = SimpleNamespace(
            npu_fused_infer_attention_score=self.original,
            _npu_fused_infer_attention_score_get_max_workspace=self.old_workspace,
            unrelated=self.unrelated,
        )
        self.ops = FakeOps()

    def install(self):
        return self.module.install_fia_bridge(self.npu, self.ops)

    def test_default_preserves_arguments_and_returned_tuple(self):
        bridge = self.install()
        query, key, value = object(), object(), object()
        ends = [2, 5]
        actual = bridge(query, key, value, input_layout="TND",
                        actual_seq_lengths=ends, sparse_mode=3)
        self.assertIs(actual, self.ops.result)
        self.assertEqual(self.ops.calls, [("default", (query, key, value), {
            "input_layout": "TND", "actual_seq_lengths": ends, "sparse_mode": 3,
        })])
        self.assertIs(self.ops.calls[0][2]["actual_seq_lengths"], ends)

    def test_out_maps_list_outputs_to_named_aliases_without_copying(self):
        bridge = self.install()
        query, key, value, workspace = object(), object(), object(), object()
        attention, lse = [None], [None]
        actual = bridge.out(query=query, key=key, value=value, workspace=workspace,
                            out=[attention, lse], input_layout="TND", sparse_mode=3)
        self.assertIs(actual[0], attention)
        self.assertIs(actual[1], lse)
        self.assertEqual(attention, ["attention-written"])
        self.assertEqual(lse, ["lse-written"])
        self.assertEqual(self.ops.calls, [("out", (query, key, value), {
            "input_layout": "TND", "sparse_mode": 3, "workspace": workspace,
            "attention_out": attention, "softmax_lse": lse,
        })])

    def test_out_accepts_tuple_and_optional_workspace(self):
        bridge = self.install()
        attention, lse = [], []
        result = bridge.out(1, 2, 3, out=(attention, lse))
        self.assertIs(result[0], attention)
        self.assertIs(result[1], lse)
        self.assertIsNone(self.ops.calls[0][2]["workspace"])

    def test_workspace_old_module_name_uses_extension_with_exact_arguments(self):
        bridge = self.install()
        result = self.npu._npu_fused_infer_attention_score_get_max_workspace(
            query=1, key=2, value=3, actual_seq_lengths=[5], input_layout="TND")
        self.assertIs(result, self.ops.workspace)
        self.assertEqual(self.ops.calls, [("workspace", (), {
            "query": 1, "key": 2, "value": 3,
            "actual_seq_lengths": [5], "input_layout": "TND",
        })])
        self.assertIs(bridge.get_max_workspace(1, 2, 3), self.ops.workspace)

    def test_infer_output_uses_extension_schema_without_key_argument(self):
        bridge = self.install()
        result = bridge.infer_output(1, 3, input_layout="TND", num_heads=16)
        self.assertIs(result, self.ops.result)
        self.assertEqual(self.ops.calls, [("infer_output", (1, 3), {
            "input_layout": "TND", "num_heads": 16,
        })])

    def test_install_is_idempotent_and_preserves_unrelated_functions(self):
        self.assertFalse(self.module.fia_bridge_installed(self.npu))
        first = self.install()
        self.assertIs(self.install(), first)
        self.assertIs(self.npu.npu_fused_infer_attention_score, first)
        self.assertIs(self.npu.unrelated, self.unrelated)
        self.assertTrue(self.module.fia_bridge_installed(self.npu))
        self.assertEqual(self.ops.calls, [])

    def test_every_missing_extension_operation_fails_before_mutation(self):
        names = (
            "npu_fused_infer_attention_score_batch_invariant",
            "npu_fused_infer_attention_score_out_batch_invariant",
            "npu_fused_infer_attention_score_get_max_workspace_batch_invariant",
            "npu_fused_infer_attention_score_infer_output_batch_invariant",
        )
        for missing in names:
            with self.subTest(missing=missing):
                ops = SimpleNamespace(**{
                    name: getattr(self.ops, name) for name in names if name != missing
                })
                with self.assertRaises((TypeError, ValueError)):
                    self.module.install_fia_bridge(self.npu, ops)
                self.assertIs(self.npu.npu_fused_infer_attention_score, self.original)
                self.assertIs(
                    self.npu._npu_fused_infer_attention_score_get_max_workspace,
                    self.old_workspace,
                )
        self.assertEqual(self.ops.calls, [])

    def test_non_callable_extension_operation_fails_before_mutation(self):
        self.ops.npu_fused_infer_attention_score_out_batch_invariant = None
        with self.assertRaises((TypeError, ValueError)):
            self.install()
        self.assertIs(self.npu.npu_fused_infer_attention_score, self.original)

    def test_non_callable_existing_fia_fails_before_mutation(self):
        self.npu.npu_fused_infer_attention_score = None
        with self.assertRaises((TypeError, ValueError)):
            self.install()
        self.assertIsNone(self.npu.npu_fused_infer_attention_score)
        self.assertIs(self.npu._npu_fused_infer_attention_score_get_max_workspace,
                      self.old_workspace)

    def test_invalid_out_is_rejected_before_extension_runs(self):
        bridge = self.install()
        for outputs in (None, [], [object()], [object(), object(), object()], "xx"):
            with self.subTest(outputs=outputs):
                with self.assertRaises((TypeError, ValueError)):
                    bridge.out(1, 2, 3, out=outputs)
        self.assertEqual(self.ops.calls, [])

    def test_missing_out_is_rejected_before_extension_runs(self):
        bridge = self.install()
        with self.assertRaises(TypeError):
            bridge.out(1, 2, 3)
        self.assertEqual(self.ops.calls, [])

    def test_out_named_alias_conflict_is_rejected_before_extension_runs(self):
        bridge = self.install()
        for name in ("attention_out", "softmax_lse"):
            with self.subTest(name=name):
                with self.assertRaises(TypeError):
                    bridge.out(1, 2, 3, out=([], []), **{name: object()})
        self.assertEqual(self.ops.calls, [])

    def test_active_probe_detects_workspace_or_default_replacement(self):
        bridge = self.install()
        self.npu._npu_fused_infer_attention_score_get_max_workspace = self.old_workspace
        self.assertFalse(self.module.fia_bridge_installed(self.npu))
        self.npu._npu_fused_infer_attention_score_get_max_workspace = bridge.get_max_workspace
        self.npu.npu_fused_infer_attention_score = self.original
        self.assertFalse(self.module.fia_bridge_installed(self.npu))

    def test_different_extension_namespace_cannot_silently_rebind_live_bridge(self):
        first = self.install()
        with self.assertRaises((TypeError, ValueError)):
            self.module.install_fia_bridge(self.npu, FakeOps())
        self.assertIs(self.npu.npu_fused_infer_attention_score, first)

    def test_broken_live_bridge_is_not_silently_reported_idempotent(self):
        first = self.install()
        self.npu._npu_fused_infer_attention_score_get_max_workspace = self.old_workspace
        with self.assertRaises((TypeError, ValueError)):
            self.install()
        self.assertIs(self.npu.npu_fused_infer_attention_score, first)


if __name__ == "__main__":
    unittest.main()
