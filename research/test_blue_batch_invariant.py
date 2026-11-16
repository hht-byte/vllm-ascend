"""CPU contract tests for the isolated AscendC FIA compatibility bridge."""

import ast
import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).with_name("blue_batch_invariant.py")

# Exact initializer from frozen Ascend 1cdb8c4; executed after transforming it.
STOCK_SOURCE = '''import os
_batch_invariant_LIB = None

def enable_batch_invariant_mode():
    global _batch_invariant_LIB
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    logger.debug(
        "Batch-invariant op registration: Triton=%s, AscendC=%s",
        HAS_TRITON,
        HAS_ASCENDC_BATCH_INVARIANT,
    )

    # Register operators only implemented in triton.
    if HAS_TRITON:
        _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::softmax", softmax_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::_softmax", softmax_batch_invariant, "NPU")

    # Register operators implemented in Ascend batch-invariant ops in priority.
    if HAS_ASCENDC_BATCH_INVARIANT:
        _batch_invariant_LIB.impl("aten::mm", torch.ops.batch_invariant_ops.npu_mm_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::matmul", torch.ops.batch_invariant_ops.npu_matmul_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::sum", torch.ops.batch_invariant_ops.npu_reduce_sum_batch_invariant, "NPU")
        # torch_npu.npu_fused_infer_attention_score is a function of torch_npu, not a torch.ops.Operator,
        # so we need to patch it directly.
        torch_npu.npu_fused_infer_attention_score = (
            torch.ops.batch_invariant_ops.npu_fused_infer_attention_score_batch_invariant
        )
        # patch npu_add_rms_norm to ensure batch invariant.
        torch_npu.npu_add_rms_norm = add_rms_norm
        # torch.sum can't be replaced by dispatch logic, so we patch it directly.
        torch.sum = reduce_sum

    # register triton implementations if ascendc is not available.
    elif HAS_TRITON:
        _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "NPU")
        _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, "NPU")

        # linear call matmul internally, so register linear only when ascendc
        # is not available. it will get better performance with ascendc.
        _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, "NPU")
'''


class FakeOps:
    """Model the four explicit 2.0 extension names and named output aliases."""

    def __init__(self):
        self.calls = []
        self.result = (object(), object())
        self.workspace = object()

    def npu_fused_infer_attention_score_batch_invariant(self, *args, **kwargs):
        self.calls.append(("default", args, kwargs))
        return self.result

    def npu_fused_infer_attention_score_batch_invariant_out(
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

    def _npu_fused_infer_attention_score_batch_invariant_get_max_workspace(
        self, *args, **kwargs,
    ):
        self.calls.append(("workspace", args, kwargs))
        return self.workspace

    def _npu_fused_infer_attention_score_batch_invariant_infer_output(
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
            "npu_fused_infer_attention_score_batch_invariant_out",
            "_npu_fused_infer_attention_score_batch_invariant_get_max_workspace",
            "_npu_fused_infer_attention_score_batch_invariant_infer_output",
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
        self.ops.npu_fused_infer_attention_score_batch_invariant_out = None
        with self.assertRaises((TypeError, ValueError)):
            self.install()
        self.assertIs(self.npu.npu_fused_infer_attention_score, self.original)

    def test_v1_only_namespace_is_rejected_before_torchnpu_mutation(self):
        legacy = SimpleNamespace(
            npu_fused_infer_attention_score_batch_invariant=(
                self.ops.npu_fused_infer_attention_score_batch_invariant),
            npu_fused_infer_attention_score_out_batch_invariant=(
                self.ops.npu_fused_infer_attention_score_batch_invariant_out),
            npu_fused_infer_attention_score_get_max_workspace_batch_invariant=(
                self.ops._npu_fused_infer_attention_score_batch_invariant_get_max_workspace),
            npu_fused_infer_attention_score_infer_output_batch_invariant=(
                self.ops._npu_fused_infer_attention_score_batch_invariant_infer_output),
        )
        with self.assertRaises((TypeError, ValueError)):
            self.module.install_fia_bridge(self.npu, legacy)
        self.assertIs(self.npu.npu_fused_infer_attention_score, self.original)
        self.assertIs(self.npu._npu_fused_infer_attention_score_get_max_workspace,
                      self.old_workspace)
        self.assertEqual(self.ops.calls, [])

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


class MatmulSourceTransformTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("blue_batch_invariant", SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.assertTrue(callable(getattr(self.module, "make_batch_invariant_matmul_patch", None)),
                        "Explicit matmul backend transform is missing")
        self.library_calls, self.registrations = [], []
        self.ops = SimpleNamespace(**{name: object() for name in (
            'npu_mm_batch_invariant', 'npu_matmul_batch_invariant',
            'npu_reduce_sum_batch_invariant',
            'npu_fused_infer_attention_score_batch_invariant',
        )})
        self.old_fia, self.old_rms, self.old_sum = object(), object(), object()
        self.npu = SimpleNamespace(npu_fused_infer_attention_score=self.old_fia,
                                   npu_add_rms_norm=self.old_rms)
        self.dim_sum_calls = []
        self.dim_sum_result = object()
        def dim_sum(x, dim, keepdim, *, dtype=None):
            self.dim_sum_calls.append((x, dim, keepdim, dtype))
            return self.dim_sum_result
        self.torch = SimpleNamespace(ops=SimpleNamespace(batch_invariant_ops=self.ops,
            aten=SimpleNamespace(sum=SimpleNamespace(dim_IntList=dim_sum))),
                                     sum=self.old_sum)
        self.functions = {name: object() for name in (
            'addmm_batch_invariant', 'bmm_batch_invariant', 'softmax_batch_invariant',
            'mm_batch_invariant', 'matmul_batch_invariant', 'linear_batch_invariant',
            'add_rms_norm', 'reduce_sum',
        )}

    def prepare(self, backend=None, *, has_triton=True, has_ascendc=True, fail_op=None):
        changed = self.module.make_batch_invariant_matmul_patch(STOCK_SOURCE)
        tree = ast.parse(changed)
        # The real stock initializer and its injected module-state assignments
        # execute with fake external operator registrations; no Torch is loaded.
        tree.body = [node for node in tree.body if not isinstance(node, ast.Import)]
        owner = self
        class Library:
            def __init__(self, *args):
                owner.library_calls.append(args)
            def impl(self, name, function, key):
                if name == fail_op:
                    raise RuntimeError('registered operator failed')
                owner.registrations.append((name, function, key))
        self.torch.library = SimpleNamespace(Library=Library)
        environment = {} if backend is None else {
            'LLM_OPT_BATCH_INVARIANT_MATMUL_BACKEND': backend}
        namespace = dict(self.functions, torch=self.torch, torch_npu=self.npu,
            os=SimpleNamespace(environ=environment),
            logger=SimpleNamespace(debug=lambda *args: None),
            HAS_TRITON=has_triton, HAS_ASCENDC_BATCH_INVARIANT=has_ascendc)
        def helper_import(name, *args, **kwargs):
            if name != 'vllm_ascend.patch.worker.blue_batch_invariant':
                raise AssertionError('Unexpected transformed initializer import')
            return self.module
        namespace['__builtins__'] = {**vars(builtins), '__import__': helper_import}
        exec(compile(tree, '<transformed frozen batch initializer>', 'exec'), namespace)
        return namespace

    def test_backend_marker_is_unconfirmed_before_registration(self):
        namespace = self.prepare('triton')
        self.assertIsNone(namespace['BLUE_BATCH_INVARIANT_MATMUL_BACKEND'])
        self.assertEqual(self.library_calls, [])
        self.assertEqual(self.registrations, [])

    def test_default_selects_ascendc_mm_and_preserves_every_other_registration(self):
        namespace = self.prepare()
        namespace['enable_batch_invariant_mode']()
        self.assertEqual(self.registrations[:-1], [
            ('aten::addmm', self.functions['addmm_batch_invariant'], 'NPU'),
            ('aten::bmm', self.functions['bmm_batch_invariant'], 'NPU'),
            ('aten::softmax', self.functions['softmax_batch_invariant'], 'NPU'),
            ('aten::_softmax', self.functions['softmax_batch_invariant'], 'NPU'),
            ('aten::mm', self.ops.npu_mm_batch_invariant, 'NPU'),
            ('aten::matmul', self.ops.npu_matmul_batch_invariant, 'NPU'),
        ])
        self.assertEqual(self.registrations[-1][::2], ('aten::sum', 'NPU'))
        sample = SimpleNamespace(dim=lambda: 2)
        self.assertIs(self.registrations[-1][1](sample), self.dim_sum_result)
        self.assertEqual(self.dim_sum_calls, [(sample, [0, 1], False, None)])
        self.assertIs(self.npu.npu_fused_infer_attention_score,
                      self.ops.npu_fused_infer_attention_score_batch_invariant)
        self.assertIs(self.npu.npu_add_rms_norm, self.functions['add_rms_norm'])
        self.assertIs(self.torch.sum, self.functions['reduce_sum'])
        self.assertEqual(namespace['BLUE_BATCH_INVARIANT_MATMUL_BACKEND'], 'ascendc')

    def test_explicit_triton_changes_only_mm_matmul_and_adds_no_linear_override(self):
        namespace = self.prepare('triton')
        namespace['enable_batch_invariant_mode']()
        functions = {name: fn for name, fn, key in self.registrations}
        self.assertIs(functions['aten::mm'], self.functions['mm_batch_invariant'])
        self.assertIs(functions['aten::matmul'], self.functions['matmul_batch_invariant'])
        sample = SimpleNamespace(dim=lambda: 1)
        requested_dtype = object()
        self.assertIs(functions['aten::sum'](sample, dtype=requested_dtype), self.dim_sum_result)
        self.assertEqual(self.dim_sum_calls, [(sample, [0], False, requested_dtype)])
        self.assertNotIn('aten::linear', functions)
        self.assertEqual(len(functions), 7)
        self.assertIs(self.npu.npu_fused_infer_attention_score,
                      self.ops.npu_fused_infer_attention_score_batch_invariant)
        self.assertIs(self.npu.npu_add_rms_norm, self.functions['add_rms_norm'])
        self.assertIs(self.torch.sum, self.functions['reduce_sum'])
        self.assertEqual(namespace['BLUE_BATCH_INVARIANT_MATMUL_BACKEND'], 'triton')

    def test_unknown_selector_fails_before_any_registration_or_torch_mutation(self):
        for backend in ('', 'auto', 'TRITON', 'cpu'):
            with self.subTest(backend=backend):
                namespace = self.prepare(backend)
                with self.assertRaises(ValueError):
                    namespace['enable_batch_invariant_mode']()
                self.assertEqual(self.library_calls, [])
                self.assertEqual(self.registrations, [])
                self.assertIsNone(namespace['BLUE_BATCH_INVARIANT_MATMUL_BACKEND'])
                self.assertIs(self.npu.npu_fused_infer_attention_score, self.old_fia)
                self.assertIs(self.torch.sum, self.old_sum)

    def test_requested_triton_without_triton_fails_before_library_creation(self):
        namespace = self.prepare('triton', has_triton=False)
        with self.assertRaises(ValueError):
            namespace['enable_batch_invariant_mode']()
        self.assertEqual(self.library_calls, [])
        self.assertEqual(self.registrations, [])
        self.assertIsNone(namespace['_batch_invariant_LIB'])
        self.assertIsNone(namespace['BLUE_BATCH_INVARIANT_MATMUL_BACKEND'])

    def test_ascendc_does_not_require_unimported_triton_functions(self):
        namespace = self.prepare('ascendc', has_triton=False)
        del namespace['mm_batch_invariant'], namespace['matmul_batch_invariant']
        namespace['enable_batch_invariant_mode']()
        self.assertEqual(self.registrations[0],
                         ('aten::mm', self.ops.npu_mm_batch_invariant, 'NPU'))
        self.assertEqual(namespace['BLUE_BATCH_INVARIANT_MATMUL_BACKEND'], 'ascendc')

    def test_failed_registration_never_sets_successful_backend_marker(self):
        namespace = self.prepare('triton', fail_op='aten::sum')
        with self.assertRaises(RuntimeError):
            namespace['enable_batch_invariant_mode']()
        self.assertIsNone(namespace['BLUE_BATCH_INVARIANT_MATMUL_BACKEND'])
        self.assertIs(self.npu.npu_fused_infer_attention_score, self.old_fia)

    def test_missing_default_sum_dim_overload_fails_before_library_or_mutations(self):
        namespace = self.prepare('triton')
        del self.torch.ops.aten.sum.dim_IntList
        with self.assertRaises(ValueError):
            namespace['enable_batch_invariant_mode']()
        self.assertEqual(self.library_calls, [])
        self.assertEqual(self.registrations, [])
        self.assertIsNone(namespace['BLUE_BATCH_INVARIANT_MATMUL_BACKEND'])
        self.assertIs(self.npu.npu_fused_infer_attention_score, self.old_fia)
        self.assertIs(self.torch.sum, self.old_sum)

    def test_stock_triton_only_branch_reports_actual_backend(self):
        namespace = self.prepare(has_ascendc=False)
        namespace['enable_batch_invariant_mode']()
        self.assertEqual(namespace['BLUE_BATCH_INVARIANT_MATMUL_BACKEND'], 'triton')
        self.assertIs(self.registrations[-1][1], self.functions['linear_batch_invariant'])
        self.assertIs(self.npu.npu_fused_infer_attention_score, self.old_fia)

    def test_no_available_backend_is_never_reported_as_registered(self):
        namespace = self.prepare(has_triton=False, has_ascendc=False)
        namespace['enable_batch_invariant_mode']()
        self.assertIsNone(namespace['BLUE_BATCH_INVARIANT_MATMUL_BACKEND'])
        self.assertEqual(self.registrations, [])

    def test_changed_stock_registration_is_rejected(self):
        changed = STOCK_SOURCE.replace('aten::addmm', 'aten::addmm_changed')
        with self.assertRaises(ValueError):
            self.module.make_batch_invariant_matmul_patch(changed)

    def test_duplicate_function_is_rejected(self):
        changed = STOCK_SOURCE + STOCK_SOURCE[STOCK_SOURCE.index('def enable_'):]
        with self.assertRaises(ValueError):
            self.module.make_batch_invariant_matmul_patch(changed)

    def test_duplicate_lib_anchor_is_rejected(self):
        with self.assertRaises(ValueError):
            self.module.make_batch_invariant_matmul_patch(
                STOCK_SOURCE + '\n_batch_invariant_LIB = None\n')

    def test_already_transformed_source_is_rejected(self):
        changed = self.module.make_batch_invariant_matmul_patch(STOCK_SOURCE)
        with self.assertRaises(ValueError):
            self.module.make_batch_invariant_matmul_patch(changed)

    def test_missing_os_import_is_rejected(self):
        with self.assertRaises(ValueError):
            self.module.make_batch_invariant_matmul_patch(
                STOCK_SOURCE.replace('import os\n', ''))


class DefaultSumContractTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('blue_batch_invariant', SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.assertTrue(callable(getattr(self.module, 'defaultSumCallable', None)),
                        'Default sum compatibility factory is missing')
        self.calls, self.result = [], object()
        def dimension_sum(x, dim, keepdim, *, dtype=None):
            self.calls.append((x, dim, keepdim, dtype))
            return self.result
        def forbidden(*args, **kwargs):
            raise AssertionError('Default/global sum recursion or tensor transfer')
        self.forbidden = forbidden
        self.sum_ops = SimpleNamespace(dim_IntList=dimension_sum, default=forbidden)
        self.torch = SimpleNamespace(ops=SimpleNamespace(aten=SimpleNamespace(sum=self.sum_ops)),
                                     sum=forbidden)

    def test_int_bool_float_rank_scalar_and_empty_inputs_use_explicit_all_dimensions(self):
        operation = self.module.defaultSumCallable(self.torch)
        for label, rank, shape, expected in (
            ('int64', 1, (3,), [0]), ('bool', 2, (2, 3), [0, 1]),
            ('float32', 3, (2, 3, 4), [0, 1, 2]),
            ('float64-scalar', 0, (), [0]), ('float32-empty', 2, (0, 3), [0, 1]),
        ):
            sample = SimpleNamespace(dim=lambda rank=rank: rank, dtype=label, shape=shape,
                                     cpu=self.forbidden, to=self.forbidden, clone=self.forbidden)
            with self.subTest(label=label):
                self.assertIs(operation(sample), self.result)
                self.assertEqual(self.calls[-1], (sample, expected, False, None))

    def test_requested_dtype_is_forwarded_by_identity_and_only_as_keyword(self):
        operation = self.module.defaultSumCallable(self.torch)
        sample, dtype = SimpleNamespace(dim=lambda: 1), object()
        self.assertIs(operation(sample, dtype=dtype), self.result)
        self.assertIs(self.calls[-1][3], dtype)
        with self.assertRaises(TypeError): operation(sample, dtype)
        with self.assertRaises(TypeError): operation(sample, dim=0)
        with self.assertRaises(TypeError): operation(sample, out=object())
        self.assertEqual(len(self.calls), 1)

    def test_factory_captures_untouched_dimension_overload_without_default_recursion(self):
        operation = self.module.defaultSumCallable(self.torch)
        self.sum_ops.dim_IntList = self.forbidden
        sample = SimpleNamespace(dim=lambda: 0)
        self.assertIs(operation(sample), self.result)
        self.assertEqual(self.calls, [(sample, [0], False, None)])

    def test_unavailable_dimension_overload_is_rejected_during_factory_construction(self):
        for value in (None, object()):
            self.sum_ops.dim_IntList = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.module.defaultSumCallable(self.torch)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
