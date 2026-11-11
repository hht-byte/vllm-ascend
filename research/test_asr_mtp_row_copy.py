"""Sparse row-copy contracts; CPU math/emulation is not NPU performance."""

import importlib
import importlib.util
import inspect
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def writer_module(test):
    try:
        import asr_mtp_row_copy
    except ImportError:
        test.fail("auxiliary row-copy writer is not implemented")
    return asr_mtp_row_copy


class TensorPointer:
    def __init__(self, values, offsets=0):
        self.values, self.offsets = values.reshape(-1), offsets

    def __add__(self, offsets):
        return TensorPointer(self.values, self.offsets + offsets)


class TorchLanguage:
    """Execute the actual pointer kernel with real CPU tensors and load/store masks."""

    def __init__(self, torch):
        self.torch, self.pid = torch, 0
        self.constexpr, self.int64 = int, torch.int64
        self.read_elements = self.written_elements = 0
        self.destination = None

    def program_id(self, axis):
        return self.pid

    def arange(self, start, end):
        return self.torch.arange(start, end)

    def load(self, pointer, mask, other=0):
        if pointer.values.data_ptr() == self.destination:
            raise AssertionError("row-copy must not read or scan the destination")
        result = self.torch.full(
            pointer.offsets.shape, other, dtype=pointer.values.dtype
        )
        result[mask] = pointer.values[pointer.offsets[mask]]
        self.read_elements += int(mask.sum())
        return result

    def store(self, pointer, values, mask):
        pointer.values[pointer.offsets[mask]] = values[mask]
        self.written_elements += int(mask.sum())


class TorchJIT:
    def __init__(self, fn, language):
        self.fn, self.language = fn, language

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs.pop("num_warps", None)
            kwargs.pop("num_stages", None)
            for pid in range(grid[0]):
                self.language.pid = pid
                self.fn(*(TensorPointer(arg) for arg in args), **kwargs)

        return launch


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Torch")
class RowWriterTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.t = torch
        torch.set_num_threads(1)

    def test_cpu_reference_never_imports_npu_or_triton(self):
        api = writer_module(self)
        with patch.object(
            api.importlib, "import_module", side_effect=AssertionError("import")
        ):
            writer, selected = api.prepare_row_writer("cpu", backend="auto")
        self.assertIsNone(writer)
        self.assertEqual(selected, "index_copy")
        with self.assertRaisesRegex(ValueError, "backend"):
            api.prepare_row_writer("cpu", backend="unknown")

    def test_auto_uses_native_api_without_loading_triton(self):
        api = writer_module(self)
        calls = []

        def scatter(cache, indices, updates):
            calls.append(indices.shape)
            cache.index_copy_(0, indices[:, 0], updates)

        with (
            patch.object(
                api, "_load_triton_writer", side_effect=AssertionError("load")
            ),
            patch.object(
                api.importlib,
                "import_module",
                return_value=SimpleNamespace(npu_scatter_nd_update_=scatter),
            ),
        ):
            writer, selected = api.prepare_row_writer(SimpleNamespace(type="npu"))
        cache, updates = self.t.zeros(31, 4), self.t.randn(3, 4)
        writer(cache, self.t.tensor([2, 30, 0]), updates)
        self.assertEqual(selected, "scatter_nd")
        self.assertEqual(calls, [self.t.Size([3, 1])])
        self.t.testing.assert_close(cache[[2, 30, 0]], updates)

    def test_explicit_scatter_never_loads_triton_and_explicit_missing_triton_errors(
        self,
    ):
        api = writer_module(self)
        with (
            patch.object(
                api, "_load_triton_writer", side_effect=AssertionError("load")
            ),
            patch.object(
                api.importlib,
                "import_module",
                return_value=SimpleNamespace(npu_scatter_nd_update_=lambda *args: None),
            ),
        ):
            _, selected = api.prepare_row_writer("npu", backend="scatter_nd")
        self.assertEqual(selected, "scatter_nd")
        with (
            patch.object(
                api, "_load_triton_writer", side_effect=ImportError("missing")
            ),
            self.assertRaisesRegex(RuntimeError, "triton"),
        ):
            api.prepare_row_writer("npu", backend="triton")

    def test_auto_stays_native_even_when_triton_is_available(self):
        api = writer_module(self)
        selected_writer = lambda *args: None
        with (
            patch.object(
                api, "_load_triton_writer", return_value=selected_writer
            ) as load_triton,
            patch.object(
                api.importlib,
                "import_module",
                return_value=SimpleNamespace(npu_scatter_nd_update_=lambda *args: None),
            ) as import_module,
        ):
            actual, selected = api.prepare_row_writer("npu")
        load_triton.assert_not_called()
        import_module.assert_called_once_with("torch_npu")
        self.assertIsNot(actual, selected_writer)
        self.assertEqual(selected, "scatter_nd")

    def test_explicit_triton_returns_writer_without_importing_native_api(self):
        api = writer_module(self)
        selected_writer = lambda *args: None
        with (
            patch.object(api, "_load_triton_writer", return_value=selected_writer),
            patch.object(
                api.importlib, "import_module", side_effect=AssertionError("import")
            ),
        ):
            actual, selected = api.prepare_row_writer("npu", backend="triton")
        self.assertIs(actual, selected_writer)
        self.assertEqual(selected, "triton")

    def test_selected_writer_runtime_failure_is_not_caught_or_retried(self):
        api = writer_module(self)

        def unavailable_at_run(*args):
            raise RuntimeError("launch failed")

        with patch.object(api, "_load_triton_writer", return_value=unavailable_at_run):
            writer, selected = api.prepare_row_writer("npu", backend="triton")
        self.assertEqual(selected, "triton")
        with self.assertRaisesRegex(RuntimeError, "launch failed"):
            writer(None, None, None)

    def make_emulated_writer(self):
        api = writer_module(self)
        language = TorchLanguage(self.t)
        functions = []

        def jit(fn):
            functions.append(fn)
            return TorchJIT(fn, language)

        triton = SimpleNamespace(
            jit=jit,
            cdiv=lambda n, d: (n + d - 1) // d,
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        )
        utility = ModuleType("vllm.triton_utils")
        utility.HAS_TRITON, utility.tl, utility.triton = True, language, triton
        package = ModuleType("vllm")
        package.__path__ = []
        with patch.dict(sys.modules, {"vllm": package, "vllm.triton_utils": utility}):
            sys.modules.pop("asr_mtp_row_copy_kernel", None)
            importlib.invalidate_caches()
            writer = api._load_triton_writer()
        self.assertIsNone(functions[0].__closure__)
        self.assertIn("def row_copy_kernel", inspect.getsource(functions[0]))
        return writer, language

    def test_actual_kernel_preserves_untouched_rows_and_unique_padding_sink(self):
        for width in (4, 128, 192, 1024):
            for dtype in (self.t.float32, self.t.float16, self.t.bfloat16):
                with self.subTest(width=width, dtype=dtype):
                    writer, language = self.make_emulated_writer()
                    cache = self.t.randn(67, width).to(dtype)
                    expected = cache.clone()
                    indices = self.t.tensor([0, 44, 2, 12, 9, 55, 62, 63, 64, 65, 66])
                    updates = self.t.randn(len(indices), width).to(dtype)
                    expected.index_copy_(0, indices, updates)
                    language.destination = cache.data_ptr()
                    writer(cache, indices[:, None], updates)
                    self.t.testing.assert_close(cache, expected, rtol=0, atol=0)
                    self.assertEqual(language.written_elements, len(indices) * width)
                    self.assertEqual(language.read_elements, len(indices) * (width + 1))

    def test_work_is_proportional_to_updates_and_does_not_read_device_indices_on_host(
        self,
    ):
        writer, language = self.make_emulated_writer()
        cache, updates = self.t.zeros(100003, 128), self.t.randn(19, 128)
        indices = self.t.tensor([i * 5000 for i in range(19)])
        language.destination = cache.data_ptr()
        # The emulation's own counters use mask.sum(); production launcher must
        # perform no item/tolist of cache addresses and no destination clone.
        with (
            patch.object(self.t.Tensor, "tolist", side_effect=AssertionError("tolist")),
            patch.object(self.t.Tensor, "clone", side_effect=AssertionError("clone")),
        ):
            writer(cache, indices, updates)
        self.assertEqual(language.written_elements, updates.numel())
        self.assertEqual(language.read_elements, updates.numel() + len(indices))

    def test_metadata_geometry_rejects_before_any_write(self):
        writer, language = self.make_emulated_writer()
        cache = self.t.zeros(17, 8)
        cases = (
            (self.t.tensor([0, 1]), self.t.zeros(2, 4)),
            (self.t.tensor([0.0, 1.0]), self.t.zeros(2, 8)),
            (self.t.tensor([[0, 1]]), self.t.zeros(2, 8)),
            (self.t.tensor([0, 1]), self.t.zeros(2, 8, dtype=self.t.float16)),
        )
        for indices, updates in cases:
            with self.subTest(indices=indices.shape), self.assertRaises(ValueError):
                writer(cache, indices, updates)
        self.assertEqual(language.written_elements, 0)


def npu_available():
    if importlib.util.find_spec("torch_npu") is None:
        return False
    import torch
    import torch_npu  # noqa: F401

    return torch.npu.is_available()


@unittest.skipUnless(npu_available(), "requires torch-npu and an available NPU")
class ActualRowCopyNPUTests(unittest.TestCase):
    def test_capture_replay_changes_sparse_addresses_without_touching_other_rows(self):
        import torch

        api = writer_module(self)
        writer, selected = api.prepare_row_writer("npu", backend="triton")
        self.assertEqual(selected, "triton")
        original = torch.randn(4096, 128, device="npu", dtype=torch.float16)
        cache = original.clone()
        addresses = torch.cat((torch.arange(97) * 31, torch.arange(5) + 4090)).to("npu")
        updates = torch.randn(len(addresses), 128, device="npu", dtype=torch.float16)
        base_cpu = original.cpu()
        with torch.inference_mode():
            writer(cache, addresses[:, None], updates)
            writer(cache, addresses[:, None], updates)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                writer(cache, addresses[:, None], updates)
            for shift in (0, 1, 2):
                cache.copy_(original)
                addresses.copy_(
                    torch.cat(
                        (torch.arange(97) * 31 + shift, torch.arange(5) + 4090)
                    ).to("npu")
                )
                updates.normal_()
                graph.replay()
                expected = base_cpu.clone()
                expected.index_copy_(0, addresses.cpu(), updates.cpu())
                torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)
