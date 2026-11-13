# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct CPU tests without importing vLLM or torch_npu."""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/sync_input_metadata.py"
spec = importlib.util.spec_from_file_location("sync_input_metadata", SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def runner():
    return NS(
        vllm_config=NS(additional_config={"asr_mtp": {}, "asr_sync_input_packing": True}),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        use_async_scheduling=False,
        use_async_spec_decode=False,
        pcp_size=1,
        dcp_size=1,
        _has_gdn=False,
        use_sparse=False,
        use_compress=False,
        is_hybrid=False,
    )


class CopyEvent:
    def __init__(self):
        self.done, self.waits, self.records = True, 0, 0

    def query(self):
        return self.done

    def synchronize(self):
        self.waits += 1
        self.done = True

    def record(self):
        self.records += 1
        self.done = False


class MetadataTests(unittest.TestCase):
    def test_rejection_reorder_shrink_and_mixed_prefill(self):
        s = mod.SyncInputMetadata(32, 8, "cpu", pin_memory=False)
        pointers = [v.data_ptr() for v in (s.positions, s.seq_lens, s.num_computed_tokens)]
        for counts, scheduled in [([12, 8, 0], [6, 4, 9]), ([10, 14], [3, 2]), ([0], [17])]:
            pos = np.concatenate([np.arange(c, c + n) for c, n in zip(counts, scheduled)])
            s.prepare(np.array(counts), np.array(counts) + scheduled, pos)
            self.assertEqual(s.positions[: len(pos)].tolist(), pos.tolist())
            self.assertEqual(s.seq_lens.tolist(), list(np.array(counts) + scheduled) + [0] * (8 - len(counts)))
            self.assertEqual(s.num_computed_tokens[: len(counts)].tolist(), counts)
            self.assertEqual(pointers, [v.data_ptr() for v in (s.positions, s.seq_lens, s.num_computed_tokens)])
            self.assertTrue(s.positions.is_contiguous())
            self.assertEqual(s.positions.dtype, torch.int64)
            self.assertEqual(s.seq_lens.dtype, torch.int32)
            self.assertEqual(s.last_copy_bytes, s.position_offset + len(pos) * 8)

    def test_inflight_host_storage(self):
        event = CopyEvent()
        s = mod.SyncInputMetadata(16, 4, "cpu", pin_memory=False, copy_event=event)
        s.prepare(np.array([9]), np.array([11]), np.array([9, 10]))
        self.assertEqual(event.waits, 0)
        s.prepare(np.array([1]), np.array([2]), np.array([1]))
        self.assertEqual(event.waits, 1)
        self.assertEqual(event.records, 2)
        event.done = True
        s.prepare(np.array([2]), np.array([3]), np.array([2]))
        self.assertEqual(event.waits, 1)

    def test_model_config_hybrid_is_rejected(self):
        r = runner()
        r.model_config = NS(is_hybrid=True)
        with self.assertRaises(ValueError):
            mod.enabled(r)

    def test_capacity_errors(self):
        s = mod.SyncInputMetadata(4, 2, "cpu", pin_memory=False)
        for c, q, p in [
            ([], [], []),
            ([0, 0, 0], [1, 1, 1], [0, 0, 0]),
            ([0], [5], list(range(5))),
            ([0], [1, 2], [0]),
        ]:
            with self.assertRaises(ValueError):
                s.prepare(np.array(c), np.array(q), np.array(p))

    def test_default_off(self):
        r = runner()
        r.vllm_config.additional_config = {}
        self.assertFalse(mod.enabled(r))

    def test_unsupported_modes(self):
        self.assertTrue(mod.enabled(runner()))
        for name, val in [
            ("use_async_scheduling", True),
            ("use_async_spec_decode", True),
            ("pcp_size", 2),
            ("dcp_size", 2),
            ("_has_gdn", True),
            ("use_sparse", True),
            ("use_compress", True),
            ("is_hybrid", True),
        ]:
            with self.subTest(name=name):
                r = runner()
                setattr(r, name, val)
                with self.assertRaises(ValueError):
                    mod.enabled(r)
        for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
            r = runner()
            setattr(r.parallel_config, name, 2)
            with self.assertRaises(ValueError):
                mod.enabled(r)
        r = runner()
        r.vllm_config.additional_config["asr_sync_input_packing"] = "true"
        with self.assertRaises(TypeError):
            mod.enabled(r)
        r = runner()
        del r.vllm_config.additional_config["asr_mtp"]
        with self.assertRaises(ValueError):
            mod.enabled(r)


class Buffer:
    def __init__(self, size, dtype):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.gpu = torch.zeros_like(self.cpu)
        self.np = self.cpu.numpy()
        self.copies = 0

    def copy_to_gpu(self, n):
        self.copies += 1
        self.gpu[:n].copy_(self.cpu[:n])


def prepare_region():
    source = SOURCE.with_name("model_runner_v1.py").read_text(encoding="utf-8-sig")
    start = source.index(
        "        valid_sampled_token_count_gpu = self.valid_sampled_token_count_gpu",
        source.index("    def _prepare_inputs("),
    )
    end = source.index("        # In async spec decode mode,", start)
    import textwrap

    code = textwrap.dedent(source[start:end])
    return compile(code, "<real _prepare_inputs region>", "exec")


class RunnerWiringTests(unittest.TestCase):
    def test_real_prepare_fast_and_fallback_match(self):
        code = prepare_region()
        stor = mod.SyncInputMetadata(32, 8, "cpu", pin_memory=False)
        for counts, scheduled in [([12, 8, 0], [6, 4, 9]), ([10, 14], [3, 2]), ([0], [17])]:
            outputs = []
            for fast in (False, True):
                n = len(counts)
                ids = np.repeat(np.arange(n), scheduled)
                offsets = np.concatenate([np.arange(q) for q in scheduled])
                pos = np.array(counts)[ids] + offsets
                r = NS(
                    _sync_input_metadata=stor if fast else None,
                    valid_sampled_token_count_gpu=None,
                    use_async_spec_decode=False,
                    pcp_size=1,
                    pcp_manager=None,
                    num_computed_tokens=stor.num_computed_tokens if fast else torch.zeros(8, dtype=torch.int32),
                    seq_lens=stor.seq_lens if fast else torch.zeros(8, dtype=torch.int32),
                    positions=stor.positions if fast else torch.zeros(32, dtype=torch.int64),
                    req_indices=Buffer(32, torch.int64),
                    query_pos=Buffer(32, torch.int64),
                    num_scheduled_tokens=Buffer(8, torch.int32),
                    optimistic_seq_lens_cpu=torch.tensor(
                        list(np.array(counts) + scheduled) + [0] * (8 - n), dtype=torch.int32
                    ),
                    input_batch=NS(num_computed_tokens_cpu_tensor=torch.tensor(counts, dtype=torch.int32)),
                )
                r.query_pos.np[: len(pos)] = offsets
                ns = dict(
                    self=r,
                    input_prepare_bucket=None,
                    num_reqs=n,
                    total_num_scheduled_tokens=len(pos),
                    req_indices=ids,
                    positions_np=pos,
                    num_scheduled_tokens=np.array(scheduled),
                    prev_req_id_to_index={},
                    torch=torch,
                    PCPAsyncSpecDecodeRebuildResult=lambda **kw: NS(**kw),
                )
                exec(code, ns)
                outputs.append((r.positions[: len(pos)].clone(), r.seq_lens.clone(), r.num_computed_tokens[:n].clone()))
                self.assertEqual(
                    r.req_indices.copies + r.query_pos.copies + r.num_scheduled_tokens.copies, 0 if fast else 3
                )
            for a, b in zip(*outputs):
                torch.testing.assert_close(a, b)

    def test_install_is_in_constructor_not_hot_path(self):
        tree = ast.parse(SOURCE.with_name("model_runner_v1.py").read_text(encoding="utf-8-sig"))
        runner_cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "NPUModelRunner")
        init = next(x for x in runner_cls.body if isinstance(x, ast.FunctionDef) and x.name == "__init__")
        self.assertTrue(
            any(
                isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "install_sync_input_metadata"
                for x in ast.walk(init)
            )
        )


class InstallTests(unittest.TestCase):
    def test_real_constructor_hook_binds_the_views(self):
        from types import MethodType
        from unittest.mock import patch

        r = runner()
        prepare = lambda self: None
        r._prepare_inputs = MethodType(prepare, r)
        r.positions = torch.zeros(16, dtype=torch.int64)
        r.seq_lens = torch.zeros(4, dtype=torch.int32)
        r.num_computed_tokens = torch.zeros(4, dtype=torch.int32)
        r.device, r.pin_memory = "cpu", False
        # Compile the actual installation call, including required keywords.
        tree = ast.parse(SOURCE.with_name("model_runner_v1.py").read_text(encoding="utf-8"))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "NPUModelRunner")
        init = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "__init__")
        stmt = next(
            x
            for x in init.body
            if isinstance(x, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr == "_sync_input_metadata" for t in x.targets)
        )
        code = compile(ast.fix_missing_locations(ast.Module(body=[stmt], type_ignores=[])), "<install>", "exec")
        with patch.object(torch, "npu", NS(Event=CopyEvent), create=True):
            exec(
                code,
                {"self": r, "install_sync_input_metadata": mod.install, "NPUModelRunner": NS(_prepare_inputs=prepare)},
            )
        self.assertIs(r.positions, r._sync_input_metadata.positions)
        self.assertIs(r.seq_lens, r._sync_input_metadata.seq_lens)
        self.assertIs(r.num_computed_tokens, r._sync_input_metadata.num_computed_tokens)

    def test_overridden_prepare_fails_without_rebinding(self):
        r = runner()
        from types import MethodType

        r._prepare_inputs = MethodType(lambda self: None, r)
        with self.assertRaisesRegex(ValueError, "overridden"):
            mod.install(r, prepare_inputs=lambda self: None)


try:
    import torch_npu  # noqa: F401

    NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    NPU_AVAILABLE = False


@unittest.skipUnless(NPU_AVAILABLE, "requires Ascend NPU")
class NpuGraphTests(unittest.TestCase):
    def test_replay_reads_new_metadata_after_shrink_and_reorder(self):
        s = mod.SyncInputMetadata(32, 8, "npu", pin_memory=True, copy_event=torch.npu.Event())
        output = torch.empty(32, dtype=torch.int64, device="npu")
        seqout = torch.empty_like(s.seq_lens)

        def body():
            torch.add(s.positions, 1, out=output)
            seqout.copy_(s.seq_lens)

        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                body()
        torch.npu.current_stream().wait_stream(stream)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            body()
        for counts, lengths in [([0, 9, 40], [9, 4, 5]), ([8], [2]), ([41, 3], [3, 6])]:
            pos = np.concatenate([np.arange(c, c + q) for c, q in zip(counts, lengths)])
            s.prepare(np.array(counts), np.array(counts) + lengths, pos)
            graph.replay()
            torch.npu.synchronize()
            self.assertEqual(output[: len(pos)].cpu().tolist(), (pos + 1).tolist())
            self.assertEqual(seqout.cpu().tolist(), list(np.array(counts) + lengths) + [0] * (8 - len(counts)))


if __name__ == "__main__":
    unittest.main()
