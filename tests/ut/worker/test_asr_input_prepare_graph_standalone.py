# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests and optional real NPU preparation replay test."""

import ast
import importlib.util
import textwrap
import unittest
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/asr_input_prepare_graph.py"


def load():
    assert SOURCE.exists(), "input preparation graph is not implemented"
    spec = importlib.util.spec_from_file_location("asr_input_prepare_graph", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Event:
    def __init__(self):
        self.waits = self.records = 0
        self.done = True

    def query(self):
        return self.done

    def synchronize(self):
        self.waits += 1
        self.done = True

    def record(self):
        self.records += 1
        self.done = False


class CPUCapture:
    def __init__(self):
        self.event = Event()
        self.captures = 0

    def capture(self, fn):
        self.captures += 1
        fn()
        return NS(replay=fn)

    def reserved(self):
        return self.captures * 1024


class Tables:
    def __init__(self, capacity=32, requests=4):
        self.block_tables = [
            NS(
                is_mamba_group=False,
                block_table=NS(gpu=torch.arange(requests * 16, dtype=torch.int32).reshape(requests, 16)),
                slot_mapping=NS(gpu=torch.full((capacity + 8,), -99, dtype=torch.int32)),
                block_size=4,
                max_num_batched_tokens=capacity,
            )
        ]

    def compute_slot_mapping(self, n, starts, positions):
        # CPU reference for the existing fixed-grid Triton kernel contract.
        for table in self.block_tables:
            for i in range(n):
                a, b = starts[i : i + 2].tolist()
                for t in range(a, b):
                    pos = int(positions[t])
                    table.slot_mapping.gpu[t] = table.block_table.gpu[i, pos // 4] * 4 + pos % 4
            table.slot_mapping.gpu[len(positions) : table.max_num_batched_tokens].fill_(-1)


def prepare_inputs(self):
    pass


def runner():
    return NS(
        vllm_config=NS(
            additional_config={"asr_mtp": {}, "asr_input_prepare_graph": {"enabled": True, "token_sizes": [8, 16, 32]}}
        ),
        compilation_config=NS(cudagraph_mode=NS(name="FULL")),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        model_config=NS(is_hybrid=False),
        use_async_scheduling=False,
        use_async_spec_decode=False,
        pcp_size=1,
        dcp_size=1,
        _has_gdn=False,
        use_sparse=False,
        use_compress=False,
        is_hybrid=False,
        num_accepted_tokens_event=None,
        max_num_tokens=32,
        device="cpu",
        pin_memory=False,
        positions=torch.zeros(32, dtype=torch.int64),
        num_computed_tokens=torch.zeros(4, dtype=torch.int32),
        seq_lens=torch.zeros(4, dtype=torch.int32),
        query_start_loc=NS(gpu=torch.zeros(6, dtype=torch.int32)),
        num_accepted_tokens=NS(gpu=torch.zeros(4, dtype=torch.int32)),
        discard_request_indices=NS(gpu=torch.zeros(4, dtype=torch.int64)),
        discard_request_mask=NS(gpu=torch.zeros(4, dtype=torch.bool)),
        input_batch=NS(block_table=Tables()),
    )


class PrepareGraphTests(unittest.TestCase):
    def setUp(self):
        self.mod = load()
        self.r = runner()
        self.backend = CPUCapture()

    def graph(self):
        return self.mod.InputPrepareGraph(self.r, (8, 16, 32), backend=self.backend)

    def test_replay_dynamic_reorder_shrink_and_padding(self):
        graph = self.graph()
        self.assertIsNone(graph.select(7))
        self.assertEqual(graph.capture(), 3072)
        self.assertEqual(graph.capture(), 0)
        ptrs = [x.data_ptr() for x in (self.r.positions, self.r.seq_lens, self.r.query_start_loc.gpu)]
        for counts, lengths in [([3, 10, 0], [6, 2, 9]), ([8], [3]), ([7, 1], [1, 6])]:
            n, t = len(counts), sum(lengths)
            qsl = np.array([0] + np.cumsum(lengths).tolist())
            pos = np.concatenate([np.arange(c, c + q) for c, q in zip(counts, lengths)])
            table = self.r.input_batch.block_table.block_tables[0]
            table.block_table.gpu.add_(1)  # Stable allocation, new block mapping.
            graph.replay(
                graph.select(t),
                counts,
                np.array(counts) + lengths,
                pos,
                qsl,
                np.array([True] + [False] * (n - 1)),
                np.array([0]),
            )
            self.assertEqual(self.r.positions[:t].tolist(), pos.tolist())
            self.assertEqual(self.r.seq_lens.tolist(), (np.array(counts) + lengths).tolist() + [0] * (4 - n))
            self.assertEqual(self.r.query_start_loc.gpu.tolist(), qsl.tolist() + [-1] * (5 - n))
            self.assertEqual(self.r.num_accepted_tokens.gpu.tolist(), [1] * 4)
            self.assertEqual(self.r.discard_request_mask.gpu.tolist(), [True] + [False] * 3)
            expected = []
            for i, (c, q) in enumerate(zip(counts, lengths)):
                expected.extend([int(table.block_table.gpu[i, p // 4]) * 4 + p % 4 for p in range(c, c + q)])
            self.assertEqual(table.slot_mapping.gpu.tolist(), expected + [-1] * (32 - t) + [-99] * 8)
            self.assertEqual(
                ptrs, [x.data_ptr() for x in (self.r.positions, self.r.seq_lens, self.r.query_start_loc.gpu)]
            )
        self.assertEqual(self.backend.captures, 3)
        self.assertEqual(self.backend.event.waits, 2)
        self.assertIsNone(graph.select(33))
        self.assertEqual(graph.replays, 3)

    def test_private_qsl_survives_attention_padding_rewrite(self):
        graph = self.graph()
        graph.capture()
        graph.replay(8, [0], [2], [0, 1], [0, 2], [False], [])
        self.r.query_start_loc.gpu.fill_(999)
        graph.replay(8, [5], [6], [5], [0, 1], [False], [])
        self.assertEqual(self.r.query_start_loc.gpu.tolist(), [0, 1, -1, -1, -1, -1])
        self.assertEqual(
            self.r.input_batch.block_table.block_tables[0].slot_mapping.gpu[1:].tolist(), [-1] * 31 + [-99] * 8
        )

    def test_rebound_table_fails_before_stale_replay(self):
        graph = self.graph()
        graph.capture()
        self.r.input_batch.block_table = Tables()
        with self.assertRaisesRegex(RuntimeError, "block table"):
            graph.select(3)

    def test_config_default_off_and_guards(self):
        r = self.r
        r._prepare_inputs = MethodType(prepare_inputs, r)
        self.assertEqual(self.mod.settings(r, prepare_inputs=prepare_inputs), (8, 16, 32))
        for key, value in [("use_async_scheduling", True), ("pcp_size", 2), ("_has_gdn", True)]:
            old = getattr(r, key)
            setattr(r, key, value)
            with self.assertRaises(ValueError):
                self.mod.settings(r, prepare_inputs=prepare_inputs)
            setattr(r, key, old)
        r.vllm_config.additional_config["asr_sync_input_packing"] = True
        with self.assertRaisesRegex(ValueError, "packing"):
            self.mod.settings(r, prepare_inputs=prepare_inputs)
        r.vllm_config.additional_config = {}
        self.assertIsNone(self.mod.settings(r, prepare_inputs=prepare_inputs))

    def test_default_buckets_use_scheduler_capacity_not_padded_position_storage(self):
        r = self.r
        r.max_num_tokens = 16
        r._prepare_inputs = MethodType(prepare_inputs, r)
        del r.vllm_config.additional_config["asr_input_prepare_graph"]["token_sizes"]
        self.assertEqual(self.mod.settings(r, prepare_inputs=prepare_inputs), (16,))

    def test_capture_rejects_bucket_beyond_slot_kernel_capacity(self):
        graph = self.graph()
        self.r.input_batch.block_table.block_tables[0].max_num_batched_tokens = 16
        with self.assertRaisesRegex(ValueError, "block table"):
            graph.capture()

    def test_bad_buckets_and_override_fail(self):
        r = self.r
        r._prepare_inputs = MethodType(prepare_inputs, r)
        for sizes in [[], [0], [True], [33], [3.5], [8, 8]]:
            r.vllm_config.additional_config["asr_input_prepare_graph"]["token_sizes"] = sizes
            with self.assertRaises((TypeError, ValueError)):
                self.mod.settings(r, prepare_inputs=prepare_inputs)
        r.vllm_config.additional_config["asr_input_prepare_graph"]["token_sizes"] = [8]
        with self.assertRaisesRegex(ValueError, "overridden"):
            self.mod.settings(r, prepare_inputs=lambda self: None)


class WiringTests(unittest.TestCase):
    def test_actual_metadata_region_selects_graph_and_keeps_fallback(self):
        mod = load()
        src = SOURCE.with_name("model_runner_v1.py").read_text(encoding="utf-8")
        start = src.index(
            "        valid_sampled_token_count_gpu = self.valid_sampled_token_count_gpu",
            src.index("    def _prepare_inputs("),
        )
        end = src.index("        # In async spec decode mode,", start)
        code = compile(textwrap.dedent(src[start:end]), "<runner metadata>", "exec")

        def buf(size, dtype):
            cpu = torch.zeros(size, dtype=dtype)
            gpu = torch.zeros_like(cpu)
            return NS(cpu=cpu, gpu=gpu, np=cpu.numpy(), copy_to_gpu=lambda n: gpu[:n].copy_(cpu[:n]))

        for use_graph in (False, True):
            r = runner()
            r._input_prepare_graph = mod.InputPrepareGraph(r, (8, 16), backend=CPUCapture())
            r._input_prepare_graph.capture()
            r._sync_input_metadata = None
            r.valid_sampled_token_count_gpu = None
            r.pcp_manager = None
            r.req_indices, r.query_pos = buf(32, torch.int64), buf(32, torch.int64)
            r.num_scheduled_tokens = buf(4, torch.int32)
            r.input_batch.num_computed_tokens_cpu_tensor = torch.tensor([9, 0, 0, 0], dtype=torch.int32)
            r.optimistic_seq_lens_cpu = torch.tensor([12, 4, 0, 0], dtype=torch.int32)
            r.query_start_loc.np = np.array([0, 3, 7, 0, 0], dtype=np.int32)
            r.query_pos.np[:7] = [0, 1, 2, 0, 1, 2, 3]
            ns = dict(
                self=r,
                torch=torch,
                num_reqs=2,
                total_num_scheduled_tokens=7,
                input_prepare_bucket=8 if use_graph else None,
                num_scheduled_tokens=np.array([3, 4], dtype=np.int32),
                positions_np=np.array([9, 10, 11, 0, 1, 2, 3]),
                req_indices=np.array([0, 0, 0, 1, 1, 1, 1]),
                discard_requests_mask=np.array([False, True]),
                discard_request_indices=np.array([1]),
                prev_req_id_to_index={},
                PCPAsyncSpecDecodeRebuildResult=lambda **kw: NS(**kw),
                logger=NS(info=lambda *a: None),
            )
            exec(code, ns)
            self.assertEqual(r.positions[:7].tolist(), [9, 10, 11, 0, 1, 2, 3])
            self.assertEqual(r.seq_lens.tolist(), [12, 4, 0, 0])
            self.assertEqual(r._input_prepare_graph.replays, int(use_graph))

    def test_actual_qsl_and_discard_copy_guards(self):
        src = SOURCE.with_name("model_runner_v1.py").read_text(encoding="utf-8")
        a = src.index("        self.query_start_loc.np[0] = 0", src.index("    def _prepare_inputs("))
        b = src.index("        # Now, query_start_loc is padded.", a)
        qsl_code = compile(textwrap.dedent(src[a:b]), "<runner qsl>", "exec")
        a = src.index("        discard_request_indices = np.nonzero(discard_requests_mask)[0]", b)
        b = src.index("        # Update num_computed_tokens on GPU.", a)
        discard_code = compile(textwrap.dedent(src[a:b]), "<runner discard>", "exec")
        for active in (False, True):
            copies = []
            r = runner()
            for field in (r.query_start_loc, r.discard_request_indices, r.discard_request_mask, r.num_accepted_tokens):
                field.np = np.zeros(field.gpu.numel(), dtype=np.int32)
                field.copy_to_gpu = lambda *a, copies=copies: copies.append(1)
            r.num_accepted_tokens.gpu.fill_(42)
            ns = dict(
                self=r,
                np=np,
                num_reqs=2,
                cu_num_tokens=np.array([3, 7]),
                input_prepare_bucket=8 if active else None,
                copy_snapshot_to_gpu=lambda *a, copies=copies: copies.append(1),
                discard_requests_mask=np.array([False, True]),
            )
            exec(qsl_code, ns)
            exec(discard_code, ns)
            self.assertEqual(len(copies), 0 if active else 3)
            self.assertEqual(r.num_discarded_requests, 1)
            self.assertEqual(r.num_accepted_tokens.gpu[0].item(), 42 if active else 1)

    def test_capture_hook_after_decoder_and_no_runtime_capture(self):
        tree = ast.parse(SOURCE.with_name("model_runner_v1.py").read_text(encoding="utf-8"))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "NPUModelRunner")
        capture = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "capture_model")
        calls = [
            x
            for x in ast.walk(capture)
            if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == "capture"
        ]
        self.assertEqual(len(calls), 1)
        prepare = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "_prepare_inputs")
        self.assertFalse(
            any(
                isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == "capture"
                for x in ast.walk(prepare)
            )
        )
        # Execute actual capture_model with a no-op parent wrapper to prove the
        # existing parent capture memory and the new preparation reserve are added.
        from contextlib import nullcontext

        env = dict(
            _get_gpu_model_runner_module_name=lambda r: "runner",
            _torch_cuda_wrapper=nullcontext,
            _replace_gpu_model_runner_function_wrapper=lambda *a: nullcontext(),
            GPUModelRunner=NS(capture_model=lambda r: 4096),
            logger=NS(info=lambda *a: None),
        )
        exec(compile(ast.fix_missing_locations(ast.Module(body=[capture], type_ignores=[])), "<capture>", "exec"), env)
        r = NS(
            _input_prepare_graph=NS(capture=lambda: 1024, token_sizes=(8,), entries={8: None}, reserved_bytes=1024),
            encoder_cudagraph_manager=None,
        )
        self.assertEqual(env["capture_model"](r), 5120)


def npu_available():
    if importlib.util.find_spec("torch_npu") is None:
        return False
    import torch_npu  # noqa: F401

    return torch.npu.is_available()


@unittest.skipUnless(npu_available(), "requires Ascend NPU, torch_npu and installed vLLM")
class NPUReplayTests(unittest.TestCase):
    def test_existing_slot_kernel_in_actual_graph_with_changed_lengths(self):
        from vllm.v1.worker.block_table import _compute_slot_mapping_kernel

        r = runner()
        r.device, r.pin_memory = "npu", True
        for name in ("positions", "seq_lens", "num_computed_tokens"):
            setattr(r, name, getattr(r, name).to("npu"))
        for name in ("query_start_loc", "num_accepted_tokens", "discard_request_mask", "discard_request_indices"):
            getattr(r, name).gpu = getattr(r, name).gpu.to("npu")
        table = r.input_batch.block_table.block_tables[0]
        table.block_table.gpu = table.block_table.gpu.to(device="npu", dtype=torch.int32)
        table.slot_mapping.gpu = table.slot_mapping.gpu.to("npu")

        def compute(n, qsl, positions):
            _compute_slot_mapping_kernel[(n + 1,)](
                len(positions),
                32,
                qsl,
                positions,
                table.block_table.gpu,
                table.block_table.gpu.stride(0),
                4,
                table.slot_mapping.gpu,
                TOTAL_CP_WORLD_SIZE=1,
                TOTAL_CP_RANK=0,
                CP_KV_CACHE_INTERLEAVE_SIZE=1,
                PAD_ID=-1,
                BLOCK_SIZE=1024,
            )

        r.input_batch.block_table.compute_slot_mapping = compute
        graph = load().InputPrepareGraph(r, (8, 16, 32))
        graph.capture()
        for counts, lengths in [([3, 10, 0], [6, 2, 9]), ([8], [3]), ([7, 1], [1, 6])]:
            n, t = len(counts), sum(lengths)
            qsl = np.array([0] + np.cumsum(lengths).tolist())
            pos = np.concatenate([np.arange(c, c + q) for c, q in zip(counts, lengths)])
            table.block_table.gpu.add_(1)
            graph.replay(graph.select(t), counts, np.array(counts) + lengths, pos, qsl, [False] * n, [])
            actual = table.slot_mapping.gpu.cpu().clone()
            expected = []
            blocks = table.block_table.gpu.cpu()
            for i, (c, q) in enumerate(zip(counts, lengths)):
                expected.extend([int(blocks[i, p // 4]) * 4 + p % 4 for p in range(c, c + q)])
            self.assertEqual(actual.tolist(), expected + [-1] * (32 - t) + [-99] * 8)
            self.assertEqual(r.positions[:t].cpu().tolist(), pos.tolist())
            self.assertEqual(r.seq_lens.cpu().tolist(), (np.array(counts) + lengths).tolist() + [0] * (4 - n))
            self.assertEqual(r.query_start_loc.gpu.cpu().tolist(), qsl.tolist() + [-1] * (5 - n))
            r.query_start_loc.gpu.fill_(999)


if __name__ == "__main__":
    unittest.main()
