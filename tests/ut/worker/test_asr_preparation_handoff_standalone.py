# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred preparation, shared host protection and pure decode graph contracts."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
from test_asr_input_prepare_graph_standalone import CPUCapture, load, npu_available, runner
from test_asr_spec_decode_metadata_standalone import load as load_metadata


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.r = runner()
        self.r.num_spec_tokens = 5
        self.r.supports_mm_inputs = True
        self.r.enable_prompt_embeds = self.r.uses_mrope = False
        self.r.uses_xdrope_dim = 0
        self.r.lora_config = None
        self.r.model_config.is_encoder_decoder = False
        self.r.model_config.quantization = None
        self.r.input_ids = NS(np=np.arange(32, dtype=np.int32), gpu=torch.arange(32, dtype=torch.int32))
        self.r.inputs_embeds = NS(gpu=torch.full((32, 4), -99.0))
        self.embedding = torch.nn.Embedding(64, 4)
        self.r.model = NS(embed_input_ids=self.embedding, requires_raw_input_tokens=False)
        self.r.input_batch.num_computed_tokens_cpu = np.array([10, 20, 0, 0])
        self.r.input_batch.num_prompt_tokens = np.array([10, 18, 0, 0])
        self.r.input_batch.prev_sampled_token_ids = None
        self.r.input_batch.vocab_size = 64
        self.r._prepare_mm_inputs = lambda n: (None, self.r.inputs_embeds.gpu[:n])
        self.r._init_model_kwargs = lambda: {"base": 1}
        self.r._extract_mm_kwargs = lambda _: {"extra": 2}
        self.backend = CPUCapture()
        self.graph = load().InputPrepareGraph(self.r, (8, 16, 32), backend=self.backend)
        self.schedule = NS(scheduled_encoder_inputs={}, total_num_scheduled_tokens=5)

    def stage(self, lengths=(3, 2), counts=(10, 20)):
        self.assertTrue(hasattr(self.graph, "stage"), "deferred preparation missing")
        n, t = len(lengths), sum(lengths)
        positions = np.concatenate([np.arange(c, c + q) for c, q in zip(counts, lengths)])
        self.graph.stage(
            self.graph.select(t),
            np.array(counts),
            np.array(counts) + lengths,
            positions,
            np.array([0] + np.cumsum(lengths).tolist()),
            [False] * n,
            [],
        )
        return n, t

    def test_deferred_outputs_final_padding_and_live_block_rows(self):
        self.graph.capture()
        self.r.query_start_loc.gpu.fill_(99)
        before = self.r.input_batch.block_table.block_tables[0].block_table.gpu.clone()
        self.stage()
        self.assertEqual(self.graph.replays, 0)
        self.assertEqual(self.r.query_start_loc.gpu[0].item(), 99)
        self.graph.finish(8, 3)
        self.assertEqual(self.graph.replays, 1)
        self.assertEqual(self.r.query_start_loc.gpu.tolist(), [0, 3, 5, 8, -1, -1])
        self.assertEqual(self.r.positions[:8].tolist(), [10, 11, 12, 20, 21, 0, 0, 0])
        self.assertEqual(self.graph.outputs["logits"][:2].tolist(), [2, 4])
        table = self.r.input_batch.block_table.block_tables[0]
        torch.testing.assert_close(table.block_table.gpu[:2], before[:2])
        self.assertEqual(table.block_table.gpu[2].sum().item(), 0)
        torch.testing.assert_close(table.block_table.gpu[3], before[3])
        self.assertEqual(table.slot_mapping.gpu[5:32].tolist(), [-1] * 27)
        self.assertTrue(self.graph.covers_padding(5, 8, 2, 3))
        self.assertFalse(self.graph.covers_padding(5, 16, 2, 3))
        self.graph.select(99)
        self.assertFalse(self.graph.covers_padding(5, 8, 2, 3))

    def test_two_payloads_share_one_event_record_and_reuse_wait(self):
        self.graph.capture()
        self.assertTrue(hasattr(self.graph, "before_upload"), "shared upload protection missing")
        packer = load_metadata().SyncSpecDecodeMetadata(self.r, upload_owner=self.graph)
        for i in range(2):
            self.stage()
            meta = packer.prepare(np.array([2, 1], np.int32), np.array([3, 5]), self.r.input_ids.np, NS)
            self.assertEqual(meta.draft_token_ids.tolist(), [1, 2, 4])
            self.assertEqual(self.backend.event.records, i)
            self.graph.finish(8, 3)
            self.assertEqual(self.backend.event.records, i + 1)
        self.assertEqual(self.backend.event.waits, 1)
        self.graph.select(99)
        packer.prepare(np.array([2], np.int32), np.array([3]), self.r.input_ids.np, NS)
        self.graph.record_upload()  # Runner's oversize/fallback exit.
        self.assertEqual(self.backend.event.records, 3)

    def test_embedding_is_in_same_replay_and_bypasses_preprocess(self):
        self.graph.capture()
        self.stage()
        self.graph.finish(8, 3, scheduler_output=self.schedule, allow_embedding=True)
        self.assertEqual(self.graph.replays, 1)
        self.assertEqual(self.graph.embedding_replays, 1)
        result = self.graph.preprocess_result(self.schedule, 8)
        self.assertIsNotNone(result)
        self.assertIsNone(result[0])
        expected = self.embedding(torch.tensor([0, 1, 2, 3, 4, 0, 0, 0]))
        torch.testing.assert_close(result[1], expected)
        self.assertEqual(result[4], {"base": 1, "extra": 2})
        self.assertIsNone(result[5])
        self.assertIsNone(self.graph.preprocess_result(NS(), 8))
        self.assertIsNone(self.graph.preprocess_result(self.schedule, 16))

    def test_embedding_guards_and_oversize_bucket(self):
        self.graph.capture()
        cases = [
            (self.r.input_batch, "num_computed_tokens_cpu", np.array([9, 20, 0, 0])),
            (self.schedule, "scheduled_encoder_inputs", {"audio": [0]}),
            (self.r.input_batch, "prev_sampled_token_ids", torch.tensor([[1]])),
            (self.r, "enable_prompt_embeds", True),
            (self.r, "uses_mrope", True),
            (self.r, "uses_xdrope_dim", 3),
            (self.r, "lora_config", object()),
        ]
        for obj, key, value in cases:
            original = getattr(obj, key)
            setattr(obj, key, value)
            self.stage()
            self.graph.finish(8, 3, scheduler_output=self.schedule, allow_embedding=True)
            self.assertIsNone(self.graph.preprocess_result(self.schedule, 8), key)
            setattr(obj, key, original)
        self.stage()
        self.graph.finish(8, 3, scheduler_output=self.schedule, allow_embedding=False)
        self.assertIsNone(self.graph.preprocess_result(self.schedule, 8))
        # Decoder padding can be larger than the graph's largest configured bucket.
        small = load().InputPrepareGraph(self.r, (8,), backend=CPUCapture())
        small.capture()
        small.stage(small.select(5), [10, 20], [13, 22], [10, 11, 12, 20, 21], [0, 3, 5], [False, False], [])
        small.finish(16, 3, scheduler_output=self.schedule, allow_embedding=True)
        self.assertIsNone(small.preprocess_result(self.schedule, 16))
        self.assertEqual(self.r.query_start_loc.gpu[:4].tolist(), [0, 3, 5, 16])

    def test_switch_off_keeps_plain_graph_and_shared_event_disabled(self):
        opts = self.r.vllm_config.additional_config["asr_input_prepare_graph"]
        opts["fuse_handoff"] = False
        graph = load().InputPrepareGraph(self.r, (8,), backend=self.backend)
        self.assertTrue(hasattr(graph, "fuse_handoff"), "handoff switch missing")
        self.assertFalse(graph.fuse_handoff)
        graph.capture()
        self.assertFalse(graph.embedding_entries)

    def test_embedding_disabled_has_no_embedding_upload_storage(self):
        self.r.vllm_config.additional_config["asr_input_prepare_graph"]["capture_decode_embedding"] = False
        graph = load().InputPrepareGraph(self.r, (8,), backend=self.backend)
        graph.capture()
        self.assertFalse(graph.embedding_entries)
        self.assertEqual(graph.cpu["embed_ids"].size, 0)

    def test_actual_runner_defer_and_padding_guards(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "NPUModelRunner")
        pad = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "_pad_query_start_loc_for_fia")
        self.assertIn("defer_upload", [x.arg for x in pad.args.args + pad.args.kwonlyargs], "qsl still uploaded early")
        pad.returns = None
        for arg in pad.args.args + pad.args.kwonlyargs:
            arg.annotation = None
        env = {"CUDAGraphMode": NS(FULL=1), "torch": torch}
        copies = []
        env["copy_snapshot_to_gpu"] = lambda _: copies.append(1)
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[pad], type_ignores=[])), "<actual FIA padding>", "exec"),
            env,
        )
        for defer in [False, True]:
            r = NS(compilation_config=NS(cudagraph_mode=1), uniform_decode_query_len=6)
            qsl = NS(np=np.array([0, 3, 5, -1], np.int32))
            self.assertEqual(env[pad.name](r, qsl, 8, 2, 2, 1, None, defer_upload=defer), 3)
        self.assertEqual(copies, [1])

    def test_attention_signature_keeps_existing_positional_contract(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "NPUModelRunner")
        fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "_build_attention_metadata")
        self.assertEqual(
            [x.arg for x in fn.args.args[:5]], ["self", "num_tokens", "num_reqs", "max_query_len", "num_tokens_padded"]
        )

    def test_actual_preprocess_hook_preserves_original_audio_path(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "NPUModelRunner")
        fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "_preprocess")
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        calls = []

        class Parent:
            def _preprocess(self, *args):
                calls.append(args)
                return "original audio path"

        wrapper = ast.ClassDef(
            name="Wrapper", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=[fn], decorator_list=[]
        )
        env = {"Parent": Parent}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), "<real preprocess>", "exec"
            ),
            env,
        )
        r = env["Wrapper"]()
        r._input_prepare_graph, r.pcp_size = self.graph, 1
        self.graph.capture()
        self.stage()
        self.graph.finish(8, 3, scheduler_output=self.schedule, allow_embedding=True)
        result = r._preprocess(self.schedule, 8)
        self.assertEqual(len(calls), 0)
        torch.testing.assert_close(result[1], self.r.inputs_embeds.gpu[:8])
        self.stage()
        self.schedule.scheduled_encoder_inputs = {"audio": [0]}
        self.graph.finish(8, 3, scheduler_output=self.schedule, allow_embedding=True)
        self.assertEqual(r._preprocess(self.schedule, 8), "original audio path")
        self.assertEqual(len(calls), 1)

    def test_live_row_reuse_max_requests_and_raw_ids(self):
        self.graph.capture()
        self.stage()
        self.graph.finish(8, 3)
        # The next CPU block-table commit restores rows that become active again.
        table = self.r.input_batch.block_table.block_tables[0]
        table.block_table.gpu[2] = torch.arange(32, 48, dtype=torch.int32)
        self.stage(lengths=(1, 1, 1, 1), counts=(0, 0, 0, 0))
        self.graph.finish(8, 5)  # Extra FIA qsl slot; no physical dummy block row.
        self.assertEqual(self.r.query_start_loc.gpu.tolist(), [0, 1, 2, 3, 4, 8])
        self.assertFalse(self.graph.covers_padding(4, 8, 4, 5))
        self.assertEqual(table.block_table.gpu[2, 0].item(), 32)
        self.r.model.requires_raw_input_tokens = True
        self.r._prepare_mm_inputs = lambda n: (self.r.input_ids.gpu[:n], self.r.inputs_embeds.gpu[:n])
        self.stage()
        self.graph.finish(8, 3, scheduler_output=self.schedule, allow_embedding=True)
        self.assertEqual(self.graph.preprocess_result(self.schedule, 8)[0].data_ptr(), self.r.input_ids.gpu.data_ptr())

    def test_early_logits_consumer_retains_immediate_replay(self):
        import textwrap

        source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
        src = source.read_text(encoding="utf-8")
        start = src.index(
            "        if input_prepare_bucket is not None:", src.index("        valid_sampled_token_count_gpu =")
        )
        end = src.index("        elif self._sync_input_metadata", start)
        code = compile(textwrap.dedent(src[start:end]), "<actual prepare dispatch>", "exec")
        self.graph.capture()
        r = self.r
        r._input_prepare_graph = self.graph
        r.input_batch.num_computed_tokens_cpu_tensor = torch.tensor([10, 20, 0, 0])
        r.optimistic_seq_lens_cpu = torch.tensor([13, 22, 0, 0])
        r.query_start_loc.np = np.array([0, 3, 5, -1, -1, -1])
        ns = dict(
            self=r,
            input_prepare_bucket=8,
            num_reqs=2,
            total_num_scheduled_tokens=5,
            positions_np=np.array([10, 11, 12, 20, 21]),
            discard_requests_mask=[False, False],
            discard_request_indices=[],
            lmhead_tp_enable=lambda: True,
        )
        exec(code, ns)
        self.assertEqual(self.graph.replays, 1, "LM-head padding reads GPU logits before deferred replay")


@unittest.skipUnless(npu_available(), "requires Ascend NPU, torch_npu and installed vLLM")
class NPUHandoffTests(unittest.TestCase):
    def test_shared_upload_and_embedding_in_actual_preparation_graph(self):
        from vllm.v1.worker.block_table import _compute_slot_mapping_kernel

        case = HandoffTests()
        case.setUp()
        r = case.r
        r.device, r.pin_memory = "npu", True
        for name in ("positions", "seq_lens", "num_computed_tokens"):
            setattr(r, name, getattr(r, name).to("npu"))
        for name in (
            "query_start_loc",
            "num_accepted_tokens",
            "discard_request_mask",
            "discard_request_indices",
            "input_ids",
        ):
            getattr(r, name).gpu = getattr(r, name).gpu.to("npu")
        r.inputs_embeds.gpu = r.inputs_embeds.gpu.to(device="npu", dtype=torch.float16)
        r.model.embed_input_ids = case.embedding.to(device="npu", dtype=torch.float16)
        table = r.input_batch.block_table.block_tables[0]
        table.block_table.gpu = table.block_table.gpu.to("npu")
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
        packer = load_metadata().SyncSpecDecodeMetadata(r, upload_owner=graph)
        for shift, padded, embed in [(0, 8, True), (4, 32, False), (9, 16, True), (3, 8, False), (7, 32, True)]:
            r.input_ids.np[:5] = np.arange(5) + shift
            graph.stage(graph.select(5), [10, 20], [13, 22], [10, 11, 12, 20, 21], [0, 3, 5], [False, False], [])
            metadata = packer.prepare(np.array([2, 1], np.int32), np.array([3, 5]), r.input_ids.np, NS)
            graph.finish(padded, 3, scheduler_output=case.schedule, allow_embedding=embed)
            result = graph.preprocess_result(case.schedule, padded)
            if embed:
                expected = r.model.embed_input_ids(
                    torch.tensor([shift + i for i in range(5)] + [0] * (padded - 5), device="npu")
                )
                torch.testing.assert_close(result[1].cpu(), expected.cpu())
            else:
                self.assertIsNone(result)
            self.assertEqual(metadata.draft_token_ids.cpu().tolist(), [shift + 1, shift + 2, shift + 4])
            self.assertEqual(r.query_start_loc.gpu.cpu().tolist(), [0, 3, 5, padded, -1, -1])
            self.assertEqual(table.slot_mapping.gpu[5:32].cpu().tolist(), [-1] * 27)
        self.assertEqual((graph.replays, graph.embedding_replays, graph.upload_records), (5, 3, 5))


if __name__ == "__main__":
    unittest.main()
