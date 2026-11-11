"""Real CPU tensor coverage of the MTP worker's graph-to-engine handoff."""

import importlib.util
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from test_asr_mtp_token_attention import FakeAPI, FakeRuntime, npu_available
from test_asr_mtp_token_graph import PackedFixture


class CPUCapture:
    def capture(self, program):
        program.run()
        return program.run


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Torch")
class RuntimeHandoffTests(PackedFixture, unittest.TestCase):
    def make_worker(self, *, max_batch=3, context=17, profile=False, acceptance=False):
        import numpy as np
        from asr_mtp_token_graph import TokenMTPGraphs
        from asr_mtp_worker import MTPWorker

        for state in self.b:
            self.new.release(state)
        options = SimpleNamespace(
            max_batch=max_batch,
            queries=(),
            tokens=(8, 32),
            context=context,
            profile_stages=profile,
        )
        graphs = TokenMTPGraphs(options, self.fixed, self.new, self.embed, self.logits)
        graphs.prepare(CPUCapture())
        rows = [row + [4] for row in self.ids]
        params = SimpleNamespace(
            extra_args={"ascend_mtp": True},
            max_tokens=20,
            temperature=0.0,
            n=1,
            min_tokens=0,
        )
        batch = SimpleNamespace(
            req_ids=["a", "b", "c"],
            max_model_len=32,
            num_tokens_no_spec=np.array([len(row) for row in rows]),
            num_computed_tokens_cpu=np.zeros(3, dtype=np.int32),
            token_ids_cpu=np.zeros((3, 32), dtype=np.int32),
            spec_decode_unsupported_reqs=set(),
        )
        for i, row in enumerate(rows):
            batch.token_ids_cpu[i, : len(row)] = row
        offsets = np.cumsum([0] + [len(row) for row in self.ids])
        runner = SimpleNamespace(
            input_batch=batch,
            query_start_loc=SimpleNamespace(np=offsets),
            speculative_config=SimpleNamespace(num_speculative_tokens=3),
            requests={key: SimpleNamespace(sampling_params=params, output_token_ids=[4]) for key in batch.req_ids},
        )
        worker = MTPWorker(
            runner,
            self.fixed,
            self.new,
            self.embed,
            self.logits,
            {63},
            graphs=graphs,
            **({"profile_stages": True} if profile else {}),
            **({"acceptance_stats": True} if acceptance else {}),
        )
        hidden = self.t.cat([state.target[: state.length] for state in self.a])
        scheduler = SimpleNamespace(num_scheduled_tokens={key: len(row) for key, row in zip(batch.req_ids, self.ids)})
        return worker, graphs, scheduler, hidden, rows

    def test_acceptance_observes_previous_mtp_before_eos_and_preserves_one_d2h(self):
        worker, _, scheduler, hidden, rows = self.make_worker(acceptance=True)
        logs = []
        worker.acceptance.emit = logs.append
        converted = []
        original_tolist = self.t.Tensor.tolist

        def tolist(tensor):
            converted.append(tuple(tensor.shape))
            return original_tolist(tensor)

        with patch.object(self.t.Tensor, "tolist", tolist):
            outputs = worker.propose([[4]] * 3, scheduler, hidden)
        self.assertEqual(converted, [(3, 3)])
        self.assertEqual(outputs, self.expected(rows))
        batch = worker.runner.input_batch
        scheduler.scheduled_spec_decode_tokens = dict(zip(batch.req_ids, outputs))
        scheduler.num_scheduled_tokens = {req: len(row) + 1 for req, row in zip(batch.req_ids, outputs)}
        batch.num_computed_tokens_cpu[:] = [len(row) - 1 for row in rows]
        result = worker.propose([[row[0], 63] for row in outputs], scheduler, hidden)
        self.assertEqual(result, [[], [], []])
        self.assertEqual(len(logs), 3)
        self.assertTrue(all(row["total"]["accepted"] == 1 for row in logs))
        self.assertTrue(all(row["total"]["drafted"] == 3 for row in logs))
        worker.finish(batch.req_ids)
        self.assertEqual(len(logs), 3)

    def test_disabled_acceptance_has_no_tracker(self):
        worker, _, _, _, _ = self.make_worker()
        self.assertIsNone(worker.acceptance)

    def test_finish_cancels_pending_acceptance_proposals(self):
        worker, _, scheduler, hidden, _ = self.make_worker(acceptance=True)
        logs = []
        worker.acceptance.emit = logs.append
        worker.propose([[4]] * 3, scheduler, hidden)
        worker.finish(worker.runner.input_batch.req_ids)
        self.assertEqual(len(logs), 3)
        self.assertTrue(all(row["total"]["rounds"] == 0 for row in logs))
        self.assertTrue(all(row["unverified_tokens"] == 3 for row in logs))

    def expected(self, rows):
        with self.t.inference_mode():
            result = self.eager.propose(self.a, rows, self.embed, self.logits, [3] * 3).tolist()
        return [row[: row.index(63) + 1] if 63 in row else row for row in result]

    def test_full_group_converts_graph_output_directly_without_aggregate_scatter(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        torch = self.t
        worker, graphs, scheduler, hidden, rows = self.make_worker()
        aggregate_ops, converted = [], []
        original_tolist = torch.Tensor.tolist

        class HandoffOps(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func == torch.ops.aten.zeros.default and tuple(args[0]) == (3, 3):
                    aggregate_ops.append("zeros")
                if func == torch.ops.aten.index_put_.default and tuple(args[0].shape) == (3, 3):
                    aggregate_ops.append("index_put")
                return func(*args, **(kwargs or {}))

        def tolist(value):
            converted.append(value.untyped_storage().data_ptr())
            return original_tolist(value)

        with HandoffOps(), patch.object(torch.Tensor, "tolist", tolist):
            actual = worker.propose([[4]] * 3, scheduler, hidden)
        self.assertEqual(actual, self.expected(rows))
        self.assertEqual(aggregate_ops, [])
        self.assertEqual(converted, [graphs.entries[32][0].output.untyped_storage().data_ptr()])

    def test_split_replays_copy_persistent_outputs_before_the_next_replay(self):
        worker, graphs, scheduler, hidden, rows = self.make_worker(max_batch=1)
        converted = []
        original_tolist = self.t.Tensor.tolist

        def tolist(value):
            converted.append(tuple(value.shape))
            return original_tolist(value)

        with patch.object(self.t.Tensor, "tolist", tolist):
            actual = worker.propose([[4]] * 3, scheduler, hidden)
        self.assertEqual(actual, self.expected(rows))
        self.assertEqual(graphs.stats["replays"], 3)
        self.assertEqual(converted, [(3, 3)])

    def test_token_graph_and_eager_groups_preserve_original_request_order(self):
        worker, graphs, scheduler, hidden, rows = self.make_worker(max_batch=1, context=6)
        actual = worker.propose([[4]] * 3, scheduler, hidden)
        self.assertEqual(actual, self.expected(rows))
        self.assertEqual(worker.stats["eager_head_batches"], 1)
        self.assertEqual(graphs.stats["replays"], 2)

    def test_disabled_timing_does_not_read_a_clock_on_the_runtime_path(self):
        worker, _, scheduler, hidden, rows = self.make_worker()
        with (
            patch(
                "asr_mtp_worker.perf_counter",
                side_effect=AssertionError("clock"),
                create=True,
            ),
            patch("asr_mtp_token_graph.perf_counter", side_effect=AssertionError("clock")),
        ):
            actual = worker.propose([[4]] * 3, scheduler, hidden)
        self.assertEqual(actual, self.expected(rows))

    def test_enabled_host_timing_reports_cpu_phases_and_physical_capacity(self):
        worker, _, scheduler, hidden, rows = self.make_worker(profile=True)
        with (
            self.assertLogs(level="INFO") as caught,
            patch("asr_mtp_worker.perf_counter", side_effect=(i / 1000 for i in range(30))),
            patch(
                "asr_mtp_token_graph.perf_counter",
                side_effect=(i / 1000 for i in range(30)),
            ),
        ):
            actual = worker.propose([[4]] * 3, scheduler, hidden)
        self.assertEqual(actual, self.expected(rows))
        records = [record.getMessage() for record in caught.records]
        worker_log = next(message for message in records if message.startswith("ASR_MTP_HOST_STAGE"))
        graph_log = next(message for message in records if message.startswith("ASR_MTP_GRAPH_HOST_STAGE"))
        for field in (
            "requests=3",
            "prepare_host_ms=",
            "proposal_host_ms=",
            "to_cpu_wait_ms=",
            "postprocess_host_ms=",
            "single_group=True",
        ):
            self.assertIn(field, worker_log)
        for field in (
            "requests=3",
            "tokens=15",
            "bucket=32",
            "query_capacity=33",
            "kv_capacity=84",
            "stage_host_ms=",
            "commit_host_ms=",
        ):
            self.assertIn(field, graph_log)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Torch")
class AttentionBindingReuseTests(unittest.TestCase):
    def setUp(self):
        import torch
        from asr_mtp_token_attention import PackedAttention

        self.t = torch
        self.runtime = FakeRuntime()
        self.api = FakeAPI(torch, self.runtime)
        self.attention = PackedAttention(
            2,
            1,
            128,
            device="cpu",
            dtype=torch.float32,
            heads=2,
            npu_api=self.api,
            npu_runtime=self.runtime,
        )
        self.q = torch.zeros(4, 2, 128)
        self.k = torch.zeros(8, 1, 128)
        self.v = torch.zeros_like(self.k)
        for head in range(2):
            self.attention.stage(head, [1, 3, 4], [2, 6, 8])
            self.attention(head, self.q, self.k, self.v, 0.25)
        self.attention.begin_capture()
        for head in range(2):
            self.attention(head, self.q, self.k, self.v, 0.25)
        self.attention.end_capture()
        self.api.calls.clear()
        self.runtime.events.clear()

    def test_repeated_boundaries_skip_task_updates_but_signal_every_head(self):
        for _ in range(2):
            self.q.add_(1)  # Content changes never require pointer rebinding.
            self.attention.update()
        self.assertEqual(self.api.calls, [])
        self.assertNotIn("update_begin", self.runtime.events)
        self.assertEqual(self.runtime.events.count("event_record"), 4)
        self.assertEqual(self.attention.task_update_reuses, 4)
        self.assertEqual(self.attention.task_updates, 0)

    def test_changed_boundaries_update_once_then_reuse_successful_binding(self):
        self.attention.stage(1, [2, 3, 4], [4, 6, 8])
        self.attention.update()
        self.assertEqual(len(self.api.calls), 1)
        self.assertEqual(self.api.calls[0]["actual_seq_lengths"], (2, 3, 4))
        self.attention.update()
        self.assertEqual(len(self.api.calls), 1)
        self.assertEqual(self.runtime.events.count("event_record"), 4)

    def test_replaced_same_shape_tensor_requires_a_new_binding(self):
        self.attention.tasks[0].query = self.q.clone()
        self.attention.update()
        self.assertEqual(len(self.api.calls), 1)
        self.assertEqual(self.runtime.events.count("event_record"), 2)

    def test_failed_preflight_never_updates_or_signals_an_earlier_valid_head(self):
        self.attention.stage(0, [2, 3, 4], [4, 6, 8])
        self.attention.stage(1, [5], [8])
        with self.assertRaises(ValueError):
            self.attention.update()
        self.assertEqual(self.api.calls, [])
        self.assertEqual(self.runtime.events, [])


class CaptureRuntime(FakeRuntime):
    """Only NPU lifecycle is simulated; the captured program uses real tensors."""

    def __init__(self):
        super().__init__()
        del self.stream
        self.active_stream = SimpleNamespace(wait_stream=lambda stream: None)
        self.synchronizations, self.replays = 0, 0

    def current_stream(self):
        return self.active_stream

    def Stream(self):
        return self.active_stream

    def stream(self, stream):
        return nullcontext()

    def graph_pool_handle(self):
        return object()

    def NPUGraph(self):
        def replay():
            self.replays += 1

        return SimpleNamespace(replay=replay)

    def graph(self, graph, pool=None):
        return nullcontext()

    def synchronize(self):
        self.synchronizations += 1


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Torch")
class CaptureHostTimingTests(unittest.TestCase):
    def capture(self, profiled):
        import torch
        from asr_mtp_token_attention import PackedAttention, PackedNPUCapture

        runtime = CaptureRuntime()
        api = FakeAPI(torch, runtime)
        attention = PackedAttention(
            2,
            1,
            128,
            device="cpu",
            dtype=torch.float32,
            heads=1,
            npu_api=api,
            npu_runtime=runtime,
        )
        attention.stage(0, [2, 4], [4, 8])
        q, k = torch.zeros(4, 2, 128), torch.zeros(8, 1, 128)
        program = SimpleNamespace(attention=attention, profile_stages=profiled)
        program.run = lambda: attention(0, q, k, k, 0.25)
        with patch.object(torch, "npu", runtime, create=True):
            replay = PackedNPUCapture().capture(program)
        return replay, attention, runtime

    def test_default_replay_does_not_read_clock_or_synchronize(self):
        replay, attention, runtime = self.capture(False)
        synchronized = runtime.synchronizations
        with patch("asr_mtp_token_attention.perf_counter", side_effect=AssertionError("clock")):
            replay()
            replay()
        self.assertEqual(runtime.synchronizations, synchronized)
        self.assertEqual(runtime.replays, 2)
        self.assertEqual(attention.task_update_reuses, 2)

    def test_profiled_replay_separates_host_rebinding_from_enqueue(self):
        replay, _, runtime = self.capture(True)
        synchronized = runtime.synchronizations
        with patch("asr_mtp_token_attention.perf_counter", side_effect=[1.0, 1.002, 1.005]):
            replay()
        self.assertAlmostEqual(replay.last_timings["task_update_host_ms"], 2.0)
        self.assertAlmostEqual(replay.last_timings["replay_enqueue_host_ms"], 3.0)
        self.assertEqual(replay.last_timings["task_updates"], 0)
        self.assertEqual(replay.last_timings["task_update_reuses"], 1)
        self.assertEqual(runtime.synchronizations, synchronized)


@unittest.skipUnless(npu_available(), "requires torch-npu and an available NPU")
class ActualAttentionBindingReuseTests(unittest.TestCase):
    def test_replay_uses_changed_tensor_content_when_task_binding_is_reused(self):
        import torch
        from asr_mtp_token_attention import PackedAttention, PackedNPUCapture

        class Program:
            def __init__(self):
                self.attention = PackedAttention(2, 1, 128, device="npu", dtype=torch.float16, heads=1)
                self.q = torch.randn(4, 2, 128, device="npu", dtype=torch.float16)
                self.k = torch.randn(8, 1, 128, device="npu", dtype=torch.float16)
                self.v = torch.randn_like(self.k)

            def run(self):
                self.output = self.attention(0, self.q, self.k, self.v, 128**-0.5)
                return self.output

        with torch.inference_mode():
            program = Program()
            program.attention.stage(0, [2, 4], [4, 8])
            replay = PackedNPUCapture().capture(program)
            reference = PackedAttention(2, 1, 128, device="cpu", dtype=torch.float32, heads=1)
            reference.stage(0, [2, 4], [4, 8])
            for _ in range(3):
                program.q.copy_(torch.randn_like(program.q))
                program.k.copy_(torch.randn_like(program.k))
                program.v.copy_(torch.randn_like(program.v))
                replay()
                expected = reference(
                    0,
                    program.q.cpu().float(),
                    program.k.cpu().float(),
                    program.v.cpu().float(),
                    128**-0.5,
                )
                torch.testing.assert_close(program.output.cpu().float(), expected, atol=5e-3, rtol=5e-3)
            self.assertEqual(program.attention.task_updates, 0)
            self.assertEqual(program.attention.task_update_reuses, 3)


if __name__ == "__main__":
    unittest.main()
