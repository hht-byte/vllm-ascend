import asyncio
import unittest
from enum import Enum
from types import SimpleNamespace
import numpy as np

from streaming_prespec import AdmissionPolicy, PreSpecCoordinator
from streaming_prespec_helpers import PreSpecModelHelpers, snapshot_stream_audio


class OutputKind(Enum):
    CUMULATIVE = 0
    DELTA = 1


class FakeEngine:
    def __init__(self, mode="complete"):
        self.mode = mode
        self.calls = []
        self.aborted = []
        self.closed = []
        self.started = asyncio.Event()

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        try:
            # Simulate engine verification: 20 survives, 21 is corrected to 99.
            yield SimpleNamespace(finished=False, outputs=[
                SimpleNamespace(token_ids=[20, 99], finish_reason=None)])
            if self.mode == "wait":
                await asyncio.Future()
            if self.mode == "error":
                raise RuntimeError("engine failure")
            if self.mode == "truncated":
                return
            final_ids = ([19, 20, 21, 22, 30] if self.mode == "full_budget"
                         else [20, 99, 30])
            yield SimpleNamespace(finished=True, outputs=[
                SimpleNamespace(token_ids=final_ids, finish_reason="length")])
        finally:
            self.closed.append(kwargs["request_id"])

    async def abort(self, request_id):
        self.aborted.append(request_id)


class Model(PreSpecModelHelpers):
    def __init__(self, engine, *, kmax=3, candidate_budget=5):
        self.prespec_engine = engine
        self.prespec = PreSpecCoordinator(
            self._run_prespec,
            AdmissionPolicy(admission_mode="force",
                            midpoint_samples=16_000,
                            max_rollback_draft_tokens=kmax,
                            max_candidate_tokens=candidate_budget),
            shadow_only=False,
        )

    def _pack_stream_request(self, *, full_audio_pcm_f32le,
                             prefix_text, stable_prefix_token_ids, draft_token_ids,
                             model_context, sampling_params):
        # Test-only transport. These are NOT vLLM external-draft field names.
        return {
            "prompt": {"audio": full_audio_pcm_f32le,
                       "text": model_context["prompt_raw"] + prefix_text,
                       "prefix_ids": stable_prefix_token_ids,
                       "context": model_context},
            "test_draft": draft_token_ids,
            "sampling_params": sampling_params,
        }


class ModelHelperTests(unittest.IsolatedAsyncioTestCase):
    def inputs(self):
        return dict(
            ws_id="ws-test", seg_id=1, window_id=0, prefix_version=2,
            full_audio_pcm_f32le=bytearray(b"\x01\x02\x03\x04" * 48000),
            current_audio_end_samples=48000, last_formal_audio_end_samples=32000,
            chunk_size_samples=32000, stable_prefix_token_ids=[10],
            prefix_text="recognized ",
            rollback_draft_token_ids=[20, 21],
            model_context={"language": "zh", "context": ["original"],
                           "prompt_raw": "<audio>"},
            sampling_params=SimpleNamespace(
                max_tokens=128, min_tokens=4, n=1, best_of=None,
                output_kind=OutputKind.DELTA),
            committed=False, audio_queue_empty=True,
        )

    async def test_full_audio_snapshot_and_corrected_continuation(self):
        engine = FakeEngine()
        model = Model(engine)
        inputs = self.inputs()
        inputs["window_id"] = 1
        try:
            self.assertTrue(model.try_start_prespec(**inputs).launched)
            # Mutations before the background task runs must not enter its input.
            inputs["full_audio_pcm_f32le"][0] = 0
            inputs["model_context"]["context"][0] = "changed"
            inputs["stable_prefix_token_ids"].append(11)
            inputs["rollback_draft_token_ids"][0] = 66
            inputs["sampling_params"].max_tokens = 256
            await engine.started.wait()
            await asyncio.sleep(0)
            candidate = model.prespec.begin_authoritative_chunk(
                "ws-test", audio_epoch=1, prefix_version=2,
                full_audio_end_samples=64000, window_id=1)
            self.assertEqual(candidate.token_ids, (20, 99, 30))
            self.assertEqual(candidate.window_id, 1)
            self.assertEqual(candidate.formal_draft_token_ids, (99, 30))
            call = engine.calls[0]
            self.assertEqual(len(call["prompt"]["audio"]), 48000 * 4)
            self.assertEqual(call["prompt"]["audio"][0], 1)
            self.assertEqual(call["prompt"]["text"], "<audio>recognized ")
            self.assertEqual(call["prompt"]["prefix_ids"], [10])
            self.assertEqual(call["prompt"]["context"]["context"], ["original"])
            self.assertEqual(call["test_draft"], [20, 21])
            self.assertEqual(call["sampling_params"].max_tokens, 5)
            self.assertEqual(call["sampling_params"].min_tokens, 0)
            self.assertEqual(call["sampling_params"].output_kind,
                             OutputKind.CUMULATIVE)
            self.assertEqual(inputs["sampling_params"].output_kind, OutputKind.DELTA)
            self.assertTrue(call["request_id"].startswith("prespec-"))
            self.assertFalse(engine.aborted)
            self.assertEqual(engine.closed, [call["request_id"]])
            model.prespec.end_authoritative_chunk(
                "ws-test", audio_epoch=1, target_continuation_tokens=(20, 99, 30), window_id=1)
        finally:
            await model.prespec.close_session("ws-test")

    async def test_skip_without_submitting_or_advancing_formal_cutoff(self):
        model = Model(FakeEngine())
        for overrides, reason in [
            ({"committed": True}, "committed"),
            ({"audio_queue_empty": False}, "audio_backlog"),
            ({"current_audio_end_samples": 40000}, "before_midpoint"),
            ({"current_audio_end_samples": 64000}, "formal_chunk_due"),
        ]:
            inputs = self.inputs()
            inputs.update(overrides)
            with self.subTest(reason=reason):
                self.assertEqual(model.try_start_prespec(**inputs).reason, reason)
                self.assertEqual(inputs["last_formal_audio_end_samples"], 32000)
        self.assertFalse(model.prespec_engine.calls)

    async def test_first_chunk_without_old_draft_can_launch(self):
        engine = FakeEngine()
        model = Model(engine)
        inputs = self.inputs()
        inputs.update(
            full_audio_pcm_f32le=bytearray(b"\x01\x02\x03\x04" * 16000),
            current_audio_end_samples=16000,
            last_formal_audio_end_samples=0,
            prefix_version=0,
            stable_prefix_token_ids=[],
            prefix_text="",
            rollback_draft_token_ids=[],
        )
        try:
            self.assertTrue(model.try_start_prespec(**inputs).launched)
            await engine.started.wait()
            await asyncio.sleep(0)
            self.assertEqual(engine.calls[0]["test_draft"], [])
            candidate = model.prespec.begin_authoritative_chunk(
                "ws-test", audio_epoch=1, window_id=0,
                prefix_version=0, full_audio_end_samples=32000)
            self.assertEqual(candidate.formal_draft_token_ids, (99, 30))
            model.prespec.end_authoritative_chunk(
                "ws-test", audio_epoch=1, window_id=0)
        finally:
            await model.prespec.close_session("ws-test")

    async def test_snapshot_joins_consumed_chunks_and_pending_buffer(self):
        committed = np.full(32000, 0.25, dtype=np.float32)
        pending = np.full(16000, -0.5, dtype=np.float32)
        audio, current_end, last_formal_end = snapshot_stream_audio(
            committed, pending)
        self.assertEqual((current_end, last_formal_end), (48000, 32000))
        self.assertTrue(np.array_equal(np.frombuffer(audio, dtype="<f4")[:32000],
                                       committed))
        self.assertTrue(np.array_equal(np.frombuffer(audio, dtype="<f4")[32000:],
                                       pending))
        pending[0] = 1
        self.assertEqual(np.frombuffer(audio, dtype="<f4")[32000], -0.5)

    async def test_old_draft_is_sent_whole_with_separate_output_budget(self):
        engine = FakeEngine()
        model = Model(engine)
        inputs = self.inputs()
        inputs["rollback_draft_token_ids"] = [20, 21, 22]
        self.assertTrue(model.try_start_prespec(**inputs).launched)
        await engine.started.wait()
        await model.prespec.close_session("ws-test")
        self.assertEqual(engine.calls[0]["test_draft"], [20, 21, 22])
        self.assertEqual(engine.calls[0]["sampling_params"].max_tokens, 5)

    async def test_oversized_old_draft_is_skipped_before_engine_submission(self):
        engine = FakeEngine()
        model = Model(engine)
        inputs = self.inputs()
        inputs["rollback_draft_token_ids"] = [20, 21, 22, 23]
        with self.assertLogs("streaming_prespec_helpers", level="INFO") as logs:
            decision = model.try_start_prespec(**inputs)
            again = model.try_start_prespec(**inputs)
        self.assertFalse(decision.launched)
        self.assertEqual(decision.reason, "rollback_draft_exceeds_kmax")
        self.assertEqual(again.reason, decision.reason)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("draft_tokens=4 kmax=3", logs.output[0])
        self.assertFalse(engine.calls)

    async def test_output_budget_covers_first_token_old_draft_and_extension(self):
        engine = FakeEngine()
        model = Model(engine, kmax=3, candidate_budget=4)
        inputs = self.inputs()
        inputs["rollback_draft_token_ids"] = [20, 21, 22]
        with self.assertLogs("streaming_prespec_helpers", level="INFO") as logs:
            decision = model.try_start_prespec(**inputs)
        self.assertFalse(decision.launched)
        self.assertEqual(decision.reason, "candidate_budget_insufficient")
        self.assertIn("candidate_budget=4 required=5", logs.output[0])
        self.assertFalse(engine.calls)

    async def test_full_kmax_old_draft_is_verified_but_oversized_result_not_consumed(self):
        engine = FakeEngine("full_budget")
        model = Model(engine, kmax=3, candidate_budget=5)
        probes = []
        model.prespec.on_probe = probes.append
        inputs = self.inputs()
        inputs["rollback_draft_token_ids"] = [20, 21, 22]
        try:
            self.assertTrue(model.try_start_prespec(**inputs).launched)
            await engine.started.wait()
            await asyncio.sleep(0)
            self.assertEqual(engine.calls[0]["test_draft"], [20, 21, 22])
            self.assertIsNone(model.prespec.begin_authoritative_chunk(
                "ws-test", audio_epoch=1, prefix_version=2,
                full_audio_end_samples=64000, window_id=0))
            self.assertEqual(probes[-1].status, "candidate_exceeds_kmax")
            model.prespec.end_authoritative_chunk("ws-test", audio_epoch=1, window_id=0)
        finally:
            await model.prespec.close_session("ws-test")

    async def test_formal_submit_cancels_shadow_and_aborts_only_its_request(self):
        engine = FakeEngine("wait")
        model = Model(engine)
        model.try_start_prespec(**self.inputs())
        await engine.started.wait()
        self.assertIsNone(model.prespec.begin_authoritative_chunk(
            "ws-test", audio_epoch=1, prefix_version=2,
            full_audio_end_samples=64000, window_id=0))
        model.prespec.end_authoritative_chunk(
            "ws-test", audio_epoch=1, target_continuation_tokens=(), window_id=0)
        await model.prespec.close_session("ws-test")
        self.assertEqual(engine.aborted, [engine.calls[0]["request_id"]])
        self.assertEqual(engine.closed, engine.aborted)

    async def test_error_or_truncated_stream_does_not_publish_partial_candidate(self):
        for mode in ("error", "truncated"):
            with self.subTest(mode=mode):
                engine = FakeEngine(mode)
                model = Model(engine)
                model.try_start_prespec(**self.inputs())
                await engine.started.wait()
                await asyncio.sleep(0)
                self.assertIsNone(model.prespec.begin_authoritative_chunk(
                    "ws-test", audio_epoch=1, prefix_version=2,
                    full_audio_end_samples=64000, window_id=0))
                model.prespec.end_authoritative_chunk(
                    "ws-test", audio_epoch=1, target_continuation_tokens=(), window_id=0)
                await model.prespec.close_session("ws-test")
                self.assertEqual(engine.aborted, [engine.calls[0]["request_id"]])

    async def test_deadline_aborts_shadow_without_waiting_for_formal_submit(self):
        engine = FakeEngine("wait")
        model = Model(engine)
        inputs = self.inputs()
        # 800 samples until the next 2 s chunk: a 50 ms deadline.
        inputs["current_audio_end_samples"] = 63200
        inputs["full_audio_pcm_f32le"] = b"\0" * (63200 * 4)
        self.assertTrue(model.try_start_prespec(**inputs).launched)
        await asyncio.wait_for(engine.started.wait(), 1)
        await asyncio.sleep(0.1)
        self.assertEqual(engine.aborted, [engine.calls[0]["request_id"]])
        self.assertIsNone(model.prespec.begin_authoritative_chunk(
            "ws-test", audio_epoch=1, prefix_version=2,
            full_audio_end_samples=64000, window_id=0))
        model.prespec.end_authoritative_chunk(
            "ws-test", audio_epoch=1, target_continuation_tokens=(), window_id=0)
        await model.prespec.close_session("ws-test")


if __name__ == "__main__":
    unittest.main()
