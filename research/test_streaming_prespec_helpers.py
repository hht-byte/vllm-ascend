import asyncio
import unittest
from enum import Enum
from types import SimpleNamespace

from streaming_prespec import AdmissionPolicy, PreSpecCoordinator
from streaming_prespec_helpers import PreSpecModelHelpers


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
            yield SimpleNamespace(finished=True, outputs=[
                SimpleNamespace(token_ids=[20, 99, 30], finish_reason="length")])
        finally:
            self.closed.append(kwargs["request_id"])

    async def abort(self, request_id):
        self.aborted.append(request_id)


class Model(PreSpecModelHelpers):
    def __init__(self, engine):
        self.prespec_engine = engine
        self.prespec = PreSpecCoordinator(
            self._run_prespec,
            AdmissionPolicy(admission_mode="force", max_candidate_tokens=3),
            shadow_only=False,
        )

    def _pack_stream_request(self, *, full_audio_pcm_f32le,
                             stable_prefix_token_ids, draft_token_ids,
                             model_context, sampling_params):
        # Test-only transport. These are NOT vLLM external-draft field names.
        return {
            "prompt": {"audio": full_audio_pcm_f32le,
                       "prefix": stable_prefix_token_ids,
                       "context": model_context},
            "test_draft": draft_token_ids,
            "sampling_params": sampling_params,
        }


class ModelHelperTests(unittest.IsolatedAsyncioTestCase):
    def inputs(self):
        return dict(
            ws_id="ws-test", seg_id=1, prefix_version=2,
            full_audio_pcm_f32le=bytearray(b"\x01\x02\x03\x04" * 48000),
            current_audio_end_samples=48000, last_formal_audio_end_samples=32000,
            chunk_size_samples=32000, stable_prefix_token_ids=[10],
            rollback_draft_token_ids=[20, 21],
            model_context={"language": "zh", "context": ["original"]},
            sampling_params=SimpleNamespace(
                max_tokens=128, min_tokens=4, n=1, best_of=None,
                output_kind=OutputKind.DELTA),
            committed=False, audio_queue_empty=True,
        )

    async def test_full_audio_snapshot_and_corrected_continuation(self):
        engine = FakeEngine()
        model = Model(engine)
        inputs = self.inputs()
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
                full_audio_end_samples=64000)
            self.assertEqual(candidate.token_ids, (20, 99, 30))
            call = engine.calls[0]
            self.assertEqual(len(call["prompt"]["audio"]), 48000 * 4)
            self.assertEqual(call["prompt"]["audio"][0], 1)
            self.assertEqual(call["prompt"]["prefix"], [10])
            self.assertEqual(call["prompt"]["context"]["context"], ["original"])
            self.assertEqual(call["test_draft"], [20, 21])
            self.assertEqual(call["sampling_params"].max_tokens, 3)
            self.assertEqual(call["sampling_params"].min_tokens, 0)
            self.assertEqual(call["sampling_params"].output_kind,
                             OutputKind.CUMULATIVE)
            self.assertEqual(inputs["sampling_params"].output_kind, OutputKind.DELTA)
            self.assertTrue(call["request_id"].startswith("prespec-"))
            self.assertFalse(engine.aborted)
            self.assertEqual(engine.closed, [call["request_id"]])
            model.prespec.end_authoritative_chunk(
                "ws-test", audio_epoch=1, target_continuation_tokens=(20, 99, 30))
        finally:
            await model.prespec.close_session("ws-test")

    async def test_skip_without_submitting_or_advancing_formal_cutoff(self):
        model = Model(FakeEngine())
        for overrides, reason in [
            ({"committed": True}, "committed"),
            ({"audio_queue_empty": False}, "audio_backlog"),
            ({"current_audio_end_samples": 40000}, "before_midpoint"),
            ({"current_audio_end_samples": 64000}, "formal_chunk_due"),
            ({"rollback_draft_token_ids": []}, "no_rollback_draft"),
        ]:
            inputs = self.inputs()
            inputs.update(overrides)
            with self.subTest(reason=reason):
                self.assertEqual(model.try_start_prespec(**inputs).reason, reason)
                self.assertEqual(inputs["last_formal_audio_end_samples"], 32000)
        self.assertFalse(model.prespec_engine.calls)

    async def test_formal_submit_cancels_shadow_and_aborts_only_its_request(self):
        engine = FakeEngine("wait")
        model = Model(engine)
        model.try_start_prespec(**self.inputs())
        await engine.started.wait()
        self.assertIsNone(model.prespec.begin_authoritative_chunk(
            "ws-test", audio_epoch=1, prefix_version=2,
            full_audio_end_samples=64000))
        model.prespec.end_authoritative_chunk(
            "ws-test", audio_epoch=1, target_continuation_tokens=())
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
                    full_audio_end_samples=64000))
                model.prespec.end_authoritative_chunk(
                    "ws-test", audio_epoch=1, target_continuation_tokens=())
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
            full_audio_end_samples=64000))
        model.prespec.end_authoritative_chunk(
            "ws-test", audio_epoch=1, target_continuation_tokens=())
        await model.prespec.close_session("ws-test")


if __name__ == "__main__":
    unittest.main()
