import asyncio
import unittest
from dataclasses import replace

from streaming_prespec import (AdmissionPolicy, LoadSnapshot,
                               PreSpecCoordinator, RepairExtendProducer,
                               ShadowRequest)


class PreSpecCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 1_000_000_000
        self.policy = AdmissionPolicy(midpoint_samples=16_000,
                                      max_utilization_pct=40,
                                      max_load_age_ms=200,
                                      cancel_margin_ms=100,
                                      safety_guard_ms=100,
                                      max_active_shadows=1)

    def load(self, *, waiting=0, formal_pending=0, utilization=25,
             age_ms=0, queue_p95_ms=57):
        return LoadSnapshot(waiting=waiting, utilization_pct=utilization,
                            observed_ns=self.now - age_ms * 1_000_000,
                            formal_queue_wait_p95_ms=queue_p95_ms,
                            formal_pending=formal_pending)

    def request(self, *, epoch=1, prefix_version=2, window_samples=16_000,
                deadline_ms=900):
        return ShadowRequest(session_id="ws-1", audio_epoch=epoch,
                             base_prefix_version=prefix_version,
                             visible_audio_end_samples=48_000,
                             window_audio_samples=window_samples,
                             audio_pcm_f32le=b"\x00" * 16,
                             stable_prefix_tokens=(10,), rollback_tokens=(20, 21),
                             deadline_ns=self.now + deadline_ms * 1_000_000)

    async def test_formal_never_waits_for_unfinished_shadow_and_cancels_it(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def producer(request, budget):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now)
        decision = coordinator.maybe_start(self.request(), self.load(),
                                           predicted_draft_ms=200)
        self.assertTrue(decision.launched)
        await started.wait()
        self.assertIsNone(coordinator.begin_formal("ws-1", 1, 2, 64_000))
        await asyncio.wait_for(cancelled.wait(), 0.2)
        self.assertIsNone(coordinator.finish_formal("ws-1", 1, (20, 21)))
        await coordinator.close_session("ws-1")

    async def test_producer_verifies_old_then_extends_on_isolated_branch(self):
        calls = []

        class Branch:
            async def verify_and_repair_old(self, rollback, budget):
                calls.append(("verify", rollback, budget))
                return (20, 99)

            async def extend(self, budget):
                calls.append(("extend", budget))
                return (30, 31)

            async def close(self):
                calls.append(("close",))

        class Backend:
            async def open_shadow(self, request):
                calls.append(("open", request.audio_epoch))
                return Branch()

        produced = await RepairExtendProducer(Backend())(self.request(), 4)
        self.assertEqual(produced, (20, 99, 30, 31))
        self.assertEqual(calls, [("open", 1), ("verify", (20, 21), 4),
                                 ("extend", 2), ("close",)])

    async def test_producer_closes_branch_when_verification_fails(self):
        closed = False

        class Branch:
            async def verify_and_repair_old(self, rollback, budget):
                raise RuntimeError("backend failed")

            async def extend(self, budget):
                self.fail("should not extend")

            async def close(self):
                nonlocal closed
                closed = True

        class Backend:
            async def open_shadow(self, request):
                return Branch()

        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            await RepairExtendProducer(Backend())(self.request(), 4)
        self.assertTrue(closed)

    async def test_shadow_candidate_is_measured_but_not_consumed(self):
        async def producer(request, budget):
            return (20, 21, 22)

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now)
        self.assertTrue(coordinator.maybe_start(self.request(), self.load(),
                                                predicted_draft_ms=200).launched)
        await asyncio.sleep(0)
        self.assertIsNone(coordinator.begin_formal("ws-1", 1, 2, 64_000))
        evaluation = coordinator.finish_formal("ws-1", 1, (20, 21, 99))
        self.assertEqual(evaluation.continuous_accepted_tokens, 2)
        self.assertEqual(evaluation.candidate_tokens, (20, 21, 22))
        self.assertTrue(evaluation.ready_before_formal)
        self.assertEqual(evaluation.ready_lead_ms, 0)
        await coordinator.close_session("ws-1")

    async def test_consume_only_ready_version_compatible_candidate(self):
        async def producer(request, budget):
            return (20, 21)

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now,
                                         shadow_only=False)
        coordinator.maybe_start(self.request(), self.load(), predicted_draft_ms=200)
        await asyncio.sleep(0)
        self.assertIsNone(coordinator.begin_formal("ws-1", 1, 3, 64_000))
        coordinator.finish_formal("ws-1", 1, (20, 21))
        request = self.request(epoch=2)
        coordinator.maybe_start(request, self.load(),
                                predicted_draft_ms=200)
        await asyncio.sleep(0)
        candidate = coordinator.begin_formal("ws-1", 2, 2, 64_000)
        self.assertEqual(candidate.token_ids, (20, 21))
        self.assertEqual(candidate.request_id, request.request_id)
        coordinator.finish_formal("ws-1", 2, (20, 21))
        await coordinator.close_session("ws-1")

    async def test_admission_requires_fresh_load_no_waiters_and_deadline_budget(self):
        async def producer(request, budget):
            return (20,)

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now)
        cases = [
            (self.request(window_samples=15_999), self.load(), 200, "before_midpoint"),
            (self.request(), self.load(waiting=1), 200, "formal_queue_nonempty"),
            (self.request(), self.load(formal_pending=1), 200,
             "formal_queue_nonempty"),
            (self.request(), self.load(utilization=50), 200, "utilization_high"),
            (self.request(), self.load(age_ms=201), 200, "stale_load"),
            (self.request(), None, 200, "missing_load"),
            (self.request(deadline_ms=400), self.load(), 200, "insufficient_slack"),
        ]
        for request, load, predicted, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(coordinator.maybe_start(request, load, predicted).reason,
                                 reason)
        self.assertTrue(coordinator.maybe_start(self.request(), self.load(), 200).launched)
        self.assertEqual(coordinator.maybe_start(self.request(epoch=2), self.load(), 200).reason,
                         "shadow_capacity_full")
        await coordinator.close_session("ws-1")

    async def test_close_waits_for_detached_shadow_cleanup(self):
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def producer(request, budget):
            started.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now)
        coordinator.maybe_start(self.request(), self.load(), predicted_draft_ms=200)
        await started.wait()
        coordinator.begin_formal("ws-1", 1, 2, 64_000)
        await coordinator.close_session("ws-1")
        self.assertTrue(cleaned.is_set())

    async def test_skipped_or_final_boundary_discards_candidate(self):
        async def producer(request, budget):
            return (20, 21)

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now,
                                         shadow_only=False)
        coordinator.maybe_start(self.request(), self.load(), predicted_draft_ms=200)
        await asyncio.sleep(0)
        self.assertIsNone(coordinator.begin_formal("ws-1", 1, 2, 64_000,
                                                    allow_candidate=False))
        self.assertIsNone(coordinator.finish_formal("ws-1", 1, ()))
        await coordinator.close_session("ws-1")

    async def test_closed_session_releases_capacity_before_next_session(self):
        async def producer(request, budget):
            return (20,)

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now)
        coordinator.maybe_start(self.request(), self.load(), 200)
        await coordinator.close_session("ws-1")
        second = self.request(epoch=2)
        second = type(second)(**{**second.__dict__, "session_id": "ws-2"})
        self.assertTrue(coordinator.maybe_start(second, self.load(), 200).launched)
        await coordinator.close_session("ws-2")

    async def test_previous_formal_inflight_blocks_midpoint_shadow(self):
        async def producer(request, budget):
            return (20,)

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now)
        coordinator.begin_formal("ws-1", 0, 2, 32_000)
        decision = coordinator.maybe_start(self.request(), self.load(), 200)
        self.assertEqual(decision.reason, "session_formal_inflight")
        coordinator.finish_formal("ws-1", 0, ())
        self.assertTrue(coordinator.maybe_start(self.request(), self.load(), 200).launched)
        await coordinator.close_session("ws-1")

    async def test_streaming_chunk_inflight_blocks_and_preempts_shadow(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def producer(request, budget):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now)
        coordinator.begin_authoritative_chunk("ws-1")
        self.assertEqual(coordinator.maybe_start(self.request(), self.load(), 200).reason,
                         "session_chunk_inflight")
        coordinator.end_authoritative_chunk("ws-1")
        self.assertTrue(coordinator.maybe_start(self.request(), self.load(), 200).launched)
        await started.wait()
        coordinator.begin_authoritative_chunk("ws-1")
        await asyncio.wait_for(cancelled.wait(), 0.2)
        coordinator.end_authoritative_chunk("ws-1")
        self.assertIsNone(coordinator.begin_formal("ws-1", 1, 2, 64_000))
        coordinator.finish_formal("ws-1", 1, ())
        await coordinator.close_session("ws-1")

    async def test_other_sessions_authoritative_work_preempts_shadow(self):
        cancelled = asyncio.Event()

        async def producer(request, budget):
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now)
        coordinator.maybe_start(self.request(), self.load(), 200)
        await asyncio.sleep(0)
        coordinator.begin_authoritative_chunk("ws-2")
        await asyncio.wait_for(cancelled.wait(), 0.2)
        coordinator.end_authoritative_chunk("ws-2")
        self.assertIsNone(coordinator.begin_formal("ws-1", 1, 2, 64_000))
        coordinator.finish_formal("ws-1", 1, ())
        coordinator.begin_formal("ws-2", 1, 2, 64_000)
        other = replace(self.request(epoch=2), session_id="ws-3")
        self.assertEqual(coordinator.maybe_start(other, self.load(), 200).reason,
                         "formal_inflight_on_device")
        coordinator.finish_formal("ws-2", 1, ())
        await coordinator.close_session("ws-1")

    async def test_ready_candidate_is_consumed_at_precommit_chunk_submit(self):
        async def producer(request, budget):
            return (20, 21, 22)

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now,
                                         shadow_only=False)
        coordinator.maybe_start(self.request(), self.load(), 200)
        await asyncio.sleep(0)
        candidate = coordinator.begin_authoritative_chunk(
            "ws-1", audio_epoch=1, prefix_version=2,
            full_audio_end_samples=64_000)
        self.assertEqual(candidate.token_ids, (20, 21, 22))
        evaluation = coordinator.end_authoritative_chunk(
            "ws-1", audio_epoch=1, target_continuation_tokens=(20, 21, 99))
        self.assertEqual(evaluation.continuous_accepted_tokens, 2)
        self.assertIsNone(coordinator.begin_formal("ws-1", 1, 2, 64_000))
        self.assertIsNone(coordinator.finish_formal("ws-1", 1, (20, 21, 99)))
        await coordinator.close_session("ws-1")

    async def test_force_mode_allows_shadow_measurement_under_load(self):
        async def producer(request, budget):
            return (20,)

        coordinator = PreSpecCoordinator(
            producer, replace(self.policy, admission_mode="force"),
            clock=lambda: self.now)
        decision = coordinator.maybe_start(
            self.request(deadline_ms=100),
            None, 200)
        self.assertTrue(decision.launched)
        await coordinator.close_session("ws-1")


if __name__ == "__main__":
    unittest.main()
