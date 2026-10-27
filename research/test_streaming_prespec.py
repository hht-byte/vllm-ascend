import asyncio
import unittest
from dataclasses import replace

from streaming_prespec import (AdmissionPolicy, LoadSnapshot,
                               PreSpecCoordinator, RepairExtendProducer,
                               ShadowRequest, VerifiedRollback)


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

    def request(self, *, epoch=1, window_id=0, prefix_version=2,
                window_samples=16_000,
                deadline_ms=900):
        return ShadowRequest(session_id="ws-1", audio_epoch=epoch,
                             window_id=window_id,
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
        self.assertIsNone(coordinator.begin_formal("ws-1", 1, 2, 64_000, window_id=0))
        await asyncio.wait_for(cancelled.wait(), 0.2)
        self.assertIsNone(coordinator.finish_formal("ws-1", 1, (20, 21), window_id=0))
        await coordinator.close_session("ws-1")

    async def test_producer_verifies_old_then_extends_on_isolated_branch(self):
        calls = []
        test_case = self
        repaired_state = object()
        verified = VerifiedRollback(19, (20, 99), repaired_state)

        class Branch:
            async def verify_and_repair_old(self, rollback, budget):
                calls.append(("verify", rollback, budget))
                return verified

            async def extend_from_verified(self, result, budget):
                test_case.assertIs(result, verified)
                test_case.assertIs(result.continuation_state, repaired_state)
                calls.append(("extend", result.token_ids, budget))
                return (30, 31)

            async def close(self):
                calls.append(("close",))

        class Backend:
            async def open_shadow(self, request):
                calls.append(("open", request.audio_epoch))
                return Branch()

        produced = await RepairExtendProducer(Backend())(self.request(), 5)
        self.assertEqual(produced, (19, 20, 99, 30, 31))
        self.assertEqual(calls, [("open", 1), ("verify", (20, 21), 5),
                                 ("extend", (20, 99), 2), ("close",)])

    async def test_producer_does_not_truncate_old_draft_before_verification(self):
        calls = []

        class Branch:
            async def verify_and_repair_old(self, rollback, budget):
                calls.append(("verify", rollback, budget))
                return VerifiedRollback(19, (20, 21, 22), object())

            async def extend_from_verified(self, result, budget):
                calls.append(("extend", budget))
                return (30,)

            async def close(self):
                pass

        class Backend:
            async def open_shadow(self, request):
                return Branch()

        request = replace(self.request(), rollback_tokens=(20, 21, 22))
        result = await RepairExtendProducer(Backend())(request, 5)
        self.assertEqual(result, (19, 20, 21, 22, 30))
        self.assertEqual(calls, [("verify", (20, 21, 22), 5), ("extend", 1)])

    async def test_coordinator_rejects_oversized_old_draft(self):
        async def producer(request, budget):
            self.fail("oversized draft must not reach producer")

        policy = replace(self.policy, max_rollback_draft_tokens=3,
                         max_candidate_tokens=3)
        coordinator = PreSpecCoordinator(producer, policy, clock=lambda: self.now)
        request = replace(self.request(), rollback_tokens=(20, 21, 22, 23))
        decision = coordinator.maybe_start(request, self.load(), 200)
        self.assertEqual(decision.reason, "rollback_draft_exceeds_kmax")
        self.assertFalse(decision.launched)

    async def test_candidate_budget_over_kmax_rejected_at_configuration(self):
        async def producer(request, budget):
            return (19, 20, 21, 22)

        policy = replace(self.policy, max_rollback_draft_tokens=2,
                         max_candidate_tokens=4)
        with self.assertRaisesRegex(ValueError, "candidate budget must fit"):
            PreSpecCoordinator(producer, policy, clock=lambda: self.now,
                               shadow_only=False)

    async def test_candidate_draft_includes_shadow_first_token(self):
        async def producer(request, budget):
            return (19, 20, 21)

        policy = replace(self.policy, max_rollback_draft_tokens=4,
                         max_candidate_tokens=4)
        coordinator = PreSpecCoordinator(producer, policy, clock=lambda: self.now,
                                         shadow_only=False)
        coordinator.maybe_start(self.request(), self.load(), 200)
        await asyncio.sleep(0)
        candidate = coordinator.begin_authoritative_chunk(
            "ws-1", audio_epoch=1, prefix_version=2,
            full_audio_end_samples=64_000, window_id=0)
        self.assertEqual(candidate.token_ids, (19, 20, 21))
        self.assertEqual(candidate.formal_draft_token_ids, (19, 20, 21))
        coordinator.end_authoritative_chunk("ws-1", audio_epoch=1, window_id=0)
        await coordinator.close_session("ws-1")

    async def test_producer_closes_branch_when_verification_fails(self):
        closed = False

        class Branch:
            async def verify_and_repair_old(self, rollback, budget):
                raise RuntimeError("backend failed")

            async def extend_from_verified(self, result, budget):
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

    async def test_producer_rejects_raw_tokens_without_verified_state(self):
        closed = False

        class Branch:
            async def verify_and_repair_old(self, rollback, budget):
                return (20, 99)  # Tokens alone do not prove corrected KV state.

            async def extend_from_verified(self, verified, budget):
                raise AssertionError("must not extend an unverified state")

            async def close(self):
                nonlocal closed
                closed = True

        class Backend:
            async def open_shadow(self, request):
                return Branch()

        with self.assertRaisesRegex(TypeError, "VerifiedRollback"):
            await RepairExtendProducer(Backend())(self.request(), 4)
        self.assertTrue(closed)

    async def test_shadow_candidate_is_measured_but_not_consumed(self):
        async def producer(request, budget):
            return (20, 21, 22)

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now)
        self.assertTrue(coordinator.maybe_start(self.request(), self.load(),
                                                predicted_draft_ms=200).launched)
        await asyncio.sleep(0)
        self.assertIsNone(coordinator.begin_formal("ws-1", 1, 2, 64_000, window_id=0))
        evaluation = coordinator.finish_formal("ws-1", 1, (20, 21, 99), window_id=0)
        self.assertEqual(evaluation.continuous_accepted_tokens, 2)
        self.assertEqual(evaluation.candidate_tokens, (20, 21, 22))
        self.assertTrue(evaluation.ready_before_formal)
        self.assertEqual(evaluation.ready_lead_ms, 0)
        await coordinator.close_session("ws-1")

    async def test_shadow_match_reports_raw_and_draft_aligned_prefixes(self):
        async def producer(request, budget):
            return (19, 20, 21, 22)

        policy = replace(self.policy, max_candidate_tokens=4)
        coordinator = PreSpecCoordinator(producer, policy, clock=lambda: self.now)
        self.assertTrue(coordinator.maybe_start(self.request(), self.load(),
                                                predicted_draft_ms=200).launched)
        await asyncio.sleep(0)
        self.assertIsNone(coordinator.begin_authoritative_chunk(
            "ws-1", audio_epoch=1, prefix_version=2,
            full_audio_end_samples=64_000, window_id=0))
        evaluation = coordinator.end_authoritative_chunk(
            "ws-1", audio_epoch=1,
            target_continuation_tokens=(99, 20, 21, 77), window_id=0)
        self.assertEqual(evaluation.target_token_count, 4)
        self.assertEqual(evaluation.raw_matched_prefix_tokens, 0)
        self.assertEqual(evaluation.draft_matched_prefix_tokens, 2)
        await coordinator.close_session("ws-1")

    async def test_match_log_shows_full_target_sequence(self):
        async def producer(request, budget):
            return (19, 20, 21, 22)

        coordinator = PreSpecCoordinator(
            producer, replace(self.policy, max_candidate_tokens=4),
            clock=lambda: self.now)
        self.assertTrue(coordinator.maybe_start(
            self.request(), self.load(), predicted_draft_ms=200).launched)
        await asyncio.sleep(0)
        coordinator.begin_authoritative_chunk(
            "ws-1", audio_epoch=1, window_id=0, prefix_version=2,
            full_audio_end_samples=64_000)
        with self.assertLogs("streaming_prespec", level="INFO") as captured:
            coordinator.end_authoritative_chunk(
                "ws-1", audio_epoch=1, window_id=0,
                target_continuation_tokens=(19, 20, 99, 77, 55))
        self.assertEqual(len(captured.output), 1)
        self.assertIn("candidate_token_ids=(19, 20, 21, 22)", captured.output[0])
        self.assertIn("target_token_ids=(19, 20, 99, 77, 55)", captured.output[0])
        self.assertIn("target_len=5", captured.output[0])
        self.assertIn("raw_matched_prefix_tokens=2", captured.output[0])
        await coordinator.close_session("ws-1")

    async def test_same_segment_starts_a_new_candidate_for_each_chunk_window(self):
        launched_windows = []
        probes = []

        async def producer(request, budget):
            launched_windows.append(request.window_id)
            return (19, 20 + request.window_id)

        coordinator = PreSpecCoordinator(
            producer, self.policy, clock=lambda: self.now,
            shadow_only=False, on_probe=probes.append)
        for window_id in (1, 2):
            request = self.request(window_id=window_id,
                                   prefix_version=window_id + 1)
            self.assertTrue(coordinator.maybe_start(
                request, self.load(), predicted_draft_ms=200).launched)
            await asyncio.sleep(0)
            candidate = coordinator.begin_authoritative_chunk(
                "ws-1", audio_epoch=1, window_id=window_id,
                prefix_version=window_id + 1,
                full_audio_end_samples=(window_id + 1) * 32_000)
            self.assertEqual(candidate.window_id, window_id)
            coordinator.end_authoritative_chunk(
                "ws-1", audio_epoch=1, window_id=window_id)
            self.assertEqual(coordinator.maybe_start(
                request, self.load(), predicted_draft_ms=200).reason,
                "epoch_already_used")

        self.assertEqual(launched_windows, [1, 2])
        self.assertEqual([probe.window_id for probe in probes], [1, 2])
        self.assertFalse(coordinator._tasks)
        self.assertFalse(coordinator._selected)
        await coordinator.close_session("ws-1")

    async def test_commit_cancels_only_the_current_unfinished_window(self):
        second_started = asyncio.Event()
        second_cancelled = asyncio.Event()

        async def producer(request, budget):
            if request.window_id == 1:
                return (19, 20)
            second_started.set()
            try:
                await asyncio.Future()
            finally:
                second_cancelled.set()

        coordinator = PreSpecCoordinator(producer, self.policy,
                                         clock=lambda: self.now)
        coordinator.maybe_start(self.request(window_id=1), self.load(), 200)
        await asyncio.sleep(0)
        coordinator.begin_authoritative_chunk(
            "ws-1", audio_epoch=1, window_id=1, prefix_version=2,
            full_audio_end_samples=64_000)
        coordinator.end_authoritative_chunk(
            "ws-1", audio_epoch=1, window_id=1)

        coordinator.maybe_start(self.request(window_id=2, prefix_version=3),
                                self.load(), 200)
        await second_started.wait()
        self.assertIsNone(coordinator.begin_formal(
            "ws-1", 1, 3, 0, window_id=2, allow_candidate=False))
        await asyncio.wait_for(second_cancelled.wait(), 0.2)
        self.assertIsNone(coordinator.finish_formal(
            "ws-1", 1, window_id=2))
        self.assertEqual(coordinator.maybe_start(
            self.request(window_id=3, prefix_version=4), self.load(), 200
        ).reason, "epoch_already_used")
        await coordinator.close_session("ws-1")

    async def test_commit_defers_ready_candidate_until_finish_request(self):
        probes = []

        async def producer(request, budget):
            return (19, 20, 21)

        coordinator = PreSpecCoordinator(
            producer, self.policy, clock=lambda: self.now,
            shadow_only=False, on_probe=probes.append)
        self.assertTrue(coordinator.maybe_start(
            self.request(window_id=1), self.load(), 200).launched)
        await asyncio.sleep(0)
        self.assertIsNone(coordinator.begin_formal(
            "ws-1", 1, 2, 0, window_id=1, defer_candidate=True))
        candidate = coordinator.claim_formal_candidate(
            "ws-1", 1, window_id=1, prefix_version=2,
            full_audio_end_samples=62_000)
        self.assertEqual(candidate.formal_draft_token_ids, (19, 20, 21))
        self.assertEqual(probes[-1].status, "ready_consumed")
        coordinator.finish_formal("ws-1", 1, window_id=1)
        await coordinator.close_session("ws-1")

    async def test_shadow_commit_observes_ready_candidate_without_submitting_it(self):
        probes = []

        async def producer(request, budget):
            return (19, 20, 21)

        coordinator = PreSpecCoordinator(
            producer, self.policy, clock=lambda: self.now,
            shadow_only=True, on_probe=probes.append)
        coordinator.maybe_start(self.request(window_id=1), self.load(), 200)
        await asyncio.sleep(0)
        coordinator.begin_formal(
            "ws-1", 1, 2, 0, window_id=1, defer_candidate=True)
        self.assertIsNone(coordinator.claim_formal_candidate(
            "ws-1", 1, window_id=1, prefix_version=2,
            full_audio_end_samples=62_000))
        self.assertEqual(probes[-1].status, "ready_shadow_only")
        self.assertEqual(probes[-1].boundary, "formal")
        with self.assertLogs("streaming_prespec", level="INFO") as captured:
            evaluation = coordinator.finish_formal(
                "ws-1", 1, (19, 20, 99), window_id=1)
        self.assertEqual(evaluation.raw_matched_prefix_tokens, 2)
        self.assertEqual(evaluation.draft_matched_prefix_tokens, 1)
        self.assertTrue(any(
            "PRESPEC_MATCH ws_id=ws-1 seg=1 window=1 boundary=formal "
            "candidate_len=3 target_len=3 raw_matched_prefix_tokens=2 "
            "draft_matched_prefix_tokens=1" in entry
            for entry in captured.output
        ))
        await coordinator.close_session("ws-1")

    async def test_commit_cancels_pending_candidate_before_finish(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def producer(request, budget):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        coordinator = PreSpecCoordinator(producer, self.policy,
                                         clock=lambda: self.now)
        coordinator.maybe_start(self.request(window_id=1), self.load(), 200)
        await started.wait()
        coordinator.begin_formal(
            "ws-1", 1, 2, 0, window_id=1, defer_candidate=True)
        await asyncio.wait_for(cancelled.wait(), 0.2)
        self.assertIsNone(coordinator.claim_formal_candidate(
            "ws-1", 1, window_id=1, prefix_version=2,
            full_audio_end_samples=62_000))
        coordinator.finish_formal("ws-1", 1, window_id=1)
        await coordinator.close_session("ws-1")

    async def test_commit_claims_published_partial_before_cancel_completes(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        probes = []
        coordinator = None

        async def producer(request, budget):
            started.set()
            self.assertFalse(coordinator.publish_partial(request, (19,)))
            self.assertTrue(coordinator.publish_partial(request, (19, 20, 21, 30)))
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        coordinator = PreSpecCoordinator(
            producer, self.policy, clock=lambda: self.now,
            shadow_only=False, on_probe=probes.append)
        coordinator.maybe_start(self.request(window_id=1), self.load(), 200)
        await started.wait()
        self.assertIsNone(coordinator.begin_formal(
            "ws-1", 1, 2, 64_000, window_id=1, defer_candidate=True))
        candidate = coordinator.claim_formal_candidate(
            "ws-1", 1, window_id=1, prefix_version=2,
            full_audio_end_samples=64_000)
        self.assertEqual(candidate.formal_draft_token_ids, (19, 20, 21, 30))
        self.assertTrue(candidate.is_partial)
        self.assertEqual(probes[-1].status, "partial_ready_consumed")
        await asyncio.wait_for(cancelled.wait(), 0.2)
        coordinator.finish_formal("ws-1", 1, window_id=1)
        await coordinator.close_session("ws-1")

    async def test_pending_chunk_uses_partial_only_when_longer_than_rollback(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        probes = []
        coordinator = None

        async def producer(request, budget):
            started.set()
            coordinator.publish_partial(request, (19, 20, 21))
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        coordinator = PreSpecCoordinator(
            producer, self.policy, clock=lambda: self.now,
            shadow_only=False, on_probe=probes.append)
        coordinator.maybe_start(self.request(), self.load(), 200)
        await started.wait()
        candidate = coordinator.begin_authoritative_chunk(
            "ws-1", audio_epoch=1, window_id=0, prefix_version=2,
            full_audio_end_samples=64_000)
        self.assertIsNone(candidate)
        self.assertEqual(probes[-1].status, "partial_not_longer_than_rollback")
        await asyncio.wait_for(cancelled.wait(), 0.2)
        coordinator.end_authoritative_chunk("ws-1", audio_epoch=1, window_id=0)
        await coordinator.close_session("ws-1")

    async def test_pending_chunk_can_consume_a_longer_partial(self):
        started = asyncio.Event()
        coordinator = None

        async def producer(request, budget):
            coordinator.publish_partial(request, (19, 20, 21, 30))
            started.set()
            await asyncio.Future()

        coordinator = PreSpecCoordinator(
            producer, self.policy, clock=lambda: self.now, shadow_only=False)
        coordinator.maybe_start(self.request(), self.load(), 200)
        await started.wait()
        candidate = coordinator.begin_authoritative_chunk(
            "ws-1", audio_epoch=1, window_id=0, prefix_version=2,
            full_audio_end_samples=64_000)
        self.assertEqual(candidate.formal_draft_token_ids, (19, 20, 21, 30))
        self.assertTrue(candidate.is_partial)
        coordinator.end_authoritative_chunk("ws-1", audio_epoch=1, window_id=0)
        await coordinator.close_session("ws-1")

    async def test_partial_rejects_stale_prefix_and_is_cleared_after_finish(self):
        started = asyncio.Event()
        probes = []
        coordinator = None

        async def producer(request, budget):
            coordinator.publish_partial(request, (19, 20, 21, 30))
            started.set()
            await asyncio.Future()

        coordinator = PreSpecCoordinator(
            producer, self.policy, clock=lambda: self.now,
            shadow_only=False, on_probe=probes.append)
        coordinator.maybe_start(self.request(window_id=1), self.load(), 200)
        await started.wait()
        coordinator.begin_formal("ws-1", 1, 2, 64_000,
                                 window_id=1, defer_candidate=True)
        self.assertIsNone(coordinator.claim_formal_candidate(
            "ws-1", 1, window_id=1, prefix_version=3,
            full_audio_end_samples=64_000))
        self.assertEqual(probes[-1].status, "version_mismatch")
        coordinator.finish_formal("ws-1", 1, window_id=1)
        self.assertFalse(coordinator._partials)
        self.assertFalse(coordinator._launch_info)
        await coordinator.close_session("ws-1")

    async def test_failed_producer_does_not_publish_partial_as_ready(self):
        probes = []
        coordinator = None

        async def producer(request, budget):
            coordinator.publish_partial(request, (19, 20, 21, 30))
            raise RuntimeError("backend failed after partial output")

        coordinator = PreSpecCoordinator(
            producer, self.policy, clock=lambda: self.now,
            shadow_only=False, on_probe=probes.append)
        coordinator.maybe_start(self.request(window_id=1), self.load(), 200)
        with self.assertRaisesRegex(RuntimeError, "backend failed after partial output"):
            await coordinator._tasks[("ws-1", 1, 1)]
        coordinator.begin_formal("ws-1", 1, 2, 64_000,
                                 window_id=1, defer_candidate=True)
        self.assertIsNone(coordinator.claim_formal_candidate(
            "ws-1", 1, window_id=1, prefix_version=2,
            full_audio_end_samples=64_000))
        self.assertEqual(probes[-1].status, "failed")
        coordinator.finish_formal("ws-1", 1, window_id=1)
        await coordinator.close_session("ws-1")

    async def test_partial_publication_requires_live_matching_request(self):
        started = asyncio.Event()
        coordinator = None
        original = self.request(window_id=1)

        async def producer(request, budget):
            self.assertFalse(coordinator.publish_partial(
                replace(request, window_id=2), (19, 20, 21, 30)))
            self.assertFalse(coordinator.publish_partial(
                replace(request, request_id="wrong"), (19, 20, 21, 30)))
            with self.assertRaises(ValueError):
                coordinator.publish_partial(request, (19, True))
            self.assertTrue(coordinator.publish_partial(request, (19, 20, 21, 30)))
            started.set()
            await asyncio.Future()

        coordinator = PreSpecCoordinator(
            producer, self.policy, clock=lambda: self.now, shadow_only=False)
        coordinator.maybe_start(original, self.load(), 200)
        await started.wait()
        self.assertFalse(coordinator.publish_partial(original, (19, 20, 21, 30, 31)))
        coordinator.begin_formal("ws-1", 1, 2, 64_000,
                                 window_id=1, defer_candidate=True)
        candidate = coordinator.claim_formal_candidate(
            "ws-1", 1, window_id=1, prefix_version=2,
            full_audio_end_samples=64_000)
        self.assertEqual(candidate.token_ids, (19, 20, 21, 30))
        coordinator.finish_formal("ws-1", 1, window_id=1)
        await coordinator.close_session("ws-1")

    async def test_duplicate_partial_snapshot_keeps_first_ready_time(self):
        started = asyncio.Event()
        probes = []
        coordinator = None

        async def producer(request, budget):
            coordinator.publish_partial(request, (19, 20, 21, 30))
            self.now += 20_000_000
            coordinator.publish_partial(request, (19, 20, 21, 30))
            started.set()
            await asyncio.Future()

        coordinator = PreSpecCoordinator(
            producer, self.policy, clock=lambda: self.now,
            shadow_only=False, on_probe=probes.append)
        coordinator.maybe_start(self.request(window_id=1), self.load(), 200)
        await started.wait()
        coordinator.begin_formal("ws-1", 1, 2, 64_000,
                                 window_id=1, defer_candidate=True)
        coordinator.claim_formal_candidate("ws-1", 1, window_id=1,
                                            prefix_version=2,
                                            full_audio_end_samples=64_000)
        self.assertEqual(probes[-1].ready_lead_ms, 20)
        coordinator.finish_formal("ws-1", 1, window_id=1)
        await coordinator.close_session("ws-1")

    async def test_commit_short_tail_discards_unclaimed_ready_candidate(self):
        async def producer(request, budget):
            return (19, 20)

        coordinator = PreSpecCoordinator(producer, self.policy,
                                         clock=lambda: self.now)
        coordinator.maybe_start(self.request(window_id=1), self.load(), 200)
        await asyncio.sleep(0)
        coordinator.begin_formal(
            "ws-1", 1, 2, 0, window_id=1, defer_candidate=True)
        coordinator.finish_formal("ws-1", 1, window_id=1)
        self.assertFalse(coordinator._tasks)
        self.assertEqual(coordinator.maybe_start(
            self.request(window_id=2), self.load(), 200).reason,
            "epoch_already_used")
        await coordinator.close_session("ws-1")

    async def test_commit_cannot_claim_previous_window_after_full_chunk(self):
        async def producer(request, budget):
            return (19, 20)

        coordinator = PreSpecCoordinator(
            producer, self.policy, clock=lambda: self.now,
            shadow_only=False)
        coordinator.maybe_start(self.request(window_id=1), self.load(), 200)
        await asyncio.sleep(0)
        coordinator.begin_formal(
            "ws-1", 1, 2, 0, window_id=1, defer_candidate=True)
        candidate = coordinator.begin_authoritative_chunk(
            "ws-1", audio_epoch=1, window_id=1, prefix_version=2,
            full_audio_end_samples=64_000)
        self.assertIsNotNone(candidate)
        coordinator.end_authoritative_chunk(
            "ws-1", audio_epoch=1, window_id=1)
        self.assertIsNone(coordinator.claim_formal_candidate(
            "ws-1", 1, window_id=2, prefix_version=3,
            full_audio_end_samples=70_000))
        coordinator.finish_formal("ws-1", 1, window_id=1)
        await coordinator.close_session("ws-1")

    async def test_chunk_end_rejects_a_different_window_id(self):
        async def producer(request, budget):
            return (19, 20)

        coordinator = PreSpecCoordinator(producer, self.policy,
                                         clock=lambda: self.now)
        coordinator.begin_authoritative_chunk(
            "ws-1", audio_epoch=1, window_id=1, prefix_version=2,
            full_audio_end_samples=64_000)
        with self.assertRaisesRegex(ValueError, "window"):
            coordinator.end_authoritative_chunk(
                "ws-1", audio_epoch=1, window_id=2)
        coordinator.end_authoritative_chunk(
            "ws-1", audio_epoch=1, window_id=1)
        await coordinator.close_session("ws-1")

    async def test_formal_end_rejects_a_different_window_id(self):
        async def producer(request, budget):
            return (19, 20)

        coordinator = PreSpecCoordinator(producer, self.policy,
                                         clock=lambda: self.now)
        coordinator.begin_formal("ws-1", 1, 2, 0, window_id=2,
                                 allow_candidate=False)
        with self.assertRaisesRegex(ValueError, "window mismatch"):
            coordinator.finish_formal("ws-1", 1, window_id=3)
        coordinator.finish_formal("ws-1", 1, window_id=2)
        await coordinator.close_session("ws-1")

    async def test_consume_only_ready_version_compatible_candidate(self):
        async def producer(request, budget):
            return (20, 21)

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now,
                                         shadow_only=False)
        coordinator.maybe_start(self.request(), self.load(), predicted_draft_ms=200)
        await asyncio.sleep(0)
        self.assertIsNone(coordinator.begin_formal("ws-1", 1, 3, 64_000, window_id=0))
        coordinator.finish_formal("ws-1", 1, (20, 21), window_id=0)
        request = self.request(epoch=2)
        coordinator.maybe_start(request, self.load(),
                                predicted_draft_ms=200)
        await asyncio.sleep(0)
        candidate = coordinator.begin_formal("ws-1", 2, 2, 64_000, window_id=0)
        self.assertEqual(candidate.token_ids, (20, 21))
        self.assertEqual(candidate.request_id, request.request_id)
        coordinator.finish_formal("ws-1", 2, (20, 21), window_id=0)
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
        coordinator.begin_formal("ws-1", 1, 2, 64_000, window_id=0)
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
                                                    allow_candidate=False, window_id=0))
        self.assertIsNone(coordinator.finish_formal("ws-1", 1, (), window_id=0))
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
        coordinator.begin_formal("ws-1", 0, 2, 32_000, window_id=0)
        decision = coordinator.maybe_start(self.request(), self.load(), 200)
        self.assertEqual(decision.reason, "session_formal_inflight")
        coordinator.finish_formal("ws-1", 0, (), window_id=0)
        self.assertTrue(coordinator.maybe_start(self.request(), self.load(), 200).launched)
        await coordinator.close_session("ws-1")

    async def test_streaming_chunk_inflight_blocks_and_preempts_shadow(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        probes = []

        async def producer(request, budget):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now,
                                         on_probe=probes.append)
        coordinator.begin_authoritative_chunk("ws-1")
        self.assertEqual(coordinator.maybe_start(self.request(), self.load(), 200).reason,
                         "session_chunk_inflight")
        coordinator.end_authoritative_chunk("ws-1")
        self.assertTrue(coordinator.maybe_start(self.request(), self.load(), 200).launched)
        await started.wait()
        coordinator.begin_authoritative_chunk("ws-1")
        await asyncio.wait_for(cancelled.wait(), 0.2)
        coordinator.end_authoritative_chunk("ws-1")
        self.assertIsNone(coordinator.begin_formal("ws-1", 1, 2, 64_000, window_id=0))
        self.assertEqual(probes[-1].status, "discarded_before_boundary")
        coordinator.finish_formal("ws-1", 1, (), window_id=0)
        await coordinator.close_session("ws-1")

    async def test_other_session_targets_do_not_preempt_or_block_shadow(self):
        release = asyncio.Event()
        cancelled = []

        async def producer(request, budget):
            try:
                await release.wait()
                return (19, 20, 21)
            except asyncio.CancelledError:
                cancelled.append(request.session_id)
                raise

        policy = replace(self.policy, max_active_shadows=3)
        coordinator = PreSpecCoordinator(producer, policy, clock=lambda: self.now,
                                         shadow_only=False)
        coordinator.maybe_start(self.request(), self.load(), 200)
        first_task = coordinator._tasks[("ws-1", 1, 0)]
        await asyncio.sleep(0)
        coordinator.begin_authoritative_chunk("ws-2")
        coordinator.begin_formal("ws-2", 1, 2, 0, window_id=0, defer_candidate=True)
        other = replace(self.request(), session_id="ws-3")
        self.assertTrue(coordinator.maybe_start(other, self.load(), 200).launched)
        second_task = coordinator._tasks[("ws-3", 1, 0)]
        release.set()
        await asyncio.gather(first_task, second_task)
        self.assertEqual(cancelled, [])

        coordinator.end_authoritative_chunk("ws-2")
        coordinator.finish_formal("ws-2", 1, window_id=0)
        for session_id in ("ws-1", "ws-3"):
            candidate = coordinator.begin_authoritative_chunk(
                session_id, audio_epoch=1, window_id=0, prefix_version=2,
                full_audio_end_samples=64_000)
            self.assertIsNotNone(candidate)
            self.assertEqual(candidate.session_id, session_id)
            coordinator.end_authoritative_chunk(session_id, audio_epoch=1, window_id=0)
            await coordinator.close_session(session_id)

    async def test_default_policy_runs_twenty_connection_shadows_concurrently(self):
        release = asyncio.Event()
        all_started = asyncio.Event()
        started = set()

        async def producer(request, budget):
            started.add(request.session_id)
            if len(started) == 20:
                all_started.set()
            await release.wait()
            return (19, 20, 21)

        coordinator = PreSpecCoordinator(producer, clock=lambda: self.now,
                                         shadow_only=False)
        requests = [replace(self.request(), session_id=f"ws-{i}") for i in range(20)]
        for request in requests:
            self.assertTrue(coordinator.maybe_start(request, self.load(), 200).launched)
        await asyncio.wait_for(all_started.wait(), 0.5)
        tasks = list(coordinator._tasks.values())
        self.assertTrue(all(not task.done() for task in tasks))
        release.set()
        await asyncio.gather(*tasks)
        for request in requests:
            candidate = coordinator.begin_authoritative_chunk(
                request.session_id, audio_epoch=1, window_id=0, prefix_version=2,
                full_audio_end_samples=64_000)
            self.assertIsNotNone(candidate)
            self.assertEqual(candidate.request_id, request.request_id)
            coordinator.end_authoritative_chunk(request.session_id,
                                                 audio_epoch=1, window_id=0)
            await coordinator.close_session(request.session_id)

    async def test_other_targets_preserve_completed_candidate_until_its_boundary(self):
        async def producer(request, budget):
            return (19, 20, 21)

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now,
                                         shadow_only=False)
        request = self.request()
        coordinator.maybe_start(request, self.load(), 200)
        await coordinator._tasks[("ws-1", 1, 0)]
        coordinator.begin_authoritative_chunk("ws-2")
        coordinator.end_authoritative_chunk("ws-2")
        coordinator.begin_formal("ws-3", 1, 2, 0, window_id=0, defer_candidate=True)
        coordinator.finish_formal("ws-3", 1, window_id=0)
        candidate = coordinator.begin_authoritative_chunk(
            "ws-1", audio_epoch=1, window_id=0, prefix_version=2,
            full_audio_end_samples=64_000)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.request_id, request.request_id)
        coordinator.end_authoritative_chunk("ws-1", audio_epoch=1, window_id=0)
        await coordinator.close_session("ws-1")

    async def test_negative_shadow_capacity_is_rejected(self):
        async def producer(request, budget):
            return (19, 20, 21)

        with self.assertRaisesRegex(ValueError, "max_active_shadows"):
            PreSpecCoordinator(producer, replace(self.policy, max_active_shadows=-1))

    async def test_ready_candidate_is_consumed_at_precommit_chunk_submit(self):
        async def producer(request, budget):
            return (20, 21, 22)

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now,
                                         shadow_only=False)
        coordinator.maybe_start(self.request(), self.load(), 200)
        await asyncio.sleep(0)
        candidate = coordinator.begin_authoritative_chunk(
            "ws-1", audio_epoch=1, prefix_version=2,
            full_audio_end_samples=64_000, window_id=0)
        self.assertEqual(candidate.token_ids, (20, 21, 22))
        evaluation = coordinator.end_authoritative_chunk(
            "ws-1", audio_epoch=1, target_continuation_tokens=(20, 21, 99), window_id=0)
        self.assertEqual(evaluation.continuous_accepted_tokens, 2)
        self.assertIsNone(coordinator.begin_formal("ws-1", 1, 2, 64_000, window_id=0))
        self.assertIsNone(coordinator.finish_formal("ws-1", 1, (20, 21, 99), window_id=0))
        await coordinator.close_session("ws-1")

    async def test_chunk_without_target_tokens_discards_candidate_without_evaluation(self):
        async def producer(request, budget):
            return (20, 21)

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now,
                                         shadow_only=False)
        coordinator.maybe_start(self.request(), self.load(), 200)
        await asyncio.sleep(0)
        candidate = coordinator.begin_authoritative_chunk(
            "ws-1", audio_epoch=1, prefix_version=2,
            full_audio_end_samples=64_000, window_id=0)
        self.assertIsNotNone(candidate)
        self.assertIsNone(coordinator.end_authoritative_chunk("ws-1", audio_epoch=1, window_id=0))
        self.assertNotIn(("ws-1", 1, 0), coordinator._selected)
        self.assertNotIn(("ws-1", 1, 0), coordinator._selected_at_ns)
        coordinator.begin_formal("ws-1", 1, 2, 64_000, window_id=0)
        self.assertIsNone(coordinator.finish_formal("ws-1", 1, window_id=0))
        await coordinator.close_session("ws-1")

    async def test_formal_without_target_tokens_discards_candidate_without_evaluation(self):
        async def producer(request, budget):
            return (20, 21)

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now,
                                         shadow_only=False)
        coordinator.maybe_start(self.request(), self.load(), 200)
        await asyncio.sleep(0)
        self.assertIsNotNone(coordinator.begin_formal("ws-1", 1, 2, 64_000, window_id=0))
        self.assertIsNone(coordinator.finish_formal("ws-1", 1, window_id=0))
        self.assertNotIn(("ws-1", 1, 0), coordinator._selected)
        self.assertNotIn(("ws-1", 1, 0), coordinator._selected_at_ns)
        await coordinator.close_session("ws-1")

    async def test_probe_records_readiness_at_chunk_submit_without_waiting(self):
        probes = []
        async def producer(request, budget):
            return (20, 21)

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now,
                                         on_probe=probes.append)
        coordinator.maybe_start(self.request(), self.load(), 200)
        await asyncio.sleep(0)
        self.now += 35_000_000
        self.assertIsNone(coordinator.begin_authoritative_chunk(
            "ws-1", audio_epoch=1, prefix_version=2,
            full_audio_end_samples=64_000, window_id=0))
        self.assertEqual(probes[-1].boundary, "chunk")
        self.assertEqual(probes[-1].status, "ready_shadow_only")
        self.assertEqual(probes[-1].ready_lead_ms, 35)
        coordinator.end_authoritative_chunk(
            "ws-1", audio_epoch=1, target_continuation_tokens=(20, 21), window_id=0)
        coordinator.begin_formal("ws-1", 1, 2, 64_000, window_id=1)
        self.assertEqual(probes[-1].status, "not_started")
        coordinator.finish_formal("ws-1", 1, (), window_id=1)
        await coordinator.close_session("ws-1")

    async def test_probe_records_pending_shadow_and_sink_failure_isolated(self):
        started = asyncio.Event()
        probes = []
        async def producer(request, budget):
            started.set()
            await asyncio.Future()

        def sink(probe):
            probes.append(probe)
            raise RuntimeError("probe writer failed")

        coordinator = PreSpecCoordinator(producer, self.policy, clock=lambda: self.now,
                                         on_probe=sink)
        coordinator.maybe_start(self.request(), self.load(), 200)
        await started.wait()
        self.assertIsNone(coordinator.begin_authoritative_chunk(
            "ws-1", audio_epoch=1, prefix_version=2,
            full_audio_end_samples=64_000, window_id=0))
        self.assertEqual(probes[-1].status, "pending")
        self.assertEqual(coordinator.probe_error_count, 1)
        coordinator.end_authoritative_chunk(
            "ws-1", audio_epoch=1, target_continuation_tokens=(), window_id=0)
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
