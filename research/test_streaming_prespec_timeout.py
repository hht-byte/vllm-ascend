"""Timeout handling without an NPU or a Windows asyncio socketpair."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from streaming_prespec import AdmissionPolicy, Candidate, PreSpecCoordinator
from streaming_prespec_helpers import PreSpecModelHelpers
import streaming_prespec_helpers


def run_without_loop(coroutine):
    while True:
        try:
            coroutine.send(None)
        except StopIteration as stopped:
            return stopped.value


class NoopTimeout:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


class TimeoutEngine:
    def __init__(self):
        self.aborted = []

    async def generate(self, **_):
        yield SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[10, 11, 12], finish_reason=None)],
            finished=False,
        )
        raise TimeoutError("deadline")

    async def abort(self, request_id):
        self.aborted.append(request_id)


class TimeoutModel(PreSpecModelHelpers):
    def __init__(self):
        self.prespec_engine = TimeoutEngine()
        self.partials = []
        self.prespec = SimpleNamespace(
            clock=lambda: 0,
            publish_partial=lambda request, tokens: self.partials.append(
                (request.request_id, tuple(tokens))),
        )

    def _build_stream_request(self, request, budget):
        return {"request_id": request.request_id}


class TimeoutTests(unittest.TestCase):
    def test_output_snapshot_is_saved_before_timeout(self):
        model = TimeoutModel()
        request = SimpleNamespace(request_id="prespec-test", deadline_ns=1_000_000_000)
        with patch.object(streaming_prespec_helpers.asyncio, "timeout",
                          lambda _: NoopTimeout()):
            with self.assertRaises(TimeoutError):
                run_without_loop(model._run_prespec(request, 15))
        self.assertEqual(model.partials, [("prespec-test", (10, 11, 12))])
        self.assertEqual(model.prespec_engine.aborted, ["prespec-test"])

    def test_timed_out_task_can_supply_published_partial(self):
        probes = []
        coordinator = PreSpecCoordinator(
            producer=None,
            policy=AdmissionPolicy(admission_mode="force",
                                   max_rollback_draft_tokens=5,
                                   max_candidate_tokens=5),
            shadow_only=False,
            on_probe=probes.append,
        )
        key = ("ws-test", 0, 0)
        partial = Candidate("prespec-test", "ws-test", 0, 0, 1, 16000,
                            (10, 11, 12), 100, is_partial=True,
                            rollback_draft_length=1)
        coordinator._partials[key] = partial

        class TimedOutTask:
            def done(self):
                return True

            def result(self):
                raise TimeoutError("deadline")

        selected = coordinator._select_candidate(
            TimedOutTask(), key, "chunk", 200, 1, 32000, True)
        self.assertIs(selected, partial)
        self.assertEqual(probes[-1].status, "partial_ready_consumed")

    def test_timed_out_task_without_snapshot_falls_back(self):
        probes = []
        coordinator = PreSpecCoordinator(
            producer=None,
            policy=AdmissionPolicy(admission_mode="force",
                                   max_rollback_draft_tokens=5,
                                   max_candidate_tokens=5),
            shadow_only=False,
            on_probe=probes.append,
        )

        class TimedOutTask:
            def done(self):
                return True

            def result(self):
                raise TimeoutError("deadline")

        self.assertIsNone(coordinator._select_candidate(
            TimedOutTask(), ("ws-test", 0, 0), "chunk", 200, 1, 32000, True))
        self.assertEqual(probes[-1].status, "timeout")


if __name__ == "__main__":
    unittest.main()
