"""Service-side coordinator for an isolated, midpoint ASR shadow request.

The model adapter supplies ``producer`` and submits the complete audio input
available at the midpoint under the existing service input contract. The engine
owns request state and decides encoder/prefix/KV reuse. The producer verifies
and repairs the old rollback suffix, then extends that corrected result. This
module never modifies authoritative state or chooses cached audio/KV ranges.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal, Protocol, Sequence
from uuid import uuid4


@dataclass(frozen=True)
class ShadowRequest:
    """Full audio snapshot plus timing metadata, not a cache-delta request.

    ``window_audio_samples`` is the audio progress since the last formal input
    cutoff, used only for admission. ``audio_pcm_f32le`` holds the complete audio
    input at this snapshot's cutoff; it must not be sliced to that progress count.
    """

    session_id: str
    audio_epoch: int
    base_prefix_version: int
    visible_audio_end_samples: int
    window_audio_samples: int
    audio_pcm_f32le: bytes
    stable_prefix_tokens: tuple[int, ...]
    rollback_tokens: tuple[int, ...]
    deadline_ns: int
    request_id: str = field(default_factory=lambda: f"prespec-{uuid4().hex}")


@dataclass(frozen=True)
class LoadSnapshot:
    waiting: int
    utilization_pct: float
    observed_ns: int
    formal_queue_wait_p95_ms: float
    formal_pending: int


@dataclass(frozen=True)
class AdmissionPolicy:
    admission_mode: Literal["gated", "force"] = "gated"
    midpoint_samples: int = 16_000
    max_utilization_pct: float = 40.0
    max_load_age_ms: float = 200.0
    cancel_margin_ms: float = 100.0
    safety_guard_ms: float = 100.0
    max_active_shadows: int = 1
    max_candidate_tokens: int = 8


@dataclass(frozen=True)
class LaunchDecision:
    launched: bool
    reason: str
    slack_ms: float | None = None


@dataclass(frozen=True)
class Candidate:
    request_id: str
    session_id: str
    audio_epoch: int
    base_prefix_version: int
    visible_audio_end_samples: int
    token_ids: tuple[int, ...]
    ready_ns: int


@dataclass(frozen=True)
class Evaluation:
    session_id: str
    audio_epoch: int
    candidate_tokens: tuple[int, ...]
    continuous_accepted_tokens: int
    ready_before_formal: bool
    ready_lead_ms: float


@dataclass(frozen=True)
class CandidateProbe:
    session_id: str
    audio_epoch: int
    boundary: Literal["chunk", "formal"]
    target_start_ns: int
    status: str
    ready_lead_ms: float | None = None
    request_id: str | None = None


Producer = Callable[[ShadowRequest, int], Awaitable[Sequence[int]]]
Clock = Callable[[], int]
ProbeSink = Callable[[CandidateProbe], None]


@dataclass(frozen=True)
class VerifiedRollback:
    """Corrected rollback suffix plus adapter-owned shadow continuation state."""

    token_ids: tuple[int, ...]
    continuation_state: object


class ShadowBranch(Protocol):
    """Optional engine adapter for a non-authoritative request's decode state.

    ``extend_from_verified`` must continue from the corrected state returned by
    ``verify_and_repair_old``, never from the raw rollback hypothesis. The engine
    decides physical cache sharing; this interface does not require the service
    to copy KV or decide reusable ranges.
    """

    async def verify_and_repair_old(self, rollback_tokens: tuple[int, ...],
                                    budget: int) -> VerifiedRollback: ...

    async def extend_from_verified(self, verified: VerifiedRollback,
                                   budget: int) -> Sequence[int]: ...

    async def close(self) -> None: ...


class ShadowBackend(Protocol):
    async def open_shadow(self, request: ShadowRequest) -> ShadowBranch: ...


class RepairExtendProducer:
    """Reuse rollback verification, then extend its corrected shadow state."""

    def __init__(self, backend: ShadowBackend) -> None:
        self.backend = backend

    async def __call__(self, request: ShadowRequest,
                       max_candidate_tokens: int) -> tuple[int, ...]:
        branch = await self.backend.open_shadow(request)
        try:
            verified = await branch.verify_and_repair_old(
                request.rollback_tokens[:max_candidate_tokens],
                max_candidate_tokens)
            if not isinstance(verified, VerifiedRollback):
                raise TypeError("rollback verifier must return VerifiedRollback")
            if verified.continuation_state is None:
                raise ValueError("rollback verifier returned no continuation state")
            old = verified.token_ids
            if not isinstance(old, tuple) or not all(
                    isinstance(token, int) and token >= 0 for token in old):
                raise ValueError("rollback verifier returned invalid token IDs")
            if len(old) > max_candidate_tokens:
                raise ValueError("repaired old suffix exceeds candidate budget")
            remaining = max_candidate_tokens - len(old)
            extension = (tuple(await branch.extend_from_verified(verified, remaining))
                         if remaining else ())
            if len(extension) > remaining:
                raise ValueError("new continuation exceeds candidate budget")
            return old + extension
        finally:
            await branch.close()


class PreSpecCoordinator:
    """One process-wide coordinator; call its methods on one asyncio event loop.

    ``shadow_only=True`` is the safe default: formal decoding never receives
    draft tokens. With ``False``, ``begin_formal`` returns a ready candidate to
    the service's existing authoritative verifier; it is never a final answer.
    """

    def __init__(self, producer: Producer,
                 policy: AdmissionPolicy = AdmissionPolicy(), *,
                 clock: Clock = time.monotonic_ns,
                 shadow_only: bool = True,
                 on_probe: ProbeSink | None = None) -> None:
        self.producer = producer
        self.policy = policy
        if policy.admission_mode not in ("gated", "force"):
            raise ValueError("admission_mode must be gated or force")
        self.clock = clock
        self.shadow_only = shadow_only
        self.on_probe = on_probe
        self.probe_error_count = 0
        self._tasks: dict[tuple[str, int], asyncio.Task[Candidate]] = {}
        self._reserved: set[asyncio.Task[Candidate]] = set()
        self._task_sessions: dict[asyncio.Task[Candidate], str] = {}
        self._launched: set[tuple[str, int]] = set()
        self._begun: set[tuple[str, int]] = set()
        self._selected: dict[tuple[str, int], Candidate] = {}
        self._selected_at_ns: dict[tuple[str, int], int] = {}
        self._active_formals: dict[str, int] = {}
        self._active_chunks: dict[str, int] = {}

    def maybe_start(self, request: ShadowRequest, load: LoadSnapshot | None,
                    predicted_draft_ms: float) -> LaunchDecision:
        """Try one midpoint request; never put a shadow task in a waiting queue."""
        key = (request.session_id, request.audio_epoch)
        now = self.clock()
        if request.window_audio_samples < self.policy.midpoint_samples:
            return LaunchDecision(False, "before_midpoint")
        if key in self._launched or key in self._begun:
            return LaunchDecision(False, "epoch_already_used")
        if self._active_formals.get(request.session_id, 0):
            return LaunchDecision(False, "session_formal_inflight")
        if self._active_formals:
            return LaunchDecision(False, "formal_inflight_on_device")
        if self._active_chunks.get(request.session_id, 0):
            return LaunchDecision(False, "session_chunk_inflight")
        if len(self._reserved) >= self.policy.max_active_shadows:
            return LaunchDecision(False, "shadow_capacity_full")
        if predicted_draft_ms < 0:
            raise ValueError("predicted durations must be nonnegative")
        if self.policy.admission_mode == "gated":
            if load is None:
                return LaunchDecision(False, "missing_load")
            if load.formal_queue_wait_p95_ms < 0:
                raise ValueError("predicted durations must be nonnegative")
            if load.waiting != 0 or load.formal_pending != 0:
                return LaunchDecision(False, "formal_queue_nonempty")
            age_ms = (now - load.observed_ns) / 1e6
            if age_ms < 0 or age_ms > self.policy.max_load_age_ms:
                return LaunchDecision(False, "stale_load")
            if not 0 <= load.utilization_pct <= self.policy.max_utilization_pct:
                return LaunchDecision(False, "utilization_high")
            slack_ms = ((request.deadline_ns - now) / 1e6
                        - load.formal_queue_wait_p95_ms
                        - predicted_draft_ms
                        - self.policy.cancel_margin_ms
                        - self.policy.safety_guard_ms)
        else:
            slack_ms = (request.deadline_ns - now) / 1e6
        if slack_ms <= 0:
            return LaunchDecision(False, "insufficient_slack", slack_ms)
        task = asyncio.get_running_loop().create_task(self._produce(request))
        self._tasks[key] = task
        self._reserved.add(task)
        self._task_sessions[task] = request.session_id
        self._launched.add(key)
        task.add_done_callback(self._release)
        return LaunchDecision(True, "launched", slack_ms)

    async def _produce(self, request: ShadowRequest) -> Candidate:
        token_ids = tuple(await self.producer(request, self.policy.max_candidate_tokens))
        if not token_ids or len(token_ids) > self.policy.max_candidate_tokens:
            raise ValueError("producer must return 1..max_candidate_tokens token IDs")
        if not all(isinstance(token, int) and token >= 0 for token in token_ids):
            raise ValueError("producer returned invalid token IDs")
        return Candidate(request.request_id, request.session_id, request.audio_epoch,
                         request.base_prefix_version,
                         request.visible_audio_end_samples, token_ids,
                         self.clock())

    def _release(self, task: asyncio.Task[Candidate]) -> None:
        self._reserved.discard(task)
        self._task_sessions.pop(task, None)
        if not task.cancelled():
            task.exception()  # Observe errors even if no formal boundary follows.

    def begin_authoritative_chunk(self, session_id: str, *,
                                  audio_epoch: int | None = None,
                                  prefix_version: int | None = None,
                                  full_audio_end_samples: int | None = None,
                                  allow_candidate: bool = True) -> Candidate | None:
        """At target chunk submit, take a ready draft or cancel shadow work."""
        if audio_epoch is not None and (prefix_version is None or
                                        full_audio_end_samples is None):
            raise ValueError("chunk candidate needs prefix version and audio end")
        started_ns = self.clock()
        previous_active = self._active_chunks.get(session_id, 0)
        self._active_chunks[session_id] = previous_active + 1
        key = (session_id, audio_epoch) if audio_epoch is not None else None
        task = self._tasks.pop(key, None) if key is not None else None
        self._cancel_pending_shadows()
        if key is None:
            return None
        if key in self._selected:
            self._emit_probe(key, "chunk", started_ns, "already_selected")
            return None
        return self._select_candidate(task, key, "chunk", started_ns, prefix_version,
                                      full_audio_end_samples,
                                      allow_candidate and not previous_active)

    def _cancel_pending_shadows(self) -> None:
        for key in list(self._tasks):
            task = self._tasks.pop(key)
            if task.done():
                self._reserved.discard(task)
            else:
                task.cancel()

    def end_authoritative_chunk(self, session_id: str, *,
                                audio_epoch: int | None = None,
                                target_continuation_tokens: Sequence[int] | None = None
                                ) -> Evaluation | None:
        """End a chunk; omit target tokens to discard without quality scoring."""
        active = self._active_chunks.get(session_id, 0)
        if active <= 0:
            raise ValueError(f"no authoritative chunk in flight: {session_id}")
        if active == 1:
            del self._active_chunks[session_id]
        else:
            self._active_chunks[session_id] = active - 1
        if audio_epoch is not None:
            return self._evaluate((session_id, audio_epoch),
                                  target_continuation_tokens)
        return None

    def _emit_probe(self, key: tuple[str, int],
                    boundary: Literal["chunk", "formal"], started_ns: int,
                    status: str, candidate: Candidate | None = None) -> None:
        if self.on_probe is None:
            return
        probe = CandidateProbe(
            key[0], key[1], boundary, started_ns, status,
            (started_ns - candidate.ready_ns) / 1e6 if candidate else None,
            candidate.request_id if candidate else None)
        try:
            self.on_probe(probe)
        except Exception:
            self.probe_error_count += 1  # Measurement must not fail target work.

    def _select_candidate(self, task: asyncio.Task[Candidate] | None,
                          key: tuple[str, int],
                          boundary: Literal["chunk", "formal"], started_ns: int,
                          prefix_version: int,
                          full_audio_end_samples: int,
                          allow_candidate: bool) -> Candidate | None:
        if task is None:
            self._emit_probe(key, boundary, started_ns,
                             "previously_used" if key in self._launched else "not_started")
            return None
        if not task.done():
            task.cancel()
            self._emit_probe(key, boundary, started_ns, "pending")
            return None
        self._reserved.discard(task)
        try:
            candidate = task.result()
        except (asyncio.CancelledError, Exception):
            self._emit_probe(key, boundary, started_ns, "failed")
            return None
        if not allow_candidate:
            self._emit_probe(key, boundary, started_ns, "disallowed", candidate)
            return None
        if candidate.base_prefix_version != prefix_version:
            self._emit_probe(key, boundary, started_ns, "version_mismatch", candidate)
            return None
        if (candidate.audio_epoch != key[1]
                or candidate.visible_audio_end_samples > full_audio_end_samples):
            self._emit_probe(key, boundary, started_ns, "audio_mismatch", candidate)
            return None
        self._selected[key] = candidate
        self._selected_at_ns[key] = started_ns
        self._emit_probe(key, boundary, started_ns,
                         "ready_shadow_only" if self.shadow_only else "ready_consumed",
                         candidate)
        return None if self.shadow_only else candidate

    def begin_formal(self, session_id: str, audio_epoch: int,
                     prefix_version: int,
                     full_audio_end_samples: int, *,
                     allow_candidate: bool = True) -> Candidate | None:
        """Nonblocking formal entry; cancel an unfinished shadow immediately."""
        key = (session_id, audio_epoch)
        if key in self._begun:
            raise ValueError(f"formal boundary already begun: {key}")
        started_ns = self.clock()
        previous_active = self._active_formals.get(session_id, 0)
        self._active_formals[session_id] = previous_active + 1
        self._begun.add(key)
        task = self._tasks.pop(key, None)
        self._cancel_pending_shadows()
        if key in self._selected:
            self._emit_probe(key, "formal", started_ns, "already_selected")
            return None
        return self._select_candidate(task, key, "formal", started_ns, prefix_version,
                                      full_audio_end_samples,
                                      allow_candidate and not previous_active)

    def finish_formal(self, session_id: str, audio_epoch: int,
                      target_continuation_tokens: Sequence[int] | None = None
                      ) -> Evaluation | None:
        """End a formal boundary; optional tokens enable quality scoring."""
        key = (session_id, audio_epoch)
        if key not in self._begun:
            raise ValueError(f"formal boundary was not begun: {key}")
        self._begun.remove(key)
        active = self._active_formals[session_id] - 1
        if active:
            self._active_formals[session_id] = active
        else:
            del self._active_formals[session_id]
        self._launched.discard(key)
        return self._evaluate(key, target_continuation_tokens)

    def _evaluate(self, key: tuple[str, int],
                  target_continuation_tokens: Sequence[int] | None
                  ) -> Evaluation | None:
        candidate = self._selected.pop(key, None)
        if candidate is None:
            return None
        ready_lead_ms = (self._selected_at_ns.pop(key) - candidate.ready_ns) / 1e6
        if target_continuation_tokens is None:
            return None
        accepted = 0
        for draft, actual in zip(candidate.token_ids, target_continuation_tokens):
            if draft != actual:
                break
            accepted += 1
        return Evaluation(key[0], key[1], candidate.token_ids, accepted,
                          ready_before_formal=True,
                          ready_lead_ms=ready_lead_ms)

    async def close_session(self, session_id: str) -> None:
        """Cancel and await shadow work when the WebSocket session closes."""
        keys = [key for key in self._tasks if key[0] == session_id]
        for key in keys:
            self._tasks.pop(key)
        tasks = [task for task, owner in self._task_sessions.items()
                 if owner == session_id]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._launched = {key for key in self._launched if key[0] != session_id}
        self._begun = {key for key in self._begun if key[0] != session_id}
        self._selected = {key: candidate for key, candidate in self._selected.items()
                          if key[0] != session_id}
        self._selected_at_ns = {key: when for key, when in self._selected_at_ns.items()
                                if key[0] != session_id}
        self._active_formals.pop(session_id, None)
        self._active_chunks.pop(session_id, None)
