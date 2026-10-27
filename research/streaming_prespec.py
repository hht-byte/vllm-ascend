"""Service-side coordinator for an isolated, midpoint ASR shadow request.

The model adapter supplies ``producer`` and submits the complete audio input
available at the midpoint under the existing service input contract. The engine
owns request state and decides encoder/prefix/KV reuse. The producer verifies
and repairs the old rollback suffix, then extends that corrected result. This
module never modifies authoritative state or chooses cached audio/KV ranges.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal, Protocol, Sequence
from uuid import uuid4


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShadowRequest:
    """Full audio snapshot plus timing metadata, not a cache-delta request.

    ``window_audio_samples`` is the audio progress since the last formal input
    cutoff, used only for admission. ``audio_pcm_f32le`` holds the complete audio
    input at this snapshot's cutoff; it must not be sliced to that progress count.
    """

    session_id: str
    audio_epoch: int
    window_id: int
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
    max_active_shadows: int = 0  # 0 disables the coordinator-wide concurrency cap.
    max_rollback_draft_tokens: int = 8
    max_candidate_tokens: int = 9


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
    window_id: int
    base_prefix_version: int
    visible_audio_end_samples: int
    token_ids: tuple[int, ...]
    ready_ns: int
    is_partial: bool = False
    rollback_draft_length: int = 0

    @property
    def formal_draft_token_ids(self) -> tuple[int, ...]:
        """The pasted proposer consumes drafts after target's first sampled token."""
        return self.token_ids[1:]


@dataclass(frozen=True)
class Evaluation:
    """Offline output comparisons; neither match count is verifier acceptance."""

    session_id: str
    audio_epoch: int
    window_id: int
    candidate_tokens: tuple[int, ...]
    raw_matched_prefix_tokens: int
    draft_matched_prefix_tokens: int
    target_token_count: int
    ready_before_formal: bool
    ready_lead_ms: float
    candidate_is_partial: bool = False

    @property
    def continuous_accepted_tokens(self) -> int:
        """Compatibility alias for the old raw-output comparison, not verifier acceptance."""
        return self.raw_matched_prefix_tokens


@dataclass(frozen=True)
class CandidateProbe:
    session_id: str
    audio_epoch: int
    window_id: int
    boundary: Literal["chunk", "formal"]
    target_start_ns: int
    status: str
    ready_lead_ms: float | None = None
    request_id: str | None = None


Producer = Callable[[ShadowRequest, int], Awaitable[Sequence[int]]]
Clock = Callable[[], int]
ProbeSink = Callable[[CandidateProbe], None]
WindowKey = tuple[str, int, int]
SegmentKey = tuple[str, int]


@dataclass(frozen=True)
class VerifiedRollback:
    """First sampled token, corrected old suffix, and shadow continuation state."""

    first_sampled_token: int
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
        if len(request.rollback_tokens) + 2 > max_candidate_tokens:
            raise ValueError("candidate budget cannot cover first token, old draft and extension")
        branch = await self.backend.open_shadow(request)
        try:
            verified = await branch.verify_and_repair_old(
                request.rollback_tokens,
                max_candidate_tokens)
            if not isinstance(verified, VerifiedRollback):
                raise TypeError("rollback verifier must return VerifiedRollback")
            if (not isinstance(verified.first_sampled_token, int)
                    or verified.first_sampled_token < 0):
                raise ValueError("rollback verifier returned invalid first token")
            if verified.continuation_state is None:
                raise ValueError("rollback verifier returned no continuation state")
            old = verified.token_ids
            if not isinstance(old, tuple) or not all(
                    isinstance(token, int) and token >= 0 for token in old):
                raise ValueError("rollback verifier returned invalid token IDs")
            if len(old) + 1 >= max_candidate_tokens:
                raise ValueError("repaired old suffix leaves no continuation budget")
            remaining = max_candidate_tokens - 1 - len(old)
            extension = (tuple(await branch.extend_from_verified(verified, remaining))
                         if remaining else ())
            if len(extension) > remaining:
                raise ValueError("new continuation exceeds candidate budget")
            return (verified.first_sampled_token,) + old + extension
        finally:
            await branch.close()


class PreSpecCoordinator:
    """One process-wide coordinator; call its methods on one asyncio event loop.

    Connection lifecycles are independent: a target only retires shadows owned
    by its own session. Concurrent sessions share the optional capacity limit
    and, in gated mode, the measured device-load admission checks.

    ``shadow_only=True`` is the safe default: formal decoding never receives
    candidate draft tokens. With ``False``, a ready candidate can be selected
    at chunk submit or deferred from commit receipt to finish submit. It is
    always passed to the existing authoritative verifier, never a final answer.
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
        if policy.max_rollback_draft_tokens <= 0 or policy.max_candidate_tokens <= 0:
            raise ValueError("draft and candidate budgets must be positive")
        if policy.max_active_shadows < 0:
            raise ValueError("max_active_shadows must be nonnegative (0 means no cap)")
        self.clock = clock
        self.shadow_only = shadow_only
        self.on_probe = on_probe
        self.probe_error_count = 0
        self._tasks: dict[WindowKey, asyncio.Task[Candidate]] = {}
        self._partials: dict[WindowKey, Candidate] = {}
        self._launch_info: dict[WindowKey, tuple[str, int, int]] = {}
        self._reserved: set[asyncio.Task[Candidate]] = set()
        self._task_sessions: dict[asyncio.Task[Candidate], str] = {}
        self._launched: set[WindowKey] = set()
        self._begun: set[SegmentKey] = set()
        self._begun_window_ids: dict[SegmentKey, int] = {}
        self._selected: dict[WindowKey, Candidate] = {}
        self._selected_at_ns: dict[WindowKey, int] = {}
        self._last_finished_window: dict[SegmentKey, int] = {}
        self._finished_segments: set[SegmentKey] = set()
        self._active_formals: dict[str, int] = {}
        self._active_chunks: dict[str, int] = {}
        self._active_window_keys: set[WindowKey] = set()
        self._logged_budget_skips: set[tuple[str, int, int, str]] = set()

    def should_log_budget_skip(self, session_id: str, audio_epoch: int,
                               window_id: int, reason: str) -> bool:
        key = (session_id, audio_epoch, window_id, reason)
        if key in self._logged_budget_skips:
            return False
        self._logged_budget_skips.add(key)
        return True

    def maybe_start(self, request: ShadowRequest, load: LoadSnapshot | None,
                    predicted_draft_ms: float) -> LaunchDecision:
        """Try one midpoint request; never put a shadow task in a waiting queue."""
        key = (request.session_id, request.audio_epoch, request.window_id)
        segment_key = key[:2]
        now = self.clock()
        if request.window_audio_samples < self.policy.midpoint_samples:
            return LaunchDecision(False, "before_midpoint")
        if len(request.rollback_tokens) > self.policy.max_rollback_draft_tokens:
            return LaunchDecision(False, "rollback_draft_exceeds_kmax")
        if len(request.rollback_tokens) + 2 > self.policy.max_candidate_tokens:
            return LaunchDecision(False, "candidate_budget_insufficient")
        if (key in self._launched or segment_key in self._begun
                or segment_key in self._finished_segments
                or request.window_id <= self._last_finished_window.get(
                    segment_key, -1)):
            return LaunchDecision(False, "epoch_already_used")
        if self._active_formals.get(request.session_id, 0):
            return LaunchDecision(False, "session_formal_inflight")
        if self._active_chunks.get(request.session_id, 0):
            return LaunchDecision(False, "session_chunk_inflight")
        if (self.policy.max_active_shadows > 0
                and len(self._reserved) >= self.policy.max_active_shadows):
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
        self._launch_info[key] = (request.request_id, request.base_prefix_version,
                                  request.visible_audio_end_samples)
        self._reserved.add(task)
        self._task_sessions[task] = request.session_id
        self._launched.add(key)
        task.add_done_callback(self._release)
        return LaunchDecision(True, "launched", slack_ms)

    def publish_partial(self, request: ShadowRequest,
                        cumulative_token_ids: Sequence[int]) -> bool:
        """Save one already-emitted CUMULATIVE snapshot without awaiting shadow work.

        The producer calls this after each usable RequestOutput. The snapshot
        belongs to the active request/task only; the formal path may later use
        it after canceling that task. A one-token snapshot has no formal draft.
        """
        key = (request.session_id, request.audio_epoch, request.window_id)
        if (self._tasks.get(key) is not asyncio.current_task()
                or self._launch_info.get(key) != (
                    request.request_id, request.base_prefix_version,
                    request.visible_audio_end_samples)):
            return False
        token_ids = tuple(cumulative_token_ids)
        if (len(token_ids) > self.policy.max_candidate_tokens
                or any(type(token) is not int or token < 0
                       for token in token_ids)):
            raise ValueError("invalid partial candidate token IDs")
        if len(token_ids) < 2 or len(token_ids) - 1 > self.policy.max_rollback_draft_tokens:
            return False
        previous = self._partials.get(key)
        if previous is not None and token_ids[:len(previous.token_ids)] != previous.token_ids:
            raise ValueError("CUMULATIVE candidate output changed an emitted token")
        if previous is not None and token_ids == previous.token_ids:
            return True
        self._partials[key] = Candidate(
            request.request_id, request.session_id, request.audio_epoch,
            request.window_id, request.base_prefix_version,
            request.visible_audio_end_samples, token_ids, self.clock(),
            is_partial=True, rollback_draft_length=len(request.rollback_tokens))
        return True

    async def _produce(self, request: ShadowRequest) -> Candidate:
        token_ids = tuple(await self.producer(request, self.policy.max_candidate_tokens))
        if not token_ids or len(token_ids) > self.policy.max_candidate_tokens:
            raise ValueError("producer must return 1..max_candidate_tokens token IDs")
        if not all(type(token) is int and token >= 0 for token in token_ids):
            raise ValueError("producer returned invalid token IDs")
        return Candidate(request.request_id, request.session_id, request.audio_epoch,
                         request.window_id, request.base_prefix_version,
                         request.visible_audio_end_samples, token_ids,
                         self.clock(), rollback_draft_length=len(request.rollback_tokens))

    def _release(self, task: asyncio.Task[Candidate]) -> None:
        self._reserved.discard(task)
        self._task_sessions.pop(task, None)
        if not task.cancelled():
            task.exception()  # Observe errors even if no formal boundary follows.

    def begin_authoritative_chunk(self, session_id: str, *,
                                  audio_epoch: int | None = None,
                                  window_id: int | None = None,
                                  prefix_version: int | None = None,
                                  full_audio_end_samples: int | None = None,
                                  allow_candidate: bool = True) -> Candidate | None:
        """At target chunk submit, take a ready draft or cancel shadow work."""
        if audio_epoch is not None and (window_id is None or
                                        prefix_version is None or
                                        full_audio_end_samples is None):
            raise ValueError("chunk candidate needs window, prefix version and audio end")
        key = ((session_id, audio_epoch, window_id)
               if audio_epoch is not None else None)
        if key is not None:
            if key in self._active_window_keys:
                raise ValueError(f"chunk window already active: {key}")
            self._active_window_keys.add(key)
        started_ns = self.clock()
        previous_active = self._active_chunks.get(session_id, 0)
        self._active_chunks[session_id] = previous_active + 1
        task = self._tasks.pop(key, None) if key is not None else None
        self._cancel_session_shadows(session_id, preserve_partial_key=key)
        if key is None:
            return None
        if key in self._selected:
            self._emit_probe(key, "chunk", started_ns, "already_selected")
            return None
        return self._select_candidate(task, key, "chunk", started_ns, prefix_version,
                                      full_audio_end_samples,
                                      allow_candidate and not previous_active)

    def _cancel_session_shadows(self, session_id: str, *,
                                preserve_ready_key: WindowKey | None = None,
                                preserve_partial_key: WindowKey | None = None) -> None:
        """Retire this connection's shadows without affecting other connections."""
        for key in list(self._tasks):
            if key[0] != session_id:
                continue
            task = self._tasks[key]
            if key == preserve_ready_key and task.done():
                continue
            self._tasks.pop(key)
            if task.done():
                self._reserved.discard(task)
            else:
                task.cancel()
        for key in list(self._partials):
            if key[0] == session_id and key != preserve_partial_key:
                self._partials.pop(key)
        for key in list(self._launch_info):
            if (key[0] == session_id and key != preserve_partial_key
                    and key != preserve_ready_key):
                self._launch_info.pop(key)

    def end_authoritative_chunk(self, session_id: str, *,
                                audio_epoch: int | None = None,
                                window_id: int | None = None,
                                target_continuation_tokens: Sequence[int] | None = None
                                ) -> Evaluation | None:
        """End a chunk; omit target tokens to discard without quality scoring."""
        if audio_epoch is not None and window_id is None:
            raise ValueError("chunk cleanup needs window id")
        key = ((session_id, audio_epoch, window_id)
               if audio_epoch is not None else None)
        if key is not None and key not in self._active_window_keys:
            raise ValueError(f"chunk window was not begun: {key}")
        active = self._active_chunks.get(session_id, 0)
        if active <= 0:
            raise ValueError(f"no authoritative chunk in flight: {session_id}")
        if active == 1:
            del self._active_chunks[session_id]
        else:
            self._active_chunks[session_id] = active - 1
        if key is not None:
            self._active_window_keys.remove(key)
            try:
                return self._evaluate(key, target_continuation_tokens)
            finally:
                self._discard_window(key)
        return None

    def _discard_window(self, key: WindowKey) -> None:
        task = self._tasks.pop(key, None)
        if task is not None:
            if task.done():
                self._reserved.discard(task)
            else:
                task.cancel()
        self._partials.pop(key, None)
        self._launch_info.pop(key, None)
        self._launched.discard(key)
        segment_key = key[:2]
        self._last_finished_window[segment_key] = max(
            key[2], self._last_finished_window.get(segment_key, -1))
        self._logged_budget_skips.discard((*key, "rollback_draft_exceeds_kmax"))
        self._logged_budget_skips.discard((*key, "candidate_budget_insufficient"))

    def _emit_probe(self, key: WindowKey,
                    boundary: Literal["chunk", "formal"], started_ns: int,
                    status: str, candidate: Candidate | None = None) -> None:
        if self.on_probe is None:
            return
        probe = CandidateProbe(
            key[0], key[1], key[2], boundary, started_ns, status,
            (started_ns - candidate.ready_ns) / 1e6 if candidate else None,
            candidate.request_id if candidate else None)
        try:
            self.on_probe(probe)
        except Exception:
            self.probe_error_count += 1  # Measurement must not fail target work.

    def _select_candidate(self, task: asyncio.Task[Candidate] | None,
                          key: WindowKey,
                          boundary: Literal["chunk", "formal"], started_ns: int,
                          prefix_version: int,
                          full_audio_end_samples: int,
                          allow_candidate: bool) -> Candidate | None:
        partial = self._partials.pop(key, None)
        if task is None:
            if partial is None:
                self._emit_probe(key, boundary, started_ns,
                                 "discarded_before_boundary" if key in self._launched
                                 else "not_started")
                return None
            candidate = partial
        elif not task.done():
            task.cancel()
            if partial is None:
                self._emit_probe(key, boundary, started_ns, "pending")
                return None
            candidate = partial
        else:
            self._reserved.discard(task)
            try:
                candidate = task.result()
            except asyncio.CancelledError:
                if partial is None:
                    self._emit_probe(key, boundary, started_ns, "failed")
                    return None
                candidate = partial
            except Exception:
                self._emit_probe(key, boundary, started_ns, "failed")
                return None
        if not allow_candidate:
            self._emit_probe(key, boundary, started_ns, "disallowed", candidate)
            return None
        if candidate.base_prefix_version != prefix_version:
            self._emit_probe(key, boundary, started_ns, "version_mismatch", candidate)
            return None
        if (candidate.audio_epoch != key[1]
                or candidate.window_id != key[2]
                or candidate.visible_audio_end_samples > full_audio_end_samples):
            self._emit_probe(key, boundary, started_ns, "audio_mismatch", candidate)
            return None
        if not candidate.formal_draft_token_ids:
            self._emit_probe(key, boundary, started_ns, "candidate_no_draft", candidate)
            return None
        if (candidate.is_partial and
                len(candidate.formal_draft_token_ids) <= candidate.rollback_draft_length):
            self._emit_probe(key, boundary, started_ns,
                             "partial_not_longer_than_rollback", candidate)
            return None
        if len(candidate.formal_draft_token_ids) > self.policy.max_rollback_draft_tokens:
            self._emit_probe(key, boundary, started_ns,
                             "candidate_exceeds_kmax", candidate)
            return None
        self._selected[key] = candidate
        self._selected_at_ns[key] = started_ns
        if candidate.is_partial:
            probe_status = ("partial_ready_shadow_only" if self.shadow_only
                            else "partial_ready_consumed")
        else:
            probe_status = "ready_shadow_only" if self.shadow_only else "ready_consumed"
        self._emit_probe(key, boundary, started_ns, probe_status, candidate)
        return None if self.shadow_only else candidate

    def begin_formal(self, session_id: str, audio_epoch: int,
                     prefix_version: int,
                     full_audio_end_samples: int, *,
                     window_id: int,
                     allow_candidate: bool = True,
                     defer_candidate: bool = False) -> Candidate | None:
        """Mark commit; optionally keep a ready candidate for finish submit."""
        segment_key = (session_id, audio_epoch)
        key = (session_id, audio_epoch, window_id)
        if segment_key in self._begun:
            raise ValueError(f"formal boundary already begun: {segment_key}")
        started_ns = self.clock()
        previous_active = self._active_formals.get(session_id, 0)
        self._active_formals[session_id] = previous_active + 1
        self._begun.add(segment_key)
        self._begun_window_ids[segment_key] = window_id
        if defer_candidate:
            self._cancel_session_shadows(session_id, preserve_ready_key=key,
                                         preserve_partial_key=key)
            return None
        task = self._tasks.pop(key, None)
        self._cancel_session_shadows(session_id, preserve_partial_key=key)
        if key in self._selected:
            self._emit_probe(key, "formal", started_ns, "already_selected")
            return None
        return self._select_candidate(task, key, "formal", started_ns, prefix_version,
                                      full_audio_end_samples,
                                      allow_candidate and not previous_active)

    def claim_formal_candidate(self, session_id: str, audio_epoch: int, *,
                               window_id: int, prefix_version: int,
                               full_audio_end_samples: int) -> Candidate | None:
        """At actual finish submit, take a ready candidate without waiting."""
        segment_key = (session_id, audio_epoch)
        if segment_key not in self._begun:
            raise ValueError(f"formal boundary was not begun: {segment_key}")
        key = (session_id, audio_epoch, window_id)
        started_ns = self.clock()
        if self._begun_window_ids[segment_key] != window_id:
            self._emit_probe(key, "formal", started_ns, "window_changed")
            return None
        task = self._tasks.pop(key, None)
        if key in self._selected:
            self._emit_probe(key, "formal", started_ns, "already_selected")
            return None
        return self._select_candidate(task, key, "formal", started_ns,
                                      prefix_version, full_audio_end_samples,
                                      self._active_formals[session_id] == 1)

    def finish_formal(self, session_id: str, audio_epoch: int,
                      target_continuation_tokens: Sequence[int] | None = None,
                      *, window_id: int
                      ) -> Evaluation | None:
        """End a formal boundary; optional tokens enable quality scoring."""
        segment_key = (session_id, audio_epoch)
        key = (session_id, audio_epoch, window_id)
        if segment_key not in self._begun:
            raise ValueError(f"formal boundary was not begun: {segment_key}")
        if self._begun_window_ids[segment_key] != window_id:
            raise ValueError(f"formal window mismatch: {segment_key}")
        self._begun.remove(segment_key)
        del self._begun_window_ids[segment_key]
        active = self._active_formals[session_id] - 1
        if active:
            self._active_formals[session_id] = active
        else:
            del self._active_formals[session_id]
        try:
            evaluation = self._evaluate(key, target_continuation_tokens)
            if evaluation is not None:
                logger.info(
                    "PRESPEC_MATCH ws_id=%s seg=%s window=%s boundary=formal "
                    "candidate_len=%s target_len=%s raw_matched_prefix_tokens=%s "
                    "draft_matched_prefix_tokens=%s candidate_partial=%s",
                    session_id, audio_epoch, window_id,
                    len(evaluation.candidate_tokens), evaluation.target_token_count,
                    evaluation.raw_matched_prefix_tokens,
                    evaluation.draft_matched_prefix_tokens,
                    evaluation.candidate_is_partial,
                )
            return evaluation
        finally:
            self._discard_window(key)
            self._finished_segments.add(segment_key)

    def _evaluate(self, key: WindowKey,
                  target_continuation_tokens: Sequence[int] | None
                  ) -> Evaluation | None:
        candidate = self._selected.pop(key, None)
        if candidate is None:
            return None
        ready_lead_ms = (self._selected_at_ns.pop(key) - candidate.ready_ns) / 1e6
        if target_continuation_tokens is None:
            return None
        def matched_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
            count = 0
            for candidate_token, target_token in zip(left, right):
                if candidate_token != target_token:
                    break
                count += 1
            return count

        raw_match = matched_prefix_len(candidate.token_ids,
                                       target_continuation_tokens)
        draft_match = matched_prefix_len(candidate.formal_draft_token_ids,
                                         target_continuation_tokens[1:])
        logger.info(
            "PRESPEC_MATCH_TOKENS ws_id=%s seg=%s window=%s mode=%s "
            "candidate_token_ids=%s target_token_ids=%s target_len=%s "
            "raw_matched_prefix_tokens=%s candidate_partial=%s",
            key[0], key[1], key[2],
            "shadow" if self.shadow_only else "consume",
            candidate.token_ids,
            tuple(target_continuation_tokens[:len(candidate.token_ids)]),
            len(target_continuation_tokens), raw_match, candidate.is_partial,
        )
        return Evaluation(
            key[0], key[1], key[2], candidate.token_ids,
            raw_matched_prefix_tokens=raw_match,
            draft_matched_prefix_tokens=draft_match,
            target_token_count=len(target_continuation_tokens),
            ready_before_formal=True,
            ready_lead_ms=ready_lead_ms,
            candidate_is_partial=candidate.is_partial,
        )

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
        self._partials = {key: candidate for key, candidate in self._partials.items()
                          if key[0] != session_id}
        self._launch_info = {key: info for key, info in self._launch_info.items()
                             if key[0] != session_id}
        self._launched = {key for key in self._launched if key[0] != session_id}
        self._begun = {key for key in self._begun if key[0] != session_id}
        self._selected = {key: candidate for key, candidate in self._selected.items()
                          if key[0] != session_id}
        self._selected_at_ns = {key: when for key, when in self._selected_at_ns.items()
                                if key[0] != session_id}
        self._last_finished_window = {
            key: window for key, window in self._last_finished_window.items()
            if key[0] != session_id
        }
        self._finished_segments = {
            key for key in self._finished_segments if key[0] != session_id
        }
        self._begun_window_ids = {
            key: window for key, window in self._begun_window_ids.items()
            if key[0] != session_id
        }
        self._active_formals.pop(session_id, None)
        self._active_chunks.pop(session_id, None)
        self._active_window_keys = {
            key for key in self._active_window_keys if key[0] != session_id
        }
        self._logged_budget_skips = {
            key for key in self._logged_budget_skips if key[0] != session_id
        }
