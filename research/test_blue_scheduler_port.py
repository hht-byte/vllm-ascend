"""Execute patched frozen schedulers without importing vLLM or device code.

Set BLUE_ENGINE_SNAPSHOT to a directory containing the unmodified vllm/ and
vllm_ascend/ trees. The local blue-task snapshot is discovered when present.
"""

from __future__ import annotations

import ast
from collections import deque
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import time
import unittest

from blue_scheduler_port import (
    base_scheduler, make_scheduler_patch, recompute_scheduler,
)


SNAPSHOT = Path(os.environ.get(
    "BLUE_ENGINE_SNAPSHOT",
    str(Path(__file__).resolve().parents[3]
        / "asr-mtp-e2e-20261116" / "engine-snapshot"),
))
BASE = SNAPSHOT / "vllm/v1/core/sched/scheduler.py"
RECOMPUTE = SNAPSHOT / "vllm_ascend/core/recompute_scheduler.py"


class Queue:
    def __init__(self):
        self.items = deque()

    def __bool__(self):
        return bool(self.items)

    def __len__(self):
        return len(self.items)

    def add_request(self, request):
        self.items.append(request)

    def peek_request(self):
        return self.items[0]

    def pop_request(self):
        return self.items.popleft()

    def prepend_request(self, request):
        self.items.appendleft(request)

    def prepend_requests(self, requests):
        self.items.extendleft(reversed(requests.items))


class Request:
    def __init__(self, *, tokens=6, prompt_tokens=None, draft=(20, 21),
                 first_verify=True, status="waiting", computed=0):
        self.request_id = "contract-request"
        self.num_tokens = tokens
        self.num_prompt_tokens = tokens if prompt_tokens is None else prompt_tokens
        self.num_computed_tokens = computed
        self.num_output_placeholders = 0
        self.max_tokens = 32
        self.next_decode_eligible_step = 0
        self.spec_token_ids = []
        self.sampling_params = SimpleNamespace(extra_args={
            "ascend_external_draft": list(draft),
            "ascend_external_draft_first_verify": first_verify,
        })
        self.status = status
        self.resumable = False
        self.streaming_queue = None
        self.has_encoder_inputs = False
        self.mm_features = []
        self.lora_request = None
        self.prefill_stats = None
        self.num_preemptions = 0
        self.use_structured_output = False
        self._all_token_ids = list(range(tokens))

    @property
    def num_tokens_with_spec(self):
        return self.num_tokens + len(self.spec_token_ids)


class Blocks:
    def get_block_ids(self):
        return ([1],)


class Cache:
    def __init__(self, *, cached=0, allocate=True):
        self.cached = cached
        self.allocate = allocate
        self.allocations = []
        self.empty_kv_cache_blocks = Blocks()

    def new_step_starts(self):
        pass

    def get_computed_blocks(self, request):
        return Blocks(), self.cached

    def allocate_slots(self, request, num_tokens, **kwargs):
        self.allocations.append((num_tokens, kwargs))
        return Blocks() if self.allocate else None

    def get_blocks(self, request_id):
        return Blocks()

    def get_num_common_prefix_blocks(self, request_id):
        return [0]


class Output(SimpleNamespace):
    has_structured_output_requests = False


def scheduler_class(source, name="Scheduler", parent=None):
    tree = ast.parse(source)
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in {"schedule", "add_request",
                                 "_prepare_initial_external_draft",
                                 "_update_after_schedule"}]
    for method in methods:
        method.decorator_list = []
    cls.bases = [] if parent is None else [ast.Name(id="Scheduler", ctx=ast.Load())]
    cls.body = methods
    status = SimpleNamespace(WAITING="waiting", PREEMPTED="preempted",
                             RUNNING="running", WAITING_FOR_REMOTE_KVS="remote",
                             WAITING_FOR_STREAMING_REQ="streaming")
    namespace = {
        "Request": Request, "RequestStatus": status, "deque": deque, "time": time,
        "PauseState": SimpleNamespace(UNPAUSED="unpaused", PAUSED_ALL="paused"),
        "SchedulingPolicy": SimpleNamespace(PRIORITY="priority"),
        "create_request_queue": lambda policy: Queue(),
        "record_function_or_nullcontext": lambda *args: nullcontext(),
        "NewRequestData": SimpleNamespace(from_request=lambda req, *args:
                                           SimpleNamespace(req_id=req.request_id,
                                                           computed=req.num_computed_tokens)),
        "SchedulerOutput": Output,
        "RecomputeSchedulerOutput": Output,
        "Scheduler": parent,
        "PLACEHOLDER_TOKEN_ID": -1,
        "StreamingUpdate": SimpleNamespace(from_request=lambda request:
            SimpleNamespace(sampling_params=request.sampling_params)),
        "logger": SimpleNamespace(info=lambda *args: None, debug=lambda *args: None),
    }
    module = ast.Module(body=[ast.ImportFrom(module="__future__", level=0,
                        names=[ast.alias(name="annotations")]), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "frozen_scheduler.py", "exec"),
         namespace)
    return namespace[name]


def make_scheduler(cls, *, budget=20, threshold=0, cached=0, allocate=True,
                   async_scheduling=False):
    scheduler = cls.__new__(cls)
    scheduler.scheduler_config = SimpleNamespace(
        long_prefill_token_threshold=threshold, enable_chunked_prefill=True,
        async_scheduling=async_scheduling)
    scheduler.num_spec_tokens = 4
    scheduler.max_model_len = 64
    scheduler.requests = {}
    scheduler.running = []
    scheduler.waiting = Queue()
    scheduler.skipped_waiting = Queue()
    scheduler.max_num_scheduled_tokens = budget
    scheduler.max_num_encoder_input_tokens = 64
    scheduler.max_num_running_reqs = 4
    scheduler.current_step = 0
    scheduler._pause_state = "unpaused"
    scheduler.kv_cache_manager = Cache(cached=cached, allocate=allocate)
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.lora_config = None
    scheduler.log_stats = False
    scheduler.policy = "fcfs"
    scheduler.need_mamba_block_aligned_split = False
    scheduler.num_lookahead_tokens = 4
    scheduler.use_eagle = False
    scheduler.is_encoder_decoder = False
    scheduler.scheduler_reserve_full_isl = False
    scheduler.connector_prefix_cache_stats = None
    scheduler._inflight_prefills = set()
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    scheduler.use_v2_model_runner = False
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler.needs_kv_cache_zeroing = False
    scheduler.finished_req_ids = set()
    scheduler.enable_return_routed_experts = False
    scheduler.is_mtp_kv_consumer = False
    scheduler.is_kv_producer = False
    scheduler.has_mamba_layers = False
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=lambda: [], free=lambda request: None,
        allocate=lambda request, i: None)
    scheduler._is_blocked_waiting_status = lambda status: False
    scheduler._select_waiting_queue_for_scheduling = lambda: (
        scheduler.waiting if scheduler.waiting else scheduler.skipped_waiting)
    scheduler._enqueue_waiting_request = scheduler.waiting.add_request
    scheduler._make_cached_request_data = lambda *args: None
    scheduler._inflight_prefill_reserved_blocks = lambda: 0
    scheduler._build_kv_connector_meta = lambda *args: None
    return scheduler


@unittest.skipUnless(BASE.exists(), "frozen blue engine snapshot unavailable")
class BaseSchedulerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = BASE.read_text(encoding="utf-8")
        cls.scheduler = scheduler_class(base_scheduler(cls.source))

    def test_remote_kv_receive_retains_first_draft_for_first_execution(self):
        scheduler = make_scheduler(self.scheduler)
        scheduler.connector = SimpleNamespace(
            on_new_request=lambda request: None,
            get_num_new_matched_tokens=lambda request, cached: (4, True),
            update_state_after_alloc=lambda *args: None)
        request = Request()
        scheduler.add_request(request)
        output = scheduler.schedule()
        self.assertEqual(output.num_scheduled_tokens, {})
        self.assertEqual(output.scheduled_spec_decode_tokens, {})
        self.assertEqual(request.status, "remote")
        self.assertEqual(request.spec_token_ids, [20, 21])
        self.assertEqual(scheduler.kv_cache_manager.allocations[0][0], 0)

    def test_async_first_verify_rejected_before_enqueue(self):
        scheduler = make_scheduler(self.scheduler, async_scheduling=True)
        with self.assertRaisesRegex(ValueError, "async"):
            scheduler.add_request(Request())
        self.assertEqual(scheduler.requests, {})
        self.assertFalse(scheduler.waiting)

    def test_new_native_resumable_first_verify_rejected_before_any_mutation(self):
        scheduler = make_scheduler(self.scheduler)
        request = Request()
        request.resumable = True
        with self.assertRaisesRegex(ValueError, "resumable"):
            scheduler.add_request(request)
        self.assertEqual(scheduler.requests, {})
        self.assertFalse(scheduler.waiting)
        self.assertIsNone(request.streaming_queue)
        self.assertEqual(request.spec_token_ids, [])

    def test_existing_native_session_first_verify_update_rejected_before_mutation(self):
        for status in ("waiting", "streaming"):
            with self.subTest(status=status):
                scheduler = make_scheduler(self.scheduler)
                session = Request(first_verify=False)
                session.resumable = True
                scheduler.add_request(session)
                session.status = status
                original_params = session.sampling_params
                update = Request()
                session_updates = []
                scheduler._update_request_as_session = lambda *args: session_updates.append(args)
                with self.assertRaisesRegex(ValueError, "resumable"):
                    scheduler.add_request(update)
                self.assertIs(scheduler.requests[session.request_id], session)
                self.assertIs(session.sampling_params, original_params)
                self.assertEqual(list(session.streaming_queue), [])
                self.assertEqual(session_updates, [])
                self.assertEqual(update.spec_token_ids, [])

    def test_full_draft_is_present_in_first_scheduler_output(self):
        scheduler = make_scheduler(self.scheduler)
        request = Request()
        scheduler.add_request(request)
        output = scheduler.schedule()
        self.assertEqual(output.num_scheduled_tokens, {"contract-request": 8})
        self.assertEqual(output.scheduled_spec_decode_tokens,
                         {"contract-request": [20, 21]})
        self.assertEqual(output.total_num_scheduled_tokens, 8)
        self.assertEqual(output.scheduled_new_reqs[0].computed, 0)
        self.assertEqual(request.num_computed_tokens, 8)
        self.assertEqual(request.spec_token_ids, [])

    def test_cached_prefix_is_counted_before_the_full_draft(self):
        scheduler = make_scheduler(self.scheduler, budget=4, cached=4)
        request = Request()
        scheduler.add_request(request)
        output = scheduler.schedule()
        self.assertEqual(output.num_scheduled_tokens, {"contract-request": 4})
        self.assertEqual(output.scheduled_spec_decode_tokens,
                         {"contract-request": [20, 21]})
        self.assertEqual(output.scheduled_new_reqs[0].computed, 4)
        self.assertEqual(request.num_computed_tokens, 8)

    def test_resumed_request_schedules_committed_output_as_well_as_prompt(self):
        scheduler = make_scheduler(self.scheduler, budget=8)
        request = Request(tokens=8, prompt_tokens=3, status="preempted", computed=2)
        scheduler.add_request(request)
        output = scheduler.schedule()
        self.assertEqual(output.num_scheduled_tokens, {"contract-request": 8})
        self.assertEqual(output.scheduled_spec_decode_tokens,
                         {"contract-request": [20, 21]})
        self.assertEqual(output.scheduled_new_reqs, [])
        self.assertEqual(request.num_computed_tokens, 10)

    def test_budget_or_threshold_fallback_never_verifies_partial_initial_draft(self):
        for budget, threshold, expected_tokens in ((7, 0, 6), (20, 7, 6), (4, 0, 4),
                                                   (20, 4, 4)):
            with self.subTest(budget=budget, threshold=threshold):
                scheduler = make_scheduler(self.scheduler, budget=budget,
                                           threshold=threshold)
                request = Request()
                scheduler.add_request(request)
                output = scheduler.schedule()
                self.assertEqual(output.num_scheduled_tokens,
                                 {"contract-request": expected_tokens})
                self.assertEqual(output.scheduled_spec_decode_tokens, {})
                self.assertEqual(request.spec_token_ids, [])

    def test_no_chunk_prefill_keeps_upstream_admission_when_prompt_does_not_fit(self):
        scheduler = make_scheduler(self.scheduler, budget=4)
        scheduler.scheduler_config.enable_chunked_prefill = False
        request = Request()
        scheduler.add_request(request)
        output = scheduler.schedule()
        self.assertEqual(output.num_scheduled_tokens, {})
        self.assertEqual(output.scheduled_spec_decode_tokens, {})
        self.assertEqual(scheduler.kv_cache_manager.allocations, [])
        self.assertTrue(scheduler.waiting)

    def test_encoder_or_alignment_shrink_cannot_schedule_partial_initial_draft(self):
        for source in ("encoder", "mamba"):
            for reduced in (4, 7):
                with self.subTest(source=source, reduced=reduced):
                    scheduler = make_scheduler(self.scheduler)
                    request = Request()
                    if source == "encoder":
                        request.has_encoder_inputs = True
                        scheduler._try_schedule_encoder_inputs = lambda *args, **kwargs: (
                            [], reduced, 64, [])
                    else:
                        scheduler.need_mamba_block_aligned_split = True
                        scheduler._mamba_block_aligned_split = lambda *args: reduced
                    scheduler.add_request(request)
                    output = scheduler.schedule()
                    self.assertEqual(output.num_scheduled_tokens,
                                     {"contract-request": min(reduced, 6)})
                    self.assertEqual(output.scheduled_spec_decode_tokens, {})
                    self.assertEqual(request.spec_token_ids, [])

    def test_allocation_failure_preserves_draft_and_does_not_execute(self):
        scheduler = make_scheduler(self.scheduler, allocate=False)
        request = Request()
        scheduler.add_request(request)
        output = scheduler.schedule()
        self.assertEqual(output.num_scheduled_tokens, {})
        self.assertEqual(output.scheduled_spec_decode_tokens, {})
        self.assertEqual(request.spec_token_ids, [20, 21])
        self.assertEqual(request.num_computed_tokens, 0)
        self.assertTrue(scheduler.waiting)
        scheduler.kv_cache_manager.allocate = True
        retry = scheduler.schedule()
        self.assertEqual(retry.scheduled_spec_decode_tokens,
                         {"contract-request": [20, 21]})

    def test_unmarked_request_keeps_original_first_prefill_even_with_async(self):
        scheduler = make_scheduler(self.scheduler, async_scheduling=True)
        request = Request(first_verify=False)
        scheduler.add_request(request)
        output = scheduler.schedule()
        self.assertEqual(output.num_scheduled_tokens, {"contract-request": 6})
        self.assertEqual(output.scheduled_spec_decode_tokens, {})

    def test_validation_rejects_invalid_drafts_before_enqueue(self):
        for draft, max_context in (((), 64), ((True,), 64), ((-1,), 64),
                                   ((1, 2, 3, 4, 5), 64), ((20, 21), 8)):
            with self.subTest(draft=draft, max_context=max_context):
                scheduler = make_scheduler(self.scheduler)
                scheduler.max_model_len = max_context
                with self.assertRaises(ValueError):
                    scheduler.add_request(Request(draft=draft))
                self.assertEqual(scheduler.requests, {})
                self.assertFalse(scheduler.waiting)

    def test_patch_applies_and_parses_with_base_scheduler_only(self):
        patch = make_scheduler_patch(SNAPSHOT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "vllm/v1/core/sched/scheduler.py"
            target.parent.mkdir(parents=True)
            target.write_text(self.source, encoding="utf-8", newline="\n")
            patch_file = root / "scheduler.patch"
            patch_file.write_text(patch, encoding="utf-8", newline="\n")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True,
                           capture_output=True)
            result = subprocess.run(["git", "apply", "--check", str(patch_file)],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run(["git", "apply", str(patch_file)], cwd=root, check=True,
                           capture_output=True)
            ast.parse(target.read_text(encoding="utf-8"))

    def test_unsupported_or_already_patched_source_fails_closed(self):
        for source in ("class Scheduler: pass\n", base_scheduler(self.source)):
            with self.subTest(source_length=len(source)):
                with self.assertRaises(ValueError):
                    base_scheduler(source)


@unittest.skipUnless(BASE.exists() and RECOMPUTE.exists(),
                     "frozen blue recompute scheduler unavailable")
class RecomputeSchedulerContracts(BaseSchedulerContracts):
    @classmethod
    def setUpClass(cls):
        cls.source = RECOMPUTE.read_text(encoding="utf-8")
        try:
            transformed = recompute_scheduler(cls.source)
        except ValueError as error:
            raise AssertionError(f"frozen recompute source must port: {error}") from error
        parent = scheduler_class(base_scheduler(BASE.read_text(encoding="utf-8")))
        cls.scheduler = scheduler_class(transformed, "RecomputeScheduler", parent)

    def test_patch_applies_and_parses_with_base_scheduler_only(self):
        self.skipTest("base-only generator case belongs to the base contracts")

    def test_unsupported_or_already_patched_source_fails_closed(self):
        for source in ("class RecomputeScheduler: pass\n",
                       recompute_scheduler(self.source)):
            with self.subTest(source_length=len(source)):
                with self.assertRaises(ValueError):
                    recompute_scheduler(source)

    def test_mtp_placeholders_do_not_overwrite_explicit_initial_draft(self):
        scheduler = make_scheduler(self.scheduler)
        scheduler.is_mtp_kv_consumer = True
        request = Request()
        scheduler.add_request(request)
        self.assertEqual(request.spec_token_ids, [20, 21])
        output = scheduler.schedule()
        self.assertEqual(output.scheduled_spec_decode_tokens,
                         {"contract-request": [20, 21]})

    def test_unmarked_mtp_consumer_keeps_original_placeholder_path(self):
        scheduler = make_scheduler(self.scheduler)
        scheduler.is_mtp_kv_consumer = True
        request = Request(first_verify=False)
        scheduler.add_request(request)
        self.assertEqual(request.spec_token_ids, [-1, -1, -1, -1])
        output = scheduler.schedule()
        self.assertEqual(output.scheduled_spec_decode_tokens,
                         {"contract-request": [-1, -1, -1, -1]})

    def test_generated_pair_applies_and_both_files_parse(self):
        patch = make_scheduler_patch(SNAPSHOT, SNAPSHOT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets = []
            for relative in ("vllm/v1/core/sched/scheduler.py",
                             "vllm_ascend/core/recompute_scheduler.py"):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text((SNAPSHOT / relative).read_text(encoding="utf-8"),
                                  encoding="utf-8", newline="\n")
                targets.append(target)
            patch_file = root / "scheduler-pair.patch"
            patch_file.write_text(patch, encoding="utf-8", newline="\n")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True,
                           capture_output=True)
            result = subprocess.run(["git", "apply", "--check", str(patch_file)],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run(["git", "apply", str(patch_file)], cwd=root, check=True,
                           capture_output=True)
            for target in targets:
                ast.parse(target.read_text(encoding="utf-8"))

    def test_explicit_recompute_root_does_not_silently_omit_missing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                make_scheduler_patch(SNAPSHOT, Path(temporary))


if __name__ == "__main__":
    unittest.main()
