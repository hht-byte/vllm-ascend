# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real scheduler bodies without importing vLLM or NPU modules.

Run directly with Python, or with pytest --confcutdir=tests/ut/core. The
inherited admission helper is loaded from the separately delivered upstream
patch, so these tests also check the Ascend/upstream scheduler contract.
"""

import ast
import logging
import textwrap
import time
import unittest
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS


class Request:
    def __init__(self, request_id="request", *, extra=None, encoder=False):
        self.request_id = request_id
        self.sampling_params = NS(extra_args=extra)
        self.resumable = False
        self.streaming_queue = None
        self.spec_token_ids = []
        self.prompt_token_ids = [101, 102, 103, 104]
        self.output_token_ids = []
        self.num_computed_tokens = 0
        self.num_output_placeholders = 0
        self.has_encoder_inputs = encoder
        self.lora_request = None
        self.prefill_stats = None
        self.status = "waiting"

    @property
    def num_tokens(self):
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_tokens_with_spec(self):
        return self.num_tokens + len(self.spec_token_ids)


class RequestQueue:
    def __init__(self):
        self.items = deque()

    def __bool__(self):
        return bool(self.items)

    def peek_request(self):
        return self.items[0]

    def pop_request(self):
        return self.items.popleft()

    def prepend_request(self, request):
        self.items.appendleft(request)

    def prepend_requests(self, requests):
        self.items.extendleft(reversed(requests.items))


def load_scheduler():
    root = Path(__file__).resolve().parents[3]
    source = root / "vllm_ascend/core/recompute_scheduler.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RecomputeScheduler")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"add_request", "schedule"}]
    upstream = root / "examples/910b/external_draft_first_verify_vllm_v023.patch"
    helper_lines = []
    for line in upstream.read_text(encoding="utf-8").splitlines():
        if line.startswith("+    def _prepare_initial_external_draft") or helper_lines:
            if line.startswith("+"):
                helper_lines.append(line[1:])
            else:
                break
    helper = ast.parse(textwrap.dedent("\n".join(helper_lines))).body
    namespace = {
        "time": time,
        "deque": deque,
        "logger": logging.getLogger(__name__),
        "PLACEHOLDER_TOKEN_ID": -1,
        "PauseState": NS(UNPAUSED="unpaused", PAUSED_ALL="paused"),
        "RequestStatus": NS(WAITING="waiting", PREEMPTED="preempted", RUNNING="running"),
        "create_request_queue": lambda policy: RequestQueue(),
        "record_function_or_nullcontext": lambda name: nullcontext(),
        "NewRequestData": NS(from_request=lambda req, blocks: NS(req_id=req.request_id, block_ids=blocks)),
        "RecomputeSchedulerOutput": NS,
    }
    body = ast.Module(body=helper + methods, type_ignores=[])
    exec(compile("from __future__ import annotations\n" + ast.unparse(body), str(source), "exec"), namespace)
    return type(
        "SchedulerUnderTest",
        (),
        {name: namespace[name] for name in ("_prepare_initial_external_draft", "add_request", "schedule")},
    )


def make_scheduler(*, budget=32, threshold=0, cached_tokens=0, encoder_limit=None, mamba_limit=None, consumer=False):
    scheduler = load_scheduler()()
    blocks = NS(get_block_ids=lambda: ([0],))
    scheduler.allocations = []

    def allocate(request, num_tokens, **kwargs):
        scheduler.allocations.append((request.request_id, num_tokens, list(request.spec_token_ids)))
        return blocks

    scheduler.kv_cache_manager = NS(
        new_step_starts=lambda: None,
        get_computed_blocks=lambda request: (blocks, cached_tokens),
        empty_kv_cache_blocks=blocks,
        allocate_slots=allocate,
        get_blocks=lambda request_id: blocks,
        get_num_common_prefix_blocks=lambda request_id: [0],
    )
    scheduler.requests = {}
    scheduler.running = []
    scheduler.waiting = RequestQueue()
    scheduler.skipped_waiting = RequestQueue()
    scheduler._enqueue_waiting_request = lambda request: scheduler.waiting.items.append(request)
    scheduler._select_waiting_queue_for_scheduling = lambda: scheduler.waiting if scheduler.waiting else None
    scheduler._is_blocked_waiting_status = lambda status: False
    scheduler._make_cached_request_data = lambda *args: NS()
    scheduler._update_after_schedule = lambda output: None
    scheduler._try_schedule_encoder_inputs = lambda req, computed, tokens, budget, **kw: (
        [],
        min(tokens, encoder_limit) if encoder_limit is not None else tokens,
        budget,
        [],
    )
    scheduler._mamba_block_aligned_split = lambda request, tokens, *args: min(tokens, mamba_limit)
    scheduler.max_num_scheduled_tokens = budget
    scheduler.max_num_running_reqs = 8
    scheduler.max_num_encoder_input_tokens = 32
    scheduler.max_model_len = 128
    scheduler.num_spec_tokens = 3
    scheduler.num_lookahead_tokens = 3
    scheduler.scheduler_config = NS(long_prefill_token_threshold=threshold, enable_chunked_prefill=True)
    scheduler.scheduler_reserve_full_isl = False
    scheduler.kv_cache_config = NS(kv_cache_groups=[object()])
    scheduler.is_mtp_kv_consumer = consumer
    scheduler.is_kv_producer = False
    scheduler.is_encoder_decoder = False
    scheduler.use_eagle = False
    scheduler.has_mamba_layers = False
    scheduler.need_mamba_block_aligned_split = mamba_limit is not None
    scheduler.use_v2_model_runner = False
    scheduler.needs_kv_cache_zeroing = False
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.lora_config = None
    scheduler.log_stats = False
    scheduler._pause_state = "unpaused"
    scheduler.policy = "fcfs"
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = NS(get_freed_mm_hashes=lambda: [])
    return scheduler


def first_verify_request(request_id="request", *, encoder=False):
    return Request(
        request_id,
        extra={
            "ascend_external_draft": [10, 11, 12],
            "ascend_external_draft_first_verify": True,
        },
        encoder=encoder,
    )


class FirstVerifySchedulerTests(unittest.TestCase):
    def test_waiting_request_verifies_full_draft_on_first_execution(self):
        scheduler = make_scheduler()
        request = first_verify_request()
        scheduler.add_request(request)
        result = scheduler.schedule()
        self.assertEqual(result.num_scheduled_tokens, {"request": 7})
        self.assertEqual(result.scheduled_spec_decode_tokens, {"request": [10, 11, 12]})
        self.assertEqual(scheduler.allocations, [("request", 7, [10, 11, 12])])
        self.assertEqual(request.spec_token_ids, [])
        self.assertEqual(request.prompt_token_ids, [101, 102, 103, 104])
        self.assertEqual(request.output_token_ids, [])

    def test_cached_prompt_keeps_all_draft_tokens_in_per_request_metadata(self):
        scheduler = make_scheduler(cached_tokens=3)
        first = first_verify_request("first")
        second = first_verify_request("second")
        second.sampling_params.extra_args["ascend_external_draft"] = [20, 21]
        scheduler.add_request(first)
        scheduler.add_request(second)
        result = scheduler.schedule()
        self.assertEqual(result.num_scheduled_tokens, {"first": 4, "second": 3})
        self.assertEqual(result.scheduled_spec_decode_tokens, {"first": [10, 11, 12], "second": [20, 21]})
        self.assertEqual(first.output_token_ids, [])
        self.assertEqual(second.output_token_ids, [])

    def test_budget_or_prefill_chunk_falls_back_without_partial_draft(self):
        for options, expected in (({"budget": 5}, 4), ({"threshold": 5}, 4), ({"budget": 2}, 2)):
            with self.subTest(options=options):
                scheduler = make_scheduler(**options)
                request = first_verify_request()
                request.spec_token_ids = [10, 11, 12]
                scheduler.add_request(request)
                result = scheduler.schedule()
                self.assertEqual(result.num_scheduled_tokens, {"request": expected})
                self.assertEqual(result.scheduled_spec_decode_tokens, {})
                self.assertEqual(request.spec_token_ids, [])
                self.assertEqual(scheduler.allocations, [("request", expected, [])])

    def test_encoder_or_mamba_split_falls_back_before_kv_allocation(self):
        for options in ({"encoder_limit": 6}, {"mamba_limit": 6}):
            with self.subTest(options=options):
                scheduler = make_scheduler(**options)
                request = first_verify_request(encoder="encoder_limit" in options)
                request.spec_token_ids = [10, 11, 12]
                scheduler.add_request(request)
                result = scheduler.schedule()
                self.assertEqual(result.num_scheduled_tokens, {"request": 4})
                self.assertEqual(result.scheduled_spec_decode_tokens, {})
                self.assertEqual(scheduler.allocations, [("request", 4, [])])

    def test_unmarked_request_keeps_prompt_only_first_step(self):
        for extra in (
            None,
            {"ascend_external_draft": [10, 11, 12]},
            {
                "ascend_external_draft": [10, 11, 12],
                "ascend_external_draft_first_verify": False,
            },
        ):
            with self.subTest(extra=extra):
                scheduler = make_scheduler()
                request = Request(extra=extra)
                scheduler.add_request(request)
                result = scheduler.schedule()
                self.assertEqual(result.num_scheduled_tokens, {"request": 4})
                self.assertEqual(result.scheduled_spec_decode_tokens, {})
                self.assertEqual(request.spec_token_ids, [])

    def test_consumer_placeholders_do_not_replace_external_draft(self):
        scheduler = make_scheduler(consumer=True)
        request = first_verify_request()
        scheduler.add_request(request)
        self.assertEqual(request.spec_token_ids, [10, 11, 12])
        result = scheduler.schedule()
        self.assertEqual(result.scheduled_spec_decode_tokens, {"request": [10, 11, 12]})

    def test_unmarked_request_works_without_upstream_first_verify_helper(self):
        for consumer in (False, True):
            with self.subTest(consumer=consumer):
                scheduler = make_scheduler(consumer=consumer)
                delattr(type(scheduler), "_prepare_initial_external_draft")
                request = Request()
                try:
                    scheduler.add_request(request)
                except AttributeError as error:
                    self.fail(f"ordinary request unexpectedly required the upstream first-verify helper: {error}")
                result = scheduler.schedule()
                self.assertEqual(result.num_scheduled_tokens, {"request": 7 if consumer else 4})
                self.assertEqual(result.scheduled_spec_decode_tokens, {"request": [-1, -1, -1]} if consumer else {})

    def test_first_verify_without_upstream_helper_reports_required_patch(self):
        scheduler = make_scheduler()
        delattr(type(scheduler), "_prepare_initial_external_draft")
        with self.assertRaisesRegex(RuntimeError, "external_draft_first_verify_vllm_v023.patch"):
            scheduler.add_request(first_verify_request())
        self.assertEqual(scheduler.requests, {})
        self.assertFalse(scheduler.waiting)

    def test_unmarked_consumer_keeps_existing_placeholder_path(self):
        scheduler = make_scheduler(consumer=True)
        request = Request()
        scheduler.add_request(request)
        self.assertEqual(request.spec_token_ids, [-1, -1, -1])
        result = scheduler.schedule()
        self.assertEqual(result.num_scheduled_tokens, {"request": 7})
        self.assertEqual(result.scheduled_spec_decode_tokens, {"request": [-1, -1, -1]})

    def test_unmarked_consumer_still_replaces_preexisting_spec_tokens(self):
        scheduler = make_scheduler(consumer=True)
        request = Request(extra={"ascend_external_draft": [10, 11, 12]})
        request.spec_token_ids = [20, 21]
        scheduler.add_request(request)
        self.assertEqual(request.spec_token_ids, [-1, -1, -1])

    def test_unmarked_consumer_preserves_existing_cached_prefix_metadata(self):
        scheduler = make_scheduler(consumer=True, cached_tokens=3)
        scheduler.add_request(Request())
        result = scheduler.schedule()
        self.assertEqual(result.num_scheduled_tokens, {"request": 4})
        self.assertEqual(result.scheduled_spec_decode_tokens, {})

    def test_invalid_draft_is_rejected_before_request_admission(self):
        for draft in ([], [-1], [True], [1.5], "12", [1, 2, 3, 4]):
            with self.subTest(draft=draft):
                scheduler = make_scheduler()
                request = first_verify_request()
                request.sampling_params.extra_args["ascend_external_draft"] = draft
                with self.assertRaises(ValueError):
                    scheduler.add_request(request)
                self.assertEqual(scheduler.requests, {})
                self.assertFalse(scheduler.waiting)

    def test_invalid_first_verify_marker_is_rejected_before_request_mutation(self):
        for marker in ("false", 1, None):
            with self.subTest(marker=marker):
                scheduler = make_scheduler()
                request = first_verify_request()
                request.sampling_params.extra_args["ascend_external_draft_first_verify"] = marker
                request.resumable = True
                request.spec_token_ids = [88]
                with self.assertRaises(TypeError):
                    scheduler.add_request(request)
                self.assertEqual(request.spec_token_ids, [88])
                self.assertIsNone(request.streaming_queue)
                self.assertEqual(scheduler.requests, {})
                self.assertFalse(scheduler.waiting)

    def test_inherited_helper_rejects_nonboolean_marker_without_mutation(self):
        for marker in ("false", 1, None):
            with self.subTest(marker=marker):
                scheduler = make_scheduler()
                request = first_verify_request()
                request.sampling_params.extra_args["ascend_external_draft_first_verify"] = marker
                request.spec_token_ids = [88]
                with self.assertRaises(TypeError):
                    scheduler._prepare_initial_external_draft(request)
                self.assertEqual(request.spec_token_ids, [88])

    def test_inherited_helper_uses_kmax_without_speculative_config_attribute(self):
        scheduler = make_scheduler()
        request = first_verify_request()
        scheduler._prepare_initial_external_draft(request)
        self.assertEqual(request.spec_token_ids, [10, 11, 12])
        scheduler.num_spec_tokens = 0
        with self.assertRaises(ValueError):
            scheduler._prepare_initial_external_draft(request)
        scheduler.num_spec_tokens = 3
        scheduler.max_model_len = 7
        with self.assertRaises(ValueError):
            scheduler._prepare_initial_external_draft(request)


if __name__ == "__main__":
    unittest.main()
