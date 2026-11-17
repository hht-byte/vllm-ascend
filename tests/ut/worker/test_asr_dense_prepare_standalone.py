# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the real runner branches while enforcing dense-ASR dependencies."""

import ast
import textwrap
import unittest
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"


def method(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "NPUModelRunner")
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)
    env = {"np": np}
    exec("from __future__ import annotations\n" + ast.unparse(fn), env)
    return env[name]


def region(begin, end):
    source = SOURCE.read_text(encoding="utf-8")
    offset = source.index(begin, source.index("    def _prepare_inputs("))
    return compile(textwrap.dedent(source[offset : source.index(end, offset)]), "<actual preparation branch>", "exec")


class AcceptedCountsTests(unittest.TestCase):
    def runner(self, helper):
        r = NS(
            _input_prepare_graph=helper,
            use_async_scheduling=False,
            input_batch=NS(num_accepted_tokens_cpu=np.array([3, 2, 7, 8], dtype=np.int32)),
            num_accepted_tokens_event=NS(synchronize=Mock()),
            num_accepted_tokens=NS(np=np.full(4, 99, dtype=np.int32), gpu=torch.zeros(4, dtype=torch.int32)),
        )
        r._sync_num_accepted_tokens = MethodType(method("_sync_num_accepted_tokens"), r)
        r.num_accepted_tokens.copy_to_gpu = Mock(
            side_effect=lambda: r.num_accepted_tokens.gpu.copy_(torch.from_numpy(r.num_accepted_tokens.np))
        )
        return r

    def run_branch(self, r, bucket, mapping=None):
        code = region("        # Sync num_accepted_tokens from CPU", "        # Update num_computed_tokens on GPU.")
        exec(code, dict(self=r, num_reqs=2, prev_req_id_to_index=mapping or {}, input_prepare_bucket=bucket))

    def test_dense_owner_skips_unrecorded_event_but_keeps_cpu_counts_and_padding(self):
        for bucket in (8, None):
            with self.subTest(bucket=bucket):
                r = self.runner(object())
                self.run_branch(r, bucket)
                self.assertEqual(r.num_accepted_tokens_event.synchronize.call_count, 0)
                self.assertEqual(r.num_accepted_tokens.np.tolist(), [3, 2, 1, 1])
                self.assertEqual(r.num_accepted_tokens.copy_to_gpu.call_count, int(bucket is None))

    def test_legacy_wait_finishes_cpu_producer_before_copying_counts(self):
        r = self.runner(None)
        r.num_accepted_tokens_event.synchronize.side_effect = lambda: r.input_batch.num_accepted_tokens_cpu.__setitem__(
            slice(0, 2), [5, 4]
        )
        self.run_branch(r, None)
        self.assertEqual(r.num_accepted_tokens_event.synchronize.call_count, 1)
        self.assertEqual(r.num_accepted_tokens.gpu.tolist(), [5, 4, 1, 1])

    def test_legacy_async_reordering_and_new_request_counts_remain_intact(self):
        r = self.runner(None)
        r.use_async_scheduling = True
        r.prev_positions = NS(np=np.array([-1, 2], dtype=np.int32))
        r.num_accepted_tokens.np[:] = [3, 2, 6, 9]
        r._remap_num_accepted_tokens = MethodType(method("_remap_num_accepted_tokens"), r)
        self.run_branch(r, None, {"old": 2})
        self.assertEqual(r.num_accepted_tokens_event.synchronize.call_count, 1)
        self.assertEqual(r.input_batch.num_accepted_tokens_cpu[:2].tolist(), [1, 6])
        self.assertEqual(r.num_accepted_tokens.gpu.tolist(), [1, 6, 1, 1])

    def test_no_event_keeps_original_default_count_behavior(self):
        r = self.runner(object())
        r.num_accepted_tokens_event = None
        self.run_branch(r, None)
        self.assertEqual(r.num_accepted_tokens.gpu.tolist(), [1, 1, 1, 1])


class DecodeDraftMetadataTests(unittest.TestCase):
    def run_branch(self, helper, *, kv_consumer):
        code = region(
            "            # Get the number of draft tokens for each request.",
            "        if self._input_prepare_graph is not None and self._input_prepare_graph.staged is None:",
        )
        observed = []

        def calculate(drafts, ends, num_pcp_pads):
            observed.append((drafts.copy(), ends.copy(), num_pcp_pads))
            return NS(logits_indices="actual-logits")

        r = NS(
            _input_prepare_graph=helper,
            input_batch=NS(
                req_id_to_index={"prefill": 0, "decode": 1, "new-kv": 2},
                num_computed_tokens_cpu=np.array([0, 10, 0]),
                num_prompt_tokens=np.array([10, 10, 10]),
            ),
            is_kv_consumer=kv_consumer,
            pcp_size=1,
            _calc_spec_decode_metadata=calculate,
            num_decode_draft_tokens=NS(np=np.full(5, -77, dtype=np.int32), copy_to_gpu=Mock()),
        )
        schedule = NS(
            scheduled_new_reqs=[NS(req_id="new-kv")],
            scheduled_spec_decode_tokens={"prefill": [11, 12], "decode": [21, 22, 23], "new-kv": [31]},
        )
        env = dict(self=r, scheduler_output=schedule, num_reqs=3, cu_num_tokens=np.array([3, 7, 9]), np=np)
        exec(code, env)
        self.assertEqual(observed[0][0].tolist(), [2, 3, 1])
        self.assertEqual(observed[0][1].tolist(), [3, 7, 9])
        self.assertIsNone(observed[0][2])
        self.assertEqual(env["num_sampled_tokens"].tolist(), [3, 4, 2])
        self.assertEqual(env["logits_indices"], "actual-logits")
        return r

    def test_dense_owner_omits_unused_hybrid_cpu_array_and_gpu_upload(self):
        r = self.run_branch(object(), kv_consumer=True)
        self.assertEqual(r.num_decode_draft_tokens.copy_to_gpu.call_count, 0)
        self.assertEqual(r.num_decode_draft_tokens.np.tolist(), [-77] * 5)

    def test_legacy_gdn_mixed_prefill_decode_and_new_kv_flags_are_preserved(self):
        for consumer in (False, True):
            with self.subTest(kv_consumer=consumer):
                r = self.run_branch(None, kv_consumer=consumer)
                self.assertEqual(r.num_decode_draft_tokens.copy_to_gpu.call_count, 1)
                self.assertEqual(r.num_decode_draft_tokens.np.tolist(), [-1, 3, 1 if consumer else -1, -1, -1])


if __name__ == "__main__":
    unittest.main()
