# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU equivalence and optional device tests for synchronous verifier inputs."""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_ascend/worker/asr_spec_decode_metadata.py"
RUNNER = ROOT / "vllm_ascend/worker/model_runner_v1.py"


def load():
    assert SOURCE.exists(), "synchronous verifier metadata packer is not implemented"
    spec = importlib.util.spec_from_file_location("asr_spec_decode_metadata", SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Event:
    def __init__(self):
        self.done = True
        self.records = self.waits = 0

    def query(self):
        return self.done

    def synchronize(self):
        self.waits += 1
        self.done = True

    def record(self):
        self.records += 1
        self.done = False


def fixture(device="cpu"):
    return NS(
        seq_lens=torch.zeros(8, dtype=torch.int32),
        max_num_tokens=256,
        num_spec_tokens=5,
        pin_memory=device != "cpu",
        device=device,
    )


def oracle(lengths, drafts, ids):
    # Per-request definition, independent of the packed implementation.
    cu_draft, cu_sample, logits, target, bonus, tokens = [], [], [], [], [], []
    end = sampled = drafted = 0
    for length, count in zip(lengths, drafts):
        end += length
        start = end - count - 1
        logits.extend(range(start, end))
        target.extend(range(sampled, sampled + count))
        bonus.append(sampled + count)
        tokens.extend(ids[start + 1 : end])
        sampled += count + 1
        drafted += count
        cu_draft.append(drafted)
        cu_sample.append(sampled)
    values = dict(
        cu_num_draft_tokens=cu_draft,
        cu_num_sampled_tokens=cu_sample,
        logits_indices=logits,
        target_logits_indices=target,
        bonus_logits_indices=bonus,
        draft_token_ids=tokens,
    )
    dtypes = dict.fromkeys(values, torch.int32)
    dtypes.update(logits_indices=torch.int64, target_logits_indices=torch.int64)
    return {key: torch.tensor(value, dtype=dtypes[key]) for key, value in values.items()}


def calc_method():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "NPUModelRunner")
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "_calc_spec_decode_metadata")
    env = dict(np=np, torch=torch, SpecDecodeMetadata=lambda **kw: NS(**kw))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "<actual runner>", "exec"), env)
    return env[fn.name]


def cumsum(r, counts, scratch, cumsum_dtype=None):
    ends = np.cumsum(counts, dtype=cumsum_dtype)
    scratch[: ends[-1]] = r.arange_np[: ends[-1]] - np.repeat(ends - counts, counts)
    return ends


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.mod = load()
        self.r = fixture()
        self.event = Event()
        self.packer = self.mod.SyncSpecDecodeMetadata(self.r, event=self.event)

    def check(self, lengths, drafts, ids):
        actual = self.packer.prepare(np.array(drafts, np.int32), np.cumsum(lengths), ids, NS)
        expected = oracle(lengths, drafts, ids)
        for key, value in expected.items():
            self.assertEqual(getattr(actual, key).dtype, value.dtype, key)
            torch.testing.assert_close(getattr(actual, key).cpu(), value)
        self.assertEqual(actual.num_draft_tokens, drafts)
        return actual

    def test_mixed_prefill_variable_drafts_reorder_shrink_and_zero(self):
        rng = np.random.default_rng(59)
        for lengths, drafts in [
            ([4, 100, 3, 100, 2], [3, 0, 2, 0, 1]),
            ([6, 4, 1], [5, 2, 0]),
            ([1], [0]),
            ([20, 6], [0, 5]),
            ([1] * 8, [0] * 8),
        ]:
            ids = rng.integers(0, 151936, sum(lengths), dtype=np.int32)
            self.check(lengths, drafts, ids)
        self.assertEqual(self.packer.uploads, 5)

    def test_random_valid_batches(self):
        rng = np.random.default_rng(884)
        for _ in range(80):
            n = int(rng.integers(1, 9))
            drafts = rng.integers(0, 6, n).tolist()
            lengths = (np.array(drafts) + rng.integers(1, 18, n)).tolist()
            self.check(lengths, drafts, rng.integers(0, 151936, sum(lengths), dtype=np.int32))

    def test_one_upload_no_device_index_and_host_reuse(self):
        copies = []
        original = torch.Tensor.copy_

        def copy(dst, src, **kw):
            copies.append((dst.numel(), kw))
            return original(dst, src, **kw)

        with patch.object(torch.Tensor, "copy_", copy):
            a = self.check([6, 1], [5, 0], np.arange(7, dtype=np.int32))
            ptr = a.logits_indices.data_ptr()
            self.check([6], [5], np.arange(6, dtype=np.int32))
        self.assertEqual(len(copies), 2)
        self.assertTrue(all(kw.get("non_blocking") for _, kw in copies))
        self.assertEqual(self.event.records, 2)
        self.assertEqual(self.event.waits, 1)
        self.event.done = True
        self.assertEqual(self.check([1], [0], np.array([7], np.int32)).logits_indices.data_ptr(), ptr)
        self.assertEqual(self.event.waits, 1)

    def test_bounds_fail_before_copy(self):
        cases = [([], []), ([6], [6]), ([2], [3]), ([1], [-1]), ([257], [0])]
        for lengths, drafts in cases:
            with self.assertRaises(ValueError):
                self.packer.prepare(np.array(drafts, np.int32), np.cumsum(lengths), np.zeros(256, np.int32), NS)
        self.assertEqual(self.event.records, 0)

    def test_actual_runner_fast_path_and_device_repair_fallback(self):
        r = self.r
        r._spec_decode_metadata_packer = self.packer
        r.use_async_scheduling = r.use_async_spec_decode = False
        r.input_batch = NS(prev_sampled_token_ids=None)
        r.pcp_size = 1
        r._arange_scratch = np.empty(256, np.int64)
        r.arange_np = np.arange(256, dtype=np.int64)
        r._get_cumsum_and_arange = MethodType(cumsum, r)
        method = MethodType(calc_method(), r)
        cpu = np.arange(17, dtype=np.int32) + 200
        gpu = torch.from_numpy(cpu.copy())
        r.input_ids = NS(np=cpu, gpu=gpu)
        lengths, drafts = [12, 5], [0, 3]
        expected = oracle(lengths, drafts, cpu)
        # Fast path must not read device input_ids or pin new small tensors.
        r.input_ids.gpu = NS(__getitem__=lambda _: self.fail("device IDs accessed"))
        with patch.object(torch.Tensor, "pin_memory", side_effect=AssertionError("per-field pinning")):
            actual = method(np.array(drafts, np.int32), np.cumsum(lengths), None)
        for key, value in expected.items():
            torch.testing.assert_close(getattr(actual, key), value)
        self.assertEqual(self.packer.uploads, 1)
        # Emulate repaired device IDs differing from the CPU copy. Must use them.
        r.input_ids.gpu = gpu + 1000
        r.input_batch.prev_sampled_token_ids = torch.tensor([[999]])
        with patch.object(torch.Tensor, "pin_memory", lambda x: x):
            repaired = method(np.array(drafts, np.int32), np.cumsum(lengths), None)
        self.assertEqual(repaired.draft_token_ids.tolist(), (expected["draft_token_ids"] + 1000).tolist())
        self.assertEqual(self.packer.uploads, 1)
        r.input_batch.prev_sampled_token_ids = None
        r._spec_decode_metadata_packer = None
        with patch.object(torch.Tensor, "pin_memory", lambda x: x):
            disabled = method(np.array(drafts, np.int32), np.cumsum(lengths), None)
        self.assertEqual(disabled.draft_token_ids.tolist(), repaired.draft_token_ids.tolist())


def npu_available():
    if importlib.util.find_spec("torch_npu") is None:
        return False
    import torch_npu  # noqa: F401

    return torch.npu.is_available()


@unittest.skipUnless(npu_available(), "requires Ascend NPU and torch_npu")
class NPUMetadataTests(unittest.TestCase):
    def test_repeated_upload_matches_verifier_reference(self):
        r = fixture("npu")
        packer = load().SyncSpecDecodeMetadata(r)
        for n in [8, 1, 5, 3, 8]:
            lengths, drafts = [7] * n, list(range(n))
            drafts = [d % 6 for d in drafts]
            ids = np.arange(7 * n, dtype=np.int32) + 500
            actual = packer.prepare(np.array(drafts, np.int32), np.cumsum(lengths), ids, NS)
            expected = oracle(lengths, drafts, ids)
            for key, value in expected.items():
                torch.testing.assert_close(getattr(actual, key).cpu(), value)


if __name__ == "__main__":
    unittest.main()
