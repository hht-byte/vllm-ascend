"""Bounded CPU bookkeeping after the device rejection handoff."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_asr_mtp_device_rejection import DeviceRejectionTests as _DeviceFixture


class _HistoryReads:
    """Record real NumPy history reads without changing the selected values."""
    def __init__(self, array):
        self.array = array
        self.sizes = []

    def __getitem__(self, key):
        selected = self.array[key]
        self.sizes.append(selected.size)
        return selected


class PendingBookkeepingTests(_DeviceFixture):
    def bookkeep(self, worker, cpu):
        sampled = [[token for token in row if token >= 0] for row in cpu.tolist()]
        batch = worker.runner.input_batch
        for i, (req, row) in enumerate(zip(batch.req_ids, sampled)):
            confirmed = int(batch.num_tokens_no_spec[i])
            batch.token_ids_cpu[i, confirmed:confirmed+len(row)] = row
            batch.num_tokens_no_spec[i] += len(row)
            worker.runner.requests[req].output_token_ids.extend(row)
        return sampled

    def test_pending_reordered_shrunk_batch_reads_only_sampled_suffix(self):
        worker, scheduler, sampler = self.make_worker()
        batch = worker.runner.input_batch
        order = [2, 0]
        batch.req_ids = [batch.req_ids[i] for i in order]
        for name in ('num_tokens_no_spec', 'num_computed_tokens_cpu', 'token_ids_cpu'):
            setattr(batch, name, getattr(batch, name)[order].copy())
        sampler.sampled_token_ids = sampler.sampled_token_ids[order]
        hidden = torch.randn(8, 12)
        cpu = worker.device_handoff.prepare(scheduler, sampler, hidden)
        sampled = self.bookkeep(worker, cpu)
        drafts = [list(row) for row in worker.device_handoff.pending[2]]
        history = _HistoryReads(batch.token_ids_cpu)
        batch.token_ids_cpu = history

        output = worker.propose(sampled, scheduler, hidden)

        self.assertEqual(output, drafts)
        self.assertEqual(history.sizes, [4, 1])
        self.assertIsNone(worker.device_handoff.pending)
        for req in batch.req_ids:
            state = worker.states[req]
            self.assertEqual([head.valid for head in state.heads], [state.length]*3)

    def test_pending_suffix_mismatch_still_rejects_post_bookkeeping_history(self):
        worker, scheduler, sampler = self.make_worker()
        hidden = torch.randn(12, 12)
        sampled = self.bookkeep(worker, worker.device_handoff.prepare(scheduler, sampler, hidden))
        batch = worker.runner.input_batch
        batch.token_ids_cpu[0, int(batch.num_tokens_no_spec[0])-1] = 62
        history = _HistoryReads(batch.token_ids_cpu)
        batch.token_ids_cpu = history

        with self.assertRaisesRegex(RuntimeError, 'confirmed tokens are bookkept'):
            worker.propose(sampled, scheduler, hidden)

        self.assertEqual(history.sizes, [1])
        self.assertIsNone(worker.device_handoff.pending)

    def test_fallback_still_proposes_from_full_confirmed_inputs(self):
        worker, scheduler, sampler = self.make_worker()
        hidden = torch.randn(12, 12)
        scheduler.num_invalid_spec_tokens['stream_a_0'] = 1
        self.assertIsNone(worker.device_handoff.prepare(scheduler, sampler, hidden))
        scheduler.num_invalid_spec_tokens.clear()
        sampled = self.bookkeep(worker, sampler.sampled_token_ids)
        batch = worker.runner.input_batch
        inputs = []
        for i, (state, row) in enumerate(zip(self.a, sampled)):
            start = int(batch.num_computed_tokens_cpu[i])
            state.length = int(batch.num_tokens_no_spec[i])-1
            state.crop_heads(state.length)
            state.target[start:state.length] = hidden[i*4:i*4+len(row)]
            inputs.append(batch.token_ids_cpu[i, :state.length+1].tolist())
        expected = self.eager.propose(self.a, inputs, self.embed, self.logits, [3]*3).tolist()
        history = _HistoryReads(batch.token_ids_cpu)
        batch.token_ids_cpu = history

        self.assertEqual(worker.propose(sampled, scheduler, hidden), expected)
        self.assertEqual(history.sizes, [9, 6, 10])

    def test_readback_copies_only_active_rows_without_replacing_host_storage(self):
        self.make_worker()
        prelude = self.program.rejection
        host_pointer = prelude.host.data_ptr()
        prelude.host.fill_(-777)
        prelude.combined.copy_(torch.arange(21).view(3, 7))

        first = prelude.readback(2).clone()

        self.assertEqual(first.tolist(), [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]])
        self.assertEqual(prelude.host[2].tolist(), [-777]*7)
        prelude.combined[:2].copy_(prelude.combined[:2].flip(0))
        second = prelude.readback(1)
        self.assertEqual(second.tolist(), [[7, 8, 9, 10, 11, 12, 13]])
        self.assertEqual(prelude.host[1].tolist(), [7, 8, 9, 10, 11, 12, 13])
        self.assertEqual(prelude.host[2].tolist(), [-777]*7)
        self.assertEqual(prelude.host.data_ptr(), host_pointer)

    def test_readback_completes_one_event_before_releasing_uploads(self):
        self.make_worker()
        prelude = self.program.rejection
        prelude.host.fill_(-777)
        prelude.combined.fill_(31)
        prelude.uploads.pending = self.program.uploads.pending = True
        calls = []
        stream = object()

        def record(actual_stream):
            self.assertIs(actual_stream, stream)
            self.assertTrue(prelude.uploads.pending)
            self.assertTrue(self.program.uploads.pending)
            calls.append('record')

        def synchronize():
            self.assertTrue(prelude.uploads.pending)
            self.assertTrue(self.program.uploads.pending)
            self.assertEqual(prelude.host[0].tolist(), [31]*7)
            calls.append('synchronize')

        prelude.event = SimpleNamespace(record=record, synchronize=synchronize)
        with patch.object(torch, 'npu', SimpleNamespace(current_stream=lambda: stream), create=True):
            result = prelude.readback(1)

        self.assertEqual(result.tolist(), [[31]*7])
        self.assertEqual(calls, ['record', 'synchronize'])
        self.assertFalse(prelude.uploads.pending)
        self.assertFalse(self.program.uploads.pending)
        self.assertEqual(prelude.host[1:].tolist(), [[-777]*7, [-777]*7])

    def test_failed_readback_completion_retains_upload_ownership(self):
        self.make_worker()
        prelude = self.program.rejection
        prelude.uploads.pending = self.program.uploads.pending = True

        def synchronize():
            raise RuntimeError('completion failed')

        prelude.event = SimpleNamespace(record=lambda stream: None, synchronize=synchronize)
        with patch.object(torch, 'npu', SimpleNamespace(current_stream=lambda: object()), create=True):
            with self.assertRaisesRegex(RuntimeError, 'completion failed'):
                prelude.readback(1)

        self.assertTrue(prelude.uploads.pending)
        self.assertTrue(self.program.uploads.pending)


# Reuse real paged/worker setup without repeating its independent parity tests.
for _name in dir(_DeviceFixture):
    if _name.startswith('test_') and _name not in PendingBookkeepingTests.__dict__:
        setattr(PendingBookkeepingTests, _name, None)
del _DeviceFixture
