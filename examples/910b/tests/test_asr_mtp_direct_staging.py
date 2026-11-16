"""Direct host staging preserves padded bytes and fences their reuse."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from asr_mtp_staging import UploadBatch
from asr_mtp_token_attention import PackedAttention
from asr_mtp_token_graph import PackedMTPProgram


def state(slot, length, valid):
    return SimpleNamespace(
        slot=slot, length=length,
        heads=[SimpleNamespace(valid=value) for value in valid],
    )


def baseline_payload(program, states, inputs, limits):
    """Original list formula, independent of the direct staging implementation."""
    r = program.max_batch + 1
    groups = [[] for _ in range(5)]
    segments, orders = [], []
    for d in range(program.heads.num_heads):
        q, kv, order = [], [], []
        starts, last = [0] * r, [0] * r
        q_total = kv_total = 0
        for i, (item, limit) in enumerate(zip(states, limits)):
            starts[i] = item.heads[d].valid
            if d < limit:
                q_total += item.length - item.heads[d].valid
                kv_total += item.length
                q.append(q_total)
                kv.append(kv_total)
                order.append(i)
                last[i] = q_total - 1
        q.append(program.query_capacity)
        kv.append(kv_total + program.query_capacity - q_total
                  if program.paged_kv else program.kv_capacity)
        order.append(program.max_batch)
        segments.append((tuple(q), tuple(kv)))
        orders.append(order)
        for group, row, fill in zip(
            groups, (q, kv, order, starts, last),
            (program.query_capacity, program.kv_capacity, program.max_batch, 0, 0),
        ):
            group.extend(row + [fill] * (r - len(row)))
    rows = [list(row) + [0] * (program.sequence_width - len(row)) for row in inputs]
    rows += [[0] * program.sequence_width for _ in range(r - len(states))]
    payload = [value for group in groups for value in group]
    for values in ([s.slot for s in states], [s.length for s in states], limits):
        payload.extend(list(values) + [0] * (r - len(states)))
    payload.extend(value for row in rows for value in row)
    tables = [
        [program.arena.block_rows[states[i].slot]
         if i < len(states) else program.arena.block_rows[-1] for i in order]
        for order in orders[:program.execution_heads]
    ] if program.paged_kv else None
    return torch.tensor(payload, dtype=torch.long), segments, tables


class ShadowEvent:
    """Inspect host bytes at both boundaries of a delayed-copy reuse fence."""

    def __init__(self):
        self.records = self.queries = self.waits = 0
        self.inspect = lambda: None
        self.done = False

    def record(self):
        self.records += 1
        self.done = False

    def query(self):
        self.queries += 1
        self.inspect()
        return self.done

    def synchronize(self):
        self.waits += 1
        self.inspect()
        self.done = True


class DirectStageTests(unittest.TestCase):
    def make(self, *, context=17, paged=False, execution_heads=3):
        program = PackedMTPProgram.__new__(PackedMTPProgram)
        program.heads = SimpleNamespace(num_heads=3)
        program.max_batch, program.context, program.tokens = 3, context, 32
        program.execution_heads = execution_heads
        program.query_capacity = 33
        program.sequence_width = context + 4
        program.kv_capacity = 128 if paged else 3 * context + 33
        program.paged_kv = paged
        program.arena = SimpleNamespace(
            storage=torch.zeros(5, 1),
            block_rows=[[i * 2, i * 2 + 1] for i in range(5)],
        )
        r = 4
        program.metadata = torch.zeros(5 * 3 * r + 3 * r + r * program.sequence_width,
                                       dtype=torch.long)
        event = ShadowEvent()
        program.uploads = UploadBatch("cpu", event=event)
        program.metadata_upload = program.uploads.buffer(program.metadata)
        offset = 0
        for name, shape in (
            *((name, (3, r)) for name in ("q_ends", "kv_ends", "order", "starts", "last")),
            *((name, (r,)) for name in ("slots", "lengths", "limits")),
            ("sequence", (r, program.sequence_width)),
        ):
            size = 1
            for extent in shape:
                size *= extent
            setattr(program, name, program.metadata[offset:offset + size].view(shape))
            offset += size
        args = dict(device="cpu", dtype=torch.float32, heads=execution_heads)
        if paged:
            from asr_mtp_paged_attention import PagedAttention
            program.attention = PagedAttention(
                2, 1, 4, max_batch=3, block_size=4, num_blocks=10,
                max_blocks=2, uploads=program.uploads, **args,
            )
        else:
            program.attention = PackedAttention(2, 1, 4, **args)
        return program, event

    def check(self, program, states, rows, limits, *, allow_inactive=False):
        expected, segments, tables = baseline_payload(program, states, rows, limits)
        program.stage(states, rows, limits, allow_inactive=allow_inactive)
        self.assertEqual(program.metadata.numpy().tobytes(), expected.numpy().tobytes())
        self.assertEqual(program.metadata_upload.host.numpy().tobytes(), expected.numpy().tobytes())
        self.assertEqual(program.attention.lengths,
                         dict(enumerate(segments[:program.execution_heads])))
        if tables is not None:
            self.assertEqual(program.attention.table_rows,
                             tuple(tuple(tuple(row) for row in table) for table in tables))

    def test_bytes_match_baseline_after_mixed_inactive_reorder_and_shrink(self):
        for paged in (False, True):
            with self.subTest(paged=paged):
                program, _ = self.make(paged=paged, execution_heads=2)
                pointer = program.metadata_upload.host.data_ptr()
                states = [state(4, 7, (3, 2, 6)), state(0, 3, (1, 0, 2)),
                          state(2, 5, (2, 4, 1))]
                rows = [list(range(s.length + 1)) for s in states]
                self.check(program, states, rows, [2, 0, 1], allow_inactive=True)
                self.check(program, states[::-1], rows[::-1], [1, 2, 0], allow_inactive=True)
                self.check(program, states[1:2], rows[1:2], [0], allow_inactive=True)
                self.assertEqual(pointer, program.metadata_upload.host.data_ptr())

    def test_long_context_and_exact_capacity_keep_dummy_padding(self):
        for paged in (False, True):
            with self.subTest(paged=paged):
                program, _ = self.make(context=4096, paged=paged)
                self.check(program, [state(3, 4096, (4064, 4095, 4088))],
                           [list(range(4097))], [3])
                self.assertEqual(program.q_ends[0].tolist(), [32, 33, 33, 33])

    def test_stages_typed_storage_without_full_python_payload(self):
        program, _ = self.make()
        # copy(values) would rebuild a complete Python representation. Direct
        # staging must fill reusable host views and upload those bytes once.
        with patch.object(program.metadata_upload, "copy",
                          side_effect=AssertionError("full Python metadata payload")):
            self.check(program, [state(2, 3, (0, 1, 2))], [[11, 12, 13, 14]], [3])

    def test_input_view_aliasing_host_storage_is_preserved_before_zero_fill(self):
        program, _ = self.make()
        states = [state(2, 3, (0, 1, 2))]
        self.check(program, states, [[11, 12, 13, 14]], [3])
        offset = 5 * 3 * 4 + 3 * 4
        tokens = program.metadata_upload.array[offset:offset + 4]
        self.check(program, states, [tokens], [3])

    def test_wait_observes_previous_metadata_and_tables_before_any_write(self):
        program, event = self.make(paged=True)
        first = [state(2, 5, (1, 2, 3)), state(0, 3, (0, 1, 2))]
        rows = [[7] * (s.length + 1) for s in first]
        self.check(program, first, rows, [3, 1])
        snapshots = [buf.host.clone() for buf in
                     (program.metadata_upload, program.attention.table_upload)]

        def inspect():
            for buf, expected in zip(
                (program.metadata_upload, program.attention.table_upload), snapshots
            ):
                self.assertTrue(torch.equal(buf.host, expected), "host changed before H2D fence")

        event.inspect = inspect
        copies = []
        original = torch.Tensor.copy_

        def copy(target, source, *args, **kwargs):
            if target is program.metadata:
                copies.append(source.data_ptr())
            return original(target, source, *args, **kwargs)

        with patch.object(torch.Tensor, "copy_", copy):
            self.check(program, first[::-1], rows[::-1], [2, 3])
        self.assertEqual(copies, [program.metadata_upload.host.data_ptr()])
        self.assertEqual((event.records, event.queries, event.waits), (2, 1, 1))

    def test_bad_final_token_does_not_modify_host_or_wait_for_previous_copy(self):
        program, event = self.make()
        states = [state(2, 3, (0, 1, 2)), state(0, 2, (0, 0, 1))]
        self.check(program, states, [[1, 2, 3, 4], [5, 6, 7]], [3, 2])
        before = program.metadata_upload.host.clone()
        with self.assertRaises((TypeError, ValueError, OverflowError)):
            program.stage(states, [[8, 9, 10, 11], [12, 13, "invalid"]], [3, 2])
        self.assertTrue(torch.equal(program.metadata_upload.host, before))
        self.assertEqual((event.records, event.queries, event.waits), (1, 0, 0))

    def test_noninteger_attention_boundaries_fail_before_host_mutation(self):
        program, event = self.make()
        states = [state(2, 3, (0, 1, 2))]
        self.check(program, states, [[1, 2, 3, 4]], [3])
        before = program.metadata_upload.host.clone()
        for invalid in (state(1, 3.0, (0, 1, 2)), state(1, 3, (0.5, 1, 2))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                program.stage([invalid], [[5, 6, 7, 8]], [3])
            self.assertTrue(torch.equal(program.metadata_upload.host, before))
        self.assertEqual((event.records, event.queries, event.waits), (1, 0, 0))

    def test_invalid_page_table_preserves_previous_host_uploads(self):
        program, event = self.make(paged=True)
        self.check(program, [state(2, 3, (0, 1, 2))], [[1, 2, 3, 4]], [3])
        metadata = program.metadata_upload.host.clone()
        tables = program.attention.table_upload.host.clone()
        program.arena.block_rows[0] = [10, 11]
        with self.assertRaises(ValueError):
            program.stage([state(0, 3, (0, 1, 2))], [[5, 6, 7, 8]], [3])
        self.assertTrue(torch.equal(program.metadata_upload.host, metadata))
        self.assertTrue(torch.equal(program.attention.table_upload.host, tables))
        self.assertEqual((event.records, event.queries, event.waits), (1, 0, 0))


if __name__ == "__main__":
    unittest.main()
