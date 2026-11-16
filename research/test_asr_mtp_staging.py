"""Host reuse must wait for H2D, including repeated graph buckets in one step."""

import importlib.util
import unittest
from unittest.mock import patch

import torch


class Event:
    def __init__(self):
        self.records = self.waits = self.queries = 0
        self.done = False
        self.before_wait = lambda: None

    def record(self):
        self.records += 1
        self.done = False

    def query(self):
        self.queries += 1
        return self.done

    def synchronize(self):
        self.before_wait()
        self.waits += 1
        self.done = True


class UploadTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch_npu"), "requires real NPU")
    def test_npu_reuses_host_storage_under_queued_copies(self):
        import torch_npu  # noqa: F401
        from asr_mtp_staging import UploadBatch

        device = torch.device("npu")
        owner = UploadBatch(device)
        dst = torch.empty(4096, dtype=torch.int64, device=device)
        upload = owner.buffer(dst)
        self.assertTrue(upload.host.is_pinned())
        snapshots = []
        for value in range(20):
            upload.copy([value] * 4096)
            snapshots.append(dst.clone())
        self.assertEqual(
            torch.stack(snapshots).cpu().tolist(), [[i] * 4096 for i in range(20)]
        )

    def make(self):
        from asr_mtp_staging import UploadBatch

        event = Event()
        batch = UploadBatch(torch.device("cpu"), event=event)
        target = torch.zeros(2, 3, dtype=torch.long)
        return batch, batch.buffer(target), target, event

    def test_repeated_upload_keeps_host_storage_and_waits_before_overwrite(self):
        _batch, buf, target, event = self.make()
        ptr = buf.host.data_ptr()
        buf.copy([[1, 2, 3], [4, 5, 6]])
        event.before_wait = lambda: self.assertEqual(
            buf.host.tolist(), [[1, 2, 3], [4, 5, 6]]
        )
        buf.copy([[7, 8, 9], [0, 0, 0]])
        self.assertEqual(ptr, buf.host.data_ptr())
        self.assertEqual(target.tolist(), [[7, 8, 9], [0, 0, 0]])
        self.assertEqual((event.records, event.waits), (2, 1))
        event.done = True
        buf.copy([[0] * 3] * 2)
        self.assertEqual(event.waits, 1)

    def test_metadata_and_tables_share_one_record_and_empty_batch_records_none(self):
        batch, buf, _target, event = self.make()
        tables = batch.buffer(torch.zeros(4, dtype=torch.int32))
        with batch:
            buf.copy([[1] * 3] * 2)
            tables.copy([2, 3, 4, 5])
        self.assertEqual(event.records, 1)
        with batch:
            pass
        self.assertEqual(event.records, 1)

    def test_exception_after_copy_still_fences_inflight_host_storage(self):
        batch, buf, _, event = self.make()
        with self.assertRaisesRegex(RuntimeError, "stage failed"), batch:
            buf.copy([[1] * 3] * 2)
            raise RuntimeError("stage failed")
        self.assertEqual(event.records, 1)
        buf.copy([[2] * 3] * 2)
        self.assertEqual(event.waits, 1)

    def test_cpu_path_never_requests_pinning_or_device_event(self):
        from asr_mtp_staging import UploadBatch

        batch = UploadBatch(torch.device("cpu"))
        target = torch.empty(2, dtype=torch.int32)
        buf = batch.buffer(target)
        buf.copy([9, 10])
        self.assertFalse(buf.host.is_pinned())
        self.assertEqual(target.tolist(), [9, 10])

    def test_paged_tables_share_owner_and_skip_unchanged_upload(self):
        from asr_mtp_paged_attention import PagedAttention
        from asr_mtp_staging import UploadBatch

        event = Event()
        owner = UploadBatch(torch.device("cpu"), event=event)
        attention = PagedAttention(
            2,
            1,
            4,
            device="cpu",
            dtype=torch.float32,
            heads=2,
            max_batch=2,
            block_size=4,
            num_blocks=8,
            max_blocks=2,
            uploads=owner,
        )
        metadata = owner.buffer(torch.zeros(3, dtype=torch.long))
        rows = [[[1, 2]], [[3, 4]]]
        ptr = attention.table_upload.host.data_ptr()
        with patch("torch.tensor", side_effect=AssertionError("runtime allocation")):
            with owner:
                metadata.copy([1, 2, 3])
                attention.set_tables(rows)
            attention.set_tables(rows)
        self.assertEqual((event.records, event.queries, event.waits), (1, 0, 0))
        attention.set_tables([[[5, 6], [1, 2]], [[7, 0]]])
        self.assertEqual(event.waits, 1)
        self.assertEqual(ptr, attention.table_upload.host.data_ptr())
        self.assertEqual(attention.tables[0].tolist(), [[5, 6], [1, 2], [0, 0]])
        self.assertEqual(attention.tables[1].tolist(), [[7, 0], [0, 0], [0, 0]])


if __name__ == "__main__":
    unittest.main()
