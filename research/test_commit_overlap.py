import unittest

from commit_overlap import analyze


class CommitOverlapTests(unittest.TestCase):
    def test_distinguishes_previous_inference_overlap_from_other_queue_wait(self):
        rows = [
            {"ws_id": "ws-a", "seg": 1, "received_at": "2026-10-19T00:00:00Z",
             "done_at": "2026-10-19T00:00:01Z", "queue_wait_ms": 0,
             "is_empty_commit": False},
            {"ws_id": "ws-a", "seg": 2, "received_at": "2026-10-19T00:00:00.800Z",
             "done_at": "2026-10-19T00:00:01.100Z", "queue_wait_ms": 60,
             "is_empty_commit": False},
            {"ws_id": "ws-b", "seg": 1, "received_at": "2026-10-19T00:00:00.900Z",
             "done_at": "2026-10-19T00:00:01.200Z", "queue_wait_ms": 40,
             "is_empty_commit": False},
            {"ws_id": "ws-a", "seg": 3, "received_at": "2026-10-19T00:00:01.300Z",
             "done_at": "2026-10-19T00:00:01.300Z", "queue_wait_ms": 0,
             "is_empty_commit": True},
        ]
        report = analyze(rows)
        self.assertEqual(report["formal_commit_count"], 3)
        self.assertEqual(report["previous_inference_overlap_count"], 1)
        self.assertEqual(report["overlap_ms"]["max"], 200)
        self.assertEqual(report["queue_wait_ms_when_overlap"]["mean"], 60)
        self.assertEqual(report["queue_wait_ms_without_overlap"]["mean"], 20)

    def test_detects_same_segment_chunk_still_decoding_when_commit_arrives(self):
        rows = [{"ws_id": "ws-cea6c68c", "seg": 24,
                 "received_at": "2026-10-19T07:03:48.371+00:00",
                 "done_at": "2026-10-19T07:03:48.489+00:00",
                 "queue_wait_ms": 118, "is_empty_commit": False}]
        lines = [
            "2026-10-19 07:03:48.117 - INFO - [ws-cea6c68c][seg-24][chunk-0] inference submit",
            "2026-10-19 07:03:48.371 - INFO - [ws-cea6c68c][seg-24] Commit received: queue_size=3",
            "2026-10-19 07:03:48.483 - DEBUG - [ws-cea6c68c][seg-24][chunk-0] token_count=12 decode_total_ms=302.79",
            "2026-10-19 07:03:48.489 - INFO - [ws-cea6c68c][seg-24][chunk-1] commit consumed",
        ]
        report = analyze(rows, service_log_lines=lines)
        self.assertEqual(report["previous_inference_overlap_count"], 0)
        self.assertEqual(report["same_segment_chunk_overlap_count"], 1)
        self.assertEqual(report["same_segment_chunk_overlap_ms"]["max"], 112)
        self.assertEqual(report["queue_wait_ms_with_chunk_overlap"]["mean"], 118)


if __name__ == "__main__":
    unittest.main()
