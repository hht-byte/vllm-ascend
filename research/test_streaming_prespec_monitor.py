"""Focused checks for test-window filtering and commit/monitor alignment."""

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from research import streaming_prespec_monitor as monitor


SCRIPT = Path(__file__).with_name("streaming_prespec_monitor.py")


class SummaryAlignmentTest(unittest.TestCase):
    def test_sample_records_separate_source_completion_times(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent) as directory:
            output = Path(directory) / "samples.jsonl"
            args = argparse.Namespace(
                run_id="c5", npu_cmd_json='["npu-smi"]', npu_util_regex=None,
                out=str(output), duration=0.01, interval=1.0,
                metrics_url="http://example.invalid/metrics", timeout=1.0,
            )
            with patch.object(monitor, "read_metrics", return_value={"waiting": 0, "running": 1}), \
                    patch.object(monitor, "read_npu", return_value={"utilization_pct": 20}):
                monitor.sample(args)
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            start = datetime.fromisoformat(row["time_utc"])
            metrics_done = datetime.fromisoformat(row["metrics_time_utc"])
            npu_done = datetime.fromisoformat(row["npu_time_utc"])
            self.assertLessEqual(start, metrics_done)
            self.assertLessEqual(metrics_done, npu_done)

    def test_excludes_pretest_samples_and_aligns_commit_by_utc(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent) as directory:
            root = Path(directory)
            samples = root / "samples.jsonl"
            commits = root / "commits.jsonl"
            summary = root / "summary.json"
            aligned = root / "aligned.jsonl"
            samples.write_text("\n".join(json.dumps(row) for row in [
                {"run_id": "c5", "time_utc": "2026-10-19T00:00:00+00:00",
                 "metrics": {"waiting": 0, "running": 0},
                 "npu": {"utilization_pct": 0}},
                {"run_id": "c5", "time_utc": "2026-10-19T00:00:10.100+00:00",
                 "metrics_time_utc": "2026-10-19T00:00:10.200+00:00",
                 "npu_time_utc": "2026-10-19T00:00:10.400+00:00",
                 "metrics": {"waiting": 2, "running": 3},
                 "npu": {"utilization_pct": 40}},
                {"run_id": "c5", "time_utc": "2026-10-19T00:00:21+00:00",
                 "metrics": {"waiting": 0, "running": 0},
                 "npu": {"utilization_pct": 0}},
            ]) + "\n", encoding="utf-8")
            commits.write_text("\n".join(json.dumps(row) for row in [
                {"ws_id": "ws-a", "seg": 1,
                 "received_at": "2026-10-19T00:00:10.500+00:00",
                 "done_at": "2026-10-19T00:00:11+00:00",
                 "commit_latency_ms": 500, "queue_size": 3},
                {"ws_id": "ws-b", "seg": 1,
                 "received_at": "2026-10-19T00:00:21+00:00",
                 "done_at": "2026-10-19T00:00:22+00:00",
                 "commit_latency_ms": 1000},
            ]) + "\n", encoding="utf-8")
            result = subprocess.run([
                sys.executable, str(SCRIPT), "summarize", "--run-id", "c5",
                "--concurrency", "5", "--samples", str(samples),
                "--commits", str(commits),
                "--test-start-utc", "2026-10-19T00:00:10Z",
                "--test-end-utc", "2026-10-19T00:00:20Z",
                "--aligned-out", str(aligned), "--out", str(summary),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(data["sample_count"], 1)
            self.assertEqual(data["queue_waiting"]["mean"], 2)
            self.assertEqual(data["npu_utilization_pct"]["mean"], 40)
            self.assertEqual(data["formal_commit_count"], 1)
            self.assertEqual(data["commit_latency_ms"]["mean"], 500)
            self.assertEqual(data["service_queue_size"]["mean"], 3)
            self.assertEqual(data["audio_blocks_pending_at_commit"]["mean"], 2)
            rows = [json.loads(line) for line in aligned.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["metrics_at_receive"]["offset_ms"], -300)
            self.assertEqual(rows[0]["metrics_at_receive"]["waiting"], 2)
            self.assertEqual(rows[0]["npu_at_receive"]["offset_ms"], -100)
            self.assertEqual(rows[0]["npu_at_receive"]["utilization_pct"], 40)


if __name__ == "__main__":
    unittest.main()
