"""CPU-only source attribution and acceptance aggregation contracts."""

import importlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("asr_mtp_metrics")
        except ImportError:
            self.fail("MTP acceptance tracker is missing")
        self.logs = []
        self.tracker = self.api.AcceptanceTracker(5, emit=self.logs.append)

    def offer(self, req="finish_1", tokens=(10, 11, 12, 13, 14), anchor=20):
        self.tracker.offer(req, tokens, anchor, fresh_positions=35)

    def observe(self, req="finish_1", scheduled=(10, 11, 12, 13, 14), sampled=(10, 11, 99), **kw):
        self.tracker.observe(req, scheduled, sampled, start=20, count=6, **kw)

    def done(self, req="finish_1"):
        self.tracker.finish(req)
        return self.logs[-1]

    def test_partial_zero_and_full_acceptance_include_zero_rounds(self):
        for sampled in ((10, 11, 99), (99,), (10, 11, 12, 13, 14, 99)):
            self.offer()
            self.observe(sampled=sampled)
        row = self.done()
        self.assertEqual(row["total"]["rounds"], 3)
        self.assertEqual(row["total"]["drafted"], 15)
        self.assertEqual(row["total"]["accepted"], 7)
        self.assertEqual(row["total"]["accepted_histogram"], [1, 0, 1, 0, 0, 1])
        self.assertEqual(row["total"]["accepted_per_position"], [2, 2, 1, 1, 1])
        self.assertEqual(row["total"]["exposed_per_position"], [3] * 5)
        self.assertEqual(row["phases"]["cold"]["rounds"], 1)
        self.assertEqual(row["phases"]["warm"]["rounds"], 2)

    def test_initial_external_and_equal_tokens_at_wrong_anchor_are_not_mtp(self):
        self.observe()  # No MTP proposal: external draft must be ignored.
        self.offer()
        self.tracker.observe("finish_1", (10, 11), (10, 99), start=0, count=3)
        row = self.done()
        self.assertEqual(row["total"]["rounds"], 0)
        self.assertEqual(row["unverified_tokens"], 5)
        self.assertEqual(row["discard_reasons"]["context_mismatch"], 1)

    def test_scheduler_truncation_and_invalid_tail_use_actual_verified_length(self):
        self.offer()
        self.tracker.observe("finish_1", (10, 11, 12), (10, 99), start=20, count=4, invalid=1)
        row = self.done()
        self.assertEqual(row["total"]["drafted"], 2)
        self.assertEqual(row["total"]["accepted"], 1)
        self.assertEqual(row["unverified_tokens"], 3)
        self.assertEqual(row["total"]["exposed_per_position"], [1, 1, 0, 0, 0])

    def test_empty_sampling_retains_pending_until_real_validation(self):
        self.offer()
        self.tracker.observe("finish_1", (), (), start=0, count=10)
        self.observe(sampled=(99,))
        row = self.done()
        self.assertEqual(row["total"]["rounds"], 1)
        self.assertEqual(row["unverified_tokens"], 0)

    def test_stop_token_in_accepted_prefix_is_counted_before_scheduler_truncation(self):
        self.offer(tokens=(10, 2, 12))
        self.tracker.observe("finish_1", (10, 2, 12), (10, 2, 99), start=20, count=4)
        self.assertEqual(self.done()["total"]["accepted"], 2)

    def test_cancel_reuse_and_replaced_proposal_are_separate_lifetimes(self):
        self.offer()
        self.offer(tokens=(6,))
        first = self.done()
        self.assertEqual(first["unverified_tokens"], 6)
        self.assertEqual(first["unverified_proposal_rounds"], 2)
        self.offer()
        self.observe()
        second = self.done()
        self.assertNotEqual(first["lifetime"], second["lifetime"])
        self.assertEqual(second["proposal_rounds"], 1)
        self.tracker.finish("finish_1")
        self.assertEqual(len(self.logs), 2)

    def test_wrong_tokens_incomplete_forward_and_impossible_acceptance_excluded(self):
        for scheduled, sampled, count in (
            ((9,), (9, 7), 2),
            ((10, 11), (10, 99), 1),
            ((10,), (10, 11, 99), 2),
        ):
            self.offer()
            self.tracker.observe("finish_1", scheduled, sampled, start=20, count=count)
        row = self.done()
        self.assertEqual(row["total"]["rounds"], 0)
        self.assertEqual(row["unverified_tokens"], 15)

    def test_request_type_and_live_snapshot_do_not_double_count(self):
        for req, kind in (
            ("stream_12_0-abc", "chunk"),
            ("finish_12-a", "finish"),
            ("prespec-xyz", "shadow"),
            ("custom", "other"),
        ):
            self.offer(req)
            self.observe(req)
            self.assertEqual(self.tracker.snapshot(req)["request_type"], kind)
            self.done(req)

    def test_analyzer_weighted_metrics_and_duplicate_guard(self):
        self.offer(tokens=(10,))
        self.observe(scheduled=(10,), sampled=(10, 99))
        self.offer()
        self.observe(sampled=(99,))
        row = self.done()
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "server.log"
            path.write_text("prefix ASR_MTP_ACCEPTANCE " + json.dumps(row) + "\n")
            report = self.api.analyze([path])
            group = report["groups"][0]
            self.assertEqual(group["acceptance_rate"], 1 / 6)
            self.assertEqual(group["mean_accepted_length"], 0.5)
            self.assertEqual(group["mean_advance_length"], 1.5)
            self.assertEqual(group["position_survival"], [0.5, 0, 0, 0, 0])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.api.analyze([path, path])

    def test_analyzer_rejects_missing_or_corrupt_summaries(self):
        self.offer()
        self.observe()
        row = self.done()
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "server.log"
            path.write_text("ASR_MTP_STEP requests=1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no ASR_MTP_ACCEPTANCE"):
                self.api.analyze([path])
            row["total"]["accepted_histogram"] = []
            path.write_text(self.api.MARKER + json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "array"):
                self.api.analyze([path])

    def test_cli_accepts_labels_and_writes_json_without_torch(self):
        self.offer()
        self.observe()
        row = self.done()
        with TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path, output = Path(directory) / "server.log", Path(directory) / "out.json"
            path.write_text(self.api.MARKER + json.dumps(row), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    self.api.__file__,
                    "--run",
                    f"h5={path}",
                    "--out",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            content = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(content["h5"]["groups"][0]["mean_accepted_length"], 2)


if __name__ == "__main__":
    unittest.main()
