"""Request-level gain accounting from formal and verifier log events."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("streaming_prespec_gain.py")


def formal(request_id, *, source="none", draft=0, output=8, rounds=4,
           boundary="chunk", **extra):
    return {"request_id": request_id, "ws_id": "ws-a", "seg": 2,
            "boundary": boundary, "draft_source": source,
            "submitted_draft_tokens": draft, "output_token_count": output,
            "decode_rounds": rounds, "prefill_ms": 70.0,
            "decode_total_ms": 90.0, "finished": True, **extra}


def verify(request_id, round_index, *, source="prespec", effective=5,
           accepted=4, reason=None, complete=True, actual_effective=None):
    row = {"request_id": request_id, "round_index": round_index,
           "effective_draft_tokens": effective,
           "accepted_draft_tokens": accepted}
    if source is not None:
        row["draft_source"] = source
    if reason is not None:
        row["reason"] = reason
    if complete:
        row["verifier_complete"] = True
        row["actual_effective_draft_tokens"] = (
            effective if actual_effective is None else actual_effective)
    return row


def line(marker, row, at="2026-10-23 06:37:56.231"):
    return f"{at} - INFO - {marker} {json.dumps(row)}\n"


class GainAnalyzerTest(unittest.TestCase):
    def run_cli(self, rollback_rows, consume_rows, *, shadow_rows=None,
                allow_legacy_time_join=False):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent) as temp:
            root = Path(temp)
            paths = {}
            for name, rows in (("rollback", rollback_rows),
                               ("consume", consume_rows),
                               ("shadow", shadow_rows)):
                if rows is None:
                    continue
                paths[name] = root / f"{name}.log"
                rendered = []
                for item in rows:
                    kind, row, *timestamp = item
                    rendered.append(line("ASR_FORMAL_STATS" if kind == "formal"
                                         else "ASR_VERIFY_STEP", row,
                                         *(timestamp or [])))
                paths[name].write_text("noise unrelated\n" + "".join(rendered),
                                       encoding="utf-8")
            output = root / "gain.json"
            cmd = [sys.executable, str(SCRIPT),
                   "--rollback-log", str(paths["rollback"]),
                   "--consume-log", str(paths["consume"]),
                   "--out", str(output)]
            if shadow_rows is not None:
                cmd.extend(["--shadow-log", str(paths["shadow"])])
            if allow_legacy_time_join:
                cmd.append("--allow-legacy-time-join")
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
            return result, data

    def test_multiple_output_tokens_per_round_are_not_counted_as_rounds(self):
        rollback = [("formal", formal("base", output=8, rounds=4))]
        consume = [("formal", formal("target", source="prespec", draft=7,
                                      output=9, rounds=3)),
                   ("verify", verify("target", 0, accepted=4,
                                      complete=False)),
                   ("verify", verify("target", 1, effective=2, accepted=1,
                                      actual_effective=7))]
        result, data = self.run_cli(rollback, consume)
        self.assertEqual(result.returncode, 0, result.stderr)
        overall = data["runs"]["consume"]["all"]
        self.assertEqual(overall["formal_request_count"], 1)
        self.assertEqual(overall["output_token_count"]["mean"], 9)
        self.assertEqual(overall["decode_rounds"]["mean"], 3)
        self.assertEqual(overall["tokens_per_round"]["mean"], 3)
        self.assertEqual(overall["accepted_draft_tokens"]["sum"], 5)
        self.assertEqual(overall["effective_draft_tokens"]["sum"], 7)
        self.assertAlmostEqual(overall["observed_acceptance_rate"], 5 / 7)
        self.assertEqual(overall["missing_verifier_count"], 0)
        self.assertEqual(data["runs"]["consume"]["by_boundary"]["chunk"]
                         ["accepted_draft_tokens"]["sum"], 5)
        self.assertEqual(data["runs"]["consume"]["by_draft_source"]["prespec"]
                         ["accepted_draft_tokens"]["sum"], 5)
        self.assertEqual(data["comparison"]["rollback_to_consume"]["all"]
                         ["decode_rounds"]["mean_delta"], -1)
        self.assertEqual(data["runs"]["consume"]["records"][0]
                         ["accepted_draft_tokens"], 5)
        self.assertEqual(data["runs"]["consume"]["records"][0]
                         ["verifier_status"], "observed")
        self.assertIn("not paired", data["methodology"].lower())

    def test_verifier_acceptance_is_not_inferred_from_prefix_match(self):
        rollback = [("formal", formal("base"))]
        consume = [("formal", formal("target", source="prespec", draft=6,
                                      raw_matched_prefix_tokens=0)),
                   ("verify", verify("target", 0, effective=6, accepted=3))]
        result, data = self.run_cli(rollback, consume)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(data["runs"]["consume"]["all"]
                         ["accepted_draft_tokens"]["sum"], 3)
        self.assertEqual(data["runs"]["consume"]["all"]
                         ["observed_acceptance_rate"], 0.5)

    def test_missing_verifier_is_not_treated_as_zero_acceptance(self):
        rollback = [("formal", formal("base"))]
        consume = [("formal", formal("missing", source="prespec", draft=7)),
                   ("formal", formal("plain", source="none", draft=0))]
        result, data = self.run_cli(rollback, consume)
        self.assertEqual(result.returncode, 0, result.stderr)
        overall = data["runs"]["consume"]["all"]
        self.assertEqual(overall["missing_verifier_count"], 1)
        self.assertEqual(overall["missing_verifier_request_ids"], ["missing"])
        self.assertEqual(overall["accepted_draft_tokens"]["count"], 1)
        self.assertEqual(overall["accepted_draft_tokens"]["sum"], 0)
        self.assertEqual(overall["acceptance_rate"]["count"], 0)
        self.assertIsNone(overall["observed_acceptance_rate"])

    def test_shadow_and_multiple_logs_per_run(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent) as temp:
            root = Path(temp)
            logs = []
            for index in (0, 1):
                path = root / f"rollback-{index}.log"
                path.write_text(line("ASR_FORMAL_STATS", formal(f"base-{index}")),
                                encoding="utf-8")
                logs.append(path)
            consume = root / "consume.log"
            consume.write_text(line("ASR_FORMAL_STATS", formal("target")), encoding="utf-8")
            shadow = root / "shadow.log"
            shadow.write_text(line("ASR_FORMAL_STATS", formal("shadow")), encoding="utf-8")
            output = root / "gain.json"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--rollback-log", *map(str, logs),
                 "--consume-log", str(consume), "--shadow-log", str(shadow),
                 "--out", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["runs"]["rollback"]["all"]["formal_request_count"], 2)
            self.assertEqual(data["runs"]["shadow"]["all"]["formal_request_count"], 1)
            self.assertIn("rollback_to_shadow", data["comparison"])
            self.assertIn("shadow_to_consume", data["comparison"])

    def test_reused_python_state_id_counts_distinct_connections(self):
        rollback = [
            ("formal", formal("stream_281470479549744_0", ws_id="ws-71ab1850",
                              seg=1, output=9)),
            ("formal", formal("stream_281470479549744_0", ws_id="ws-1cf924ed",
                              seg=6, output=12)),
        ]
        consume = [("formal", formal("consume"))]
        result, data = self.run_cli(rollback, consume)
        self.assertEqual(result.returncode, 0, result.stderr)
        records = data["runs"]["rollback"]["records"]
        self.assertEqual(len(records), 2)
        self.assertEqual({(row["ws_id"], row["seg"]) for row in records},
                         {("ws-71ab1850", 1), ("ws-1cf924ed", 6)})
        self.assertEqual(data["runs"]["rollback"]["all"]
                         ["output_token_count"]["mean"], 10.5)

    def test_reused_request_id_joins_verifier_to_its_later_formal_completion(self):
        rollback = [("formal", formal("base"))]
        reused = "stream_281470479549744_0"
        consume = [
            ("verify", verify(reused, 0, source=None, effective=3, accepted=2),
             "2026-10-26 07:05:17.600"),
            ("formal", formal(reused, ws_id="ws-first", seg=1,
                              source="prespec", draft=3),
             "2026-10-26 07:05:17.670"),
            ("verify", verify(reused, 0, source=None, effective=4, accepted=1),
             "2026-10-26 07:05:25.200"),
            ("formal", formal(reused, ws_id="ws-second", seg=6,
                              source="rollback", draft=4),
             "2026-10-26 07:05:25.326"),
        ]
        result, data = self.run_cli(rollback, consume,
                                    allow_legacy_time_join=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        records = data["runs"]["consume"]["records"]
        self.assertEqual(len(records), 2)
        by_ws = {row["ws_id"]: row for row in records}
        self.assertEqual(by_ws["ws-first"]["accepted_draft_tokens"], 2)
        self.assertEqual(by_ws["ws-second"]["accepted_draft_tokens"], 1)
        self.assertEqual(data["runs"]["consume"]["all"]
                         ["observed_acceptance_rate"], 3 / 7)

    def test_reused_id_with_verifier_requires_explicit_legacy_join(self):
        reused = "stream_281470479549744_0"
        rollback = [("formal", formal("base"))]
        consume = [
            ("verify", verify(reused, 0, source=None, effective=3, accepted=2),
             "2026-10-26 07:05:17.600"),
            ("formal", formal(reused, ws_id="ws-first", seg=1,
                              source="prespec", draft=3),
             "2026-10-26 07:05:17.670"),
            ("formal", formal(reused, ws_id="ws-second", seg=6,
                              source="prespec", draft=3),
             "2026-10-26 07:05:25.326"),
        ]
        result, data = self.run_cli(rollback, consume)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("allow-legacy-time-join", result.stderr)
        self.assertIsNone(data)

    def test_verifier_source_can_be_inherited_after_formal_event(self):
        rollback = [("formal", formal("base"))]
        consume = [("verify", verify("target", 0, source=None,
                                      effective=3, accepted=2)),
                   ("formal", formal("target", source="prespec", draft=3))]
        result, data = self.run_cli(rollback, consume)
        self.assertEqual(result.returncode, 0, result.stderr)
        record = data["runs"]["consume"]["records"][0]
        self.assertEqual(record["draft_source"], "prespec")
        self.assertEqual(record["accepted_draft_tokens"], 2)

    def test_nonfinal_step_allows_null_actual_effective(self):
        first = verify("target", 0, source=None, effective=3,
                       accepted=2, complete=False)
        first["actual_effective_draft_tokens"] = None
        rollback = [("formal", formal("base"))]
        consume = [("formal", formal("target", source="prespec", draft=5)),
                   ("verify", first),
                   ("verify", verify("target", 1, source=None, effective=2,
                                      accepted=1, actual_effective=5))]
        result, data = self.run_cli(rollback, consume)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(data["runs"]["consume"]["records"][0]
                         ["accepted_draft_tokens"], 3)

    def test_explicit_zero_effective_skip_is_distinct_from_missing(self):
        rollback = [("formal", formal("base"))]
        consume = [("formal", formal("missing", source="prespec", draft=5)),
                   ("formal", formal("skipped", source="prespec", draft=5)),
                   ("verify", verify("skipped", 0, source=None,
                                      effective=0, accepted=0,
                                      reason="spec_unsupported"))]
        result, data = self.run_cli(rollback, consume)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = data["runs"]["consume"]["all"]
        self.assertEqual(summary["missing_verifier_count"], 1)
        self.assertEqual(summary["explicit_zero_effective_count"], 1)
        self.assertEqual(summary["explicit_zero_effective_request_ids"], ["skipped"])
        records = {row["request_id"]: row for row in data["runs"]["consume"]["records"]}
        self.assertEqual(records["missing"]["verifier_status"], "missing")
        self.assertIsNone(records["missing"]["accepted_draft_tokens"])
        self.assertEqual(records["skipped"]["verifier_status"], "zero_effective")
        self.assertEqual(records["skipped"]["verifier_skip_reasons"],
                         ["spec_unsupported"])
        self.assertEqual(records["skipped"]["effective_draft_tokens"], 0)
        self.assertEqual(summary["acceptance_rate"]["count"], 0)

    def test_partial_verifier_logs_do_not_understate_accepted_counts(self):
        rollback = [("formal", formal("base"))]
        consume = [("formal", formal("no-end", source="prespec", draft=7)),
                   ("verify", verify("no-end", 0, effective=4, accepted=3,
                                      complete=False)),
                   ("formal", formal("mismatch", source="prespec", draft=7)),
                   ("verify", verify("mismatch", 0, effective=4, accepted=3,
                                      actual_effective=7))]
        result, data = self.run_cli(rollback, consume)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = data["runs"]["consume"]["all"]
        self.assertEqual(summary["incomplete_verifier_count"], 2)
        self.assertEqual(summary["missing_verifier_count"], 0)
        self.assertEqual(summary["accepted_draft_tokens"]["count"], 0)
        self.assertIsNone(summary["observed_acceptance_rate"])
        records = {row["request_id"]: row for row in data["runs"]["consume"]["records"]}
        self.assertEqual(records["no-end"]["verifier_status"], "incomplete")
        self.assertEqual(records["mismatch"]["verifier_status"], "incomplete")
        self.assertIsNone(records["mismatch"]["accepted_draft_tokens"])
        self.assertEqual(records["mismatch"]["logged_accepted_draft_tokens"], 3)

    def test_observed_acceptance_rate_excludes_partial_requests(self):
        rollback = [("formal", formal("base"))]
        consume = [("formal", formal("partial", source="prespec", draft=5)),
                   ("verify", verify("partial", 0, effective=5, accepted=4,
                                      complete=False)),
                   ("formal", formal("complete", source="prespec", draft=5)),
                   ("verify", verify("complete", 0, effective=5, accepted=2))]
        result, data = self.run_cli(rollback, consume)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = data["runs"]["consume"]["all"]
        self.assertEqual(summary["incomplete_verifier_count"], 1)
        self.assertEqual(summary["accepted_draft_tokens"]["sum"], 2)
        self.assertEqual(summary["effective_draft_tokens"]["sum"], 5)
        self.assertEqual(summary["observed_acceptance_rate"], 0.4)

    def test_zero_output_and_absent_timings_remain_unknown(self):
        rollback = [("formal", formal("base"))]
        consume = [("formal", formal("zero", output=0, rounds=0,
                                      prefill_ms=None, decode_total_ms=None))]
        result, data = self.run_cli(rollback, consume)
        self.assertEqual(result.returncode, 0, result.stderr)
        overall = data["runs"]["consume"]["all"]
        self.assertEqual(overall["output_token_count"]["mean"], 0)
        self.assertEqual(overall["decode_rounds"]["mean"], 0)
        self.assertEqual(overall["tokens_per_round"]["count"], 0)
        self.assertEqual(overall["prefill_ms"]["count"], 0)
        self.assertIsNone(overall["prefill_ms"]["mean"])
        self.assertEqual(overall["accepted_draft_tokens"]["sum"], 0)
        self.assertIsNone(data["comparison"]["rollback_to_consume"]["all"]
                          ["prefill_ms"]["mean_delta"])

    def test_chunk_finish_groups_and_p95_differences(self):
        rollback = [("formal", formal("base-chunk", output=4)),
                    ("formal", formal("base-finish", output=12,
                                      boundary="finish", case_id="audio-2"))]
        consume = [("formal", formal("target-chunk", output=7)),
                   ("formal", formal("target-finish", output=9,
                                     boundary="finish", case_id="audio-2"))]
        result, data = self.run_cli(rollback, consume)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(data["runs"]["consume"]["all"]["case_id_count"], 1)
        comparison = data["comparison"]["rollback_to_consume"]
        self.assertEqual(comparison["by_boundary"]["chunk"]
                         ["output_token_count"]["mean_delta"], 3)
        self.assertEqual(comparison["by_boundary"]["finish"]
                         ["output_token_count"]["mean_delta"], -3)
        self.assertAlmostEqual(comparison["all"]["output_token_count"]
                               ["p95_delta"], -2.7)

    def test_rejects_duplicate_or_orphan_or_out_of_bounds_events(self):
        good = [("formal", formal("base"))]
        invalid_sets = [
            ([("formal", formal("target")), ("formal", formal("target"))],
             "duplicate formal"),
            ([("formal", formal("target", source="prespec", draft=5)),
              ("verify", verify("target", 0)),
              ("verify", verify("target", 0))], "duplicate verifier"),
            ([("verify", verify("orphan", 0))], "orphan verifier"),
            ([("formal", formal("target", source="prespec", draft=5)),
              ("verify", verify("target", 0, effective=2, accepted=3))],
             "accepted_draft_tokens"),
            ([("formal", formal("target", source="prespec", draft=5)),
              ("verify", verify("target", 0, effective=6, accepted=4))],
             "effective_draft_tokens"),
            ([("formal", formal("target", source="prespec", draft=5)),
              ("verify", verify("target", 0, effective=4, accepted=3,
                                actual_effective=6))],
             "actual_effective_draft_tokens"),
            ([("formal", formal("target", source="prespec", draft=5)),
              ("verify", verify("target", 0, source="rollback"))],
             "draft_source"),
            ([("formal", formal("target", source="prespec", draft=5)),
              ("verify", verify("target", 0, source=None,
                                effective=0, accepted=0))],
             "reason"),
            ([("formal", formal("target", source="prespec", draft=5)),
              ("verify", verify("target", 0, effective=3, accepted=2,
                                actual_effective=5)),
              ("verify", verify("target", 1, effective=2, accepted=1,
                                complete=False))],
             "final verifier round_index"),
        ]
        for rows, message in invalid_sets:
            with self.subTest(message=message):
                result, data = self.run_cli(good, rows)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr.lower())
                self.assertIsNone(data)

    def test_rejects_missing_formal_instrumentation(self):
        result, data = self.run_cli([], [("formal", formal("target"))])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no asr_formal_stats", result.stderr.lower())
        self.assertIsNone(data)


if __name__ == "__main__":
    unittest.main()
