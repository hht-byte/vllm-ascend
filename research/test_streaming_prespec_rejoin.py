"""Offline candidate rejoin estimates from PRESPEC_MATCH_TOKENS logs."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from research.streaming_prespec_rejoin import (
    analyze_record, parse_match_line, summarize,
)


SCRIPT = Path(__file__).with_name("streaming_prespec_rejoin.py")


def log_line(candidate, target, *, target_len=None, mode="consume"):
    if target_len is None:
        target_len = len(target)
    raw_match = 0
    for left, right in zip(candidate, target):
        if left != right:
            break
        raw_match += 1
    return (
        "7635:2026-10-26 07:28:37.805 - INFO - PRESPEC_MATCH_TOKENS "
        "ws_id=ws-191306f2 seg=3 window=2 mode=%s "
        "candidate_token_ids=%s target_token_ids=%s target_len=%d "
        "raw_matched_prefix_tokens=%d\n"
        % (mode, tuple(candidate), tuple(target), target_len, raw_match)
    )


class RejoinTest(unittest.TestCase):
    def test_user_sample_only_reuses_terminal_eos_without_step_saving(self):
        candidate = (43424, 79555, 8452, 15131, 949, 84110, 409,
                     64066, 13, 151645)
        target = (43424, 79555, 8452, 15131, 949, 84110, 409,
                  138188, 30, 151645)
        record = parse_match_line(log_line(candidate, target))
        result = analyze_record(record, kmax=15, eos_token_id=151645)
        self.assertEqual(result["first_mismatch_index"], 7)
        self.assertEqual(result["initial_draft_matched"], 6)
        self.assertEqual(result["later_same_position_tokens"], 1)
        self.assertEqual(result["later_same_position_nonterminal_tokens"], 0)
        aligned = result["policies"]["aligned"]
        self.assertEqual(aligned["attempts"], 2)
        self.assertEqual(aligned["zero_accept_attempts"], 1)
        self.assertEqual(aligned["accepted_tokens"], 1)
        self.assertEqual(aligned["accepted_nonterminal_tokens"], 0)
        self.assertEqual(aligned["estimated_saved_steps"], 0)
        for n in (1, 2, 3):
            self.assertEqual(result["policies"][f"ngram_{n}"]["attempts"], 0)

    def test_ngram_can_rejoin_after_confirmed_anchor(self):
        candidate = (1, 2, 8, 7, 4, 5, 6)
        target = (1, 2, 9, 4, 5, 6)
        result = analyze_record(parse_match_line(log_line(candidate, target)),
                                kmax=15, eos_token_id=151645)
        self.assertEqual(result["policies"]["aligned"]["accepted_tokens"], 0)
        ngram = result["policies"]["ngram_1"]
        self.assertEqual(ngram["attempts"], 1)
        self.assertEqual(ngram["accepted_tokens"], 2)
        self.assertEqual(ngram["estimated_saved_steps"], 1)
        self.assertEqual(result["policies"]["ngram_2"]["accepted_tokens"], 1)
        self.assertEqual(result["policies"]["ngram_3"]["attempts"], 0)

    def test_first_token_difference_is_already_handled_by_initial_draft(self):
        result = analyze_record(
            parse_match_line(log_line((101, 2, 3, 4), (102, 2, 3, 4))),
            kmax=15, eos_token_id=151645)
        self.assertEqual(result["initial_draft_matched"], 3)
        self.assertIsNone(result["first_mismatch_index"])
        self.assertIsNone(result["policies"])

    def test_truncated_target_replays_aligned_but_not_ngram(self):
        result = analyze_record(
            parse_match_line(log_line((1, 2, 3), (1, 9, 3), target_len=5)),
            kmax=15, eos_token_id=151645)
        self.assertTrue(result["censored"])
        self.assertEqual(result["skip_reason"], "censored_target")
        self.assertEqual(result["policies"]["aligned"]["estimated_saved_steps"], 1)
        self.assertIsNone(result["policies"]["ngram_1"])
        self.assertIsNone(result["policies"]["ngram_2"])
        self.assertIsNone(result["policies"]["ngram_3"])

    def test_user_censored_case_has_zero_incremental_aligned_gain(self):
        result = analyze_record(
            parse_match_line(log_line((1, 2, 3, 4, 5, 6),
                                      (1, 2, 3, 4, 9, 8), target_len=12)),
            kmax=15, eos_token_id=151645)
        self.assertEqual(result["initial_draft_matched"], 3)
        self.assertEqual(result["first_mismatch_index"], 4)
        aligned = result["policies"]["aligned"]
        self.assertEqual(aligned["attempts"], 1)
        self.assertEqual(aligned["zero_accept_attempts"], 1)
        self.assertEqual(aligned["accepted_tokens"], 0)
        self.assertEqual(aligned["estimated_model_steps"], 7)
        self.assertEqual(aligned["estimated_saved_steps"], 0)

    def test_censored_aligned_replay_counts_unknown_bonus_token(self):
        result = analyze_record(
            parse_match_line(log_line((1, 2, 3, 4, 5, 6),
                                      (1, 2, 3, 4, 9, 6), target_len=12)),
            kmax=15, eos_token_id=151645)
        aligned = result["policies"]["aligned"]
        self.assertEqual(aligned["accepted_tokens"], 1)
        self.assertEqual(aligned["autoregressive_steps"], 5)
        self.assertEqual(aligned["estimated_model_steps"], 6)
        self.assertEqual(aligned["estimated_saved_steps"], 1)

    def test_summary_reports_exclusive_replay_skip_reasons(self):
        examples = (
            ((1, 2, 3), (1, 9, 3), 5),  # censored first
            ((1, 2, 3), (1, 2, 3), 3),  # no post-first-draft mismatch
            ((1, 2, 3, 4), (1, 9, 3, 4), 4),  # draft exceeds kmax
            ((1, 2), (1, 9), 2),  # replayed
        )
        rows = [analyze_record(parse_match_line(log_line(c, t, target_len=length)),
                               kmax=2, eos_token_id=151645)
                for c, t, length in examples]
        summary = summarize(rows)
        self.assertEqual(summary["replayed_count"], 1)
        self.assertEqual(summary["policies"]["aligned"]["evaluated_request_count"], 2)
        self.assertEqual(summary["policies"]["ngram_1"]["evaluated_request_count"], 1)
        self.assertEqual(summary["excluded_replay_reasons"], {
            "censored_target": 1,
            "no_post_first_draft_mismatch": 1,
            "candidate_draft_exceeds_kmax": 1,
        })

    def test_cli_counts_complete_and_censored_separately(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent) as temp:
            source = Path(temp) / "server.log"
            output = Path(temp) / "rejoin.json"
            source.write_text(
                "unrelated\n"
                + log_line((1, 2, 3, 4), (1, 2, 9, 4))
                + log_line((101, 2, 3), (102, 2, 3), target_len=5),
                encoding="utf-8")
            run = subprocess.run([sys.executable, str(SCRIPT), "--log",
                                  str(source), "--out", str(output)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["match_events"], 2)
            self.assertEqual(data["complete_target_count"], 1)
            self.assertEqual(data["censored_target_count"], 1)
            self.assertEqual(data["post_first_draft_mismatch_count"], 1)

    def test_optional_partial_candidate_flag_and_grouping(self):
        candidate = (1, 2, 3, 4)
        target = (1, 2, 9, 4)
        old_line = log_line(candidate, target)
        new_line = old_line.rstrip("\n") + " candidate_partial=True\n"
        full_line = old_line.rstrip("\n") + " candidate_partial=False\n"
        self.assertIsNone(parse_match_line(old_line)["candidate_partial"])
        self.assertIs(parse_match_line(new_line)["candidate_partial"], True)
        self.assertIs(parse_match_line(full_line)["candidate_partial"], False)
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent) as temp:
            source = Path(temp) / "server.log"
            output = Path(temp) / "rejoin.json"
            source.write_text(old_line + new_line + full_line,
                              encoding="utf-8")
            run = subprocess.run([sys.executable, str(SCRIPT), "--log",
                                  str(source), "--out", str(output)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["by_candidate_partial"]["true"]["match_events"], 1)
            self.assertEqual(data["by_candidate_partial"]["unknown"]["match_events"], 1)
            self.assertEqual(data["by_candidate_partial"]["false"]["match_events"], 1)


if __name__ == "__main__":
    unittest.main()
