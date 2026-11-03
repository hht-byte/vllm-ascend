"""Streaming replay retains each arm's own rollback draft."""

import unittest

from mtp_streaming_replay import replay_utterance


class ReplayTests(unittest.TestCase):
    def test_third_chunk_verifies_previous_unfixed_tail(self):
        utterance = {
            "dataset": "pt", "utterance_id": "u", "reference_text": "",
            "audio": "audio.wav", "duration_seconds": 6.0,
            "chunks": [{"chunk_index": i, "audio_end_seconds": 2.0 * (i + 1)}
                       for i in range(3)],
        }
        seen = []

        def decode(record, arm, prefix, rollback):
            seen.append((record["chunk_index"], arm, prefix, rollback))
            ids = ((10, 11, 99), (10, 11, 12, 99), (11, 12, 13, 99))
            return {"token_ids": list(ids[record["chunk_index"]])}

        row = replay_utterance(utterance, decode, lambda ids: str(ids),
                               {99}, rollback_tokens=2, unfixed_chunks=2)
        self.assertEqual(seen, [
            (0, "baseline", (), ()), (0, "mtp", (), ()),
            (1, "baseline", (), ()), (1, "mtp", (), ()),
            (2, "baseline", (10,), (11, 12)),
            (2, "mtp", (10,), (11, 12)),
        ])
        self.assertEqual(row["mtp"]["token_ids"], [10, 11, 12, 13])
        self.assertTrue(row["token_exact_match"])

    def test_arms_keep_separate_actual_drafts_after_divergence(self):
        utterance = {
            "dataset": "pt", "utterance_id": "u", "reference_text": "",
            "audio": "audio.wav", "duration_seconds": 4.0,
            "chunks": [{"chunk_index": i, "audio_end_seconds": 2.0 * (i + 1)}
                       for i in range(2)],
        }
        seen = []

        def decode(record, arm, prefix, rollback):
            seen.append((arm, record["chunk_index"], prefix, rollback))
            if record["chunk_index"] == 0:
                return {"token_ids": [1, 2, 99] if arm == "baseline" else [1, 3, 99]}
            return {"token_ids": [2, 4, 99] if arm == "baseline" else [3, 4, 99]}

        row = replay_utterance(utterance, decode, lambda ids: str(ids), {99},
                               rollback_tokens=1, unfixed_chunks=1)
        self.assertEqual(seen[-2:], [
            ("baseline", 1, (1,), (2,)), ("mtp", 1, (1,), (3,)),
        ])
        self.assertEqual(row["baseline"]["token_ids"], [1, 2, 4])
        self.assertEqual(row["mtp"]["token_ids"], [1, 3, 4])
        self.assertFalse(row["token_exact_match"])


if __name__ == "__main__":
    unittest.main()
