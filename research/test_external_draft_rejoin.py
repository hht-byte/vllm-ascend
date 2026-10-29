"""Request-local rejoin checks; no NPU runtime required."""

import unittest

from external_draft_rejoin import first_mismatch, next_draft


class RejoinTests(unittest.TestCase):
    def test_first_round_keeps_first_candidate_token(self):
        self.assertEqual(next_draft((1, 2, 3), (), max_draft_tokens=3), (1, 2, 3))

    def test_first_token_mismatch_uses_aligned_suffix(self):
        self.assertEqual(first_mismatch((1, 2, 3, 4), (9,)), 0)
        self.assertEqual(next_draft((1, 2, 3, 4), (9,),
                                    policy="ngram_3", max_draft_tokens=3),
                         (2, 3, 4))

    def test_first_token_mismatch_uses_ngram_after_aligned_retry(self):
        candidate = (1, 2, 3, 4, 5)
        self.assertEqual(next_draft(candidate, (9, 3), policy="ngram_1",
                                    max_draft_tokens=4,
                                    after_aligned_retry=True), (4, 5))

    def test_first_token_mismatch_without_ngram_anchor_returns_no_draft(self):
        candidate = (325, 5580, 26241, 2328, 151645)
        self.assertEqual(next_draft(candidate, (1514, 438), policy="ngram_1",
                                    max_draft_tokens=4,
                                    after_aligned_retry=True), ())

    def test_later_mismatch_uses_rightmost_ngram_anchor(self):
        candidate = (10, 20, 30, 40, 20, 30, 50, 60)
        confirmed = (10, 20, 30, 99, 20, 30)
        self.assertEqual(next_draft(candidate, confirmed, policy="ngram_2",
                                    max_draft_tokens=3), (50, 60))

    def test_ngram_1_default_and_ngram_3(self):
        candidate = (1, 2, 3, 4, 5, 6, 7, 8)
        confirmed = (1, 2, 3, 99, 4, 5, 6)
        self.assertEqual(next_draft(candidate, confirmed,
                                    max_draft_tokens=3), (7, 8))
        self.assertEqual(next_draft(candidate, confirmed, policy="ngram_3",
                                    max_draft_tokens=3), (7, 8))

    def test_later_mismatch_with_no_anchor_falls_back_to_autoregressive(self):
        self.assertEqual(next_draft((1, 2, 3), (1, 9),
                                    max_draft_tokens=3), ())

    def test_aligned_after_later_mismatch(self):
        self.assertEqual(next_draft((1, 2, 3, 4), (1, 9),
                                    policy="aligned", max_draft_tokens=2),
                         (3, 4))

    def test_limit_and_validation(self):
        self.assertEqual(next_draft((1, 2, 3, 4), (1,),
                                    max_draft_tokens=2), (2, 3))
        with self.assertRaises(ValueError):
            next_draft((1,), (), policy="ngram_4", max_draft_tokens=1)


if __name__ == "__main__":
    unittest.main()
