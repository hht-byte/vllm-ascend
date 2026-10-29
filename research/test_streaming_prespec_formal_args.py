import unittest

from streaming_prespec_formal_args import (
    formal_external_draft_extra_args,
    formal_rollback_draft_tokens,
)
from streaming_prespec import AdmissionPolicy, Candidate, PreSpecCoordinator


class FormalArgsTests(unittest.TestCase):
    def test_candidate_full_draft_and_budget_contract(self):
        candidate = Candidate("req", "ws", 0, 0, 0, 16000,
                              (10, 20, 30), 0)
        self.assertEqual(candidate.formal_draft_token_ids, (10, 20, 30))
        with self.assertRaisesRegex(ValueError, "candidate budget must fit"):
            PreSpecCoordinator(lambda request, budget: (),
                               AdmissionPolicy(max_rollback_draft_tokens=2,
                                               max_candidate_tokens=3))

    def test_prespec_includes_first_token_and_policy(self):
        original = {"other": 123}
        args = formal_external_draft_extra_args(
            original, candidate_tokens=(10, 20, 30), rollback_tokens=(20,),
            rollback_full_tokens=(10, 20),
            kmax=3)
        self.assertEqual(args["ascend_external_draft"], [10, 20, 30])
        self.assertTrue(args["ascend_external_draft_first_verify"])
        self.assertEqual(args["ascend_external_draft_rejoin"], "ngram_1")
        self.assertTrue(args["ascend_external_draft_ngram_after_first_mismatch"])
        self.assertEqual(original, {"other": 123})

    def test_candidate_first_mismatch_ngram_can_be_disabled_independently(self):
        args = formal_external_draft_extra_args(
            None, candidate_tokens=(10, 20), rollback_tokens=(),
            rollback_full_tokens=(), kmax=3,
            prespec_first_mismatch_ngram=False,
        )
        self.assertEqual(args["ascend_external_draft"], [10, 20])
        self.assertTrue(args["ascend_external_draft_first_verify"])
        self.assertEqual(args["ascend_external_draft_rejoin"], "ngram_1")
        self.assertNotIn("ascend_external_draft_ngram_after_first_mismatch", args)

    def test_candidate_first_mismatch_ngram_requires_boolean(self):
        with self.assertRaisesRegex(TypeError, "prespec_first_mismatch_ngram must be a bool"):
            formal_external_draft_extra_args(
                None, candidate_tokens=(10, 20), rollback_tokens=(),
                rollback_full_tokens=(), kmax=3,
                prespec_first_mismatch_ngram="false",
            )

    def test_rollback_replaces_stale_candidate_and_first_verifies(self):
        args = formal_external_draft_extra_args(
            {"ascend_external_draft_first_verify": True,
             "ascend_external_draft_rejoin": "ngram_3",
             "ascend_external_draft_ngram_after_first_mismatch": True},
            candidate_tokens=None, rollback_tokens=(20, 30),
            rollback_full_tokens=(10, 20, 30), kmax=3)
        self.assertEqual(args["ascend_external_draft"], [10, 20, 30])
        self.assertTrue(args["ascend_external_draft_first_verify"])
        self.assertEqual(args["ascend_external_draft_rejoin"], "ngram_1")
        self.assertNotIn("ascend_external_draft_ngram_after_first_mismatch", args)

    def test_no_silent_candidate_truncation(self):
        with self.assertRaisesRegex(ValueError, "exceeds engine kmax"):
            formal_external_draft_extra_args(
                None, candidate_tokens=(1, 2, 3, 4), rollback_tokens=(),
                rollback_full_tokens=(), kmax=3)

    def test_full_formal_rollback_starts_at_first_unfixed_token(self):
        self.assertEqual(
            formal_rollback_draft_tokens((20, 21, 22), (), kmax=2),
            (20, 21),
        )
        self.assertEqual(
            formal_rollback_draft_tokens((10, 11, 20, 21, 22),
                                         (10, 11), kmax=3),
            (20, 21, 22),
        )

    def test_full_formal_rollback_rejects_misaligned_prefix(self):
        with self.assertRaisesRegex(ValueError, "stable prefix"):
            formal_rollback_draft_tokens((10, 11, 20), (10, 12), kmax=3)

    def test_no_candidate_verifies_full_rollback_from_first_token(self):
        args = formal_external_draft_extra_args(
            {"other": 123}, candidate_tokens=None,
            rollback_tokens=(21, 22),
            rollback_full_tokens=(20, 21, 22),
            kmax=3, rejoin_policy="ngram_2",
        )
        self.assertEqual(args["ascend_external_draft"], [20, 21, 22])
        self.assertTrue(args["ascend_external_draft_first_verify"])
        self.assertEqual(args["ascend_external_draft_rejoin"], "ngram_2")
        self.assertEqual(args["other"], 123)

    def test_disabled_rollback_first_verify_restores_legacy_suffix(self):
        args = formal_external_draft_extra_args(
            {"other": 123, "ascend_external_draft_first_verify": True,
             "ascend_external_draft_rejoin": "ngram_3"},
            candidate_tokens=None,
            rollback_tokens=(21, 22, 23),
            rollback_full_tokens=(20, 21, 22),
            kmax=3,
            rollback_first_verify=False,
        )
        self.assertEqual(args, {"other": 123,
                                "ascend_external_draft": [21, 22, 23]})

    def test_disabled_rollback_first_verify_does_not_change_candidate(self):
        args = formal_external_draft_extra_args(
            None, candidate_tokens=(30, 31),
            rollback_tokens=(21, 22),
            rollback_full_tokens=(20, 21, 22),
            kmax=3, rollback_first_verify=False,
        )
        self.assertEqual(args["ascend_external_draft"], [30, 31])
        self.assertTrue(args["ascend_external_draft_first_verify"])
        self.assertEqual(args["ascend_external_draft_rejoin"], "ngram_1")

    def test_disabled_rollback_first_verify_with_only_one_token_is_target_only(self):
        args = formal_external_draft_extra_args(
            None, candidate_tokens=None, rollback_tokens=(),
            rollback_full_tokens=(20,), kmax=3,
            rollback_first_verify=False,
        )
        self.assertEqual(args, {"ascend_external_draft": []})

    def test_rollback_first_verify_requires_boolean(self):
        with self.assertRaisesRegex(TypeError, "rollback_first_verify must be a bool"):
            formal_external_draft_extra_args(
                None, candidate_tokens=None, rollback_tokens=(21,),
                rollback_full_tokens=(20, 21), kmax=3,
                rollback_first_verify="false",
            )

    def test_candidate_takes_priority_over_full_rollback(self):
        args = formal_external_draft_extra_args(
            None, candidate_tokens=(30, 31),
            rollback_tokens=(21, 22), rollback_full_tokens=(20, 21, 22),
            kmax=3,
        )
        self.assertEqual(args["ascend_external_draft"], [30, 31])
        self.assertTrue(args["ascend_external_draft_first_verify"])

    def test_empty_full_rollback_keeps_target_only(self):
        args = formal_external_draft_extra_args(
            {"ascend_external_draft_first_verify": True},
            candidate_tokens=None, rollback_tokens=(),
            rollback_full_tokens=(), kmax=3,
        )
        self.assertEqual(args, {"ascend_external_draft": []})

    def test_full_rollback_must_extend_legacy_suffix(self):
        with self.assertRaisesRegex(ValueError, "full rollback draft"):
            formal_external_draft_extra_args(
                None, candidate_tokens=None, rollback_tokens=(21, 22),
                rollback_full_tokens=(20, 99, 22), kmax=3,
            )


if __name__ == "__main__":
    unittest.main()
