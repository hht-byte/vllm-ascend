import unittest

from streaming_prespec_formal_args import formal_external_draft_extra_args
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
            kmax=3)
        self.assertEqual(args["ascend_external_draft"], [10, 20, 30])
        self.assertTrue(args["ascend_external_draft_first_verify"])
        self.assertEqual(args["ascend_external_draft_rejoin"], "ngram_1")
        self.assertEqual(original, {"other": 123})

    def test_rollback_clears_stale_prespec_flags(self):
        args = formal_external_draft_extra_args(
            {"ascend_external_draft_first_verify": True,
             "ascend_external_draft_rejoin": "ngram_3"},
            candidate_tokens=None, rollback_tokens=(20, 30), kmax=3)
        self.assertEqual(args, {"ascend_external_draft": [20, 30]})

    def test_no_silent_candidate_truncation(self):
        with self.assertRaisesRegex(ValueError, "exceeds engine kmax"):
            formal_external_draft_extra_args(
                None, candidate_tokens=(1, 2, 3, 4), rollback_tokens=(), kmax=3)


if __name__ == "__main__":
    unittest.main()
