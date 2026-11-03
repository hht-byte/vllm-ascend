"""Rollback-first MTP handoff with a deterministic target backend."""

import unittest

from mtp_rollback_handoff import Verification, confirm_draft, run_handoff


class SequenceBackend:
    def __init__(self, target, proposals):
        self.target = tuple(target)
        self.proposals = list(proposals)
        self.confirmed = []
        self.events = []

    def verify(self, draft):
        self.events.append(("verify", tuple(draft), tuple(self.confirmed)))
        expected = self.target[len(self.confirmed):]
        accepted = 0
        for offered, actual in zip(draft, expected):
            if offered != actual:
                break
            accepted += 1
        tokens = expected[:accepted if accepted == len(draft) else accepted + 1]
        self.confirmed.extend(tokens)
        return Verification(tuple(tokens), accepted, 1)

    def propose_mtp(self, max_heads):
        self.events.append(("mtp", max_heads, tuple(self.confirmed)))
        return self.proposals.pop(0)


class HandoffTests(unittest.TestCase):
    def test_target_correction_replaces_first_bad_rollback_token(self):
        self.assertEqual(confirm_draft((10, 70, 80), (10, 11, 12)),
                         ((10, 11), 1))
        self.assertEqual(confirm_draft((70, 80), (10, 11)), ((10,), 0))
        self.assertEqual(confirm_draft((10, 11), (10, 11)), ((10, 11), 2))

    def test_full_rollback_is_verified_before_mtp_uses_confirmed_context(self):
        backend = SequenceBackend((10, 11, 12, 13, 14, 99), ((12, 77), (14, 99)))
        result = run_handoff(backend, (10, 11), max_new_tokens=8, kmax=3,
                             draft_heads=2, eos_token_ids={99})
        self.assertEqual(result["token_ids"], [10, 11, 12, 13, 14, 99])
        self.assertEqual(result["rollback_accepted_tokens"], 2)
        self.assertEqual(result["mtp_accepted_draft_tokens"], 1)
        self.assertEqual(backend.events, [
            ("verify", (10, 11), ()),
            ("mtp", 2, (10, 11)),
            ("verify", (12, 77), (10, 11)),
            ("mtp", 2, (10, 11, 12, 13)),
            ("verify", (14, 99), (10, 11, 12, 13)),
        ])

    def test_rollback_mismatch_hands_corrected_token_to_mtp(self):
        backend = SequenceBackend((10, 11, 12, 99), ((12, 99),))
        result = run_handoff(backend, (10, 70, 80), max_new_tokens=6,
                             kmax=4, draft_heads=2, eos_token_ids={99})
        self.assertEqual(result["token_ids"], [10, 11, 12, 99])
        self.assertEqual(result["rollback_accepted_tokens"], 1)
        self.assertEqual(backend.events[1], ("mtp", 2, (10, 11)))

    def test_no_rollback_starts_with_mtp_and_eos_stops_it(self):
        backend = SequenceBackend((5, 99), ((5, 99),))
        result = run_handoff(backend, (), max_new_tokens=4, kmax=3,
                             draft_heads=2, eos_token_ids={99})
        self.assertEqual(result["token_ids"], [5, 99])
        self.assertEqual(result["rollback_offered_tokens"], 0)
        self.assertEqual(backend.events[0], ("mtp", 2, ()))

    def test_rollback_eos_finishes_without_mtp(self):
        backend = SequenceBackend((10, 99), ())
        result = run_handoff(backend, (10, 99, 7), max_new_tokens=6,
                             kmax=3, draft_heads=2, eos_token_ids={99})
        self.assertEqual(result["token_ids"], [10, 99])
        self.assertEqual(len(backend.events), 1)

    def test_rejects_draft_over_engine_budget(self):
        backend = SequenceBackend((1, 2, 3), ())
        with self.assertRaisesRegex(ValueError, "kmax"):
            run_handoff(backend, (1, 2, 3), max_new_tokens=6,
                        kmax=2, draft_heads=1, eos_token_ids={99})
        self.assertEqual(backend.events, [])

    def test_rejects_unverified_backend_tail(self):
        class BrokenBackend(SequenceBackend):
            def verify(self, draft):
                return Verification((draft[0], 999, 998), 1, 1)

        with self.assertRaisesRegex(ValueError, "verification"):
            run_handoff(BrokenBackend((1, 2, 3), ()), (1, 2),
                        max_new_tokens=5, kmax=2, draft_heads=1,
                        eos_token_ids={99})


if __name__ == "__main__":
    unittest.main()
