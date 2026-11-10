"""Native-head wiring must cover eager and both graph tensor programs."""

import unittest
from unittest.mock import patch

from test_asr_mtp_graph import TinyMTPFixture
from test_asr_mtp_token_graph import PackedFixture


class FusionPathChecks:
    def check_path(self, *, eager=False):
        from asr_mtp_fused import prepare_fused_heads
        from torch.nn import functional as F

        model = self.eager if eager else self.fixed
        model.requires_grad_(False)
        prepare_fused_heads(model)
        rows = [row + [4] for row in self.ids]
        limits = [3] * len(rows)
        with (
            self.t.inference_mode(),
            patch("torch.nn.functional.linear", wraps=F.linear) as linear,
        ):
            if eager:
                actual = self.eager.propose(self.a, rows, self.embed, self.logits, limits).clone()
            else:
                self.program.stage(self.b, rows, limits)
                actual = self.program.run()[: len(rows)].clone()
                self.program.commit(self.b, limits)
        # Fusion + packed QKV + O + packed gate/up + down + LM head.
        self.assertEqual(linear.call_count, 6 * model.num_heads)
        with self.t.inference_mode():
            if eager:
                self.program.stage(self.b, rows, limits)
                expected = self.program.run()[: len(rows)].clone()
                self.program.commit(self.b, limits)
            else:
                expected = self.eager.propose(self.a, rows, self.embed, self.logits, limits)
        self.t.testing.assert_close(actual, expected)
        # Reject all auxiliary tokens: only the former drive executes next.
        for left, right, row in zip(self.a, self.b, self.ids):
            old = len(row)
            row.append(4)
            hidden = self.t.randn(1, self.cfg.hidden_size)
            left.target[old : old + 1] = right.target[old : old + 1] = hidden
            left.length = right.length = len(row)
            left.crop_heads(len(row))
            right.crop_heads(len(row))
        self.compare(self.a, self.b, [row + [5] for row in self.ids], [3] * len(rows))


class RectangularFusionTests(FusionPathChecks, TinyMTPFixture, unittest.TestCase):
    def test_eager_uses_packed_heads_and_preserves_rejection_cache(self):
        self.check_path(eager=True)

    def test_rectangular_graph_uses_packed_heads_and_preserves_rejection_cache(self):
        self.check_path()


class PackedFusionTests(FusionPathChecks, PackedFixture, unittest.TestCase):
    def test_token_graph_uses_packed_heads_and_preserves_rejection_cache(self):
        self.check_path()


class FusionGraphOptionsTests(unittest.TestCase):
    def test_profiling_is_explicit_and_propagates_to_both_graph_layouts(self):
        from asr_mtp_graph import graph_options

        for extra in ({}, {"graph_token_sizes": [8, 32]}):
            settings = {"graph_mode": "full", **extra}
            options = graph_options(settings, context=128, slots=4, max_batch=4)
            self.assertFalse(options.profile_stages)
            options = graph_options(
                {**settings, "profile_stages": True},
                context=128,
                slots=4,
                max_batch=4,
            )
            self.assertTrue(options.profile_stages)
            with self.assertRaisesRegex(ValueError, "profile_stages"):
                graph_options(
                    {**settings, "profile_stages": 1},
                    context=128,
                    slots=4,
                    max_batch=4,
                )
