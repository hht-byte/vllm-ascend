"""CPU boundaries for the trained-weight verifier; no NPU result is simulated."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def verifier_module(test):
    try:
        import verify_asr_mtp_weights
    except ImportError:
        test.fail("trained-weight verifier is not implemented")
    return verify_asr_mtp_weights


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Torch")
class WeightVerifierTests(unittest.TestCase):
    def setUp(self):
        import torch
        from asr_mtp_heads import BatchedMTPHeads
        self.t = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        config = dict(hidden_size=8, intermediate_size=16, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=4, attention_bias=False,
                      rms_norm_eps=1e-6, hidden_act="silu", rope_theta=10000,
                      rope_scaling=None, vocab_size=64)
        self.cfg = SimpleNamespace(**config)
        (self.model / "config.json").write_text(json.dumps(
            {"thinker_config": {"text_config": config}}), encoding="utf-8")
        self.keys = ("thinker.model.embed_tokens.weight", "thinker.lm_head.weight")
        self.index = {key: "weights.safetensors" for key in self.keys}
        self.write_index()
        (self.model / "weights.safetensors").touch()
        self.weights = BatchedMTPHeads(self.cfg, 5).state_dict()
        self.checkpoint = self.root / "inference.pt"
        torch.save({"metadata": {"num_heads": 5, "base_model": "PRIVATE_PATH"},
                    "mtp": {**self.weights, "backbone.private": torch.zeros(1)}}, self.checkpoint)

    def write_index(self):
        (self.model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": self.index}), encoding="utf-8")

    def fake_safe_open(self, path, *, framework, device):
        self.assertEqual(Path(path).parent, self.model)
        self.assertEqual((framework, device), ("pt", "cpu"))
        tensors = {key: self.t.arange(512).reshape(64, 8).float() for key in self.keys}
        return contextlib.nullcontext(SimpleNamespace(get_tensor=tensors.__getitem__))

    def test_restricted_mmap_load_selects_all_five_heads_and_strictly_loads(self):
        api = verifier_module(self)
        with patch.dict("sys.modules", {"safetensors": SimpleNamespace(safe_open=self.fake_safe_open)}), \
             patch.object(self.t, "load", wraps=self.t.load) as load:
            cfg, weights, embedding, logits = api._load_inputs(self.checkpoint, self.model)
        self.assertEqual(load.call_args.kwargs,
                         {"map_location": "cpu", "weights_only": True, "mmap": True})
        self.assertEqual(set(weights), set(self.weights))
        heads = api._make_heads(cfg, weights, device="cpu", dtype=self.t.float32)
        self.assertEqual(heads.num_heads, 5)
        self.t.testing.assert_close(heads.state_dict()["blocks.4.fusion.weight"],
                                    self.weights["blocks.4.fusion.weight"])
        self.assertEqual((embedding.shape, logits.shape), ((64, 8), (64, 8)))
        del weights["blocks.4.fusion.weight"]
        with self.assertRaises(RuntimeError):
            api._make_heads(cfg, weights, device="cpu", dtype=self.t.float32)

    def test_index_cannot_read_a_shard_outside_model_directory(self):
        api = verifier_module(self)
        self.index[self.keys[0]] = "../private.safetensors"
        self.write_index()
        with patch.dict("sys.modules", {"safetensors": SimpleNamespace(safe_open=self.fake_safe_open)}):
            with self.assertRaises(ValueError):
                api._load_inputs(self.checkpoint, self.model)

    def test_cli_projects_success_and_provenance_without_private_values(self):
        api = verifier_module(self)
        output = self.root / "result.json"
        checks = dict(initial_proposal=True, partial_rejection=True, confirmed_cache=True,
                      tokens=[123], checkpoint_path="PRIVATE_PATH")
        with patch.object(api, "verify", return_value=checks):
            code = api.main(["--checkpoint", str(self.checkpoint), "--model", str(self.model),
                             "--output", str(output), "--checkpoint-commit", "a" * 40,
                             "--training-checkpoint-sha256", "b" * 64])
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertTrue(result["passed"])
        self.assertEqual(result["scope"], "synthetic_component")
        self.assertEqual(result["input_checkpoint_sha256"], api._sha256(self.checkpoint))
        self.assertEqual(result["provided_provenance"]["training_checkpoint_sha256"], "b" * 64)
        self.assertNotIn("PRIVATE_PATH", output.read_text())
        self.assertEqual(set(result["checks"]),
                         {"initial_proposal", "partial_rejection", "confirmed_cache"})

    def test_cli_failure_never_exposes_exception_text(self):
        api = verifier_module(self)
        output, stderr = self.root / "failed.json", io.StringIO()
        with patch.object(api, "verify", side_effect=RuntimeError("PRIVATE_PATH secret tokens")), \
             contextlib.redirect_stderr(stderr):
            code = api.main(["--checkpoint", str(self.checkpoint), "--model", str(self.model),
                             "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(output.read_text())["passed"])
        self.assertNotIn("PRIVATE_PATH", output.read_text() + stderr.getvalue())
        self.assertNotIn("secret", output.read_text() + stderr.getvalue())

    def test_cli_output_cannot_overwrite_checkpoint_or_model_files(self):
        api = verifier_module(self)
        for output in (self.checkpoint, self.model / "config.json"):
            with self.subTest(output=output.name), contextlib.redirect_stderr(io.StringIO()):
                before = output.read_bytes()
                code = api.main(["--checkpoint", str(self.checkpoint), "--model", str(self.model),
                                 "--output", str(output)])
                self.assertEqual(code, 1)
                self.assertEqual(output.read_bytes(), before)

    def test_existing_attempt_report_is_preserved_without_reexecution(self):
        api = verifier_module(self)
        output = self.root / 'prior-attempt.json'
        output.write_text('{"passed":false}', encoding='utf-8')
        with patch.object(api, 'verify', return_value=dict.fromkeys(api.CHECKS, True)) as verify, \
             contextlib.redirect_stderr(io.StringIO()):
            code = api.main(['--checkpoint', str(self.checkpoint), '--model', str(self.model),
                             '--output', str(output)])
        self.assertEqual(code, 1)
        self.assertEqual(output.read_text(), '{"passed":false}')
        verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
