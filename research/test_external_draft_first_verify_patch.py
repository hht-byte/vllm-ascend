"""Validate the generated 0.23 patch against the local reference trees."""

import ast
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

from external_draft_rejoin import first_mismatch, next_draft

from build_external_draft_first_verify_patch import (
    base_scheduler,
    make_patch,
    proposer,
    recompute_scheduler,
    worker_patch_registry,
)


VLLM = Path(r"E:\26B-multimodal\vllm")
ASCEND = Path(r"E:\26B-multimodal\opencode-project\asr-optimize\vllm-ascend")


@unittest.skipUnless(
    (VLLM / "vllm/v1/core/sched/scheduler.py").exists()
    and (ASCEND / "vllm_ascend/spec_decode/external_draft_proposer.py").exists(),
    "local 0.23 reference source unavailable",
)
class EnginePatchTests(unittest.TestCase):
    def test_first_verify_reads_bookkept_prefix_and_stops_after_failed_aligned_retry(self):
        source = proposer((ASCEND / "vllm_ascend/spec_decode/external_draft_proposer.py")
                          .read_text(encoding="utf-8"))
        cls = next(node for node in ast.parse(source).body
                   if isinstance(node, ast.ClassDef)
                   and node.name == "AscendExternalDraftProposer")
        method = next(node for node in cls.body
                      if isinstance(node, ast.FunctionDef) and node.name == "propose")
        method.decorator_list = []
        namespace = {
            "EXTERNAL_DRAFT_ATTR": "_external_draft_tokens",
            "EXTERNAL_DRAFT_FIRST_VERIFY_ATTR": "_external_draft_first_verify",
            "EXTERNAL_DRAFT_POLICIES_ATTR": "_external_draft_policies",
            "EXTERNAL_DRAFT_LAST_PROPOSAL_ATTR": "_external_draft_last_proposal",
            "EXTERNAL_DRAFT_ALIGNED_BLOCKED_ATTR": "_external_draft_aligned_blocked",
            "first_mismatch": first_mismatch,
            "next_draft": next_draft,
            "logger": types.SimpleNamespace(debug=lambda *args: None),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), "propose", "exec"),
             namespace)

        class TokenTable:
            def __init__(self):
                self.row = [101, 102, 103, 9, 0, 0, 0]

            def __getitem__(self, key):
                row, columns = key
                self_owner.assertEqual(row, 0)
                return self.row[columns]

            def __setitem__(self, key, value):
                self_owner.fail("first-verify path must not rewrite bookkept tokens")

        self_owner = self
        batch = types.SimpleNamespace(
            req_ids=["request"], spec_decode_unsupported_reqs=set(),
            num_tokens_no_spec=[4], num_prompt_tokens=[3], max_model_len=20,
            token_ids_cpu=TokenTable(),
            _external_draft_tokens={"request": [1, 2, 3, 4]},
            _external_draft_first_verify={"request"},
            _external_draft_policies={"request": "ngram_1"},
            _external_draft_last_proposal={},
            _external_draft_aligned_blocked=set(),
        )
        proposer_obj = types.SimpleNamespace(runner=types.SimpleNamespace(input_batch=batch), k=15)
        self.assertEqual(namespace["propose"](proposer_obj, [[9]]), [[2, 3, 4]])
        batch.num_tokens_no_spec[0] = 5
        batch.token_ids_cpu.row[4] = 8
        self.assertEqual(namespace["propose"](proposer_obj, [[8]]), [[]])
        self.assertIn("request", batch._external_draft_aligned_blocked)

        class LegacyTokenTable(TokenTable):
            def __setitem__(self, key, value):
                row, columns = key
                self.row[columns] = value

        batch.token_ids_cpu = LegacyTokenTable()
        batch._external_draft_first_verify.clear()
        batch._external_draft_tokens["request"] = [5, 6]
        self.assertEqual(namespace["propose"](proposer_obj, [[8]]), [[5, 6]])
        self.assertEqual(namespace["propose"](proposer_obj, [[8]]), [[]])

    def test_first_verify_accepts_scheduler_without_speculative_config_attribute(self):
        source = base_scheduler((VLLM / "vllm/v1/core/sched/scheduler.py")
                                .read_text(encoding="utf-8"))
        scheduler_class = next(node for node in ast.parse(source).body
                               if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
        helper = next(node for node in scheduler_class.body
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "_prepare_initial_external_draft")
        namespace = {"Request": object}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "helper", "exec"),
             namespace)
        scheduler = types.SimpleNamespace(num_spec_tokens=15, max_model_len=100)
        request = types.SimpleNamespace(
            sampling_params=types.SimpleNamespace(extra_args={
                "ascend_external_draft_first_verify": True,
                "ascend_external_draft": [4, 5],
            }),
            num_tokens=10,
            spec_token_ids=[],
        )
        namespace["_prepare_initial_external_draft"](scheduler, request)
        self.assertEqual(request.spec_token_ids, [4, 5])

    def test_changed_engine_files_parse(self):
        cases = (
            (VLLM / "vllm/v1/core/sched/scheduler.py", base_scheduler),
            (ASCEND / "vllm_ascend/core/recompute_scheduler.py", recompute_scheduler),
            (ASCEND / "vllm_ascend/spec_decode/external_draft_proposer.py", proposer),
            (ASCEND / "vllm_ascend/patch/worker/__init__.py", worker_patch_registry),
        )
        for path, transform in cases:
            with self.subTest(path=path):
                ast.parse(transform(path.read_text(encoding="utf-8")), filename=str(path))

    def test_patch_contains_first_schedule_and_rejoin(self):
        patch = make_patch(VLLM, ASCEND,
                           Path(__file__).with_name("external_draft_rejoin.py"))
        self.assertIn("status=scheduled draft_tokens", patch)
        self.assertIn("next_draft(", patch)
        self.assertIn("/dev/null", patch)

    def test_existing_monkey_patch_factory_and_runner_dispatch_are_preserved(self):
        patch = make_patch(VLLM, ASCEND,
                           Path(__file__).with_name("external_draft_rejoin.py"))
        self.assertNotIn("--- a/vllm_ascend/worker/model_runner_v1.py", patch)
        transformed = proposer(
            (ASCEND / "vllm_ascend/spec_decode/external_draft_proposer.py")
            .read_text(encoding="utf-8"))
        self.assertIn("class AscendExternalDraftProposer(AscendNgramProposer):",
                      transformed)
        runner = (ASCEND / "vllm_ascend/worker/model_runner_v1.py").read_text(
            encoding="utf-8")
        self.assertIn("isinstance(self.drafter, AscendNgramProposer)", runner)

    def test_first_verify_transport_layers_on_existing_monkey_patch(self):
        patch = make_patch(VLLM, ASCEND,
                           Path(__file__).with_name("external_draft_rejoin.py"))
        self.assertFalse("--- a/vllm_ascend/patch/worker/patch_external_draft.py" in patch)
        self.assertTrue("+++ b/vllm_ascend/patch/worker/patch_external_draft_first_verify.py" in patch)
        self.assertTrue("import vllm_ascend.patch.worker.patch_external_draft_first_verify" in patch)

    def test_patch_applies_to_reference_files(self):
        workspace_tmp = Path(__file__).parents[1] / ".tmp"
        workspace_tmp.mkdir(exist_ok=True)
        assert workspace_tmp.resolve().is_relative_to(Path(__file__).parents[1].resolve())
        with tempfile.TemporaryDirectory(dir=workspace_tmp) as tmp:
            root = Path(tmp)
            for source_root, name in (
                (VLLM, "vllm/v1/core/sched/scheduler.py"),
                (ASCEND, "vllm_ascend/core/recompute_scheduler.py"),
                (ASCEND, "vllm_ascend/spec_decode/external_draft_proposer.py"),
                (ASCEND, "vllm_ascend/patch/worker/__init__.py"),
            ):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                original = (source_root / name).read_text(encoding="utf-8")
                target.write_bytes(original.replace("\r\n", "\n").encode("utf-8"))
            patch = make_patch(VLLM, ASCEND,
                               Path(__file__).with_name("external_draft_rejoin.py"))
            patch_file = root / "first_verify.patch"
            patch_file.write_bytes(patch.encode("utf-8"))
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            for include in ("vllm/*", "vllm_ascend/*"):
                result = subprocess.run(
                    ["git", "apply", "--check", f"--include={include}",
                     str(patch_file)], cwd=root,
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                subprocess.run(
                    ["git", "apply", f"--include={include}", str(patch_file)],
                    cwd=root, check=True, capture_output=True, text=True,
                )
            for name in (
                "vllm/v1/core/sched/scheduler.py",
                "vllm_ascend/core/recompute_scheduler.py",
                "vllm_ascend/spec_decode/external_draft_proposer.py",
                "vllm_ascend/patch/worker/__init__.py",
                "vllm_ascend/patch/worker/patch_external_draft_first_verify.py",
                "vllm_ascend/spec_decode/external_draft_rejoin.py",
            ):
                ast.parse((root / name).read_text(encoding="utf-8"), filename=name)


if __name__ == "__main__":
    unittest.main()
