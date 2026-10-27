"""Validate the generated 0.23 patch against the local reference trees."""

import ast
import subprocess
import tempfile
import unittest
from pathlib import Path

from build_external_draft_first_verify_patch import (
    base_scheduler,
    make_patch,
    model_runner,
    proposer,
    recompute_scheduler,
    worker_patch,
)


VLLM = Path(r"E:\26B-multimodal\vllm")
ASCEND = Path(r"E:\26B-multimodal\opencode-project\asr-optimize\vllm-ascend")


@unittest.skipUnless(
    (VLLM / "vllm/v1/core/sched/scheduler.py").exists()
    and (ASCEND / "vllm_ascend/spec_decode/external_draft_proposer.py").exists(),
    "local 0.23 reference source unavailable",
)
class EnginePatchTests(unittest.TestCase):
    def test_changed_engine_files_parse(self):
        cases = (
            (VLLM / "vllm/v1/core/sched/scheduler.py", base_scheduler),
            (ASCEND / "vllm_ascend/core/recompute_scheduler.py", recompute_scheduler),
            (ASCEND / "vllm_ascend/spec_decode/external_draft_proposer.py", proposer),
            (ASCEND / "vllm_ascend/patch/worker/patch_external_draft.py", worker_patch),
            (ASCEND / "vllm_ascend/worker/model_runner_v1.py", model_runner),
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
                (ASCEND, "vllm_ascend/patch/worker/patch_external_draft.py"),
                (ASCEND, "vllm_ascend/worker/model_runner_v1.py"),
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
                "vllm_ascend/patch/worker/patch_external_draft.py",
                "vllm_ascend/worker/model_runner_v1.py",
                "vllm_ascend/spec_decode/external_draft_rejoin.py",
            ):
                ast.parse((root / name).read_text(encoding="utf-8"), filename=name)


if __name__ == "__main__":
    unittest.main()
