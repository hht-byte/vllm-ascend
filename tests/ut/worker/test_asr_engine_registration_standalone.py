# SPDX-License-Identifier: Apache-2.0
"""Execute the worker's late registration blocks without requiring an NPU."""

import ast
import builtins
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class LateASRRegistrationTests(unittest.TestCase):
    def run_entry(self, supported):
        tree = ast.parse((ROOT / "vllm_ascend/worker/worker.py").read_text(encoding="utf-8"))
        runner_import = next(
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "vllm_ascend.worker.model_runner_v1"
        )
        worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUWorker")
        startup_end = next(
            node.lineno for node in tree.body if isinstance(node, ast.Expr) and node.lineno > runner_import.lineno
        )
        blocks = [
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and (runner_import.lineno < node.lineno < startup_end or node.lineno > worker.end_lineno)
        ]

        class Runner:
            pass

        class Worker:
            pass

        def base(cls):
            cls.external_draft = True

        def first_verify():
            self.assertTrue(getattr(Runner, "external_draft", False), "transport must precede metadata")
            Runner.first_verify = True

        def memory(cls):
            cls.mtp_memory = True

        modules = {
            "patch_external_draft": types.SimpleNamespace(install=base),
            "patch_external_draft_first_verify": types.SimpleNamespace(install=first_verify),
            "patch_external_draft_mtp": types.SimpleNamespace(install_worker_memory_hook=memory),
            "patch_full_graph_spec_buckets": types.SimpleNamespace(install_full_graph_spec_buckets=lambda cls: None),
        }

        def import_module(name, globals=None, locals=None, fromlist=(), level=0):
            if name.endswith("patch_external_draft_mtp"):
                self.assertTrue(getattr(Runner, "first_verify", False), "MTP requires the completed base install")
                Runner.mtp = True
            return modules[name.rsplit(".", 1)[-1]]

        namespace = {
            "__builtins__": {**vars(builtins), "__import__": import_module},
            "vllm_version_is": lambda version: supported,
            "NPUModelRunner": Runner,
            "NPUWorker": Worker,
        }
        exec(compile(ast.Module(body=blocks, type_ignores=[]), "worker-registration", "exec"), namespace)
        return Runner, Worker

    def test_v023_registers_transport_then_first_verify_then_mtp_and_memory(self):
        runner, worker = self.run_entry(True)
        self.assertTrue(getattr(runner, "external_draft", False))
        self.assertTrue(getattr(runner, "first_verify", False))
        self.assertTrue(getattr(runner, "mtp", False))
        self.assertTrue(getattr(worker, "mtp_memory", False))

    def test_other_versions_keep_their_original_worker(self):
        runner, worker = self.run_entry(False)
        self.assertFalse(hasattr(runner, "external_draft"))
        self.assertFalse(hasattr(runner, "mtp"))
        self.assertFalse(hasattr(worker, "mtp_memory"))


if __name__ == "__main__":
    unittest.main()
