"""Exercise the optional worker patch without importing torch or NPU modules."""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE = Path(__file__).with_name("patch_external_draft_first_verify.py")


class TransportPatchTests(unittest.TestCase):
    def _load(self):
        class Batch:
            def __init__(self):
                self.requests = set()
                self._external_draft_tokens = {}

            def add_request(self, request):
                self.requests.add(request.req_id)
                draft = request.sampling_params.extra_args.get("ascend_external_draft")
                if draft:
                    self._external_draft_tokens[request.req_id] = list(draft)
                return 3

            def remove_request(self, req_id):
                self._external_draft_tokens.pop(req_id, None)
                self.requests.remove(req_id)

        proposer = types.ModuleType(
            "vllm_ascend.spec_decode.external_draft_proposer")
        proposer.EXTERNAL_DRAFT_ATTR = "_external_draft_tokens"
        proposer.EXTERNAL_DRAFT_FIRST_VERIFY_ATTR = "_external_draft_first_verify"
        proposer.EXTERNAL_DRAFT_FIRST_VERIFY_KEY = "ascend_external_draft_first_verify"
        proposer.EXTERNAL_DRAFT_POLICIES_ATTR = "_external_draft_policies"
        proposer.EXTERNAL_DRAFT_POLICY_KEY = "ascend_external_draft_rejoin"
        batch_module = types.ModuleType("vllm_ascend.worker.npu_input_batch")
        batch_module.NPUInputBatch = Batch
        modules = {
            "vllm_ascend": types.ModuleType("vllm_ascend"),
            "vllm_ascend.spec_decode": types.ModuleType("vllm_ascend.spec_decode"),
            proposer.__name__: proposer,
            "vllm_ascend.worker": types.ModuleType("vllm_ascend.worker"),
            batch_module.__name__: batch_module,
        }
        with patch.dict(sys.modules, modules):
            exec(compile(MODULE.read_text(encoding="utf-8"), str(MODULE), "exec"),
                 {"__name__": "first_verify_test_patch"})
        return Batch

    @staticmethod
    def _request(req_id, extra):
        return types.SimpleNamespace(
            req_id=req_id,
            sampling_params=types.SimpleNamespace(extra_args=extra),
        )

    def test_candidate_metadata_and_original_draft_transport(self):
        batch = self._load()()
        request = self._request("a", {
            "ascend_external_draft": [10, 11],
            "ascend_external_draft_first_verify": True,
            "ascend_external_draft_rejoin": "ngram_2",
        })
        self.assertEqual(batch.add_request(request), 3)
        self.assertEqual(batch._external_draft_tokens["a"], [10, 11])
        self.assertEqual(batch._external_draft_first_verify, {"a"})
        self.assertEqual(batch._external_draft_policies, {"a": "ngram_2"})
        batch.remove_request("a")
        self.assertEqual(batch.requests, set())
        self.assertEqual(batch._external_draft_tokens, {})
        self.assertEqual(batch._external_draft_first_verify, set())
        self.assertEqual(batch._external_draft_policies, {})

    def test_rollback_does_not_get_first_verify_metadata(self):
        batch = self._load()()
        batch.add_request(self._request("b", {"ascend_external_draft": [3]}))
        self.assertFalse(hasattr(batch, "_external_draft_first_verify"))
        self.assertEqual(batch._external_draft_tokens["b"], [3])

    def test_invalid_policy_rejected_before_batch_mutation(self):
        batch = self._load()()
        with self.assertRaisesRegex(ValueError, "invalid external draft rejoin policy"):
            batch.add_request(self._request("c", {
                "ascend_external_draft": [4],
                "ascend_external_draft_first_verify": True,
                "ascend_external_draft_rejoin": "ngram_4",
            }))
        self.assertEqual(batch.requests, set())


if __name__ == "__main__":
    unittest.main()
