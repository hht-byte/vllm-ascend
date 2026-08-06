# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OP_ROOT = ROOT / "csrc" / "attention" / "chunk_gated_delta_rule" / "op_host"


class TestChunkGatedDeltaRuleSource(unittest.TestCase):

    def test_platform_initialization_failure_is_propagated(self):
        header = (OP_ROOT / "chunk_gated_delta_rule_tiling.h").read_text(
            encoding="utf-8")
        source = (OP_ROOT / "chunk_gated_delta_rule_tiling.cpp").read_text(
            encoding="utf-8")

        self.assertIn("platformStatus_ = InitCompileInfo();", header)
        self.assertIn("ge::graphStatus InitCompileInfo();", header)
        self.assertIn(
            "ge::graphStatus platformStatus_{ge::GRAPH_FAILED};", header)
        self.assertIn("return platformStatus_;", source)

    def test_op_api_includes_header_for_strstr(self):
        source = (OP_ROOT / "op_api" /
                  "aclnn_chunk_gated_delta_rule.cpp").read_text(
                      encoding="utf-8")

        self.assertIn("#include <cstring>", source)

    def test_infer_shape_uses_output_index_constants(self):
        source = (OP_ROOT /
                  "chunk_gated_delta_rule_infershape.cpp").read_text(
                      encoding="utf-8")

        self.assertIn("GetOutputShape(OUTPUT_OUT_IDX)", source)
        self.assertIn("GetOutputShape(OUTPUT_FINAL_STATE_IDX)", source)


if __name__ == "__main__":
    unittest.main()
