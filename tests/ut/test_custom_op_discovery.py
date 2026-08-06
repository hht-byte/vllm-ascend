# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
OP_DISCOVERY_CMAKE = ROOT_DIR / "csrc" / "cmake" / "func.cmake"


class TestCustomOpDiscovery(unittest.TestCase):
    def test_nested_attention_custom_op_is_discovered(self) -> None:
        cmake = os.environ.get("CMAKE_EXECUTABLE") or shutil.which("cmake")
        if cmake is None:
            self.skipTest("cmake is required to verify custom-op discovery")

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_dir = Path(tmp_dir)
            op_host_dir = source_dir / "attention" / "chunk_gated_delta_rule" / "op_host"
            op_host_dir.mkdir(parents=True)
            (op_host_dir / "CMakeLists.txt").write_text("", encoding="utf-8")

            cmake_lists = f"""
cmake_minimum_required(VERSION 3.16)
project(custom_op_discovery NONE)
include("{OP_DISCOVERY_CMAKE.as_posix()}")
op_add_subdirectory(OP_LIST OP_DIR_LIST)
file(WRITE "${{CMAKE_BINARY_DIR}}/ops.txt" "${{OP_LIST}}")
"""
            (source_dir / "CMakeLists.txt").write_text(cmake_lists, encoding="utf-8")

            build_dir = source_dir / "build"
            subprocess.run(
                [cmake, "-S", str(source_dir), "-B", str(build_dir)],
                check=True,
                capture_output=True,
                text=True,
            )

            discovered_ops = (build_dir / "ops.txt").read_text(encoding="utf-8")
            self.assertEqual(discovered_ops, "chunk_gated_delta_rule")


if __name__ == "__main__":
    unittest.main()
