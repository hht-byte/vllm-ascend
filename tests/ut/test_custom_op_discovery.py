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
CHUNK_GATED_DELTA_RULE_OP_HOST = (
    ROOT_DIR / "csrc" / "attention" / "chunk_gated_delta_rule" / "op_host"
)


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

    def test_chunk_gated_delta_rule_registers_with_legacy_targets(self) -> None:
        cmake = os.environ.get("CMAKE_EXECUTABLE") or shutil.which("cmake")
        if cmake is None:
            self.skipTest("cmake is required to verify custom-op registration")

        with tempfile.TemporaryDirectory() as tmp_dir:
            source_dir = Path(tmp_dir)
            cmake_lists = f"""
cmake_minimum_required(VERSION 3.16)
project(custom_op_registration NONE)
function(add_ops_compile_options)
endfunction()
function(target_sources target)
    file(APPEND "${{CMAKE_BINARY_DIR}}/registrations.txt" "${{target}}=${{ARGN}}\n")
endfunction()
function(target_include_directories)
endfunction()
function(install)
endfunction()
set(BUILD_OPEN_PROJECT ON)
set(ACLNN_INC_INSTALL_DIR include)
file(WRITE "${{CMAKE_BINARY_DIR}}/registrations.txt" "")
add_subdirectory("{CHUNK_GATED_DELTA_RULE_OP_HOST.as_posix()}" chunk_op_host)
"""
            (source_dir / "CMakeLists.txt").write_text(cmake_lists, encoding="utf-8")

            build_dir = source_dir / "build"
            configure = subprocess.run(
                [cmake, "-S", str(source_dir), "-B", str(build_dir)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(configure.returncode, 0, configure.stdout + configure.stderr)

            registrations = (build_dir / "registrations.txt").read_text(encoding="utf-8").replace("\\", "/")
            expected_sources = {
                "op_host_aclnnExc": ["chunk_gated_delta_rule_def.cpp"],
                "opapi": [
                    "op_api/aclnn_chunk_gated_delta_rule.cpp",
                    "op_api/chunk_gated_delta_rule.cpp",
                ],
                "optiling": ["chunk_gated_delta_rule_tiling.cpp"],
                "opsproto": ["chunk_gated_delta_rule_infershape.cpp"],
            }
            for target, sources in expected_sources.items():
                registration = next(
                    line for line in registrations.splitlines() if line.startswith(f"{target}=")
                )
                for source in sources:
                    self.assertIn(source, registration)


if __name__ == "__main__":
    unittest.main()
