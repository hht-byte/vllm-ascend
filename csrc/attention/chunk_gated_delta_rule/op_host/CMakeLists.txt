# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_ops_compile_options(
    OP_NAME ChunkGatedDeltaRule
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
)

target_sources(op_host_aclnnExc PRIVATE
    chunk_gated_delta_rule_def.cpp
)

target_sources(opapi PRIVATE
    op_api/aclnn_chunk_gated_delta_rule.cpp
    op_api/chunk_gated_delta_rule.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
        op_api/aclnn_chunk_gated_delta_rule.cpp
        op_api/chunk_gated_delta_rule.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
        op_api/aclnn_chunk_gated_delta_rule.cpp
        op_api/chunk_gated_delta_rule.cpp
    )
endif()

target_sources(optiling PRIVATE
    chunk_gated_delta_rule_tiling.cpp
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_sources(opsproto PRIVATE
    chunk_gated_delta_rule_infershape.cpp
)

target_include_directories(opsproto PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

file(GLOB _CGDR_Aclnn_header
    "${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_chunk_gated_delta_rule.h"
)

install(FILES ${_CGDR_Aclnn_header}
    DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
