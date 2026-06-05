/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * ------------------------------------------------------------------------- */

#ifndef STREAMING_ATTENTION_TILING_H
#define STREAMING_ATTENTION_TILING_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct StreamingAttentionTilingData {
    uint32_t batch;
    uint32_t heads;
    uint32_t seqLen;
    uint32_t headDim;
    uint32_t rowCount;
    uint32_t totalElements;
    uint32_t blockM;
    uint32_t blockN;
    uint32_t mode;
    AscendC::tiling::TCubeTiling qkMatmulTiling;
    AscendC::tiling::TCubeTiling pvMatmulTiling;
};

#endif // STREAMING_ATTENTION_TILING_H
