//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include <string>

// Matrix multiplication shader with selectable input and output buffer layouts.
const char* Test02_MxM_VecToMat = R"(
#version 450 core
#pragma use_vulkan_memory_model
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_KHR_cooperative_matrix : require
#extension GL_QCOM_cooperative_matrix_conversion : require
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_debug_printf : enable

#extension GL_EXT_shader_explicit_arithmetic_types_float32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32   : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8    : enable

layout(constant_id = 0) const uint lsx = 64;
layout(constant_id = 1) const uint lsy = 2;
layout(constant_id = 2) const uint lsz = 2;
layout(constant_id = 3) const uint TOTAL_M = 1;
layout(constant_id = 4) const uint TOTAL_N = 1;
layout(constant_id = 5) const uint TOTAL_K = 1;
layout(constant_id = 6) const uint TILE_M = 1;
layout(constant_id = 7) const uint TILE_N = 1;
layout(constant_id = 8) const uint TILE_K = 1;
layout(constant_id = 9)  const bool layoutA_Mfirst = false;
layout(constant_id = 10) const bool layoutB_Nfirst = false;
layout(constant_id = 11) const bool layoutA_TiledKfirst = true;
layout(constant_id = 12) const bool layoutB_TiledKfirst = true;
layout(constant_id = 13) const bool layoutC_Mfirst = false;
layout(constant_id = 14) const bool layoutR_Mfirst = false;
layout(constant_id = 15) const uint strideAinElements = 1;
layout(constant_id = 16) const uint strideBinElements = 1;
layout(constant_id = 17) const uint strideCinElements = 1;
layout(constant_id = 18) const uint strideRinElements = 1;

#if PACKED_A
layout(set=0, binding=0) readonly buffer InputA { uint32_t x[]; } inputA;
#else
layout(set=0, binding=0) readonly buffer InputA { A_TYPE x[]; } inputA;
#endif
layout(set=0, binding=1) readonly buffer InputB { A_TYPE x[]; } inputB;
layout(set=0, binding=2) readonly buffer InputC { R_TYPE x[]; } inputC;
layout(set=0, binding=3)  buffer Output { R_TYPE x[]; } outputO;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main()
{
    const uint32_t block_id_m = gl_GlobalInvocationID.y;
    const uint32_t block_id_n = gl_GlobalInvocationID.z;
    if ((block_id_m >= TOTAL_M/TILE_M) || (block_id_n >= TOTAL_N/TILE_N)) return;

    const uint32_t row = block_id_m * TILE_M;
    const uint32_t col = block_id_n * TILE_N;

    coopmat<R_TYPE, gl_ScopeSubgroup, TILE_M, TILE_N, gl_MatrixUseAccumulator> matR;
    matR = coopmat<R_TYPE, gl_ScopeSubgroup, TILE_M, TILE_N, gl_MatrixUseAccumulator>(0.0);

    for (uint32_t step = 0; step < TOTAL_K; step += TILE_K)
    {
        coopmat<A_TYPE, gl_ScopeSubgroup, TILE_M, TILE_K, gl_MatrixUseA> matA;
        const uint laneRow = row + gl_SubgroupInvocationID;
#if PACKED_A
        // QCOM conversion interprets each word as NUM_PACK input components.
        uint32_t vecA[TILE_K / NUM_PACK];
        if (layoutA_Mfirst)
        {
            // Gather strided INT8 components without changing their bit patterns.
            const uint bitsPerComponent = 32 / NUM_PACK;
            const uint componentMask = (1u << bitsPerComponent) - 1u;
            for (uint k = 0; k < TILE_K / NUM_PACK; ++k)
            {
                uint packed = 0;
                for (uint c = 0; c < NUM_PACK; ++c)
                {
                    const uint offset = laneRow + (step + k * NUM_PACK + c) * strideAinElements;
                    const uint component = (inputA.x[offset / NUM_PACK] >> ((offset % NUM_PACK) * bitsPerComponent)) & componentMask;
                    packed |= component << (c * bitsPerComponent);
                }
                vecA[k] = packed;
            }
        }
        else
        {
            const uint wordOffset = (layoutA_TiledKfirst ? laneRow * TILE_K + step * TOTAL_M
                                                       : laneRow * strideAinElements + step) / NUM_PACK;
            for (uint k = 0; k < TILE_K / NUM_PACK; ++k)
                vecA[k] = inputA.x[wordOffset + k];
        }
#else
        A_TYPE vecA[TILE_K];
        for (uint k = 0; k < TILE_K; ++k)
        {
            const uint offset = layoutA_Mfirst ? laneRow + (step + k) * strideAinElements
                : (layoutA_TiledKfirst ? laneRow * TILE_K + step * TOTAL_M + k
                                      : laneRow * strideAinElements + step + k);
            vecA[k] = inputA.x[offset];
        }
#endif
        vectorToCoopmatQCOM(vecA, matA);

        coopmat<A_TYPE, gl_ScopeSubgroup, TILE_K, TILE_N, gl_MatrixUseB> matB;
        if (layoutB_Nfirst)
        {
            coopMatLoad(matB, inputB.x, col + step * strideBinElements, strideBinElements, 0);
        }
        else if (layoutB_TiledKfirst)
        {
            coopMatLoad(matB, inputB.x, col * TILE_K + step * TOTAL_N, TILE_K, 1);
        }
        else
        {
            coopMatLoad(matB, inputB.x, col * strideBinElements + step, strideBinElements, 1);
        }

        matR = coopMatMulAdd(matA, matB, matR);
    }

    uint32_t subMatrixRStartInElements = layoutR_Mfirst ? col * strideRinElements + row : row * strideRinElements + col;
    coopMatStore(matR, outputO.x, subMatrixRStartInElements, strideRinElements, int(layoutR_Mfirst));
}
)";
