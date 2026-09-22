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
const char* Test01_MxM_Basic = R"(
#version 450 core
#pragma use_vulkan_memory_model
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_KHR_cooperative_matrix : enable
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

layout(set=0, binding=0) readonly buffer InputA { A_TYPE x[]; } inputA;
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
        if (layoutA_Mfirst)
        {
            coopMatLoad(matA, inputA.x, row + step * strideAinElements, strideAinElements, 1);
        }
        else if (layoutA_TiledKfirst)
        {
            coopMatLoad(matA, inputA.x, row * TILE_K + step * TOTAL_M, TILE_K, 0);
        }
        else
        {
            coopMatLoad(matA, inputA.x, row * strideAinElements + step, strideAinElements, 0);
        }

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
