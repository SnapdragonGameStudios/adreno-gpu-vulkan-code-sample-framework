//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include <string>

// Tiled-K-first layout MxM shader.
//
// Matrices A and B are pre-transformed on the host using TransformMatrixToTiledKfirst<T>
// before being uploaded to GPU memory.  C and D (result) remain in their original layout.
//
// Tiled-K-first addressing:
//   A offset = row * TILE_K + step * TOTAL_N
//   B offset = col * TILE_K + step * TOTAL_M
//   stride   = TILE_K  (Kfirst = 1)
//
// This matches the host-side tiling formula:
//   out[(kk/tileK)*tileK*M + kk%tileK + mm*tileK] = in[mm*K + kk]
//
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
#extension GL_EXT_debug_printf : enable // Enable this extension if you want to use printf() inside the shader

#extension GL_EXT_shader_explicit_arithmetic_types_float32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32   : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8    : enable

// These specialized constants are set inside the host
layout(constant_id = 0) const uint lsx = 64; // local_size_x set inside the host and map to constant_id = 0
layout(constant_id = 1) const uint lsy = 2;  // local_size_y set inside the host and map to constant_id = 1
layout(constant_id = 2) const uint lsz = 2;  // local_size_z set inside the host and map to constant_id = 2
layout(constant_id = 3) const uint TOTAL_M = 1;
layout(constant_id = 4) const uint TOTAL_N = 1;
layout(constant_id = 5) const uint TOTAL_K = 1;
layout(constant_id = 6) const uint TILE_M = 1;
layout(constant_id = 7) const uint TILE_N = 1;
layout(constant_id = 8) const uint TILE_K = 1;
// constant_id 9-14: layout/stride for A and B are unused in Tiled-K-first path;
// kept so the host specialization-constant array layout is unchanged.
layout(constant_id = 9)  const bool layoutA_Mfirst = false;
layout(constant_id = 10) const bool layoutB_Kfirst = false;
layout(constant_id = 11) const bool layoutC_Mfirst = false;
layout(constant_id = 12) const bool layoutR_Mfirst = false;
layout(constant_id = 13) const uint strideAinElements = 1;
layout(constant_id = 14) const uint strideBinElements = 1;
layout(constant_id = 15) const uint strideCinElements = 1;
layout(constant_id = 16) const uint strideRinElements = 1;

// #defines set on compiler GLSL to SPIR-V command line:
// A_TYPE = e.g. float or float16_t
// R_TYPE = e.g. float or float16_t

layout(set=0, binding=0) readonly buffer InputA { A_TYPE x[]; } inputA;
layout(set=0, binding=1) readonly buffer InputB { A_TYPE x[]; } inputB;
layout(set=0, binding=2) readonly buffer InputC { R_TYPE x[]; } inputC;
layout(set=0, binding=3)  buffer Output { R_TYPE x[]; } outputO;

// Set work-group size at dispatch time using specialized constant_id 0,1,2, see host source code for detail
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main()
{
    const uint32_t block_id_m = gl_GlobalInvocationID.y;
    const uint32_t block_id_n = gl_GlobalInvocationID.z;
    if ((block_id_m >= TOTAL_M/TILE_M) || (block_id_n >= TOTAL_N/TILE_N)) return;

    const uint32_t row = block_id_m * TILE_M;
    const uint32_t col = block_id_n * TILE_N;

    // Initialize result matR to zero, not using matC in this shader
    coopmat<R_TYPE, gl_ScopeSubgroup, TILE_M, TILE_N, gl_MatrixUseAccumulator> matR;
    matR = coopmat<R_TYPE, gl_ScopeSubgroup, TILE_M, TILE_N, gl_MatrixUseAccumulator>(0.0);

    for (uint32_t step = 0; step < TOTAL_K; step += TILE_K)
    {
        // Tiled-K-first addressing.
        // Host pre-transforms A[M,K] and B[K,N] into tiled layout before upload.
        //
        // A tiled layout: (kk/TILE_K)*TILE_K*M + mm*TILE_K + kk%TILE_K
        //   => start for this workgroup's row at K-step:  row * TILE_K + step * TOTAL_N
        //
        // B tiled layout: (kk/TILE_K)*TILE_K*N + nn*TILE_K + kk%TILE_K
        //   => start for this workgroup's col at K-step:  col * TILE_K + step * TOTAL_M
        uint32_t subMatrixAStartInElements = row * TILE_K + step * TOTAL_N;
        uint32_t subMatrixBStartInElements = col * TILE_K + step * TOTAL_M;

        coopmat<A_TYPE, gl_ScopeSubgroup, TILE_M, TILE_K, gl_MatrixUseA> matA;
        coopMatLoad(matA, inputA.x, subMatrixAStartInElements, TILE_K, 1 /*Kfirst*/);

        coopmat<A_TYPE, gl_ScopeSubgroup, TILE_K, TILE_N, gl_MatrixUseB> matB;
        coopMatLoad(matB, inputB.x, subMatrixBStartInElements, TILE_K, 1 /*Kfirst*/);

        matR = coopMatMulAdd(matA, matB, matR);
    }

    // Store results — D/R matrix remains in original (non-tiled) layout
    uint32_t subMatrixRStartInElements = layoutR_Mfirst ? col * strideRinElements + row : row * strideRinElements + col;
    coopMatStore(matR, outputO.x, subMatrixRStartInElements, strideRinElements, int(layoutR_Mfirst));
}
)";
