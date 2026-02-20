//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include <string>

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
layout(constant_id = 9) const bool layoutA_Mfirst = false;
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

//layout(set=0, binding=0, std430) uniform Params { InputA inputA; InputB inputB; InputC inputC; Output outputO; } params;

// Set work-group size at dispacth time using specialized constant_id 0,1,2, see host source code for detail
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
        // On each iteration, load a row of cooperative matrices from matrix A,
        // load a column of cooperative matrices from matrix B, and multiply all
        // pairs of those matrices.
        uint32_t subMatrixAStartInElements = layoutA_Mfirst ? row + step * strideAinElements : row * strideAinElements + step;
        uint32_t subMatrixBStartInElements = layoutB_Kfirst ? col * strideBinElements + step : col + step * strideBinElements;

        coopmat<A_TYPE, gl_ScopeSubgroup, TILE_M, TILE_K, gl_MatrixUseA> matA;
        coopMatLoad(matA, inputA.x, subMatrixAStartInElements, strideAinElements, int(layoutA_Mfirst));

        coopmat<A_TYPE, gl_ScopeSubgroup, TILE_K, TILE_N, gl_MatrixUseB> matB;
        coopMatLoad(matB, inputB.x, subMatrixBStartInElements, strideBinElements, int(layoutB_Kfirst));

        //for (int i = (gl_LocalInvocationID.x > 63 ? 20 : 0); i < 100; i++) // diable unroll, test gpu_freq, should around 1%
        matR = coopMatMulAdd(matA, matB, matR);
    }

    // Store results
    uint32_t subMatrixRStartInElements = layoutR_Mfirst ? col * strideRinElements + row : row * strideRinElements + col;

    coopMatStore(matR, outputO.x, subMatrixRStartInElements, strideRinElements, int(layoutR_Mfirst));
}
)";