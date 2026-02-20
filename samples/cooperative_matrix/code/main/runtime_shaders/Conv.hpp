//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include <string>

const char* Test03_CONV = R"(
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
#extension GL_QCOM_cooperative_matrix_conversion : enable

// These specialized constants are set inside the host
layout(constant_id = 0) const uint lsx = 64; // local_size_x set inside the host and map to constant_id = 0
layout(constant_id = 1) const uint lsy = 2;  // local_size_y set inside the host and map to constant_id = 1
layout(constant_id = 2) const uint lsz = 2;  // local_size_z set inside the host and map to constant_id = 2
layout(constant_id = 3)  const uint TOTAL_M = 1;
layout(constant_id = 4)  const uint TOTAL_N = 1;
layout(constant_id = 5)  const uint TOTAL_K = 1;
layout(constant_id = 6)  const uint TILE_M = 1;
layout(constant_id = 7)  const uint TILE_N = 1;
layout(constant_id = 8)  const uint TILE_K = 1;
layout(constant_id = 9)  const uint INPUT_W = 1;
layout(constant_id = 10)  const uint INPUT_H = 1;
layout(constant_id = 11)  const uint FILTER_W = 1;
layout(constant_id = 12)  const uint FILTER_H = 1;
layout(constant_id = 13) const uint DILATION = 1;
layout(constant_id = 14) const uint STRIDE  = 1;
layout(constant_id = 15) const uint strideAinElements = 1;
layout(constant_id = 16) const uint strideBinElements = 1;
layout(constant_id = 17) const uint strideCinElements = 1;
layout(constant_id = 18) const uint strideRinElements = 1;

// #defines set on compiler GLSL to SPIR-V command line:
// A_TYPE = e.g. float or float16_t
// R_TYPE = e.g. float or float16_t

layout(set=0, binding=0) readonly buffer InputA     { A_TYPE   x[]; } inputA;
layout(set=0, binding=0) readonly buffer InputAuint { uint32_t x[]; } inputAuint;
layout(set=0, binding=1) readonly buffer InputB { A_TYPE x[]; } inputB;
layout(set=0, binding=2) readonly buffer InputC { R_TYPE x[]; } inputC;
layout(set=0, binding=3) buffer Output { R_TYPE x[]; } outputO;

// Set work-group size at dispacth time using specialized constant_id 0,1,2, see host source code for detail
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in; 

void main()
{
    const uint32_t block_id_m = gl_GlobalInvocationID.y;
    const uint32_t block_id_n = gl_GlobalInvocationID.z;
    if ((block_id_m >= TOTAL_M/TILE_M) || (block_id_n >= TOTAL_N/TILE_N)) return;

    const uint32_t row = block_id_m * TILE_M;
    const uint32_t col = block_id_n * TILE_N;
    
    uint32_t gidx_m = gl_GlobalInvocationID.x + TILE_M * gl_GlobalInvocationID.y; // fibers along M
    uint32_t out_col_id = gidx_m % INPUT_W;
    uint32_t out_row_id = gidx_m / INPUT_W;

    uint32_t filter_offset_h = (FILTER_H % 2 == 0)? 0 : FILTER_H/2;
    uint32_t filter_offset_w = (FILTER_W % 2 == 0)? 0 : FILTER_W/2;

    // Initialize result matR to zero, not using matC in this shader
    coopmat<R_TYPE, gl_ScopeSubgroup, TILE_M, TILE_N, gl_MatrixUseAccumulator> matR;
    matR = coopmat<R_TYPE, gl_ScopeSubgroup, TILE_M, TILE_N, gl_MatrixUseAccumulator>(0.0);
    
    for (uint32_t step = 0; step < TOTAL_K; step += TILE_K)
    {
        uint32_t subMatrixBStartInElements = col * FILTER_H * FILTER_W * strideBinElements + step; // B is Kfirst
        for (uint32_t filter_row = 0; filter_row < FILTER_H; filter_row++)
        {
            for (uint32_t filter_col = 0; filter_col < FILTER_W; filter_col++)
            {
                coopmat<A_TYPE, gl_ScopeSubgroup, TILE_K, TILE_N, gl_MatrixUseB> matB;
                coopmat<A_TYPE, gl_ScopeSubgroup, TILE_M, TILE_K, gl_MatrixUseA> matA;

                // load B matrix input data using coop_mat extension
                coopMatLoad(matB, inputB.x, subMatrixBStartInElements, FILTER_H * FILTER_W * strideBinElements, int(true));

                // load A matrix input data as vectors using regular vector load
                uint32_t input_row_id = STRIDE * out_row_id + DILATION * (filter_row - filter_offset_h);
                uint32_t input_col_id = STRIDE * out_col_id + DILATION * (filter_col - filter_offset_w);

                // load A vector data from memory
                uint32_t vecA[TILE_K/NUM_PACK];
                for (int i=0; i<TILE_K/NUM_PACK; i++)
                    vecA[i] = inputAuint.x[(input_row_id * INPUT_W + input_col_id) * strideAinElements/NUM_PACK + step/NUM_PACK + i];

                // zero fill A vector data for out of boundary cases
                if ((input_row_id < 0) || (input_row_id >= INPUT_H) || (input_col_id < 0) || (input_col_id >= INPUT_W))
                  for (int i=0; i<TILE_K/NUM_PACK; i++) vecA[i] = uint32_t(0);

                // convert A vector to A matrix
                vectorToCoopmatQCOM(vecA, matA);

                matR = coopMatMulAdd(matA, matB, matR);

                subMatrixBStartInElements += strideBinElements;
            }
        }
    }

    // Store results
    uint32_t subMatrixRStartInElements = row * strideRinElements + col;
    coopMatStore(matR, outputO.x, subMatrixRStartInElements, strideRinElements, int(false));
}
)";