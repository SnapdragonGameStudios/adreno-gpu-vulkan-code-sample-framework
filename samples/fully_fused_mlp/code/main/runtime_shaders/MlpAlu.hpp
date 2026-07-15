//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

// ALU-only fused MLP forward pass.
//
// Faithful port of Vulkan_MLP's
//   shader/breda_baseline/fused_mlp/forward_single_fiber_tile_interleaved.comp
//
// The original macro / shared-memory structure is preserved intact (the
// #defines, localWeights_0[256], LOAD_WEIGHTS_TO_LOCAL_MEM_256, the
// FOUR_NODE_X_ALL / ONE_LAYER_BLOCK / ONE_LAYER_UNROLL unrolls, the
// memoryBarrierShared() placement, and the load_bias spec constant): these
// drive the compiler's register/LDS scheduling and must not be flattened.
// Each fiber processes one sample, holding its activation vector in fp16
// registers while the 64 fibers of a workgroup cooperatively stream per-tile
// weights through 256-element shared memory.
//
// Only deliberate change vs upstream: main() wraps the original per-fiber body
// in a grid-stride loop so a bounded workgroup count covers batches beyond
// maxComputeWorkGroupCount[0]*64. batch_size is a multiple of FORWARD_GROUP_SIZE
// (host-enforced), so the loop bound is workgroup-uniform and the shared-memory
// barriers stay balanced. When the dispatch already covers the batch the loop
// runs exactly once == upstream behaviour.
//
// Network: 4 weight matrices — W0 (input proj) + W1 + W2 (hidden) + W3 (output).
// in == hidden == NETWORK_MAX_WIDTH (16/32/64), out = 4 (RGBA).
//
// Host injects (RuntimeShader::AddDefine):
//   NETWORK_MAX_WIDTH  = 16 | 32 | 64
//   FORWARD_GROUP_SIZE = 64   (the LDS weight load is hardwired to 64 fibers)
// Spec constants 0..5 carry in/hidden/out/batch/activation/load_bias.
//
// Buffers (match FusedMlpRunner::RunAlu descriptor set):
//   binding 0 UBO constants
//   binding 1 input   (f16vec4[], row-major [batch x inFeatures])
//   binding 2 weights (f16vec4[], 4 layers concatenated, each [out x in] row-major)
//   binding 3 biases  (f16vec4[], 4 layers concatenated, flat per-layer [out])
//   binding 4 output  (f16vec2[], row-major [batch x outFeatures])

inline const char* kMlpAluShader = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require

layout(constant_id = 0) const uint in_features = 16;
layout(constant_id = 1) const uint hidden_features = 16;
layout(constant_id = 2) const uint out_feature = 4;
layout(constant_id = 3) const uint batch_size = 64;
layout(constant_id = 4) const uint activation = 0; // 0: none, 1: ReLU
layout(constant_id = 5) const uint load_bias = 0;  // 0: none, 1: Load_Bias

// keep configurable via -D or earlier #defines
#ifndef FORWARD_GROUP_SIZE
#define FORWARD_GROUP_SIZE 64
#endif

#ifndef NETWORK_MAX_WIDTH
#define NETWORK_MAX_WIDTH 16
#endif

#define ACCUMULATION NETWORK_MAX_WIDTH
#define LOCAL_WEIGHT_SIZE (NETWORK_MAX_WIDTH * ACCUMULATION)
#define LOCAL_WEIGHT_FIBER_LOAD_SIZE 4

layout(local_size_x = FORWARD_GROUP_SIZE) in;

layout(binding = 0) uniform FusedMlpConstantsUBO {
    uint batchSize;
    uint inFeatures;
    uint outFeatures;
    uint hiddenLayers;
    uint hiddenFeatures;
    uint activation;
    uint outActivation;
} mlpConstants;

layout(std430, binding = 1) readonly buffer Inputs {
    f16vec4 data[];   // packed half pairs
} inputsBuf;

layout(std430, binding = 2) readonly buffer Weights {
    f16vec4 data[];   // packed half pairs
} weightsBuf;

layout(std430, binding = 3) readonly buffer Biases {
    f16vec4 data[];   // packed half pairs or floats depending on layout
} biasesBuf;

layout(std430, binding = 4) buffer Output {
    f16vec2 data[];   // packed half pairs for output
} outputBuf;

// shared (groupshared) memory
shared float16_t localWeights_0[256];

void main() {
    uint localId = gl_LocalInvocationID.x;

    // accumulation registers
    float16_t acc_reg[ACCUMULATION];
    float16_t input_reg[ACCUMULATION];

    const uint localWeightIndex = localId * LOCAL_WEIGHT_FIBER_LOAD_SIZE;

//
// macros translated to GLSL-compatible constructs
//
#define LOAD_BIAS(activation_cnt) \
    {\
        if(load_bias == 1u){\
            [[unroll]]for (uint i = 0u; i < activation_cnt; ++i) {\
                acc_reg[i] += float16_t(biasesBuf.data[biasOffset][i % 4u]);\
            }\
        }\
        if (activation == 1u) { /* ReLU */ \
            [[unroll]]for (uint i = 0u; i < activation_cnt; ++i) {\
                acc_reg[i] = acc_reg[i] > float16_t(0.0) ? acc_reg[i] : float16_t(0.0);\
            }\
        }\
    }

// Output-layer epilogue: add bias but DO NOT apply activation. A standard MLP
// has a linear output layer, so the final layer is never ReLU'd (this lets the
// network produce signed outputs). Only the bias/activation epilogue differs
// from LOAD_BIAS; the optimization-critical weight-streaming / FOUR_NODE inner
// loops are unchanged.
#define LOAD_BIAS_ONLY(activation_cnt) \
    {\
        if(load_bias == 1u){\
            [[unroll]]for (uint i = 0u; i < activation_cnt; ++i) {\
                acc_reg[i] += float16_t(biasesBuf.data[biasOffset][i % 4u]);\
            }\
        }\
    }

#define LOAD_WEIGHTS_TO_LOCAL_MEM_256(targetLmBank) \
    {\
        targetLmBank[localWeightIndex + 0u] = float16_t(weightsBuf.data[globalWeightIndex].x);\
        targetLmBank[localWeightIndex + 1u] = float16_t(weightsBuf.data[globalWeightIndex].y);\
        targetLmBank[localWeightIndex + 2u] = float16_t(weightsBuf.data[globalWeightIndex].z);\
        targetLmBank[localWeightIndex + 3u] = float16_t(weightsBuf.data[globalWeightIndex].w);\
        globalWeightIndex += 64;\
    }

#define LOAD_INPUT_WITH_INPUTS(activation_cnt)\
    {\
        [[unroll]]for (uint i = 0u; i < activation_cnt; i ++){\
            input_reg[i] = float16_t(inputsBuf.data[inputIndexBase + i/4u][i%4u]);\
        }\
    }

#define LOAD_INPUT_WITH_ACC(activation_cnt) \
    [[unroll]]for (uint i = 0u; i < activation_cnt; ++i) {\
        input_reg[i] = acc_reg[i];\
     }

#define CLEAR_ACC(activation_cnt) \
    [[unroll]]for (uint i = 0u; i < activation_cnt; ++i) {\
        acc_reg[i] = float16_t(0.0);\
    }

#define FOUR_NODE_X_4(indexBase, targetLocalWeights, indexTargetLmBase, inputIndex, weightIndex, offset)\
    [[unroll]]for (uint j = 0; j < 4; j++){\
        float16_t weight = targetLocalWeights[(indexTargetLmBase + j) * ACCUMULATION + weightIndex + offset];\
        acc_reg[indexBase + j] += input_reg[inputIndex + offset] * weight;\
    }

#define FOUR_NODE_X_ALL(activation_cnt, indexBase, targetLocalWeights, indexTargetLmBase, offset)\
    [[unroll]]for (uint k = 0; k < activation_cnt; k++){\
        FOUR_NODE_X_4(indexBase, targetLocalWeights, indexTargetLmBase, k, k, offset);\
    }

// Block of all nodes for one layer, ensures 256 wights are used per block
#define ONE_LAYER_BLOCK_16(targetLocalWeights)\
    [[unroll]]for (uint _i = 0; _i < 4; ++_i) {\
        FOUR_NODE_X_ALL(16, _i*4, targetLocalWeights, _i*4, 0);\
    }

#define ONE_LAYER_BLOCK_32(offset, targetLocalWeights)\
    [[unroll]]for (uint _i = 0; _i < 2; ++_i) {\
        FOUR_NODE_X_ALL(32, offset + _i*4, targetLocalWeights, _i*4, 0);\
    }

#define ONE_LAYER_BLOCK_64(offset, targetLocalWeights)\
    FOUR_NODE_X_ALL(64, offset + 0, targetLocalWeights, 0, 0);

// Whole layer unroll macros
#define ONE_LAYER_UNROLL_16\
    LOAD_WEIGHTS_TO_LOCAL_MEM_256(localWeights_0)\
    memoryBarrierShared();\
    ONE_LAYER_BLOCK_16(localWeights_0)\
    memoryBarrierShared();

#define ONE_LAYER_UNROLL_32\
    [[unroll]]for (uint i = 0; i < 4; ++i) {\
        LOAD_WEIGHTS_TO_LOCAL_MEM_256(localWeights_0)\
        memoryBarrierShared();\
        ONE_LAYER_BLOCK_32(i * 8, localWeights_0)\
        memoryBarrierShared();\
    }

#define ONE_LAYER_UNROLL_64\
    [[unroll]]for (uint i = 0; i < 16; ++i) {\
        LOAD_WEIGHTS_TO_LOCAL_MEM_256(localWeights_0)\
        memoryBarrierShared();\
        ONE_LAYER_BLOCK_64(i * 4, localWeights_0)\
        memoryBarrierShared();\
    }

#define INPUT_LAYER_16\
        LOAD_INPUT_WITH_INPUTS(16)\
        CLEAR_ACC(16)\
        ONE_LAYER_UNROLL_16\
        LOAD_BIAS(16)

#define HIDDEN_LAYER_16\
        LOAD_INPUT_WITH_ACC(16)\
        CLEAR_ACC(16)\
        ONE_LAYER_UNROLL_16\
        LOAD_BIAS(16)

#define FINAL_LAYER_16\
        LOAD_WEIGHTS_TO_LOCAL_MEM_256(localWeights_0)\
        LOAD_INPUT_WITH_ACC(16)\
        CLEAR_ACC(16)\
        memoryBarrierShared();\
        FOUR_NODE_X_ALL(16, 0, localWeights_0, 0, 0);\
        LOAD_BIAS_ONLY(4)

#define INPUT_LAYER_32\
        LOAD_INPUT_WITH_INPUTS(32)\
        CLEAR_ACC(32)\
        ONE_LAYER_UNROLL_32\
        LOAD_BIAS(32)

#define HIDDEN_LAYER_32\
        LOAD_INPUT_WITH_ACC(32)\
        CLEAR_ACC(32)\
        ONE_LAYER_UNROLL_32\
        LOAD_BIAS(32)

#define FINAL_LAYER_32\
        LOAD_WEIGHTS_TO_LOCAL_MEM_256(localWeights_0)\
        LOAD_INPUT_WITH_ACC(32)\
        CLEAR_ACC(32)\
        memoryBarrierShared();\
        FOUR_NODE_X_ALL(32, 0, localWeights_0, 0, 0)\
        LOAD_BIAS_ONLY(4)


#define INPUT_LAYER_64\
        LOAD_INPUT_WITH_INPUTS(64)\
        CLEAR_ACC(64)\
        ONE_LAYER_UNROLL_64\
        LOAD_BIAS(64)

#define HIDDEN_LAYER_64\
        LOAD_INPUT_WITH_ACC(64)\
        CLEAR_ACC(64)\
        ONE_LAYER_UNROLL_64\
        LOAD_BIAS(64)

#define FINAL_LAYER_64\
        LOAD_WEIGHTS_TO_LOCAL_MEM_256(localWeights_0)\
        LOAD_INPUT_WITH_ACC(64)\
        CLEAR_ACC(64)\
        memoryBarrierShared();\
        FOUR_NODE_X_ALL(64, 0, localWeights_0, 0, 0)\
        LOAD_BIAS_ONLY(4)

#if (NETWORK_MAX_WIDTH == 16)
    #define INPUT_LAYER INPUT_LAYER_16
    #define HIDDEN_LAYER HIDDEN_LAYER_16
    #define FINAL_LAYER FINAL_LAYER_16
#elif(NETWORK_MAX_WIDTH == 32)
    #define INPUT_LAYER INPUT_LAYER_32
    #define HIDDEN_LAYER HIDDEN_LAYER_32
    #define FINAL_LAYER FINAL_LAYER_32
#elif(NETWORK_MAX_WIDTH == 64)
    #define INPUT_LAYER INPUT_LAYER_64
    #define HIDDEN_LAYER HIDDEN_LAYER_64
    #define FINAL_LAYER FINAL_LAYER_64
#else
    #error "Unsupported NETWORK_MAX_WIDTH. Please define it as 16, 32 or 64."
#endif

    // Grid-stride over workgroup-aligned blocks of FORWARD_GROUP_SIZE fibers so
    // any batch size is legal. batch_size is a multiple of FORWARD_GROUP_SIZE
    // (host-enforced) => fiberBase < batch_size is workgroup-uniform and every
    // participating fiber is valid, keeping the shared-memory barriers balanced.
    // When the dispatch already covers the batch this loops once == upstream.
    const uint fiberGridStride = gl_NumWorkGroups.x * FORWARD_GROUP_SIZE;
    for (uint fiberBase = gl_WorkGroupID.x * FORWARD_GROUP_SIZE;
         fiberBase < batch_size;
         fiberBase += fiberGridStride)
    {
        uint fiberId = fiberBase + localId;
        uint inputIndexBase = fiberId * in_features / 4u;
        uint biasOffset = 0u;
        uint globalWeightIndex = localId;

        // INPUT_LAYER macro will load weights into shared memory
        INPUT_LAYER
        biasOffset += in_features/4u; // adjust depending on layout

        // Hidden Layers
        HIDDEN_LAYER
        biasOffset += hidden_features/4u;

        HIDDEN_LAYER
        biasOffset += hidden_features/4u;

        FINAL_LAYER

        // pack first 4 accumulators as halves into two uints (results.x, results.y)
        uint outputIndexBase = fiberId * out_feature / 2u;
        if (fiberId < batch_size) {
            outputBuf.data[outputIndexBase + 0u] = f16vec2(acc_reg[0], acc_reg[1]);
            outputBuf.data[outputIndexBase + 1u] = f16vec2(acc_reg[2], acc_reg[3]);
        }
    }
}
)";

// ---------------------------------------------------------------------------
// WIDE_IO ALU shader: 12 -> W -> 10 (one hidden layer). Straightforward
// one-sample-per-fiber kernel (64-fiber workgroups, grid-stride). No upstream
// tile-interleave to preserve for this net, so it is written for clarity and to
// match CpuReference's fp16 branch bit-for-bit:
//   - pairwise fp16 summation: sum += half(a[2i]*w[2i]) + half(a[2i+1]*w[2i+1])
//   - weights in [in][out] row-major (stride = out) — same as host storage, no
//     transpose (host uploads m_host.weights verbatim, W-padded rows for L0)
//   - full per-node bias[o]; hidden ReLU, output linear
// Bindings (5): 0 UBO constants, 1 input (f16vec4 [batch x IN_PAD/4]), 2 W0
// [IN_PAD x W], 3 W1 [W x OUT], 4 biases (b0[W] ++ b1[OUT]), output
// (f16vec2 [batch x OUT_PAD/2]). IN_PAD = 16 (12 padded), OUT_PAD host-provided.
inline const char* kMlpAluWideIOShader = R"(
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require

layout(constant_id = 0) const uint in_features  = 12;  // real input count
layout(constant_id = 1) const uint hidden_features = 16;
layout(constant_id = 2) const uint out_features = 10;
layout(constant_id = 3) const uint batch_size   = 64;
layout(constant_id = 4) const uint activation   = 0;   // 0 none, 1 ReLU (hidden)
layout(constant_id = 5) const uint load_bias    = 0;

#ifndef FORWARD_GROUP_SIZE
#define FORWARD_GROUP_SIZE 64
#endif
#ifndef IN_PAD
#define IN_PAD 16   // input row stride in the packed f16vec4 input buffer (12 -> 16)
#endif
#ifndef OUT_PAD
#define OUT_PAD 12  // output row stride (halfs); f16vec2 view => OUT_PAD/2 vec2 per row
#endif
#ifndef HIDDEN_MAX
#define HIDDEN_MAX 64
#endif

layout(local_size_x = FORWARD_GROUP_SIZE) in;

layout(binding = 0) uniform FusedMlpConstantsUBO {
    uint batchSize; uint inFeatures; uint outFeatures;
    uint hiddenLayers; uint hiddenFeatures; uint activation; uint outActivation;
} mlpConstants;

layout(std430, binding = 1) readonly buffer Inputs  { f16vec4   data[]; } inputsBuf;   // [batch x IN_PAD/4]
layout(std430, binding = 2) readonly buffer Weights0 { float16_t data[]; } w0Buf;       // [IN_PAD x hidden]
layout(std430, binding = 3) readonly buffer Weights1 { float16_t data[]; } w1Buf;       // [hidden x out]
layout(std430, binding = 4) readonly buffer Biases   { float16_t data[]; } biasBuf;     // b0[hidden] ++ b1[out]
layout(std430, binding = 5) buffer Output            { f16vec2   data[]; } outputBuf;   // [batch x OUT_PAD/2]

void main() {
    uint localId = gl_LocalInvocationID.x;
    const uint fiberGridStride = gl_NumWorkGroups.x * FORWARD_GROUP_SIZE;
    for (uint fiberBase = gl_WorkGroupID.x * FORWARD_GROUP_SIZE;
         fiberBase < batch_size;
         fiberBase += fiberGridStride)
    {
        uint fiberId = fiberBase + localId;
        if (fiberId >= batch_size) break;

        // load input (real in_features; remainder of IN_PAD is zero padding)
        float16_t inp[IN_PAD];
        uint inVec4Base = fiberId * (IN_PAD / 4u);
        [[unroll]] for (uint i = 0u; i < IN_PAD / 4u; ++i) {
            f16vec4 v = inputsBuf.data[inVec4Base + i];
            inp[i*4u+0u] = v.x; inp[i*4u+1u] = v.y; inp[i*4u+2u] = v.z; inp[i*4u+3u] = v.w;
        }

        // L0: hidden[h] = sum_i inp[i] * W0[i*hidden + h]  (pairwise fp16), +bias0[h], ReLU
        float16_t hid[HIDDEN_MAX];
        for (uint h = 0u; h < hidden_features; ++h) {
            float16_t sum = float16_t(0.0);
            uint pairs = in_features / 2u;
            for (uint p = 0u; p < pairs; ++p) {
                uint i0 = p*2u; uint i1 = i0 + 1u;
                float16_t pr0 = float16_t(inp[i0]) * w0Buf.data[i0*hidden_features + h];
                float16_t pr1 = float16_t(inp[i1]) * w0Buf.data[i1*hidden_features + h];
                sum = sum + (pr0 + pr1);
            }
            if ((in_features & 1u) != 0u) {
                uint i0 = in_features - 1u;
                sum = sum + float16_t(inp[i0]) * w0Buf.data[i0*hidden_features + h];
            }
            if (load_bias == 1u) sum = sum + biasBuf.data[h];
            if (activation == 1u) sum = max(sum, float16_t(0.0));
            hid[h] = sum;
        }

        // L1 (output): out[o] = sum_k hid[k] * W1[k*out + o] (pairwise fp16), +bias1[o], linear
        float16_t outv[16];
        for (uint o = 0u; o < out_features; ++o) {
            float16_t sum = float16_t(0.0);
            uint pairs = hidden_features / 2u;
            for (uint p = 0u; p < pairs; ++p) {
                uint k0 = p*2u; uint k1 = k0 + 1u;
                float16_t pr0 = hid[k0] * w1Buf.data[k0*out_features + o];
                float16_t pr1 = hid[k1] * w1Buf.data[k1*out_features + o];
                sum = sum + (pr0 + pr1);
            }
            if ((hidden_features & 1u) != 0u) {
                uint k0 = hidden_features - 1u;
                sum = sum + hid[k0] * w1Buf.data[k0*out_features + o];
            }
            if (load_bias == 1u) sum = sum + biasBuf.data[hidden_features + o];
            outv[o] = sum;
        }

        // store OUT (f16vec2 pairs). OUT_PAD is even; write ceil(out/2) pairs,
        // zero-filling the odd tail lane.
        uint outVec2Base = fiberId * (OUT_PAD / 2u);
        uint fullPairs = out_features / 2u;
        [[unroll]] for (uint q = 0u; q < 8u; ++q) {
            if (q < fullPairs)
                outputBuf.data[outVec2Base + q] = f16vec2(outv[q*2u], outv[q*2u+1u]);
        }
        if ((out_features & 1u) != 0u) {
            uint q = fullPairs;
            outputBuf.data[outVec2Base + q] = f16vec2(outv[out_features - 1u], float16_t(0.0));
        }
    }
}
)";
