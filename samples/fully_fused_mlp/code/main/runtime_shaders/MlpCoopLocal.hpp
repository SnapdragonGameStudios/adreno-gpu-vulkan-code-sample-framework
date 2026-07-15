//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

// Local-fusing strategy.
//
// Body of Vulkan_MLP's shader/coopmat_glsl/fwdPass_coopmat_local_inputs.comp.
// Input X and all inter-layer hidden state live in shared (LDS) memory
// (local_cache_x); only coopMatLoad/coopMatMulAdd/coopMatStore are used for the
// hidden layers, so NO QCOM conversion is required there.
//
// Output-layer change vs the original: the original used coopmatToVectorQCOM +
// f16vec4 store (QCOM-only). To make this strategy runnable without
// VK_QCOM_cooperative_matrix_conversion we add a non-QCOM path that stores the
// [64 x 16] accumulator with a plain coopMatStore into a tiled [batch x 16]
// output buffer; the host reads the first 4 columns per sample. The QCOM path
// is kept under USE_QCOM_CONV for parity when the extension is present.
//
// Element types:  X_ELEM = f16vec4, W*_ELEM = float16_t, B*_ELEM = float16_t.
//   USE_QCOM_CONV=1 -> Y_ELEM = f16vec4   (Y is [batch x 4])
//   USE_QCOM_CONV=0 -> Y_ELEM = float16_t (Y is [batch x 16], tiled)
//
// Width (DIM_K = hiddenFeatures) supports 16/32/64.
// waves_per_group must be 1 (local_cache_x is per-subgroup).

inline const char* kMlpCoopLocalBody = R"(
#define cTILE_M 64
#define cTILE_K 16
#define cTILE_N hiddenFeatures
#define wTILE_K hiddenFeatures
#define DIM_K   hiddenFeatures

uint32_t group_idx = gl_WorkGroupID.x;
// This driver returns gl_SubgroupInvocationID == 0 for every lane (subgroup
// size is correctly 64); gl_LocalInvocationID.x is a correct 0..63 across the
// 64-thread single-subgroup workgroup, so use it as the per-fiber index.
uint32_t wave_idx  = 0u;
uint32_t fiber_idx = gl_LocalInvocationID.x;

// DIM_K (= hiddenFeatures) is a spec constant, which GLSL does not allow as a
// shared-array dimension (must be a compile-time constant). Declare the cache at
// the maximum supported width (64) so the size is a literal; only the first
// 64*DIM_K elements are used at runtime.
shared float16_t local_cache_x[64 * 64];

void tile_relu(inout coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_N, gl_MatrixUseAccumulator> acc) {
    [[unroll]] for (uint32_t l = 0; l < DIM_K; l++)
        acc[l] = max(acc[l], float16_t(0.0hf));
}

#if ADD_BIAS == 1
#define _BIAS_ADD_KxK(acc, B_BUF)                                                                        \
{                                                                                                        \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_N, gl_MatrixUseAccumulator> bias_tile;          \
    coopMatLoad(bias_tile, B_BUF.x, 0, cTILE_N, gl_CooperativeMatrixLayoutRowMajor);                    \
    acc = acc + bias_tile;                                                                               \
}
#define _BIAS_ADD_Kx4(acc, B_BUF)                                                                        \
{                                                                                                        \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, 16, gl_MatrixUseAccumulator> bias_tile;               \
    coopMatLoad(bias_tile, B_BUF.x, 0, 16, gl_CooperativeMatrixLayoutRowMajor);                         \
    acc = acc + bias_tile;                                                                               \
}
#else
#define _BIAS_ADD_KxK(acc, B_BUF)
#define _BIAS_ADD_Kx4(acc, B_BUF)
#endif

#if ACTIVATION == 1
#define _ACTIVATION_KxK(acc) tile_relu(acc);
#else
#define _ACTIVATION_KxK(acc)
#endif

void load_inputs_to_local() {
    const uint32_t feats_per_row = DIM_K / 4;
    const uint32_t xload_offset  = group_idx * 64 * feats_per_row;
    [[unroll]] for (uint32_t i = 0; i < feats_per_row; i++) {
        uint32_t index   = i * 64 + fiber_idx;
        f16vec4  v       = X.x[xload_offset + index];
        uint32_t s  = index / feats_per_row;
        uint32_t feat_g  = index % feats_per_row;
        local_cache_x[s * DIM_K + feat_g * 4 + 0] = v.x;
        local_cache_x[s * DIM_K + feat_g * 4 + 1] = v.y;
        local_cache_x[s * DIM_K + feat_g * 4 + 2] = v.z;
        local_cache_x[s * DIM_K + feat_g * 4 + 3] = v.w;
    }
}

#define LAYER_KxK(W_BUF, B_BUF)                                                                                          \
{                                                                                                                        \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_N, gl_MatrixUseAccumulator> cache_acc =                        \
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_N, gl_MatrixUseAccumulator>(0.0hf);                        \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_K, gl_MatrixUseA> cache_x;                                     \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_K, cTILE_N, gl_MatrixUseB> cache_w;                                     \
    [[unroll]] for (uint32_t step_k = 0; step_k < wTILE_K; step_k += cTILE_K) {                                        \
        coopMatLoad(cache_x, local_cache_x, step_k, DIM_K, gl_CooperativeMatrixLayoutRowMajor);                        \
        coopMatLoad(cache_w, W_BUF.x, step_k*DIM_K, DIM_K, gl_CooperativeMatrixLayoutRowMajor);                          \
        cache_acc = coopMatMulAdd(cache_x, cache_w, cache_acc);                                                         \
    }                                                                                                                    \
    _BIAS_ADD_KxK(cache_acc, B_BUF)                                                                                     \
    _ACTIVATION_KxK(cache_acc)                                                                                           \
    coopMatStore(cache_acc, local_cache_x, 0, DIM_K, gl_CooperativeMatrixLayoutRowMajor);                               \
}

#define LAYER_Kx4(W_BUF, B_BUF)                                                                                         \
{                                                                                                                        \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, 16, gl_MatrixUseAccumulator> cache_acc =                             \
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, 16, gl_MatrixUseAccumulator>(0.0hf);                             \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_K, gl_MatrixUseA> cache_x;                                     \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_K, 16, gl_MatrixUseB> cache_w;                                          \
    [[unroll]] for (uint32_t step_k = 0; step_k < wTILE_K; step_k += cTILE_K) {                                        \
        coopMatLoad(cache_x, local_cache_x, step_k, DIM_K, gl_CooperativeMatrixLayoutRowMajor);                        \
        coopMatLoad(cache_w, W_BUF.x, step_k*16, 16, gl_CooperativeMatrixLayoutRowMajor);                              \
        cache_acc = coopMatMulAdd(cache_x, cache_w, cache_acc);                                                         \
    }                                                                                                                    \
    _BIAS_ADD_Kx4(cache_acc, B_BUF)                                                                                     \
    PSTORE_Kx4(cache_acc)                                                                                               \
}

#if USE_QCOM_CONV
// QCOM path — extract first 4 fiber outputs, store as a vec4 to [batch x 4].
#define PSTORE_Kx4(acc)                                                                                                 \
{                                                                                                                        \
    float16_t fiber_out[16];                                                                                             \
    coopmatToVectorQCOM(acc, fiber_out);                                                                                 \
    uint32_t store_idx = group_idx * 64 + fiber_idx;                                                                    \
    _DBG_OR_STORE(store_idx, fiber_out)                                                                                  \
}
#else
// Non-QCOM path — store the whole [64 x 16] accumulator tile row-major to a
// tiled [batch x 16] output; host reads columns 0..3 of each sample row.
#define PSTORE_Kx4(acc)                                                                                                 \
{                                                                                                                        \
    coopMatStore(acc, Y.x, group_idx * 64 * 16, 16, gl_CooperativeMatrixLayoutRowMajor);                               \
}
#endif

void fusedMLP_FWDPass() {
    load_inputs_to_local();
    LAYER_KxK(W0, B0)
    LAYER_KxK(W1, B1)
    LAYER_KxK(W2, B2)
    LAYER_Kx4(W3, B3)
}
)";

// ---------------------------------------------------------------------------
// WIDE_IO variant: 12 -> W -> 10 (one hidden layer). Input and the hidden
// activation live in LDS (local_cache_x). Assembled with X_ELEM = f16vec4,
// WIDE_IO_OUT defined, and host defines IN_K (=16), OUT_N (=16), OUT_CH (=10).
// The input row is 12 features padded to IN_K = 16 in the host buffer; the
// cache holds [64 x IN_K] then is overwritten with the [64 x DIM_W] hidden.
inline const char* kMlpCoopLocalWideIOBody = R"(
#define cTILE_M 64
#define cTILE_K 16
#define DIM_W   hiddenFeatures

uint32_t group_idx = gl_WorkGroupID.x;
uint32_t fiber_idx = gl_LocalInvocationID.x;   // driver-safe (subgroup id is 0 on every lane)

// Sized at the max footprint: input tile is [64 x IN_K=16], hidden is [64 x 64].
shared float16_t local_cache_x[64 * 64];

void tile_relu_w(inout coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, DIM_W, gl_MatrixUseAccumulator> acc) {
    [[unroll]] for (uint32_t l = 0; l < DIM_W; l++)
        acc[l] = max(acc[l], float16_t(0.0hf));
}

// Load [64 x IN_K] input into the cache. Host input row stride is IN_K (=16);
// X is viewed as f16vec4 so IN_K/4 vec4 per row. IN_K == 16 => 4 vec4 per row,
// exactly one vec4 per fiber-iteration across 64 fibers (64*4 = 256 = 64*IN_K/4).
void load_inputs_w() {
    const uint32_t feats_per_row = IN_K / 4u;                 // 4
    const uint32_t xload_offset  = group_idx * 64u * feats_per_row;
    [[unroll]] for (uint32_t i = 0; i < feats_per_row; i++) {
        uint32_t index  = i * 64u + fiber_idx;
        f16vec4  v      = X.x[xload_offset + index];
        uint32_t s      = index / feats_per_row;
        uint32_t feat_g = index % feats_per_row;
        local_cache_x[s * IN_K + feat_g * 4u + 0u] = v.x;
        local_cache_x[s * IN_K + feat_g * 4u + 1u] = v.y;
        local_cache_x[s * IN_K + feat_g * 4u + 2u] = v.z;
        local_cache_x[s * IN_K + feat_g * 4u + 3u] = v.w;
    }
}

void fusedMLP_FWDPass() {
    load_inputs_w();

    // ---- L0: hidden[64 x DIM_W] = cache[64 x IN_K] * W0[IN_K x DIM_W] ----
    {
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, DIM_W, gl_MatrixUseAccumulator> acc =
            coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, DIM_W, gl_MatrixUseAccumulator>(0.0hf);
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_K, gl_MatrixUseA> cx;
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_K, DIM_W,   gl_MatrixUseB> cw;
        coopMatLoad(cx, local_cache_x, 0, IN_K,  gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(cw, W0.x,          0, DIM_W, gl_CooperativeMatrixLayoutRowMajor);
        acc = coopMatMulAdd(cx, cw, acc);
#if ADD_BIAS == 1
        {
            coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, DIM_W, gl_MatrixUseAccumulator> bt;
            coopMatLoad(bt, B0.x, 0, DIM_W, gl_CooperativeMatrixLayoutRowMajor);
            acc = acc + bt;
        }
#endif
#if ACTIVATION == 1
        tile_relu_w(acc);
#endif
        coopMatStore(acc, local_cache_x, 0, DIM_W, gl_CooperativeMatrixLayoutRowMajor);
    }

    // ---- L1 (output): [64 x OUT_N] = cache[64 x DIM_W] * W1[DIM_W x OUT_N] ----
    {
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, OUT_N, gl_MatrixUseAccumulator> acc =
            coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, OUT_N, gl_MatrixUseAccumulator>(0.0hf);
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_K, gl_MatrixUseA> cx;
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_K, OUT_N,   gl_MatrixUseB> cw;
        [[unroll]] for (uint32_t step_k = 0; step_k < DIM_W; step_k += cTILE_K) {
            coopMatLoad(cx, local_cache_x, step_k,        DIM_W, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(cw, W1.x,          step_k * OUT_N, OUT_N, gl_CooperativeMatrixLayoutRowMajor);
            acc = coopMatMulAdd(cx, cw, acc);
        }
#if ADD_BIAS == 1
        {
            coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, OUT_N, gl_MatrixUseAccumulator> bt;
            coopMatLoad(bt, B1.x, 0, OUT_N, gl_CooperativeMatrixLayoutRowMajor);
            acc = acc + bt;
        }
#endif
        float16_t fiber_out[16];
        coopmatToVectorQCOM(acc, fiber_out);
        uint32_t store_idx = group_idx * 64u + fiber_idx;
        WIDE_STORE_10(store_idx, fiber_out)
    }
}
)";
