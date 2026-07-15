//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

// Global-fusing strategy.
//
// Body of Vulkan_MLP's shader/coopmat_glsl/fwdPass_coopmat_global.comp.
// Hidden state ping-pongs through global memory (in-place X); no LDS cache.
// Only coopMatLoad/coopMatMulAdd/coopMatStore are used for the hidden layers,
// so NO QCOM conversion is required there.
//
// Output-layer change vs the original: same non-QCOM tiled-store path as the
// local strategy (see MlpCoopLocal.hpp). QCOM path kept under USE_QCOM_CONV.
//
// Element types:  X_ELEM = float16_t, W*_ELEM = float16_t, B*_ELEM = float16_t.
//   USE_QCOM_CONV=1 -> Y_ELEM = f16vec4   (Y is [batch x 4])
//   USE_QCOM_CONV=0 -> Y_ELEM = float16_t (Y is [batch x 16], tiled)
//
// Width (DIM_K = hiddenFeatures) supports 16/32/64.
// USE_X_DUMMY is forced 0 (in-place X ping-pong); X is re-uploaded by the host
// before each validated dispatch is not needed because we validate a single
// dispatch.

inline const char* kMlpCoopGlobalBody = R"(
#define cTILE_M 64
#define cTILE_K 16
#define cTILE_N hiddenFeatures
#define wTILE_K hiddenFeatures
#define DIM_K   hiddenFeatures

void tile_relu(inout coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_N, gl_MatrixUseAccumulator> acc) {
    [[unroll]] for (uint32_t l = 0; l < DIM_K; l++)
        acc[l] = max(acc[l], float16_t(0.0hf));
}

#if ACTIVATION == 1
#define _ACTIVATION_KxK(acc) tile_relu(acc);
#else
#define _ACTIVATION_KxK(acc)
#endif

#if ADD_BIAS == 1
#define _BIAS_ADD_KxK(acc, B_BUF)                                                          \
{                                                                                          \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_N,                               \
            gl_MatrixUseAccumulator> bias_tile;                                            \
    coopMatLoad(bias_tile, B_BUF.x, 0, cTILE_N, gl_CooperativeMatrixLayoutRowMajor);     \
    acc = acc + bias_tile;                                                                 \
}
#define _BIAS_ADD_Kx4(acc, B_BUF)                                                          \
{                                                                                          \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, 16,                                    \
            gl_MatrixUseAccumulator> bias_tile;                                            \
    coopMatLoad(bias_tile, B_BUF.x, 0, 16, gl_CooperativeMatrixLayoutRowMajor);           \
    acc = acc + bias_tile;                                                                 \
}
#else
#define _BIAS_ADD_KxK(acc, B_BUF)
#define _BIAS_ADD_Kx4(acc, B_BUF)
#endif

#define DO_LAYER_KxK_FIRST(W_BUF, B_BUF, xrow)                                            \
{                                                                                          \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_N,                               \
            gl_MatrixUseAccumulator> acc =                                                 \
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_N,                           \
                gl_MatrixUseAccumulator>(0.0hf);                                           \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_K, gl_MatrixUseA> cx;            \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_K, cTILE_N, gl_MatrixUseB> cw;            \
    for (uint32_t step_k = 0; step_k < wTILE_K; step_k += cTILE_K) {                    \
        coopMatLoad(cx, X.x,     xrow * DIM_K + step_k, DIM_K,                           \
                    gl_CooperativeMatrixLayoutRowMajor);                                   \
        coopMatLoad(cw, W_BUF.x, step_k * DIM_K,        DIM_K,                           \
                    gl_CooperativeMatrixLayoutRowMajor);                                   \
        acc = coopMatMulAdd(cx, cw, acc);                                                 \
    }                                                                                      \
    _BIAS_ADD_KxK(acc, B_BUF)                                                             \
    _ACTIVATION_KxK(acc)                                                                   \
    coopMatStore(acc, X.x, xrow * DIM_K, DIM_K,                                          \
                 gl_CooperativeMatrixLayoutRowMajor);                                      \
}

#define DO_LAYER_KxK(W_BUF, B_BUF, xrow)                                                  \
{                                                                                          \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_N,                               \
            gl_MatrixUseAccumulator> acc =                                                 \
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_N,                           \
                gl_MatrixUseAccumulator>(0.0hf);                                           \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_K, gl_MatrixUseA> cx;            \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_K, cTILE_N, gl_MatrixUseB> cw;            \
    for (uint32_t step_k = 0; step_k < wTILE_K; step_k += cTILE_K) {                    \
        coopMatLoad(cx, X.x, xrow * DIM_K + step_k, DIM_K,                               \
                    gl_CooperativeMatrixLayoutRowMajor);                                   \
        coopMatLoad(cw, W_BUF.x, step_k * DIM_K,        DIM_K,                           \
                    gl_CooperativeMatrixLayoutRowMajor);                                   \
        acc = coopMatMulAdd(cx, cw, acc);                                                 \
    }                                                                                      \
    _BIAS_ADD_KxK(acc, B_BUF)                                                             \
    _ACTIVATION_KxK(acc)                                                                   \
    coopMatStore(acc, X.x, xrow * DIM_K, DIM_K,                                          \
                 gl_CooperativeMatrixLayoutRowMajor);                                      \
}

#define DO_LAYER_Kx4(W_BUF, B_BUF, xrow)                                                  \
{                                                                                          \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, 16,                                    \
            gl_MatrixUseAccumulator> acc =                                                 \
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, 16,                                \
                gl_MatrixUseAccumulator>(0.0hf);                                           \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_K, gl_MatrixUseA> cx;            \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_K, 16,      gl_MatrixUseB> cw;            \
    [[unroll]] for (uint32_t step_k = 0; step_k < wTILE_K; step_k += cTILE_K) {         \
        coopMatLoad(cx, X.x, xrow * DIM_K + step_k, DIM_K,                               \
                    gl_CooperativeMatrixLayoutRowMajor);                                   \
        coopMatLoad(cw, W_BUF.x, step_k * 16,           16,                              \
                    gl_CooperativeMatrixLayoutRowMajor);                                   \
        acc = coopMatMulAdd(cx, cw, acc);                                                 \
    }                                                                                      \
    _BIAS_ADD_Kx4(acc, B_BUF)                                                             \
    PSTORE_Kx4(acc, xrow)                                                                 \
}

#if USE_QCOM_CONV
#define PSTORE_Kx4(acc, xrow)                                                             \
{                                                                                          \
    float16_t fiber_out[16];                                                               \
    coopmatToVectorQCOM(acc, fiber_out);                                                   \
    /* gl_SubgroupInvocationID is 0 on every lane on this driver; scatter by the   */     \
    /* workgroup thread index instead (0..63 across the single 64-wide subgroup).  */     \
    uint32_t store_idx = group_idx * 64u + gl_LocalInvocationID.x;                        \
    _DBG_OR_STORE(store_idx, fiber_out)                                                    \
}
#else
// Non-QCOM — store the [64 x 16] accumulator tile to a tiled [batch x 16] Y.
#define PSTORE_Kx4(acc, xrow)                                                             \
{                                                                                          \
    coopMatStore(acc, Y.x, xrow * 16, 16, gl_CooperativeMatrixLayoutRowMajor);           \
}
#endif

void fusedMLP_FWDPass() {
    uint32_t group_idx = gl_WorkGroupID.x;
    // waves_per_group is 1 and there is a single subgroup per workgroup; the
    // tile base row is just the workgroup's 64-sample block. (Avoid gl_SubgroupID
    // here — subgroup builtins are unreliable on this driver.)
    uint32_t xrow      = group_idx * 64u;

    DO_LAYER_KxK_FIRST(W0, B0, xrow)
    DO_LAYER_KxK(W1, B1, xrow)
    DO_LAYER_KxK(W2, B2, xrow)
    DO_LAYER_Kx4(W3, B3, xrow)
}
)";

// ---------------------------------------------------------------------------
// WIDE_IO variant: 12 -> W -> 10 (one hidden layer). Same global-memory coopmat
// math, but a 2-matrix chain and rectangular tiles. Assembled with X_ELEM =
// float16_t, WIDE_IO_OUT defined (binding 18 is the WideOutRow struct), and host
// defines IN_K (=16, padded from 12), OUT_N (=16, padded from 10), OUT_CH (=10).
// The hidden activation round-trips through the scratch buffer at binding 19
// (Xdummy, sized [batch x W]) since in-place X ping-pong needs in == hidden.
inline const char* kMlpCoopGlobalWideIOBody = R"(
#define cTILE_M 64
#define cTILE_K 16
#define DIM_W   hiddenFeatures

void tile_relu_w(inout coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, DIM_W, gl_MatrixUseAccumulator> acc) {
    [[unroll]] for (uint32_t l = 0; l < DIM_W; l++)
        acc[l] = max(acc[l], float16_t(0.0hf));
}

void fusedMLP_FWDPass() {
    uint32_t group_idx = gl_WorkGroupID.x;
    uint32_t xrow      = group_idx * 64u;

    // ---- L0: hidden[64 x DIM_W] = X[64 x IN_K] * W0[IN_K x DIM_W] ----
    // IN_K == cTILE_K (16), so a single matmul step (no K loop).
    {
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, DIM_W, gl_MatrixUseAccumulator> acc =
            coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, DIM_W, gl_MatrixUseAccumulator>(0.0hf);
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_K, gl_MatrixUseA> cx;
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_K, DIM_W,   gl_MatrixUseB> cw;
        coopMatLoad(cx, X.x,  xrow * IN_K, IN_K,  gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(cw, W0.x, 0,           DIM_W, gl_CooperativeMatrixLayoutRowMajor);
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
        coopMatStore(acc, Xdummy.x, xrow * DIM_W, DIM_W, gl_CooperativeMatrixLayoutRowMajor);
    }

    // ---- L1 (output): [64 x OUT_N] = hidden[64 x DIM_W] * W1[DIM_W x OUT_N] ----
    {
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, OUT_N, gl_MatrixUseAccumulator> acc =
            coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, OUT_N, gl_MatrixUseAccumulator>(0.0hf);
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_K, gl_MatrixUseA> cx;
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_K, OUT_N,   gl_MatrixUseB> cw;
        [[unroll]] for (uint32_t step_k = 0; step_k < DIM_W; step_k += cTILE_K) {
            coopMatLoad(cx, Xdummy.x, xrow * DIM_W + step_k, DIM_W, gl_CooperativeMatrixLayoutRowMajor);
            coopMatLoad(cw, W1.x,     step_k * OUT_N,         OUT_N, gl_CooperativeMatrixLayoutRowMajor);
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
        uint32_t store_idx = group_idx * 64u + gl_LocalInvocationID.x;   // driver-safe fiber index
        WIDE_STORE_10(store_idx, fiber_out)
    }
}
)";
