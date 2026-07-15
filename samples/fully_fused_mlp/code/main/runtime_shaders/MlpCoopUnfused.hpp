//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

// Unfused (per-layer dispatch) baseline — single-layer coopmat kernel.
//
// This is the SAME coopmat global-memory matmul as one layer of the Global
// fused strategy (MlpCoopGlobal.hpp), but each network layer runs as its own
// dispatch with the intermediate activations round-tripping through global
// memory between dispatches. It exists as a baseline to measure the cost that
// fusion saves (extra global round-trips + per-dispatch launch overhead + loss
// of cross-layer on-chip reuse).
//
// The kernel is LAYER-AGNOSTIC: it always reads its input from X (binding 1),
// its weights from W0 (binding 2) and bias from B0 (binding 10), and writes its
// output to Y (binding 18). The HOST rebinds those four buffers per layer
// (src activation, that layer's weights/bias, dst activation) — see
// FusedMlpRunner::RunUnfused. So the shader needs no layer index.
//
// Two kinds, selected by LAYER_KIND (host injects it):
//   LAYER_KIND=0  hidden layer  [64 x DIM_K] = [64 x DIM_K] * [DIM_K x DIM_K]
//                 + bias tile, + ReLU (if ACTIVATION), coopMatStore -> Y[batch x DIM_K]
//                 (Y_ELEM = float16_t)
//   LAYER_KIND=1  output layer  [64 x 16]    = [64 x DIM_K] * [DIM_K x 16]
//                 + bias, NO ReLU, coopmatToVectorQCOM -> f16vec4 Y[batch x 4]
//                 (Y_ELEM = f16vec4)
//
// Element types (host-injected before kCoopBuffers):
//   X_ELEM = float16_t (activation buffers are plain fp16 [batch x width])
//   W0_ELEM/B0_ELEM = float16_t
//   LAYER_KIND=0 -> Y_ELEM = float16_t ; LAYER_KIND=1 -> Y_ELEM = f16vec4
//
// Driver-safe indexing: per-fiber scatter uses gl_LocalInvocationID.x (this
// device returns gl_SubgroupInvocationID == 0 on every lane).

inline const char* kMlpCoopUnfusedBody = R"(
#define cTILE_M 64
#define cTILE_K 16
#define cTILE_N hiddenFeatures
#define wTILE_K hiddenFeatures
#define DIM_K   hiddenFeatures

#ifndef LAYER_KIND
#define LAYER_KIND 0
#endif

void tile_relu(inout coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_N, gl_MatrixUseAccumulator> acc) {
    [[unroll]] for (uint32_t l = 0; l < DIM_K; l++)
        acc[l] = max(acc[l], float16_t(0.0hf));
}

#if ADD_BIAS == 1
#define _BIAS_ADD_KxK(acc)                                                                 \
{                                                                                          \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_N, gl_MatrixUseAccumulator> bt;   \
    coopMatLoad(bt, B0.x, 0, cTILE_N, gl_CooperativeMatrixLayoutRowMajor);                 \
    acc = acc + bt;                                                                        \
}
#define _BIAS_ADD_Kx4(acc)                                                                 \
{                                                                                          \
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, 16, gl_MatrixUseAccumulator> bt;        \
    coopMatLoad(bt, B0.x, 0, 16, gl_CooperativeMatrixLayoutRowMajor);                      \
    acc = acc + bt;                                                                        \
}
#else
#define _BIAS_ADD_KxK(acc)
#define _BIAS_ADD_Kx4(acc)
#endif

#if ACTIVATION == 1
#define _ACTIVATION_KxK(acc) tile_relu(acc);
#else
#define _ACTIVATION_KxK(acc)
#endif

void fusedMLP_FWDPass() {
    uint32_t group_idx = gl_WorkGroupID.x;
    uint32_t xrow      = group_idx * 64u;   // this workgroup's 64-sample tile base

#if LAYER_KIND == 0
    // ---- hidden layer: X[64 x DIM_K] * W0[DIM_K x DIM_K] -> Y[64 x DIM_K] ----
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_N, gl_MatrixUseAccumulator> acc =
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_N, gl_MatrixUseAccumulator>(0.0hf);
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_K, gl_MatrixUseA> cx;
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_K, cTILE_N, gl_MatrixUseB> cw;
    for (uint32_t step_k = 0; step_k < wTILE_K; step_k += cTILE_K) {
        coopMatLoad(cx, X.x,  xrow * DIM_K + step_k, DIM_K, gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(cw, W0.x, step_k * DIM_K,        DIM_K, gl_CooperativeMatrixLayoutRowMajor);
        acc = coopMatMulAdd(cx, cw, acc);
    }
    _BIAS_ADD_KxK(acc)
    _ACTIVATION_KxK(acc)
    coopMatStore(acc, Y.x, xrow * DIM_K, DIM_K, gl_CooperativeMatrixLayoutRowMajor);
#else
    // ---- output layer: X[64 x DIM_K] * W0[DIM_K x 16] -> Y[batch x 4] (QCOM) ----
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, 16, gl_MatrixUseAccumulator> acc =
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, 16, gl_MatrixUseAccumulator>(0.0hf);
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_K, gl_MatrixUseA> cx;
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_K, 16,      gl_MatrixUseB> cw;
    [[unroll]] for (uint32_t step_k = 0; step_k < wTILE_K; step_k += cTILE_K) {
        coopMatLoad(cx, X.x,  xrow * DIM_K + step_k, DIM_K, gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(cw, W0.x, step_k * 16,           16,    gl_CooperativeMatrixLayoutRowMajor);
        acc = coopMatMulAdd(cx, cw, acc);
    }
    _BIAS_ADD_Kx4(acc)
    float16_t fiber_out[16];
    coopmatToVectorQCOM(acc, fiber_out);
    uint32_t store_idx = group_idx * 64u + gl_LocalInvocationID.x;   // driver-safe fiber index
    Y.x[store_idx] = f16vec4(fiber_out[0], fiber_out[1], fiber_out[2], fiber_out[3]);
#endif
}
)";

// ---------------------------------------------------------------------------
// WIDE_IO unfused kinds (host selects via LAYER_KIND):
//   LAYER_KIND=2  input layer  hidden[64 x W] = X[64 x IN_K] * W0[IN_K x W]
//                 + bias, + ReLU (if ACTIVATION), coopMatStore -> Y[batch x W]
//                 (Y_ELEM = float16_t; IN_K == 16 so one matmul step).
//   LAYER_KIND=3  output layer out[64 x OUT_N] = X[64 x W] * W0[W x OUT_N]
//                 + bias, NO ReLU, coopmatToVectorQCOM -> WIDE_STORE_10 (10 ch).
//                 (WIDE_IO_OUT defined; src X is the [batch x W] hidden.)
// Host defines IN_K (=16), OUT_N (=16); DIM_W = hiddenFeatures = W.
inline const char* kMlpCoopUnfusedWideIOBody = R"(
#define cTILE_M 64
#define cTILE_K 16
#define DIM_W   hiddenFeatures

#ifndef LAYER_KIND
#define LAYER_KIND 2
#endif

void tile_relu_w(inout coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, DIM_W, gl_MatrixUseAccumulator> acc) {
    [[unroll]] for (uint32_t l = 0; l < DIM_W; l++)
        acc[l] = max(acc[l], float16_t(0.0hf));
}

void fusedMLP_FWDPass() {
    uint32_t group_idx = gl_WorkGroupID.x;
    uint32_t xrow      = group_idx * 64u;

#if LAYER_KIND == 2
    // ---- input layer: X[64 x IN_K] * W0[IN_K x DIM_W] -> Y[64 x DIM_W] ----
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
    coopMatStore(acc, Y.x, xrow * DIM_W, DIM_W, gl_CooperativeMatrixLayoutRowMajor);
#else
    // ---- output layer: X[64 x DIM_W] * W0[DIM_W x OUT_N] -> WIDE_STORE_10 ----
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, OUT_N, gl_MatrixUseAccumulator> acc =
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, OUT_N, gl_MatrixUseAccumulator>(0.0hf);
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, cTILE_K, gl_MatrixUseA> cx;
    coopmat<float16_t, gl_ScopeSubgroup, cTILE_K, OUT_N,   gl_MatrixUseB> cw;
    [[unroll]] for (uint32_t step_k = 0; step_k < DIM_W; step_k += cTILE_K) {
        coopMatLoad(cx, X.x,  xrow * DIM_W + step_k, DIM_W, gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(cw, W0.x, step_k * OUT_N,        OUT_N, gl_CooperativeMatrixLayoutRowMajor);
        acc = coopMatMulAdd(cx, cw, acc);
    }
#if ADD_BIAS == 1
    {
        coopmat<float16_t, gl_ScopeSubgroup, cTILE_M, OUT_N, gl_MatrixUseAccumulator> bt;
        coopMatLoad(bt, B0.x, 0, OUT_N, gl_CooperativeMatrixLayoutRowMajor);
        acc = acc + bt;
    }
#endif
    float16_t fiber_out[16];
    coopmatToVectorQCOM(acc, fiber_out);
    uint32_t store_idx = group_idx * 64u + gl_LocalInvocationID.x;   // driver-safe fiber index
    WIDE_STORE_10(store_idx, fiber_out)
#endif
}
)";
