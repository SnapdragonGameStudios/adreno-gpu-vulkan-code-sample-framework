//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

// GPR-fusing strategy (a.k.a. coopvec).
//
// Body of Vulkan_MLP's shader/coopmat_glsl/fwdPass_coopvec_global.comp (verbatim).
// The whole MLP state lives in per-fiber registers (fiber_xvector[16]); the
// QCOM conversion ops vectorToCoopmatQCOM / coopmatToVectorQCOM move it in and
// out of coopmat tiles each layer.  Requires GL_QCOM_cooperative_matrix_conversion
// (USE_QCOM_CONV must be 1) and is 16-wide only.
//
// Element types for this strategy:
//   X_ELEM = f16vec4, W*_ELEM = float16_t, B*_ELEM = f16vec4, Y_ELEM = f16vec4
// Output Y is [batch x 4] (store_output writes the first 4 channels).
//
// Assembled by FusedMlpRunner::BuildCoopmatSource() with USE_QCOM_CONV=1.

inline const char* kMlpCoopGprBody = R"(
uint32_t group_idx = gl_WorkGroupID.x;
// NOTE: this driver returns gl_SubgroupInvocationID == 0 for every lane (while
// gl_SubgroupSize correctly reports 64), which collapsed every lane onto sample
// 0. gl_LocalInvocationID.x is a correct 0..63 across the 64-thread, single-
// subgroup workgroup, so use it as the per-lane (fiber) index instead.
uint32_t wave_idx  = 0u;
uint32_t fiber_idx = gl_LocalInvocationID.x;

float16_t fiber_xvector[16];

shared float16_t lm_w0[16 * 16];
shared float16_t lm_w1[16 * 16];
shared float16_t lm_w2[16 * 16];
shared float16_t lm_w3[16 * 16];

void load_xvector() {
    uint32_t xrow = (group_idx * waves_per_group + wave_idx) * 64 + fiber_idx;
    uint32_t xEl  = xrow * 4;
    [[unroll]] for (uint32_t i = 0; i < 4; i++) {
        f16vec4 v = X.x[xEl + i];
        fiber_xvector[i*4 + 0] = v.x;
        fiber_xvector[i*4 + 1] = v.y;
        fiber_xvector[i*4 + 2] = v.z;
        fiber_xvector[i*4 + 3] = v.w;
    }
}

void load_w_to_local() {
    [[unroll]] for (uint32_t i = 0; i < 256 / 64; i++) { uint32_t idx = i*64 + fiber_idx; lm_w0[idx] = W0.x[idx]; }
    [[unroll]] for (uint32_t i = 0; i < 256 / 64; i++) { uint32_t idx = i*64 + fiber_idx; lm_w1[idx] = W1.x[idx]; }
    [[unroll]] for (uint32_t i = 0; i < 256 / 64; i++) { uint32_t idx = i*64 + fiber_idx; lm_w2[idx] = W2.x[idx]; }
    [[unroll]] for (uint32_t i = 0; i < 256 / 64; i++) { uint32_t idx = i*64 + fiber_idx; lm_w3[idx] = W3.x[idx]; }
}

void store_output() {
    uint32_t out_idx = group_idx * waves_per_group * 64 + wave_idx * 64 + fiber_idx;
#if DEBUG_FIBER_INDEX
    // Debug marker (see MlpCoopCommon.hpp): scatter by workgroup thread index.
    // x=localInvocation, y=workgroupSize, z=subgroupInvocation, w=subgroupSize.
    uint32_t _dbg = gl_WorkGroupID.x * 64u + gl_LocalInvocationID.x;
    Y.x[_dbg] = f16vec4(float16_t(gl_LocalInvocationID.x), float16_t(gl_WorkGroupSize.x),
                        float16_t(gl_SubgroupInvocationID), float16_t(gl_SubgroupSize));
#else
    Y.x[out_idx] = f16vec4(fiber_xvector[0], fiber_xvector[1], fiber_xvector[2], fiber_xvector[3]);
#endif
}

#define GPR_BIAS(B_BUF) \
    [[unroll]] for (uint32_t i = 0; i < 4; i++) { \
        f16vec4 b = B_BUF.x[i]; \
        f16vec4 f = f16vec4(fiber_xvector[i*4+0], fiber_xvector[i*4+1], fiber_xvector[i*4+2], fiber_xvector[i*4+3]) + b; \
        fiber_xvector[i*4+0] = f.x; fiber_xvector[i*4+1] = f.y; fiber_xvector[i*4+2] = f.z; fiber_xvector[i*4+3] = f.w; \
    }

#define GPR_RELU \
    [[unroll]] for (uint32_t i = 0; i < 16; i++) fiber_xvector[i] = max(fiber_xvector[i], float16_t(0.0hf));

#define GPR_LAYER(W_LM, B_BUF) \
{ \
    coopmat<float16_t, gl_ScopeSubgroup, 64, 16, gl_MatrixUseA>           cx; \
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>           cw; \
    coopmat<float16_t, gl_ScopeSubgroup, 64, 16, gl_MatrixUseAccumulator> ca = \
        coopmat<float16_t, gl_ScopeSubgroup, 64, 16, gl_MatrixUseAccumulator>(0.0hf); \
    vectorToCoopmatQCOM(fiber_xvector, cx); \
    coopMatLoad(cw, W_LM, 0, 16, gl_CooperativeMatrixLayoutRowMajor); \
    ca = coopMatMulAdd(cx, cw, ca); \
    coopmatToVectorQCOM(ca, fiber_xvector); \
}

void fusedMLP_FWDPass() {
    load_xvector();
    load_w_to_local();

    // L0
    GPR_LAYER(lm_w0, B0)
#if ADD_BIAS
    GPR_BIAS(B0)
#endif
#if ACTIVATION == 1
    GPR_RELU
#endif

    // L1
    GPR_LAYER(lm_w1, B1)
#if ADD_BIAS
    GPR_BIAS(B1)
#endif
#if ACTIVATION == 1
    GPR_RELU
#endif

    // L2
    GPR_LAYER(lm_w2, B2)
#if ADD_BIAS
    GPR_BIAS(B2)
#endif
#if ACTIVATION == 1
    GPR_RELU
#endif

    // L3 (output) — no activation
    GPR_LAYER(lm_w3, B3)
#if ADD_BIAS
    {
        f16vec4 b = B3.x[0];
        f16vec4 f = f16vec4(fiber_xvector[0], fiber_xvector[1], fiber_xvector[2], fiber_xvector[3]) + b;
        fiber_xvector[0] = f.x; fiber_xvector[1] = f.y; fiber_xvector[2] = f.z; fiber_xvector[3] = f.w;
    }
#endif

    store_output();
}
)";

// ---------------------------------------------------------------------------
// WIDE_IO variant: 12 -> 16 -> 10 (W=16 only; the coopvec state is 16-wide).
// Two GPR layers. The input row is 12 features padded to 16 in the host buffer
// (X is f16vec4[], 4 vec4 = 16 halfs per sample; channels 12..15 are zero).
// Bias B*_ELEM = f16vec4; output stored via WIDE_STORE_10 (WIDE_IO_OUT).
inline const char* kMlpCoopGprWideIOBody = R"(
uint32_t group_idx = gl_WorkGroupID.x;
uint32_t wave_idx  = 0u;
uint32_t fiber_idx = gl_LocalInvocationID.x;   // driver-safe fiber index

float16_t fiber_xvector[16];

shared float16_t lm_w0[16 * 16];   // input->hidden [16 x 16] (12 real rows, padded)
shared float16_t lm_w1[16 * 16];   // hidden->output [16 x 16] (10 real cols, padded)

void load_xvector() {
    uint32_t xrow = (group_idx * waves_per_group + wave_idx) * 64 + fiber_idx;
    uint32_t xEl  = xrow * 4;
    [[unroll]] for (uint32_t i = 0; i < 4; i++) {
        f16vec4 v = X.x[xEl + i];
        fiber_xvector[i*4 + 0] = v.x;
        fiber_xvector[i*4 + 1] = v.y;
        fiber_xvector[i*4 + 2] = v.z;
        fiber_xvector[i*4 + 3] = v.w;
    }
}

void load_w_to_local() {
    [[unroll]] for (uint32_t i = 0; i < 256 / 64; i++) { uint32_t idx = i*64 + fiber_idx; lm_w0[idx] = W0.x[idx]; }
    [[unroll]] for (uint32_t i = 0; i < 256 / 64; i++) { uint32_t idx = i*64 + fiber_idx; lm_w1[idx] = W1.x[idx]; }
}

#define GPR_LAYER(W_LM) \
{ \
    coopmat<float16_t, gl_ScopeSubgroup, 64, 16, gl_MatrixUseA>           cx; \
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>           cw; \
    coopmat<float16_t, gl_ScopeSubgroup, 64, 16, gl_MatrixUseAccumulator> ca = \
        coopmat<float16_t, gl_ScopeSubgroup, 64, 16, gl_MatrixUseAccumulator>(0.0hf); \
    vectorToCoopmatQCOM(fiber_xvector, cx); \
    coopMatLoad(cw, W_LM, 0, 16, gl_CooperativeMatrixLayoutRowMajor); \
    ca = coopMatMulAdd(cx, cw, ca); \
    coopmatToVectorQCOM(ca, fiber_xvector); \
}

#define GPR_BIAS4(B_BUF) \
    [[unroll]] for (uint32_t i = 0; i < 4; i++) { \
        f16vec4 b = B_BUF.x[i]; \
        f16vec4 f = f16vec4(fiber_xvector[i*4+0], fiber_xvector[i*4+1], fiber_xvector[i*4+2], fiber_xvector[i*4+3]) + b; \
        fiber_xvector[i*4+0] = f.x; fiber_xvector[i*4+1] = f.y; fiber_xvector[i*4+2] = f.z; fiber_xvector[i*4+3] = f.w; \
    }

#define GPR_RELU16 \
    [[unroll]] for (uint32_t i = 0; i < 16; i++) fiber_xvector[i] = max(fiber_xvector[i], float16_t(0.0hf));

void fusedMLP_FWDPass() {
    load_xvector();
    load_w_to_local();

    // L0 (input->hidden): bias + ReLU
    GPR_LAYER(lm_w0)
#if ADD_BIAS
    GPR_BIAS4(B0)
#endif
#if ACTIVATION == 1
    GPR_RELU16
#endif

    // L1 (output): bias, linear
    GPR_LAYER(lm_w1)
#if ADD_BIAS
    GPR_BIAS4(B1)
#endif

    uint32_t store_idx = group_idx * waves_per_group * 64 + wave_idx * 64 + fiber_idx;
    WIDE_STORE_10(store_idx, fiber_xvector)
}
)";
