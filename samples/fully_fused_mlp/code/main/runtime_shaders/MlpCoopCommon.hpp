//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

// Shared GLSL fragments for the cooperative-matrix MLP shaders.
//
// Vulkan_MLP composes its coopmat shaders from an include chain
//   main.comp -> extensions.glsl + <experiment>.comp -> buffers.glsl
// RuntimeShader compiles a single in-memory string with NO -I include path, so
// FusedMlpRunner::BuildCoopmatSource() concatenates these fragments instead:
//
//   "#version 460\n"                         (RuntimeShader injects the -D defines here)
//   + kCoopExtensions                        (all #extension lines; QCOM gated by USE_QCOM_CONV)
//   + <per-strategy element-type #defines>   (X_ELEM / W*_ELEM / B*_ELEM / Y_ELEM)
//   + kCoopSpecConstants                     (local size + spec constants + ADD_BIAS/ACTIVATION defaults)
//   + kCoopBuffers                           (buffers.glsl, verbatim)
//   + <per-strategy body>                    (the experiment .comp body)
//   + kCoopMain                              ("void main(){ fusedMLP_FWDPass(); }")
//
// Injected defines (RuntimeShader::AddDefine): ADD_BIAS, ACTIVATION,
// UNIFORM_BIAS, USE_QCOM_CONV.

// All #extension directives. GL_QCOM_cooperative_matrix_conversion is required
// only for the GPR/coopvec strategy (USE_QCOM_CONV=1).
inline const char* kCoopExtensions = R"(
#ifndef USE_QCOM_CONV
#define USE_QCOM_CONV 0
#endif

#extension GL_KHR_cooperative_matrix                        : require
#extension GL_KHR_shader_subgroup_basic                     : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16  : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32    : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32  : require
#extension GL_KHR_memory_scope_semantics                    : require
#extension GL_EXT_control_flow_attributes                   : enable
#extension GL_EXT_scalar_block_layout                       : require
#if USE_QCOM_CONV
#extension GL_QCOM_cooperative_matrix_conversion            : require
#endif
)";

// Local-size + spec constants + compile-time bias/activation defaults.
// NOTE: the workgroup size is a LITERAL 64 (not local_size_x_id). Using a
// spec-constant workgroup size emits the SPIR-V LocalSizeId execution mode,
// which requires Vulkan 1.3 + the maintenance4 feature and a SPIR-V 1.6 target.
// These shaders always run exactly 64 fibers per workgroup (waves_per_group=1),
// so a literal size lets us compile to SPIR-V 1.5 / Vulkan 1.2 and avoid forcing
// a higher device API version. constant_ids 3..6 remain spec constants.
inline const char* kCoopSpecConstants = R"(
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 3) const uint waves_per_group  = 1;
layout(constant_id = 4) const uint inFeatures       = 16;
layout(constant_id = 5) const uint hiddenFeatures   = 16;
layout(constant_id = 6) const uint outFeatures      = 4;

#ifndef ADD_BIAS
#define ADD_BIAS 0
#endif
#ifndef ACTIVATION
#define ACTIVATION 0
#endif
#ifndef UNIFORM_BIAS
#define UNIFORM_BIAS 0
#endif

// Debug hook: when DEBUG_FIBER_INDEX=1 the output store writes a diagnostic
// vector instead of the computed result, AND scatters by gl_LocalInvocationID
// (the workgroup thread index, always 0..local_size-1) rather than by the
// subgroup lane, so we can separate "workgroup isn't 64 threads" from
// "subgroup lanes don't span 0..63":
//   .x = gl_LocalInvocationID.x   (which workgroup thread wrote this slot)
//   .y = gl_WorkGroupSize.x       (compiled local size — should be 64)
//   .z = gl_SubgroupInvocationID  (subgroup lane — should equal .x if 1 subgroup)
//   .w = gl_SubgroupSize          (subgroup width — should be 64)
// Read as: if samples group*64 + 0..63 are ALL written with .x=0..63 and .y=64,
// the workgroup is a full 64 threads (bug is the subgroup-lane scatter). If only
// group*64+0 is written and .y reads 1, the workgroup is a single thread.
#ifndef DEBUG_FIBER_INDEX
#define DEBUG_FIBER_INDEX 0
#endif
#if DEBUG_FIBER_INDEX
#define _DBG_OR_STORE(store_idx, fiber_out) \
    { uint32_t _dbg = gl_WorkGroupID.x * 64u + gl_LocalInvocationID.x; \
      Y.x[_dbg] = f16vec4(float16_t(gl_LocalInvocationID.x), float16_t(gl_WorkGroupSize.x), \
                          float16_t(gl_SubgroupInvocationID), float16_t(gl_SubgroupSize)); }
#else
#define _DBG_OR_STORE(store_idx, fiber_out) \
    Y.x[store_idx] = f16vec4(fiber_out[0], fiber_out[1], fiber_out[2], fiber_out[3]);
#endif
)";

// buffers.glsl — verbatim (per-layer binding layout, element types overridable).
inline const char* kCoopBuffers = R"(
#ifndef X_ELEM
#define X_ELEM  float16_t
#endif
#ifndef W0_ELEM
#define W0_ELEM float16_t
#endif
#ifndef W1_ELEM
#define W1_ELEM float16_t
#endif
#ifndef W2_ELEM
#define W2_ELEM float16_t
#endif
#ifndef W3_ELEM
#define W3_ELEM float16_t
#endif
#ifndef W4_ELEM
#define W4_ELEM float16_t
#endif
#ifndef W5_ELEM
#define W5_ELEM float16_t
#endif
#ifndef W6_ELEM
#define W6_ELEM float16_t
#endif
#ifndef W7_ELEM
#define W7_ELEM float16_t
#endif
#ifndef B0_ELEM
#define B0_ELEM float16_t
#endif
#ifndef B1_ELEM
#define B1_ELEM float16_t
#endif
#ifndef B2_ELEM
#define B2_ELEM float16_t
#endif
#ifndef B3_ELEM
#define B3_ELEM float16_t
#endif
#ifndef B4_ELEM
#define B4_ELEM float16_t
#endif
#ifndef B5_ELEM
#define B5_ELEM float16_t
#endif
#ifndef B6_ELEM
#define B6_ELEM float16_t
#endif
#ifndef B7_ELEM
#define B7_ELEM float16_t
#endif
#ifndef Y_ELEM
#define Y_ELEM  float16_t
#endif

layout(set=0, binding=0) buffer ConstantsBuffer {
    uint batchSize;
    uint inFeatures;
    uint outFeatures;
    uint hiddenLayers;
    uint hiddenFeatures;
    uint activation;
    uint initMatrixDataType;
    uint biasType;
} constants;

layout(set=0, binding=1) buffer InputBuffer  { X_ELEM  x[]; } X;

layout(set=0, binding=2)  readonly buffer WB0 { W0_ELEM x[]; } W0;
layout(set=0, binding=3)  readonly buffer WB1 { W1_ELEM x[]; } W1;
layout(set=0, binding=4)  readonly buffer WB2 { W2_ELEM x[]; } W2;
layout(set=0, binding=5)  readonly buffer WB3 { W3_ELEM x[]; } W3;
layout(set=0, binding=6)  buffer WB4 { W4_ELEM x[]; } W4;
layout(set=0, binding=7)  readonly buffer WB5 { W5_ELEM x[]; } W5;
layout(set=0, binding=8)  readonly buffer WB6 { W6_ELEM x[]; } W6;
layout(set=0, binding=9)  readonly buffer WB7 { W7_ELEM x[]; } W7;

#ifndef BIAS_BINDING
#define BIAS_BINDING readonly buffer
#endif
#ifndef BIAS_SIZE
#define BIAS_SIZE    []
#endif

layout(set=0, binding=10) BIAS_BINDING BB0 { B0_ELEM x BIAS_SIZE; } B0;
layout(set=0, binding=11) BIAS_BINDING BB1 { B1_ELEM x BIAS_SIZE; } B1;
layout(set=0, binding=12) BIAS_BINDING BB2 { B2_ELEM x BIAS_SIZE; } B2;
layout(set=0, binding=13) BIAS_BINDING BB3 { B3_ELEM x BIAS_SIZE; } B3;
layout(set=0, binding=14) BIAS_BINDING BB4 { B4_ELEM x BIAS_SIZE; } B4;
layout(set=0, binding=15) BIAS_BINDING BB5 { B5_ELEM x BIAS_SIZE; } B5;
layout(set=0, binding=16) BIAS_BINDING BB6 { B6_ELEM x BIAS_SIZE; } B6;
layout(set=0, binding=17) BIAS_BINDING BB7 { B7_ELEM x BIAS_SIZE; } B7;

#ifdef WIDE_IO_OUT
// WIDE_IO output layout: each sample's 10 output channels are written with two
// f16vec4 stores (channels 0-7) + one f16vec2 store (channels 8-9). Under std430
// the struct { f16vec4; f16vec4; f16vec2; } occupies 20 bytes and the array
// stride rounds up to 24 bytes = 12 halfs, so the host sets paddedOut = 12 and
// reads columns 0..9 (columns 10,11 are stride padding). Only kernels that emit
// the final output define WIDE_IO_OUT; activation-producing kernels keep the
// plain Y (float16_t) below.
struct WideOutRow { f16vec4 a; f16vec4 b; f16vec2 c; };
layout(set=0, binding=18) buffer WideOutputBuffer { WideOutRow rows[]; } Yw;
#else
layout(set=0, binding=18) buffer OutputBuffer { Y_ELEM  x[]; } Y;
#endif

#ifdef WIDE_IO_OUT
// Emit one sample's 10 outputs (from a float16_t v[16] — the QCOM-extracted
// fiber vector, or the GPR fiber register file) as exactly two f16vec4 stores
// (channels 0-7) + one f16vec2 store (channels 8-9). Row stride is 12 halfs
// (std430 struct array stride); host paddedOut = 12, reads columns 0..9.
#define WIDE_STORE_10(store_idx, v)                              \
{                                                                \
    Yw.rows[store_idx].a = f16vec4(v[0], v[1], v[2], v[3]);      \
    Yw.rows[store_idx].b = f16vec4(v[4], v[5], v[6], v[7]);      \
    Yw.rows[store_idx].c = f16vec2(v[8], v[9]);                  \
}
#endif

layout(set=0, binding=19) buffer XDummyBuffer { float16_t x[]; } Xdummy;
)";

inline const char* kCoopMain = R"(
void main() {
    fusedMLP_FWDPass();
}
)";
