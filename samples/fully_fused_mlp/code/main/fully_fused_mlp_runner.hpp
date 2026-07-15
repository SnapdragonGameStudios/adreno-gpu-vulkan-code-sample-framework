//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include <vulkan/vulkan.hpp>
#include "runtime_shader.hpp"
#include <cstdint>
#include <vector>
#include <string>
#include <array>
#include <optional>

#pragma push_macro("BOOL")
#define BOOL HALF_BOOL
#include "half/half.h"
#pragma pop_macro("BOOL")

class Vulkan;

// ----------------------------------------------------------------------------
// Fully Fused MLP runner.
//
// Ports the Vulkan_MLP forward-pass + CPU reference into a framework "runner"
// (mirrors CooperativeMatrixRunner): owns the ImGui UI, host data, GPU
// dispatch and CPU/GPU validation.  The owning Application keeps presenting
// frames + FPS; this object runs a single forward pass on demand (deferred to
// TriggerPendingTests so the window stays responsive).
//
// Network is fixed: 1 input + 2 hidden + 1 output, out = 4 (RGBA).
// Width (in == hidden) selectable 16/32/64.  The output layer always has 4
// logical channels.
// ----------------------------------------------------------------------------

enum class ExecMode  { ALU = 0, COOPMAT = 1, UNFUSED = 2 };  // UNFUSED = per-layer coopmat baseline
enum class FuseMode  { GPR = 0, LOCAL = 1, GLOBAL = 2 }; // coopmat only
enum class BiasMode  { ZERO = 0, RANDOM = 1 };

// Selectable network topology.
//   RGBA    : the original fixed net — in == hidden == width{16,32,64},
//             2 hidden layers, out = 4 (RGBA). 4 weight matrices.
//   WIDE_IO : in = 12 (fixed) -> one hidden layer of width{16,64} -> out = 10.
//             2 weight matrices [12 x W] and [W x 10]. GPR (coopvec) supports
//             W=16 only (per-fiber register state is 16-wide).
enum class NetKind   { RGBA = 0, WIDE_IO = 1 };

class FusedMlpRunner
{
public:
    explicit FusedMlpRunner(Vulkan& vulkan_instance);
    ~FusedMlpRunner();

    // Detect capabilities, set initial defaults. Returns false on hard failure.
    bool InitializeRunner();

    // Called once per frame by the Application. If a run was requested via the
    // UI, build+dispatch+validate the selected configuration this frame.
    bool TriggerPendingTests();

    // Draw the ImGui configuration + results panel.
    void RenderUI();

private:
    // ---- network constants (mirror Vulkan_MLP FusedMlpConstants) ----
    struct FusedMlpConstants
    {
        uint32_t batchSize;
        uint32_t inFeatures;
        uint32_t outFeatures;
        uint32_t hiddenLayers;
        uint32_t hiddenFeatures;
        uint32_t activation;          // 0 none, 1 ReLU
        uint32_t initMatrixDataType;  // 0 deterministic, 1 constant, 2 rng
        uint32_t biasType;            // 0 zero, 1 random
    };

    static constexpr uint32_t kMaxLayers = 8; // matches buffers.glsl slot count

    // The framework's bundled half library is ramenhut/half, whose type is the
    // global class FLOAT16 (declared in half/half.h, included above). It has no
    // arithmetic operators and its operator float() is non-const, so convert via
    // the static FLOAT16::ToFloat32()/ctor helpers (see toF32() in the .cpp).
    using FLOAT16 = ::FLOAT16;

    // ---- per-sample displayed result ----
    // cpu/gpu sized for the widest supported output (WIDE_IO out = 10; the RGBA
    // net uses only [0..3]). kMaxOut leaves headroom (snapN(10) = 16).
    static constexpr uint32_t kMaxOut = 16;
    struct SampleResult
    {
        uint32_t sample;
        float    cpu[kMaxOut];
        float    gpu[kMaxOut];
        bool     pass;
    };

    // ---- host data (mirror MLP_Resources_Host) ----
    struct HostData
    {
        FusedMlpConstants            constants{};
        std::vector<FLOAT16>         input;                  // [batch x inFeatures]
        std::vector<FLOAT16>         weights[kMaxLayers];    // logical [inF x outF] row-major
        std::vector<FLOAT16>         biases[kMaxLayers];     // logical [outF]
        std::vector<FLOAT16>         output;                 // [batch x paddedOut]
        uint32_t                     paddedOut = 4;          // output row stride
    };

    // ---- a Vulkan buffer + memory pair ----
    struct Buffer
    {
        VkBuffer       buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize   size   = 0;
        bool valid() const { return buffer != VK_NULL_HANDLE; }
    };

private:
    // Setup
    void InitHostData();                 // fill deterministic weights/inputs/biases (verbatim port)
    void CpuReference(std::vector<FLOAT16>& out, uint32_t sampleIndex) const;

    // The deferred run
    bool RunForwardPass();               // returns true on success
    bool RunAlu();
    bool RunAluWideIO();                 // ALU path for the 12->W->10 net (6-binding layout)
    bool RunCoopmat();
    bool RunUnfused();                   // per-layer coopmat dispatch baseline
    void Validate();                     // fill m_sample_results + summary

    // True if the device advertises Vulkan >= 1.3 (so the runtime compiler may
    // target SPIR-V 1.6). When false we target SPIR-V 1.5 to pass spirv-val.
    bool DeviceSupportsVulkan13() const;

    // Buffer helpers
    int32_t FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const;
    bool CreateAndFillBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                             VkMemoryPropertyFlags memProps, Buffer& out, const void* data) const;
    void DestroyBuffer(Buffer& b) const;
    void CopyBufferToHost(const Buffer& b, void* dst, VkDeviceSize size) const;

    // Dispatch a compute pipeline `perf_loop` times, fill timing.
    // If refresh != nullptr, re-upload `refreshSize` bytes from `refreshData`
    // into `refresh->memory` before every dispatch (needed for the global
    // strategy, which ping-pongs activations in-place through X — without this
    // dispatches 2..N would read corrupted input).
    void DispatchCompute(VkPipeline pipeline, VkPipelineLayout layout,
                         VkDescriptorSet set, uint32_t groupCountX,
                         const Buffer* refresh = nullptr, const void* refreshData = nullptr,
                         VkDeviceSize refreshSize = 0);

    // Dispatch a chain of compute pipelines (one per network layer) once per
    // perf_loop iteration, with a global COMPUTE->COMPUTE barrier between layers
    // so each layer's global writes are visible to the next. The whole N-layer
    // chain is timed between one timestamp pair => the reported steady-state
    // avg/min is the summed per-inference latency. Used by the unfused baseline.
    struct DispatchLayer { VkPipeline pipeline; VkPipelineLayout layout; VkDescriptorSet set; };
    void DispatchComputeLayers(const std::vector<DispatchLayer>& layers, uint32_t groupCountX);

    // Build the assembled GLSL string for the selected coopmat fuse mode.
    std::string BuildCoopmatSource() const;

    // Build the assembled GLSL for the unfused single-layer kernel.
    // layerKind: 0 = RGBA hidden (KxK, ReLU-able), 1 = RGBA output (Kx4, QCOM,
    //            linear); 2 = WIDE_IO input (IN_K x W, ReLU, plain fp16 store),
    //            3 = WIDE_IO output (W x OUT_N, vectorized 10-ch store, linear).
    std::string BuildUnfusedSource(int layerKind) const;

    // Width helpers
    static uint32_t alignK(uint32_t v) { return (v + 15u) & ~15u; }
    static uint32_t snapN(uint32_t v)  { return v <= 16 ? 16 : (v <= 32 ? 32 : 64); }

private:
    Vulkan& m_vulkan_instance;

    // capabilities
    bool m_coopmat_supported   = false;
    bool m_qcom_conv_supported = false;

    // editable configuration
    NetKind  m_network   = NetKind::RGBA;   // selectable topology
    ExecMode m_exec_mode  = ExecMode::ALU;
    FuseMode m_fuse_mode  = FuseMode::LOCAL;
    int      m_width      = 16;             // 16/32/64
    bool     m_relu       = false;
    BiasMode m_bias_mode  = BiasMode::ZERO;
    int      m_batch_size = 65536;          // multiple of 64 (total number of samples)
    int      m_perf_loop  = 50;              // dispatch repeats for timing
    int      m_warmup_iterations = 25;       // leading dispatches excluded from steady-state timing
    int      m_validate_sample_count = 256;  // samples validated from head + tail of batch
    int      m_display_rows = 8;             // rows shown in the results table (head + tail)
    bool     m_debug_fiber_index = false;    // coopmat debug: store gl_SubgroupInvocationID instead of result
    float    m_validate_eps = 0.01f;

    // run state
    bool        m_run_pending = false;
    HostData    m_host{};
    std::vector<SampleResult> m_sample_results;
    std::string m_status_line = "No run yet.";
    bool        m_last_run_ok = false;
    size_t      m_mismatch_count = 0;
    size_t      m_invalid_count  = 0;
    float       m_avg_ms = -1.0f;
    float       m_min_ms = -1.0f;
    uint32_t    m_timed_iterations = 0;      // dispatches included in steady-state avg (after warm-up)
};
