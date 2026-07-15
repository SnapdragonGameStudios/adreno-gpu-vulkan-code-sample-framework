//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "fully_fused_mlp_runner.hpp"
#include "vulkan/vulkan.hpp"
#include "vulkan/extensionHelpers.hpp"
#include "vulkan/extensionLib.hpp"
#include "system/os_common.h"

#include "runtime_shaders/MlpAlu.hpp"
#include "runtime_shaders/MlpCoopCommon.hpp"
#include "runtime_shaders/MlpCoopGpr.hpp"
#include "runtime_shaders/MlpCoopLocal.hpp"
#include "runtime_shaders/MlpCoopGlobal.hpp"
#include "runtime_shaders/MlpCoopUnfused.hpp"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <sstream>
#include <iomanip>

// Decode a VkResult to a readable name for diagnostics.
static const char* VkResultStr(VkResult r)
{
    switch (r) {
        case VK_SUCCESS:                        return "VK_SUCCESS";
        case VK_NOT_READY:                      return "VK_NOT_READY";
        case VK_TIMEOUT:                        return "VK_TIMEOUT";
        case VK_INCOMPLETE:                     return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY:       return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:     return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED:    return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST:              return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED:        return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT:        return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT:    return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT:      return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER:      return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS:         return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED:     return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL:          return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_OUT_OF_POOL_MEMORY:       return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE:  return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        default:                                return "VK_ERROR_<other>";
    }
}

#define CHECK_VK(cmd)                                                                           \
    {                                                                                           \
        VkResult local_result = cmd;                                                            \
        if (local_result != VK_SUCCESS)                                                         \
            LOGE("CHECK_VK: %s returned %s (%d)", #cmd, VkResultStr(local_result),              \
                 static_cast<int>(local_result));                                               \
    }

namespace
{
    // Network shape depends on FusedMlpRunner::m_network:
    //   RGBA    : the fused-MLP shaders perform exactly 4 weight-matrix multiplies
    //             — W0 (input projection) + W1 + W2 (two hidden) + W3 (output).
    //             In Vulkan_MLP's totalWeights = hiddenLayers + 1 convention that
    //             is hiddenLayers = 3. in == hidden == width; out = 4 (RGBA).
    //   WIDE_IO : 2 weight matrices — W0 [12 x W] (input->hidden) + W1 [W x 10]
    //             (hidden->output), i.e. hiddenLayers = 1. in = 12, out = 10.
    constexpr uint32_t kHiddenLayers = 3;   // RGBA net
    constexpr uint32_t kOutFeatures  = 4;   // RGBA net

    // WIDE_IO net constants.
    constexpr uint32_t kWideInFeatures  = 12;
    constexpr uint32_t kWideOutFeatures = 10;
    constexpr uint32_t kWideHiddenLayers = 1;
    // WIDE_IO output row stride in halfs. The coopmat output is written via the
    // WideOutRow struct { f16vec4; f16vec4; f16vec2; }, whose std430 array stride
    // rounds up to 24 bytes = 12 halfs. The ALU path uses the same stride so the
    // host reads columns 0..9 uniformly (paddedOut = 12). Columns 10,11 are pad.
    constexpr uint32_t kWideOutStride = 12;

    // ramenhut/half FLOAT16 has a non-const operator float(), so it cannot be
    // static_cast on a const reference. ToFloat32() takes by value (through the
    // const copy-ctor) and works on const lvalues.
    inline float toF32(const FLOAT16& h) { return FLOAT16::ToFloat32(h); }

    // fp16-rounded primitives. The bundled ramenhut/half FLOAT16 is a
    // storage-only type (no arithmetic operators), so each op converts to
    // float, performs ONE operation, and rounds the result back to fp16. The
    // ALU CPU reference builds its dot product from these to mirror the shader's
    // native-fp16 accumulation (see CpuReference for the exact pairing order).
    inline FLOAT16 h16(float f)                    { return FLOAT16(f); }
    inline FLOAT16 hadd(const FLOAT16& a, const FLOAT16& b) { return FLOAT16(toF32(a) + toF32(b)); }
    inline FLOAT16 hmul(const FLOAT16& a, const FLOAT16& b) { return FLOAT16(toF32(a) * toF32(b)); }
    inline FLOAT16 hrelu(const FLOAT16& a)         { float f = toF32(a); return FLOAT16(f > 0.0f ? f : 0.0f); }
}


FusedMlpRunner::FusedMlpRunner(Vulkan& vulkan_instance)
    : m_vulkan_instance(vulkan_instance)
{
}

FusedMlpRunner::~FusedMlpRunner()
{
}

bool FusedMlpRunner::InitializeRunner()
{
    const bool coopmatExt = m_vulkan_instance.HasLoadedVulkanDeviceExtension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    // Real detection of the QCOM cooperative-matrix conversion extension. Every
    // coopmat strategy's output store goes through the QCOM vector<->coopmat
    // conversion ops (coopmatToVectorQCOM / vectorToCoopmatQCOM), which are only
    // legal when the extension is enabled. It is registered OPTIONAL by literal
    // name in application.cpp (the VK_QCOM_..._EXTENSION_NAME macro is absent from
    // the vendored Vulkan-Headers, so we must NOT reference it here either — use
    // the literal string to match the registration). It is loaded iff the driver
    // advertises it. Gate coopmat on BOTH extensions: without QCOM conversion,
    // coopmat mode is not presented (rather than dispatching QCOM ops the device
    // cannot legally run).
    m_qcom_conv_supported = m_vulkan_instance.HasLoadedVulkanDeviceExtension("VK_QCOM_cooperative_matrix_conversion");
    m_coopmat_supported   = coopmatExt && m_qcom_conv_supported;

    LOGI("FusedMlpRunner: cooperative_matrix ext=%s, QCOM_conversion=%s -> coopmat modes %s",
         coopmatExt ? "yes" : "no",
         m_qcom_conv_supported ? "yes" : "no",
         m_coopmat_supported ? "enabled" : "disabled");

    // Independently confirm against the raw driver extension list (the framework
    // only reports 'loaded' for extensions it registered). This disambiguates
    // "the app didn't ask for it" from "the driver doesn't expose it".
    {
        uint32_t n = 0;
        vkEnumerateDeviceExtensionProperties(m_vulkan_instance.m_VulkanGpu, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> exts(n);
        if (n) vkEnumerateDeviceExtensionProperties(m_vulkan_instance.m_VulkanGpu, nullptr, &n, exts.data());
        bool driverHasQcomConv = false, driverHasCoopMat = false;
        for (const auto& e : exts) {
            if (std::strcmp(e.extensionName, "VK_QCOM_cooperative_matrix_conversion") == 0) driverHasQcomConv = true;
            if (std::strcmp(e.extensionName, "VK_KHR_cooperative_matrix") == 0)             driverHasCoopMat = true;
        }
        LOGI("FusedMlpRunner: driver advertises VK_KHR_cooperative_matrix=%s, VK_QCOM_cooperative_matrix_conversion=%s (of %u device extensions)",
             driverHasCoopMat ? "yes" : "no", driverHasQcomConv ? "yes" : "no", n);
    }

    // --- Device capability dump (helps diagnose device-specific submit/pipeline
    //     failures such as VK_ERROR_INITIALIZATION_FAILED). ---
    {
        const auto& props = m_vulkan_instance.GetGpuProperties().Base.properties;
        LOGI("FusedMlpRunner: device '%s' apiVersion=%u.%u.%u driver=0x%08X vendor=0x%04X",
             props.deviceName,
             VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion),
             props.driverVersion, props.vendorID);
        LOGI("FusedMlpRunner: timestampPeriod=%.3f maxComputeWorkGroupInvocations=%u maxComputeSharedMemorySize=%u",
             props.limits.timestampPeriod,
             props.limits.maxComputeWorkGroupInvocations,
             props.limits.maxComputeSharedMemorySize);
        LOGI("FusedMlpRunner: maxComputeWorkGroupCount=[%u,%u,%u] maxStorageBufferRange=%u",
             props.limits.maxComputeWorkGroupCount[0],
             props.limits.maxComputeWorkGroupCount[1],
             props.limits.maxComputeWorkGroupCount[2],
             props.limits.maxStorageBufferRange);

        const uint32_t gqf = m_vulkan_instance.m_VulkanQueues[Vulkan::eGraphicsQueue].QueueFamilyIndex;
        if (gqf < m_vulkan_instance.m_pVulkanQueueProps.size())
            LOGI("FusedMlpRunner: graphics queue family %u timestampValidBits=%u queueFlags=0x%X",
                 gqf,
                 m_vulkan_instance.m_pVulkanQueueProps[gqf].timestampValidBits,
                 m_vulkan_instance.m_pVulkanQueueProps[gqf].queueFlags);

        // fp16 / 16-bit-storage features the MLP shaders rely on.
        VkPhysicalDeviceShaderFloat16Int8Features f16{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES };
        VkPhysicalDevice16BitStorageFeatures     s16{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES };
        f16.pNext = &s16;
        VkPhysicalDeviceFeatures2 feats2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        feats2.pNext = &f16;
        vkGetPhysicalDeviceFeatures2(m_vulkan_instance.m_VulkanGpu, &feats2);
        LOGI("FusedMlpRunner: shaderFloat16=%s shaderInt8=%s storageBuffer16BitAccess=%s uniformAndStorageBuffer16BitAccess=%s",
             f16.shaderFloat16 ? "yes" : "no",
             f16.shaderInt8 ? "yes" : "no",
             s16.storageBuffer16BitAccess ? "yes" : "no",
             s16.uniformAndStorageBuffer16BitAccess ? "yes" : "no");
    }

    // If coopmat is unsupported, fall back to ALU mode for the initial state.
    if (!m_coopmat_supported)
        m_exec_mode = ExecMode::ALU;

    return true;
}

bool FusedMlpRunner::DeviceSupportsVulkan13() const
{
    const uint32_t apiVersion = m_vulkan_instance.GetGpuProperties().Base.properties.apiVersion;
    return VK_VERSION_MAJOR(apiVersion) > 1 ||
           (VK_VERSION_MAJOR(apiVersion) == 1 && VK_VERSION_MINOR(apiVersion) >= 3);
}

// ----------------------------------------------------------------------------
// Host data fill — verbatim port of fused_mlp::initialize (deterministic).
// ----------------------------------------------------------------------------
void FusedMlpRunner::InitHostData()
{
    FusedMlpConstants c{};
    c.batchSize          = static_cast<uint32_t>(m_batch_size);
    if (m_network == NetKind::WIDE_IO) {
        // in = 12 (fixed) -> one hidden layer of width{16,64} -> out = 10.
        c.inFeatures     = kWideInFeatures;
        c.outFeatures    = kWideOutFeatures;
        c.hiddenLayers   = kWideHiddenLayers;
        c.hiddenFeatures = static_cast<uint32_t>(m_width);
    } else {
        // RGBA: in == hidden == width; 2 hidden layers; out = 4.
        c.inFeatures     = static_cast<uint32_t>(m_width);
        c.outFeatures    = kOutFeatures;
        c.hiddenLayers   = kHiddenLayers;
        c.hiddenFeatures = static_cast<uint32_t>(m_width);
    }
    c.activation         = m_relu ? 1u : 0u;
    c.initMatrixDataType = 0;                                  // deterministic
    c.biasType           = (m_bias_mode == BiasMode::RANDOM) ? 1u : 0u;

    m_host = HostData{};
    m_host.constants = c;

    // Input data.
    // Real PRNG (fixed seed => deterministic, so CPU and GPU read identical
    // buffers). Inputs in [0,1]; hidden layers are ReLU'd, so positive inputs
    // keep the hidden units alive. The output layer is linear (see below).
    {
        std::mt19937 rng(1234u);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        uint32_t inputCnt = c.batchSize * c.inFeatures;
        m_host.input.resize(inputCnt);
        for (uint32_t i = 0; i < inputCnt; ++i)
            m_host.input[i] = FLOAT16(dist(rng));
    }

    // weights: totalWeights = hiddenLayers + 1
    // Hidden layers (l < last): STRICTLY POSITIVE weights U(0, 2*g/inF) so the
    // per-node mean gain is g (=0.9). Combined with positive inputs this keeps
    // the ReLU'd hidden activations alive and magnitude-stable (mean ~0.3-0.45),
    // so no hidden unit dies and the signal neither explodes nor collapses.
    //
    // Output layer (l == last): the shader/CPU output layer is LINEAR (no ReLU),
    // so use ZERO-MEAN signed weights U(-b, b) with b = 2/inF. The output node
    // sum_k h_k * W_ko then has zero mean and, thanks to random-sign
    // cancellation over inF terms, stays well within [-1, 1] while producing
    // signed, per-sample-varying values on all four RGBA channels.
    // Fixed per-layer seed => deterministic (CPU and GPU read identical buffers).
    const float kLayerGain = 0.9f;
    const uint32_t totalWeights = c.hiddenLayers + 1;
    for (uint32_t l = 0; l < totalWeights; ++l) {
        bool isLast = (l == totalWeights - 1);
        uint32_t inF  = (l == 0)   ? c.inFeatures  : c.hiddenFeatures;
        uint32_t outF = isLast     ? c.outFeatures : c.hiddenFeatures;
        uint32_t layerCnt = inF * outF;
        m_host.weights[l].resize(layerCnt);
        std::mt19937 rng(9001u + l);
        if (isLast) {
            const float b = 2.0f / static_cast<float>(inF);     // signed, zero-mean
            std::uniform_real_distribution<float> dist(-b, b);
            for (uint32_t i = 0; i < layerCnt; ++i)
                m_host.weights[l][i] = FLOAT16(dist(rng));
        } else {
            const float wmax = 2.0f * kLayerGain / static_cast<float>(inF); // mean gain = kLayerGain
            std::uniform_real_distribution<float> dist(0.0f, wmax);
            for (uint32_t i = 0; i < layerCnt; ++i)
                m_host.weights[l][i] = FLOAT16(dist(rng));
        }
    }

    // biases: same count as weights
    for (uint32_t l = 0; l < totalWeights; ++l) {
        uint32_t biasCnt = (l == totalWeights - 1) ? c.outFeatures : c.hiddenFeatures;
        m_host.biases[l].resize(biasCnt);
        for (uint32_t i = 0; i < biasCnt; ++i) {
            float val = (c.biasType == 1)
                ? -0.05f + 0.1f * ((float)((l * 12349 + i * 7919) % 10000) / 9999.0f)
                : 0.0f;
            m_host.biases[l][i] = FLOAT16(val);
        }
    }

    // output: paddedOut per row. WIDE_IO uses the vectorized WideOutRow stride
    // (12 halfs); RGBA uses out_features (=4).
    m_host.paddedOut = (m_network == NetKind::WIDE_IO) ? kWideOutStride : c.outFeatures;
    m_host.output.assign(static_cast<size_t>(c.batchSize) * m_host.paddedOut, FLOAT16(0));
}

// ----------------------------------------------------------------------------
// CPU reference. Accumulation precision matches the GPU path under test:
//   ALU      -> fp16 accumulation (each mul/add/ReLU rounds to fp16), matching
//               the native-fp16 ALU shader step-for-step.
//   coopmat  -> fp32 accumulation, single fp16 round per layer, matching the
//               coopmat hardware accumulator.
// ReLU on hidden layers only; one fp16 value flows between layers either way.
// ----------------------------------------------------------------------------
void FusedMlpRunner::CpuReference(std::vector<FLOAT16>& output, uint32_t sampleIndex) const
{
    const auto& c = m_host.constants;
    uint32_t in_features    = c.inFeatures;
    uint32_t hidden_layers  = c.hiddenLayers;
    uint32_t hidden_features = c.hiddenFeatures;
    uint32_t out_features   = c.outFeatures;

    const uint32_t inOff = sampleIndex * in_features;
    std::vector<FLOAT16> current(m_host.input.begin() + inOff,
                                 m_host.input.begin() + inOff + in_features);

    // Accumulation precision must match the GPU path being validated:
    //  - ALU shader accumulates natively in fp16 (each mul/add/relu rounds).
    //  - coopmat accumulators are fp32 internally, rounding to fp16 only on
    //    store between layers.
    // Mirror that here so the reference is bit-consistent with whichever path
    // produced the GPU output.
    const bool fp16_accum = (m_exec_mode == ExecMode::ALU);

    auto dense = [&](const std::vector<FLOAT16>& src, std::vector<FLOAT16>& dst,
                     uint32_t in_f, uint32_t out_f, uint32_t layer, bool applyActivation)
    {
        for (uint32_t o = 0; o < out_f; ++o) {
            if (fp16_accum) {
                // Match the ALU shader's fp16 accumulation exactly. Upstream
                // Vulkan_MLP sums two fp16-rounded products per step:
                //   sum += half(a[2i]*w[2i]) + half(a[2i+1]*w[2i+1])
                // ("decrease the fp16 precision to match the shader"). src[i]
                // and the weight are fp16; each product rounds to fp16, the pair
                // is added, then accumulated into the fp16 running sum.
                // Weight index i*out_f + o is our [in][out] storage == upstream's
                // [out][in] weightsData_h[layer][o*in_f + i].
                FLOAT16 sum = h16(0.0f);
                uint32_t pairs = in_f / 2u;
                for (uint32_t p = 0; p < pairs; ++p) {
                    uint32_t i0 = p * 2u;
                    uint32_t i1 = i0 + 1u;
                    FLOAT16 prod0 = hmul(src[i0], m_host.weights[layer][i0 * out_f + o]);
                    FLOAT16 prod1 = hmul(src[i1], m_host.weights[layer][i1 * out_f + o]);
                    sum = hadd(sum, hadd(prod0, prod1));
                }
                if (in_f & 1u) { // odd tail (widths here are even, kept for safety)
                    uint32_t i0 = in_f - 1u;
                    sum = hadd(sum, hmul(src[i0], m_host.weights[layer][i0 * out_f + o]));
                }
                if (m_bias_mode == BiasMode::RANDOM) {
                    // RGBA: match the ALU shader's LOAD_BIAS, which reads
                    // biasesBuf.data[biasOffset][i%4] — only the first 4 bias
                    // values, broadcast cyclically across all nodes (o % 4).
                    // WIDE_IO: the new-net ALU shader reads full per-node bias[o].
                    uint32_t bi = (m_network == NetKind::WIDE_IO) ? o : (o % 4u);
                    sum = hadd(sum, m_host.biases[layer][bi]);
                }
                if (applyActivation && c.activation == 1)
                    sum = hrelu(sum);                           // fp16 ReLU
                dst[o] = sum;
            } else {
                float sum = 0.0f;                              // fp32 accumulator (coopmat)
                for (uint32_t i = 0; i < in_f; ++i) {
                    uint32_t wi = i * out_f + o;
                    sum += toF32(src[i]) * toF32(m_host.weights[layer][wi]);
                }
                if (m_bias_mode == BiasMode::RANDOM)
                    sum += toF32(m_host.biases[layer][o]);
                if (applyActivation && c.activation == 1 && sum < 0.0f) sum = 0.0f;
                dst[o] = FLOAT16(sum);                         // single fp16 round per layer
            }
        }
    };

    const uint32_t totalWeights = hidden_layers + 1;
    // The output layer is LINEAR on every path: the ALU shader's FINAL_LAYER now
    // uses LOAD_BIAS_ONLY (bias, no ReLU) and the coopmat shaders skip activation
    // on the last layer. ReLU is applied to hidden layers only, so the network
    // can emit signed outputs.
    const bool activateOutput = false;
    for (uint32_t l = 0; l < totalWeights; ++l) {
        bool     isLast = (l == totalWeights - 1);
        uint32_t in_f   = (l == 0)   ? in_features   : hidden_features;
        uint32_t out_f  = isLast     ? out_features  : hidden_features;
        bool     doAct  = isLast ? activateOutput : true;
        std::vector<FLOAT16> next(out_f, FLOAT16(0));
        dense(current, next, in_f, out_f, l, doAct);
        current = std::move(next);
    }
    output = std::move(current);
}

// ----------------------------------------------------------------------------
// Buffer helpers
// ----------------------------------------------------------------------------
int32_t FusedMlpRunner::FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(m_vulkan_instance.m_VulkanGpu, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return static_cast<int32_t>(i);
    }
    return -1;
}

bool FusedMlpRunner::CreateAndFillBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                         VkMemoryPropertyFlags memProps, Buffer& out, const void* data) const
{
    if (size == 0) size = 16; // keep VkCreateBuffer happy for empty/unused slots

    VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK_VK(vkCreateBuffer(m_vulkan_instance.m_VulkanDevice, &bi, nullptr, &out.buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(m_vulkan_instance.m_VulkanDevice, out.buffer, &req);

    int32_t memIdx = FindMemoryType(req.memoryTypeBits, memProps);
    if (memIdx < 0) { LOGE("FusedMlpRunner: no suitable memory type"); return false; }

    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = static_cast<uint32_t>(memIdx);
    CHECK_VK(vkAllocateMemory(m_vulkan_instance.m_VulkanDevice, &ai, nullptr, &out.memory));
    CHECK_VK(vkBindBufferMemory(m_vulkan_instance.m_VulkanDevice, out.buffer, out.memory, 0));
    out.size = size;

    if (data) {
        void* mapped = nullptr;
        CHECK_VK(vkMapMemory(m_vulkan_instance.m_VulkanDevice, out.memory, 0, size, 0, &mapped));
        std::memcpy(mapped, data, static_cast<size_t>(size));
        vkUnmapMemory(m_vulkan_instance.m_VulkanDevice, out.memory);
    }
    return true;
}

void FusedMlpRunner::DestroyBuffer(Buffer& b) const
{
    if (b.buffer) vkDestroyBuffer(m_vulkan_instance.m_VulkanDevice, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(m_vulkan_instance.m_VulkanDevice, b.memory, nullptr);
    b = Buffer{};
}

void FusedMlpRunner::CopyBufferToHost(const Buffer& b, void* dst, VkDeviceSize size) const
{
    void* mapped = nullptr;
    CHECK_VK(vkMapMemory(m_vulkan_instance.m_VulkanDevice, b.memory, 0, size, 0, &mapped));
    std::memcpy(dst, mapped, static_cast<size_t>(size));
    vkUnmapMemory(m_vulkan_instance.m_VulkanDevice, b.memory);
}

void FusedMlpRunner::DispatchCompute(VkPipeline pipeline, VkPipelineLayout layout,
                                     VkDescriptorSet set, uint32_t groupCountX,
                                     const Buffer* refresh, const void* refreshData,
                                     VkDeviceSize refreshSize)
{
    VkDevice       device = m_vulkan_instance.m_VulkanDevice;
    const uint32_t qfi   = m_vulkan_instance.m_VulkanQueues[Vulkan::eGraphicsQueue].QueueFamilyIndex;
    VkQueue        queue = m_vulkan_instance.m_VulkanQueues[Vulkan::eGraphicsQueue].Queue;

    // Guard against dispatching with objects that failed to create earlier. A
    // null pipeline is the classic cause of VK_ERROR_INITIALIZATION_FAILED at
    // vkQueueSubmit (e.g. vkCreateComputePipelines rejected the runtime SPIR-V
    // on this device). CHECK_VK only logs, so verify here before recording.
    if (pipeline == VK_NULL_HANDLE || layout == VK_NULL_HANDLE || set == VK_NULL_HANDLE) {
        LOGE("FusedMlpRunner::DispatchCompute: aborting - invalid handle(s): pipeline=%p layout=%p set=%p "
             "(pipeline/shader creation likely failed earlier for this config on this device)",
             (void*)pipeline, (void*)layout, (void*)set);
        return;
    }
    if (queue == VK_NULL_HANDLE) {
        LOGE("FusedMlpRunner::DispatchCompute: graphics queue is VK_NULL_HANDLE (family=%d)", (int)qfi);
        return;
    }
    LOGI("FusedMlpRunner::DispatchCompute: groupCountX=%u perf_loop=%d queueFamily=%d",
         groupCountX, m_perf_loop, (int)qfi);

    m_vulkan_instance.QueueWaitIdle(Vulkan::eGraphicsQueue);

    VkCommandPoolCreateInfo cpci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfi };
    VkCommandPool pool = VK_NULL_HANDLE;
    CHECK_VK(vkCreateCommandPool(device, &cpci, nullptr, &pool));

    // Timestamp queries are only legal when the target queue family reports
    // timestampValidBits > 0 (and the device a nonzero timestampPeriod). Some
    // devices/queues don't support them; issuing vkCmdWriteTimestamp there makes
    // the driver reject the submit (seen as VK_ERROR_INITIALIZATION_FAILED from
    // vkQueueSubmit). Detect support and gate all timestamp usage — when
    // unsupported we simply report timing as unavailable; the compute is
    // unaffected.
    const double tsPeriod = m_vulkan_instance.GetGpuProperties().Base.properties.limits.timestampPeriod;
    const bool useTimestamps = (tsPeriod > 0.0)
        && (qfi >= 0)
        && (static_cast<size_t>(qfi) < m_vulkan_instance.m_pVulkanQueueProps.size())
        && (m_vulkan_instance.m_pVulkanQueueProps[qfi].timestampValidBits > 0);

    uint32_t perf_loop = std::max(1, m_perf_loop);
    uint32_t queryCount = perf_loop * 2;
    VkQueryPool qpool = VK_NULL_HANDLE;
    if (useTimestamps) {
        VkQueryPoolCreateInfo qpci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0,
            VK_QUERY_TYPE_TIMESTAMP, queryCount, 0 };
        CHECK_VK(vkCreateQueryPool(device, &qpci, nullptr, &qpool));
    }

    const bool   doRefresh = (refresh && refresh->valid() && refreshData && refreshSize > 0);

    auto recordOne = [&](VkCommandBuffer cb, uint32_t i, bool resetWhole)
    {
        VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cb, &bi);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        if (useTimestamps) {
            if (resetWhole) vkCmdResetQueryPool(cb, qpool, 0, queryCount);
            else            vkCmdResetQueryPool(cb, qpool, i * 2, 2);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, i * 2);
        }
        vkCmdDispatch(cb, groupCountX, 1, 1);
        if (useTimestamps)
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, i * 2 + 1);
        vkEndCommandBuffer(cb);
    };

    if (doRefresh) {
        // Per-dispatch command buffer; re-upload X before each so every
        // dispatch sees the original input (global strategy ping-pongs X).
        for (uint32_t i = 0; i < perf_loop; ++i) {
            void* mapped = nullptr;
            CHECK_VK(vkMapMemory(device, refresh->memory, 0, refreshSize, 0, &mapped));
            std::memcpy(mapped, refreshData, static_cast<size_t>(refreshSize));
            vkUnmapMemory(device, refresh->memory);

            VkCommandBufferAllocateInfo cbai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
                pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
            VkCommandBuffer cb = VK_NULL_HANDLE;
            vkAllocateCommandBuffers(device, &cbai, &cb);
            recordOne(cb, i, /*resetWhole=*/false);

            VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            si.commandBufferCount = 1; si.pCommandBuffers = &cb;
            VkResult subRes = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
            if (subRes != VK_SUCCESS) {
                LOGE("FusedMlpRunner: vkQueueSubmit (refresh iter %u) failed: %s (%d)",
                     i, VkResultStr(subRes), (int)subRes);
                vkFreeCommandBuffers(device, pool, 1, &cb);
                break;
            }
            CHECK_VK(vkQueueWaitIdle(queue));
            vkFreeCommandBuffers(device, pool, 1, &cb);
        }
    } else {
        // Submit each iteration as its OWN small submit (reusing one command
        // buffer) rather than packing all perf_loop dispatches into a single
        // command buffer + single submit. A single submit that runs longer than
        // the OS GPU watchdog (TDR, ~2 s on Windows) triggers a device reset —
        // which is what "crashes at high perf-loop / width 64" was: the giant
        // batched submit exceeded the watchdog. Per-iteration submits keep each
        // submit short; the timestamps still measure per-dispatch GPU time.
        VkCommandBufferAllocateInfo cbai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
            pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
        VkCommandBuffer cb = VK_NULL_HANDLE;
        CHECK_VK(vkAllocateCommandBuffers(device, &cbai, &cb));

        // Reset the whole timestamp pool up front (own tiny submit / not timed).
        if (useTimestamps) {
            VkCommandBufferBeginInfo rbi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            vkBeginCommandBuffer(cb, &rbi);
            vkCmdResetQueryPool(cb, qpool, 0, queryCount);
            vkEndCommandBuffer(cb);
            VkSubmitInfo rsi{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            rsi.commandBufferCount = 1; rsi.pCommandBuffers = &cb;
            CHECK_VK(vkQueueSubmit(queue, 1, &rsi, VK_NULL_HANDLE));
            CHECK_VK(vkQueueWaitIdle(queue));
        }

        for (uint32_t i = 0; i < perf_loop; ++i) {
            CHECK_VK(vkResetCommandBuffer(cb, 0));
            VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            vkBeginCommandBuffer(cb, &bi);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
            if (useTimestamps)
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, i * 2);
            vkCmdDispatch(cb, groupCountX, 1, 1);
            if (useTimestamps)
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, i * 2 + 1);
            vkEndCommandBuffer(cb);

            VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            si.commandBufferCount = 1; si.pCommandBuffers = &cb;
            VkResult subRes = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
            if (subRes != VK_SUCCESS) {
                LOGE("FusedMlpRunner: vkQueueSubmit (iter %u) failed: %s (%d) [groupCountX=%u]",
                     i, VkResultStr(subRes), (int)subRes, groupCountX);
                break;
            }
            CHECK_VK(vkQueueWaitIdle(queue));
        }
        vkFreeCommandBuffers(device, pool, 1, &cb);
    }

    if (useTimestamps) {
        std::vector<uint64_t> ts(queryCount, 0);
        vkGetQueryPoolResults(device, qpool, 0, queryCount,
            ts.size() * sizeof(uint64_t), ts.data(), sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

        // Report STEADY-STATE timing only: the first several dispatches run while
        // the GPU clock ramps up (DVFS warm-up) and are much slower, so exclude
        // the leading m_warmup_iterations from the average. Clamp so at least one
        // iteration is always counted (fall back to all iterations if the warm-up
        // count would leave none).
        uint32_t warmup = static_cast<uint32_t>(std::max(0, m_warmup_iterations));
        uint32_t firstCounted = (warmup < perf_loop) ? warmup : 0u;

        double total = 0.0, mn = 1e300;
        uint32_t counted = 0;
        for (uint32_t i = firstCounted; i < perf_loop; ++i) {
            if (ts[i*2+1] < ts[i*2]) continue;
            double us = static_cast<double>(ts[i*2+1] - ts[i*2]) * tsPeriod / 1000.0;
            total += us;
            mn = std::min(mn, us);
            ++counted;
        }
        m_avg_ms = counted ? static_cast<float>((total / counted) / 1000.0) : -1.0f;
        m_min_ms = counted ? static_cast<float>(mn / 1000.0) : -1.0f;
        m_timed_iterations = counted;
        vkDestroyQueryPool(device, qpool, nullptr);
    } else {
        // Timing not available on this device/queue.
        m_avg_ms = -1.0f;
        m_min_ms = -1.0f;
        m_timed_iterations = 0;
    }

    vkDestroyCommandPool(device, pool, nullptr);
}

// ----------------------------------------------------------------------------
// DispatchComputeLayers — run a chain of per-layer pipelines once per perf-loop
// iteration, timed as ONE inference. Each iteration: for each layer, bind its
// pipeline+set and dispatch, with a global COMPUTE->COMPUTE barrier between
// layers so layer L's global writes are visible to layer L+1. The whole chain
// is bracketed by one timestamp pair, so the existing steady-state avg/min
// (warm-up excluded) reports the summed per-inference latency.
// ----------------------------------------------------------------------------
void FusedMlpRunner::DispatchComputeLayers(const std::vector<DispatchLayer>& layers, uint32_t groupCountX)
{
    VkDevice       device = m_vulkan_instance.m_VulkanDevice;
    const uint32_t qfi   = m_vulkan_instance.m_VulkanQueues[Vulkan::eGraphicsQueue].QueueFamilyIndex;
    VkQueue        queue = m_vulkan_instance.m_VulkanQueues[Vulkan::eGraphicsQueue].Queue;

    if (layers.empty() || queue == VK_NULL_HANDLE) {
        LOGE("FusedMlpRunner::DispatchComputeLayers: no layers or null queue"); return;
    }
    for (const auto& L : layers)
        if (L.pipeline == VK_NULL_HANDLE || L.set == VK_NULL_HANDLE) {
            LOGE("FusedMlpRunner::DispatchComputeLayers: invalid layer handle"); return;
        }

    m_vulkan_instance.QueueWaitIdle(Vulkan::eGraphicsQueue);

    VkCommandPoolCreateInfo cpci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfi };
    VkCommandPool pool = VK_NULL_HANDLE;
    CHECK_VK(vkCreateCommandPool(device, &cpci, nullptr, &pool));

    const double tsPeriod = m_vulkan_instance.GetGpuProperties().Base.properties.limits.timestampPeriod;
    const bool useTimestamps = (tsPeriod > 0.0)
        && (qfi >= 0)
        && (static_cast<size_t>(qfi) < m_vulkan_instance.m_pVulkanQueueProps.size())
        && (m_vulkan_instance.m_pVulkanQueueProps[qfi].timestampValidBits > 0);

    uint32_t perf_loop = std::max(1, m_perf_loop);
    uint32_t queryCount = perf_loop * 2;
    VkQueryPool qpool = VK_NULL_HANDLE;
    if (useTimestamps) {
        VkQueryPoolCreateInfo qpci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0,
            VK_QUERY_TYPE_TIMESTAMP, queryCount, 0 };
        CHECK_VK(vkCreateQueryPool(device, &qpci, nullptr, &qpool));
    }

    VkCommandBufferAllocateInfo cbai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
        pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
    VkCommandBuffer cb = VK_NULL_HANDLE;
    CHECK_VK(vkAllocateCommandBuffers(device, &cbai, &cb));

    VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    // Reset the timestamp pool up front in its own tiny submit.
    if (useTimestamps) {
        VkCommandBufferBeginInfo rbi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cb, &rbi);
        vkCmdResetQueryPool(cb, qpool, 0, queryCount);
        vkEndCommandBuffer(cb);
        VkSubmitInfo rsi{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        rsi.commandBufferCount = 1; rsi.pCommandBuffers = &cb;
        CHECK_VK(vkQueueSubmit(queue, 1, &rsi, VK_NULL_HANDLE));
        CHECK_VK(vkQueueWaitIdle(queue));
    }

    // Submit ONE inference (the whole N-layer chain) per submit, rather than
    // packing all perf_loop inferences into a single command buffer + submit.
    // The unfused baseline runs N dispatches per inference over a large batch;
    // batching all iterations into one submit can exceed the OS GPU watchdog
    // (TDR, ~2 s) and reset the device ("crash at high perf-loop / width 64").
    // One inference per submit keeps each submit short; the timestamp pair still
    // brackets the full N-layer chain, so avg/min = summed per-inference latency.
    for (uint32_t i = 0; i < perf_loop; ++i) {
        CHECK_VK(vkResetCommandBuffer(cb, 0));
        VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cb, &bi);
        if (useTimestamps)
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, i * 2);
        for (size_t l = 0; l < layers.size(); ++l) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layers[l].pipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layers[l].layout, 0, 1, &layers[l].set, 0, nullptr);
            vkCmdDispatch(cb, groupCountX, 1, 1);
            // barrier between layers so this layer's global writes feed the next
            if (l + 1 < layers.size())
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, 1, &barrier, 0, nullptr, 0, nullptr);
        }
        if (useTimestamps)
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, i * 2 + 1);
        vkEndCommandBuffer(cb);

        VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VkResult subRes = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        if (subRes != VK_SUCCESS) {
            LOGE("FusedMlpRunner: vkQueueSubmit (layers iter %u) failed: %s (%d)", i, VkResultStr(subRes), (int)subRes);
            break;
        }
        CHECK_VK(vkQueueWaitIdle(queue));
    }
    vkFreeCommandBuffers(device, pool, 1, &cb);

    if (useTimestamps) {
        std::vector<uint64_t> ts(queryCount, 0);
        vkGetQueryPoolResults(device, qpool, 0, queryCount,
            ts.size() * sizeof(uint64_t), ts.data(), sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

        uint32_t warmup = static_cast<uint32_t>(std::max(0, m_warmup_iterations));
        uint32_t firstCounted = (warmup < perf_loop) ? warmup : 0u;
        double total = 0.0, mn = 1e300;
        uint32_t counted = 0;
        for (uint32_t i = firstCounted; i < perf_loop; ++i) {
            if (ts[i*2+1] < ts[i*2]) continue;
            double us = static_cast<double>(ts[i*2+1] - ts[i*2]) * tsPeriod / 1000.0;
            total += us; mn = std::min(mn, us); ++counted;
        }
        m_avg_ms = counted ? static_cast<float>((total / counted) / 1000.0) : -1.0f;
        m_min_ms = counted ? static_cast<float>(mn / 1000.0) : -1.0f;
        m_timed_iterations = counted;
        vkDestroyQueryPool(device, qpool, nullptr);
    } else {
        m_avg_ms = -1.0f; m_min_ms = -1.0f; m_timed_iterations = 0;
    }

    vkDestroyCommandPool(device, pool, nullptr);
}

// ----------------------------------------------------------------------------
// ALU path — 5-binding packed layout, single concatenated weights/biases.
// ----------------------------------------------------------------------------
bool FusedMlpRunner::RunAlu()
{
    if (m_network == NetKind::WIDE_IO)
        return RunAluWideIO();

    const auto& c = m_host.constants;
    VkDevice device = m_vulkan_instance.m_VulkanDevice;

    const VkBufferUsageFlags    ubo = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    const VkBufferUsageFlags    ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkMemoryPropertyFlags hostMem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    const uint32_t totalWeights = c.hiddenLayers + 1;
    const uint32_t W = c.hiddenFeatures; // NETWORK_MAX_WIDTH (in == hidden)

    // The tile-interleaved ALU shader (faithful port of the upstream kernel)
    // streams each layer's weights through a 256-element LDS bank in fixed
    // NETWORK_MAX_WIDTH x NETWORK_MAX_WIDTH blocks, indexing them as
    //   weight = localWeights[(node) * ACCUMULATION + k]   (ACCUMULATION = WIDTH)
    // i.e. [out][in] (transposed) row-major with stride WIDTH. The host stores
    // weights as [in][out], so transpose each layer into a WIDTH x WIDTH block
    // (the output layer's WIDTH x 4 is zero-padded up to WIDTH x WIDTH; only
    // output rows 0..3 are real and read back). Biases concatenate as
    // [width, width, width, out] to match the per-layer LOAD_BIAS_* offsets.
    std::vector<FLOAT16> weightsConcat, biasesConcat;
    for (uint32_t l = 0; l < totalWeights; ++l) {
        bool     isFirst = (l == 0);
        bool     isLast  = (l == totalWeights - 1);
        uint32_t inF  = isFirst ? c.inFeatures  : c.hiddenFeatures;
        uint32_t outF = isLast  ? c.outFeatures : c.hiddenFeatures;
        std::vector<FLOAT16> block(static_cast<size_t>(W) * W, FLOAT16(0));
        for (uint32_t o = 0; o < outF; ++o)
            for (uint32_t k = 0; k < inF; ++k)
                block[o * W + k] = m_host.weights[l][k * outF + o]; // [in][out] -> [out][in]
        weightsConcat.insert(weightsConcat.end(), block.begin(), block.end());
    }
    for (uint32_t l = 0; l < totalWeights; ++l)
        biasesConcat.insert(biasesConcat.end(), m_host.biases[l].begin(), m_host.biases[l].end());

    Buffer constantsBuf, inputBuf, weightsBuf, biasesBuf, outputBuf;
    bool bufOk = true;
    bufOk &= CreateAndFillBuffer(sizeof(FusedMlpConstants), ubo, hostMem, constantsBuf, &c);
    bufOk &= CreateAndFillBuffer(m_host.input.size() * sizeof(FLOAT16), ssbo, hostMem, inputBuf, m_host.input.data());
    bufOk &= CreateAndFillBuffer(weightsConcat.size() * sizeof(FLOAT16), ssbo, hostMem, weightsBuf, weightsConcat.data());
    bufOk &= CreateAndFillBuffer(biasesConcat.size()  * sizeof(FLOAT16), ssbo, hostMem, biasesBuf,  biasesConcat.data());
    bufOk &= CreateAndFillBuffer(m_host.output.size()  * sizeof(FLOAT16), ssbo, hostMem, outputBuf, nullptr);
    if (!bufOk) {
        LOGE("ALU buffer allocation failed (batch=%u width=%d) - out of device memory or no host-visible type?",
             c.batchSize, m_width);
        DestroyBuffer(constantsBuf); DestroyBuffer(inputBuf); DestroyBuffer(weightsBuf);
        DestroyBuffer(biasesBuf); DestroyBuffer(outputBuf);
        return false;
    }
    LOGI("ALU buffers OK: input=%zu weights=%zu biases=%zu output=%zu (halfs)",
         m_host.input.size(), weightsConcat.size(), biasesConcat.size(), m_host.output.size());

    // compile shader
    RuntimeShader shader;
    shader.AddDefine("NETWORK_MAX_WIDTH", std::to_string(m_width));
    // The tile-interleaved weight streaming is hardwired to 64 fibers per
    // workgroup (each fiber loads 4 halfs, 64 * 4 = 256 = one LDS bank, and
    // globalWeightIndex advances by 64). FORWARD_GROUP_SIZE must therefore be
    // 64; large batches are handled by the in-shader grid-stride loop, not by a
    // bigger workgroup.
    shader.AddDefine("FORWARD_GROUP_SIZE", std::string("64"));
    // Target SPIR-V 1.5 / Vulkan 1.2 (the framework's default instance version).
    // The shaders use a literal workgroup size (no LocalSizeId) and coopmat via
    // the SPV_KHR_cooperative_matrix extension, so 1.5 is sufficient and avoids
    // needing to force the instance/device to 1.3.
    if (!shader.Build(kMlpAluShader, device, "main", GLSLANG_STAGE_COMPUTE, /*target_vulkan_1_3=*/false)) {
        LOGE("ALU shader failed to compile");
        DestroyBuffer(constantsBuf); DestroyBuffer(inputBuf); DestroyBuffer(weightsBuf);
        DestroyBuffer(biasesBuf); DestroyBuffer(outputBuf);
        return false;
    }

    // descriptor set layout: 0 UBO, 1..4 SSBO
    VkDescriptorSetLayoutBinding lb[5] = {
        { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    };
    VkDescriptorSetLayoutCreateInfo li{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    li.bindingCount = 5; li.pBindings = lb;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDescriptorSetLayout(device, &li, nullptr, &dsl));

    VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    CHECK_VK(vkCreatePipelineLayout(device, &pli, nullptr, &pl));

    // spec constants 0..5: in/hidden/out/batch/activation/load_bias
    const uint32_t loadBias = (m_bias_mode == BiasMode::RANDOM) ? 1u : 0u;
    uint32_t spec[6] = { c.inFeatures, c.hiddenFeatures, c.outFeatures, c.batchSize, c.activation, loadBias };
    VkSpecializationMapEntry me[6];
    for (uint32_t i = 0; i < 6; ++i) me[i] = { i, i * (uint32_t)sizeof(uint32_t), sizeof(uint32_t) };
    VkSpecializationInfo specInfo{ 6, me, sizeof(spec), spec };

    VkPipelineShaderStageCreateInfo ss{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ss.module = shader.GetShaderModule();
    ss.pName = "main";
    ss.pSpecializationInfo = &specInfo;
    VkComputePipelineCreateInfo pci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pci.stage = ss; pci.layout = pl;
    VkPipeline pipeline = VK_NULL_HANDLE;
    {
        VkResult pRes = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline);
        if (pRes != VK_SUCCESS || pipeline == VK_NULL_HANDLE) {
            LOGE("ALU vkCreateComputePipelines failed: %s (%d) [width=%d]. Aborting run.",
                 VkResultStr(pRes), (int)pRes, m_width);
            vkDestroyPipelineLayout(device, pl, nullptr);
            vkDestroyDescriptorSetLayout(device, dsl, nullptr);
            vkDestroyShaderModule(device, shader.GetShaderModule(), nullptr);
            DestroyBuffer(constantsBuf); DestroyBuffer(inputBuf); DestroyBuffer(weightsBuf);
            DestroyBuffer(biasesBuf); DestroyBuffer(outputBuf);
            return false;
        }
    }

    // pool + set
    VkDescriptorPoolSize ps[2] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 },
    };
    VkDescriptorPoolCreateInfo pi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pi.poolSizeCount = 2; pi.pPoolSizes = ps; pi.maxSets = 1;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDescriptorPool(device, &pi, nullptr, &dpool));

    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = dpool; ai.descriptorSetCount = 1; ai.pSetLayouts = &dsl;
    VkDescriptorSet set = VK_NULL_HANDLE;
    CHECK_VK(vkAllocateDescriptorSets(device, &ai, &set));

    VkDescriptorBufferInfo bufInfos[5] = {
        { constantsBuf.buffer, 0, constantsBuf.size },
        { inputBuf.buffer,     0, inputBuf.size },
        { weightsBuf.buffer,   0, weightsBuf.size },
        { biasesBuf.buffer,    0, biasesBuf.size },
        { outputBuf.buffer,    0, outputBuf.size },
    };
    VkDescriptorType types[5] = {
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    VkWriteDescriptorSet writes[5];
    for (uint32_t i = 0; i < 5; ++i)
        writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, i, 0, 1, types[i], nullptr, &bufInfos[i], nullptr };
    vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);

    // Grid-stride dispatch. The kernel uses a fixed workgroup of 64 fibers
    // (required by the LDS weight streaming); enough workgroups to cover the
    // batch once, clamped to the device's max workgroup count so any batch size
    // is legal — the shader's grid-stride loop handles the remainder.
    const uint32_t kAluWorkgroup = 64u;
    const uint32_t maxGroups = m_vulkan_instance.GetGpuProperties().Base.properties.limits.maxComputeWorkGroupCount[0];
    uint32_t groupCountX = (c.batchSize + kAluWorkgroup - 1u) / kAluWorkgroup;
    groupCountX = std::clamp(groupCountX, 1u, maxGroups);
    DispatchCompute(pipeline, pl, set, groupCountX);

    // readback
    CopyBufferToHost(outputBuf, m_host.output.data(), outputBuf.size);

    // cleanup
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pl, nullptr);
    vkDestroyDescriptorPool(device, dpool, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    vkDestroyShaderModule(device, shader.GetShaderModule(), nullptr);
    DestroyBuffer(constantsBuf); DestroyBuffer(inputBuf); DestroyBuffer(weightsBuf);
    DestroyBuffer(biasesBuf); DestroyBuffer(outputBuf);
    return true;
}

// ----------------------------------------------------------------------------
// ALU path for the WIDE_IO net (12 -> W -> 10). 6-binding layout, dedicated
// kernel (kMlpAluWideIOShader). Weights uploaded in [in][out] row-major (the
// shader's native layout — same as host storage, no transpose). Input padded
// 12 -> IN_PAD (16). Output [batch x OUT_PAD (=12)], columns 0..9 real.
// ----------------------------------------------------------------------------
bool FusedMlpRunner::RunAluWideIO()
{
    const auto& c = m_host.constants;
    VkDevice device = m_vulkan_instance.m_VulkanDevice;

    const VkBufferUsageFlags    ubo = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    const VkBufferUsageFlags    ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkMemoryPropertyFlags hostMem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    const uint32_t inF     = c.inFeatures;      // 12
    const uint32_t W       = c.hiddenFeatures;  // 16 or 64
    const uint32_t outF    = c.outFeatures;     // 10
    const uint32_t inPad   = alignK(inF);       // 16 (input row stride, f16vec4-friendly)
    const uint32_t outPad  = kWideOutStride;    // 12 (matches coopmat WideOutRow stride)

    // input padded 12 -> inPad
    std::vector<FLOAT16> inputPadded(static_cast<size_t>(c.batchSize) * inPad, FLOAT16(0));
    for (uint32_t r = 0; r < c.batchSize; ++r)
        for (uint32_t i = 0; i < inF; ++i)
            inputPadded[r * inPad + i] = m_host.input[r * inF + i];

    // weights in [in][out] row-major (== host storage). L0 [12 x W], L1 [W x 10].
    const std::vector<FLOAT16>& w0 = m_host.weights[0];   // size inF*W
    const std::vector<FLOAT16>& w1 = m_host.weights[1];   // size W*outF

    // biases concatenated: b0[W] ++ b1[outF]
    std::vector<FLOAT16> biasesConcat;
    biasesConcat.insert(biasesConcat.end(), m_host.biases[0].begin(), m_host.biases[0].end());
    biasesConcat.insert(biasesConcat.end(), m_host.biases[1].begin(), m_host.biases[1].end());

    Buffer constantsBuf, inputBuf, w0Buf, w1Buf, biasesBuf, outputBuf;
    bool bufOk = true;
    bufOk &= CreateAndFillBuffer(sizeof(FusedMlpConstants), ubo, hostMem, constantsBuf, &c);
    bufOk &= CreateAndFillBuffer(inputPadded.size() * sizeof(FLOAT16), ssbo, hostMem, inputBuf, inputPadded.data());
    bufOk &= CreateAndFillBuffer(w0.size() * sizeof(FLOAT16), ssbo, hostMem, w0Buf, w0.data());
    bufOk &= CreateAndFillBuffer(w1.size() * sizeof(FLOAT16), ssbo, hostMem, w1Buf, w1.data());
    bufOk &= CreateAndFillBuffer(biasesConcat.size() * sizeof(FLOAT16), ssbo, hostMem, biasesBuf, biasesConcat.data());
    bufOk &= CreateAndFillBuffer(m_host.output.size() * sizeof(FLOAT16), ssbo, hostMem, outputBuf, nullptr);
    auto cleanup = [&]() {
        DestroyBuffer(constantsBuf); DestroyBuffer(inputBuf); DestroyBuffer(w0Buf);
        DestroyBuffer(w1Buf); DestroyBuffer(biasesBuf); DestroyBuffer(outputBuf);
    };
    if (!bufOk) { LOGE("ALU WIDE_IO buffer allocation failed (batch=%u width=%d)", c.batchSize, m_width); cleanup(); return false; }

    RuntimeShader shader;
    shader.AddDefine("FORWARD_GROUP_SIZE", std::string("64"));
    shader.AddDefine("IN_PAD",  std::to_string(inPad));
    shader.AddDefine("OUT_PAD", std::to_string(outPad));
    shader.AddDefine("HIDDEN_MAX", std::string("64"));
    if (!shader.Build(kMlpAluWideIOShader, device, "main", GLSLANG_STAGE_COMPUTE, /*target_vulkan_1_3=*/false)) {
        LOGE("ALU WIDE_IO shader failed to compile");
        cleanup();
        return false;
    }

    // descriptor set layout: 0 UBO, 1..5 SSBO
    VkDescriptorSetLayoutBinding lb[6];
    lb[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    for (uint32_t i = 1; i < 6; ++i)
        lb[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    VkDescriptorSetLayoutCreateInfo li{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    li.bindingCount = 6; li.pBindings = lb;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDescriptorSetLayout(device, &li, nullptr, &dsl));

    VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    CHECK_VK(vkCreatePipelineLayout(device, &pli, nullptr, &pl));

    // spec constants 0..5: in/hidden/out/batch/activation/load_bias
    const uint32_t loadBias = (m_bias_mode == BiasMode::RANDOM) ? 1u : 0u;
    uint32_t spec[6] = { c.inFeatures, c.hiddenFeatures, c.outFeatures, c.batchSize, c.activation, loadBias };
    VkSpecializationMapEntry me[6];
    for (uint32_t i = 0; i < 6; ++i) me[i] = { i, i * (uint32_t)sizeof(uint32_t), sizeof(uint32_t) };
    VkSpecializationInfo specInfo{ 6, me, sizeof(spec), spec };

    VkPipelineShaderStageCreateInfo ss{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ss.module = shader.GetShaderModule();
    ss.pName = "main";
    ss.pSpecializationInfo = &specInfo;
    VkComputePipelineCreateInfo pci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pci.stage = ss; pci.layout = pl;
    VkPipeline pipeline = VK_NULL_HANDLE;
    {
        VkResult pRes = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline);
        if (pRes != VK_SUCCESS || pipeline == VK_NULL_HANDLE) {
            LOGE("ALU WIDE_IO vkCreateComputePipelines failed: %s (%d) [width=%d]. Aborting run.",
                 VkResultStr(pRes), (int)pRes, m_width);
            vkDestroyPipelineLayout(device, pl, nullptr);
            vkDestroyDescriptorSetLayout(device, dsl, nullptr);
            vkDestroyShaderModule(device, shader.GetShaderModule(), nullptr);
            cleanup();
            return false;
        }
    }

    VkDescriptorPoolSize ps[2] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 },
    };
    VkDescriptorPoolCreateInfo pi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pi.poolSizeCount = 2; pi.pPoolSizes = ps; pi.maxSets = 1;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDescriptorPool(device, &pi, nullptr, &dpool));

    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = dpool; ai.descriptorSetCount = 1; ai.pSetLayouts = &dsl;
    VkDescriptorSet set = VK_NULL_HANDLE;
    CHECK_VK(vkAllocateDescriptorSets(device, &ai, &set));

    VkDescriptorBufferInfo bufInfos[6] = {
        { constantsBuf.buffer, 0, constantsBuf.size },
        { inputBuf.buffer,     0, inputBuf.size },
        { w0Buf.buffer,        0, w0Buf.size },
        { w1Buf.buffer,        0, w1Buf.size },
        { biasesBuf.buffer,    0, biasesBuf.size },
        { outputBuf.buffer,    0, outputBuf.size },
    };
    VkDescriptorType types[6] = {
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    VkWriteDescriptorSet writes[6];
    for (uint32_t i = 0; i < 6; ++i)
        writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, i, 0, 1, types[i], nullptr, &bufInfos[i], nullptr };
    vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);

    const uint32_t kAluWorkgroup = 64u;
    const uint32_t maxGroups = m_vulkan_instance.GetGpuProperties().Base.properties.limits.maxComputeWorkGroupCount[0];
    uint32_t groupCountX = (c.batchSize + kAluWorkgroup - 1u) / kAluWorkgroup;
    groupCountX = std::clamp(groupCountX, 1u, maxGroups);
    DispatchCompute(pipeline, pl, set, groupCountX);

    CopyBufferToHost(outputBuf, m_host.output.data(), outputBuf.size);

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pl, nullptr);
    vkDestroyDescriptorPool(device, dpool, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    vkDestroyShaderModule(device, shader.GetShaderModule(), nullptr);
    cleanup();
    return true;
}

// ----------------------------------------------------------------------------
// Assemble the coopmat GLSL string for the selected fuse mode.
// ----------------------------------------------------------------------------
std::string FusedMlpRunner::BuildCoopmatSource() const
{
    std::string src = "#version 460\n";
    src += kCoopExtensions;

    if (m_network == NetKind::WIDE_IO) {
        // 12 -> W -> 10. Output written via the WideOutRow struct (WIDE_IO_OUT),
        // so no Y_ELEM is needed for the output block. Inject the shape literals:
        // IN_K = alignK(12) = 16, OUT_N = snapN(10) = 16.
        const uint32_t inK  = alignK(m_host.constants.inFeatures);   // 16
        const uint32_t outN = snapN(m_host.constants.outFeatures);   // 16
        src += "#define WIDE_IO_OUT 1\n";
        src += "#define IN_K "  + std::to_string(inK)  + "\n";
        src += "#define OUT_N " + std::to_string(outN) + "\n";
        src += "#define OUT_CH " + std::to_string(m_host.constants.outFeatures) + "\n";
        // Element types per strategy (input buffer view).
        if (m_fuse_mode == FuseMode::GLOBAL) {
            src += "#define X_ELEM  float16_t\n";
        } else { // LOCAL and GPR read X as f16vec4
            src += "#define X_ELEM  f16vec4\n";
            if (m_fuse_mode == FuseMode::GPR)
                src += "#define B0_ELEM f16vec4\n#define B1_ELEM f16vec4\n";
        }
        src += kCoopSpecConstants;
        src += kCoopBuffers;
        switch (m_fuse_mode) {
            case FuseMode::GPR:    src += kMlpCoopGprWideIOBody;    break;
            case FuseMode::LOCAL:  src += kMlpCoopLocalWideIOBody;  break;
            case FuseMode::GLOBAL: src += kMlpCoopGlobalWideIOBody; break;
        }
        src += kCoopMain;
        return src;
    }

    // ---- RGBA net (original) ----
    // All coopmat strategies use the QCOM vector<->coopmat conversion for their
    // output store (this is the upstream design; Y_ELEM = f16vec4 and the store
    // goes through coopmatToVectorQCOM). We assume the QCOM conversion extension
    // is available (the app force-loads it). This is what makes the output
    // buffer indexing correct — a float16_t/tiled Y layout produced wrong
    // results for LOCAL.
    const bool qcom = true;

    // per-strategy element type defines
    if (m_fuse_mode == FuseMode::GPR) {
        src += "#define X_ELEM  f16vec4\n";
        src += "#define B0_ELEM f16vec4\n#define B1_ELEM f16vec4\n#define B2_ELEM f16vec4\n#define B3_ELEM f16vec4\n";
        src += "#define Y_ELEM  f16vec4\n";
    } else if (m_fuse_mode == FuseMode::LOCAL) {
        src += "#define X_ELEM  f16vec4\n";
        src += "#define Y_ELEM  f16vec4\n";   // vec4 store via coopmatToVectorQCOM
    } else { // GLOBAL
        src += "#define X_ELEM  float16_t\n";
        src += "#define Y_ELEM  f16vec4\n";   // vec4 store via coopmatToVectorQCOM
    }

    src += kCoopSpecConstants;
    src += kCoopBuffers;

    switch (m_fuse_mode) {
        case FuseMode::GPR:    src += kMlpCoopGprBody;    break;
        case FuseMode::LOCAL:  src += kMlpCoopLocalBody;  break;
        case FuseMode::GLOBAL: src += kMlpCoopGlobalBody; break;
    }
    src += kCoopMain;
    return src;
}

// ----------------------------------------------------------------------------
// Assemble the unfused single-layer kernel GLSL.
//   layerKind 0 = hidden (KxK): X float16_t -> Y float16_t [batch x width]
//   layerKind 1 = output (Kx4): X float16_t -> Y f16vec4  [batch x 4] (QCOM)
// ----------------------------------------------------------------------------
std::string FusedMlpRunner::BuildUnfusedSource(int layerKind) const
{
    std::string src = "#version 460\n";
    src += kCoopExtensions;

    if (m_network == NetKind::WIDE_IO) {
        // layerKind 2 = input (IN_K x W, plain fp16 store), 3 = output (W x OUT_N,
        // vectorized 10-ch store). X is plain float16_t for both.
        const uint32_t inK  = alignK(m_host.constants.inFeatures);   // 16
        const uint32_t outN = snapN(m_host.constants.outFeatures);   // 16
        src += "#define IN_K "  + std::to_string(inK)  + "\n";
        src += "#define OUT_N " + std::to_string(outN) + "\n";
        src += "#define OUT_CH " + std::to_string(m_host.constants.outFeatures) + "\n";
        src += "#define X_ELEM  float16_t\n";
        if (layerKind == 3)
            src += "#define WIDE_IO_OUT 1\n";          // output: WideOutRow store
        else
            src += "#define Y_ELEM  float16_t\n";      // input: plain fp16 hidden tile
        src += kCoopSpecConstants;
        src += kCoopBuffers;
        src += kMlpCoopUnfusedWideIOBody;
        src += kCoopMain;
        return src;
    }

    src += "#define X_ELEM  float16_t\n";
    src += (layerKind == 1) ? "#define Y_ELEM  f16vec4\n"      // output: QCOM vec4 store
                            : "#define Y_ELEM  float16_t\n";   // hidden: plain fp16 tile
    src += kCoopSpecConstants;
    src += kCoopBuffers;
    src += kMlpCoopUnfusedBody;
    src += kCoopMain;
    return src;
}

// ----------------------------------------------------------------------------
// Coopmat path — 20-binding layout (W0..W7, B0..B7), padded weights.
// ----------------------------------------------------------------------------
bool FusedMlpRunner::RunCoopmat()
{
    const auto& c = m_host.constants;
    VkDevice device = m_vulkan_instance.m_VulkanDevice;

    // All coopmat strategies use the QCOM conversion output store (assumed
    // available; app force-loads the extension). Output Y is [batch x 4]:
    // coopmatToVectorQCOM extracts each fiber's 4 outputs, stored as an f16vec4
    // at store_idx = group*64 + fiber, so the row stride is outFeatures (4).
    const bool qcom = true;
    const bool wideIO = (m_network == NetKind::WIDE_IO);
    const uint32_t totalWeights = c.hiddenLayers + 1;
    const uint32_t W = c.hiddenFeatures; // width

    // Output element layout:
    //   RGBA    -> Y = [batch x 4] (f16vec4 store), paddedOut = 4.
    //   WIDE_IO -> Y = [batch x 12] (WideOutRow 2xf16vec4+f16vec2 store, std430
    //              stride 12 halfs), paddedOut = 12, host reads columns 0..9.
    const uint32_t outRowStride = wideIO ? kWideOutStride : kOutFeatures;
    m_host.paddedOut = outRowStride;
    m_host.output.assign(static_cast<size_t>(c.batchSize) * outRowStride, FLOAT16(0));

    const VkBufferUsageFlags    ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkMemoryPropertyFlags hostMem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    Buffer constantsBuf, inputBuf, outputBuf, dummyBuf;
    Buffer scratchBuf;                    // WIDE_IO GLOBAL hidden activation [batch x W]
    Buffer weightBufs[kMaxLayers];
    Buffer biasBufs[kMaxLayers];

    CreateAndFillBuffer(sizeof(FusedMlpConstants), ssbo, hostMem, constantsBuf, &c);

    // input — padded cols to alignK(inFeatures). For all three strategies W is a
    // multiple of 16, so no padding occurs; kept for generality.
    std::vector<FLOAT16> uploadedInput;
    {
        uint32_t padCols = alignK(c.inFeatures);
        if (padCols == c.inFeatures) {
            uploadedInput = m_host.input;
        } else {
            uploadedInput.assign(static_cast<size_t>(c.batchSize) * padCols, FLOAT16(0));
            for (uint32_t r = 0; r < c.batchSize; ++r)
                for (uint32_t cc = 0; cc < c.inFeatures; ++cc)
                    uploadedInput[r * padCols + cc] = m_host.input[r * c.inFeatures + cc];
        }
        CreateAndFillBuffer(uploadedInput.size() * sizeof(FLOAT16), ssbo, hostMem, inputBuf, uploadedInput.data());
    }

    // weights — row-major [inF x outF]; first matrix pads rows to alignK,
    // last matrix pads cols to snapN.
    for (uint32_t l = 0; l < totalWeights; ++l) {
        bool isFirst = (l == 0);
        bool isLast  = (l == totalWeights - 1);
        uint32_t outF = isLast  ? c.outFeatures : c.hiddenFeatures;
        uint32_t inF  = isFirst ? c.inFeatures  : c.hiddenFeatures;
        uint32_t padRows = isFirst ? alignK(c.inFeatures)  : inF;
        uint32_t padCols = isLast  ? snapN(c.outFeatures)  : outF;

        std::vector<FLOAT16> w(static_cast<size_t>(padRows) * padCols, FLOAT16(0));
        for (uint32_t r = 0; r < inF; ++r)
            for (uint32_t cc = 0; cc < outF; ++cc)
                w[r * padCols + cc] = m_host.weights[l][r * outF + cc];
        CreateAndFillBuffer(w.size() * sizeof(FLOAT16), ssbo, hostMem, weightBufs[l], w.data());
    }

    // biases.
    //  - GPR (coopvec): flat per-layer vectors read as f16vec4[] (B*.x[i]).
    //    Hidden layers need 16 halfs (4 vec4); output layer needs >=4 halfs (1 vec4).
    //  - LOCAL/GLOBAL: tiled [64 x N] so the shader coopMatLoads them as
    //    accumulator-shaped bias tiles.
    for (uint32_t l = 0; l < totalWeights; ++l) {
        bool isLast = (l == totalWeights - 1);
        std::vector<FLOAT16> b;
        if (m_fuse_mode == FuseMode::GPR) {
            uint32_t biasN = isLast ? 16u : 16u; // flat, hidden=16 vec elems, output reads first vec4
            b.assign(biasN, FLOAT16(0));
            for (uint32_t i = 0; i < m_host.biases[l].size() && i < biasN; ++i)
                b[i] = m_host.biases[l][i];
        } else {
            uint32_t biasN = isLast ? snapN(c.outFeatures) : c.hiddenFeatures;
            b.assign(static_cast<size_t>(64) * biasN, FLOAT16(0));
            for (uint32_t row = 0; row < 64; ++row)
                for (uint32_t col = 0; col < biasN; ++col)
                    b[row * biasN + col] = (col < m_host.biases[l].size()) ? m_host.biases[l][col] : FLOAT16(0);
        }
        CreateAndFillBuffer(b.size() * sizeof(FLOAT16), ssbo, hostMem, biasBufs[l], b.data());
    }

    // dummy buffer for unused W/B slots (and Xdummy binding 19)
    CreateAndFillBuffer(128 * sizeof(FLOAT16), ssbo, hostMem, dummyBuf, nullptr);

    // WIDE_IO GLOBAL: the hidden activation round-trips through binding 19
    // (Xdummy) since in-place X ping-pong needs in == hidden. Size [batch x W].
    if (wideIO && m_fuse_mode == FuseMode::GLOBAL)
        CreateAndFillBuffer(static_cast<VkDeviceSize>(c.batchSize) * W * sizeof(FLOAT16),
                            ssbo, hostMem, scratchBuf, nullptr);

    // output
    CreateAndFillBuffer(m_host.output.size() * sizeof(FLOAT16), ssbo, hostMem, outputBuf, nullptr);

    // compile
    RuntimeShader shader;
    shader.AddDefine("ADD_BIAS",     std::to_string(c.biasType));
    shader.AddDefine("ACTIVATION",   std::to_string(c.activation));
    shader.AddDefine("UNIFORM_BIAS", std::string("0"));
    shader.AddDefine("USE_QCOM_CONV", std::string(qcom ? "1" : "0"));
    shader.AddDefine("DEBUG_FIBER_INDEX", std::string(m_debug_fiber_index ? "1" : "0"));
    std::string src = BuildCoopmatSource();
    if (!shader.Build(src, device, "main", GLSLANG_STAGE_COMPUTE, /*target_vulkan_1_3=*/false)) {
        LOGE("Coopmat shader failed to compile");
        DestroyBuffer(constantsBuf); DestroyBuffer(inputBuf); DestroyBuffer(outputBuf); DestroyBuffer(dummyBuf);
        for (uint32_t l = 0; l < kMaxLayers; ++l) { DestroyBuffer(weightBufs[l]); DestroyBuffer(biasBufs[l]); }
        return false;
    }

    // descriptor set layout: 0 constants, 1 X, 2..9 W0..W7, 10..17 B0..B7, 18 Y, 19 Xdummy
    const uint32_t totalBindings = 2 + 2 * kMaxLayers + 1 + 1; // 20
    std::vector<VkDescriptorSetLayoutBinding> bindings(totalBindings);
    for (uint32_t i = 0; i < totalBindings; ++i)
        bindings[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    VkDescriptorSetLayoutCreateInfo li{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    li.bindingCount = totalBindings; li.pBindings = bindings.data();
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDescriptorSetLayout(device, &li, nullptr, &dsl));

    VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    CHECK_VK(vkCreatePipelineLayout(device, &pli, nullptr, &pl));

    // spec constants 0..6
    struct { uint32_t lsx, lsy, lsz, wpg, inF, hidF, outF; } spec =
        { 64, 1, 1, 1, c.inFeatures, c.hiddenFeatures, c.outFeatures };
    VkSpecializationMapEntry me[7];
    for (uint32_t i = 0; i < 7; ++i) me[i] = { i, i * (uint32_t)sizeof(uint32_t), sizeof(uint32_t) };
    VkSpecializationInfo specInfo{ 7, me, sizeof(spec), &spec };

    VkPipelineShaderStageCreateInfo ss{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ss.module = shader.GetShaderModule();
    ss.pName = "main";
    ss.pSpecializationInfo = &specInfo;

    // Cooperative-matrix SPIR-V (SPV_KHR_cooperative_matrix) built at SPIR-V 1.5
    // is only legal in a pipeline created with
    // VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT, which also
    // requires pinning the subgroup size (VK_EXT_subgroup_size_control /
    // computeFullSubgroups, both enabled by the framework when available). This
    // avoids needing a SPIR-V 1.6 / Vulkan 1.3 device. Our workgroup is 64-wide
    // and matches the device subgroup size, so require a full 64-wide subgroup.
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT reqSubgroup{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT };
    const auto* sgProps = m_vulkan_instance.GetExtension<ExtensionLib::Vulkan_SubgroupPropertiesHook>();
    const uint32_t subgroupSize = sgProps ? sgProps->Properties.subgroupSize : 0u;
    const bool useFullSubgroups =
        m_vulkan_instance.HasLoadedVulkanDeviceExtension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)
        && subgroupSize == 64u;
    if (useFullSubgroups) {
        reqSubgroup.requiredSubgroupSize = subgroupSize; // 64
        reqSubgroup.pNext = nullptr;
        ss.pNext = &reqSubgroup;
        ss.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
    } else {
        LOGE("Coopmat: VK_EXT_subgroup_size_control with 64-wide subgroups is required to run "
             "cooperative-matrix shaders at SPIR-V 1.5 (subgroupSize=%u). Aborting run.", subgroupSize);
        vkDestroyPipelineLayout(device, pl, nullptr);
        vkDestroyDescriptorSetLayout(device, dsl, nullptr);
        vkDestroyShaderModule(device, shader.GetShaderModule(), nullptr);
        DestroyBuffer(constantsBuf); DestroyBuffer(inputBuf); DestroyBuffer(outputBuf); DestroyBuffer(dummyBuf);
        for (uint32_t l = 0; l < kMaxLayers; ++l) { DestroyBuffer(weightBufs[l]); DestroyBuffer(biasBufs[l]); }
        return false;
    }

    VkComputePipelineCreateInfo pci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pci.stage = ss; pci.layout = pl;
    VkPipeline pipeline = VK_NULL_HANDLE;
    {
        VkResult pRes = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline);
        if (pRes != VK_SUCCESS || pipeline == VK_NULL_HANDLE) {
            LOGE("Coopmat vkCreateComputePipelines failed: %s (%d) [fuse=%d width=%d]. Aborting run.",
                 VkResultStr(pRes), (int)pRes, (int)m_fuse_mode, m_width);
            vkDestroyPipelineLayout(device, pl, nullptr);
            vkDestroyDescriptorSetLayout(device, dsl, nullptr);
            vkDestroyShaderModule(device, shader.GetShaderModule(), nullptr);
            DestroyBuffer(constantsBuf); DestroyBuffer(inputBuf); DestroyBuffer(outputBuf); DestroyBuffer(dummyBuf);
            for (uint32_t l = 0; l < kMaxLayers; ++l) { DestroyBuffer(weightBufs[l]); DestroyBuffer(biasBufs[l]); }
            return false;
        }
    }

    // pool + set
    VkDescriptorPoolSize psz{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, totalBindings };
    VkDescriptorPoolCreateInfo pi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pi.poolSizeCount = 1; pi.pPoolSizes = &psz; pi.maxSets = 1;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDescriptorPool(device, &pi, nullptr, &dpool));

    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = dpool; ai.descriptorSetCount = 1; ai.pSetLayouts = &dsl;
    VkDescriptorSet set = VK_NULL_HANDLE;
    CHECK_VK(vkAllocateDescriptorSets(device, &ai, &set));

    auto bufOrDummy = [&](const Buffer& b) -> const Buffer& { return b.valid() ? b : dummyBuf; };

    std::vector<VkDescriptorBufferInfo> infos(totalBindings);
    infos[0] = { constantsBuf.buffer, 0, constantsBuf.size };
    infos[1] = { inputBuf.buffer, 0, inputBuf.size };
    for (uint32_t l = 0; l < kMaxLayers; ++l) {
        const Buffer& wb = (l < totalWeights) ? bufOrDummy(weightBufs[l]) : dummyBuf;
        infos[2 + l] = { wb.buffer, 0, wb.size };
    }
    for (uint32_t l = 0; l < kMaxLayers; ++l) {
        const Buffer& bb = (l < totalWeights) ? bufOrDummy(biasBufs[l]) : dummyBuf;
        infos[2 + kMaxLayers + l] = { bb.buffer, 0, bb.size };
    }
    infos[2 + 2 * kMaxLayers]     = { outputBuf.buffer, 0, outputBuf.size };
    // Xdummy (binding 19): WIDE_IO GLOBAL uses it as the hidden-activation
    // scratch; otherwise unused (dummy).
    infos[2 + 2 * kMaxLayers + 1] = scratchBuf.valid()
        ? VkDescriptorBufferInfo{ scratchBuf.buffer, 0, scratchBuf.size }
        : VkDescriptorBufferInfo{ dummyBuf.buffer, 0, dummyBuf.size };

    std::vector<VkWriteDescriptorSet> writes(totalBindings);
    for (uint32_t i = 0; i < totalBindings; ++i)
        writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, i, 0, 1,
                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[i], nullptr };
    vkUpdateDescriptorSets(device, totalBindings, writes.data(), 0, nullptr);

    // dispatch: one workgroup per 64 samples (waves_per_group=1)
    uint32_t groupCountX = c.batchSize / 64u;
    // RGBA GLOBAL ping-pongs activations in-place through X, so re-upload the
    // input before each perf-loop dispatch to keep multi-iteration timing valid
    // and the final read-back correct. WIDE_IO GLOBAL instead round-trips the
    // hidden through the binding-19 scratch (X stays read-only), so no re-upload.
    if (m_fuse_mode == FuseMode::GLOBAL && !wideIO)
        DispatchCompute(pipeline, pl, set, groupCountX,
                        &inputBuf, uploadedInput.data(), inputBuf.size);
    else
        DispatchCompute(pipeline, pl, set, groupCountX);

    CopyBufferToHost(outputBuf, m_host.output.data(), outputBuf.size);

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pl, nullptr);
    vkDestroyDescriptorPool(device, dpool, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    vkDestroyShaderModule(device, shader.GetShaderModule(), nullptr);
    DestroyBuffer(constantsBuf); DestroyBuffer(inputBuf); DestroyBuffer(outputBuf); DestroyBuffer(dummyBuf);
    DestroyBuffer(scratchBuf);
    for (uint32_t l = 0; l < kMaxLayers; ++l) { DestroyBuffer(weightBufs[l]); DestroyBuffer(biasBufs[l]); }
    return true;
}

// ----------------------------------------------------------------------------
// Unfused baseline — one coopmat dispatch PER LAYER, intermediates in global
// memory. Same math as the Global fused strategy, but N separate dispatches
// (with a global barrier between) instead of one fused kernel. Reports the
// summed per-inference latency so it can be compared against the fused paths.
// ----------------------------------------------------------------------------
bool FusedMlpRunner::RunUnfused()
{
    const auto& c = m_host.constants;
    VkDevice device = m_vulkan_instance.m_VulkanDevice;

    const uint32_t totalWeights = c.hiddenLayers + 1; // == number of layer dispatches
    const uint32_t W = c.hiddenFeatures;              // width
    const bool wideIO = (m_network == NetKind::WIDE_IO);

    // Output layer store:
    //   RGBA    -> QCOM f16vec4 store, Y = [batch x 4], paddedOut = 4.
    //   WIDE_IO -> WideOutRow store, Y = [batch x 12] (cols 0..9 real), paddedOut = 12.
    m_host.paddedOut = wideIO ? kWideOutStride : kOutFeatures;
    m_host.output.assign(static_cast<size_t>(c.batchSize) * m_host.paddedOut, FLOAT16(0));

    const VkBufferUsageFlags    ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkMemoryPropertyFlags hostMem = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    Buffer constantsBuf, inputBuf, outputBuf, dummyBuf;
    Buffer actA, actB;                 // ping-pong activation buffers ([batch x W] fp16)
    Buffer weightBufs[kMaxLayers];
    Buffer biasBufs[kMaxLayers];

    // WIDE_IO input-layer kernel reads X at stride IN_K = alignK(12) = 16, so pad
    // the input rows 12 -> 16; RGBA uploads the input verbatim ([batch x W]).
    std::vector<FLOAT16> inputUpload;
    if (wideIO) {
        uint32_t inPad = alignK(c.inFeatures); // 16
        inputUpload.assign(static_cast<size_t>(c.batchSize) * inPad, FLOAT16(0));
        for (uint32_t r = 0; r < c.batchSize; ++r)
            for (uint32_t i = 0; i < c.inFeatures; ++i)
                inputUpload[r * inPad + i] = m_host.input[r * c.inFeatures + i];
    } else {
        inputUpload = m_host.input;
    }

    bool bufOk = true;
    bufOk &= CreateAndFillBuffer(sizeof(FusedMlpConstants), ssbo, hostMem, constantsBuf, &c);
    bufOk &= CreateAndFillBuffer(inputUpload.size() * sizeof(FLOAT16), ssbo, hostMem, inputBuf, inputUpload.data());
    bufOk &= CreateAndFillBuffer(static_cast<VkDeviceSize>(c.batchSize) * W * sizeof(FLOAT16), ssbo, hostMem, actA, nullptr);
    bufOk &= CreateAndFillBuffer(static_cast<VkDeviceSize>(c.batchSize) * W * sizeof(FLOAT16), ssbo, hostMem, actB, nullptr);
    bufOk &= CreateAndFillBuffer(m_host.output.size() * sizeof(FLOAT16), ssbo, hostMem, outputBuf, nullptr);
    bufOk &= CreateAndFillBuffer(128 * sizeof(FLOAT16), ssbo, hostMem, dummyBuf, nullptr);

    // weights [in x out] padded (first pads rows to alignK, last pads cols to snapN)
    for (uint32_t l = 0; l < totalWeights; ++l) {
        bool isFirst = (l == 0);
        bool isLast  = (l == totalWeights - 1);
        uint32_t outF = isLast  ? c.outFeatures : c.hiddenFeatures;
        uint32_t inF  = isFirst ? c.inFeatures  : c.hiddenFeatures;
        uint32_t padRows = isFirst ? alignK(c.inFeatures)  : inF;
        uint32_t padCols = isLast  ? snapN(c.outFeatures)  : outF;
        std::vector<FLOAT16> w(static_cast<size_t>(padRows) * padCols, FLOAT16(0));
        for (uint32_t r = 0; r < inF; ++r)
            for (uint32_t cc = 0; cc < outF; ++cc)
                w[r * padCols + cc] = m_host.weights[l][r * outF + cc];
        bufOk &= CreateAndFillBuffer(w.size() * sizeof(FLOAT16), ssbo, hostMem, weightBufs[l], w.data());
    }
    // biases tiled [64 x N] (coopMatLoad as accumulator-shaped bias tile)
    for (uint32_t l = 0; l < totalWeights; ++l) {
        bool isLast = (l == totalWeights - 1);
        uint32_t biasN = isLast ? snapN(c.outFeatures) : c.hiddenFeatures;
        std::vector<FLOAT16> b(static_cast<size_t>(64) * biasN, FLOAT16(0));
        for (uint32_t row = 0; row < 64; ++row)
            for (uint32_t col = 0; col < biasN; ++col)
                b[row * biasN + col] = (col < m_host.biases[l].size()) ? m_host.biases[l][col] : FLOAT16(0);
        bufOk &= CreateAndFillBuffer(b.size() * sizeof(FLOAT16), ssbo, hostMem, biasBufs[l], b.data());
    }

    auto cleanup = [&]() {
        DestroyBuffer(constantsBuf); DestroyBuffer(inputBuf); DestroyBuffer(outputBuf);
        DestroyBuffer(dummyBuf); DestroyBuffer(actA); DestroyBuffer(actB);
        for (uint32_t l = 0; l < kMaxLayers; ++l) { DestroyBuffer(weightBufs[l]); DestroyBuffer(biasBufs[l]); }
    };
    if (!bufOk) { LOGE("Unfused buffer allocation failed (batch=%u width=%d)", c.batchSize, m_width); cleanup(); return false; }

    // Two pipelines: a "hidden/input" kernel and an "output" kernel.
    //   RGBA    -> LAYER_KIND 0 (hidden KxK) / 1 (output Kx4, QCOM).
    //   WIDE_IO -> LAYER_KIND 2 (input IN_KxW) / 3 (output WxOUT_N, WIDE_STORE_10).
    const int kindHidden = wideIO ? 2 : 0;
    const int kindOutput = wideIO ? 3 : 1;
    RuntimeShader shaderHidden, shaderOutput;
    for (int slot = 0; slot < 2; ++slot) {
        RuntimeShader& sh = (slot == 0) ? shaderHidden : shaderOutput;
        const int kind = (slot == 0) ? kindHidden : kindOutput;
        sh.AddDefine("ADD_BIAS",       std::to_string(c.biasType));
        sh.AddDefine("ACTIVATION",     std::to_string(c.activation));
        sh.AddDefine("UNIFORM_BIAS",   std::string("0"));
        // Only the output kernel uses coopmatToVectorQCOM; the hidden/input
        // kernel is a plain coopMatStore, so it must NOT require the QCOM ext.
        sh.AddDefine("USE_QCOM_CONV",  std::string(slot == 1 ? "1" : "0"));
        sh.AddDefine("DEBUG_FIBER_INDEX", std::string(m_debug_fiber_index ? "1" : "0"));
        sh.AddDefine("LAYER_KIND",     std::to_string(kind));
        if (!sh.Build(BuildUnfusedSource(kind), device, "main", GLSLANG_STAGE_COMPUTE, /*target_vulkan_1_3=*/false)) {
            LOGE("Unfused shader (kind=%d) failed to compile", kind);
            cleanup();
            return false;
        }
    }

    // descriptor set layout: same 20-binding layout as the coopmat path.
    const uint32_t totalBindings = 2 + 2 * kMaxLayers + 1 + 1; // 20
    std::vector<VkDescriptorSetLayoutBinding> bindings(totalBindings);
    for (uint32_t i = 0; i < totalBindings; ++i)
        bindings[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    VkDescriptorSetLayoutCreateInfo li{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    li.bindingCount = totalBindings; li.pBindings = bindings.data();
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDescriptorSetLayout(device, &li, nullptr, &dsl));

    VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    CHECK_VK(vkCreatePipelineLayout(device, &pli, nullptr, &pl));

    struct { uint32_t lsx, lsy, lsz, wpg, inF, hidF, outF; } spec =
        { 64, 1, 1, 1, c.inFeatures, c.hiddenFeatures, c.outFeatures };
    VkSpecializationMapEntry me[7];
    for (uint32_t i = 0; i < 7; ++i) me[i] = { i, i * (uint32_t)sizeof(uint32_t), sizeof(uint32_t) };
    VkSpecializationInfo specInfo{ 7, me, sizeof(spec), &spec };

    // Full-subgroups pinning (same requirement as RunCoopmat).
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT reqSubgroup{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT };
    const auto* sgProps = m_vulkan_instance.GetExtension<ExtensionLib::Vulkan_SubgroupPropertiesHook>();
    const uint32_t subgroupSize = sgProps ? sgProps->Properties.subgroupSize : 0u;
    const bool useFullSubgroups =
        m_vulkan_instance.HasLoadedVulkanDeviceExtension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)
        && subgroupSize == 64u;
    if (!useFullSubgroups) {
        LOGE("Unfused: VK_EXT_subgroup_size_control with 64-wide subgroups required (subgroupSize=%u). Aborting.", subgroupSize);
        vkDestroyPipelineLayout(device, pl, nullptr); vkDestroyDescriptorSetLayout(device, dsl, nullptr);
        vkDestroyShaderModule(device, shaderHidden.GetShaderModule(), nullptr);
        vkDestroyShaderModule(device, shaderOutput.GetShaderModule(), nullptr);
        cleanup();
        return false;
    }
    reqSubgroup.requiredSubgroupSize = subgroupSize;

    auto makePipeline = [&](RuntimeShader& sh, VkPipeline& outPipe) -> bool {
        VkPipelineShaderStageCreateInfo ss{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        ss.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ss.module = sh.GetShaderModule();
        ss.pName = "main";
        ss.pSpecializationInfo = &specInfo;
        ss.pNext = &reqSubgroup;
        ss.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
        VkComputePipelineCreateInfo pci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pci.stage = ss; pci.layout = pl;
        VkResult pRes = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &outPipe);
        if (pRes != VK_SUCCESS || outPipe == VK_NULL_HANDLE) {
            LOGE("Unfused vkCreateComputePipelines failed: %s (%d)", VkResultStr(pRes), (int)pRes);
            return false;
        }
        return true;
    };
    VkPipeline pipeHidden = VK_NULL_HANDLE, pipeOutput = VK_NULL_HANDLE;
    if (!makePipeline(shaderHidden, pipeHidden)) {
        LOGE("Unfused: HIDDEN/INPUT pipeline (LAYER_KIND=%d) creation failed. Assembled source:\n%s",
             kindHidden, BuildUnfusedSource(kindHidden).c_str());
    }
    if (pipeHidden != VK_NULL_HANDLE && !makePipeline(shaderOutput, pipeOutput)) {
        LOGE("Unfused: OUTPUT pipeline (LAYER_KIND=%d) creation failed. Assembled source:\n%s",
             kindOutput, BuildUnfusedSource(kindOutput).c_str());
    }
    if (pipeHidden == VK_NULL_HANDLE || pipeOutput == VK_NULL_HANDLE) {
        if (pipeHidden) vkDestroyPipeline(device, pipeHidden, nullptr);
        vkDestroyPipelineLayout(device, pl, nullptr); vkDestroyDescriptorSetLayout(device, dsl, nullptr);
        vkDestroyShaderModule(device, shaderHidden.GetShaderModule(), nullptr);
        vkDestroyShaderModule(device, shaderOutput.GetShaderModule(), nullptr);
        cleanup();
        return false;
    }

    // Descriptor pool: one set per layer.
    VkDescriptorPoolSize psz{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, totalBindings * totalWeights };
    VkDescriptorPoolCreateInfo pi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pi.poolSizeCount = 1; pi.pPoolSizes = &psz; pi.maxSets = totalWeights;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDescriptorPool(device, &pi, nullptr, &dpool));

    // Per-layer descriptor sets. Each binds: constants, X=src, W0=layer weights,
    // B0=layer bias, Y=dst; all other W/B slots + Xdummy -> dummy buffer.
    // Ping-pong: L0 input->actA, L1 actA->actB, L2 actB->actA, ... last->Y.
    std::vector<VkDescriptorSet> sets(totalWeights, VK_NULL_HANDLE);
    std::vector<VkDescriptorSetLayout> layouts(totalWeights, dsl);
    {
        VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorPool = dpool; ai.descriptorSetCount = totalWeights; ai.pSetLayouts = layouts.data();
        CHECK_VK(vkAllocateDescriptorSets(device, &ai, sets.data()));
    }

    // storage for per-set buffer infos (must outlive vkUpdateDescriptorSets)
    std::vector<std::array<VkDescriptorBufferInfo, 20>> infoStore(totalWeights);
    std::vector<VkWriteDescriptorSet> allWrites;
    allWrites.reserve(totalWeights * totalBindings);

    for (uint32_t l = 0; l < totalWeights; ++l) {
        bool isLast = (l == totalWeights - 1);
        // src: L0 = input; else the buffer the previous layer wrote (actA/actB alternating)
        const Buffer& src = (l == 0) ? inputBuf : ((l % 2 == 1) ? actA : actB);
        // dst: last layer -> Y; hidden layers alternate actA/actB
        const Buffer& dst = isLast ? outputBuf : ((l % 2 == 0) ? actA : actB);

        auto& info = infoStore[l];
        info[0] = { constantsBuf.buffer, 0, constantsBuf.size };
        info[1] = { src.buffer, 0, src.size };
        for (uint32_t k = 0; k < kMaxLayers; ++k)                       // W0..W7: W0=this layer's weights, rest dummy
            info[2 + k] = (k == 0) ? VkDescriptorBufferInfo{ weightBufs[l].buffer, 0, weightBufs[l].size }
                                   : VkDescriptorBufferInfo{ dummyBuf.buffer, 0, dummyBuf.size };
        for (uint32_t k = 0; k < kMaxLayers; ++k)                       // B0..B7: B0=this layer's bias, rest dummy
            info[2 + kMaxLayers + k] = (k == 0) ? VkDescriptorBufferInfo{ biasBufs[l].buffer, 0, biasBufs[l].size }
                                                : VkDescriptorBufferInfo{ dummyBuf.buffer, 0, dummyBuf.size };
        info[2 + 2 * kMaxLayers]     = { dst.buffer, 0, dst.size };     // Y = dst
        info[2 + 2 * kMaxLayers + 1] = { dummyBuf.buffer, 0, dummyBuf.size }; // Xdummy unused

        for (uint32_t i = 0; i < totalBindings; ++i)
            allWrites.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sets[l], i, 0, 1,
                                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &info[i], nullptr });
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(allWrites.size()), allWrites.data(), 0, nullptr);

    // Build the layer chain and dispatch (timed as one inference).
    std::vector<DispatchLayer> chain(totalWeights);
    for (uint32_t l = 0; l < totalWeights; ++l) {
        bool isLast = (l == totalWeights - 1);
        chain[l] = { isLast ? pipeOutput : pipeHidden, pl, sets[l] };
    }
    uint32_t groupCountX = c.batchSize / 64u;
    DispatchComputeLayers(chain, groupCountX);

    CopyBufferToHost(outputBuf, m_host.output.data(), outputBuf.size);

    vkDestroyPipeline(device, pipeHidden, nullptr);
    vkDestroyPipeline(device, pipeOutput, nullptr);
    vkDestroyPipelineLayout(device, pl, nullptr);
    vkDestroyDescriptorPool(device, dpool, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    vkDestroyShaderModule(device, shaderHidden.GetShaderModule(), nullptr);
    vkDestroyShaderModule(device, shaderOutput.GetShaderModule(), nullptr);
    cleanup();
    return true;
}

// ----------------------------------------------------------------------------
// Validate — fill m_sample_results + summary (port of fused_mlp::validate).
// ----------------------------------------------------------------------------
void FusedMlpRunner::Validate()
{
    const auto& c = m_host.constants;
    const uint32_t batch    = c.batchSize;
    const uint32_t outF     = c.outFeatures;
    const uint32_t padOut   = m_host.paddedOut;
    const float    eps      = m_validate_eps;

    m_sample_results.clear();
    m_mismatch_count = 0;
    m_invalid_count  = 0;

    // sample list: first-N and last-N (dedup)
    std::vector<uint32_t> samples;
    uint32_t n = std::min<uint32_t>(static_cast<uint32_t>(m_validate_sample_count), batch);
    for (uint32_t s = 0; s < n; ++s) samples.push_back(s);
    for (uint32_t k = 0; k < n; ++k) {
        uint32_t s = (batch == 0) ? 0 : (batch - 1 - k);
        if (std::find(samples.begin(), samples.end(), s) == samples.end()) samples.push_back(s);
    }

    // rows displayed in the UI table: first kDisplay and last kDisplay of the
    // validated set (user-adjustable via "Results rows shown").
    const size_t kDisplay = static_cast<size_t>(std::max(1, m_display_rows));

    for (size_t si = 0; si < samples.size(); ++si) {
        uint32_t sample = samples[si];
        std::vector<FLOAT16> ref;
        CpuReference(ref, sample);

        bool sample_pass = true;
        SampleResult sr{};
        sr.sample = sample;
        for (uint32_t o = 0; o < outF; ++o) {
            float cpu_f = toF32(ref[o]);
            size_t i = static_cast<size_t>(sample) * padOut + o;
            float gpu_f = toF32(m_host.output[i]);
            sr.cpu[o] = cpu_f;
            sr.gpu[o] = gpu_f;
            if (!std::isfinite(gpu_f)) { ++m_invalid_count; ++m_mismatch_count; sample_pass = false; continue; }
            if (std::fabs(cpu_f - gpu_f) > eps) { ++m_mismatch_count; sample_pass = false; }
        }
        sr.pass = sample_pass;

        bool inHead = si < kDisplay;
        bool inTail = si >= samples.size() - kDisplay;
        if (inHead || inTail)
            m_sample_results.push_back(sr);
    }

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    if (m_mismatch_count == 0)
        ss << "PASS - GPU matches CPU reference (eps=" << eps << ")";
    else {
        ss << "FAIL - " << m_mismatch_count << " mismatches";
        if (m_invalid_count) ss << " (" << m_invalid_count << " NaN/inf)";
    }
    if (m_avg_ms >= 0.0f)
        ss << "  |  steady-state avg " << m_avg_ms << " ms, min " << m_min_ms
           << " ms (over " << m_timed_iterations << " iters, " << m_warmup_iterations << " warm-up excluded)";
    else
        ss << "  |  timing n/a (no timestamp support on this queue)";
    m_status_line = ss.str();
    m_last_run_ok = (m_mismatch_count == 0);
}

bool FusedMlpRunner::RunForwardPass()
{
    InitHostData();

    bool ok = false;
    if (m_exec_mode == ExecMode::ALU)
        ok = RunAlu();
    else if (m_exec_mode == ExecMode::UNFUSED)
        ok = RunUnfused();
    else
        ok = RunCoopmat();

    if (!ok) {
        m_status_line = "Run failed (see log).";
        m_last_run_ok = false;
        m_sample_results.clear();
        return false;
    }
    Validate();
    return true;
}

bool FusedMlpRunner::TriggerPendingTests()
{
    if (!m_run_pending) return true;
    m_run_pending = false;
    m_vulkan_instance.WaitUntilIdle();
    RunForwardPass();
    return true;
}

// ----------------------------------------------------------------------------
// ImGui UI
// ----------------------------------------------------------------------------
void FusedMlpRunner::RenderUI()
{
    if (ImGui::CollapsingHeader("MLP Configuration", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // --- Network topology ---
        const char* netNames[] = { "RGBA (in=hidden=W, out=4)", "Wide-IO (12 -> W -> 10)" };
        int net = static_cast<int>(m_network);
        if (ImGui::BeginCombo("Network", netNames[net]))
        {
            for (int i = 0; i < 2; ++i) {
                if (ImGui::Selectable(netNames[i], net == i)) {
                    m_network = static_cast<NetKind>(i);
                    // Wide-IO supports width 16/64 only; snap 32 -> 64. GPR (16-wide)
                    // is handled by the gprForces16 path below.
                    if (m_network == NetKind::WIDE_IO && m_width == 32) m_width = 64;
                }
            }
            ImGui::EndCombo();
        }

        // --- Execution mode ---
        // Coopmat + Unfused both dispatch the coopmat kernels (which use the QCOM
        // conversion ops), so both require m_coopmat_supported (KHR coopmat AND
        // QCOM conversion). Force the mode back to ALU if a coopmat mode is somehow
        // active without support, so a disabled mode can never be dispatched.
        if (!m_coopmat_supported && m_exec_mode != ExecMode::ALU)
            m_exec_mode = ExecMode::ALU;
        const char* modeNames[] = { "ALU", "Coopmat", "Unfused (baseline)" };
        int mode = static_cast<int>(m_exec_mode);
        if (ImGui::BeginCombo("Mode", modeNames[mode]))
        {
            for (int i = 0; i < 3; ++i) {
                // Coopmat and the Unfused baseline both use the coopmat kernel.
                bool disabled = ((i == 1 || i == 2) && !m_coopmat_supported);
                ImGui::BeginDisabled(disabled);
                if (ImGui::Selectable(modeNames[i], mode == i))
                    m_exec_mode = static_cast<ExecMode>(i);
                ImGui::EndDisabled();
            }
            ImGui::EndCombo();
        }
        if (!m_coopmat_supported)
            ImGui::TextColored(ImVec4(1, 0.6f, 0, 1),
                "Coopmat / Unfused modes disabled: VK_KHR_cooperative_matrix + "
                "VK_QCOM_cooperative_matrix_conversion required.");

        // --- Fuse strategy (coopmat only) ---
        ImGui::BeginDisabled(m_exec_mode != ExecMode::COOPMAT);
        const char* fuseNames[] = { "GPR fusing", "Local fusing", "Global fusing" };
        int fuse = static_cast<int>(m_fuse_mode);
        if (ImGui::BeginCombo("Fuse Strategy", fuseNames[fuse]))
        {
            for (int i = 0; i < 3; ++i) {
                bool disabled = (i == 0 && !m_qcom_conv_supported); // GPR needs QCOM conversion
                ImGui::BeginDisabled(disabled);
                if (ImGui::Selectable(fuseNames[i], fuse == i)) {
                    m_fuse_mode = static_cast<FuseMode>(i);
                    if (m_fuse_mode == FuseMode::GPR) m_width = 16; // GPR is 16-wide only
                }
                ImGui::EndDisabled();
            }
            ImGui::EndCombo();
        }
        ImGui::EndDisabled();
        if (m_exec_mode == ExecMode::COOPMAT && !m_qcom_conv_supported)
            ImGui::TextColored(ImVec4(1, 0.6f, 0, 1), "GPR fusing disabled: VK_QCOM_cooperative_matrix_conversion not supported.");
        if (m_exec_mode == ExecMode::UNFUSED)
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1, 1),
                "Unfused baseline: one coopmat dispatch per layer (intermediates via global memory). "
                "Timing is the summed per-inference latency.");

        // --- Width ---
        const int  widths[]     = { 16, 32, 64 };
        const char* widthNames[] = { "16", "32", "64" };
        bool gprForces16 = (m_exec_mode == ExecMode::COOPMAT && m_fuse_mode == FuseMode::GPR);
        bool wideIO = (m_network == NetKind::WIDE_IO);
        int  widthIdx = (m_width == 16) ? 0 : (m_width == 32 ? 1 : 2);
        const char* widthLabel = wideIO ? "Hidden width" : "Width (in == hidden)";
        if (ImGui::BeginCombo(widthLabel, widthNames[widthIdx]))
        {
            for (int i = 0; i < 3; ++i) {
                // GPR locks width to 16; Wide-IO supports 16/64 only (no 32).
                bool disabled = (gprForces16 && i != 0) || (wideIO && i == 1);
                ImGui::BeginDisabled(disabled);
                if (ImGui::Selectable(widthNames[i], widthIdx == i))
                    m_width = widths[i];
                ImGui::EndDisabled();
            }
            ImGui::EndCombo();
        }
        if (gprForces16)
            ImGui::TextColored(ImVec4(1, 0.6f, 0, 1), "GPR fusing is 16-wide only.");
        else if (wideIO)
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1, 1), "Wide-IO: hidden width 16 or 64 (GPR fusing is 16-wide only).");

        // --- Activation / bias ---
        ImGui::Checkbox("ReLU activation (hidden layers)", &m_relu);
        bool biasRandom = (m_bias_mode == BiasMode::RANDOM);
        if (ImGui::Checkbox("Random bias (else zero)", &biasRandom))
            m_bias_mode = biasRandom ? BiasMode::RANDOM : BiasMode::ZERO;

        // --- Batch / validation / perf ---
        // Total number of samples in the batch (must be a multiple of 64 —
        // the ALU kernel uses 64-fiber workgroups). Large batches are covered by
        // the shader's grid-stride loop, so there is no upper limit here.
        if (ImGui::InputInt("Total samples (batch, x64)", &m_batch_size, 64, 4096))
            m_batch_size = std::max(64, (m_batch_size / 64) * 64);
        m_batch_size = std::max(64, (m_batch_size / 64) * 64);

        // Perf loop: how many times the compute pass is dispatched for timing.
        if (ImGui::InputInt("Perf loop (dispatch repeats)", &m_perf_loop, 1, 10))
            m_perf_loop = std::clamp(m_perf_loop, 1, 100000);
        m_perf_loop = std::clamp(m_perf_loop, 1, 100000);

        // Warm-up iterations excluded from the reported (steady-state) timing.
        // The GPU clock ramps up over the first several dispatches (DVFS), so
        // those are much slower; only the iterations after warm-up are averaged.
        if (ImGui::InputInt("Warm-up iterations (excluded)", &m_warmup_iterations, 1, 10))
            m_warmup_iterations = std::clamp(m_warmup_iterations, 0, std::max(0, m_perf_loop - 1));
        m_warmup_iterations = std::clamp(m_warmup_iterations, 0, std::max(0, m_perf_loop - 1));

        // How many samples (from head and tail of the batch) are validated.
        if (ImGui::InputInt("Validate sample count", &m_validate_sample_count, 1, 64))
            m_validate_sample_count = std::clamp(m_validate_sample_count, 1, 1 << 20);
        m_validate_sample_count = std::clamp(m_validate_sample_count, 1, 1 << 20);

        // How many rows the results table shows (head + tail of the validated set).
        if (ImGui::InputInt("Results rows shown (head+tail)", &m_display_rows, 1, 8))
            m_display_rows = std::clamp(m_display_rows, 1, 1024);
        m_display_rows = std::clamp(m_display_rows, 1, 1024);

        ImGui::DragFloat("Validate epsilon", &m_validate_eps, 0.001f, 0.0f, 1.0f, "%.4f");

        // Coopmat debug: store each fiber's subgroup-invocation id (0..63) as the
        // output instead of the MLP result. Correct behaviour => each 64-sample
        // block reads 0,1,2,...,63. If instead only every 64th sample is nonzero,
        // the fibers are collapsing (subgroup not 64-wide / QCOM scatter wrong).
        ImGui::BeginDisabled(m_exec_mode != ExecMode::COOPMAT);
        ImGui::Checkbox("DEBUG: store fiber index (coopmat)", &m_debug_fiber_index);
        ImGui::EndDisabled();

        if (m_network == NetKind::WIDE_IO)
            ImGui::TextDisabled("Network: in=12 -> 1 hidden (W) -> out=10 (linear output)");
        else
            ImGui::TextDisabled("Network: 1 input + 2 hidden + 1 output, out=4 (RGBA)");
    }

    ImGui::Separator();

    if (ImGui::Button("Run"))
        m_run_pending = true;
    ImGui::SameLine();
    ImGui::TextColored(m_last_run_ok ? ImVec4(0.2f, 1, 0.2f, 1) : ImVec4(1, 0.5f, 0.2f, 1), "%s", m_status_line.c_str());

    if (!m_sample_results.empty() && ImGui::CollapsingHeader("Results (subset)", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // Number of output channels shown per row (4 for RGBA, 10 for Wide-IO).
        const uint32_t outF = m_host.constants.outFeatures;
        const char* cpuHdr = (m_network == NetKind::WIDE_IO) ? "CPU (10 ch)" : "CPU (RGBA)";
        const char* gpuHdr = (m_network == NetKind::WIDE_IO) ? "GPU (10 ch)" : "GPU (RGBA)";

        // Build a per-row "%.4f %.4f ..." string from the first outF channels.
        auto fmtChannels = [outF](const float* v) {
            std::ostringstream os;
            os << std::fixed << std::setprecision(4);
            for (uint32_t o = 0; o < outF; ++o) {
                if (o) os << ' ';
                os << v[o];
            }
            return os.str();
        };

        if (ImGui::BeginTable("MlpResults", 4,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY,
            ImVec2(0, 280)))
        {
            ImGui::TableSetupColumn("Sample", ImGuiTableColumnFlags_WidthFixed, 70.0f);
            ImGui::TableSetupColumn(cpuHdr);
            ImGui::TableSetupColumn(gpuHdr);
            ImGui::TableSetupColumn("Result", ImGuiTableColumnFlags_WidthFixed, 70.0f);
            ImGui::TableHeadersRow();

            for (const auto& r : m_sample_results) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%u", r.sample);
                ImGui::TableSetColumnIndex(1);
                ImGui::TextUnformatted(fmtChannels(r.cpu).c_str());
                ImGui::TableSetColumnIndex(2);
                ImGui::TextUnformatted(fmtChannels(r.gpu).c_str());
                ImGui::TableSetColumnIndex(3);
                ImGui::PushStyleColor(ImGuiCol_Text, r.pass ? ImVec4(0.2f, 1, 0.2f, 1) : ImVec4(1, 0.2f, 0.2f, 1));
                ImGui::Text(r.pass ? "PASS" : "FAIL");
                ImGui::PopStyleColor();
            }
            ImGui::EndTable();
        }
    }
}
