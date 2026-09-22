// Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "anf_gpu_profiler.hpp"
#include "vulkan/commandBuffer.hpp"
#include "vulkan/timerPool.hpp"

#include <array>
#include <span>
#include <vector>

class Vulkan;

class AnfDispatchTimingController final
{
public:
    enum class Technique : uint32_t
    {
        Sr = 0,
        Fg,
        Count
    };

    struct DispatchToken
    {
        bool        UsesIndirect         = false;
        bool        UsesImmediateMarkers = false;
        VkCommandBuffer DispatchCommandBuffer = VK_NULL_HANDLE;
        int         TimerId              = -1;
        std::array<VkSemaphore, 1> DispatchWaitSemaphores{};
        std::array<VkSemaphore, 1> DispatchSignalSemaphores{};
        uint32_t    DispatchWaitCount    = 0;
        uint32_t    DispatchSignalCount  = 0;
    };

    bool Initialize(Vulkan* vulkan, TimerPoolBase* timerPool, bool dispatchImmediate);
    void Release();

    void SetDispatchImmediate(bool dispatchImmediate) { m_DispatchImmediate = dispatchImmediate; }
    bool IsDispatchImmediate() const { return m_DispatchImmediate; }

    bool CreateSemaphores(VkDevice device);
    void DestroySemaphores(VkDevice device);

    DispatchToken BeginDispatch(
        Technique                    technique,
        uint32_t                     whichBuffer,
        std::span<const VkSemaphore> waitSemaphores,
        AnfGpuProfiler*             profiler,
        AnfGpuProfiler::Region      profilerRegion);

    std::span<const VkSemaphore> GetDispatchWaitSemaphores(
        const DispatchToken&            token,
        std::span<const VkSemaphore> fallbackWaitSemaphores) const;

    std::span<const VkSemaphore> GetDispatchSignalSemaphores(
        const DispatchToken&            token,
        std::span<const VkSemaphore> fallbackSignalSemaphores) const;

    VkCommandBuffer GetDispatchCommandBuffer(const DispatchToken& token) const
    {
        return token.DispatchCommandBuffer;
    }

    void EndDispatch(
        Technique                          technique,
        uint32_t                           whichBuffer,
        std::span<const VkSemaphore>       submitWaitSemaphores,
        std::span<const VkPipelineStageFlags> submitWaitStageMasks,
        VkSemaphore                        finalSignalSemaphore,
        const DispatchToken&               token);

    void CancelDispatch(
        Technique            technique,
        uint32_t             whichBuffer,
        const DispatchToken& token);

private:
    struct TechniqueTiming
    {
        std::array<CommandListVulkan, NUM_VULKAN_BUFFERS> DispatchCmdList;
        std::array<VkSemaphore, NUM_VULKAN_BUFFERS> BeginSemaphores{};
        std::array<VkSemaphore, NUM_VULKAN_BUFFERS> EndSemaphores{};
        std::array<CommandListVulkan, NUM_VULKAN_BUFFERS> StartCmdList;
        std::array<CommandListVulkan, NUM_VULKAN_BUFFERS> StopCmdList;
    };

    static const char* TechniqueName(Technique technique);
    TechniqueTiming& GetTechniqueTiming(Technique technique);
    const TechniqueTiming& GetTechniqueTiming(Technique technique) const;

private:
    Vulkan*         m_Vulkan          = nullptr;
    TimerPoolBase* m_TimerPool = nullptr;
    bool            m_Initialized     = false;
    bool            m_DispatchImmediate = true;
    std::array<TechniqueTiming, static_cast<size_t>(Technique::Count)> m_Techniques{};
};
