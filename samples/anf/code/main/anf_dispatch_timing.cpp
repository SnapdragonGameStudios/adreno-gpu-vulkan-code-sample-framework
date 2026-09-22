// Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "anf_dispatch_timing.hpp"

#include "vulkan/vulkan.hpp"

#include <cstdio>

namespace
{
    constexpr size_t TechniqueToIndex(AnfDispatchTimingController::Technique technique)
    {
        return static_cast<size_t>(technique);
    }
}

bool AnfDispatchTimingController::Initialize(Vulkan* vulkan, TimerPoolBase* timerPool, bool dispatchImmediate)
{
    Release();

    m_Vulkan            = vulkan;
    m_TimerPool         = timerPool;
    m_DispatchImmediate = dispatchImmediate;
    if (!m_Vulkan)
        return false;

    char name[128]{};
    for (uint32_t techniqueIdx = 0; techniqueIdx < static_cast<uint32_t>(Technique::Count); ++techniqueIdx)
    {
        auto& technique = m_Techniques[techniqueIdx];
        const auto tech = static_cast<Technique>(techniqueIdx);
        const char* techniqueName = TechniqueName(tech);

        for (uint32_t i = 0; i < NUM_VULKAN_BUFFERS; ++i)
        {
            std::snprintf(name, sizeof(name), "ANF %s Dispatch CMD Buffer %u", techniqueName, i);
            if (!technique.DispatchCmdList[i].Initialize(m_Vulkan, name, CommandListBase::Type::Primary, Vulkan::eGraphicsQueue, m_TimerPool))
            {
                Release();
                return false;
            }

            std::snprintf(name, sizeof(name), "ANF %s Timing Start CMD Buffer %u", techniqueName, i);
            if (!technique.StartCmdList[i].Initialize(m_Vulkan, name, CommandListBase::Type::Primary, Vulkan::eGraphicsQueue, m_TimerPool))
            {
                Release();
                return false;
            }

            std::snprintf(name, sizeof(name), "ANF %s Timing Stop CMD Buffer %u", techniqueName, i);
            if (!technique.StopCmdList[i].Initialize(m_Vulkan, name, CommandListBase::Type::Primary, Vulkan::eGraphicsQueue, m_TimerPool))
            {
                Release();
                return false;
            }
        }
    }

    m_Initialized = true;
    return true;
}

void AnfDispatchTimingController::Release()
{
    for (auto& technique : m_Techniques)
    {
        for (auto& cmdList : technique.DispatchCmdList)
            cmdList.Release();
        for (auto& cmdList : technique.StartCmdList)
            cmdList.Release();
        for (auto& cmdList : technique.StopCmdList)
            cmdList.Release();
    }

    m_Initialized = false;
    m_TimerPool   = nullptr;
    m_Vulkan      = nullptr;
}

bool AnfDispatchTimingController::CreateSemaphores(VkDevice device)
{
    const VkSemaphoreCreateInfo semaphoreInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    for (auto& technique : m_Techniques)
    {
        for (auto& semaphore : technique.BeginSemaphores)
        {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS)
            {
                DestroySemaphores(device);
                return false;
            }
        }
        for (auto& semaphore : technique.EndSemaphores)
        {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS)
            {
                DestroySemaphores(device);
                return false;
            }
        }
    }

    return true;
}

void AnfDispatchTimingController::DestroySemaphores(VkDevice device)
{
    for (auto& technique : m_Techniques)
    {
        for (auto& semaphore : technique.BeginSemaphores)
        {
            if (semaphore != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(device, semaphore, nullptr);
                semaphore = VK_NULL_HANDLE;
            }
        }
        for (auto& semaphore : technique.EndSemaphores)
        {
            if (semaphore != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(device, semaphore, nullptr);
                semaphore = VK_NULL_HANDLE;
            }
        }
    }
}

AnfDispatchTimingController::DispatchToken AnfDispatchTimingController::BeginDispatch(
    Technique                    technique,
    uint32_t                     whichBuffer,
    std::span<const VkSemaphore> waitSemaphores,
    AnfGpuProfiler*             profiler,
    AnfGpuProfiler::Region      profilerRegion)
{
    DispatchToken token{};
    auto& timing = GetTechniqueTiming(technique);

    if (!m_DispatchImmediate)
    {
        auto& dispatchCmd = timing.DispatchCmdList[whichBuffer];
        dispatchCmd.Reset();
        dispatchCmd.Begin();
        token.TimerId = profiler ? profiler->BeginRegion(dispatchCmd, profilerRegion) : -1;
        token.UsesIndirect         = true;
        token.DispatchCommandBuffer = dispatchCmd.m_VkCommandBuffer;
        return token;
    }

    if (!profiler || !profiler->IsInitialized())
        return token;

    auto& startCmd = timing.StartCmdList[whichBuffer];
    startCmd.Reset();
    startCmd.Begin();
    token.TimerId = profiler->BeginRegion(startCmd, profilerRegion);
    startCmd.End();

    std::vector<VkPipelineStageFlags> waitStages(waitSemaphores.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    startCmd.QueueSubmit(waitSemaphores, waitStages, { &timing.BeginSemaphores[whichBuffer], 1 });

    token.UsesImmediateMarkers = true;
    token.DispatchWaitSemaphores[0]   = timing.BeginSemaphores[whichBuffer];
    token.DispatchSignalSemaphores[0] = timing.EndSemaphores[whichBuffer];
    token.DispatchWaitCount           = 1;
    token.DispatchSignalCount         = 1;
    return token;
}

std::span<const VkSemaphore> AnfDispatchTimingController::GetDispatchWaitSemaphores(
    const DispatchToken&         token,
    std::span<const VkSemaphore> fallbackWaitSemaphores) const
{
    if (token.UsesImmediateMarkers && token.DispatchWaitCount > 0)
        return { token.DispatchWaitSemaphores.data(), token.DispatchWaitCount };
    return fallbackWaitSemaphores;
}

std::span<const VkSemaphore> AnfDispatchTimingController::GetDispatchSignalSemaphores(
    const DispatchToken&         token,
    std::span<const VkSemaphore> fallbackSignalSemaphores) const
{
    if (token.UsesImmediateMarkers && token.DispatchSignalCount > 0)
        return { token.DispatchSignalSemaphores.data(), token.DispatchSignalCount };
    return fallbackSignalSemaphores;
}

void AnfDispatchTimingController::EndDispatch(
    Technique                          technique,
    uint32_t                           whichBuffer,
    std::span<const VkSemaphore>       submitWaitSemaphores,
    std::span<const VkPipelineStageFlags> submitWaitStageMasks,
    VkSemaphore                        finalSignalSemaphore,
    const DispatchToken&               token)
{
    auto& timing = GetTechniqueTiming(technique);

    if (token.UsesIndirect)
    {
        auto& dispatchCmd = timing.DispatchCmdList[whichBuffer];
        AnfGpuProfiler::EndRegion(dispatchCmd, token.TimerId);
        dispatchCmd.End();

        std::vector<VkPipelineStageFlags> defaultWaitStages;
        auto waitStages = submitWaitStageMasks;
        if (submitWaitSemaphores.size() > 0 && submitWaitStageMasks.size() != submitWaitSemaphores.size())
        {
            defaultWaitStages.assign(submitWaitSemaphores.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            waitStages = defaultWaitStages;
        }

        if (finalSignalSemaphore != VK_NULL_HANDLE)
        {
            const std::array<VkSemaphore, 1> signalSemaphores = { finalSignalSemaphore };
            dispatchCmd.QueueSubmit(submitWaitSemaphores, waitStages, signalSemaphores);
        }
        else
        {
            dispatchCmd.QueueSubmit(submitWaitSemaphores, waitStages, std::span<const VkSemaphore>{});
        }
        return;
    }

    if (!token.UsesImmediateMarkers)
        return;

    auto& stopCmd = timing.StopCmdList[whichBuffer];
    stopCmd.Reset();
    stopCmd.Begin();
    AnfGpuProfiler::EndRegion(stopCmd, token.TimerId);
    stopCmd.End();

    static const VkPipelineStageFlags stopWaitStage[] = { VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
    const VkSemaphore timingEnd = timing.EndSemaphores[whichBuffer];

    if (finalSignalSemaphore != VK_NULL_HANDLE)
    {
        const std::array<VkSemaphore, 1> signalSemaphores = { finalSignalSemaphore };
        stopCmd.QueueSubmit({ &timingEnd, 1 }, { stopWaitStage, 1 }, signalSemaphores);
    }
    else
    {
        stopCmd.QueueSubmit({ &timingEnd, 1 }, { stopWaitStage, 1 }, std::span<const VkSemaphore>{});
    }
}

void AnfDispatchTimingController::CancelDispatch(
    Technique            technique,
    uint32_t             whichBuffer,
    const DispatchToken& token)
{
    if (!token.UsesIndirect)
        return;

    auto& dispatchCmd = GetTechniqueTiming(technique).DispatchCmdList[whichBuffer];
    AnfGpuProfiler::EndRegion(dispatchCmd, token.TimerId);
    dispatchCmd.End();
}

const char* AnfDispatchTimingController::TechniqueName(Technique technique)
{
    switch (technique)
    {
    case Technique::Sr:   return "SR";
    case Technique::Fg:   return "FG";
    case Technique::Count: break;
    }
    return "Unknown";
}

AnfDispatchTimingController::TechniqueTiming& AnfDispatchTimingController::GetTechniqueTiming(Technique technique)
{
    return m_Techniques[TechniqueToIndex(technique)];
}

const AnfDispatchTimingController::TechniqueTiming& AnfDispatchTimingController::GetTechniqueTiming(Technique technique) const
{
    return m_Techniques[TechniqueToIndex(technique)];
}
