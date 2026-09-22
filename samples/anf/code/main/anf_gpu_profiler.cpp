// Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "anf_gpu_profiler.hpp"

#include "vulkan/vulkan.hpp"

namespace
{
    constexpr size_t RegionToIndex(AnfGpuProfiler::Region region)
    {
        return static_cast<size_t>(region);
    }
}

double AnfGpuProfiler::Metrics::AppBaselineLastMs() const
{
    const auto& scene = Get(Region::SceneReal);
    if (!scene.Valid) return 0.0;

    double result = scene.LastMs;
    if (const auto& hud = Get(Region::HudReal); hud.Valid) result += hud.LastMs;
    if (const auto& blit = Get(Region::BlitReal); blit.Valid) result += blit.LastMs;
    return result;
}

double AnfGpuProfiler::Metrics::AppBaselineAvgMs() const
{
    const auto& scene = Get(Region::SceneReal);
    if (!scene.Valid) return 0.0;

    double result = scene.AvgMs;
    if (const auto& hud = Get(Region::HudReal); hud.Valid) result += hud.AvgMs;
    if (const auto& blit = Get(Region::BlitReal); blit.Valid) result += blit.AvgMs;
    return result;
}

double AnfGpuProfiler::Metrics::FgTotalLastMs() const
{
    double result = 0.0;
    if (const auto& prep = Get(Region::FgPrepareReal); prep.Valid) result += prep.LastMs;
    if (const auto& dispatch = Get(Region::FgDispatch); dispatch.Valid) result += dispatch.LastMs;
    return result;
}

double AnfGpuProfiler::Metrics::FgTotalAvgMs() const
{
    double result = 0.0;
    if (const auto& prep = Get(Region::FgPrepareReal); prep.Valid) result += prep.AvgMs;
    if (const auto& dispatch = Get(Region::FgDispatch); dispatch.Valid) result += dispatch.AvgMs;
    return result;
}

AnfGpuProfiler::AnfGpuProfiler(Vulkan& vulkan) noexcept
    : m_Vulkan(vulkan)
    , m_TimerPool(vulkan)
{
    m_GraphicsQueueFamily = static_cast<uint32_t>(m_Vulkan.m_VulkanQueues[Vulkan::eGraphicsQueue].QueueFamilyIndex);
}

bool AnfGpuProfiler::Initialize(uint32_t maxTimersPerFrame)
{
    if (m_Initialized) return true;
    m_Initialized = m_TimerPool.Initialize(maxTimersPerFrame);
    return m_Initialized;
}

void AnfGpuProfiler::Destroy()
{
    if (!m_Initialized) return;
    m_TimerPool.Destroy();
    m_Initialized = false;
    m_Metrics     = {};
}

TimerPoolBase* AnfGpuProfiler::GetTimerPoolBase()
{
    return static_cast<TimerPoolBase*>(&m_TimerPool);
}

int AnfGpuProfiler::BeginRegion(CommandListVulkan& cmd, Region region) const
{
    if (!m_Initialized) return -1;
    return cmd.StartGpuTimer(RegionName(region));
}

void AnfGpuProfiler::EndRegion(CommandListVulkan& cmd, int timerId)
{
    cmd.StopGpuTimer(timerId);
}

void AnfGpuProfiler::RecordReadback(CommandListVulkan& cmd, uint32_t whichBuffer)
{
    if (!m_Initialized) return;
    m_TimerPool.ReadResults(cmd.m_VkCommandBuffer, whichBuffer);
}

void AnfGpuProfiler::UpdateCompleted(uint32_t whichBuffer)
{
    if (!m_Initialized) return;

    m_TimerPool.UpdateResults(whichBuffer);

    for (size_t i = 0; i < RegionToIndex(Region::Count); ++i)
    {
        UpdateRegionSample(static_cast<Region>(i));
    }
}

const char* AnfGpuProfiler::RegionName(Region region)
{
    switch (region)
    {
    case Region::SceneReal:     return "ANF Scene Real";
    case Region::SrDispatch:    return "ANF SR Dispatch";
    case Region::FgPrepareReal: return "ANF FG Prepare Real";
    case Region::FgDispatch:    return "ANF FG Dispatch";
    case Region::HudReal:       return "ANF HUD Real";
    case Region::HudFake:       return "ANF HUD Fake";
    case Region::BlitReal:      return "ANF Blit Real";
    case Region::BlitFake:      return "ANF Blit Fake";
    case Region::Count:         break;
    }
    return "ANF Unknown";
}

const TimerSimple* AnfGpuProfiler::FindRegionTimer(Region region) const
{
    const auto& timers = m_TimerPool.GetResults();
    auto timerIt = timers.find(std::pair{ std::string_view(RegionName(region)), m_GraphicsQueueFamily });
    if (timerIt == timers.end() || timerIt->CompletedCount <= 0)
        return nullptr;
    return &(*timerIt);
}

void AnfGpuProfiler::UpdateRegionSample(Region region)
{
    const auto* timer = FindRegionTimer(region);
    if (!timer) return;

    auto& sample  = m_Metrics.Regions[RegionToIndex(region)];
    sample.Valid  = true;
    sample.LastMs = m_TimerPool.GetTimeInMs(*timer);
    sample.AvgMs  = m_TimerPool.GetAverageTimeInMs(*timer);
}
