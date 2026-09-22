// Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "vulkan/commandBuffer.hpp"
#include "vulkan/timerSimple.hpp"

#include <array>
#include <cstdint>
#include <string_view>

class Vulkan;

class AnfGpuProfiler final
{
public:
    enum class Region : uint32_t
    {
        SceneReal = 0,
        SrDispatch,
        FgPrepareReal,
        FgDispatch,
        HudReal,
        HudFake,
        BlitReal,
        BlitFake,
        Count
    };

    struct Sample
    {
        bool   Valid  = false;
        double LastMs = 0.0;
        double AvgMs  = 0.0;
    };

    struct Metrics
    {
        std::array<Sample, static_cast<size_t>(Region::Count)> Regions{};

        const Sample& Get(Region region) const { return Regions[static_cast<size_t>(region)]; }

        double AppBaselineLastMs() const;
        double AppBaselineAvgMs() const;
        double FgTotalLastMs() const;
        double FgTotalAvgMs() const;
    };

    explicit AnfGpuProfiler(Vulkan& vulkan) noexcept;

    bool Initialize(uint32_t maxTimersPerFrame = 32);
    void Destroy();

    bool IsInitialized() const { return m_Initialized; }
    TimerPoolBase* GetTimerPoolBase();

    int BeginRegion(CommandListVulkan& cmd, Region region) const;
    static void EndRegion(CommandListVulkan& cmd, int timerId);

    void RecordReadback(CommandListVulkan& cmd, uint32_t whichBuffer);
    void UpdateCompleted(uint32_t whichBuffer);

    const Metrics& GetMetrics() const { return m_Metrics; }

private:
    static const char* RegionName(Region region);
    const TimerSimple* FindRegionTimer(Region region) const;
    void UpdateRegionSample(Region region);

private:
    Vulkan&         m_Vulkan;
    TimerPoolSimple m_TimerPool;
    uint32_t        m_GraphicsQueueFamily = 0;
    bool            m_Initialized         = false;
    Metrics         m_Metrics{};
};
