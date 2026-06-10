//============================================================================================================
//
//                  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include "vulkan/timerSimple.hpp"
#include <array>
#include <chrono>
#include <string_view>

enum class TimingSlot : uint32_t
{
    Scene        = 0,
    LightCull    = 1,
    DeferredLight= 2,
    TotalFrame   = 3,
    kCount
};

static constexpr uint32_t kTimingSlotCount = static_cast<uint32_t>(TimingSlot::kCount);

static constexpr std::array<const char*, kTimingSlotCount> kTimingSlotNames = {
    "Scene",
    "Light Cull",
    "Deferred Light",
    "Total Frame",
};

struct PassTimingResult
{
    double cpuMs = 0.0;
    double gpuMs = 0.0;
};

class PassTimings
{
public:
    bool Initialize(Vulkan& vulkan, uint32_t maxTimers = 32)
    {
        m_TimerPool = std::make_unique<TimerPoolSimple>(vulkan);
        return m_TimerPool->Initialize(maxTimers);
    }

    void Destroy()
    {
        if (m_TimerPool)
        {
            m_TimerPool->Destroy();
            m_TimerPool.reset();
        }
    }

    void BeginFrame(VkCommandBuffer cmdBuf, uint32_t bufferIdx)
    {
        if (m_TimerPool)
            m_TimerPool->ReadResults(cmdBuf, bufferIdx);
    }

    TimerPoolBase::TimerId GpuBegin(CommandListVulkan& cmdList, TimingSlot slot)
    {
        if (!m_TimerPool) return -1;
        return cmdList.StartGpuTimer(kTimingSlotNames[static_cast<uint32_t>(slot)]);
    }

    void GpuEnd(CommandListVulkan& cmdList, TimerPoolBase::TimerId id)
    {
        if (id >= 0)
            cmdList.StopGpuTimer(id);
    }

    void UpdateResults(uint32_t bufferIdx)
    {
        if (m_TimerPool)
            m_TimerPool->UpdateResults(bufferIdx);

        if (m_TimerPool)
        {
            for (const auto& timer : m_TimerPool->GetResults())
            {
                for (uint32_t s = 0; s < kTimingSlotCount; ++s)
                {
                    if (timer.Name == kTimingSlotNames[s])
                    {
                        m_Results[s].gpuMs = m_TimerPool->GetAverageTimeInMs(timer);
                        break;
                    }
                }
            }
        }
    }

    using Clock    = std::chrono::high_resolution_clock;
    using TimePoint= Clock::time_point;

    void CpuBegin(TimingSlot slot)
    {
        m_CpuStart[static_cast<uint32_t>(slot)] = Clock::now();
    }

    void CpuEnd(TimingSlot slot)
    {
        const uint32_t idx = static_cast<uint32_t>(slot);
        const double dt = std::chrono::duration<double, std::milli>(
            Clock::now() - m_CpuStart[idx]).count();
        // EMA (α=0.1)
        m_Results[idx].cpuMs = m_Results[idx].cpuMs * 0.9 + dt * 0.1;
    }

    const PassTimingResult& GetResult(TimingSlot slot) const
    {
        return m_Results[static_cast<uint32_t>(slot)];
    }

    TimerPoolBase* GetTimerPool() const { return m_TimerPool.get(); }

    void ResetGpuAverages()
    {
        if (m_TimerPool)
            m_TimerPool->ResetTimers(-1);
        for (auto& r : m_Results)
        {
            r.gpuMs = 0.0;
            r.cpuMs = 0.0;
        }
    }

private:
    std::unique_ptr<TimerPoolSimple>                       m_TimerPool;
    std::array<PassTimingResult, kTimingSlotCount>         m_Results{};
    std::array<TimePoint,        kTimingSlotCount>         m_CpuStart{};
};
