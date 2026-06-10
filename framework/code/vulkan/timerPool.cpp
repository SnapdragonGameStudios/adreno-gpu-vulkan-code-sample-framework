//============================================================================================================
//
//
//                  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "timerPool.hpp"
#include "extensionLib.hpp"
#include <algorithm>
#include <iterator>


TimerPoolBase::TimerPoolBase( Vulkan& vulkan ) noexcept : m_Vulkan( vulkan )
{
}


TimerPoolBase::~TimerPoolBase()
{
    Destroy();
}


bool TimerPoolBase::Initialize( uint32_t maxTimers )
{
    assert(m_VulkanQueryPool == VK_NULL_HANDLE);
    assert(m_MaxUsedTimerIndex == -1);

    if (m_Vulkan.GetGpuProperties().Base.properties.limits.timestampComputeAndGraphics != VK_TRUE)
        return false;
    m_TimeStampPeriod = m_Vulkan.GetGpuProperties().Base.properties.limits.timestampPeriod;

    VkQueryPoolCreateInfo QueryInfo = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    QueryInfo.flags = 0;
    QueryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    uint32_t timerCount = NUM_VULKAN_BUFFERS * maxTimers;
    QueryInfo.queryCount = timerCount * 2/*one for start time, one for stop time*/;      // size based on number of frames, may fall-down if we want/support timers that run across frame boundaries.
    QueryInfo.pipelineStatistics = 0;

    const auto* hostQueryResetExt = m_Vulkan.GetExtension<ExtensionLib::Ext_VK_EXT_host_query_reset>();
    if (hostQueryResetExt && hostQueryResetExt->Status == VulkanExtensionStatus::eLoaded)
    {
        m_vkResetQueryPoolEXT = hostQueryResetExt->m_vkResetQueryPoolEXT;
    }
    else
    {
        // VK_EXT_host_query_reset was promoted to Vulkan 1.2 core as vkResetQueryPool.
        // On 1.2+ devices the extension may not be listed; try loading both entry points.
        m_vkResetQueryPoolEXT = (PFN_vkResetQueryPoolEXT) vkGetDeviceProcAddr( m_Vulkan.m_VulkanDevice, "vkResetQueryPoolEXT" );
        if (!m_vkResetQueryPoolEXT)
            m_vkResetQueryPoolEXT = (PFN_vkResetQueryPoolEXT) vkGetDeviceProcAddr( m_Vulkan.m_VulkanDevice, "vkResetQueryPool" );
    }
    if (!m_vkResetQueryPoolEXT)
    {
        LOGE("TimerPoolBase: vkResetQueryPool(EXT) not available, GPU timers disabled.");
        return false;
    }

    auto RetVal = vkCreateQueryPool( m_Vulkan.m_VulkanDevice, &QueryInfo, nullptr, &m_VulkanQueryPool );
    if (!CheckVkError( "vkCreateQueryPool()", RetVal ))
    {
        return false;
    }

    // Setup the GPU buffers that the timing results are copied in to.
    void* zeroData = calloc(QueryInfo.queryCount, sizeof(VulkanQueryResult));
    if (!CreateUniformBuffer(&m_Vulkan, m_VulkanQueryResults, QueryInfo.queryCount * sizeof(VulkanQueryResult), zeroData, BufferUsageFlags::TransferDst | BufferUsageFlags::TransferSrc))
    {
        return false;
    }
    free(zeroData); zeroData = nullptr;

    // Setup the storage for the 'inflight' timers (one per timer)
    m_TimerQueries.resize( timerCount, {} );

    ResetQueryPool( 0, QueryInfo.queryCount );

    // Queues can have different timer 'valid bits', grab the values for each queue.
    std::transform( m_Vulkan.m_pVulkanQueueProps.begin(), m_Vulkan.m_pVulkanQueueProps.end(), std::back_inserter(m_DeviceQueueValidTimerBitMask), []( const auto& a ) -> uint64_t {
        uint64_t v = a.timestampValidBits < 64 ? (uint64_t(1) << uint64_t(a.timestampValidBits)) : 0;
        return v - uint64_t(1);
    });

    // Fill the 'free' pool entries with 'first' timerId (fill backwards so first value popped off list will be 0)
    for (int timerId = timerCount - 1; timerId >= 0; timerId--)
        m_FreeQueryPoolEntries.push( timerId );

    m_TimerCount = timerCount;

    return true;
}


void TimerPoolBase::Destroy()
{
    if (m_VulkanQueryPool != VK_NULL_HANDLE)
    {
        vkDestroyQueryPool( m_Vulkan.m_VulkanDevice, m_VulkanQueryPool, nullptr );
        m_VulkanQueryPool = VK_NULL_HANDLE;
    }
    ReleaseUniformBuffer( &m_Vulkan, m_VulkanQueryResults );
}


TimerPoolBase::TimerId TimerPoolBase::StartGpuTimer( VkCommandBuffer commandBuffer, const std::string_view& timerName, int deviceQueueFamilyIndex )
{
    assert( deviceQueueFamilyIndex >= 0 );

    ///TODO: mutex here if multithreading
    if (m_FreeQueryPoolEntries.empty())
        return -1;
    uint32_t timerId = m_FreeQueryPoolEntries.top();
    m_FreeQueryPoolEntries.pop();

    if (m_MaxUsedTimerIndex < (int)timerId)
        m_MaxUsedTimerIndex = timerId;

    TimerBase* pTimer = FindOrAddTimer(timerName, deviceQueueFamilyIndex );
    TimerQuery inflightTimer = { pTimer, true/* recorded in command list*/};
    std::swap( m_TimerQueries[timerId], inflightTimer );
    assert( inflightTimer.pTimer == nullptr );

    ///TODO: end mutex

    auto queryPoolId = timerId * 2; // 2 queries per timer (start time and stop time)
    vkCmdWriteTimestamp( commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_VulkanQueryPool, queryPoolId );
    return timerId;
}


void TimerPoolBase::StopGpuTimer( VkCommandBuffer commandBuffer, TimerId timerId )
{
    if (timerId == -1)
        return;
    assert( m_TimerQueries[timerId].pTimer != nullptr );

    int queryPoolId = timerId*2 + 1;  // the 'stop' timer query is next to its start
    vkCmdWriteTimestamp( commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_VulkanQueryPool, queryPoolId );
}

void TimerPoolBase::TimerCommandBufferReset(std::span<TimerId> timerIds)
{
    for (int timerId : timerIds)
    {
        assert(m_TimerQueries[timerId].RecordedInCommandList);
        m_TimerQueries[timerId].RecordedInCommandList = false;
    }
}

void TimerPoolBase::ReadResults(VkCommandBuffer commandBuffer, uint32_t whichFrame)
{
    uint32_t maxUsedQueries = m_MaxUsedTimerIndex * 2 + 2;
    if (m_MaxUsedTimerIndex >= 0)
    {
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT/*src stage*/, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT/* dest stage bit*/,
            0, 0, nullptr, 0, nullptr, 0, nullptr);
        // Copy the timing queries that may have been written this frame to a buffer (ready for mapping back to the cpu).
        vkCmdCopyQueryPoolResults(commandBuffer, m_VulkanQueryPool, 0, maxUsedQueries, m_VulkanQueryResults.bufferHandles[whichFrame], 0, sizeof(VulkanQueryResult), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        // Reset the timing queries that may have been written this frame.
        vkCmdResetQueryPool(commandBuffer, m_VulkanQueryPool, 0, maxUsedQueries);
    }
}

void TimerPoolBase::UpdateResults(uint32_t whichFrame)
{
    assert( m_VulkanQueryPool != VK_NULL_HANDLE );
    if (m_MaxUsedTimerIndex < 0)
    {
        assert( m_MaxUsedTimerIndex == -1 );
        return;
    }

    ///TODO: assume we are mutexed here with with commandlist generation

    // Map the 'current' (last completed) timing buffer (written by gpu commands setup by ReadResults) to the CPU.
    auto& gpuBuffer = m_VulkanQueryResults.buf[whichFrame];
    auto mapped = m_Vulkan.GetMemoryManager().Map<VulkanQueryResult>(gpuBuffer);
    if (mapped.data() == nullptr)
    {
        return;
    }

    // Parse results and reset any timers that completed (both the start and stop portions)
    for (uint32_t timerIdx = 0; timerIdx <= (uint32_t/*checked on entry that this is not < 0*/)m_MaxUsedTimerIndex; ++timerIdx)
    {
        const uint32_t resultIdx = timerIdx * 2;
        auto& StartResult = mapped.data()[resultIdx];
        auto& StopResult = mapped.data()[resultIdx + 1];
        if (StartResult.Availability != 0 && StopResult.Availability != 0)
        {
            // Timer fully completed.
            const TimerQuery& inflightTimer = m_TimerQueries[timerIdx];
            if (!inflightTimer.pTimer)
            {
                // Stale result for a slot that was already freed (can occur across mode switches).
                // Clear the availability so this slot is not processed again.
                StartResult.Availability = 0;
                StopResult.Availability  = 0;
                continue;
            }
            inflightTimer.pTimer->Update(whichFrame, StartResult.Timestamp, StopResult.Timestamp);

            StartResult.Availability = 0;
            StopResult.Availability = 0;
        }
    }
    // Cleanup any completed timers that are no longer recorded in any command lists
    for (uint32_t timerIdx = 0; timerIdx <= (uint32_t/*checked on entry that this is not < 0*/)m_MaxUsedTimerIndex; ++timerIdx)
    {
        TimerQuery& inflightTimer = m_TimerQueries[timerIdx];
        if (inflightTimer.RecordedInCommandList == false && inflightTimer.pTimer)
        {
            m_TimerQueries[timerIdx] = {};
            m_FreeQueryPoolEntries.push(timerIdx);
        }
    }

    m_Vulkan.GetMemoryManager().Unmap(gpuBuffer, std::move(mapped));
    ///TODO: end mutex
}

void TimerPoolBase::ResetQueryPool(uint32_t firstResetQuery, uint32_t queryResetCount)
{
    m_vkResetQueryPoolEXT(m_Vulkan.m_VulkanDevice, m_VulkanQueryPool, firstResetQuery, queryResetCount);
}
