// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include "descriptorHeapManager.hpp"
#include "dx12.hpp"
#include <algorithm>

// Forward declarations
using namespace Microsoft::WRL; // for ComPtr


//-----------------------------------------------------------------------------
bool DescriptorHeapManager::Initialize(ComPtr<ID3D12DescriptorHeap> descriptorHeap, uint32_t maxHandles, uint32_t handleSize)
//-----------------------------------------------------------------------------
{
    m_DescriptorHeap = std::move(descriptorHeap);
    m_HandleSize = handleSize;
    m_FreeDescriptors.clear();
    m_FreeDescriptors.push_back({ 0,maxHandles });
    m_TemporaryAllocations.fill({});
    return true;
}

//-----------------------------------------------------------------------------
DescriptorTableHandle DescriptorHeapManager::Allocate(uint32_t numHandles)
//-----------------------------------------------------------------------------
{
    auto foundIt = std::find_if(m_FreeDescriptors.begin(), m_FreeDescriptors.end(), [numHandles](const auto& freeRange) -> bool {
        uint32_t freeSize = freeRange.end - freeRange.start;
        return freeSize >= numHandles;
    });
    if (foundIt == m_FreeDescriptors.end())
        return {};  // out of space!

    const uint32_t handleStartIdx = foundIt->start;
    foundIt->start += numHandles;
    if (foundIt->start == foundIt->end)
        // Block used completely
        m_FreeDescriptors.erase(foundIt);

    return DescriptorTableHandle{ m_DescriptorHeap->GetGPUDescriptorHandleForHeapStart().ptr + UINT64(handleStartIdx) * m_HandleSize,
                                  m_DescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + UINT64(handleStartIdx) * m_HandleSize,
                                  numHandles,
                                  m_HandleSize};
}

//-----------------------------------------------------------------------------
DescriptorTableHandle DescriptorHeapManager::AllocateTemporary(uint32_t frameIndex, uint32_t numHandles)
//-----------------------------------------------------------------------------
{
    // Search/allocate in reverse (opposite direction from Allocate, so temporaries are at the top of the descriptor heap)
    auto foundIt = std::find_if(m_FreeDescriptors.rbegin(), m_FreeDescriptors.rend(), [numHandles](const auto& freeRange) -> bool {
        uint32_t freeSize = freeRange.end - freeRange.start;
        return freeSize >= numHandles;
    });
    if (foundIt == m_FreeDescriptors.rend())
        return {};  // out of space!

    const uint32_t handleStartIdx = foundIt->end - numHandles;
    foundIt->end -= numHandles;
    if (foundIt->start == foundIt->end)
        // Block used completely
        m_FreeDescriptors.erase((++foundIt).base());//convert reverse iterator to forward and erase from free

    DescriptorTableHandle handle{ m_DescriptorHeap->GetGPUDescriptorHandleForHeapStart().ptr + UINT64(handleStartIdx) * m_HandleSize,
                                  m_DescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T(handleStartIdx) * m_HandleSize,
                                  numHandles,
                                  m_HandleSize};
    m_TemporaryAllocations[frameIndex].push_back({ handleStartIdx, handleStartIdx + numHandles });

    return handle;
}

//-----------------------------------------------------------------------------
void DescriptorHeapManager::Free(DescriptorTableHandle&& handle)
//-----------------------------------------------------------------------------
{
    assert(handle.numHandles > 0);
    const uint32_t handleStartIdx = (handle.GetCpuHandle(0).ptr - m_DescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr) / m_HandleSize;
    const uint32_t handleEndIdx = handleStartIdx + handle.numHandles;
}

//-----------------------------------------------------------------------------
void DescriptorHeapManager::Free(Range range)
//-----------------------------------------------------------------------------
{
    const uint32_t startIdx = range.start;
    const uint32_t endIdx = range.end;
    assert(startIdx < endIdx);

    auto nextFreeIt = std::find_if(m_FreeDescriptors.begin(), m_FreeDescriptors.end(), [startIdx](const auto& freeRange) -> bool {
        return (startIdx < freeRange.start);
    });
    if (nextFreeIt == m_FreeDescriptors.end())
    {
        assert(m_FreeDescriptors.empty() || m_FreeDescriptors.back().end <= startIdx);
        if (!m_FreeDescriptors.empty() && m_FreeDescriptors.back().end == startIdx)
            // add to the end of the final free block
            m_FreeDescriptors.back().end = endIdx;
        else
            // add a new 'final' free block
            m_FreeDescriptors.push_back({ startIdx, endIdx });
    }
    else
    {
        assert(nextFreeIt->start <= endIdx);
        if (endIdx == nextFreeIt->start)
            // Extend the next free block to include this newly freed block
            nextFreeIt->start = startIdx;
        else
        {
            if (nextFreeIt != m_FreeDescriptors.begin() && (nextFreeIt - 1)->end == startIdx)
            {
                if (nextFreeIt->start == endIdx)
                {
                    // Combine previous free block with the next free block (we just plugged the gap)
                    (nextFreeIt - 1)->end = nextFreeIt->end;
                    m_FreeDescriptors.erase(nextFreeIt);
                }
                else
                    // Just extend the previous free block
                    (nextFreeIt - 1)->end = endIdx;
            }
            else
                // Make a new free block in the middle of the allocations (add a hole)
                m_FreeDescriptors.insert(nextFreeIt, { startIdx, endIdx });
        }
    }
}

//-----------------------------------------------------------------------------
void DescriptorHeapManager::FreeTemporaries(uint32_t frameIndex)
//-----------------------------------------------------------------------------
{
    for (const auto& alloc : m_TemporaryAllocations[frameIndex])
    {
        Free(alloc);
    }
    m_TemporaryAllocations[frameIndex].clear();
}
