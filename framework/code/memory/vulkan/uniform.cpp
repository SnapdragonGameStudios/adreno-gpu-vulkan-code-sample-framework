// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2020 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#include <cstring>
#include <cstdint>
#include "memory/memoryMapped.hpp"
#include "vulkan/vulkan.hpp"
#include "uniform.hpp"


//-----------------------------------------------------------------------------
template<>
bool CreateUniformBuffer<Vulkan>(Vulkan* pVulkan, Uniform<Vulkan>* pNewUniform, size_t dataSize, const void* const pData, BufferUsageFlags usage)
//-----------------------------------------------------------------------------
{
    auto& memoryManager = pVulkan->GetMemoryManager();

    // Create the memory buffer...
    pNewUniform->buf = memoryManager.CreateBuffer(dataSize, usage, MemoryUsage::CpuToGpu, &pNewUniform->bufferInfo);

    // If we have initial data, add it now
    if (pData != nullptr)
    {
        UpdateUniformBuffer(pVulkan, pNewUniform, dataSize, pData);
    }

    return true;
}

//-----------------------------------------------------------------------------
template<>
void UpdateUniformBuffer<Vulkan>(Vulkan* pVulkan, Uniform<Vulkan>* pUniform, size_t dataSize, const void* const pNewData)
//-----------------------------------------------------------------------------
{
    void* pMappedData = NULL;
    auto mapped = pVulkan->GetMemoryManager().Map<uint8_t>(pUniform->buf);
    if (mapped.data() == nullptr)
    {
        return;
    }

    memcpy(mapped.data(), pNewData, dataSize);

    pVulkan->GetMemoryManager().Unmap(pUniform->buf, std::move(mapped));
}


//-----------------------------------------------------------------------------
template<>
void ReleaseUniformBuffer<Vulkan>(Vulkan* pVulkan, Uniform<Vulkan>* pUniform)
//-----------------------------------------------------------------------------
{
    if (pUniform->buf)
    {
        pVulkan->GetMemoryManager().Destroy(std::move(pUniform->buf));
    }
}


//-----------------------------------------------------------------------------
bool CreateUniformBuffer(Vulkan* pVulkan, MemoryAllocatedBuffer<Vulkan, VkBuffer>& rNewUniformBuffer, size_t dataSize, const void* const pData, BufferUsageFlags usage)
//-----------------------------------------------------------------------------
{
    auto& memoryManager = pVulkan->GetMemoryManager();

    // Create the memory buffer...
    rNewUniformBuffer = memoryManager.CreateBuffer(dataSize, usage, MemoryUsage::CpuToGpu, nullptr);

    // If we have initial data, add it now
    if (pData != nullptr)
    {
        UpdateUniformBuffer(pVulkan, rNewUniformBuffer, dataSize, pData);
    }

    return true;
}

//-----------------------------------------------------------------------------
void UpdateUniformBuffer(Vulkan* pVulkan, MemoryAllocatedBuffer<Vulkan, VkBuffer>& rUniform, size_t dataSize, const void* const pNewData)
//-----------------------------------------------------------------------------
{
    auto mapped = pVulkan->GetMemoryManager().Map<uint8_t>(rUniform);
    if (mapped.data() == nullptr)
    {
        return;
    }

    memcpy(mapped.data(), pNewData, dataSize);

    pVulkan->GetMemoryManager().Unmap(rUniform, std::move(mapped));
}

//-----------------------------------------------------------------------------
void ReleaseUniformBuffer(Vulkan* pVulkan, MemoryAllocatedBuffer<Vulkan, VkBuffer>& rUniform)
//-----------------------------------------------------------------------------
{
    pVulkan->GetMemoryManager().Destroy(std::move(rUniform));
}

//-----------------------------------------------------------------------------
Uniform<Vulkan>::Uniform(Uniform<Vulkan>&& other) noexcept
//-----------------------------------------------------------------------------
    : buf(std::move(other.buf))
    , bufferInfo(other.bufferInfo)
{
    other.bufferInfo = {};
}
