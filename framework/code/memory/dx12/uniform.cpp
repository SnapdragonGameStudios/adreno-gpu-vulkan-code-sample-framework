// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2020 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#include <cstdint>
#include "memory/memoryMapped.hpp"
#include "dx12/dx12.hpp"
#include "uniform.hpp"


//-----------------------------------------------------------------------------
bool CreateUniformBuffer(Dx12* pDx12, Uniform<Dx12>* pNewUniform, size_t dataSize, const void* const pData, BufferUsageFlags usage)
//-----------------------------------------------------------------------------
{
    auto& memoryManager = pDx12->GetMemoryManager();

    // Create the memory buffer...
    pNewUniform->buf = memoryManager.CreateBuffer(dataSize, usage, MemoryUsage::CpuToGpu);

    // If we have initial data, add it now
    if (pData != nullptr)
    {
        UpdateUniformBuffer( pDx12, pNewUniform, dataSize, pData);
    }

    return true;
}

//-----------------------------------------------------------------------------
void UpdateUniformBuffer(Dx12* pDx12, Uniform<Dx12>* pUniform, size_t dataSize, const void* const pNewData)
//-----------------------------------------------------------------------------
{
    void* pMappedData = NULL;
    auto mapped = pDx12->GetMemoryManager().Map<uint8_t>(pUniform->buf);
    if (mapped.data() == nullptr)
    {
        return;
    }

    memcpy(mapped.data(), pNewData, dataSize);

    pDx12->GetMemoryManager().Unmap(pUniform->buf, std::move(mapped));
}


//-----------------------------------------------------------------------------
void ReleaseUniformBuffer(Dx12* pDx12, Uniform<Dx12>* pUniform)
//-----------------------------------------------------------------------------
{
    if (pUniform->buf)
    {
        pDx12->GetMemoryManager().Destroy(std::move(pUniform->buf));
    }
}


//-----------------------------------------------------------------------------
bool CreateUniformBuffer(Dx12* pDx12, MemoryAllocatedBuffer<Dx12, ID3D12Resource*>& rNewUniformBuffer, size_t dataSize, const void* const pData, BufferUsageFlags usage)
//-----------------------------------------------------------------------------
{
    auto& memoryManager = pDx12->GetMemoryManager();

    // Create the memory buffer...
    rNewUniformBuffer = memoryManager.CreateBuffer(dataSize, usage, MemoryUsage::CpuToGpu);

    // If we have initial data, add it now
    if (pData != nullptr)
    {
        UpdateUniformBuffer(pDx12, rNewUniformBuffer, dataSize, pData);
    }

    return true;
}

//-----------------------------------------------------------------------------
void UpdateUniformBuffer(Dx12* pDx12, MemoryAllocatedBuffer<Dx12, ID3D12Resource*>& rUniform, size_t dataSize, const void* const pNewData)
//-----------------------------------------------------------------------------
{
    auto mapped = pDx12->GetMemoryManager().Map<uint8_t>(rUniform);
    if (mapped.data() == nullptr)
    {
        return;
    }

    memcpy(mapped.data(), pNewData, dataSize);

    pDx12->GetMemoryManager().Unmap(rUniform, std::move(mapped));
}

//-----------------------------------------------------------------------------
void ReleaseUniformBuffer(Dx12* pDx12, MemoryAllocatedBuffer<Dx12, ID3D12Resource*>& rUniform)
//-----------------------------------------------------------------------------
{
    pDx12->GetMemoryManager().Destroy(std::move(rUniform));
}
