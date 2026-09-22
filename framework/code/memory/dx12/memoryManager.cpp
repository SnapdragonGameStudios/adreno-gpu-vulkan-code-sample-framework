// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "memoryManager.hpp"
#include "memoryMapped.hpp"
#include "dx12/dx12.hpp"
#include "D3D12MemoryAllocator/include/D3D12MemAlloc.h"  // we should be the only file to include this header!
#include <cassert>


MemoryManager<Dx12>::MemoryManager()
{
}

///////////////////////////////////////////////////////////////////////////////

MemoryManager<Dx12>::~MemoryManager()
{
    Destroy();
}

///////////////////////////////////////////////////////////////////////////////

bool MemoryManager<Dx12>::Initialize(IDXGIAdapter* adapter, ID3D12Device* device)
{
    assert(mDxmaAllocator == nullptr);
    D3D12MA::ALLOCATOR_DESC allocatorDesc = {};
    allocatorDesc.pDevice = device;
    allocatorDesc.pAdapter = adapter;

    HRESULT hr = D3D12MA::CreateAllocator(&allocatorDesc, &mDxmaAllocator);
    if (hr != S_OK)
    {
        mDxmaAllocator = nullptr;
        return false;
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////

void MemoryManager<Dx12>::Destroy()
{
    auto* const tmp = mDxmaAllocator;
    mDxmaAllocator = nullptr;
    if (tmp!=nullptr)
        tmp->Release();
}

///////////////////////////////////////////////////////////////////////////////

MemoryManager<Dx12>::MemoryAllocatedBuffer MemoryManager<Dx12>::CreateBuffer(size_t size, BufferUsageFlags bufferUsage, MemoryUsage memoryUsage)
{
    assert(memoryUsage != MemoryUsage::Unknown);
    assert(bufferUsage != BufferUsageFlags::Unknown);

    auto heapType = D3D12_HEAP_TYPE_DEFAULT;
    switch (memoryUsage) {
    case MemoryUsage::CpuToGpu:
        heapType = D3D12_HEAP_TYPE_UPLOAD;
        break;
    case MemoryUsage::GpuToCpu:
        heapType = D3D12_HEAP_TYPE_READBACK;
        break;
    case MemoryUsage::GpuExclusive:
        heapType = D3D12_HEAP_TYPE_DEFAULT;
        break;
    default:
        assert(0);
        break;
    }
    if ((bufferUsage&BufferUsageFlags::Uniform) != 0)
        size = (size + D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1) & ~(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT-1);

    const D3D12_RESOURCE_DESC resourceDesc = {  .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
                                                .Alignment = 0,
                                                .Width = size,
                                                .Height = 1,
                                                .DepthOrArraySize = 1,
                                                .MipLevels = 1,
                                                .Format = DXGI_FORMAT_UNKNOWN,
                                                .SampleDesc = {.Count = 1, .Quality = 0 },
                                                .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                                                .Flags = D3D12_RESOURCE_FLAG_NONE };
    D3D12MA::ALLOCATION_DESC allocationDesc = {.HeapType = heapType};
    MemoryAllocatedBuffer allocatedBuffer;
    HRESULT hr = mDxmaAllocator->CreateResource(&allocationDesc,
                                                &resourceDesc,
                                                D3D12_RESOURCE_STATE_GENERIC_READ,
                                                NULL,
                                                &allocatedBuffer.allocation.allocation,
                                                IID_PPV_ARGS(&allocatedBuffer.resource));
    if (hr != S_OK)
        return {};

    return allocatedBuffer;
}

///////////////////////////////////////////////////////////////////////////////

MemoryManager<Dx12>::MemoryAllocatedBuffer MemoryManager<Dx12>::CreateImage(const D3D12_RESOURCE_DESC& imageDesc, MemoryUsage memoryUsage, D3D12_RESOURCE_STATES initialResourceState, const D3D12_CLEAR_VALUE& clearValue)
{
    assert(memoryUsage != MemoryUsage::Unknown);

    auto heapType = D3D12_HEAP_TYPE_DEFAULT;
    switch (memoryUsage) {
    case MemoryUsage::CpuToGpu:
        heapType = D3D12_HEAP_TYPE_UPLOAD;
        break;
    case MemoryUsage::GpuToCpu:
        heapType = D3D12_HEAP_TYPE_READBACK;
        break;
    case MemoryUsage::GpuExclusive:
        heapType = D3D12_HEAP_TYPE_DEFAULT;
        break;
    default:
        assert(0);
        break;
    }

    D3D12MA::ALLOCATION_DESC allocationDesc = { .HeapType = heapType };
    MemoryAllocatedBuffer allocatedBuffer;
    HRESULT hr = mDxmaAllocator->CreateResource(&allocationDesc,
                                                &imageDesc,
                                                initialResourceState,
                                                clearValue.Format != DXGI_FORMAT_UNKNOWN ? &clearValue : nullptr,
                                                &allocatedBuffer.allocation.allocation,
                                                IID_PPV_ARGS(&allocatedBuffer.resource));
    if (hr != S_OK)
        return {};

    return allocatedBuffer;
}

///////////////////////////////////////////////////////////////////////////////

void MemoryManager<Dx12>::Destroy(MemoryAllocatedBuffer allocatedBuffer)
{
    if (allocatedBuffer)
    {
        allocatedBuffer.resource->Release();
        allocatedBuffer.resource = nullptr;
        allocatedBuffer.allocation.allocation->Release();
        allocatedBuffer.allocation.allocation = nullptr;
    }
}

///////////////////////////////////////////////////////////////////////////////

void MemoryManager<Dx12>::MapInternal(D3D12MA::Allocation* allocation, void** outCpuLocation)
{
    assert(outCpuLocation != nullptr);
    assert(*outCpuLocation == nullptr);	// mapped twice?
    assert(allocation != nullptr);	// Likely caused by a double mapping of this data
    if (allocation->GetResource()->Map(0, nullptr, outCpuLocation) != S_OK)
    {
        outCpuLocation = nullptr;
        assert(0);
    }
}

///////////////////////////////////////////////////////////////////////////////

void MemoryManager<Dx12>::UnmapInternal(D3D12MA::Allocation* allocation, void* cpuLocation)
{
    assert(cpuLocation);
    allocation->GetResource()->Unmap(0, nullptr);
}
