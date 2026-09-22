// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include <cassert>
#include "memory/memory.hpp"
#include "memory/memoryManager.hpp"
#include "memory/dx12/memoryMapped.hpp"
#include "material/materialPass.hpp"

// forward declarations
class Dx12;
struct IDXGIAdapter;
struct ID3D12Device;
struct D3D12_CLEAR_VALUE;
struct D3D12_RESOURCE_DESC;
enum D3D12_RESOURCE_STATES;
namespace D3D12MA {
    class Allocator;
    class Allocation;
};


/// Top level API for allocating memory buffers for use by DX12
/// template specialization of MemoryManager.
/// @ingroup Memory
template<>
class MemoryManager<Dx12>
{
    MemoryManager(const MemoryManager<Dx12>&) = delete;
    MemoryManager& operator=(const MemoryManager<Dx12>&) = delete;
public:
    MemoryManager();
    ~MemoryManager();
    using tGfxApi = Dx12;
    using MemoryAllocatedBuffer = MemoryAllocatedBuffer<Dx12, ID3D12Resource*>;

    /// Initialize the memory manager (must be initialized before using CreateBuffer etc)
    /// @param EnableBufferDeviceAddress enable ability to call GetBufferDeviceAddress
    /// @return true if successfully initialized
    bool Initialize(IDXGIAdapter* adapter, ID3D12Device* device);
    /// Destroy the memory manager (do before you destroy the Dx12 device)
    void Destroy();

    /// Create buffer in memory and create the associated Dx12 objects
    MemoryAllocatedBuffer CreateBuffer(size_t size, BufferUsageFlags bufferUsage, MemoryUsage memoryUsage);
    /// Create buffer in memory 
    MemoryAllocatedBuffer CreateImage(const D3D12_RESOURCE_DESC& imageDesc, MemoryUsage memoryUsage, D3D12_RESOURCE_STATES initialResourceState, const D3D12_CLEAR_VALUE& clearValue);

    /// Destruction of created buffer
    void Destroy(MemoryAllocatedBuffer);

    /// Map a buffer to cpu memory
    template<typename T_DATATYPE>
    MemoryCpuMapped<Dx12, T_DATATYPE> Map(MemoryAllocatedBuffer& buffer);

    /// Unmap a buffer from cpu memory
    void Unmap(MemoryAllocatedBuffer& buffer, MemoryCpuMappedUntyped<Dx12> allocation);

protected:
    MemoryCpuMappedUntyped<Dx12> MapInt(MemoryAllocation<Dx12> allocation);
private:
    void MapInternal(D3D12MA::Allocation*, void** outCpuLocation);
    void UnmapInternal(D3D12MA::Allocation*, void* cpuLocation);

private:
    D3D12MA::Allocator* mDxmaAllocator = nullptr;
};

template<typename T_DATATYPE>
MemoryCpuMapped<Dx12, T_DATATYPE> MemoryManager<Dx12>::Map(MemoryAllocatedBuffer& buffer)
{
    return MapInt(std::move(buffer.allocation));
}

inline void MemoryManager<Dx12>::Unmap(MemoryAllocatedBuffer& buffer, MemoryCpuMappedUntyped<Dx12> allocation)
{
    assert(allocation.mCpuLocation);
    UnmapInternal(allocation.mAllocation.allocation, allocation.mCpuLocation);
    allocation.mCpuLocation = nullptr;	// clear ownership
    buffer.allocation = std::move(allocation.mAllocation);
}

inline MemoryCpuMappedUntyped<Dx12> MemoryManager<Dx12>::MapInt(MemoryAllocation<Dx12> allocation)
{
    MemoryCpuMappedUntyped<Dx12> guard(std::move(allocation));
    MapInternal(guard.mAllocation.allocation, &guard.mCpuLocation);
    return guard;
}

