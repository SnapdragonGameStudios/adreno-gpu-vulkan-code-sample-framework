// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include <wrl.h>
#include <d3d12.h>
#include <array>
#include <vector>

#define NUM_SWAPCHAIN_BUFFERS (8)

// Forward declarations
struct ID3D12DescriptorHeap;
class Dx12;


struct DescriptorTableHandle
{
    DescriptorTableHandle(const DescriptorTableHandle&) = default;
    DescriptorTableHandle& operator=(const DescriptorTableHandle&) = default;
    DescriptorTableHandle() noexcept {};
    DescriptorTableHandle(UINT64 _gpu, UINT64 _cpu, uint32_t _numHandles, uint32_t _handleSize) noexcept
        : gpu{.ptr=_gpu}, cpu{.ptr=_cpu}, numHandles(_numHandles), handleSize(_handleSize) {}
    DescriptorTableHandle(DescriptorTableHandle && other) noexcept {
        *this = std::move(other);
    }
    DescriptorTableHandle& operator=(DescriptorTableHandle && other) noexcept {
        if (this != &other)
        {
            gpu = other.gpu;
            other.gpu = {};
            cpu = other.cpu;
            other.cpu = {};
            numHandles = other.numHandles;
            other.numHandles = 0;
            handleSize = other.handleSize;
            other.handleSize = 0;
        }
        return *this;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE GetCpuHandle(size_t index) const { return {cpu.ptr + size_t(index)*handleSize}; }
    D3D12_GPU_DESCRIPTOR_HANDLE GetGpuHandle() const { return gpu; }
    uint32_t numHandles = 0;
    uint32_t handleSize = 0;
private:
    D3D12_GPU_DESCRIPTOR_HANDLE gpu{};
    D3D12_CPU_DESCRIPTOR_HANDLE cpu{};
};


struct DescriptorTableHandleAndRootIndex
{
    uint32_t                rootIndex;
    DescriptorTableHandle   descriptorTable;
};



class DescriptorHeapManager
{
    DescriptorHeapManager(const DescriptorHeapManager&) = delete;
    DescriptorHeapManager& operator=(const DescriptorHeapManager&) = delete;
public:
    DescriptorHeapManager() noexcept {}
    bool Initialize(Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptorHeap, uint32_t maxHandles, uint32_t handleSize);
    DescriptorTableHandle Allocate(uint32_t numHandles);
    DescriptorTableHandle AllocateTemporary(uint32_t frameIndex, uint32_t numHandles);
    void Free(DescriptorTableHandle&&);
    void FreeTemporaries(uint32_t frameIndex);
    ID3D12DescriptorHeap* const GetHeap() const { return m_DescriptorHeap.Get(); }
private:
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_DescriptorHeap;
    uint32_t                        m_HandleSize = 0;
    struct Range { uint32_t start = 0, end = 0; };
    std::vector<Range>              m_FreeDescriptors;
    std::array<std::vector<Range>, NUM_SWAPCHAIN_BUFFERS> m_TemporaryAllocations;
protected:
    void Free(Range);
};
