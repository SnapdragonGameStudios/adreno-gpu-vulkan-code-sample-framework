// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2020 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

/// @file uniform.hpp
/// @brief Dx12 Uniform buffer containers and support functions.

#include <cstdint>
#include "memoryMapped.hpp"
#include "memory/uniform.hpp"

#include <d3d12.h>

// forward defines
class Dx12;
struct ID3D12Resource;


/// All the variables needed to create a uniform buffer
/// Template specialization of Uniform<> for Dx12
template<>
struct Uniform<Dx12>
{
    Uniform() = default;
    Uniform(const Uniform<Dx12>&) = delete;
    Uniform& operator=(const Uniform<Dx12>&) = delete;
    Uniform(Uniform<Dx12>&&) noexcept = default;
    MemoryAllocatedBuffer<Dx12, ID3D12Resource*> buf;
    //VkDescriptorBufferInfo bufferInfo;
};

/// Uniform buffer array that can be updated every frame without stomping the in-flight uniform buffers.
template<size_t T_NUM_BUFFERS>
struct UniformArray<Dx12, T_NUM_BUFFERS>
{
    UniformArray() = default;
    UniformArray(const UniformArray<Dx12, T_NUM_BUFFERS>&) = delete;
    UniformArray& operator=(const UniformArray<Dx12, T_NUM_BUFFERS>&) = delete;
    UniformArray(UniformArray<Dx12, T_NUM_BUFFERS>&&) noexcept { assert(0); }
    std::array<MemoryAllocatedBuffer<Dx12, ID3D12Resource*>, T_NUM_BUFFERS> buf;
    std::array<ID3D12Resource*, T_NUM_BUFFERS> bufferHandles{};  ///< copy of ID3D12Resource handles (for easy use in bindings etc)
    constexpr size_t size() const { return T_NUM_BUFFERS; }
};

bool CreateUniformBuffer(Dx12*, Uniform<Dx12>* pNewUniform, size_t dataSize, const void* const pData = NULL, BufferUsageFlags usage = BufferUsageFlags::Uniform);
void UpdateUniformBuffer(Dx12*, Uniform<Dx12>* pUniform, size_t dataSize, const void* const pNewData);
void ReleaseUniformBuffer(Dx12*, Uniform<Dx12>* pUniform);

template<typename T>
bool CreateUniformBuffer(Dx12* pGfxApi, UniformT<Dx12, T>& newUniform, const T* const pData = nullptr, BufferUsageFlags usage = BufferUsageFlags::Uniform)
{
    return CreateUniformBuffer(pGfxApi, &newUniform, sizeof(T), pData, usage);
}
template<typename T>
void UpdateUniformBuffer(Dx12* pGfxApi, Uniform<Dx12>& uniform, const T& newData)
{
    return UpdateUniformBuffer(pGfxApi, &uniform, sizeof(T), &newData);
}
template<typename T, typename TT>
void UpdateUniformBuffer(Dx12* pGfxApi, UniformT<Dx12, T>& uniform, const TT& newData)
{
    static_assert(std::is_same<T, TT>::value, "UpdateUniformBuffer, uniform is different type to newData");
    return UpdateUniformBuffer(pGfxApi, static_cast<Uniform<Dx12>&>(uniform), newData);
}

bool CreateUniformBuffer(Dx12* pGfxApi, MemoryAllocatedBuffer<Dx12, ID3D12Resource*>& rNewUniformBuffer, size_t dataSize, const void* const pData, BufferUsageFlags usage);
void UpdateUniformBuffer(Dx12* pGfxApi, MemoryAllocatedBuffer<Dx12, ID3D12Resource*>& rUniform, size_t dataSize, const void* const pNewData);
void ReleaseUniformBuffer(Dx12* pGfxApi, MemoryAllocatedBuffer<Dx12, ID3D12Resource*>& rUniform);

template<size_t T_NUM_BUFFERS>
bool CreateUniformBuffer(Dx12* pGfxApi, UniformArray<Dx12, T_NUM_BUFFERS>& rNewUniform, size_t dataSize, const void* const pData = NULL, BufferUsageFlags usage = BufferUsageFlags::Uniform)
{
    for (size_t i = 0; i < T_NUM_BUFFERS; ++i)
    {
        if (!CreateUniformBuffer(pGfxApi, rNewUniform.buf[i], dataSize, pData, usage))
            return false;
        rNewUniform.bufferHandles[i] = rNewUniform.buf[i].GetResource();
    }
    return true;
}
template<size_t T_NUM_BUFFERS>
void UpdateUniformBuffer(Dx12* pGfxApi, UniformArray<Dx12, T_NUM_BUFFERS>& rUniform, size_t dataSize, const void* const pNewData, uint32_t bufferIdx)
{
    UpdateUniformBuffer(pGfxApi, rUniform.buf[bufferIdx], dataSize, pNewData);
}
template<size_t T_NUM_BUFFERS>
void ReleaseUniformBuffer(Dx12* pGfxApi, UniformArray<Dx12, T_NUM_BUFFERS>& rUniform)
{
    for (size_t i = 0; i < T_NUM_BUFFERS; ++i)
    {
        rUniform.bufferHandles[i] = nullptr;
        ReleaseUniformBuffer(pGfxApi, rUniform.buf[i]);
    }
}

template<typename T, size_t T_NUM_BUFFERS>
bool CreateUniformBuffer(Dx12* pGfxApi, UniformArrayT<Dx12, T, T_NUM_BUFFERS>& newUniform, const T* const pData = nullptr, BufferUsageFlags usage = BufferUsageFlags::Uniform)
{
    return CreateUniformBuffer(pGfxApi, newUniform, sizeof(T), pData, usage);
}
template<typename T, size_t T_NUM_BUFFERS>
void UpdateUniformBuffer(Dx12* pGfxApi, UniformArrayT<Dx12, T, T_NUM_BUFFERS>& uniform, const T& newData, uint32_t bufferIdx)
{
    return UpdateUniformBuffer(pGfxApi, uniform, sizeof(T), &newData, bufferIdx);
}
template<typename T, typename TT, size_t T_NUM_BUFFERS>
void UpdateUniformBuffer(Dx12* pGfxApi, UniformArrayT<Dx12, T, T_NUM_BUFFERS>& uniform, const TT& newData, uint32_t bufferIdx)
{
    static_assert(std::is_same<T, TT>::value, "UpdateUniformBuffer, uniform is different type to newData");
    return UpdateUniformBuffer(pGfxApi, static_cast<UniformArray<Dx12, T_NUM_BUFFERS>&>(uniform), newData, bufferIdx);
}
