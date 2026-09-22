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
/// @brief Vulkan Uniform buffer containers and support functions.

#include <cstdint>
#include "memoryMapped.hpp"
#include "memory/memory.hpp"
#include "memory/uniform.hpp"
#include <volk/volk.h>

// forward defines
class Vulkan;
template<typename T_GFXAPI> struct Uniform;
using UniformVulkan = Uniform<Vulkan>;
template<size_t T_NUM_BUFFERS> using UniformVulkanArray = UniformArray<Vulkan, T_NUM_BUFFERS>;

/// All the variables needed to create a uniform buffer
template<>
struct Uniform<Vulkan>
{
    Uniform() = default;
    Uniform(const Uniform&) = delete;
    Uniform& operator=(const Uniform&) = delete;
    Uniform(Uniform<Vulkan>&&) noexcept;
    MemoryAllocatedBuffer<Vulkan, VkBuffer> buf;
    VkDescriptorBufferInfo bufferInfo;
    bool IsEmpty() const { return !buf; }
};

/// Uniform buffer array that can be updated every frame without stomping the in-flight uniform buffers.
template<size_t T_NUM_BUFFERS>
struct UniformArray<Vulkan, T_NUM_BUFFERS>
{
    UniformArray() = default;
    UniformArray(const UniformArray&) = delete;
    UniformArray& operator=(const UniformArray&) = delete;
    UniformArray(UniformArray&&) noexcept { assert(0); }
    std::array<MemoryAllocatedBuffer<Vulkan, VkBuffer>, T_NUM_BUFFERS> buf;
    std::array<VkBuffer, T_NUM_BUFFERS> bufferHandles{};  ///< copy of VkBuffer handles (for easy use in bindings etc)
    constexpr size_t size() const { return T_NUM_BUFFERS; }
    bool IsEmpty() const { return !buf[0]; }
};


template<>
bool CreateUniformBuffer(Vulkan* pVulkan, Uniform<Vulkan>* pNewUniform, size_t dataSize, const void* const pData, BufferUsageFlags usage);
template<>
void UpdateUniformBuffer(Vulkan* pVulkan, Uniform<Vulkan>* pUniform, size_t dataSize, const void* const pNewData);
template<>
void ReleaseUniformBuffer(Vulkan* pVulkan, Uniform<Vulkan>* pUniform);

template<typename T>
bool CreateUniformBuffer(Vulkan* pVulkan, UniformT<Vulkan, T>& newUniform, const T* const pData = nullptr, BufferUsageFlags usage = BufferUsageFlags::Uniform)
{
    return CreateUniformBuffer(pVulkan, &newUniform, sizeof(T), pData, usage);
}
template<typename T>
void UpdateUniformBuffer(Vulkan* pVulkan, Uniform<Vulkan>& uniform, const T& newData)
{
    return UpdateUniformBuffer(pVulkan, &uniform, sizeof(T), &newData);
}
template<typename T, typename TT>
void UpdateUniformBuffer(Vulkan* pVulkan, UniformT<Vulkan, T>& uniform, const TT& newData)
{
    static_assert(std::is_same<T, TT>::value, "UpdateUniformBuffer, uniform is different type to newData");
    return UpdateUniformBuffer(pVulkan, static_cast<Uniform<Vulkan>&>(uniform), newData);
}

bool CreateUniformBuffer(Vulkan* pVulkan, MemoryAllocatedBuffer<Vulkan, VkBuffer>& rNewUniformBuffer, size_t dataSize, const void* const pData, BufferUsageFlags usage);
void UpdateUniformBuffer(Vulkan* pVulkan, MemoryAllocatedBuffer<Vulkan, VkBuffer>& rUniform, size_t dataSize, const void* const pNewData);
void ReleaseUniformBuffer(Vulkan* pVulkan, MemoryAllocatedBuffer<Vulkan, VkBuffer>& rUniform);

template<size_t T_NUM_BUFFERS>
bool CreateUniformBuffer(Vulkan* pVulkan, UniformArray<Vulkan, T_NUM_BUFFERS>& rNewUniform, size_t dataSize, const void* const pData = NULL, BufferUsageFlags usage = BufferUsageFlags::Uniform)
{
    for (size_t i = 0; i < T_NUM_BUFFERS; ++i)
    {
        if (!CreateUniformBuffer(pVulkan, rNewUniform.buf[i], dataSize, pData, usage))
            return false;
        rNewUniform.bufferHandles[i] = rNewUniform.buf[i].GetVkBuffer();
    }
    return true;
}
template<size_t T_NUM_BUFFERS>
void UpdateUniformBuffer(Vulkan* pVulkan, UniformArray<Vulkan, T_NUM_BUFFERS>& rUniform, size_t dataSize, const void* const pNewData, uint32_t bufferIdx)
{
    UpdateUniformBuffer(pVulkan, rUniform.buf[bufferIdx], dataSize, pNewData);
}
template<size_t T_NUM_BUFFERS>
void ReleaseUniformBuffer(Vulkan* pVulkan, UniformArray<Vulkan, T_NUM_BUFFERS>& rUniform)
{
    for (size_t i = 0; i < T_NUM_BUFFERS; ++i)
    {
        rUniform.bufferHandles[i] = VK_NULL_HANDLE;
        ReleaseUniformBuffer(pVulkan, rUniform.buf[i]);
    }
}

template<typename T, size_t T_NUM_BUFFERS>
bool CreateUniformBuffer(Vulkan* pVulkan, UniformArrayT<Vulkan, T, T_NUM_BUFFERS>& newUniform, const T* const pData = nullptr, BufferUsageFlags usage = BufferUsageFlags::Uniform)
{
    return CreateUniformBuffer(pVulkan, newUniform, sizeof(T), pData, usage);
}
template<typename T, size_t T_NUM_BUFFERS>
void UpdateUniformBuffer(Vulkan* pVulkan, UniformArrayT<Vulkan, T, T_NUM_BUFFERS>& uniform, const T& newData, uint32_t bufferIdx)
{
    return UpdateUniformBuffer(pVulkan, uniform, sizeof(T), &newData, bufferIdx);
}
template<typename T, typename TT, size_t T_NUM_BUFFERS>
void UpdateUniformBuffer(Vulkan* pVulkan, UniformArrayT<Vulkan, T, T_NUM_BUFFERS>& uniform, const TT& newData, uint32_t bufferIdx)
{
    static_assert(std::is_same<T, TT>::value, "UpdateUniformBuffer, uniform is different type to newData");
    return UpdateUniformBuffer(pVulkan, static_cast<UniformArray<Vulkan, T_NUM_BUFFERS>&>(uniform), newData, bufferIdx);
}
template<typename T, size_t T_NUM_BUFFERS>
void ReleaseUniformBuffer(Vulkan* pVulkan, UniformArrayT<Vulkan, T, T_NUM_BUFFERS>& uniform)
{
    ReleaseUniformBuffer(pVulkan, static_cast<UniformArray<Vulkan, T_NUM_BUFFERS>&>(uniform));
}
