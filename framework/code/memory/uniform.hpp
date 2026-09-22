// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

/// @file uniform.hpp
/// @brief Vulkan Uniform buffer containers and support functions.
/// Graphics API agnostic templates (expected to be specialized by Vulkan / Dx12 etc)

#include <cstdint>
#include "memory/memory.hpp"
#include "memory/memoryMapped.hpp"

// forward defines


template<typename T_GFXAPI>
struct Uniform
{
    Uniform(const Uniform&) = delete;
    Uniform& operator=(const Uniform&) = delete;
    Uniform() = delete;                     // template class expected to be specialized
    Uniform(Uniform&&) noexcept = delete;   // template class expected to be specialized

    static_assert(sizeof(Uniform<T_GFXAPI>) > 1);   // Ensure this class template is specialized (and not used as-is)
};

template<typename T_GFXAPI, typename T>
struct UniformT : public Uniform<T_GFXAPI>
{
    typedef T type;
};

/// Uniform buffer array that can be updated every frame without stomping the in-flight uniform buffers.
template<typename T_GFXAPI, size_t T_NUM_BUFFERS>
struct UniformArray
{
    UniformArray() = delete;                                                    // template class expected to be specialized
    UniformArray(const UniformArray<T_GFXAPI, T_NUM_BUFFERS> &) = delete;
    UniformArray& operator=(const UniformArray<T_GFXAPI, T_NUM_BUFFERS>&) = delete;
    UniformArray(UniformArray<T_GFXAPI, T_NUM_BUFFERS>&&) noexcept = delete;   // template class expected to be specialized

    static_assert(sizeof( UniformArray<T_GFXAPI, T_NUM_BUFFERS> ) > 1);   // Ensure this class template is specialized (and not used as-is)
};

/// @brief UniformArray buffer helper template
/// @tparam T struct type that is contained in the Uniform
template<typename T_GFXAPI, typename T, size_t T_NUM_BUFFERS>
struct UniformArrayT : public UniformArray<T_GFXAPI, T_NUM_BUFFERS>
{
    typedef T type;
};


template<typename T_GFXAPI>
bool CreateUniformBuffer( T_GFXAPI*, Uniform<T_GFXAPI>* pNewUniform, size_t dataSize, const void* const pData = NULL, BufferUsageFlags usage = BufferUsageFlags::Uniform )
{
    static_assert(sizeof( T_GFXAPI ) != sizeof( T_GFXAPI ), "Must use the specialized version of this function.  Your are likely missing #include \"memory/<GFXAPI>/uniform.hpp\"");
}
template<typename T_GFXAPI>
void UpdateUniformBuffer(T_GFXAPI* , Uniform<T_GFXAPI>* pUniform, size_t dataSize, const void* const pNewData)
{
    static_assert(sizeof( T_GFXAPI ) != sizeof( T_GFXAPI ), "Must use the specialized version of this function.  Your are likely missing #include \"memory/<GFXAPI>/uniform.hpp\"");
}
template<typename T_GFXAPI>
void ReleaseUniformBuffer(T_GFXAPI* , Uniform<T_GFXAPI>* pUniform)
{
    static_assert(sizeof( T_GFXAPI ) != sizeof( T_GFXAPI ), "Must use the specialized version of this function.  Your are likely missing #include \"memory/<GFXAPI>/uniform.hpp\"");
}

template<typename T_GFXAPI, typename T>
bool CreateUniformBuffer(T_GFXAPI* pGfxApi, UniformT<T_GFXAPI, T>& newUniform, const T* const pData = nullptr, BufferUsageFlags usage = BufferUsageFlags::Uniform)
{
    return CreateUniformBuffer(pGfxApi, &newUniform, sizeof(T), pData, usage);
}
template<typename T_GFXAPI, typename T>
void UpdateUniformBuffer(T_GFXAPI* pGfxApi, Uniform<T_GFXAPI>& uniform, const T& newData)
{
    return UpdateUniformBuffer(pGfxApi, &uniform, sizeof(T), &newData);
}
template<typename T_GFXAPI, typename T, typename TT>
void UpdateUniformBuffer(T_GFXAPI* pGfxApi, UniformT<T_GFXAPI, T>& uniform, const TT& newData)
{
    static_assert(std::is_same<T, TT>::value, "UpdateUniformBuffer, uniform is different type to newData");
    return UpdateUniformBuffer(pGfxApi, static_cast<Uniform<T_GFXAPI>&>(uniform), newData);
}

template<typename T_GFXAPI, size_t T_NUM_BUFFERS>
bool CreateUniformBuffer(T_GFXAPI* pGfxApi, UniformArray<T_GFXAPI, T_NUM_BUFFERS>& rNewUniform, size_t dataSize, const void* const pData = NULL, BufferUsageFlags usage = BufferUsageFlags::Uniform)
{
    for (size_t i = 0; i < T_NUM_BUFFERS; ++i)
    {
        if (!CreateUniformBuffer(pGfxApi, rNewUniform.buf[i], dataSize, pData, usage))
            return false;
        //rNewUniform.vkBuffers[i] = rNewUniform.buf[i].GetVkBuffer();
    }
    return true;
}
template<typename T_GFXAPI, size_t T_NUM_BUFFERS>
void UpdateUniformBuffer(T_GFXAPI* pGfxApi, UniformArray<T_GFXAPI, T_NUM_BUFFERS>& rUniform, size_t dataSize, const void* const pNewData, uint32_t bufferIdx)
{
    UpdateUniformBuffer(pGfxApi, rUniform.buf[bufferIdx], dataSize, pNewData);
}
template<typename T_GFXAPI, size_t T_NUM_BUFFERS>
void ReleaseUniformBuffer(T_GFXAPI* pGfxApi, UniformArray<T_GFXAPI, T_NUM_BUFFERS>& rUniform)
{
    for (size_t i = 0; i < T_NUM_BUFFERS; ++i)
    {
        rUniform.bufferHandles[i] = {};
        ReleaseUniformBuffer(pGfxApi, rUniform.buf[i]);
    }
}

template<typename T_GFXAPI, typename T, size_t T_NUM_BUFFERS>
bool CreateUniformBuffer(T_GFXAPI* pGfxApi, UniformArrayT<T_GFXAPI, T, T_NUM_BUFFERS>& newUniform, const T* const pData = nullptr, BufferUsageFlags usage = BufferUsageFlags::Uniform)
{
    return CreateUniformBuffer(pGfxApi, newUniform, sizeof(T), pData, usage);
}
template<typename T_GFXAPI, typename T, size_t T_NUM_BUFFERS>
void UpdateUniformBuffer(T_GFXAPI* pGfxApi, UniformArrayT<T_GFXAPI, T, T_NUM_BUFFERS>& uniform, const T& newData, uint32_t bufferIdx)
{
    return UpdateUniformBuffer(pGfxApi, uniform, sizeof(T), &newData, bufferIdx);
}
template<typename T_GFXAPI, typename T, typename TT, size_t T_NUM_BUFFERS>
void UpdateUniformBuffer(T_GFXAPI* pGfxApi, UniformArrayT<T_GFXAPI, T, T_NUM_BUFFERS>& uniform, const TT& newData, uint32_t bufferIdx)
{
    static_assert(std::is_same<T, TT>::value, "UpdateUniformBuffer, uniform is different type to newData");
    return UpdateUniformBuffer(pGfxApi, static_cast<UniformArray<T_GFXAPI, T_NUM_BUFFERS>&>(uniform), newData, bufferIdx);
}
