// SPDX-License-Identifier: BSD-3-Clause

//============================================================================================================
//
//                  Copyright (c) QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//============================================================================================================
#pragma once

#include <atomic>
#include <cassert>
#include <volk/volk.h>
#include "refHandleDestroyFuncs.hpp"

///
/// Reference counted handle to Vulkan object.
/// Use for sharing VkHandle across objects (and allowing this container to do destruction).
/// There is no DX12 equivalent because their objects are already ref counted with ComPtr.
///
/// CANNOT be used by multiple threads
///

template<typename T>
class RefHandle
{
public:
    RefHandle() noexcept
    {}

    RefHandle(VkDevice _device, T _handle) noexcept
        : handle(_handle)
        , device(_device)
    {
        if (handle)
            m_shared_ref_count = new std::atomic<uint32_t>(1);
    }

    RefHandle(const RefHandle& other) noexcept
        : handle(other.handle)
        , device(other.device)
        , m_shared_ref_count(other.m_shared_ref_count)
    {
        if (m_shared_ref_count)
        {
            auto old_ref_count = m_shared_ref_count->fetch_add(1);
            assert(old_ref_count > 0);
        }
    }

    RefHandle& operator=(const RefHandle& other) noexcept
    {
        if (this != &other)
        {
            free();
            if (other.m_shared_ref_count)
            {
                auto old_ref_count = other.m_shared_ref_count->fetch_add(1);
                if (old_ref_count > 0)
                {
                    handle = other.handle;
                    device = other.device;
                    m_shared_ref_count = other.m_shared_ref_count;
                }
            }
        }
        return *this;
    }

    ~RefHandle() noexcept
    {
        free();
    }

    RefHandle(RefHandle&& other) noexcept
        : handle(other.handle)
        , device(other.device)
        , m_shared_ref_count(other.m_shared_ref_count)
    {
        other.handle = VK_NULL_HANDLE;
        other.device = VK_NULL_HANDLE;
        other.m_shared_ref_count = nullptr;
    }

    RefHandle& operator=(RefHandle&& other) noexcept
    {
        if (&other != this)
        {
            free();
            std::swap(handle, other.handle);
            std::swap(device, other.device);
            std::swap(m_shared_ref_count, other.m_shared_ref_count);
        }
        return *this;
    }

    operator T() const { return get(); }
    T get() const { return handle; }
    VkDevice getDevice() const { return device; }

    T handle = {};

private:
    VkDevice device = VK_NULL_HANDLE;

    void free()
    {
        if (m_shared_ref_count)
        {
            auto old_count = m_shared_ref_count->fetch_add(-1);
            m_shared_ref_count = nullptr;
            if (old_count == 1)
            {
                VulkanTraits::DestroyFn<T>::Call(device, handle, nullptr);
            }
            handle = VK_NULL_HANDLE;
            device = VK_NULL_HANDLE;
        }
        else
        {
            assert(handle == VK_NULL_HANDLE);
        }
    }

    std::atomic<uint32_t>* m_shared_ref_count = nullptr;

};
