// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include "graphicsApi/graphicsApiBase.hpp"
#include "vulkan/refHandle.hpp"

// Forward declarations


/// Simple wrapper around VkSemaphore.
/// Simplifies creation (and checks for leaks on destruction - is up to the owner to call Destroy)
/// @ingroup Vulkan
class SemaphoreVulkan final
{
public:
    SemaphoreVulkan() noexcept;
    ~SemaphoreVulkan() noexcept;
    SemaphoreVulkan( VkDevice, VkSemaphore ) noexcept;
    SemaphoreVulkan( SemaphoreVulkan&& src ) noexcept;
    SemaphoreVulkan& operator=( SemaphoreVulkan&& src ) noexcept;
    SemaphoreVulkan Copy() const { return SemaphoreVulkan{*this}; }

    VkSemaphore GetVkSemaphore() const { return m_Semaphore; }
    bool IsEmpty() const { return m_Semaphore == VK_NULL_HANDLE; }

private:
    SemaphoreVulkan( const SemaphoreVulkan& src ) noexcept {
        m_Semaphore = src.m_Semaphore;
    }

    RefHandle<VkSemaphore> m_Semaphore;
};
