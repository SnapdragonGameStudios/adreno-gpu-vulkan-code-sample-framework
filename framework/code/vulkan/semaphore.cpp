// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "semaphore.hpp"

//
// Constructors/move-operators for SemaphoreVulkan (wrapper around vkSemaphore to handle destroy and ref counting).
//
SemaphoreVulkan::SemaphoreVulkan() noexcept
    : m_Semaphore( VK_NULL_HANDLE, VK_NULL_HANDLE )
{}
SemaphoreVulkan::~SemaphoreVulkan() noexcept
{}
SemaphoreVulkan::SemaphoreVulkan( VkDevice device, VkSemaphore semaphore ) noexcept
    : m_Semaphore( device, semaphore )
{}
SemaphoreVulkan::SemaphoreVulkan( SemaphoreVulkan&& src ) noexcept
    : m_Semaphore( std::move( src.m_Semaphore ) )
{}
SemaphoreVulkan& SemaphoreVulkan::operator=( SemaphoreVulkan&& src ) noexcept
{
    if (this != &src)
    {
        m_Semaphore = src.m_Semaphore;
        src.m_Semaphore = {};
    }
    return *this;
}
