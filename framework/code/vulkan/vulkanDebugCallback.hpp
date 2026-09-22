// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2020 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include "vulkan.hpp"

/// @brief Encapsulate Vulkan debug callback functionality.
class VulkanDebugCallback
{
    VulkanDebugCallback(const VulkanDebugCallback&) = delete;
    VulkanDebugCallback& operator=(const VulkanDebugCallback&) = delete;
public:
    VulkanDebugCallback(Vulkan& vulkan);
    VulkanDebugCallback(VulkanDebugCallback&&);
    VulkanDebugCallback& operator=(VulkanDebugCallback&&);
    virtual ~VulkanDebugCallback();
    bool Init();

private:
    /// Static entry point for the debug callback (called from Vulkan).
    /// Call in to DebugCallback (pUserData must point to 'this' VulkanDebugCallback class)
    static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallbackFn(VkDebugReportFlagsEXT Flags, VkDebugReportObjectTypeEXT ObjectType, uint64_t SrcObj, size_t Location, int32_t Code, const char* pLayerPrefix, const char* pMessage, void* pUserData);
protected:
    /// Member function called by DebugCallbackFn
    /// Can be overridden if we want to handle the messages differently.
    virtual VkBool32 DebugCallback(VkDebugReportFlagsEXT Flags, VkDebugReportObjectTypeEXT ObjectType, uint64_t SrcObj, size_t Location, int32_t Code, const char* pLayerPrefix, const char* pMessage);

private:
    const Vulkan& m_Vulkan;

    PFN_vkCreateDebugReportCallbackEXT              m_fpCreateDebugReportCallback = nullptr;
    PFN_vkDestroyDebugReportCallbackEXT             m_fpDestroyDebugReportCallback = nullptr;
    PFN_vkDebugReportMessageEXT                     m_fpDebugReportMessage = nullptr;
    VkDebugReportCallbackEXT                        m_DebugCallbackHandle = VK_NULL_HANDLE;
};
