// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2020 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "vulkanDebugCallback.hpp"
#include "system/os_common.h"


//-----------------------------------------------------------------------------
VulkanDebugCallback::VulkanDebugCallback(Vulkan& vulkan)
//-----------------------------------------------------------------------------
    : m_Vulkan(vulkan)
{
}

//-----------------------------------------------------------------------------
VulkanDebugCallback::~VulkanDebugCallback()
//-----------------------------------------------------------------------------
{
    if (m_fpDestroyDebugReportCallback)
    {
        m_fpDestroyDebugReportCallback(m_Vulkan.GetVulkanInstance(), m_DebugCallbackHandle, nullptr);
    }
}

//-----------------------------------------------------------------------------
bool VulkanDebugCallback::Init()
//-----------------------------------------------------------------------------
{
#if defined(USES_VULKAN_DEBUG_LAYERS)
    VkResult RetVal = VK_SUCCESS;

    const VkInstance vulkanInstance = m_Vulkan.GetVulkanInstance();

    // ********************************
    // Get Function Pointers
    // ********************************
    PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr(vulkanInstance, "vkGetDeviceProcAddr");
    if (fpGetDeviceProcAddr == nullptr)
    {
        LOGE("Unable to get function pointer from instance: vkGetInstanceProcAddr");
        return false;
    }

    // Get the function pointers needed for debug
    m_fpCreateDebugReportCallback = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(vulkanInstance, "vkCreateDebugReportCallbackEXT");
    if (m_fpCreateDebugReportCallback == nullptr)
    {
        LOGE("Unable to get function pointer from instance: vkCreateDebugReportCallbackEXT");
        return false;
    }

    m_fpDestroyDebugReportCallback = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(vulkanInstance, "vkDestroyDebugReportCallbackEXT");
    if (m_fpDestroyDebugReportCallback == nullptr)
    {
        LOGE("Unable to get function pointer from instance: vkDestroyDebugReportCallbackEXT");
        return false;
    }

    m_fpDebugReportMessage = (PFN_vkDebugReportMessageEXT)vkGetInstanceProcAddr(vulkanInstance, "vkDebugReportMessageEXT");
    if (m_fpDebugReportMessage == nullptr)
    {
        LOGE("Unable to get function pointer from instance: vkDebugReportMessageEXT");
        return false;
    }

    VkDebugReportCallbackCreateInfoEXT CallbackInfo{ VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT };

    VkDebugReportFlagsEXT LocalFlags = 0;
    LocalFlags |= VK_DEBUG_REPORT_INFORMATION_BIT_EXT;
    LocalFlags |= VK_DEBUG_REPORT_WARNING_BIT_EXT;
    LocalFlags |= VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
    LocalFlags |= VK_DEBUG_REPORT_ERROR_BIT_EXT;
    LocalFlags |= VK_DEBUG_REPORT_DEBUG_BIT_EXT;

    CallbackInfo.flags = LocalFlags;
    CallbackInfo.pfnCallback = VulkanDebugCallback::DebugCallbackFn;
    CallbackInfo.pUserData = this;

    RetVal = m_fpCreateDebugReportCallback(vulkanInstance, &CallbackInfo, nullptr, &m_DebugCallbackHandle);
    if (!CheckVkError("m_fpCreateDebugReportCallback()", RetVal))
    {
        return false;
    }
#endif // defined(USES_VULKAN_DEBUG_LAYERS)

    return true;    // when USES_VULKAN_DEBUG_LAYERS is not defined we still return 'success' from the init
}

#if defined(USES_VULKAN_DEBUG_LAYERS)
// Static function entry point
VKAPI_ATTR VkBool32 VKAPI_CALL VulkanDebugCallback::DebugCallbackFn(VkDebugReportFlagsEXT Flags, VkDebugReportObjectTypeEXT ObjectType, uint64_t SrcObj, size_t Location, int32_t Code, const char* pLayerPrefix, const char* pMessage, void* pUserData)
{
    VulkanDebugCallback* pThis = static_cast<VulkanDebugCallback*>(pUserData);
    return pThis->DebugCallback(Flags, ObjectType, SrcObj, Location, Code, pLayerPrefix, pMessage);
}
#endif // defined(USES_VULKAN_DEBUG_LAYERS)

// Class member function
VkBool32 VulkanDebugCallback::DebugCallback(VkDebugReportFlagsEXT Flags, VkDebugReportObjectTypeEXT ObjectType, uint64_t SrcObj, size_t Location, int32_t Code, const char* pLayerPrefix, const char* pMessage)
{
#if defined(USES_VULKAN_DEBUG_LAYERS)
    // Flags
    bool IsError = false;
    char* pFlagString = (char*)"Unknown";
    if (Flags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT)
    {
        pFlagString = (char*)"Info";
        return false;   // Don't want to be spammed with this type of message!
    }
    if (Flags & VK_DEBUG_REPORT_WARNING_BIT_EXT)
    {
        pFlagString = (char*)"Warning";
    }
    if (Flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT)
    {
        pFlagString = (char*)"Perf";
    }
    if (Flags & VK_DEBUG_REPORT_ERROR_BIT_EXT)
    {
        IsError = true;
        pFlagString = (char*)"Error";
    }
    if (Flags & VK_DEBUG_REPORT_DEBUG_BIT_EXT)
    {
        pFlagString = (char*)"Debug";
        return false;   // Don't want to be spammed with this type of message!
    }

    // ObjectType is what caused the error, not where
    char* pObjectName;
    switch (ObjectType)
    {
    case VK_DEBUG_REPORT_OBJECT_TYPE_UNKNOWN_EXT:               pObjectName = (char*)"UNKNOWN";                break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_INSTANCE_EXT:              pObjectName = (char*)"INSTANCE";               break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT:       pObjectName = (char*)"PHYSICAL_DEVICE";        break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_EXT:                pObjectName = (char*)"DEVICE";                 break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_QUEUE_EXT:                 pObjectName = (char*)"QUEUE";                  break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_SEMAPHORE_EXT:             pObjectName = (char*)"SEMAPHORE";              break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT:        pObjectName = (char*)"COMMAND_BUFFER";         break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_FENCE_EXT:                 pObjectName = (char*)"FENCE";                  break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_MEMORY_EXT:         pObjectName = (char*)"DEVICE_MEMORY";          break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT:                pObjectName = (char*)"BUFFER";                 break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT:                 pObjectName = (char*)"IMAGE";                  break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_EVENT_EXT:                 pObjectName = (char*)"EVENT";                  break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_QUERY_POOL_EXT:            pObjectName = (char*)"QUERY_POOL";             break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_VIEW_EXT:           pObjectName = (char*)"BUFFER_VIEW";            break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_VIEW_EXT:            pObjectName = (char*)"IMAGE_VIEW";             break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT:         pObjectName = (char*)"SHADER_MODULE";          break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_CACHE_EXT:        pObjectName = (char*)"PIPELINE_CACHE";         break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_LAYOUT_EXT:       pObjectName = (char*)"PIPELINE_LAYOUT";        break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_RENDER_PASS_EXT:           pObjectName = (char*)"RENDER_PASS";            break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT:              pObjectName = (char*)"PIPELINE";               break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT_EXT: pObjectName = (char*)"DESCRIPTOR_SET_LAYOUT";  break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_EXT:               pObjectName = (char*)"SAMPLER";                break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_POOL_EXT:       pObjectName = (char*)"DESCRIPTOR_POOL";        break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT:        pObjectName = (char*)"DESCRIPTOR_SET";         break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_FRAMEBUFFER_EXT:           pObjectName = (char*)"FRAMEBUFFER";            break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_POOL_EXT:          pObjectName = (char*)"COMMAND_POOL";           break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_SURFACE_KHR_EXT:           pObjectName = (char*)"SURFACE_KHR";            break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_SWAPCHAIN_KHR_EXT:         pObjectName = (char*)"SWAPCHAIN_KHR";          break;
    case VK_DEBUG_REPORT_OBJECT_TYPE_DEBUG_REPORT_EXT:          pObjectName = (char*)"DEBUG_REPORT";           break;
    default:                                                    pObjectName = (char*)"Unknown";                break;

    }

    // SrcObj gives handle to object of type "ObjectType" above.  Can use it to access the object if we want.

    char szBuffer[2048];    // Bumped to 2048 based on Vulkan validation errors
    snprintf(szBuffer, sizeof(szBuffer) - 1, "Vulkan Debug: (%s) %s => %s: %s", pFlagString, pObjectName, pLayerPrefix, pMessage);
    szBuffer[sizeof(szBuffer) - 1] = 0;

    LOGE("********** Validation Message - Start **********");
    if (IsError)
    {
        LOGE("%s", szBuffer);
    }
    else
    {
        LOGI("%s", szBuffer);
    }
    LOGE("********** Validation Message - End **********"); 

    // Return "True" to cause layer to bail out and command is NOT sent to Vulkan
    // Return "False" to send it to the Vulkan layer (behave like final product)
    // Note: returning "false" may still not give correct behaviour for errors from private/unknown extensions (such as QCOM_render_pass_transform)
#endif // defined(USES_VULKAN_DEBUG_LAYERS)
    return false;
}
