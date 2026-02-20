//============================================================================================================
//
//
//                  Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "vulkanDebugCallback.hpp"
#include "vulkan.hpp"
#include "extensionLib.hpp"
#include "system/os_common.h"
#include "system/config.h"
#include "texture/vulkan/texture.hpp"
#include "vulkan/renderContext.hpp"
#include "vulkan/renderPass.hpp"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <map>
#include <memory>
#if defined(OS_LINUX)
#include <GLFW/glfw3.h>
#endif // OS_LINUX

#define VOLK_IMPLEMENTATION
#include <volk/volk.h>

#if defined(OS_ANDROID)
#if __ANDROID_API__ < 29
PFN_vkGetAndroidHardwareBufferPropertiesANDROID fpGetAndroidHardwareBufferPropertiesANDROID = nullptr;
PFN_vkGetImageMemoryRequirements2               fpGetImageMemoryRequirements2 = nullptr;
#endif // __ANDROID_API__ < 29
#endif // OS_ANDROID

#if defined(USES_VULKAN_DEBUG_LAYERS)
PFN_vkCreateDebugReportCallbackEXT              fpCreateDebugReportCallback = nullptr;
PFN_vkDestroyDebugReportCallbackEXT             fpDestroyDebugReportCallback = nullptr;
PFN_vkDebugReportMessageEXT                     fpDebugReportMessage = nullptr;

VkDebugReportCallbackEXT                        gDebugCallbackHandle = VK_NULL_HANDLE;
#endif // USES_VULKAN_DEBUG_LAYERS

// Global access to GetDeviceProcAddr
PFN_vkGetDeviceProcAddr                         fpGetDeviceProcAddr = nullptr;

// Validation enables (from config)
VAR(bool, gEnableValidation, true, kVariableNonpersistent);
VAR(bool, gEnableValidationBestPractices, false, kVariableNonpersistent);
VAR(bool, gEnableValidationGpu, false, kVariableNonpersistent);
VAR(bool, gEnableValidationDebugPrintf, false, kVariableNonpersistent);
VAR(bool, gEnableValidationSynchronization, false, kVariableNonpersistent);

// Override the physical device number
VAR(int, gPhysicalDevice, -1, kVariableNonpersistent);

// Forward declaration of helper functions
bool CheckVkError(const char* pPrefix, VkResult CheckVal);

namespace
{
    bool GetMemoryType(
        VkPhysicalDeviceMemoryProperties& memory_properties,
        uint32_t typeBits,
        VkFlags requirements_mask,
        uint32_t& typeIndex)
    {
        for (uint32_t i = 0; i < 32; i++)
        {
            if ((typeBits & 1) == 1)
            {
                if ((memory_properties.memoryTypes[i].propertyFlags &
                    requirements_mask) == requirements_mask)
                {
                    typeIndex = i;
                    return true;
                }
            }

            typeBits >>= 1;
        }

        return false;
    }
}

//-----------------------------------------------------------------------------
Vulkan::Vulkan()
//-----------------------------------------------------------------------------
{
#if defined(OS_WINDOWS)
    m_hInstance = 0;
    m_hWnd = 0;
#endif // defined(OS_WINDOWS)

    // Surface dimensions should be set by the platform specific code (eg the window size in Windows, screen size from Android).  Set some defaults.
    m_SurfaceWidth = 1280;
    m_SurfaceHeight = 720;

    m_SwapchainPreTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    m_UseRenderPassTransform = false; // default to not bothering the app with this (but potentially a surfaceflinger performance hit and may be outputting hdr surfaces through a 8bit composition buffer)
    m_UsePreTransform = false;  // default to letting the device (or m_UseRenderPassTransform) handle any rotation.  See m_UseRenderPassTransform for performance/hdr notes.

    m_SwapchainImageCount = 0;
    m_VulkanSwapchain = VK_NULL_HANDLE;
    m_SwapchainCurrentIndx = 0;
    // m_SwapchainDepth;

    // Vulkan Objects
    m_VulkanInstance = VK_NULL_HANDLE;
    m_VulkanApiVersion = 0;
    m_VulkanGpuCount = 0;
    m_VulkanGpuIdx = 0;

    m_pDeviceLayerNames.clear();
    
    m_LayerKhronosValidationAvailable = false;
    m_ExtGlobalPriorityAvailable = false;
    m_ExtRenderPassTransformAvailable = false;
    m_ExtRenderPassShaderResolveAvailable = false;
    m_ExtRenderPassTransformLegacy = false;
    m_ExtRenderPassTransformEnabled = false;
    m_ExtPortability = false;
#if defined (OS_ANDROID)
    m_ExtAndroidExternalMemoryAvailable = false;
#endif // defined (OS_ANDROID)

    m_VulkanGraphicsQueueSupportsCompute = false;
    m_VulkanGraphicsQueueSupportsDataGraph = false;

    m_VulkanDevice = VK_NULL_HANDLE;

    m_VulkanSurfaceCaps = {};

    m_SetupCmdBuffer = VK_NULL_HANDLE;

    m_PipelineCache = VK_NULL_HANDLE;
}

//-----------------------------------------------------------------------------
Vulkan::~Vulkan()
//-----------------------------------------------------------------------------
{
    DestroyFrameBuffers();
    DestroySwapchainRenderPass();
    DestroySwapChain();

    m_MemoryManager.Destroy();

    for (auto& queue : m_VulkanQueues)
    {
        if (queue.CommandPool != VK_NULL_HANDLE)
            vkDestroyCommandPool(m_VulkanDevice, queue.CommandPool, nullptr);
    }

    if (m_PipelineCache != VK_NULL_HANDLE)
    {
        vkDestroyPipelineCache(m_VulkanDevice, m_PipelineCache, nullptr);
    }

    //DestroySyncElements();
    vkDestroySemaphore(m_VulkanDevice, m_RenderCompleteSemaphore, nullptr);

    // Destroy Device
    vkDestroyDevice( m_VulkanDevice, nullptr );
    m_VulkanDevice = VK_NULL_HANDLE;

    vkDestroySurfaceKHR( m_VulkanInstance, m_VulkanSurface, nullptr );

#if defined(USES_VULKAN_DEBUG_LAYERS)
    if(gEnableValidation)
        DestroyDebugCallback();
#endif // USES_VULKAN_DEBUG_LAYERS

    m_pVulkanQueueProps.clear();
        
    m_VulkanGpu = VK_NULL_HANDLE;
    m_VulkanGpuCount = 0;
    m_VulkanGpuIdx = 0;

    volkFinalize();
}


//-----------------------------------------------------------------------------
bool Vulkan::Init(uintptr_t windowHandle, uintptr_t hInst, const Vulkan::tSelectSurfaceFormatFn& SelectSurfaceFormatFn, const Vulkan::tConfigurationFn& CustomConfigurationFn)
//-----------------------------------------------------------------------------
{
    if (volkInitialize() != VK_SUCCESS)
    {
        return false;
    }

#if defined (OS_WINDOWS)
    m_hWnd = (HWND)windowHandle;
    m_hInstance = (HINSTANCE)hInst;
#elif defined (OS_ANDROID)
    m_pAndroidWindow = (ANativeWindow*)windowHandle;
#elif defined (OS_LINUX)
    m_pGlfwWindow = (GLFWwindow*)windowHandle;
#endif // defined (OS_WINDOWS|OS_ANDROID)

    if (CustomConfigurationFn)
        CustomConfigurationFn( m_ConfigOverride );

    if (!RegisterKnownExtensions())
        return false;

    if (!CreateInstance())
        return false;

    if (!InitInstanceFunctions())
        return false;

    if (!GetPhysicalDevices())
        return false;

    if (!InitQueue())
        return false;

#if defined(USES_VULKAN_DEBUG_LAYERS)
    if (gEnableValidation && m_LayerKhronosValidationAvailable && !InitDebugCallback())
        return false;
#endif // defined(USES_VULKAN_DEBUG_LAYERS)

    if (!InitSurface())
        return false;

    if (!InitDataGraph())
        return false;

    if (!InitCompute())
        return false;

    if (!InitDevice())
        return false;

    if (!GetDataGraphProcessingEngine())
        return false;

    if (!InitCommandPools())
        return false;

    if (!InitSyncElements())
        return false;

    if (!InitMemoryManager())
        return false;

    InitPipelineCache();    //ok for this to fail!

    if (!QuerySurfaceCapabilities())
        return false;

    if (SelectSurfaceFormatFn)
    {
        int surfaceFormatIndex = SelectSurfaceFormatFn(m_SurfaceFormats);
        if (surfaceFormatIndex >= 0)
        {
            m_SurfaceFormat = m_SurfaceFormats[surfaceFormatIndex].format;
            if (m_SurfaceFormat == TextureFormat::UNDEFINED) // this is allowed in the spec
                m_SurfaceFormat = TextureFormat::B8G8R8A8_UNORM;
            m_SurfaceColorSpace = m_SurfaceFormats[surfaceFormatIndex].colorSpace;
        }
    }

    if (!InitSwapChain())
        return false;

    if (!InitSwapchainRenderPass())
        return false;

    if (!InitFrameBuffers())
        return false;

    return true;
}

//-----------------------------------------------------------------------------
void Vulkan::Terminate()
//-----------------------------------------------------------------------------
{
    // Destroy the callback
#if defined(USES_VULKAN_DEBUG_LAYERS)
    if(gEnableValidation)
        DestroyDebugCallback();
#endif // USES_VULKAN_DEBUG_LAYERS
}

//-----------------------------------------------------------------------------
bool Vulkan::ReInit(uintptr_t windowHandle)
//-----------------------------------------------------------------------------
{
#if defined (OS_WINDOWS)
    return false;   ///TODO: implement ReInit on Windows
#elif defined (OS_ANDROID)
    if (m_pAndroidWindow != (ANativeWindow*)windowHandle)
    {
        m_pAndroidWindow = (ANativeWindow*)windowHandle;

        vkDestroySurfaceKHR( m_VulkanInstance, m_VulkanSurface, nullptr );
        m_VulkanSurface = nullptr;

        InitSurface();

        return true;
    }
    return true;    // Success
#elif defined (OS_LINUX)
    return false;   ///TODO: implement ReInit on Linux
#endif // defined (OS_WINDOWS|OS_ANDROID|OS_LINUX)
}
    
//-----------------------------------------------------------------------------
VkCompositeAlphaFlagBitsKHR Vulkan::GetBestVulkanCompositeAlpha()
//-----------------------------------------------------------------------------
{
    // Find a supported composite alpha format
    VkCompositeAlphaFlagBitsKHR DesiredCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    // Win32 seems to support Opaque, but Android likes Inherit
    // Looking through to find the supported one seems the best way to handle it.
    VkCompositeAlphaFlagBitsKHR PossibleCompositAlphaFormats[] = {
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
    };
    uint32_t NumPossibleFormats = sizeof(PossibleCompositAlphaFormats) / sizeof(VkCompositeAlphaFlagBitsKHR);

    for (uint32_t WhichFormat = 0; WhichFormat < NumPossibleFormats; WhichFormat++)
    {
        VkCompositeAlphaFlagBitsKHR OneFormat = PossibleCompositAlphaFormats[WhichFormat];
        if (m_VulkanSurfaceCaps.supportedCompositeAlpha & OneFormat)
        {
            // Found a supported format
            DesiredCompositeAlpha = OneFormat;
            break;
        }
    }

    return DesiredCompositeAlpha;
}

//-----------------------------------------------------------------------------
TextureFormat Vulkan::GetBestSurfaceDepthFormat(bool NeedStencil)
//-----------------------------------------------------------------------------
{
    // Start with the highest depth format we can, and work down.
    // We could start with VK_FORMAT_UNDEFINED and then check if we
    // didn't find an optimized format, but that doesn't mean the format
    // is not supported, just not optimized.
    TextureFormat DesiredDepthFormat = NeedStencil ? TextureFormat::D32_SFLOAT_S8_UINT : TextureFormat::D32_SFLOAT;

    // Order of the highest precision depth formats, favoring the ones without stencil if available.
    const auto PossibleDepthFormats = {
        TextureFormat::D32_SFLOAT,
        TextureFormat::D32_SFLOAT_S8_UINT,
        TextureFormat::D24_UNORM_S8_UINT,
        TextureFormat::D16_UNORM,
        TextureFormat::D16_UNORM_S8_UINT
    };
    // Highest to lowest precision depth formats, with stencil bits.
    const auto PossibleDepthFormatsWithStencil = {
        TextureFormat::D32_SFLOAT_S8_UINT,
        TextureFormat::D24_UNORM_S8_UINT,
        TextureFormat::D16_UNORM_S8_UINT
    };

    if( NeedStencil )
    {
        for(const auto& DepthFormat: (NeedStencil ? PossibleDepthFormatsWithStencil : PossibleDepthFormats) )
        {
            const VkFormatProperties& DepthFormatProps = GetFormatProperties(TextureFormatToVk(DepthFormat) );

            // If depth/stencil supported with optimal tiling we have found what we want
            if (DepthFormatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
            {
                DesiredDepthFormat = DepthFormat;
                break;
            }
        }
    }

    // TODO: Start with TextureFormat::UNDEFINED and check if we found something
    return DesiredDepthFormat;
}

//-----------------------------------------------------------------------------
bool Vulkan::IsTextureFormatSupported( TextureFormat format) const
//-----------------------------------------------------------------------------
{
    switch (format) {
    case TextureFormat::BC1_RGB_UNORM_BLOCK:
    case TextureFormat::BC1_RGB_SRGB_BLOCK:
    case TextureFormat::BC1_RGBA_UNORM_BLOCK:
    case TextureFormat::BC1_RGBA_SRGB_BLOCK:
    case TextureFormat::BC2_UNORM_BLOCK:
    case TextureFormat::BC2_SRGB_BLOCK:
    case TextureFormat::BC3_UNORM_BLOCK:
    case TextureFormat::BC3_SRGB_BLOCK:
    case TextureFormat::BC4_UNORM_BLOCK:
    case TextureFormat::BC4_SNORM_BLOCK:
    case TextureFormat::BC5_UNORM_BLOCK:
    case TextureFormat::BC5_SNORM_BLOCK:
    case TextureFormat::BC6H_UFLOAT_BLOCK:
    case TextureFormat::BC6H_SFLOAT_BLOCK:
    case TextureFormat::BC7_UNORM_BLOCK:
    case TextureFormat::BC7_SRGB_BLOCK:
        return m_VulkanGpuFeatures.Base.features.textureCompressionBC;
    case TextureFormat::ETC2_R8G8B8_UNORM_BLOCK:
    case TextureFormat::ETC2_R8G8B8_SRGB_BLOCK:
    case TextureFormat::ETC2_R8G8B8A1_UNORM_BLOCK:
    case TextureFormat::ETC2_R8G8B8A1_SRGB_BLOCK:
    case TextureFormat::ETC2_R8G8B8A8_UNORM_BLOCK:
    case TextureFormat::ETC2_R8G8B8A8_SRGB_BLOCK:
        return m_VulkanGpuFeatures.Base.features.textureCompressionETC2;
    case TextureFormat::ASTC_4x4_UNORM_BLOCK:
    case TextureFormat::ASTC_4x4_SRGB_BLOCK:
    case TextureFormat::ASTC_5x4_UNORM_BLOCK:
    case TextureFormat::ASTC_5x4_SRGB_BLOCK:
    case TextureFormat::ASTC_5x5_UNORM_BLOCK:
    case TextureFormat::ASTC_5x5_SRGB_BLOCK:
    case TextureFormat::ASTC_6x5_UNORM_BLOCK:
    case TextureFormat::ASTC_6x5_SRGB_BLOCK:
    case TextureFormat::ASTC_6x6_UNORM_BLOCK:
    case TextureFormat::ASTC_6x6_SRGB_BLOCK:
    case TextureFormat::ASTC_8x5_UNORM_BLOCK:
    case TextureFormat::ASTC_8x5_SRGB_BLOCK:
    case TextureFormat::ASTC_8x6_UNORM_BLOCK:
    case TextureFormat::ASTC_8x6_SRGB_BLOCK:
    case TextureFormat::ASTC_8x8_UNORM_BLOCK:
    case TextureFormat::ASTC_8x8_SRGB_BLOCK:
    case TextureFormat::ASTC_10x5_UNORM_BLOCK:
    case TextureFormat::ASTC_10x5_SRGB_BLOCK:
    case TextureFormat::ASTC_10x6_UNORM_BLOCK:
    case TextureFormat::ASTC_10x6_SRGB_BLOCK:
    case TextureFormat::ASTC_10x8_UNORM_BLOCK:
    case TextureFormat::ASTC_10x8_SRGB_BLOCK:
    case TextureFormat::ASTC_10x10_UNORM_BLOCK:
    case TextureFormat::ASTC_10x10_SRGB_BLOCK:
    case TextureFormat::ASTC_12x10_UNORM_BLOCK:
    case TextureFormat::ASTC_12x10_SRGB_BLOCK:
    case TextureFormat::ASTC_12x12_UNORM_BLOCK:
    case TextureFormat::ASTC_12x12_SRGB_BLOCK:
        return m_VulkanGpuFeatures.Base.features.textureCompressionASTC_LDR;
    case TextureFormat::PVRTC1_2BPP_UNORM_BLOCK_IMG:
    case TextureFormat::PVRTC1_4BPP_UNORM_BLOCK_IMG:
    case TextureFormat::PVRTC2_2BPP_UNORM_BLOCK_IMG:
    case TextureFormat::PVRTC2_4BPP_UNORM_BLOCK_IMG:
    case TextureFormat::PVRTC1_2BPP_SRGB_BLOCK_IMG:
    case TextureFormat::PVRTC1_4BPP_SRGB_BLOCK_IMG:
    case TextureFormat::PVRTC2_2BPP_SRGB_BLOCK_IMG:
    case TextureFormat::PVRTC2_4BPP_SRGB_BLOCK_IMG:
        // disable all the PVR formats (not supported on any of our target hardware)
        return false;
    default:
        // Assume everything else is supported.
        return true;
    }
}

//-----------------------------------------------------------------------------
const VkFormatProperties& Vulkan::GetFormatProperties(const VkFormat format) const
//-----------------------------------------------------------------------------
{
    auto it = m_FormatProperties.try_emplace(format, VkFormatProperties{});
    if (it.second)
    {
        vkGetPhysicalDeviceFormatProperties(m_VulkanGpu, format, &it.first->second);
    }
    return it.first->second;
}

//-----------------------------------------------------------------------------
void Vulkan::SetImageLayout(VkImage image,
    VkCommandBuffer cmdBuffer,
    VkImageAspectFlags aspect,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkPipelineStageFlags srcMask,
    VkPipelineStageFlags dstMask,
    uint32_t baseMipLevel,
    uint32_t mipLevelCount,
    uint32_t baseLayer,
    uint32_t layerCount)
    //-----------------------------------------------------------------------------
{
    VkPipelineStageFlags srcStageFlags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;    //  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags destStageFlags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;   //  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    // Create an image barrier object
    VkImageMemoryBarrier imageMemoryBarrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    imageMemoryBarrier.oldLayout = oldLayout;
    imageMemoryBarrier.newLayout = newLayout;
    imageMemoryBarrier.image = image;
    imageMemoryBarrier.subresourceRange.aspectMask = aspect;
    imageMemoryBarrier.subresourceRange.baseMipLevel = baseMipLevel;
    imageMemoryBarrier.subresourceRange.levelCount = mipLevelCount;
    imageMemoryBarrier.subresourceRange.baseArrayLayer = baseLayer;
    imageMemoryBarrier.subresourceRange.layerCount = layerCount;
    imageMemoryBarrier.srcAccessMask = 0;
    imageMemoryBarrier.dstAccessMask = 0;
    imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    switch (oldLayout)
    {
    case VK_IMAGE_LAYOUT_UNDEFINED:
        // There is no need to barrier from 'Undefined'.  We need to know what we are coming from in order to barrier.
        // If we are clearing the image in the upcoming pipeline then Undefined oldLayout is valid, but no need to create a barrier!
        //assert(0);
        break;
    case VK_IMAGE_LAYOUT_GENERAL:
        // Old layout is General, assume this comes from Compute
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        break;

    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        // Old layout is color attachment
        // Make sure any writes to the color buffer have been finished
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        srcStageFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        break;

    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        // Old layout is depth/stencil attachment
        // Make sure any writes to the depth/stencil buffer have been finished
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        srcStageFlags = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        break;

    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        srcStageFlags = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        break;

    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        // Old layout is shader read (sampler, input attachment)
        // Make sure any shader reads from the image have been finished
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        break;

    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        // Old layout is transfer source
        // Make sure any reads from the image have been finished
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        break;

    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        imageMemoryBarrier.srcAccessMask = 0;
        break;

    case VK_IMAGE_LAYOUT_PREINITIALIZED:
        // Only allowed as initial layout!
        // Make sure any writes to the image have been finished
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        srcStageFlags = VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        break;

    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        // LOGE("Don't know how to set image layout if starting with VK_IMAGE_LAYOUT_PRESENT_SRC_KHR!");
        // return;
        break;

    default:
        // Compiler warning for enumeration not handled in switch
        break;
    }

    // ********************************
    // New Layout
    // ********************************
    switch (newLayout)
    {
    case VK_IMAGE_LAYOUT_UNDEFINED:
        // Don't know what to do with this!
        LOGE("Don't know how to set image layout if transitioning to undefined!");
        assert(0);
        // return;
        break;

    case VK_IMAGE_LAYOUT_GENERAL:
        // New layout is 'general' (for compute storage buffer)
        imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        destStageFlags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        break;

    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        // New layout is color attachment
        // Make sure any writes to the color buffer hav been finished
        imageMemoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        destStageFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        if (oldLayout != VK_IMAGE_LAYOUT_UNDEFINED)
        {
            // This causes a validation warning if layout is undefined
            imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            srcStageFlags = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        break;

    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        // New layout is depth attachment
        // Make sure any writes to depth/stencil buffer have been finished
        imageMemoryBarrier.dstAccessMask = imageMemoryBarrier.dstAccessMask | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        destStageFlags = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        break;

    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
        break;

    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        // New layout is shader read (sampler, input attachment)
        // Make sure any writes to the image have been finished
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStageFlags = VK_PIPELINE_STAGE_TRANSFER_BIT;
        imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        destStageFlags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        break;

    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        // New layout is transfer source (copy, blit)
        // Make sure any reads from and writes to the image have been finished
        imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        destStageFlags = VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;

        // Make sure any reads from and writes to the image have been finished
        // imageMemoryBarrier.srcAccessMask = imageMemoryBarrier.srcAccessMask | VK_ACCESS_TRANSFER_READ_BIT;
        // imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        break;

    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        // New layout is transfer destination (copy, blit)
        // Make sure any reads from and writes to the image have been finished
        imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        destStageFlags = VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;

        // Make sure any copyies to the image have been finished
        // imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        break;

    case VK_IMAGE_LAYOUT_PREINITIALIZED:
        break;

    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        imageMemoryBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        destStageFlags = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        break;

    default:
        // Compiler warning for enumeration not handled in switch
        break;
    }


    // Barrier on image memory, with correct layouts set.
    vkCmdPipelineBarrier(cmdBuffer,
        srcStageFlags,     // srcMask,
        destStageFlags,    // dstMask,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &imageMemoryBarrier);
}

//-----------------------------------------------------------------------------
bool Vulkan::RegisterKnownExtensions()
//-----------------------------------------------------------------------------
{
    //
    // Add any Vulkan Instance and Device Extensions we must have (expected to be a short list)
    //

    for (auto& appExtension : m_ConfigOverride.AdditionalVulkanInstanceExtensions)
    {
        auto insertIt = m_InstanceExtensions.m_Extensions.insert(std::move(appExtension));
        assert(insertIt.second);  // check we didnt add a duplicate!
    }

    for (auto& appExtension : m_ConfigOverride.AdditionalVulkanDeviceExtensions)
    {
        auto insertIt = m_DeviceExtensions.m_Extensions.insert( std::move( appExtension ) );
        assert( insertIt.second );  // check we didnt add a duplicate!
    }

    for (auto& appExtension : m_ConfigOverride.AdditionalVulkanInstanceLayers)
    {
        auto insertIt = m_InstanceLayers.m_Extensions.insert( std::move( appExtension ) );
        assert( insertIt.second );  // check we didnt add a duplicate!
    }

#if defined(USES_VULKAN_DEBUG_LAYERS)
    //
    // Add validation layer and layer extensions
    //
    // If this is NOT set, we cannot use the debug extensions! (Find vkCreateDebugReportCallback)
    if (gEnableValidation)
    {
        m_InstanceLayers.AddExtension( "VK_LAYER_KHRONOS_validation", VulkanExtensionStatus::eOptional );
        m_InstanceExtensions.AddExtension( VK_EXT_DEBUG_REPORT_EXTENSION_NAME, VulkanExtensionStatus::eOptional );
        m_InstanceExtensions.AddExtension( VK_EXT_DEBUG_UTILS_EXTENSION_NAME, VulkanExtensionStatus::eOptional );
        // This extension allows us to use VkValidationFeaturesEXT (>0 if available)
        m_ExtValidationFeatures = m_InstanceExtensions.AddExtension( VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME, VulkanExtensionStatus::eOptional );
    }
#endif // USES_VULKAN_DEBUG_LAYERS

    //
    // Add the Instance extensions we need
    //
    m_ExtKhrSurface = m_InstanceExtensions.AddExtension<ExtensionLib::Ext_VK_KHR_surface>();

#if defined(VK_USE_PLATFORM_WIN32_KHR)
    m_InstanceExtensions.AddExtension( VK_KHR_WIN32_SURFACE_EXTENSION_NAME, VulkanExtensionStatus::eRequired );
#endif // VK_USE_PLATFORM_WIN32_KHR

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
    m_InstanceExtensions.AddExtension( VK_KHR_ANDROID_SURFACE_EXTENSION_NAME, VulkanExtensionStatus::eRequired );
#endif // VK_USE_PLATFORM_ANDROID_KHR

#if defined(OS_LINUX)
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    for(int i=0;i<glfwExtensionCount;++i)
	m_InstanceExtensions.AddExtension( glfwExtensions[i], VulkanExtensionStatus::eRequired );
#endif // OS_LINUX

    // This extension may enable more than the default VK_COLOR_SPACE_SRGB_NONLINEAR_KHR color space (enable if available)
    m_InstanceExtensions.AddExtension( VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME, VulkanExtensionStatus::eOptional );

    // Extension to allow vkGetPhysicalDeviceProperties2KHR (which is provided by Vulkan 1.1 as vkGetPhysicalDeviceProperties2).  May already have been requested/required by the application.
    m_ExtKhrGetPhysicalDeviceProperties2 = m_InstanceExtensions.GetExtension<ExtensionLib::Ext_VK_KHR_get_physical_device_properties2>();
    if (!m_ExtKhrGetPhysicalDeviceProperties2)
        m_ExtKhrGetPhysicalDeviceProperties2 = m_InstanceExtensions.AddExtension<ExtensionLib::Ext_VK_KHR_get_physical_device_properties2>( VulkanExtensionStatus::eOptional );

    // This extension allows us to call VkPhysicalDeviceSurfaceInfo2KHR (enable if available)
    m_ExtSurfaceCapabilities2 = m_InstanceExtensions.AddExtension<ExtensionLib::Ext_VK_KHR_get_surface_capabilities2>( VulkanExtensionStatus::eOptional );

#if defined (OS_ANDROID)
    // This extension allow us to use Android Hardware Buffers
    m_InstanceExtensions.AddExtension( "VK_KHR_external_memory_capabilities", VulkanExtensionStatus::eOptional );
#endif // defined (OS_ANDROID)

#if defined (OS_LINUX)
    // May want portability
    m_InstanceExtensions.AddExtension( "VK_KHR_portability_enumeration", VulkanExtensionStatus::eOptional );
#endif // defined (OS_LINUX)

    // This extension says the device must be able to present images directly to the screen.
    m_ExtSwapchain = m_DeviceExtensions.AddExtension<ExtensionLib::Ext_VK_KHR_swapchain>( VulkanExtensionStatus::eRequired );

    //
    // Add extensions we would always LIKE to initialize (if they exist).
    //
    m_DeviceExtensions.AddExtension<ExtensionLib::Ext_VK_KHR_shader_float16_int8>( VulkanExtensionStatus::eOptional );
#if defined(USES_VULKAN_DEBUG_LAYERS)
    m_ExtDebugUtils  = m_DeviceExtensions.AddExtension<ExtensionLib::Ext_VK_EXT_debug_utils>( VulkanExtensionStatus::eOptional );
    m_ExtDebugMarker = m_DeviceExtensions.AddExtension<ExtensionLib::Ext_VK_EXT_debug_marker>( VulkanExtensionStatus::eOptional );
#else
    m_ExtDebugUtils  = m_DeviceExtensions.AddExtension<ExtensionLib::Ext_VK_EXT_debug_utils>( VulkanExtensionStatus::eUninitialized );
    m_ExtDebugMarker = m_DeviceExtensions.AddExtension<ExtensionLib::Ext_VK_EXT_debug_marker>( VulkanExtensionStatus::eUninitialized );
#endif
    m_DeviceExtensions.AddExtension( VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME, VulkanExtensionStatus::eOptional );
    m_ExtHdrMetadata = m_DeviceExtensions.AddExtension<ExtensionLib::Ext_VK_EXT_hdr_metadata>( VulkanExtensionStatus::eOptional );
    m_DeviceExtensions.AddExtension( VK_EXT_SAMPLE_LOCATIONS_EXTENSION_NAME, VulkanExtensionStatus::eOptional );
    m_DeviceExtensions.AddExtension( "VK_QCOM_render_pass_transform", VulkanExtensionStatus::eOptional);
    // This extension allows us to set  VK_SUBPASS_DESCRIPTION_SHADER_RESOLVE_BIT_QCOM (enable if available)
    m_DeviceExtensions.AddExtension( "VK_QCOM_render_pass_shader_resolve", VulkanExtensionStatus::eOptional);
    m_DeviceExtensions.AddExtension<ExtensionLib::Ext_VK_KHR_portability_subset>( VulkanExtensionStatus::eOptional );
    m_DeviceExtensions.AddExtension<ExtensionLib::Ext_VK_EXT_subgroup_size_control>( VulkanExtensionStatus::eOptional );
    m_DeviceExtensions.AddExtension( VK_EXT_SHADER_SUBGROUP_BALLOT_EXTENSION_NAME, VulkanExtensionStatus::eOptional );

    m_ExtQcomTileProperties = m_DeviceExtensions.AddExtension<ExtensionLib::Ext_VK_QCOM_tile_properties>( VulkanExtensionStatus::eOptional );

    m_SubgroupProperties = m_Vulkan11ProvidedExtensions.AddExtension<ExtensionLib::Vulkan_SubgroupPropertiesHook>();
    m_StorageFeatures = m_Vulkan11ProvidedExtensions.AddExtension<ExtensionLib::Vulkan_StorageFeaturesHook>();

#if defined (OS_ANDROID)
    m_DeviceExtensions.AddExtension( VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME, VulkanExtensionStatus::eOptional );
#endif

    m_ExtKhrSynchronization2 = m_DeviceExtensions.GetExtension<ExtensionLib::Ext_VK_KHR_synchronization2>();
    m_ExtKhrDrawIndirectCount = m_DeviceExtensions.GetExtension<ExtensionLib::Ext_VK_KHR_draw_indirect_count>();
    m_ExtRenderPass2 = m_DeviceExtensions.GetExtension<ExtensionLib::Ext_VK_KHR_create_renderpass2>();
    m_ExtBufferDeviceAddress = m_DeviceExtensions.GetExtension<ExtensionLib::Ext_VK_KHR_buffer_device_address>();
    m_Ext8BitStorage = m_DeviceExtensions.GetExtension<ExtensionLib::Ext_VK_KHR_8bit_storage>();
    m_ExtCooperativeMatrix = m_DeviceExtensions.GetExtension<ExtensionLib::Ext_VK_KHR_cooperative_matrix>();
    m_ExtFragmentShadingRate = m_DeviceExtensions.GetExtension<ExtensionLib::Ext_VK_KHR_fragment_shading_rate>();
    m_ExtMeshShader = m_DeviceExtensions.GetExtension<ExtensionLib::Ext_VK_KHR_mesh_shader>();
    m_ExtDynamicRendering = m_DeviceExtensions.GetExtension<ExtensionLib::Ext_VK_KHR_dynamic_rendering>();
    m_ExtQcomTileShading = m_DeviceExtensions.GetExtension<ExtensionLib::Ext_VK_QCOM_tile_shading>();
    m_ExtQcomTileMemoryHeap = m_DeviceExtensions.GetExtension<ExtensionLib::Ext_VK_QCOM_tile_memory_heap>();
    m_ExtKhrGetMemoryRequirements = m_DeviceExtensions.GetExtension<ExtensionLib::Ext_VK_KHR_get_memory_requirements2>();

    // Now we have a list of all the extensions we know about we can ask them to Register themselves with whatever 'hooks' they require.
    m_InstanceLayers.RegisterAll(*this);
    m_InstanceExtensions.RegisterAll(*this);
    m_DeviceExtensions.RegisterAll( *this );
    m_Vulkan11ProvidedExtensions.RegisterAll( *this );

    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::CreateInstance()
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    // ********************************
    // Debug/Validation/Whatever Layers
    // ********************************
    if (!InitInstanceExtensions())
    {
        return false;
    }

    // ********************************
    // Create the Vulkan Instance
    // ********************************

    // We specify the Vulkan version our application was built with,
    // as well as names and versions for our application and engine,
    // if applicable. This allows the driver to gain insight to what
    // is utilizing the vulkan driver, and serve appropriate versions.
    VkApplicationInfo AppInfoStruct{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    AppInfoStruct.pApplicationName = "VkFrameworkSample";
    AppInfoStruct.applicationVersion = 0;
    AppInfoStruct.pEngineName = "VkFrameworkEngine";
    AppInfoStruct.engineVersion = 0;
    AppInfoStruct.apiVersion = m_ConfigOverride.ApiVerson.value_or( VK_MAKE_VERSION( 1, 2, 0 ) );

    // Create the list of layers to enable.
    std::vector<const char*> InstanceLayerNames;
    InstanceLayerNames.reserve( m_InstanceLayers.m_Extensions.size() );
    for (const auto& e : m_InstanceLayers.m_Extensions)
        if (e.second->Status == VulkanExtensionStatus::eLoaded && e.second->LoadMode != VulkanExtensionLoadMode::eSkipRequest)
            InstanceLayerNames.push_back( e.first.c_str() );

    // Create the list of instance extensions to enable
    std::vector<const char*> InstanceExtensionNames;
    InstanceExtensionNames.reserve(m_InstanceExtensions.m_Extensions.size());
    for (const auto& e : m_InstanceExtensions.m_Extensions)
        if (e.second->Status == VulkanExtensionStatus::eLoaded && e.second->LoadMode != VulkanExtensionLoadMode::eSkipRequest)
            InstanceExtensionNames.push_back(e.first.c_str());

    fvk::VkStructWrapper<VkInstanceCreateInfo, VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO> InstanceInfoStruct{{
        .flags = 0,
        .pApplicationInfo = &AppInfoStruct,
        .enabledLayerCount = (uint32_t)InstanceLayerNames.size(),
        .ppEnabledLayerNames = InstanceLayerNames.data(),
        .enabledExtensionCount = (uint32_t)InstanceExtensionNames.size(),
        .ppEnabledExtensionNames = InstanceExtensionNames.data()
    }};
    InstanceInfoStruct->flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

    // ********************************
    // Potentially add Validation layer feature settings.
    // ********************************
    std::vector<VkValidationFeatureEnableEXT> ValidationFeaturesEnables;
    if (m_ExtValidationFeatures && m_ExtValidationFeatures->Version >= 2)    // spec version 1 does not support 'best practices'
    {
        auto& ValidationFeaturesStruct = InstanceInfoStruct.Add<VkValidationFeaturesEXT, VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT>();

#if (VK_EXT_VALIDATION_FEATURES_SPEC_VERSION < 4)   // if our header does not define the spec version 4 then add the 'missing' enable
#define VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT (4)
#endif
        if (gEnableValidationBestPractices)
        {
            LOGI( "Enabling VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT in Validation Layer" );
            ValidationFeaturesEnables.push_back(VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT);
        }
        if (gEnableValidationGpu)
        {
            LOGI("Enabling VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT in Validation Layer");
            LOGI("Enabling VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT in Validation Layer");
            ValidationFeaturesEnables.push_back((VkValidationFeatureEnableEXT)VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT);
            ValidationFeaturesEnables.push_back((VkValidationFeatureEnableEXT)VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT );
        }
        if (gEnableValidationDebugPrintf && m_ExtValidationFeatures->Version >= 3) // spec versions before 3 do not support printf
        {
            if (gEnableValidationGpu)
            {
                LOGW("Cannot use gEnableValidationDebugPrintf at the same time as gEnableValidationGpu");
            }
            else
            {
                LOGI("Enabling VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT in Validation Layer");
                ValidationFeaturesEnables.push_back((VkValidationFeatureEnableEXT)VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT);
            }
        }
        if (gEnableValidationSynchronization && m_ExtValidationFeatures->Version >= 4) // spec versions before 4 do not support synchronization validation
        {
            LOGI("Enabling VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT in Validation Layer");
            ValidationFeaturesEnables.push_back((VkValidationFeatureEnableEXT)VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT);
        }

        if (!ValidationFeaturesEnables.empty())
        {
            ValidationFeaturesStruct->enabledValidationFeatureCount = (uint32_t) ValidationFeaturesEnables.size();
            ValidationFeaturesStruct->pEnabledValidationFeatures = ValidationFeaturesEnables.data();
        }
    }

    uint32_t MajorVersion = VK_VERSION_MAJOR(AppInfoStruct.apiVersion);
    uint32_t MinorVersion = VK_VERSION_MINOR(AppInfoStruct.apiVersion);
    uint32_t PatchVersion = VK_VERSION_PATCH(AppInfoStruct.apiVersion);
    LOGI("Requesting Vulkan version %d.%d.%d", MajorVersion, MinorVersion, PatchVersion);

    LOGI("Requesting Layers:");
    for (const auto& LayerName : InstanceLayerNames)
    {
        LOGI("    %s", LayerName);
    }

    LOGI("Requesting Extensions:");
    for (const auto& ExtensionName: InstanceExtensionNames)
    {
        LOGI("    %s", ExtensionName);
    }

    LOGI("Creating Vulkan Instance...");

    LOGI("    InstanceInfo.enabledLayerCount: %d", InstanceInfoStruct->enabledLayerCount);
    for (uint32_t Which = 0; Which < InstanceInfoStruct->enabledLayerCount; Which++)
    {
        LOGI("        %d: %s", Which, InstanceInfoStruct->ppEnabledLayerNames[Which]);
    }

    LOGI("    InstanceInfo.enabledExtensionCount: %d", InstanceInfoStruct->enabledExtensionCount);
    for (uint32_t Which = 0; Which < InstanceInfoStruct->enabledExtensionCount; Which++)
    {
        LOGI("        %d: %s", Which, InstanceInfoStruct->ppEnabledExtensionNames[Which]);
    }

    // The main Vulkan instance is created with the creation infos above.
    // We do not specify a custom memory allocator for instance creation.
    retVal = vkCreateInstance(&InstanceInfoStruct, nullptr, &m_VulkanInstance);

    // ********************************
    // Handle any errors from create instance
    // ********************************
    // Vulkan API return values can expose further information on a failure.
    // For instance, INCOMPATIBLE_DRIVER may be returned if the API level
    // an application is built with, exposed through VkApplicationInfo, is
    // newer than the driver present on a device.
    if (retVal == VK_ERROR_INCOMPATIBLE_DRIVER)
    {
        LOGE("Cannot find a compatible Vulkan installable client driver: vkCreateInstance Failure");
        return false;
    }
    else if (retVal == VK_ERROR_EXTENSION_NOT_PRESENT)
    {
        LOGE("Cannot find a specified extension library: vkCreateInstance Failure");
        return false;
    }
    else if (!CheckVkError("vkCreateInstance()", retVal))
    {
        return false;
    }
    volkLoadInstance(m_VulkanInstance);

    m_VulkanApiVersion = AppInfoStruct.apiVersion;

    // ********************************
    // Call the PostLoad for all instance extensions
    // ********************************
    for (auto& extension : m_InstanceExtensions.m_Extensions)
    {
        if (extension.second->Status == VulkanExtensionStatus::eLoaded)
            extension.second->PostLoad();
    }

    return true;
}

//-----------------------------------------------------------------------------
// Parse ExtensionProps for extensions in RegisteredExtensions.
// If found set the registered extension's version and change Status to eLoaded if requested to load (status eOptional or eRequired).
// If not found add to the list of RegisteredExtensions (with a eUninitialized status)
template<VulkanExtensionType T_TYPE>
static bool ParseExtensionProperties( const std::vector<VkExtensionProperties>& ExtensionProps, Vulkan::RegisteredExtensions<T_TYPE>& RegisteredExtensions )
//-----------------------------------------------------------------------------
{
    bool success = true;
    for (uint32_t uiIndx = 0; uiIndx < (uint32_t)ExtensionProps.size(); uiIndx++)
    {
        const auto* const pOneItem = &ExtensionProps[uiIndx];
        LOGI( "    %d: %s", uiIndx, pOneItem->extensionName );

        // Need to check if we found specific device extensions
        {
            std::string extensionName = pOneItem->extensionName;
            auto* pExtension = RegisteredExtensions.GetExtension( extensionName );
            if (pExtension)
            {
                // Found an already known extension, see if we want to do anything with it.
                switch (pExtension->Status)
                {
                case VulkanExtensionStatus::eOptional:
                case VulkanExtensionStatus::eRequired:
                    pExtension->Status = VulkanExtensionStatus::eLoaded;  // strictly not 'yet' loaded but will be shortly
                    break;
                case VulkanExtensionStatus::eLoaded:
                case VulkanExtensionStatus::eUninitialized:
                    break;
                }
                pExtension->Version = pOneItem->specVersion;
            }
            else
            {
                // Add to the list of known extensions (assume we dont want to load it)
                RegisteredExtensions.AddExtension( extensionName, VulkanExtensionStatus::eUninitialized, pOneItem->specVersion );
            }
        }
    }

    // Do a final check of extensions we requested (as required or optional) that are not present in the list of available extensions.
    for (auto& extension : RegisteredExtensions.m_Extensions)
    {
        if (extension.second->Status == VulkanExtensionStatus::eRequired)
        {
            if (extension.second->LoadMode == VulkanExtensionLoadMode::eDefault)
            {
                LOGE( "Required Vulkan extension \"%s\" was not found.", extension.first.c_str() );
                success = false;
            }
            else
            {
                LOGW( "Required Vulkan extension \"%s\" not found but VulkanExtensionLoadMode requesting we continue.", extension.first.c_str() );
                extension.second->Status = VulkanExtensionStatus::eLoaded;
            }
        }
        else if (extension.second->Status == VulkanExtensionStatus::eOptional)
        {
            // If requested as optional but not available then set to be uninitialized.
            extension.second->Status = VulkanExtensionStatus::eUninitialized;
        }
    }
    return success;
}

//-----------------------------------------------------------------------------
// Parse LayerProps for layers in RegisteredLayers.
// If found set the registered layer's version and change Status to eLoaded if requested to load (status eOptional or eRequired).
// If not found add to the list of RegisteredLayers (with a eUninitialized status)
static bool ParseLayerProperties( const std::vector<VkLayerProperties>& LayerProps, Vulkan::RegisteredExtensions<VulkanExtensionType::eLayer>& RegisteredLayers )
//-----------------------------------------------------------------------------
{
    bool success = true;
    for (uint32_t uiIndx = 0; uiIndx < (uint32_t)LayerProps.size(); uiIndx++)
    {
        const auto* const pOneItem = &LayerProps[uiIndx];
        LOGI( "    %d: %s", uiIndx, pOneItem->layerName );

        // Need to check if we found specific device extensions
        {
            std::string name = pOneItem->layerName;
            auto* pLayer = RegisteredLayers.GetExtension( name );
            if (pLayer)
            {
                // Found an already known extension, see if we want to do anything with it.
                switch (pLayer->Status)
                {
                case VulkanExtensionStatus::eOptional:
                case VulkanExtensionStatus::eRequired:
                    pLayer->Status = VulkanExtensionStatus::eLoaded;  // strictly not 'yet' loaded but will be shortly
                    break;
                case VulkanExtensionStatus::eLoaded:
                case VulkanExtensionStatus::eUninitialized:
                    break;
                }
                pLayer->Version = pOneItem->specVersion;
            }
            else
            {
                // Add to the list of known extensions (assume we dont want to load it)
                RegisteredLayers.AddExtension( name, VulkanExtensionStatus::eUninitialized, pOneItem->specVersion );
            }
        }
    }

    // Do a final check of extensions we requested (as required or optional) that are not present in the list of available extensions.
    for (auto& extension : RegisteredLayers.m_Extensions)
    {
        if (extension.second->Status == VulkanExtensionStatus::eRequired)
        {
            if (extension.second->LoadMode == VulkanExtensionLoadMode::eDefault)
            {
                LOGE( "Required Vulkan extension \"%s\" was not found.", extension.first.c_str() );
                success = false;
            }
            else
            {
                LOGW( "Required Vulkan extension \"%s\" not found but VulkanExtensionLoadMode requesting we continue.", extension.first.c_str() );
                extension.second->Status = VulkanExtensionStatus::eLoaded;
            }
        }
        else if (extension.second->Status == VulkanExtensionStatus::eOptional)
        {
            // If requested as optional but not available then set to be uninitialized.
            extension.second->Status = VulkanExtensionStatus::eUninitialized;
        }
    }
    return success;
}


//-----------------------------------------------------------------------------
bool Vulkan::InitInstanceExtensions()
//-----------------------------------------------------------------------------
{
    VkResult retVal;

    // ********************************
    // Look for Instance Layers (eg Validation)
    // ********************************
    {
        uint32_t NumInstanceLayerProps = 0;
        retVal = vkEnumerateInstanceLayerProperties(&NumInstanceLayerProps, nullptr);
        if (!CheckVkError("vkEnumerateInstanceLayerProperties(nullptr)", retVal))
        {
            return false;
        }

        LOGI("Found %d Vulkan Instance Layer Properties", NumInstanceLayerProps);
        if (NumInstanceLayerProps > 0)
        {
            // Allocate memory for the structures...
            std::vector<VkLayerProperties> InstanceLayerProps;
            InstanceLayerProps.resize(NumInstanceLayerProps);

            // ... then read the structures from the driver
            retVal = vkEnumerateInstanceLayerProperties(&NumInstanceLayerProps, InstanceLayerProps.data());
            if (!CheckVkError("vkEnumerateInstanceLayerProperties()", retVal))
            {
                return false;
            }

            if (!ParseLayerProperties( InstanceLayerProps, m_InstanceLayers ))
            {
                LOGE( "Required Vulkan Instance Layer(s) missing." );
                return false;
            }

            auto* validationLayer = m_InstanceLayers.GetExtension( "VK_LAYER_KHRONOS_validation" );
            if (validationLayer && validationLayer->Status == VulkanExtensionStatus::eLoaded)
            {
                m_LayerKhronosValidationAvailable = true;
            }
        }
    }

    // ********************************
    // Look for Instance Extensions
    // ********************************
    {
        uint32_t NumInstanceExtensionProps = 0;
        retVal = vkEnumerateInstanceExtensionProperties(nullptr, &NumInstanceExtensionProps, nullptr);
        if (!CheckVkError("vkEnumerateInstanceExtensionProperties(nullptr)", retVal))
        {
            return false;
        }

        std::vector<VkExtensionProperties> InstanceExtensionProps;

        LOGI("Found %d Vulkan Instance Extension Properties", NumInstanceExtensionProps);
        if (NumInstanceExtensionProps > 0)
        {
            // Allocate memory for the structures...
            InstanceExtensionProps.resize(NumInstanceExtensionProps);

            // ... then read the structures from the driver
            retVal = vkEnumerateInstanceExtensionProperties(nullptr, &NumInstanceExtensionProps, InstanceExtensionProps.data());
            if (!CheckVkError("vkEnumerateInstanceExtensionProperties()", retVal))
            {
                return false;
            }
        }

        // Now add in any instance extensions exposed by layers (that we care about)
        for (auto& layer : m_InstanceLayers.m_Extensions)
        {
            if (layer.second->Status == VulkanExtensionStatus::eLoaded)
            {
                uint32_t NumLayerInstanceExtensionProps = 0;
                const char* const pLayerName = layer.first.c_str();

                retVal = vkEnumerateInstanceExtensionProperties( pLayerName, &NumLayerInstanceExtensionProps, nullptr );
                if (!CheckVkError( "vkEnumerateInstanceExtensionProperties(VK_LAYER_KHRONOS_validation)", retVal ))
                {
                    return false;
                }

                LOGI( "Found %d Vulkan %s Instance Extension Properties", NumLayerInstanceExtensionProps, pLayerName );
                if (NumLayerInstanceExtensionProps > 0)
                {
                    // Allocate memory for the structures...
                    InstanceExtensionProps.resize( NumInstanceExtensionProps + NumLayerInstanceExtensionProps );

                    // ... then read the structures from the driver
                    retVal = vkEnumerateInstanceExtensionProperties( pLayerName, &NumLayerInstanceExtensionProps, &InstanceExtensionProps[NumInstanceExtensionProps] );
                    if (!CheckVkError( "vkEnumerateInstanceExtensionProperties(VK_LAYER_KHRONOS_validation)", retVal ))
                    {
                        return false;
                    }
                }
                NumInstanceExtensionProps += NumLayerInstanceExtensionProps;
            }
        }

        if (!ParseExtensionProperties( InstanceExtensionProps, m_InstanceExtensions ))
        {
            LOGE("Required Vulkan Instance extensions missing.");
            return false;
        }
    }

    return true;
}



//-----------------------------------------------------------------------------
bool Vulkan::GetDataGraphProcessingEngine()
//-----------------------------------------------------------------------------
{
    if (!m_VulkanGraphicsQueueSupportsDataGraph)
    {
        return true;
    }

    const auto* Ext_VK_QCOM_data_graph_model =
        static_cast<ExtensionLib::Ext_VK_QCOM_data_graph_model*>(m_DeviceExtensions.GetExtension(VK_QCOM_DATA_GRAPH_MODEL_EXTENSION_NAME));

    const bool is_qcom_data_graph_model_supported = (Ext_VK_QCOM_data_graph_model != nullptr);

    LOGI("************************************");
    LOGI("*** DATA GRAPH PROCESSING ENGINE ***");
    LOGI("************************************");

    const uint32_t queue_family_index = m_VulkanQueues[Vulkan::eDataGraphQueue].QueueFamilyIndex;

    uint32_t prop_count = 0;
    VkResult result = vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM(
        m_VulkanGpu,
        queue_family_index,
        &prop_count,
        nullptr);

    if (result != VK_SUCCESS || prop_count == 0)
    {
        LOGW("*** No data graph properties returned (result=%d, propCount=%u). Disabling data graph.", static_cast<int>(result), prop_count);
        m_VulkanGraphicsQueueSupportsDataGraph = false;
        return true;
    }

    std::vector<VkQueueFamilyDataGraphPropertiesARM> data_graph_props(prop_count);
    for (auto& p : data_graph_props)
    {
        p.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_PROPERTIES_ARM; 
        p.pNext = nullptr;
    }

    result = vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM(
        m_VulkanGpu,
        queue_family_index,
        &prop_count,
        data_graph_props.data());

    if (result != VK_SUCCESS || prop_count == 0)
    {
        LOGW("*** Failed to query data graph properties (result=%d, propCount=%u). Disabling data graph.", static_cast<int>(result), prop_count);
        m_VulkanGraphicsQueueSupportsDataGraph = false;
        return true;
    }

    LOGI("*** Checking queue data graph props:");
    LOGI("*** \tpropCount: %u", prop_count);

    for (uint32_t j = 0; j < prop_count; ++j)
    {
        LOGI("*** \t\tEngine:");
        LOGI("*** \t\t\tType:           0x%x", data_graph_props[j].engine.type);
        LOGI("*** \t\t\tisForeign:      %d", static_cast<int>(data_graph_props[j].engine.isForeign));
        LOGI("*** \t\tOperation:");
        LOGI("*** \t\t\toperationType:  0x%x", data_graph_props[j].operation.operationType);
        LOGI("*** \t\t\tname:           %s", data_graph_props[j].operation.name);
        LOGI("*** \t\t\tversion:        %u", data_graph_props[j].operation.version);
    }

    auto Select_engine_and_op = [&](
        VkPhysicalDeviceDataGraphProcessingEngineTypeARM engine_type,
        VkPhysicalDeviceDataGraphOperationTypeARM op_type) -> bool
    {

        for (uint32_t j = 0; j < prop_count; ++j)
        {
            auto& entry = data_graph_props[j];
            if (entry.engine.type == engine_type && entry.operation.operationType == op_type)
            {
                m_VulkanDataGraphProcessingEngine = entry.engine;

                LOGI("*** Selected engine/op:");
                LOGI("*** \tengine.type=0x%x  isForeign=%d",
                    entry.engine.type,
                    static_cast<int>(entry.engine.isForeign));
                LOGI("*** \top.type=0x%x     name=%s  version=%u",
                    entry.operation.operationType,
                    entry.operation.name,
                    entry.operation.version);

                return true;
            }
        }

        return false;
    };

 
    bool selected = false;

    if (is_qcom_data_graph_model_supported)
    {
        // Prefer QCOM Neural engine + QCOM NeuralModel op, then QCOM Compute engine + QCOM NeuralModel op.
        // If unavailable, try QCOM BuiltinModel op.
        //
        // QCOM op/engine enums are provided by VK_QCOM_data_graph_model
        selected =
            Select_engine_and_op(VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_NEURAL_QCOM,
                VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_NEURAL_MODEL_QCOM) ||
            Select_engine_and_op(VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_COMPUTE_QCOM,
                VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_NEURAL_MODEL_QCOM) ||
            Select_engine_and_op(VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_NEURAL_QCOM,
                VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_BUILTIN_MODEL_QCOM) ||
            Select_engine_and_op(VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_COMPUTE_QCOM,
                VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_BUILTIN_MODEL_QCOM);

        if (!selected)
        {
            LOGW("*** VK_QCOM_data_graph_model is active, but no QCOM engine/op pair was found on this queue family.");
        }
    }

    if (!selected)
    {
        // Fallback: first available
        m_VulkanDataGraphProcessingEngine = data_graph_props[0].engine;
        selected = true;

        LOGI("*** Fallback selected first engine:");
        LOGI("*** \tengine.type=0x%x  isForeign=%d",
            m_VulkanDataGraphProcessingEngine.type,
            static_cast<int>(m_VulkanDataGraphProcessingEngine.isForeign));
    }

    LOGI("Checking for Tensor Storage Format Support");
    {
        if (HasLoadedVulkanDeviceExtension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
        {
            VkTensorFormatPropertiesARM tensor_fmt_props = {};
            VkFormatProperties2         format_props = {};

            tensor_fmt_props.sType = VK_STRUCTURE_TYPE_TENSOR_FORMAT_PROPERTIES_ARM;
            tensor_fmt_props.pNext = nullptr;

            format_props.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
            format_props.pNext = &tensor_fmt_props;

            vkGetPhysicalDeviceFormatProperties2KHR(m_VulkanGpu, VK_FORMAT_R32_SFLOAT, &format_props);

            LOGI("*** \t\t\ttensorFmtProps.linearTilingTensorFeatures:   %lld", static_cast<long long>(tensor_fmt_props.linearTilingTensorFeatures));
            LOGI("*** \t\t\ttensorFmtProps.optimalTilingTensorFeatures:  %lld", static_cast<long long>(tensor_fmt_props.optimalTilingTensorFeatures));

            if ((tensor_fmt_props.linearTilingTensorFeatures & VK_FORMAT_FEATURE_2_TENSOR_DATA_GRAPH_BIT_ARM) == 0)
            {
                LOGI("*** \t\t\t - NOTE: Device doesn't support tensor storage format");
            }
        }
    }

    LOGI("Ensuring Engine Synchronization Support");
    {
        VkPhysicalDeviceQueueFamilyDataGraphProcessingEngineInfoARM info = {};
        info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_QUEUE_FAMILY_DATA_GRAPH_PROCESSING_ENGINE_INFO_ARM;
        info.pNext = nullptr;
        info.queueFamilyIndex = queue_family_index;
        info.engineType = m_VulkanDataGraphProcessingEngine.type;

        VkQueueFamilyDataGraphProcessingEnginePropertiesARM engine_props = {};
        engine_props.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_PROCESSING_ENGINE_PROPERTIES_ARM; // correct [3](https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/546)
        engine_props.pNext = nullptr;

        vkGetPhysicalDeviceQueueFamilyDataGraphProcessingEnginePropertiesARM(m_VulkanGpu, &info, &engine_props);

#if 0
        if ((engine_props.foreignSemaphoreHandleTypes & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT) == 0)
        {
            return false;
        }
        if ((engine_props.foreignMemoryHandleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID) == 0)
        {
            return false;
        }
#endif
    }

    LOGI("******************************************");
    LOGI("* DATA Graph Processing Engine Completed *");
    LOGI("******************************************");

    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::GetPhysicalDevices()
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    // Query number of physical devices available
    retVal = vkEnumeratePhysicalDevices(m_VulkanInstance, &m_VulkanGpuCount, nullptr);
    if (!CheckVkError("vkEnumeratePhysicalDevices()", retVal))
    {
        return false;
    }

    if (m_VulkanGpuCount == 0)
    {
        LOGE("Could not find a Vulkan GPU!!!!");
        return false;
    }

    LOGI("Found %d Vulkan GPU[s]", m_VulkanGpuCount);

    // Allocate space the the correct number of devices, before requesting their data
    std::vector<VkPhysicalDevice> pDevices;
    pDevices.resize(m_VulkanGpuCount, VK_NULL_HANDLE);

    retVal = vkEnumeratePhysicalDevices(m_VulkanInstance, &m_VulkanGpuCount, pDevices.data());
    if (!CheckVkError("vkEnumeratePhysicalDevices()", retVal))
    {
        return false;
    }

    // Query and display the available GPU devices and determine the 'best' GPU to use.
    {
        std::vector<PhysicalDeviceFeatures> DeviceFeatures;
        std::vector<PhysicalDeviceProperties> DeviceProperties;
        DeviceFeatures.reserve( pDevices.size() );
        DeviceProperties.reserve( pDevices.size() );

        for (VkPhysicalDevice device : pDevices)
        {
            QueryPhysicalDeviceFeatures( device, DeviceFeatures.emplace_back() );
            QueryPhysicalDeviceProperties( device, DeviceFeatures.back(), DeviceProperties.emplace_back() );
        }

        DumpDeviceInfo( DeviceFeatures, DeviceProperties );

        if (gPhysicalDevice >= 0)
        {
            LOGI( "Forcing physical device: (config gPhysicalDevice=%d).\n", gPhysicalDevice );

            if (gPhysicalDevice >= DeviceProperties.size())
            {
                LOGE( "Forced physical device out of range.  Reverting to automatic selection.\n" );
                m_VulkanGpuIdx = (uint32_t)GetBestVulkanPhysicalDeviceId( DeviceProperties );
            }
            else
            {
                m_VulkanGpuIdx = gPhysicalDevice;
            }
        }

        m_VulkanGpuIdx = (uint32_t) GetBestVulkanPhysicalDeviceId( DeviceProperties );

        LOGI("Using Vulkan Device (GPU): %d \"%s\"", m_VulkanGpuIdx, DeviceProperties[m_VulkanGpuIdx].Base.properties.deviceName);
    }

    m_VulkanGpu = pDevices[m_VulkanGpuIdx];

    // ********************************
    // Debug/Validation/Whatever Layers and extensions
    // ********************************
    if (!InitDeviceExtensions())
    {
        return false;
    }

    // Query the available features and properties for this device.
    PhysicalDeviceFeatures AvailableFeatures{};
    QueryPhysicalDeviceFeatures( m_VulkanGpu, AvailableFeatures );
    QueryPhysicalDeviceProperties( m_VulkanGpu, AvailableFeatures, m_VulkanGpuProperties );

    // Get Memory information and properties - this is required later, when we begin
    // allocating buffers to store data.
    if (m_ExtKhrGetPhysicalDeviceProperties2->Status == VulkanExtensionStatus::eLoaded)
        m_ExtKhrGetPhysicalDeviceProperties2->m_vkGetPhysicalDeviceMemoryProperties2KHR(m_VulkanGpu, &m_PhysicalDeviceMemoryProperties);
    else
        vkGetPhysicalDeviceMemoryProperties(m_VulkanGpu, &m_PhysicalDeviceMemoryProperties.memoryProperties);

    // Memory manager not available yet so can't use MemoryManager<Vulkan>::GetHeapWithHeapFlags
    // Go directly to Vulkan api for memry heap data
    {
        const auto& memoryProperties = m_PhysicalDeviceMemoryProperties.memoryProperties;
        LOGI( "VkPhysicalDeviceMemoryProperties:" );
        LOGI( "    VkMemoryType" );
        for (auto i = 0; i < memoryProperties.memoryTypeCount; ++i)
            LOGI( "    %d: propertyFlags 0x%x heapIndex %d", i, memoryProperties.memoryTypes[i].propertyFlags, memoryProperties.memoryTypes[i].heapIndex );
        LOGI( "    VkMemoryHeap" );
        for (auto i = 0; i < memoryProperties.memoryHeapCount; ++i)
            LOGI( "    %d: size %zu VkMemoryHeapFlags 0x%x", i, (size_t)memoryProperties.memoryHeaps[i].size, memoryProperties.memoryHeaps[i].flags );
    }

    // VK_QCOM_render_pass_transform had 2 versions, switch to 'legacy' if on an older driver verson.
    m_ExtRenderPassTransformLegacy = (m_ExtRenderPassTransformAvailable && (m_VulkanGpuProperties.Base.properties.driverVersion < VK_MAKE_VERSION(512, 467, 0)));
    // Internal builds have version 366 hardcoded, use the supported Vulkan version instead (the switch to legacy before 1.1.126.0 may be incorrect, 1.1.124.0 cartainly was legacy though and 1.1.128 is not!)
    m_ExtRenderPassTransformLegacy = m_ExtRenderPassTransformLegacy && ((m_VulkanGpuProperties.Base.properties.driverVersion != VK_MAKE_VERSION(512, 366, 0) || m_VulkanGpuProperties.Base.properties.apiVersion < VK_MAKE_VERSION(1, 1, 126)));

    if( m_ExtRenderPassTransformLegacy )
    {
        LOGI( "VK_QCOM_render_pass_transform legacy values" );
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::InitDeviceExtensions()
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    // ********************************
    // Look for Device Validation Layers
    // ********************************
    {
        uint32_t NumDeviceLayerProps = 0;
        retVal = vkEnumerateDeviceLayerProperties(m_VulkanGpu, &NumDeviceLayerProps, nullptr);
        if (!CheckVkError("vkEnumerateDeviceLayerProperties(nullptr)", retVal))
        {
            return false;
        }

        LOGI("Found %u Vulkan Device Layer Properties", NumDeviceLayerProps);
        if (NumDeviceLayerProps > 0)
        {
            // Allocate memory for the structures...
            std::vector<VkLayerProperties> DeviceLayerProps;
            DeviceLayerProps.resize(NumDeviceLayerProps, {});

            // ... then read the structures from the driver
            retVal = vkEnumerateDeviceLayerProperties(m_VulkanGpu, &NumDeviceLayerProps, DeviceLayerProps.data());
            if (!CheckVkError("vkEnumerateDeviceLayerProperties()", retVal))
            {
                return false;
            }

            for (uint32_t uiIndx = 0; uiIndx < (uint32_t)DeviceLayerProps.size(); uiIndx++)
            {
                VkLayerProperties* pOneItem = &DeviceLayerProps[uiIndx];
                LOGI("    %d: %s => %s", uiIndx, pOneItem->layerName, pOneItem->description);
            }
        }
    }

    // ********************************
    // Look for Device Extensions
    // ********************************
    {
        uint32_t NumDeviceExtensionProps = 0;
        retVal = vkEnumerateDeviceExtensionProperties(m_VulkanGpu, nullptr, &NumDeviceExtensionProps, nullptr);
        if (!CheckVkError("vkEnumerateDeviceExtensionProperties(nullptr)", retVal))
        {
            return false;
        }

        std::vector<VkExtensionProperties> DeviceExtensionProps;

        LOGI("Found %u Vulkan Device Extension Properties", NumDeviceExtensionProps );
        if (NumDeviceExtensionProps > 0)
        {
            // Allocate memory for the structures...
            DeviceExtensionProps.resize( NumDeviceExtensionProps );

            // ... then read the structures from the driver
            retVal = vkEnumerateDeviceExtensionProperties(m_VulkanGpu, nullptr, &NumDeviceExtensionProps, DeviceExtensionProps.data());
            if (!CheckVkError("vkEnumerateDeviceExtensionProperties()", retVal))
            {
                return false;
            }
        }

        if (!ParseExtensionProperties( DeviceExtensionProps, m_DeviceExtensions ))
        {
            LOGE( "Required Vulkan Device extensions missing." );
            return false;
        }
    }

    // Cache off any 'has loaded' flags that we want to hold on to.
    m_ExtGlobalPriorityAvailable = HasLoadedVulkanDeviceExtension( VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME );
    m_ExtRenderPassTransformAvailable = HasLoadedVulkanDeviceExtension( "VK_QCOM_render_pass_transform" );
    m_ExtRenderPassShaderResolveAvailable = HasLoadedVulkanDeviceExtension( "VK_QCOM_render_pass_shader_resolve" );
    m_ExtPortability = HasLoadedVulkanDeviceExtension( "VK_KHR_portability_subset" );
#if defined (OS_ANDROID)
    m_ExtAndroidExternalMemoryAvailable = HasLoadedVulkanDeviceExtension(VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME);
#endif

    // Add any extensions that we only need if other extensions are available (ie dependancies).
#if defined (OS_ANDROID)
    auto loadExtension = [this]( const std::string& name ) {
        auto it = m_DeviceExtensions.m_Extensions.find( name );
        if (it == m_DeviceExtensions.m_Extensions.end())
        {
        }
        else
        {
            if (it->second->Status == VulkanExtensionStatus::eUninitialized)
                it->second->Status = VulkanExtensionStatus::eLoaded;
        }
    };
    if (m_ExtAndroidExternalMemoryAvailable)
    {
        // Extensions required by the device extension VK_ANDROID_external_memory_android_hardware_buffer : 
        loadExtension( "VK_KHR_sampler_ycbcr_conversion" );
        loadExtension( "VK_KHR_external_memory" );
        loadExtension( "VK_EXT_queue_family_foreign" );
        // Extensions required by the device extension VK_KHR_sampler_ycbcr_conversion : 
        loadExtension( "VK_KHR_maintenance1" );
        loadExtension( "VK_KHR_bind_memory2" );
        loadExtension( "VK_KHR_get_memory_requirements2" );
    }
#if defined(ANDROID_HARDWARE_BUFFER_SUPPORT)
    // Need support for External Memory
    loadExtension( "VK_KHR_external_memory_fd" );
    loadExtension( "VK_KHR_external_memory" );
    loadExtension( "VK_KHR_descriptor_update_template" );
#endif // defined (ANDROID_HARDWARE_BUFFER_SUPPORT)
#endif // defined (OS_ANDROID)

    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::InitQueue()
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    // Before we create our main Vulkan device, we must ensure our physical device
    // has queue families which can perform the actions we require. For this, we request
    // the number of queue families, and their properties.
    uint32_t VulkanQueuePropertiesCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_VulkanGpu, &VulkanQueuePropertiesCount, nullptr);
    LOGI("Found %d Vulkan Device Queues Properties", VulkanQueuePropertiesCount);

    if (VulkanQueuePropertiesCount == 0)
    {
        LOGE("No Vulkan Device Queues Present!");
        return false;
    }

    // Allocate memory for the structures...
    m_pVulkanQueueProps.resize(VulkanQueuePropertiesCount);

    // ... then read the structures from the driver
    vkGetPhysicalDeviceQueueFamilyProperties(m_VulkanGpu, &VulkanQueuePropertiesCount, m_pVulkanQueueProps.data());

    for (uint32_t uiIndx = 0; uiIndx < VulkanQueuePropertiesCount; uiIndx++)
    {
        // Need to check that we found specific flag
        VkQueueFamilyProperties* pOneItem = &m_pVulkanQueueProps[uiIndx];
        if (pOneItem->queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            LOGI("    %d: Supports GRAPHICS", uiIndx);
        }
        else
        {
            LOGI("    %d: Does NOT Support GRAPHICS", uiIndx);
        }

        if (pOneItem->queueFlags & VK_QUEUE_COMPUTE_BIT)
        {
            LOGI("    %d: Supports COMPUTE", uiIndx);
        }
        else
        {
            LOGI("    %d: Does NOT Support COMPUTE", uiIndx);
        }

        if (pOneItem->queueFlags & VK_QUEUE_TRANSFER_BIT)
        {
            LOGI("    %d: Supports TRANSFER", uiIndx);
        }
        else
        {
            LOGI("    %d: Does NOT Support TRANSFER", uiIndx);
        }

        if (pOneItem->queueFlags & VK_QUEUE_DATA_GRAPH_BIT_ARM)
        {
            LOGI("    %d: Supports DATA GRAPH", uiIndx);
        }
        else
        {
            LOGI("    %d: Does NOT Support DATA GRAPH", uiIndx);
        }

        if (pOneItem->timestampValidBits > 0)
        {
            LOGI("    %d: Supports TimeStamps bits (%d valid bits)", uiIndx, pOneItem->timestampValidBits);
        }
        else
        {
            LOGI("    %d: Does NOT Support TimeStamps", uiIndx);
        }
    }


    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::InitInstanceFunctions()
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

#if defined (OS_ANDROID)
#if __ANDROID_API__ < 29
    fpGetAndroidHardwareBufferPropertiesANDROID = (PFN_vkGetAndroidHardwareBufferPropertiesANDROID)vkGetInstanceProcAddr(m_VulkanInstance, "vkGetAndroidHardwareBufferPropertiesANDROID");
    if (fpGetAndroidHardwareBufferPropertiesANDROID == nullptr)
    {
        LOGE("Unable to get function pointer from instance: vkGetAndroidHardwareBufferPropertiesANDROID");
    }
    LOGI("Initialized function pointer from instance: vkGetAndroidHardwareBufferPropertiesANDROID");
    
    fpGetImageMemoryRequirements2 = (PFN_vkGetImageMemoryRequirements2)vkGetInstanceProcAddr(m_VulkanInstance, "vkGetImageMemoryRequirements2");
    if (fpGetImageMemoryRequirements2 == nullptr)
    {
        LOGE("Unable to get function pointer from instance: vkGetImageMemoryRequirements2");
    }
    LOGI("Initialized function pointer from instance: vkGetImageMemoryRequirements2");
#endif // __ANDROID_API__ < 29

#endif // defined (OS_ANDROID)

    // Call any registered extensions for instance funtion pointer lookups.
    {
        VulkanInstanceFunctionPointerLookup fn { m_VulkanInstance };
        m_VulkanInstanceFunctionPointerLookupExtensions.PushExtensions( &fn );
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::InitDebugCallback()
//-----------------------------------------------------------------------------
{
    std::unique_ptr<VulkanDebugCallback> debugCallback = std::make_unique<VulkanDebugCallback>(*this);
    if (!debugCallback->Init())
    {
        return false;
    }
    m_DebugCallback.swap(debugCallback);
    return true;
}

//-----------------------------------------------------------------------------
void Vulkan::DestroyDebugCallback()
//-----------------------------------------------------------------------------
{
    m_DebugCallback.reset();
}

//-----------------------------------------------------------------------------
bool Vulkan::InitSurface()
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    // ********************************
    // Create surface
    // ********************************
    LOGI("Creating Vulkan Surface...");

#if defined (OS_WINDOWS)

    VkWin32SurfaceCreateInfoKHR SurfaceInfoStruct {VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    SurfaceInfoStruct.flags = 0;
    SurfaceInfoStruct.hinstance = m_hInstance;
    SurfaceInfoStruct.hwnd = m_hWnd;
    retVal = vkCreateWin32SurfaceKHR(m_VulkanInstance, &SurfaceInfoStruct, nullptr, &m_VulkanSurface);

#elif defined (OS_ANDROID)

    VkAndroidSurfaceCreateInfoKHR SurfaceInfoStruct {VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
    SurfaceInfoStruct.flags = 0;
    SurfaceInfoStruct.window = m_pAndroidWindow;
    retVal = vkCreateAndroidSurfaceKHR(m_VulkanInstance, &SurfaceInfoStruct, nullptr, &m_VulkanSurface);

#elif defined (OS_LINUX)

    retVal = glfwCreateWindowSurface(m_VulkanInstance, m_pGlfwWindow, nullptr, &m_VulkanSurface);

#endif // defined (OS_WINDOWS|OS_ANDROID|OS_LINUX)

    if (!CheckVkError("vkCreateXXXXSurfaceKHR()", retVal))
    {
        return false;
    }

    // ********************************
    // Look for queue that supports presenting
    // ********************************
    // Iterate over each queue to learn whether it supports presenting:
    std::vector<VkBool32> SupportFlags;
    SupportFlags.resize( m_pVulkanQueueProps.size(), {} );

    for (uint32_t uiIndx = 0; uiIndx < m_pVulkanQueueProps.size(); uiIndx++)
    {
        m_ExtKhrSurface->m_vkGetPhysicalDeviceSurfaceSupportKHR(m_VulkanGpu, uiIndx, m_VulkanSurface, &SupportFlags[uiIndx]);
    }

    // Look for graphics and present queues and hope there is one that supports both
    uint32_t GraphicsIndx = -1;
    uint32_t PresentIndx = -1;
    for (uint32_t uiIndx = 0; uiIndx < m_pVulkanQueueProps.size(); uiIndx++)
    {
        if (SupportFlags[uiIndx] == VK_TRUE && PresentIndx == -1)
        {
            // This gives us a back up of at least the index.
            PresentIndx = uiIndx;
        }

        if ((m_pVulkanQueueProps[uiIndx].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
        {
            // If this is the first one that supports graphics grab it
            if (GraphicsIndx == -1)
                GraphicsIndx = uiIndx;

            if (SupportFlags[uiIndx] == VK_TRUE)
            {
                // Found a queue that supports both graphics and present.  Grab it
                GraphicsIndx = uiIndx;
                PresentIndx = uiIndx;
                break;
            }
        }
    }

    // Look specifically for a transfer queue
    for (uint32_t uiIndx = 0; uiIndx < m_pVulkanQueueProps.size(); uiIndx++)
    {
        if ((m_pVulkanQueueProps[uiIndx].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0 &&
            (m_pVulkanQueueProps[uiIndx].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0 &&
            (m_pVulkanQueueProps[uiIndx].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0)
        {
            // Found a dedicated transfer queue
            LOGI("Found dedicated transfer queue (%d).", uiIndx);
            m_VulkanQueues[eTransferQueue].QueueFamilyIndex = uiIndx;
            break;
        }
    }

    // If we didn't find either queue or they are not the same then we have a problem!
    if (GraphicsIndx == -1 || PresentIndx == -1 || GraphicsIndx != PresentIndx)
    {
        LOGE("Unable to find a queue that supports both graphics and present!");
        return false;
    }

    // We now have the queue we can use
    m_VulkanQueues[eGraphicsQueue].QueueFamilyIndex = GraphicsIndx;

    // If no dedicated transfer queue just use the graphics queue
    if (m_VulkanQueues[eTransferQueue].QueueFamilyIndex == -1)
    {
        LOGI("No dedicated transfer queue found. Use the graphics queue (%d).", m_VulkanQueues[eGraphicsQueue].QueueFamilyIndex);
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::InitDataGraph()
//-----------------------------------------------------------------------------
{
    VkResult RetVal = VK_SUCCESS;

    // Look for a queue that supports data graph (note that this queue, if present, can ONLY support data graph operations)
    for (int i = 0; i < m_pVulkanQueueProps.size(); ++i)
    {
        if ((m_pVulkanQueueProps[i].queueFlags & VK_QUEUE_DATA_GRAPH_BIT_ARM) == VK_QUEUE_DATA_GRAPH_BIT_ARM)
        {
            m_VulkanQueues[eDataGraphQueue].QueueFamilyIndex = i;
            m_VulkanGraphicsQueueSupportsDataGraph = true;
            break;
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::InitCompute()
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    // See if the present queue supports compute.
    if (m_pVulkanQueueProps[m_VulkanQueues[eGraphicsQueue].QueueFamilyIndex].queueFlags & VK_QUEUE_COMPUTE_BIT)
    {
        m_VulkanGraphicsQueueSupportsCompute = true;
    }

    // Look for a queue that supports Compute (but not graphics)
    for (int i = 0; i < m_pVulkanQueueProps.size(); ++i)
    {
        if ((m_pVulkanQueueProps[i].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) == VK_QUEUE_COMPUTE_BIT)
        {
            m_VulkanQueues[eComputeQueue].QueueFamilyIndex = i;
            break;
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::InitDevice()
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    // ********************************
    // Query the available device features
    // ********************************
    PhysicalDeviceFeatures AvailableFeatures{};
    QueryPhysicalDeviceFeatures(m_VulkanGpu, AvailableFeatures);

    // Check for and enable the Vulkan Device features that we need to use.
    if (!m_ExtPortability && AvailableFeatures.Base.features.shaderStorageImageExtendedFormats == VK_FALSE)
    {
        LOGE("Vulkan physical device must support shaderStorageImageExtendedFormats");
        return false;
    }
    if (!m_ExtPortability && AvailableFeatures.Base.features.sampleRateShading == VK_FALSE)
    {
        LOGE("Vulkan physical device must support sampleRateShading");
        return false;
    }

    m_VulkanGpuFeatures = {};
    m_VulkanGpuFeatures.Base.features.geometryShader = AvailableFeatures.Base.features.geometryShader;
    m_VulkanGpuFeatures.Base.features.shaderStorageImageExtendedFormats = AvailableFeatures.Base.features.shaderStorageImageExtendedFormats;
    m_VulkanGpuFeatures.Base.features.sampleRateShading = AvailableFeatures.Base.features.sampleRateShading;
    m_VulkanGpuFeatures.Base.features.samplerAnisotropy = AvailableFeatures.Base.features.samplerAnisotropy;
    m_VulkanGpuFeatures.Base.features.shaderImageGatherExtended = AvailableFeatures.Base.features.shaderImageGatherExtended;
    m_VulkanGpuFeatures.Base.features.shaderStorageImageWriteWithoutFormat = AvailableFeatures.Base.features.shaderStorageImageWriteWithoutFormat;

    m_VulkanGpuFeatures.Base.features.multiDrawIndirect = AvailableFeatures.Base.features.multiDrawIndirect;
    m_VulkanGpuFeatures.Base.features.drawIndirectFirstInstance = AvailableFeatures.Base.features.drawIndirectFirstInstance;
    m_VulkanGpuFeatures.Base.features.depthClamp = AvailableFeatures.Base.features.depthClamp;

    m_VulkanGpuFeatures.Base.features.textureCompressionETC2 = AvailableFeatures.Base.features.textureCompressionETC2;
    m_VulkanGpuFeatures.Base.features.textureCompressionASTC_LDR = AvailableFeatures.Base.features.textureCompressionASTC_LDR;
    m_VulkanGpuFeatures.Base.features.textureCompressionBC = AvailableFeatures.Base.features.textureCompressionBC;

    m_VulkanGpuFeatures.Base.features.shaderFloat64 = AvailableFeatures.Base.features.shaderFloat64;
    m_VulkanGpuFeatures.Base.features.shaderInt64 = AvailableFeatures.Base.features.shaderInt64;
    m_VulkanGpuFeatures.Base.features.shaderInt16 = AvailableFeatures.Base.features.shaderInt16;

    // ********************************
    // Create the Device
    // ********************************
    float GraphicsPriority = 1.0f;
    float AsycComputePriority = 0.0f;   // values of 1 and 0 are guaranteed by the spec (and required), more than that needs discreteQueuePriorities check.
    float DataGraphPriority = 1.0f;
    uint32_t QueueCount = 1;

    VkDeviceQueueGlobalPriorityCreateInfoEXT DeviceQueueGlobalPriorityInfo {VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_EXT};
    VkDeviceQueueCreateInfo DeviceQueueInfoStructs[3] = {};
    DeviceQueueInfoStructs[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    DeviceQueueInfoStructs[0].flags = 0;
    DeviceQueueInfoStructs[0].queueFamilyIndex = m_VulkanQueues[eGraphicsQueue].QueueFamilyIndex;
    DeviceQueueInfoStructs[0].queueCount = 1;
    DeviceQueueInfoStructs[0].pQueuePriorities = &GraphicsPriority;
    if (m_VulkanQueues[eComputeQueue].QueueFamilyIndex >= 0)
    {
        DeviceQueueInfoStructs[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        DeviceQueueInfoStructs[1].flags = 0;
        DeviceQueueInfoStructs[1].queueFamilyIndex = m_VulkanQueues[eComputeQueue].QueueFamilyIndex;
        DeviceQueueInfoStructs[1].queueCount = 1;
        DeviceQueueInfoStructs[1].pQueuePriorities = &AsycComputePriority;

        // Also attach a VkDeviceQueueGlobalPriorityCreateInfoEXT to enable low priority compute.
        if (m_ExtGlobalPriorityAvailable)
        {
            DeviceQueueGlobalPriorityInfo.globalPriority = m_ConfigOverride.AsyncQueuePriority.value_or(VK_QUEUE_GLOBAL_PRIORITY_LOW_EXT);
            DeviceQueueInfoStructs[1].pNext = &DeviceQueueGlobalPriorityInfo;
        }
        ++QueueCount;
    }
    if (m_VulkanQueues[eDataGraphQueue].QueueFamilyIndex >= 0)
    {
        DeviceQueueInfoStructs[2].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        DeviceQueueInfoStructs[2].flags = 0;
        DeviceQueueInfoStructs[2].queueFamilyIndex = m_VulkanQueues[eDataGraphQueue].QueueFamilyIndex;
        DeviceQueueInfoStructs[2].queueCount = 1;
        DeviceQueueInfoStructs[2].pQueuePriorities = &DataGraphPriority;

        ++QueueCount;
    }

    std::vector<const char*> DeviceExtensionNames;
    DeviceExtensionNames.reserve(m_DeviceExtensions.m_Extensions.size());
    for (const auto& e : m_DeviceExtensions.m_Extensions)
        if (e.second->Status == VulkanExtensionStatus::eLoaded && e.second->LoadMode != VulkanExtensionLoadMode::eSkipRequest)
            DeviceExtensionNames.push_back(e.first.c_str());

    if (m_VulkanApiVersion >= VK_MAKE_VERSION( 1, 1, 0 ))
    {
        // Mark all Vulkan 1.1 provided (built-in) extensions as 'loaded' so they get populated for vkCreateDevice
        for (auto& extPair : m_Vulkan11ProvidedExtensions.m_Extensions)
        {
            assert( extPair.second->Status == VulkanExtensionStatus::eRequired );
            extPair.second->Status = VulkanExtensionStatus::eLoaded;
        }
    }

    VkDeviceCreateInfo DeviceInfoStruct = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    DeviceInfoStruct.pNext = &m_VulkanGpuFeatures.Base;             // Hardcoded features enabled through the 'pNext chain'
    DeviceInfoStruct.flags = 0;
    DeviceInfoStruct.queueCreateInfoCount = QueueCount;
    DeviceInfoStruct.pQueueCreateInfos = DeviceQueueInfoStructs;
    DeviceInfoStruct.enabledLayerCount = (uint32_t) m_pDeviceLayerNames.size();
    DeviceInfoStruct.ppEnabledLayerNames = m_pDeviceLayerNames.data();
    DeviceInfoStruct.enabledExtensionCount = (uint32_t) DeviceExtensionNames.size();
    DeviceInfoStruct.ppEnabledExtensionNames = DeviceExtensionNames.data();
    m_DeviceCreateInfoExtensions.PushExtensions(&DeviceInfoStruct);   // Push registered features into the 'pNext chain'

    LOGI("Creating Vulkan Device...");
    for(uint32_t i = 0;i<QueueCount; ++i)
    {
        LOGI("    DeviceQueueInfo[%u].queueFamilyIndex: %d", i, DeviceQueueInfoStructs[i].queueFamilyIndex);
        LOGI("    DeviceQueueInfo[%u].queuePriority: %0.2f", i, DeviceQueueInfoStructs[i].pQueuePriorities[0]);
        if (m_ExtGlobalPriorityAvailable && DeviceQueueInfoStructs[i].pNext)
            LOGI("    DeviceQueueInfo[%u].pNext.DeviceQueueGlobalPriorityInfo.globalPriority: %d", i, DeviceQueueGlobalPriorityInfo.globalPriority);
        else if (i>0)
        {
            LOGI( "    %s not found (cannot set global priority for LPAC)", VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME );
        }
    }
    LOGI("    DeviceInfo.DeviceLayers: %d", DeviceInfoStruct.enabledLayerCount);
    for (uint32_t Which = 0; Which < DeviceInfoStruct.enabledLayerCount; Which++)
    {
        LOGI("        %d: %s", Which, DeviceInfoStruct.ppEnabledLayerNames[Which]);
    }

    LOGI("    DeviceInfo.DeviceExtensions: %d", DeviceInfoStruct.enabledExtensionCount);
    for (uint32_t Which = 0; Which < DeviceInfoStruct.enabledExtensionCount; Which++)
    {
        LOGI("        %d: %s", Which, DeviceInfoStruct.ppEnabledExtensionNames[Which]);
    }

    retVal = vkCreateDevice(m_VulkanGpu, &DeviceInfoStruct, nullptr, &m_VulkanDevice);
    if (!CheckVkError("vkCreateDevice()", retVal))
    {
        return false;
    }

    volkLoadDevice(m_VulkanDevice);

    // Pop the extensions back off the VkPhysicalDeviceFeatures2 chain (most registered extensions will do nothing here)
    m_DeviceCreateInfoExtensions.PopExtensions(&DeviceInfoStruct);

    // ********************************
    // Get (Device) Function Pointers
    // ********************************
    // These functions are part of the device so we must query for them using
    // "vkGetDeviceProcAddr" once the device has been created
    if (!fpGetDeviceProcAddr)
    {
        fpGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr(m_VulkanInstance, "vkGetDeviceProcAddr");
        if (fpGetDeviceProcAddr == nullptr)
        {
            LOGE("Unable to get function pointer from instance: vkGetInstanceProcAddr");
            return false;
        }
    }

    // Call any extensions that registered for device function pointer lookups.
    {
        VulkanDeviceFunctionPointerLookup fn { m_VulkanDevice, fpGetDeviceProcAddr };
        m_VulkanDeviceFunctionPointerLookupExtensions.PushExtensions( &fn );
    }

    // ********************************
    // Call the PostLoad for all device extensions.
    // This is done after the LookupFunctionPointers calls (where applicable) but maybe should go before?
    // ********************************
    for (auto& extension : m_DeviceExtensions.m_Extensions)
    {
        if (extension.second->Status == VulkanExtensionStatus::eLoaded)
            extension.second->PostLoad();
    }

    // ********************************
    // Create the Device Queue
    // ********************************
    vkGetDeviceQueue(m_VulkanDevice, m_VulkanQueues[eGraphicsQueue].QueueFamilyIndex, 0, &m_VulkanQueues[eGraphicsQueue].Queue);

    // ********************************
    // Create the Async Compute Device Queue
    // ********************************
    if (m_VulkanQueues[eComputeQueue].QueueFamilyIndex >= 0)
    {
        vkGetDeviceQueue(m_VulkanDevice, m_VulkanQueues[eComputeQueue].QueueFamilyIndex, 0, &m_VulkanQueues[eComputeQueue].Queue);
    }

    // ********************************
    // Create the Data Graph Device Queue
    // ********************************
    if (m_VulkanQueues[eDataGraphQueue].QueueFamilyIndex >= 0)
    {
        vkGetDeviceQueue(m_VulkanDevice, m_VulkanQueues[eDataGraphQueue].QueueFamilyIndex, 0, &m_VulkanQueues[eDataGraphQueue].Queue);
    }

    // ********************************
    // Get Supported Formats
    // ********************************
    uint32_t NumFormats;
    retVal = m_ExtKhrSurface->m_vkGetPhysicalDeviceSurfaceFormatsKHR(m_VulkanGpu, m_VulkanSurface, &NumFormats, nullptr);
    if (!CheckVkError("vkGetPhysicalDeviceSurfaceFormatsKHR()", retVal))
    {
        return false;
    }

    LOGI("Found %d Vulkan Supported Formats", NumFormats);

    if (NumFormats > 0)
    {
        // Allocate memory for the formats...
        m_SurfaceFormats.clear();
        m_SurfaceFormats.reserve( NumFormats );
        std::vector<VkSurfaceFormatKHR> vkSurfaceFormats;
        vkSurfaceFormats.resize( NumFormats );

        // ... then read the structures from the driver
        retVal = m_ExtKhrSurface->m_vkGetPhysicalDeviceSurfaceFormatsKHR(m_VulkanGpu, m_VulkanSurface, &NumFormats, vkSurfaceFormats.data());
        if (!CheckVkError("vkGetPhysicalDeviceSurfaceFormatsKHR()", retVal))
        {
            return false;
        }
        if (NumFormats == 0)
        {
            return false;
        }

        for (uint32_t uiIndx = 0; uiIndx < NumFormats; uiIndx++)
        {
            const auto& surfaceFormat = vkSurfaceFormats[uiIndx];
            LOGI("    %d: (%d) %s", uiIndx, surfaceFormat.format, VulkanFormatString(surfaceFormat.format));
            LOGI("    %d: (%d) %s", uiIndx, surfaceFormat.colorSpace, VulkanColorSpaceString(surfaceFormat.colorSpace));
        }
        for (const auto& vk : vkSurfaceFormats)
        {
            m_SurfaceFormats.push_back( {VkToTextureFormat( vk.format ), vk.colorSpace} );
        }

        // If the surface come back as VK_FORMAT_UNDEFINED and there is only one then
        // it doesn't have a preferred format
        if (NumFormats == 1 && m_SurfaceFormats[0].format == TextureFormat::UNDEFINED)
        {
            m_SurfaceFormat = TextureFormat::B8G8R8A8_UNORM;
        }
        else
        {
            // Just take the first one by default
            m_SurfaceFormat = m_SurfaceFormats[0].format;

            // See if we have a srgb format (and use it if we do)
            for(const auto& f: m_SurfaceFormats)
            {
                if (FormatIsSrgb(f.format))
                {
                    m_SurfaceFormat = f.format;
                    break;
                }
            }
        }

        // Taking the first colorspace
        m_SurfaceColorSpace = m_SurfaceFormats[0].colorSpace;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::InitSyncElements()
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    // We have semaphores for rendering and backbuffer signalling.
    VkSemaphoreCreateInfo semCreateInfo {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semCreateInfo.flags = 0;

    m_RenderCompleteSemaphore = VK_NULL_HANDLE;
    retVal = vkCreateSemaphore(m_VulkanDevice, &semCreateInfo, nullptr, &m_RenderCompleteSemaphore);
    if (!CheckVkError("vkCreateSemaphore()", retVal))
    {
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::InitCommandPools()
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    // ********************************
    // Create the command pool
    // ********************************
    LOGI("Creating Vulkan Command Pool...");

    const VkCommandPoolCreateInfo CmdPoolInfoStruct{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = (uint32_t)m_VulkanQueues[eGraphicsQueue].QueueFamilyIndex
    };

    retVal = vkCreateCommandPool(m_VulkanDevice, &CmdPoolInfoStruct, nullptr, &m_VulkanQueues[eGraphicsQueue].CommandPool);
    if (!CheckVkError("vkCreateCommandPool()", retVal))
    {
        return false;
    }

    // Allocate a command pool for (async) Compute.
    if (m_VulkanQueues[eComputeQueue].Queue)
    {
        const VkCommandPoolCreateInfo CmdPoolInfoStruct{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = (uint32_t)m_VulkanQueues[eComputeQueue].QueueFamilyIndex
        };

        retVal = vkCreateCommandPool(m_VulkanDevice, &CmdPoolInfoStruct, nullptr, &m_VulkanQueues[eComputeQueue].CommandPool);
        if (!CheckVkError("vkCreateCommandPool()", retVal))
        {
            return false;
        }
    }

    // Allocate a command pool for Data Graph.
    if (m_VulkanGraphicsQueueSupportsDataGraph && m_VulkanQueues[eDataGraphQueue].Queue)
    {
        VkDataGraphProcessingEngineCreateInfoARM engineInfo = {};
        engineInfo.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PROCESSING_ENGINE_CREATE_INFO_ARM;
        engineInfo.pNext = NULL;
        engineInfo.processingEngineCount = 1;
        engineInfo.pProcessingEngines = &m_VulkanDataGraphProcessingEngine;

        VkCommandPoolCreateInfo CmdPoolInfoStruct{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        CmdPoolInfoStruct.pNext = &engineInfo;
        CmdPoolInfoStruct.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        CmdPoolInfoStruct.queueFamilyIndex = m_VulkanQueues[eDataGraphQueue].QueueFamilyIndex;

        retVal = vkCreateCommandPool(m_VulkanDevice, &CmdPoolInfoStruct, nullptr, &m_VulkanQueues[eDataGraphQueue].CommandPool);
        if (!CheckVkError("vkCreateCommandPool()", retVal))
        {
            return false;
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::InitMemoryManager()
//-----------------------------------------------------------------------------
{
    return m_MemoryManager.Initialize(m_VulkanGpu, m_VulkanDevice, m_VulkanInstance, HasLoadedVulkanDeviceExtension("VK_KHR_buffer_device_address"));
}

//-----------------------------------------------------------------------------
bool Vulkan::InitPipelineCache()
//-----------------------------------------------------------------------------
{
    assert(m_PipelineCache == VK_NULL_HANDLE);
    VkPipelineCacheCreateInfo CreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
    auto retVal = vkCreatePipelineCache( m_VulkanDevice, &CreateInfo, nullptr, &m_PipelineCache );
    if (!CheckVkError( "vkCreatePipelineCache()", retVal ))
    {
        return false;
    }
    return true;
}


//-----------------------------------------------------------------------------
bool Vulkan::QuerySurfaceCapabilities(VkSurfaceCapabilitiesKHR& outVulkanSurfaceCaps)
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    if (m_ExtSurfaceCapabilities2 && m_ExtSurfaceCapabilities2->Status == VulkanExtensionStatus::eLoaded)
    {
        // Have the extension for vkGetPhysicalDeviceSurfaceCapabilities2KHR.
        // Ideally we could query the VkHdrMetadataEXT for this surface and use that as part of colormapping/output.
        // Doesnt seem to be supported for Qualcomm or Nvidia drivers ('undocumented' feature on AMD)
        const VkPhysicalDeviceSurfaceInfo2KHR SurfaceInfo {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
            .surface = m_VulkanSurface
        };
        VkSurfaceCapabilities2KHR SurfaceCapabilities { VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR };

        retVal = m_ExtSurfaceCapabilities2->m_vkGetPhysicalDeviceSurfaceCapabilities2KHR(m_VulkanGpu, &SurfaceInfo, &SurfaceCapabilities);
        if (!CheckVkError("vkGetPhysicalDeviceSurfaceCapabilities2KHR()", retVal))
        {
            return false;
        }

        outVulkanSurfaceCaps = SurfaceCapabilities.surfaceCapabilities;
    }
    else
    {
        retVal = m_ExtKhrSurface->m_vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_VulkanGpu, m_VulkanSurface, &outVulkanSurfaceCaps);
        if (!CheckVkError("vkGetPhysicalDeviceSurfaceCapabilitiesKHR()", retVal))
        {
            return false;
        }
    }
    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::QuerySurfaceCapabilities()
//-----------------------------------------------------------------------------
{
    if (!QuerySurfaceCapabilities(m_VulkanSurfaceCaps))
        return false;

    LOGI("Vulkan Surface Capabilities:");
    LOGI("    minImageCount: %d", m_VulkanSurfaceCaps.minImageCount);
    LOGI("    maxImageCount: %d", m_VulkanSurfaceCaps.maxImageCount);
    LOGI("    currentExtent: %dx%d", m_VulkanSurfaceCaps.currentExtent.width, m_VulkanSurfaceCaps.currentExtent.height);
    LOGI("    minImageExtent: %dx%d", m_VulkanSurfaceCaps.minImageExtent.width, m_VulkanSurfaceCaps.minImageExtent.height);
    LOGI("    maxImageExtent: %dx%d", m_VulkanSurfaceCaps.maxImageExtent.width, m_VulkanSurfaceCaps.maxImageExtent.height);
    LOGI("    maxImageArrayLayers: %d", m_VulkanSurfaceCaps.maxImageArrayLayers);
    LOGI("    supportedTransforms: 0x%08x", m_VulkanSurfaceCaps.supportedTransforms);
    LOGI("    currentTransform: 0x%08x", m_VulkanSurfaceCaps.currentTransform);

    if (m_VulkanSurfaceCaps.currentTransform & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
        LOGI("        VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR");
    if (m_VulkanSurfaceCaps.currentTransform & VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR)
        LOGI("        VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR");
    if (m_VulkanSurfaceCaps.currentTransform & VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR)
        LOGI("        VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR");
    if (m_VulkanSurfaceCaps.currentTransform & VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR)
        LOGI("        VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR");
    if (m_VulkanSurfaceCaps.currentTransform & VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_BIT_KHR)
        LOGI("        VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_BIT_KHR");
    if (m_VulkanSurfaceCaps.currentTransform & VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR)
        LOGI("        VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_90_BIT_KHR");
    if (m_VulkanSurfaceCaps.currentTransform & VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR)
        LOGI("        VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_180_BIT_KHR");
    if (m_VulkanSurfaceCaps.currentTransform & VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR)
        LOGI("        VK_SURFACE_TRANSFORM_HORIZONTAL_MIRROR_ROTATE_270_BIT_KHR");

    LOGI("    supportedCompositeAlpha: 0x%08x", m_VulkanSurfaceCaps.supportedCompositeAlpha);
    LOGI("    supportedUsageFlags: 0x%08x", m_VulkanSurfaceCaps.supportedUsageFlags);

    return true;
}


//-----------------------------------------------------------------------------
bool Vulkan::InitSwapChain()
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    // ********************************
    // Presentation Modes
    // ********************************
    uint32_t numPresentModes;

    retVal = m_ExtKhrSurface->m_vkGetPhysicalDeviceSurfacePresentModesKHR( m_VulkanGpu, m_VulkanSurface, &numPresentModes, nullptr );
    if (!CheckVkError( "vkGetPhysicalDeviceSurfacePresentModesKHR()", retVal ))
    {
        return false;
    }

    std::vector<VkPresentModeKHR> presentModes;
    presentModes.resize( numPresentModes );

    retVal = m_ExtKhrSurface->m_vkGetPhysicalDeviceSurfacePresentModesKHR( m_VulkanGpu, m_VulkanSurface, &numPresentModes, presentModes.data() );
    if (!CheckVkError( "vkGetPhysicalDeviceSurfacePresentModesKHR()", retVal ))
    {
        return false;
    }

    LOGI( "Supported Present Modes:" );
    for (uint32_t whichMode = 0; whichMode < numPresentModes; whichMode++)
    {
        switch (presentModes[whichMode])
        {
        case VK_PRESENT_MODE_IMMEDIATE_KHR:
            LOGI( "    VK_PRESENT_MODE_IMMEDIATE_KHR" );
            break;
        case VK_PRESENT_MODE_MAILBOX_KHR:
            LOGI( "    VK_PRESENT_MODE_MAILBOX_KHR" );
            break;
        case VK_PRESENT_MODE_FIFO_KHR:
            LOGI( "    VK_PRESENT_MODE_FIFO_KHR" );
            break;
        case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
            LOGI( "    VK_PRESENT_MODE_FIFO_RELAXED_KHR" );
            break;
        case VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR:
            LOGI( "    VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR" );
            break;
        case VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR:
            LOGI( "    VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR" );
            break;
        default:
            LOGI( "    Unknown! (%d)", presentModes[whichMode] );
            break;
        }
    }

    // FIFO: is the same as VSync mode.  It also makes the
    // swapchain images come back in order (0, 1, 2, 3, 0, 1, ...)
    // Mailbox: is similar to double buffering. One is displayed and one is queued.
    // The problem with mailbox is that the swapchain order that is returned is not
    // sequential if there are more than two images
    // Immediate: Requests are applied immediately and tearing may occur.
    // Unless there is a vsync, the GPU is rendering frames that are not displayed.
    VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;   // FIFO should always be available as a fall-back

    for (uint32_t uiIndx = 0; uiIndx < numPresentModes; uiIndx++)
    {
        if (m_ConfigOverride.PresentMode.has_value() && m_ConfigOverride.PresentMode.value() == presentModes[uiIndx])
        {
            // Found the app selected presentation mode!
            swapchainPresentMode = m_ConfigOverride.PresentMode.value();
            break;
        }
        else if (presentModes[uiIndx] == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            // Mailbox is our 1st choice if we dont have (or can't find) an app selected override.
            swapchainPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
        }
        else if ((swapchainPresentMode != VK_PRESENT_MODE_MAILBOX_KHR) && (presentModes[uiIndx] == VK_PRESENT_MODE_IMMEDIATE_KHR))
        {
            // Immediate is our 2nd choice if we dont have (or can't find) an app selected override.
            swapchainPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        }
    }

    switch (swapchainPresentMode)
    {
    case VK_PRESENT_MODE_IMMEDIATE_KHR:
        LOGI( "Setting SwapChain Present Mode: VK_PRESENT_MODE_IMMEDIATE_KHR" );
        break;
    case VK_PRESENT_MODE_MAILBOX_KHR:
        LOGI( "Setting SwapChain Present Mode: VK_PRESENT_MODE_MAILBOX_KHR" );
        break;
    case VK_PRESENT_MODE_FIFO_KHR:
        LOGI( "Setting SwapChain Present Mode: VK_PRESENT_MODE_FIFO_KHR" );
        break;
    case VK_PRESENT_MODE_FIFO_RELAXED_KHR:
        LOGI( "Setting SwapChain Present Mode: VK_PRESENT_MODE_FIFO_RELAXED_KHR" );
        break;
    case VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR:
        LOGI( "Setting SwapChain Present Mode: VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR" );
        break;
    case VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR:
        LOGI( "Setting SwapChain Present Mode: VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR" );
        break;
    default:
        LOGI( "Setting SwapChain Present Mode: Unknown! (%d)", swapchainPresentMode );
        break;
    }

    //
    // Swapchain Extent (grab this again as the swapchain rotate may have swapped the dimensions)
    //
    VkExtent2D swapchainExtent;

    // If the surface capabilities comes back with current extent set to -1 it means
    // the size is undefined and we can set to what we want.
    // If it comes back as something it had better match what we set it to.
    if (m_VulkanSurfaceCaps.currentExtent.width == (uint32_t)-1 || m_VulkanSurfaceCaps.currentExtent.height == (uint32_t)-1 ||
         m_VulkanSurfaceCaps.currentExtent.width == 0 || m_VulkanSurfaceCaps.currentExtent.height == 0)
    {
        swapchainExtent.width = m_SurfaceWidth;
        swapchainExtent.height = m_SurfaceHeight;
        LOGI( "Surface Caps returned an extent with at least one dimension invalid!  Setting to %dx%d", swapchainExtent.width, swapchainExtent.height );
    }
    else
    {
        swapchainExtent = m_VulkanSurfaceCaps.currentExtent;
    }

    // Take the surface width from the Vulkan surface
    m_SurfaceWidth = swapchainExtent.width;
    m_SurfaceHeight = swapchainExtent.height;

    LOGI( "Vulkan surface size: %dx%d", m_SurfaceWidth, m_SurfaceHeight );

    uint32_t desiredSwapchainImages = NUM_VULKAN_BUFFERS;
    if (swapchainPresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
    {
        // In mailbox mode we request the number of buffers the application needs and Vulkan will want one or two more!  https://github.com/KhronosGroup/Vulkan-Docs/issues/909
        desiredSwapchainImages = std::max( (uint32_t)3, m_VulkanSurfaceCaps.minImageCount + 1/*when running multithreaded and mailbox present we need even more swapchains on Android!  However many Android needs (which includes one we can be preparing on the CPU) plus one if we have a thread simultaneously submitting the last frames commands*/ );
    }

    if (m_VulkanSurfaceCaps.minImageCount > desiredSwapchainImages)
    {
        LOGE( "****************************************" );
        LOGE( "Minimum image count (%d) is greater than NUM_VULKAN_BUFFERS (%d)!", m_VulkanSurfaceCaps.minImageCount, NUM_VULKAN_BUFFERS );
        LOGE( "You will have all sorts of problems!" );
        LOGE( "****************************************" );
        assert( 0 );
    }

    if ((m_VulkanSurfaceCaps.maxImageCount > 0) && (desiredSwapchainImages > m_VulkanSurfaceCaps.maxImageCount))
    {
        LOGE( "****************************************" );
        LOGE( "We desired %d swapchain images but surface limits us to %d!", desiredSwapchainImages, m_VulkanSurfaceCaps.maxImageCount );
        LOGE( "You will have all sorts of problems!" );
        LOGE( "****************************************" );
        desiredSwapchainImages = m_VulkanSurfaceCaps.maxImageCount;
    }

    // If we have the render pass transform extension (and want to use it), then make the pre-transform match the current transform and enable flag to setup renderpasses/commandbuffers appropriately.
    // Otherwise we want to leave the pretransform alone and let the hardware do the rotation, if supported.
    {
        if (m_ExtRenderPassTransformAvailable && m_LayerKhronosValidationAvailable)
        {
            LOGE( "Disabling QCOM_Render_Pass_Transform as it is not supported while Validation layers are enabled" );
            m_ExtRenderPassTransformAvailable = false;
        }

        VkSurfaceTransformFlagsKHR desiredPreTransform;
        LOGI( "QCOM_Render_Pass_Transform - ExtRenderPassTransformAvailable=%s UseRenderPassTransform=%s", m_ExtRenderPassTransformAvailable ? "True" : "False", m_UseRenderPassTransform ? "True" : "False" );
        if (m_UseRenderPassTransform && m_ExtRenderPassTransformAvailable)
        {
            desiredPreTransform = m_VulkanSurfaceCaps.currentTransform;
            m_ExtRenderPassTransformEnabled = (m_VulkanSurfaceCaps.currentTransform == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR) || (m_VulkanSurfaceCaps.currentTransform == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR);
        }
        else if (m_UsePreTransform)
        {
            desiredPreTransform = m_VulkanSurfaceCaps.currentTransform;
        }
        else
        {
            m_ExtRenderPassTransformEnabled = false;
            if (m_VulkanSurfaceCaps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
            {
                desiredPreTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
            }
            else
            {
                desiredPreTransform = m_VulkanSurfaceCaps.currentTransform;
            }
        }
        if ((desiredPreTransform & VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR)
             || (desiredPreTransform & VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR))
        {
            LOGI( "Vulkan surface extents swapped (width <-> height)" );
            std::swap( swapchainExtent.width, swapchainExtent.height );
        }

        m_SwapchainPreTransform = (VkSurfaceTransformFlagBitsKHR)desiredPreTransform;
    }

    // ********************************
    // Swapchain
    // ********************************
    VkSwapchainCreateInfoKHR SwapchainInfo{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .flags = 0,
        .surface = m_VulkanSurface,
        .minImageCount = desiredSwapchainImages,
        .imageFormat = TextureFormatToVk( m_SurfaceFormat ),
        .imageColorSpace = m_SurfaceColorSpace,
        .imageExtent = swapchainExtent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,      // Image usage changes if we can blit it
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .preTransform = m_SwapchainPreTransform,
        .compositeAlpha = GetBestVulkanCompositeAlpha(),        // Get the supported composite alpha flag
        .presentMode = swapchainPresentMode,
        .clipped = true,
        .oldSwapchain = m_VulkanSwapchain,
    };
    const VkFormatProperties& FormatProps = GetFormatProperties(TextureFormatToVk(m_SurfaceFormat));
    if ((FormatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT))
    {
        SwapchainInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }
    if ((FormatProps.optimalTilingFeatures & (VK_FORMAT_FEATURE_BLIT_SRC_BIT|VK_FORMAT_FEATURE_TRANSFER_SRC_BIT_KHR)))
    {
        SwapchainInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;    // screenshottable
    }

    LOGI("Trying to get %d swapchain images...", desiredSwapchainImages);
#if defined (OS_WINDOWS) || defined (OS_LINUX)
    retVal = m_ExtSwapchain->m_vkCreateSwapchainKHR(m_VulkanDevice, &SwapchainInfo, nullptr, &m_VulkanSwapchain);
#elif defined (OS_ANDROID)
    retVal = vkCreateSwapchainKHR(m_VulkanDevice, &SwapchainInfo, nullptr, &m_VulkanSwapchain);
#endif // defined (OS_WINDOWS|OS_ANDROID|OS_LINUX)
    if (!CheckVkError("vkCreateSwapchainKHR()", retVal))
    {
        return false;
    }

#if defined (OS_WINDOWS) || defined (OS_LINUX)
    retVal = m_ExtSwapchain->m_vkGetSwapchainImagesKHR(m_VulkanDevice, m_VulkanSwapchain, &m_SwapchainImageCount, nullptr);
#elif defined (OS_ANDROID)
    retVal = vkGetSwapchainImagesKHR(m_VulkanDevice, m_VulkanSwapchain, &m_SwapchainImageCount, nullptr);
#else
#error "Unsupported platform"
#endif // defined (OS_WINDOWS|OS_ANDROID|OS_LINUX)
    if (!CheckVkError("vkGetSwapchainImagesKHR(nullptr)", retVal))
    {        return false;
    }

    LOGI("SwapChain has %d images", m_SwapchainImageCount);
    if (m_SwapchainImageCount == 0)
    {
        LOGE("    Initialization Failed!  Could not get swapchain images");
        return false;
    }
    m_SwapchainCurrentIndx = 0;

    if ((m_VulkanSurfaceCaps.maxImageCount > 0) && (m_SwapchainImageCount > m_VulkanSurfaceCaps.maxImageCount))
    {
        LOGE("****************************************");
        LOGE("We asked for %d swapchain images but got back %d!", desiredSwapchainImages, m_SwapchainImageCount);
        LOGE("****************************************");
    }

    if (m_SwapchainImageCount > NUM_VULKAN_BUFFERS)
    {
        LOGE("****************************************");
        LOGE("vkGetSwapchainImagesKHR() returned %d swapchain images, but NUM_VULKAN_BUFFERS = %d!", m_SwapchainImageCount, NUM_VULKAN_BUFFERS);
        LOGE("Application cannot proceed.  Consider forcing a different swap-mode.");
        LOGE("****************************************");

        // Later: I can't just set m_SwapchainImageCount = NUM_VULKAN_BUFFERS!
        // I get error "Vulkan Error (VK_INCOMPLETE) from vkGetSwapchainImagesKHR()"
    }

    std::vector<VkImage> swapchainImages;
    swapchainImages.resize(m_SwapchainImageCount);

#if defined (OS_WINDOWS) || defined (OS_LINUX)
    retVal = m_ExtSwapchain->m_vkGetSwapchainImagesKHR(m_VulkanDevice, m_VulkanSwapchain, &m_SwapchainImageCount, swapchainImages.data());
#elif defined (OS_ANDROID)
    retVal = vkGetSwapchainImagesKHR(m_VulkanDevice, m_VulkanSwapchain, &m_SwapchainImageCount, swapchainImages.data());
#endif // defined (OS_WINDOWS|OS_ANDROID|OS_LINUX)
    if (!CheckVkError("vkGetSwapchainImagesKHR()", retVal))
    {
        return false;
    }

    // ********************************
    // Setup Depth
    // ********************************
    m_SwapchainDepth.format = m_ConfigOverride.SwapchainDepthFormat.value_or( GetBestSurfaceDepthFormat() );

    if (m_SwapchainDepth.format != TextureFormat::UNDEFINED)
    {
        const VkFormat vkSwapchainDepthFormat = TextureFormatToVk( m_SwapchainDepth.format );
        const VkImageCreateInfo imageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = vkSwapchainDepthFormat,
            .extent {
                .width = swapchainExtent.width,
                .height = swapchainExtent.height,
                .depth = 1 }, // Spec says for VK_IMAGE_TYPE_2D depth must be 1
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
        };

        m_SwapchainDepth.image = m_MemoryManager.CreateImage( imageInfo, MemoryUsage::GpuExclusive );
        if (!m_SwapchainDepth.image)
        {
            LOGE( "Error creating swapchain back buffer depth image" );
            return false;
        }

        // Create the view for this image
        const VkImageViewCreateInfo ImageViewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .flags = 0,
            .image = m_SwapchainDepth.image.GetVkBuffer(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = vkSwapchainDepthFormat,
            .components {
                .r = VK_COMPONENT_SWIZZLE_R,
                .g = VK_COMPONENT_SWIZZLE_G,
                .b = VK_COMPONENT_SWIZZLE_B,
                .a = VK_COMPONENT_SWIZZLE_A
            },
            .subresourceRange {
                .aspectMask = (VkImageAspectFlags)(FormatHasStencil( m_SwapchainDepth.format ) ? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) : VK_IMAGE_ASPECT_DEPTH_BIT),
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
        };
        retVal = vkCreateImageView( m_VulkanDevice, &ImageViewInfo, nullptr, &m_SwapchainDepth.view );
        if (!CheckVkError( "vkCreateImageView()", retVal ))
        {
            return false;
        }
    }

    assert(m_SwapchainBuffers.empty());
    m_SwapchainBuffers.reserve(m_SwapchainImageCount);

    for (uint32_t uiIndx = 0; uiIndx < m_SwapchainImageCount; uiIndx++)
    {
        auto& swapchainBuffer = m_SwapchainBuffers.emplace_back( SwapchainBuffers{});
        swapchainBuffer.image = swapchainImages[uiIndx];

        // Render loop will expect image to have been used before and in
        // VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        // layout and will change to COLOR_ATTACHMENT_OPTIMAL, so init the image
        // to that state

        const VkImageViewCreateInfo ImageViewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .flags = 0,
            .image = swapchainBuffer.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = TextureFormatToVk( m_SurfaceFormat ),
            .components {
                .r = VK_COMPONENT_SWIZZLE_R,
                .g = VK_COMPONENT_SWIZZLE_G,
                .b = VK_COMPONENT_SWIZZLE_B,
                .a = VK_COMPONENT_SWIZZLE_A },
            .subresourceRange {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        retVal = vkCreateImageView( m_VulkanDevice, &ImageViewInfo, nullptr, &swapchainBuffer.view );
        if (!CheckVkError( "vkCreateImageView()", retVal ))
        {
            return false;
        }

        // Create the wait fences, one per swapchain image
        const VkFenceCreateInfo fenceInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT
        };
        retVal = vkCreateFence( m_VulkanDevice, &fenceInfo, nullptr, &swapchainBuffer.fence );
        if (!CheckVkError( "vkCreateFence()", retVal ))
        {
            return false;
        }

        // Create semaphores for backbuffer signalling, one per swapchain.
        const VkSemaphoreCreateInfo semCreateInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .flags = 0
        };
        retVal = vkCreateSemaphore(m_VulkanDevice, &semCreateInfo, nullptr, &swapchainBuffer.semaphore);
        if (!CheckVkError("vkCreateSemaphore()", retVal))
        {
            return false;
        }

    }   // Each swapchain image

    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::InitSwapchainRenderPass()
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    // The renderpass defines the attachments to the framebuffer object that gets
    // used in the pipelines. We have two attachments, the colour buffer, and the
    // depth buffer. The operations and layouts are set to defaults for this type
    // of attachment.
    uint32_t numAttachmentDescriptions = 0;
    std::array<VkAttachmentDescription, 2> attachmentDescriptions;
    attachmentDescriptions.fill( {} );
    attachmentDescriptions[0].flags                 = 0;
    attachmentDescriptions[0].format                = TextureFormatToVk( m_SurfaceFormat );
    attachmentDescriptions[0].samples               = VK_SAMPLE_COUNT_1_BIT;
    attachmentDescriptions[0].loadOp                = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachmentDescriptions[0].storeOp               = VK_ATTACHMENT_STORE_OP_STORE;
    attachmentDescriptions[0].stencilLoadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescriptions[0].stencilStoreOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachmentDescriptions[0].initialLayout         = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    attachmentDescriptions[0].finalLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    ++numAttachmentDescriptions;

    const bool HasDepth = m_SwapchainDepth.format != TextureFormat::UNDEFINED;

    if( HasDepth )
    {
        attachmentDescriptions[1].flags             = 0;
        attachmentDescriptions[1].format            = TextureFormatToVk( m_SwapchainDepth.format );
        attachmentDescriptions[1].samples           = VK_SAMPLE_COUNT_1_BIT;
        attachmentDescriptions[1].loadOp            = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachmentDescriptions[1].storeOp           = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescriptions[1].stencilLoadOp     = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDescriptions[1].stencilStoreOp    = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescriptions[1].initialLayout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachmentDescriptions[1].finalLayout       = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        ++numAttachmentDescriptions;
    }

    // We have references to the attachment offsets, stating the layout type.
    const VkAttachmentReference colorReference {
        .attachment                 = 0,
        .layout                     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };
    const VkAttachmentReference depthReference{
        .attachment                 = 1,
        .layout                     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    };

    // There can be multiple subpasses in a renderpass, but this example has only one.
    // We set the color and depth references at the grahics bind point in the pipeline.
    const VkSubpassDescription subpassDescription {
        .flags                      = 0,
        .pipelineBindPoint          = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount       = 0,
        .pInputAttachments          = nullptr,
        .colorAttachmentCount       = 1,
        .pColorAttachments          = &colorReference,
        .pResolveAttachments        = nullptr,
        .pDepthStencilAttachment    = HasDepth ? &depthReference : nullptr,
        .preserveAttachmentCount    = 0,
        .pPreserveAttachments       = nullptr
    };

    // Dependencies
    m_SwapchainRenderPassDependencies.fill( {} );
    // m_SwapchainRenderPassDependencies[0] is an acquire dependency
    // m_SwapchainRenderPassDependencies[1] is a present dependency
    // We use subpass dependencies to define the color image layout transitions rather than
    // explicitly do them in the command buffer, as it is more efficient to do it this way.

    // Before we can use the back buffer from the swapchain, we must change the
    // image layout from the PRESENT mode to the COLOR_ATTACHMENT mode.
    m_SwapchainRenderPassDependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
    m_SwapchainRenderPassDependencies[0].dstSubpass      = 0;
    m_SwapchainRenderPassDependencies[0].srcStageMask    = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    m_SwapchainRenderPassDependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    m_SwapchainRenderPassDependencies[0].srcAccessMask   = 0;
    m_SwapchainRenderPassDependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    m_SwapchainRenderPassDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // After writing to the back buffer of the swapchain, we need to change the
    // image layout from the COLOR_ATTACHMENT mode to the PRESENT mode which
    // is optimal for sending to the screen for users to see the completed rendering.
    m_SwapchainRenderPassDependencies[1].srcSubpass      = 0;
    m_SwapchainRenderPassDependencies[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
    m_SwapchainRenderPassDependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    m_SwapchainRenderPassDependencies[1].dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    m_SwapchainRenderPassDependencies[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    m_SwapchainRenderPassDependencies[1].dstAccessMask   = 0;
    m_SwapchainRenderPassDependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // The renderpass itself is created with the number of subpasses, and the
    // list of attachments which those subpasses can reference.
    fvk::VkRenderPassCreateInfo renderPassCreateInfo{{
        .flags = (m_ExtRenderPassTransformEnabled) ? VK_RENDER_PASS_CREATE_TRANSFORM_BIT_QCOM : (VkRenderPassCreateFlags)0,
        .attachmentCount = numAttachmentDescriptions,
        .pAttachments = attachmentDescriptions.data(),
        .subpassCount = 1,
        .pSubpasses = &subpassDescription,
        .dependencyCount = (uint32_t)m_SwapchainRenderPassDependencies.size(),
        .pDependencies = m_SwapchainRenderPassDependencies.data(),
    }};

    // Create renderpass
    return CreateRenderPass( *renderPassCreateInfo, m_SwapchainRenderPass );
}

//-----------------------------------------------------------------------------
bool Vulkan::Create2SubpassRenderPass(const std::span<const TextureFormat> InternalColorFormats, const std::span<const TextureFormat> OutputColorFormats, TextureFormat InternalDepthFormat, const std::span<const Msaa> InternalMsaa/*two passes*/, Msaa OutputMsaa, RenderPass& rRenderPass/*out*/)
//-----------------------------------------------------------------------------
{
    assert(!rRenderPass);                       // check not already allocated
    assert(!InternalColorFormats.empty());      // not supporting a depth only pass
    assert(!OutputColorFormats.empty());
    assert( InternalMsaa.size() == 2 );

    //const bool NeedsResolve = InternalMsaa != OutputMsaa;

    // Color attachments and Depth attachment
    std::vector<VkAttachmentDescription2> PassAttachDescs;
    PassAttachDescs.reserve(InternalColorFormats.size() + OutputColorFormats.size() + 2);

    // Each subpass needs a reference to its attachments
    std::array<VkSubpassDescription2, 2> SubpassDesc{
        VkSubpassDescription2 {.sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2, .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS},
        VkSubpassDescription2 {.sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2, .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS} };
    std::vector<VkAttachmentReference2> ColorReferencesPass0;
    std::vector<VkAttachmentReference2> ResolveReferencesPass0;
    ColorReferencesPass0.reserve(InternalColorFormats.size());
    ResolveReferencesPass0.reserve(InternalColorFormats.size());

    std::vector<VkAttachmentReference2> InputReferencesPass1;
    std::vector<VkAttachmentReference2> ColorReferencesPass1;
    std::vector<VkAttachmentReference2> ResolveReferencesPass1;
    InputReferencesPass1.reserve(InternalColorFormats.size());
    ColorReferencesPass1.reserve(OutputColorFormats.size());
    ResolveReferencesPass1.reserve(OutputColorFormats.size());

    const bool HasDepth = InternalDepthFormat != TextureFormat::UNDEFINED;

    bool Pass0NeedsResolve = InternalMsaa[1] != InternalMsaa[0];
    bool Pass1NeedsResolve = OutputMsaa != InternalMsaa[1];

    //
    // First Pass Color Attachments (what is written by the first pass)
    // Also setup as inputs to the second pass.
    for (const auto& ColorFormat : InternalColorFormats)
    {
        // Pass0 color and depth buffers setup to clear on load, discard on end (of entire pass).
        const VkAttachmentDescription2 AttachmentDescPass0 = {
            VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
            nullptr, /*pNext*/
            0/*flags*/,
            TextureFormatToVk(ColorFormat)/*format*/,
            EnumToVk(InternalMsaa[0])/*samples*/,
            VK_ATTACHMENT_LOAD_OP_CLEAR/*loadOp*/,
            VK_ATTACHMENT_STORE_OP_DONT_CARE/*storeOp*/,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE/*stencilLoadOp*/,
            VK_ATTACHMENT_STORE_OP_DONT_CARE/*stencilStoreOp*/,
            VK_IMAGE_LAYOUT_UNDEFINED/*initialLayout*/,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL/*finalLayout*/ };
        PassAttachDescs.push_back(AttachmentDescPass0);

        ColorReferencesPass0.push_back({ .sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, .attachment = (uint32_t)PassAttachDescs.size() - 1, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL/*output of first pass*/ });
        InputReferencesPass1.push_back({ .sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, .attachment = (uint32_t)PassAttachDescs.size() - 1, .layout = Pass0NeedsResolve ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL/*input of second pass*/ });
    }

    if (Pass0NeedsResolve)
    {
        //
        // First pass Resolve Attachments (intermediate)
        for (const auto& ColorFormat : InternalColorFormats)
        {
            // Pass1 color buffers to resolve to at end of pass.
            VkAttachmentDescription2 AttachmentDescResolvePass0 = {
                .sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
                .format = TextureFormatToVk(ColorFormat),
                .samples = EnumToVk(OutputMsaa),
                .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

            PassAttachDescs.push_back( AttachmentDescResolvePass0 );
            ResolveReferencesPass0.push_back( {.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, .attachment = (uint32_t) PassAttachDescs.size() - 1, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL} );
        }
        //
        // Setup the resolve (first subpass)
        SubpassDesc[0].pResolveAttachments = ResolveReferencesPass0.data();
    }

    //
    // Second Pass Color Attachments (these are the outputs from the second pass)
    for (const auto& ColorFormat : OutputColorFormats)
    {
        // Pass1 color buffers setup to, store on end (of entire pass).
        VkAttachmentDescription2 AttachmentDescPass1 = {
            .sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
            .format = TextureFormatToVk(ColorFormat),
            .samples = EnumToVk((!Pass1NeedsResolve) ? OutputMsaa : InternalMsaa[1]),
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = (!Pass1NeedsResolve) ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        PassAttachDescs.push_back(AttachmentDescPass1);

        ColorReferencesPass1.push_back({.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, .attachment = (uint32_t)PassAttachDescs.size() - 1, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
    }

    if (Pass1NeedsResolve)
    {
        //
        // Second pass Resolve Attachments (output)
        for (const auto& ColorFormat : OutputColorFormats)
        {
            // Pass1 color buffers to resolve to at end of pass.
            VkAttachmentDescription2 AttachmentDescResolvePass1 = {
                VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
                nullptr, // pNext
                0, TextureFormatToVk(ColorFormat)/*format*/,
                EnumToVk(OutputMsaa)/*samples*/,
                VK_ATTACHMENT_LOAD_OP_DONT_CARE/*loadOp*/,
                VK_ATTACHMENT_STORE_OP_STORE/*storeOp*/,
                VK_ATTACHMENT_LOAD_OP_DONT_CARE/*stencilLoadOp*/,
                VK_ATTACHMENT_STORE_OP_DONT_CARE/*stencilStoreOp*/,
                VK_IMAGE_LAYOUT_UNDEFINED/*initialLayout*/,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL/*finalLayout*/ };

            PassAttachDescs.push_back( AttachmentDescResolvePass1 );
            ResolveReferencesPass1.push_back( {.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, .attachment = (uint32_t) PassAttachDescs.size() - 1, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL} );
        }
        //
        // Setup the resolve (second subpass)
        SubpassDesc[1].pResolveAttachments = ResolveReferencesPass1.data();
    }

    //
    // Depth Attachment (cleared at start of pass, written by first subpass, discarded after pass)
    VkAttachmentReference2 DepthReference = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2};
    if (HasDepth)
    {
        VkAttachmentDescription2 AttachmentDescDepthPass0 = {
            VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2,
            nullptr,/*pNext*/
            0/*flags*/,
            TextureFormatToVk(InternalDepthFormat)/*format*/,
            EnumToVk(InternalMsaa[0])/*samples*/,
            VK_ATTACHMENT_LOAD_OP_CLEAR/*loadOp*/,
            VK_ATTACHMENT_STORE_OP_DONT_CARE/*storeOp*/,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE/*stencilLoadOp*/,
            VK_ATTACHMENT_STORE_OP_DONT_CARE/*stencilStoreOp*/,
            VK_IMAGE_LAYOUT_UNDEFINED/*initialLayout*/,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL/*finalLayout*/ };

        PassAttachDescs.push_back(AttachmentDescDepthPass0);
        DepthReference = {.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, .attachment = (uint32_t)PassAttachDescs.size() - 1, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
    }

    //
    // Subpass dependencies
    std::array<VkSubpassDependency2, 1> PassDependencies {{{.sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2}}};
    if (ColorReferencesPass0.size() > 0)
    {
        SubpassDesc[0].colorAttachmentCount = (uint32_t)ColorReferencesPass0.size();
        SubpassDesc[0].pColorAttachments = ColorReferencesPass0.data();
        SubpassDesc[1].inputAttachmentCount = (uint32_t)InputReferencesPass1.size();
        SubpassDesc[1].pInputAttachments = InputReferencesPass1.data();
    }
    if (ColorReferencesPass1.size() > 0)
    {
        SubpassDesc[1].colorAttachmentCount = (uint32_t)ColorReferencesPass1.size();
        SubpassDesc[1].pColorAttachments = ColorReferencesPass1.data();
    }

    // Only first pass writes the Depth
    if (HasDepth)
    {
        SubpassDesc[0].pDepthStencilAttachment = &DepthReference;
    }

    // Color subpass (takes output of first subpass as input to fragment shader)
    PassDependencies[0].srcSubpass = 0;
    PassDependencies[0].dstSubpass = 1;
    PassDependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    PassDependencies[0].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    PassDependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    PassDependencies[0].dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
    PassDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // Now ready to actually create the render pass
    fvk::VkRenderPassCreateInfo2 RenderPassInfo{{
        .flags = 0,
        .attachmentCount = (uint32_t)PassAttachDescs.size(),
        .pAttachments = PassAttachDescs.data(),
        .subpassCount = (uint32_t)SubpassDesc.size(),
        .pSubpasses = SubpassDesc.data(),
        .dependencyCount = (uint32_t)PassDependencies.size(),
        .pDependencies = PassDependencies.data()
    }};

    return CreateRenderPass( *RenderPassInfo, rRenderPass );
}

//-----------------------------------------------------------------------------
bool Vulkan::CreateRenderPass( const VkRenderPassCreateInfo& createInfo, RenderPass& rRenderPass/*out*/ )
//-----------------------------------------------------------------------------
{
    VkRenderPass handle = VK_NULL_HANDLE;
    VkResult retVal = vkCreateRenderPass( m_VulkanDevice, &createInfo, NULL, &handle );
    if (!CheckVkError( "vkCreateRenderPass()", retVal ))
    {
        rRenderPass = {};
        return false;
    }
    rRenderPass = {m_VulkanDevice, handle};
    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::CreateRenderPass( const VkRenderPassCreateInfo2KHR& createInfo, RenderPass& rRenderPass/*out*/ )
//-----------------------------------------------------------------------------
{
    assert( m_ExtRenderPass2 );
    if (!m_ExtRenderPass2)
    {
        rRenderPass = {};
        return false;
    }

    VkRenderPass handle = VK_NULL_HANDLE;
    VkResult retVal = m_ExtRenderPass2->m_vkCreateRenderPass2KHR( m_VulkanDevice, &createInfo, nullptr, &rRenderPass.mRenderPass.handle );
    if (!CheckVkError( "vkCreateRenderPass2KHR()", retVal ))
    {
        rRenderPass = {};
        return false;
    }

    rRenderPass = {m_VulkanDevice, handle};
    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::Create2SubpassRenderPass( const std::span<const TextureFormat> InternalColorFormats,
    const std::span<const TextureFormat> OutputColorFormats,
    TextureFormat InternalDepthFormat,
    Msaa InternalPassMsaa, /* same for both passes*/
    Msaa OutputMsaa,
    RenderPass& rRenderPass/*out*/ )
//-----------------------------------------------------------------------------
{
    const Msaa InternalPassMsaa2[2] = { InternalPassMsaa, InternalPassMsaa };
    return Create2SubpassRenderPass( InternalColorFormats , OutputColorFormats , InternalDepthFormat, InternalPassMsaa2, OutputMsaa, rRenderPass );
}

//-----------------------------------------------------------------------------
bool Vulkan::CreateSubpassShaderResolveRenderPass( const std::span<const TextureFormat> InternalColorFormats,
    const std::span<const TextureFormat> OutputColorFormats,
    TextureFormat InternalDepthFormat,
    Msaa InternalMsaa,
    Msaa OutputMsaa,
    RenderPass& rRenderPass/*out*/ )
//-----------------------------------------------------------------------------
{
    assert( !rRenderPass);                          // check not already allocated and that we have a location to place the renderpass handle
    assert( !InternalColorFormats.empty() );        // not supporting a depth only pass
    assert( !OutputColorFormats.empty() );

    if (InternalMsaa == OutputMsaa)
        // no resolve needed - return 'error'
        return false;
    if (!GetExtRenderPassShaderResolveAvailable())
        // hardware/driver cannot do shader resolve - return 'error'
        return false;

    // Color attachments and Depth attachment
    std::vector<VkAttachmentDescription> PassAttachDescs;
    PassAttachDescs.reserve( InternalColorFormats.size() + OutputColorFormats.size() + 2 );

    // Each subpass needs a reference to its attachments
    std::array<VkSubpassDescription, 2> SubpassDesc {
        VkSubpassDescription {0/*flags*/, VK_PIPELINE_BIND_POINT_GRAPHICS},
        VkSubpassDescription {0/*flags*/, VK_PIPELINE_BIND_POINT_GRAPHICS} };
    std::vector<VkAttachmentReference> ColorReferencesPass0;
    ColorReferencesPass0.reserve( InternalColorFormats.size() );

    std::vector<VkAttachmentReference> InputReferencesPass1;
    std::vector<VkAttachmentReference> ColorReferencesPass1;
    InputReferencesPass1.reserve( InternalColorFormats.size() );
    ColorReferencesPass1.reserve( OutputColorFormats.size() );

    const bool HasDepth = InternalDepthFormat != TextureFormat::UNDEFINED;
    const bool HasPass2ReadDepth = HasDepth && false;

    //
    // First Pass Color Attachments (what is written by the first pass)
    // Also setup as inputs to the second pass.
    for (const auto& ColorFormat : InternalColorFormats)
    {
        // Pass0 color and depth buffers setup to clear on load, discard on end (of entire pass).
        VkAttachmentDescription AttachmentDescPass0 = { 0/*flags*/,
            TextureFormatToVk(ColorFormat)/*format*/,
            EnumToVk(InternalMsaa)/*samples*/,
            VK_ATTACHMENT_LOAD_OP_CLEAR/*loadOp*/,
            VK_ATTACHMENT_STORE_OP_DONT_CARE/*storeOp*/,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE/*stencilLoadOp*/,
            VK_ATTACHMENT_STORE_OP_DONT_CARE/*stencilStoreOp*/,
            VK_IMAGE_LAYOUT_UNDEFINED/*initialLayout*/,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL/*finalLayout*/ };
        PassAttachDescs.push_back( AttachmentDescPass0 );

        ColorReferencesPass0.push_back( { (uint32_t) PassAttachDescs.size() - 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL/*output of first pass*/ } );
        InputReferencesPass1.push_back( { (uint32_t) PassAttachDescs.size() - 1, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL/*input of second pass*/ } );
    }

    //
    // Second Pass Color Attachments (these are the outputs from the second pass)
    for (const auto& ColorFormat : OutputColorFormats)
    {
        // Pass1 color buffers setup to, store on end (of entire pass).
        VkAttachmentDescription AttachmentDescPass1 = { 0, TextureFormatToVk(ColorFormat)/*format*/,
            EnumToVk(OutputMsaa)/*samples*/,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE/*loadOp*/,
            VK_ATTACHMENT_STORE_OP_STORE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE/*stencilLoadOp*/,
            VK_ATTACHMENT_STORE_OP_DONT_CARE/*stencilStoreOp*/,
            VK_IMAGE_LAYOUT_UNDEFINED/*initialLayout*/,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL/*finalLayout*/ };
        PassAttachDescs.push_back( AttachmentDescPass1 );

        ColorReferencesPass1.push_back( { (uint32_t) PassAttachDescs.size() - 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL } );
    }

    //
    // Depth Attachment (cleared at start of pass, written by first subpass, discarded after pass)
    VkAttachmentReference DepthReference = {};
    if (HasDepth)
    {
        VkAttachmentDescription AttachmentDescDepthPass0 = { 0/*flags*/,
            TextureFormatToVk(InternalDepthFormat)/*format*/,
            EnumToVk(InternalMsaa)/*samples*/,
            VK_ATTACHMENT_LOAD_OP_CLEAR/*loadOp*/,
            VK_ATTACHMENT_STORE_OP_DONT_CARE/*storeOp*/,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE/*stencilLoadOp*/,
            VK_ATTACHMENT_STORE_OP_DONT_CARE/*stencilStoreOp*/,
            VK_IMAGE_LAYOUT_UNDEFINED/*initialLayout*/,
            HasPass2ReadDepth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL/*finalLayout*/ };

        PassAttachDescs.push_back( AttachmentDescDepthPass0 );
        DepthReference = { (uint32_t) PassAttachDescs.size() - 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
        if (HasPass2ReadDepth)
            InputReferencesPass1.push_back( { (uint32_t) PassAttachDescs.size() - 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL/*input of second pass*/ } );
    }

    // ShaderBase Resolve extension (VK_QCOM_render_pass_shader_resolve) is available, use it. 
    assert( OutputMsaa == Msaa::Samples1 );
    SubpassDesc[1].flags = VK_SUBPASS_DESCRIPTION_SHADER_RESOLVE_BIT_QCOM | VK_SUBPASS_DESCRIPTION_FRAGMENT_REGION_BIT_QCOM;

    //
    // Subpass dependencies
    std::array<VkSubpassDependency, 1> PassDependencies = {};
    if (ColorReferencesPass0.size() > 0)
    {
        SubpassDesc[0].colorAttachmentCount = (uint32_t) ColorReferencesPass0.size();
        SubpassDesc[0].pColorAttachments = ColorReferencesPass0.data();
        SubpassDesc[1].inputAttachmentCount = (uint32_t) InputReferencesPass1.size();
        SubpassDesc[1].pInputAttachments = InputReferencesPass1.data();
    }
    if (ColorReferencesPass1.size() > 0)
    {
        SubpassDesc[1].colorAttachmentCount = (uint32_t) ColorReferencesPass1.size();
        SubpassDesc[1].pColorAttachments = ColorReferencesPass1.data();
    }

    // Only first pass writes the Depth
    if (HasDepth)
    {
        SubpassDesc[0].pDepthStencilAttachment = &DepthReference;
    }

    // Color subpass (takes output of first subpass as input to fragment shader)
    PassDependencies[0].srcSubpass = 0;
    PassDependencies[0].dstSubpass = 1;
    PassDependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    PassDependencies[0].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    PassDependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    PassDependencies[0].dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
    PassDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // Now ready to actually create the render pass
    fvk::VkRenderPassCreateInfo RenderPassInfo{{
        .flags = 0,
        .attachmentCount = (uint32_t)PassAttachDescs.size(),
        .pAttachments = PassAttachDescs.data(),
        .subpassCount = (uint32_t)SubpassDesc.size(),
        .pSubpasses = SubpassDesc.data(),
        .dependencyCount = (uint32_t)PassDependencies.size(),
        .pDependencies = PassDependencies.data()
    }};
    return CreateRenderPass( *RenderPassInfo, rRenderPass );
}

//-----------------------------------------------------------------------------
bool Vulkan::InitFrameBuffers()
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    //The framebuffer objects reference the renderpass, and allow
    // the references defined in that renderpass to now attach to views.
    // The views in this example are the colour view, which is our swapchain image,
    // and (optionally) the depth buffer created manually earlier.
    std::array<VkImageView, 2> attachments;
    attachments.fill( {} );
    uint32_t numAttachments = 1;
    if( m_SwapchainDepth.format != TextureFormat::UNDEFINED )
    {
        attachments[1] = m_SwapchainDepth.view;
        ++numAttachments;
    }

    fvk::VkFramebufferCreateInfo framebufferCreateInfo;
    framebufferCreateInfo->renderPass = m_SwapchainRenderPass.mRenderPass;
    framebufferCreateInfo->attachmentCount = numAttachments;
    framebufferCreateInfo->pAttachments = attachments.data();
    framebufferCreateInfo->width = m_SurfaceWidth;
    framebufferCreateInfo->height = m_SurfaceHeight;
    if ((m_SwapchainPreTransform & VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR) || (m_SwapchainPreTransform & VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR))
    {
        std::swap(framebufferCreateInfo->width, framebufferCreateInfo->height);
    }
    framebufferCreateInfo->layers = 1;

    LOGI("Creating %d frame buffers... (%d x %d)", m_SwapchainImageCount, framebufferCreateInfo->width, framebufferCreateInfo->height);

    assert(!m_SwapchainBuffers.empty());

    // Reusing the framebufferCreateInfo to create framebuffers,
    // only the attachment to the relevent image view changes each time.
    for (auto& buffer: m_SwapchainBuffers)
    {
        assert( !buffer.framebuffer );
        attachments[0] = buffer.view;
        VkRect2D renderArea{
            .extent {
                .width = framebufferCreateInfo->width, 
                .height = framebufferCreateInfo->height
        }};
        VkViewport renderViewport{
            .width = float( renderArea.extent.width ),
            .height = float( renderArea.extent.height ),
            .minDepth = 0.0f,
            .maxDepth = 1.0f
        };
        std::vector<VkClearValue> clearValues;
        clearValues.resize( numAttachments );   // use 0 filled struct as default.
        RenderPassClearData clearData {renderArea, renderViewport, std::move(clearValues)};

        VkFramebuffer framebuffer{};
        retVal = vkCreateFramebuffer(m_VulkanDevice, &framebufferCreateInfo, nullptr, &framebuffer);
        buffer.framebuffer = {{m_VulkanDevice, framebuffer}, std::move(clearData), "Swapchain Framebuffer"};
        if (!CheckVkError("vkCreateFramebuffer()", retVal))
        {
            return false;
        }
    }

    ReportFramebufferProperties( m_SwapchainBuffers.front().framebuffer.m_FrameBuffer );

    return true;
}

//-----------------------------------------------------------------------------
void Vulkan::DestroySwapChain()
//-----------------------------------------------------------------------------
{
    if( m_SwapchainDepth.format != TextureFormat::UNDEFINED )
    {
        m_MemoryManager.Destroy( std::move( m_SwapchainDepth.image ) );
        m_SwapchainDepth.format = TextureFormat::UNDEFINED;
        vkDestroyImageView( m_VulkanDevice, m_SwapchainDepth.view, nullptr );
        m_SwapchainDepth.view = VK_NULL_HANDLE;
    }

    for (auto& swapchainBuffer : m_SwapchainBuffers)
    {
        vkDestroySemaphore(m_VulkanDevice, swapchainBuffer.semaphore, nullptr);
        vkDestroyFence(m_VulkanDevice, swapchainBuffer.fence, nullptr);
        vkDestroyImageView(m_VulkanDevice, swapchainBuffer.view, nullptr);
        assert(!swapchainBuffer.framebuffer);                   // framebuffers destroyed by DestroyFramebuffers
        swapchainBuffer.image = VK_NULL_HANDLE;                 // images are owned by the m_VulkanSwapchain
    }
    m_SwapchainBuffers.clear();
    
#if defined (OS_WINDOWS) || defined (OS_LINUX)
    m_ExtSwapchain->m_vkDestroySwapchainKHR(m_VulkanDevice, m_VulkanSwapchain, nullptr);
#elif defined (OS_ANDROID)
    vkDestroySwapchainKHR(m_VulkanDevice, m_VulkanSwapchain, nullptr);
#endif // defined (OS_WINDOWS|OS_ANDROID|OS_LINUX)
    m_VulkanSwapchain = VK_NULL_HANDLE;
}

//-----------------------------------------------------------------------------
void Vulkan::DestroySwapchainRenderPass()
//-----------------------------------------------------------------------------
{
    m_SwapchainRenderPass = {};
}

//-----------------------------------------------------------------------------
void Vulkan::DestroyFrameBuffers()
//-----------------------------------------------------------------------------
{
    m_SwapchainBuffers.clear();
}

//-----------------------------------------------------------------------------
bool Vulkan::AllocateCommandBuffer(VkCommandBufferLevel CmdBuffLevel, uint32_t QueueIndex, VkCommandBuffer* pCmdBuffer/*out*/) const
//-----------------------------------------------------------------------------
{
    assert(pCmdBuffer && *pCmdBuffer == VK_NULL_HANDLE);  // check not already allocated and that we have a location to place the command buffer handle

    // Allocate the command buffer from the pool
    VkCommandBufferAllocateInfo AllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    AllocInfo.commandPool = m_VulkanQueues[QueueIndex].CommandPool;
    AllocInfo.level = CmdBuffLevel;
    AllocInfo.commandBufferCount = 1;

    VkResult retVal = vkAllocateCommandBuffers(m_VulkanDevice, &AllocInfo, pCmdBuffer);
    return CheckVkError("vkAllocateCommandBuffers()", retVal);
}

//-----------------------------------------------------------------------------
void Vulkan::FreeCommandBuffer(uint32_t QueueIndex, VkCommandBuffer CmdBuffer) const
//-----------------------------------------------------------------------------
{
    auto cmdPool = m_VulkanQueues[QueueIndex].CommandPool;
    vkFreeCommandBuffers(m_VulkanDevice, cmdPool, 1, &CmdBuffer);
}

//-----------------------------------------------------------------------------
bool Vulkan::CreateRenderPass(std::span<const TextureFormat> ColorFormats, TextureFormat DepthFormat, Msaa Msaa, RenderPassInputUsage ColorInputUsage, RenderPassOutputUsage ColorOutputUsage, bool ShouldClearDepth, RenderPassOutputUsage DepthOutputUsage, RenderPass& rRenderPass/*out*/, std::span<const TextureFormat> ResolveFormats)
//-----------------------------------------------------------------------------
{
    std::vector<RenderPassInputUsage> colorInputUsagePerColor;
    std::fill_n( std::back_inserter( colorInputUsagePerColor ), ColorFormats.size(), ColorInputUsage );
    std::vector<RenderPassOutputUsage> colorOutputUsagePerColor;
    std::fill_n( std::back_inserter( colorOutputUsagePerColor ), ColorFormats.size(), ColorOutputUsage );

    const RenderPassCreateData createData{
        .ColorFormats = ColorFormats,
        .DepthFormat = DepthFormat,
        .Msaa = Msaa,
        .ColorInputUsage = colorInputUsagePerColor,
        .ColorOutputUsage = colorOutputUsagePerColor,
        .DepthInputUsage = ShouldClearDepth ? RenderPassInputUsage::Clear : RenderPassInputUsage::Load,
        .DepthOutputUsage = DepthOutputUsage,
        .ResolveFormats = ResolveFormats,
    };

    auto renderPass = CreateRenderPass( createData );
    rRenderPass = std::move(renderPass);
    return !!rRenderPass;
}

static constexpr VkAttachmentLoadOp InputUsageToVkAttachmentLoadOp( const RenderPassInputUsage t ) {
    constexpr VkAttachmentLoadOp cRenderPassInputUsageToVk[]{
        VK_ATTACHMENT_LOAD_OP_CLEAR,
        VK_ATTACHMENT_LOAD_OP_LOAD,
        VK_ATTACHMENT_LOAD_OP_DONT_CARE
    };
    return cRenderPassInputUsageToVk[(int)t];
}

//-----------------------------------------------------------------------------
RenderPass<Vulkan> Vulkan::CreateRenderPass( const RenderPassCreateData& createData )
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    // Color attachments and Depth attachment
    std::vector<VkAttachmentDescription> PassAttachDescs;
    PassAttachDescs.reserve(createData.ColorFormats.size()+1);

    bool bPresentPass = false;
    bool bHasDepth = createData.DepthFormat != TextureFormat::UNDEFINED;
    auto vkMsaa = EnumToVk(createData.Msaa);

    // Color Attachment
    for(auto i = 0;i<createData.ColorFormats.size(); ++i)
    {
        const auto ColorFormat = createData.ColorFormats[i];
        const auto vkColorFormat = TextureFormatToVk(ColorFormat);
        assert(vkColorFormat != VK_FORMAT_UNDEFINED);

        PassAttachDescs.push_back({
            /*flags*/ 0,
            /*format*/ vkColorFormat,
            /*samples*/ vkMsaa,
            /*loadOp*/ InputUsageToVkAttachmentLoadOp( createData.ColorInputUsage[i] ),
            /*storeOp*/ VK_ATTACHMENT_STORE_OP_STORE,
            /*stencilLoadOp*/ VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            /*stencilStoreOp*/ VK_ATTACHMENT_STORE_OP_DONT_CARE,
            /*initialLayout*/ VK_IMAGE_LAYOUT_UNDEFINED,
            /*finalLayout*/ VK_IMAGE_LAYOUT_UNDEFINED
            });
        auto& PassAttachDesc = PassAttachDescs.back();

        switch(createData.ColorInputUsage[i]) {
        case RenderPassInputUsage::Clear:
            PassAttachDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            break;
        case RenderPassInputUsage::Load:
            PassAttachDesc.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            break;
        case RenderPassInputUsage::DontCare:
            PassAttachDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            break;
        };

        switch (createData.ColorOutputUsage[i]) {
        case RenderPassOutputUsage::Discard:
            PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // not sure why Vulkan spec says this can't be VK_IMAGE_LAYOUT_UNDEFINED - we aren't storing it afterall!
            break;
        case RenderPassOutputUsage::Store:
            PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            break;
        case RenderPassOutputUsage::StoreReadOnly:
            PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            break;
        case RenderPassOutputUsage::StoreTransferSrc:
            PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            break;
        case RenderPassOutputUsage::Clear:
            assert(0); // currently unsupported
            break;
        case RenderPassOutputUsage::Present:
            if (createData.ResolveFormats.empty())
            {
                // Nothing to resolve - this is the output to present
                PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            }
            else
            {
                // Dont care what happens to this pass once the resolve has happened
                PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            bPresentPass = true;
            break;
        }
    }

    // Depth Attachment
    if (bHasDepth)
    {
        PassAttachDescs.push_back({
            /*flags*/ 0,
            /*format*/ TextureFormatToVk( createData.DepthFormat),
            /*samples*/ vkMsaa,
            /*loadOp*/ InputUsageToVkAttachmentLoadOp(createData.DepthInputUsage),
            /*storeOp*/ VK_ATTACHMENT_STORE_OP_STORE,
            /*stencilLoadOp*/ VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            /*stencilStoreOp*/ VK_ATTACHMENT_STORE_OP_DONT_CARE,
            /*initialLayout*/ createData.DepthInputUsage == RenderPassInputUsage::Clear ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            /*finalLayout*/ VK_IMAGE_LAYOUT_UNDEFINED
            });
        auto& PassAttachDesc = PassAttachDescs.back();
        switch (createData.DepthOutputUsage) {
        case RenderPassOutputUsage::Discard:
            PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; // not sure why Vulkan spec says this can't be VK_IMAGE_LAYOUT_UNDEFINED - we aren't storing it afterall!
            break;
        case RenderPassOutputUsage::Store:
            PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            break;
        case RenderPassOutputUsage::StoreReadOnly:
            PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            break;
        case RenderPassOutputUsage::StoreTransferSrc:
            PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            break;
        case RenderPassOutputUsage::Clear:
        case RenderPassOutputUsage::Present:
            assert(0); // currently unsupported
        }
    }

    // We need a reference to the attachments
    std::vector<VkAttachmentReference> ColorReferences;
    ColorReferences.reserve(createData.ColorFormats.size());
    uint32_t AttachmentIndex = 0;
    for(; AttachmentIndex < createData.ColorFormats.size(); ++AttachmentIndex)
    {
        ColorReferences.push_back({ AttachmentIndex, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
    }
    VkAttachmentReference DepthReference = {AttachmentIndex++, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    // We only have one subpass
    VkSubpassDescription SubpassDesc = {};
    SubpassDesc.flags = 0;
    SubpassDesc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    SubpassDesc.inputAttachmentCount = 0;
    SubpassDesc.pInputAttachments = nullptr;

    if (ColorReferences.size() > 0)
    {
        SubpassDesc.colorAttachmentCount = (uint32_t)ColorReferences.size();
        SubpassDesc.pColorAttachments = ColorReferences.data();
    }

    // When we have ResolveFormats we resolve each member of ColorFormats out to a matching ResolveFormats buffer
    std::vector<VkAttachmentReference> ResolveReferences;
    if( createData.Msaa != Msaa::Samples1 && !createData.ResolveFormats.empty() )
    {
        ResolveReferences.resize( createData.ColorFormats.size(), {VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED} ); // must match the number of Color buffers (even if they are not resolving)
        uint32_t ColorIdx = 0;
        for( const TextureFormat ResolveFormat : createData.ResolveFormats )
        {
            VkFormat vkResolveFormat = TextureFormatToVk(ResolveFormat);
            if( vkResolveFormat != VK_FORMAT_UNDEFINED)
            {
                ResolveReferences[ColorIdx] = { (uint32_t) PassAttachDescs.size(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

                // Pass1 color buffers to resolve to at end of pass.
                VkAttachmentDescription AttachmentDescResolvePass1 = { 0, vkResolveFormat/*format*/,
                    VK_SAMPLE_COUNT_1_BIT/*samples*/,
                    VK_ATTACHMENT_LOAD_OP_DONT_CARE/*loadOp*/,
                    VK_ATTACHMENT_STORE_OP_STORE/*storeOp*/,
                    VK_ATTACHMENT_LOAD_OP_DONT_CARE/*stencilLoadOp*/, 
                    VK_ATTACHMENT_STORE_OP_DONT_CARE/*stencilStoreOp*/,
                    VK_IMAGE_LAYOUT_UNDEFINED/*initialLayout*/, 
                    bPresentPass ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL/*finalLayout*/};
                PassAttachDescs.push_back( AttachmentDescResolvePass1 );
            }
            ++ColorIdx;
        }
        SubpassDesc.pResolveAttachments = ResolveReferences.data();
    }
    if (bHasDepth)
    {
        SubpassDesc.pDepthStencilAttachment = &DepthReference;
    }
    SubpassDesc.preserveAttachmentCount = 0;
    SubpassDesc.pPreserveAttachments = nullptr;

    // Subpass dependencies
    std::array<VkSubpassDependency, 2> PassDependencies = {};

    if (bPresentPass)
    {
        // Use the same dependencies as the swapchain was created with
        PassDependencies = m_SwapchainRenderPassDependencies;
    }
    else if (createData.ColorFormats.empty())
    {
        // Depth only pass
        PassDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        PassDependencies[0].dstSubpass = 0;
        PassDependencies[0].srcStageMask = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
        PassDependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        PassDependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        PassDependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        PassDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        PassDependencies[1].srcSubpass = 0;
        PassDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        PassDependencies[1].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        PassDependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        PassDependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        PassDependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        PassDependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    }
    else
    {
        // Color (and maybe depth) pass
        PassDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        PassDependencies[0].dstSubpass = 0;
        PassDependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        PassDependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        PassDependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        PassDependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        PassDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        PassDependencies[1].srcSubpass = 0;
        PassDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        PassDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        PassDependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        PassDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        PassDependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        PassDependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    }

    // Now ready to actually create the render pass
    fvk::VkRenderPassCreateInfo RenderPassCreateInfo {{
        .flags = (VkRenderPassCreateFlags) ((bPresentPass && m_ExtRenderPassTransformEnabled) ? VK_RENDER_PASS_CREATE_TRANSFORM_BIT_QCOM : 0),
        .attachmentCount = (uint32_t)PassAttachDescs.size(),
        .pAttachments = PassAttachDescs.data(),
        .subpassCount = 1,
        .pSubpasses = &SubpassDesc,
        .dependencyCount = (uint32_t)PassDependencies.size(),
        .pDependencies = PassDependencies.data()
    }};

    if (createData.TileShading != RenderPassTileShadingMode::Disabled)
    {
        VkTileShadingRenderPassFlagsQCOM tileShadingFlags = VK_TILE_SHADING_RENDER_PASS_ENABLE_BIT_QCOM;
        if (createData.TileShading == RenderPassTileShadingMode::PerTileShading)
            tileShadingFlags |= VK_TILE_SHADING_RENDER_PASS_PER_TILE_EXECUTION_BIT_QCOM;

        RenderPassCreateInfo.Add<VkRenderPassTileShadingCreateInfoQCOM, VK_STRUCTURE_TYPE_RENDER_PASS_TILE_SHADING_CREATE_INFO_QCOM>( {
            .flags = tileShadingFlags,//
            .tileApronSize = {createData.TileApron[0], createData.TileApron[1]}
        } );
        if (createData.TileApron[0]>0 || createData.TileApron[1] > 0)
            SubpassDesc.flags |= VK_SUBPASS_DESCRIPTION_TILE_SHADING_APRON_BIT_QCOM;
    }

    RenderPass renderPass;
    if (CreateRenderPass( *RenderPassCreateInfo, renderPass ))
        return renderPass;
    return {};
}


//-----------------------------------------------------------------------------
bool Vulkan::CreateRenderPassVRS(std::span<const TextureFormat> ColorFormats, TextureFormat DepthFormat, Msaa Msaa,
    RenderPassInputUsage ColorInputUsage, RenderPassOutputUsage ColorOutputUsage, bool ShouldClearDepth,
    RenderPassOutputUsage DepthOutputUsage, RenderPass& rRenderPass/*out*/, std::span<const TextureFormat> ResolveFormats, bool HasDensityMap)
    //-----------------------------------------------------------------------------
{
    assert( !rRenderPass );                             // check not already allocated

    std::vector<VkAttachmentDescription2> PassAttachDescs;
    PassAttachDescs.reserve(ColorFormats.size());

    bool bPresentPass = false;
    bool bHasDepth = DepthFormat != TextureFormat::UNDEFINED;
    auto vkMsaa = EnumToVk( Msaa );

    // Color Attachment
    for (const auto ColorFormat : ColorFormats)
    {
        const auto vkColorFormat = TextureFormatToVk(ColorFormat);
        assert(vkColorFormat != VK_FORMAT_UNDEFINED);

        PassAttachDescs.push_back({
            /*stype*/ VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2_KHR,
            /*pNext*/ nullptr,
            /*flags*/ 0,
            /*format*/ vkColorFormat,
            /*samples*/ vkMsaa,
            /*loadOp*/ InputUsageToVkAttachmentLoadOp(ColorInputUsage),
            /*storeOp*/ VK_ATTACHMENT_STORE_OP_STORE,
            /*stencilLoadOp*/ VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            /*stencilStoreOp*/ VK_ATTACHMENT_STORE_OP_DONT_CARE,
            /*initialLayout*/ VK_IMAGE_LAYOUT_UNDEFINED,
            /*finalLayout*/ VK_IMAGE_LAYOUT_UNDEFINED
            });
        auto& PassAttachDesc = PassAttachDescs.back();

        switch (ColorInputUsage) {
        case RenderPassInputUsage::Clear:
            PassAttachDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            break;
        case RenderPassInputUsage::Load:
            PassAttachDesc.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            break;
        case RenderPassInputUsage::DontCare:
            PassAttachDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            break;
        };

        switch (ColorOutputUsage) {
        case RenderPassOutputUsage::Discard:
            PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // not sure why Vulkan spec says this can't be VK_IMAGE_LAYOUT_UNDEFINED - we aren't storing it afterall!
            break;
        case RenderPassOutputUsage::Store:
            PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            break;
        case RenderPassOutputUsage::StoreReadOnly:
            PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            break;
        case RenderPassOutputUsage::StoreTransferSrc:
            PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            break;
        case RenderPassOutputUsage::Clear:
            assert(0); // currently unsupported
            break;
        case RenderPassOutputUsage::Present:
            if (ResolveFormats.empty())
            {
                // Nothing to resolve - this is the output to present
                PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            }
            else
            {
                // Dont care what happens to this pass once the resolve has happened
                PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            bPresentPass = true;
            break;
        }
    }

    // Depth Attachment
    if (bHasDepth)
    {
        PassAttachDescs.push_back({
            /*sType*/ VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2_KHR,
            /*pNext*/ nullptr,
            /*flags*/ 0,
            /*format*/ TextureFormatToVk(DepthFormat),
            /*samples*/ vkMsaa,
            /*loadOp*/ ShouldClearDepth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
            /*storeOp*/ VK_ATTACHMENT_STORE_OP_STORE,
            /*stencilLoadOp*/ VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            /*stencilStoreOp*/ VK_ATTACHMENT_STORE_OP_DONT_CARE,
            /*initialLayout*/ ShouldClearDepth ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            /*finalLayout*/ VK_IMAGE_LAYOUT_UNDEFINED
            });
        auto& PassAttachDesc = PassAttachDescs.back();
        switch (DepthOutputUsage) {
        case RenderPassOutputUsage::Discard:
            PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; // not sure why Vulkan spec says this can't be VK_IMAGE_LAYOUT_UNDEFINED - we aren't storing it afterall!
            break;
        case RenderPassOutputUsage::Store:
            PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            break;
        case RenderPassOutputUsage::StoreReadOnly:
            PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            break;
        case RenderPassOutputUsage::StoreTransferSrc:
            PassAttachDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            PassAttachDesc.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            break;
        case RenderPassOutputUsage::Clear:
        case RenderPassOutputUsage::Present:
            assert(0); // currently unsupported
        }
    }

    PassAttachDescs.push_back({
        /*sType*/ VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2_KHR,
        /*pNext*/ nullptr,
        /*flags*/ 0,
        /*format*/ VK_FORMAT_R8_UINT,
        /*samples*/ VK_SAMPLE_COUNT_1_BIT,
        /*loadOp*/ VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        /*storeOp*/ VK_ATTACHMENT_STORE_OP_STORE,
        /*stencilLoadOp*/ VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        /*stencilStoreOp*/ VK_ATTACHMENT_STORE_OP_DONT_CARE,
        /*initialLayout*/ VK_IMAGE_LAYOUT_GENERAL,// VK_IMAGE_LAYOUT_GENERAL ?
        /*finalLayout*/ VK_IMAGE_LAYOUT_GENERAL // VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL ?
        });

    // We need a reference to the attachments
    std::vector<VkAttachmentReference2KHR> ColorReferences;
    ColorReferences.reserve(ColorFormats.size());
    uint32_t AttachmentIndex = 0;
    for (; AttachmentIndex < ColorFormats.size(); ++AttachmentIndex)
    {
        ColorReferences.push_back({ VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2_KHR, nullptr,  AttachmentIndex, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
    }

    VkAttachmentReference2KHR DepthReference;
    if (bHasDepth) {
        DepthReference = { VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2_KHR, nullptr, AttachmentIndex++, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
    }

    VkAttachmentReference2KHR VRS_AttachRef = { VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2_KHR, nullptr, AttachmentIndex++, VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR };

    // We only have one subpass
    VkSubpassDescription2KHR SubpassDesc = { VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2_KHR , nullptr };
    SubpassDesc.flags = 0;
    SubpassDesc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    SubpassDesc.inputAttachmentCount = 0;
    SubpassDesc.pInputAttachments = nullptr;

    if (ColorReferences.size() > 0)
    {
        SubpassDesc.colorAttachmentCount = (uint32_t)ColorReferences.size();
        SubpassDesc.pColorAttachments = ColorReferences.data();
    }

    // When we have ResolveFormats we resolve each member of ColorFormats out to a matching ResolveFormats buffer
    std::vector<VkAttachmentReference2KHR> ResolveReferences;
    if (!ResolveFormats.empty())
    {
        ResolveReferences.resize(ColorFormats.size(), { VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2_KHR, nullptr, VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED }); // must match the number of Color buffers (even if they are not resolving)
        uint32_t ColorIdx = 0;
        for (const auto ResolveFormat : ResolveFormats)
        {
            const VkFormat vkResolveFormat = TextureFormatToVk(ResolveFormat);
            if (vkResolveFormat != VK_FORMAT_UNDEFINED)
            {
                ResolveReferences[ColorIdx] = { VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2_KHR, nullptr, (uint32_t)PassAttachDescs.size(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

                // Pass1 color buffers to resolve to at end of pass.
                VkAttachmentDescription2 AttachmentDescResolvePass1 = {
                    VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2_KHR, /*sType*/
                    nullptr, /*pNext*/
                    0, vkResolveFormat/*format*/,
                    VK_SAMPLE_COUNT_1_BIT/*samples*/,
                    VK_ATTACHMENT_LOAD_OP_DONT_CARE/*loadOp*/,
                    VK_ATTACHMENT_STORE_OP_STORE/*storeOp*/,
                    VK_ATTACHMENT_LOAD_OP_DONT_CARE/*stencilLoadOp*/,
                    VK_ATTACHMENT_STORE_OP_DONT_CARE/*stencilStoreOp*/,
                    VK_IMAGE_LAYOUT_UNDEFINED/*initialLayout*/,
                    bPresentPass ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL/*finalLayout*/ };
                PassAttachDescs.push_back(AttachmentDescResolvePass1);
            }
            ++ColorIdx;
        }
        SubpassDesc.pResolveAttachments = ResolveReferences.data();
    }
    if (bHasDepth)
    {
        SubpassDesc.pDepthStencilAttachment = &DepthReference;
    }

    assert( m_ExtFragmentShadingRate );
    assert( m_ExtFragmentShadingRate->Status == VulkanExtensionStatus::eLoaded );

    VkFragmentShadingRateAttachmentInfoKHR shading_rate_attachment = {};
    shading_rate_attachment.sType = VK_STRUCTURE_TYPE_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR;
    shading_rate_attachment.pFragmentShadingRateAttachment = &VRS_AttachRef;
    shading_rate_attachment.shadingRateAttachmentTexelSize.width = m_ExtFragmentShadingRate->Properties.maxFragmentShadingRateAttachmentTexelSize.width;
    shading_rate_attachment.shadingRateAttachmentTexelSize.height = m_ExtFragmentShadingRate->Properties.maxFragmentShadingRateAttachmentTexelSize.height;
    SubpassDesc.pNext = &shading_rate_attachment;
    SubpassDesc.preserveAttachmentCount = 0;
    SubpassDesc.pPreserveAttachments = nullptr;

    // Subpass dependencies
    std::array<VkSubpassDependency2KHR, 2> PassDependencies = {};

    if (ColorFormats.empty())
    {
        // Depth only pass
        PassDependencies[0].sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
        PassDependencies[0].pNext = nullptr;
        PassDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        PassDependencies[0].dstSubpass = 0;
        PassDependencies[0].srcStageMask = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
        PassDependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        PassDependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        PassDependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        PassDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        PassDependencies[1].sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
        PassDependencies[1].pNext = nullptr;
        PassDependencies[1].srcSubpass = 0;
        PassDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        PassDependencies[1].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        PassDependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        PassDependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        PassDependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        PassDependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    }
    else
    {
        // Color (and maybe depth) pass
        PassDependencies[0].sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
        PassDependencies[0].pNext = nullptr;
        PassDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        PassDependencies[0].dstSubpass = 0;
        PassDependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        PassDependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        PassDependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        PassDependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        PassDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        PassDependencies[1].sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
        PassDependencies[1].pNext = nullptr;
        PassDependencies[1].srcSubpass = 0;
        PassDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        PassDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        PassDependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        PassDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        PassDependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        PassDependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    }


    // We use subpass dependencies to define the color image layout transitions rather than
// explicitly do them in the command buffer, as it is more efficient to do it this way.
#if 0
// Before we can use the back buffer from the swapchain, we must change the
// image layout from the PRESENT mode to the COLOR_ATTACHMENT mode.
    PassDependencies[0].sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
    PassDependencies[0].pNext = nullptr;
    PassDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    PassDependencies[0].dstSubpass = 0;
    PassDependencies[0].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    PassDependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    PassDependencies[0].srcAccessMask = 0;
    PassDependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    PassDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // After writing to the back buffer of the swapchain, we need to change the
    // image layout from the COLOR_ATTACHMENT mode to the PRESENT mode which
    // is optimal for sending to the screen for users to see the completed rendering.
    PassDependencies[1].sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
    PassDependencies[1].pNext = nullptr;
    PassDependencies[1].srcSubpass = 0;
    PassDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    PassDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    PassDependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    PassDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    PassDependencies[1].dstAccessMask = 0;
    PassDependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
#endif

    // Now ready to actually create the render pass
    VkRenderPassCreateInfo2KHR RenderPassInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2 , nullptr };
    RenderPassInfo.flags = (bPresentPass && m_ExtRenderPassTransformEnabled) ? VK_RENDER_PASS_CREATE_TRANSFORM_BIT_QCOM : 0;
    RenderPassInfo.attachmentCount = (uint32_t)PassAttachDescs.size();
    RenderPassInfo.pAttachments = PassAttachDescs.data();
    RenderPassInfo.subpassCount = 1;
    RenderPassInfo.pSubpasses = &SubpassDesc;
    RenderPassInfo.dependencyCount = (uint32_t)PassDependencies.size();
    RenderPassInfo.pDependencies = PassDependencies.data();

    return CreateRenderPass( RenderPassInfo, rRenderPass );
}

//-----------------------------------------------------------------------------
bool Vulkan::CreatePipeline(
    VkPipelineCache                         pipelineCache,
    const VkPipelineVertexInputStateCreateInfo* visci,
    VkPipelineLayout                        pipelineLayout,
    const RenderContext&                    renderingPassContext,
    const VkPipelineRasterizationStateCreateInfo* providedRS,
    const VkPipelineDepthStencilStateCreateInfo*  providedDSS,
    const VkPipelineColorBlendStateCreateInfo*    providedCBS,
    const VkPipelineMultisampleStateCreateInfo*   providedMS,
    const VkPipelineInputAssemblyStateCreateInfo* providedIA,
    std::span<const VkDynamicState>         dynamicStates,
    const VkViewport*                       viewport,
    const VkRect2D*                         scissor,
    VkShaderModule                          taskShaderModule,
    VkShaderModule                          meshShaderModule,
    VkShaderModule                          vertShaderModule,
    VkShaderModule                          fragShaderModule,
    const VkSpecializationInfo*             specializationInfo,
    bool                                    bAllowDerivation,
    VkPipeline                              deriveFromPipeline,
    VkPipeline*                             pipeline)
//-----------------------------------------------------------------------------
{
    // We support both dynamic rendering (no render pass) and old-style vkrenderpass pipelines
    const bool dynamicRendering = renderingPassContext.IsDynamic();

    // Our vertex buffer describes a triangle list.
    VkPipelineInputAssemblyStateCreateInfo ia{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
    };

    // State for rasterization, such as polygon fill mode is defined.
    VkPipelineRasterizationStateCreateInfo rs {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.0f
    };

    // Setup default blending state (disabled).  Will be ignored if providedCBS is set.
    std::array< VkPipelineColorBlendAttachmentState, 10> att_state;
    att_state.fill({ .blendEnable = VK_FALSE, .colorWriteMask = 0xf });
    assert(att_state.size() >= renderingPassContext.GetNumColorAttachmentFormats());

    VkPipelineColorBlendStateCreateInfo cb {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = std::max(uint32_t(1), uint32_t(renderingPassContext.GetNumColorAttachmentFormats())),
        .pAttachments = att_state.data()
    };

    // Standard depth and stencil state is defined
    VkPipelineDepthStencilStateCreateInfo ds { 
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .back = {.failOp = VK_STENCIL_OP_KEEP,
                .passOp = VK_STENCIL_OP_KEEP,
                .compareOp = VK_COMPARE_OP_ALWAYS},
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 1.0f,
    };
    ds.front = ds.back;

    // Default to no msaa
    VkPipelineMultisampleStateCreateInfo ms { 
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .pSampleMask = nullptr,
    };

    // We define four shader stages:  vertex, fragment, task and mesh shaders.
    uint32_t stageCount = 1;
    std::array<VkPipelineShaderStageCreateInfo, 4> shaderStages {};

    uint32_t EVertBit = 0;
    uint32_t EFragBit = 1;
    uint32_t EMeshBit = 2;
    uint32_t ETaskBit = 3;

    const uint32_t EVertMask = 1 << EVertBit;
    const uint32_t EFragMask = 1 << EFragBit;
    const uint32_t EMeshMask = 1 << EMeshBit;
    const uint32_t ETaskMask = 1 << ETaskBit;
    const uint32_t EVertFragMask = EFragMask | EVertMask;
    const uint32_t EMeshFragMask = EFragMask | EMeshMask;
    const uint32_t ETaskMeshFragMask = ETaskMask | EMeshFragMask;

    uint32_t drawbleType = 0;
    drawbleType |= (fragShaderModule != VK_NULL_HANDLE) << EFragBit;
    drawbleType |= (vertShaderModule != VK_NULL_HANDLE) << EVertBit;
    drawbleType |= (meshShaderModule != VK_NULL_HANDLE) << EMeshBit;
    drawbleType |= (taskShaderModule != VK_NULL_HANDLE) << ETaskBit;

    if (drawbleType == EVertFragMask) {
        stageCount = 2;

        shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[0].module = vertShaderModule;
        shaderStages[0].pName = "main";
        shaderStages[0].pSpecializationInfo = specializationInfo;

        shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[1].module = fragShaderModule;
        shaderStages[1].pName = "main";
        shaderStages[1].pSpecializationInfo = specializationInfo;
    }
    else if (drawbleType == EMeshFragMask) {
        stageCount = 2;

        shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[0].stage = VK_SHADER_STAGE_MESH_BIT_EXT;
        shaderStages[0].module = meshShaderModule;
        shaderStages[0].pName = "main";
        shaderStages[0].pSpecializationInfo = specializationInfo;

        shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[1].module = fragShaderModule;
        shaderStages[1].pName = "main";
        shaderStages[1].pSpecializationInfo = specializationInfo;
    }
    else if (drawbleType == ETaskMeshFragMask) {
        stageCount = 3;

        shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[0].stage = VK_SHADER_STAGE_TASK_BIT_EXT;
        shaderStages[0].module = taskShaderModule;
        shaderStages[0].pName = "main";
        shaderStages[0].pSpecializationInfo = specializationInfo;

        shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[1].stage = VK_SHADER_STAGE_MESH_BIT_EXT;
        shaderStages[1].module = meshShaderModule;
        shaderStages[1].pName = "main";
        shaderStages[1].pSpecializationInfo = specializationInfo;

        shaderStages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[2].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[2].module = fragShaderModule;
        shaderStages[2].pName = "main";
        shaderStages[2].pSpecializationInfo = specializationInfo;
    }
    else if (drawbleType == EVertMask) {
        stageCount = 1;

        shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[0].module = vertShaderModule;
        shaderStages[0].pName = "main";
        shaderStages[0].pSpecializationInfo = specializationInfo;
    }
    else
    {
        assert( 0 && "Unsupported combination of shader modules" );
    }

    // Set up the flags
    VkPipelineCreateFlags flags = 0;
    if (bAllowDerivation)
    {
        flags |= VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT;
    }
    if (deriveFromPipeline != VK_NULL_HANDLE)
    {
        flags |= VK_PIPELINE_CREATE_DERIVATIVE_BIT;
    }

    // Create some dynamic states.
#ifndef VK_DYNAMIC_STATE_RANGE_SIZE
#define VK_DYNAMIC_STATE_RANGE_SIZE     (VK_DYNAMIC_STATE_STENCIL_REFERENCE - VK_DYNAMIC_STATE_VIEWPORT + 1)
#endif // VK_DYNAMIC_STATE_RANGE_SIZE

    //
    // Populate dynamic states and viewport/scissor at the same time (if no viewport or scissor is defined we can assume it wants a dynamic state)
    //
    VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };

    // Do we needs to add dynamic states for the viewport/scissor?
    bool setDefaultViewport = viewport == nullptr;
    bool setDefaultScissor = scissor == nullptr;

    std::array<VkDynamicState, VK_DYNAMIC_STATE_RANGE_SIZE> dynamicStateEnables;
    if (setDefaultViewport)
        dynamicStateEnables[dynamicStateCreateInfo.dynamicStateCount++] = VK_DYNAMIC_STATE_VIEWPORT;
    if (setDefaultScissor)
        dynamicStateEnables[dynamicStateCreateInfo.dynamicStateCount++] = VK_DYNAMIC_STATE_SCISSOR;
    if (m_ExtFragmentShadingRate &&
        m_ExtFragmentShadingRate->Status == VulkanExtensionStatus::eLoaded &&
        m_ExtFragmentShadingRate->RequestedFeatures.attachmentFragmentShadingRate == VK_TRUE)
        dynamicStateEnables[dynamicStateCreateInfo.dynamicStateCount++] = VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR;
    dynamicStateCreateInfo.pDynamicStates = dynamicStateEnables.data();

#if 0
    if (!dynamicStates.empty())
    {
        dynamicStateCreateInfo.pDynamicStates = dynamicStates.data();
        dynamicStateCreateInfo.dynamicStateCount = (uint32_t)dynamicStates.size();
        // see if we already have viewport and/or scissor in the dynamic states passed to this function
        for (const auto& dynamicState : dynamicStates)
        {
            if (dynamicState == VK_DYNAMIC_STATE_VIEWPORT)
                setDefaultViewport = false;
            else if (dynamicState == VK_DYNAMIC_STATE_SCISSOR)
                setDefaultScissor = false;
        }
    }
#endif

    VkViewport defaultViewport;
    if (setDefaultViewport)
    {
        defaultViewport = {};
        defaultViewport.height = (float)m_SurfaceHeight;
        defaultViewport.width = (float)m_SurfaceWidth;
        defaultViewport.minDepth = (float)0.0f;
        defaultViewport.maxDepth = (float)1.0f;
    }

    VkRect2D defaultScissor;
    if (setDefaultScissor)
    {
        defaultScissor = {};
        defaultScissor.extent.width = m_SurfaceWidth;
        defaultScissor.extent.height = m_SurfaceHeight;
        defaultScissor.offset.x = 0;
        defaultScissor.offset.y = 0;
    }

    VkPipelineViewportStateCreateInfo ViewportInfo{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    ViewportInfo.viewportCount = (setDefaultViewport || viewport) ? 1 : 0;
    ViewportInfo.pViewports = setDefaultViewport ? &defaultViewport : viewport;
    ViewportInfo.scissorCount = (setDefaultScissor || scissor) ? 1 : 0;
    ViewportInfo.pScissors = setDefaultScissor ? &defaultScissor : scissor;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipelineCreateInfo.flags = flags;
    pipelineCreateInfo.layout = pipelineLayout;
    pipelineCreateInfo.pVertexInputState = visci;
    pipelineCreateInfo.pInputAssemblyState = (providedIA != nullptr) ? providedIA : &ia;

    if (drawbleType == EMeshFragMask || drawbleType == ETaskMeshFragMask) {
        pipelineCreateInfo.pVertexInputState = nullptr;
        pipelineCreateInfo.pInputAssemblyState = nullptr;
    }
    else {
        pipelineCreateInfo.pVertexInputState = visci;
        pipelineCreateInfo.pInputAssemblyState = &ia;
    }

    pipelineCreateInfo.pRasterizationState = (providedRS != nullptr) ? providedRS : &rs;
    pipelineCreateInfo.pColorBlendState = (providedCBS != nullptr) ? providedCBS : &cb;
    pipelineCreateInfo.pMultisampleState = (providedMS != nullptr) ? providedMS : &ms;
    pipelineCreateInfo.pViewportState = &ViewportInfo;
    pipelineCreateInfo.pDepthStencilState = (providedDSS != nullptr) ? providedDSS : &ds;
    pipelineCreateInfo.pStages = &shaderStages[0];
    pipelineCreateInfo.pDynamicState = &dynamicStateCreateInfo;// dynamicStateCreateInfo.dynamicStateCount > 0 ? : nullptr;
    pipelineCreateInfo.stageCount = stageCount;
    pipelineCreateInfo.basePipelineHandle = (deriveFromPipeline != VK_NULL_HANDLE) ? deriveFromPipeline : VK_NULL_HANDLE;
    pipelineCreateInfo.basePipelineIndex = -1; // indicates this field isn't used
    pipelineCreateInfo.subpass = renderingPassContext.subPass;

    fvk::VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo;
    if (dynamicRendering) // don't move the pipelineRenderingCreateInfo local variable inside this if block; want to keep it in scope for vkCreateGraphicsPipelines
    {
        pipelineRenderingCreateInfo = std::move( renderingPassContext.GetPipelineRenderingCreateInfo() );
        pipelineCreateInfo.pNext = &pipelineRenderingCreateInfo;
    }
    else
    {
        pipelineCreateInfo.renderPass = renderingPassContext.GetRenderPass().mRenderPass;
    }

    VkResult retVal = vkCreateGraphicsPipelines(m_VulkanDevice, pipelineCache, 1, &pipelineCreateInfo, nullptr, pipeline);
    if (!CheckVkError("vkCreateGraphicsPipelines()", retVal))
    {
        return false;
    }

    return true;

}

//-----------------------------------------------------------------------------
bool Vulkan::CreateComputePipeline(
    VkPipelineCache  pipelineCache,
    VkPipelineLayout pipelineLayout,
    VkShaderModule   computeModule,
    const VkSpecializationInfo* specializationInfo,
    VkPipeline*      pipeline)
//-----------------------------------------------------------------------------
{
    VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.flags = 0;
    info.layout = pipelineLayout;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = computeModule;
    info.stage.pName = "main";
    info.stage.pSpecializationInfo = specializationInfo;

    m_VulkanPipelineShaderStageCreateInfoExtensions.PushExtensions( &info.stage );
    VkResult retVal = vkCreateComputePipelines(m_VulkanDevice, pipelineCache, 1, &info, nullptr, pipeline);
    m_VulkanPipelineShaderStageCreateInfoExtensions.PopExtensions( &info.stage );
    if (!CheckVkError("vkCreateComputePipelines()", retVal))
    {
        return false;
    }
    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::SetSwapchainHrdMetadata(const VkHdrMetadataEXT& RenderingHdrMetaData)
//-----------------------------------------------------------------------------
{
    // Set the HDR transfer function of the mastering device.
    assert(RenderingHdrMetaData.sType == VK_STRUCTURE_TYPE_HDR_METADATA_EXT);

    if (m_ExtHdrMetadata == nullptr || m_ExtHdrMetadata->Status != VulkanExtensionStatus::eLoaded || m_ExtHdrMetadata->m_vkSetHdrMetadataEXT == nullptr)
        return false;

    m_ExtHdrMetadata->m_vkSetHdrMetadataEXT(m_VulkanDevice, 1, &m_VulkanSwapchain, &RenderingHdrMetaData);
    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::ChangeSurfaceFormat(SurfaceFormat newSurfaceFormat)
//-----------------------------------------------------------------------------
{
    if (newSurfaceFormat.format == m_SurfaceFormat && newSurfaceFormat.colorSpace == m_SurfaceColorSpace)
    {
        return true;    // didn't change anything but is successful
    }

    m_SurfaceFormat = newSurfaceFormat.format;
    m_SurfaceColorSpace = newSurfaceFormat.colorSpace;

    return RecreateSwapChain();
}

//-----------------------------------------------------------------------------
bool Vulkan::RecreateSwapChain()
//-----------------------------------------------------------------------------
{
    // Wait for device to be done rendering
    if (!WaitUntilIdle())
    {
        return false;
    }

    //
    // Destroy everything we need to recreate in order to change the underlying swapchain
    //
    DestroyFrameBuffers();
    DestroySwapchainRenderPass();
    DestroySwapChain();

    //
    // Build the swapchain back up again
    //
    if (!InitSwapChain())
        return false;

    if (!InitSwapchainRenderPass())
        return false;

    if (!InitFrameBuffers())
        return false;

    return true;
}

//-----------------------------------------------------------------------------
Vulkan::BufferIndexAndFence Vulkan::SetNextBackBuffer()
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    VkFence Fence = m_SwapchainBuffers[m_SwapchainCurrentIndx].fence;
    VkSemaphore BackBufferSemaphore = m_SwapchainBuffers[m_SwapchainCurrentIndx].semaphore;

    // Ensure the command buffer we will use to render the 'next image' is done rendering.
    retVal = vkWaitForFences(m_VulkanDevice, 1, &Fence, VK_TRUE, UINT64_MAX);
    CheckVkError( "vkWaitForFences()", retVal );

    // Reset Fence, ready to be set by the GPU when the command buffer has been submitted and completed.
    vkResetFences(m_VulkanDevice, 1, &Fence);

    // Get the next image to render to, then queue a wait until the image is ready
    uint32_t SwapchainPresentIndx = 0;
    retVal = vkAcquireNextImageKHR(m_VulkanDevice, m_VulkanSwapchain, UINT64_MAX, BackBufferSemaphore, VK_NULL_HANDLE, &SwapchainPresentIndx);
    if (retVal == VK_ERROR_OUT_OF_DATE_KHR)
    {
        LOGI("VK_ERROR_OUT_OF_DATE_KHR not handled in sample");
    }
    else if (retVal == VK_SUBOPTIMAL_KHR)
    {
        LOGI("VK_SUBOPTIMAL_KHR not handled in sample");
    }
    else if (!CheckVkError("vkAcquireNextImageKHR()", retVal))
    {
    }

    // Grab the swapchain index and then increment.
    // This index is decoupled from the present index returned by vkAcquireNextImageKHR (which may not run in sequence depending on present modes etc).
    const uint32_t CurrentIndex = m_SwapchainCurrentIndx++;
    if( m_SwapchainCurrentIndx == m_SwapchainImageCount )
        m_SwapchainCurrentIndx = 0;
    return {CurrentIndex, SwapchainPresentIndx, Fence, BackBufferSemaphore};
}

//-----------------------------------------------------------------------------
bool Vulkan::QueueSubmit(const std::span<const VkCommandBuffer> CommandBuffers, const std::span<const VkSemaphore> WaitSemaphores, const std::span<const VkPipelineStageFlags> WaitDstStageMasks, const std::span<const VkSemaphore> SignalSemaphores, uint32_t QueueIndex, VkFence CompletedFence)
//-----------------------------------------------------------------------------
{
    assert(!CommandBuffers.empty());

    // ... submit to the queue
    VkSubmitInfo SubmitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    SubmitInfo.commandBufferCount = (uint32_t)CommandBuffers.size();
    SubmitInfo.pCommandBuffers = CommandBuffers.data();

    if (!WaitSemaphores.empty())
    {
        SubmitInfo.waitSemaphoreCount = (uint32_t)WaitSemaphores.size();
        SubmitInfo.pWaitSemaphores = WaitSemaphores.data();
        assert(WaitDstStageMasks.size() == WaitSemaphores.size());   // expects wait semaphores to have a corresponding mask
        SubmitInfo.pWaitDstStageMask = WaitDstStageMasks.data();
    }

    if (!SignalSemaphores.empty())
    {
        SubmitInfo.signalSemaphoreCount = (uint32_t)SignalSemaphores.size();
        SubmitInfo.pSignalSemaphores = SignalSemaphores.data();
    }

    return QueueSubmit({ &SubmitInfo, 1 }, QueueIndex, CompletedFence);
}

//-----------------------------------------------------------------------------
bool Vulkan::QueueSubmit(const std::span<const VkSubmitInfo> SubmitInfo, uint32_t QueueIndex, VkFence CompletedFence)
//-----------------------------------------------------------------------------
{
    VkQueue Queue = m_VulkanQueues[QueueIndex].Queue;
    assert(Queue != VK_NULL_HANDLE);
    VkResult retVal = vkQueueSubmit(Queue, (uint32_t)SubmitInfo.size(), SubmitInfo.data(), CompletedFence);
    if (!CheckVkError("vkQueueSubmit()", retVal))
    {
        return false;
    }
    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::QueueSubmit(const std::span<const VkSubmitInfo2KHR> SubmitInfo, uint32_t QueueIndex, VkFence CompletedFence)
//-----------------------------------------------------------------------------
{
    VkQueue Queue = m_VulkanQueues[QueueIndex].Queue;
    assert(Queue != VK_NULL_HANDLE);
    assert( m_ExtKhrSynchronization2 && m_ExtKhrSynchronization2->Status == VulkanExtensionStatus::eLoaded );
    VkResult retVal = m_ExtKhrSynchronization2->m_vkQueueSubmit2KHR(Queue, (uint32_t)SubmitInfo.size(), SubmitInfo.data(), CompletedFence);
    if (!CheckVkError("vkQueueSubmit2KHR()", retVal))
    {
        return false;
    }
    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::PresentQueue(const std::span<const VkSemaphore> pWaitSemaphores, uint32_t SwapchainPresentIndx)
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    // Build up the present info and present the queue
    VkPresentInfoKHR PresentInfo {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    PresentInfo.waitSemaphoreCount = 0;
    PresentInfo.pWaitSemaphores = nullptr;
    PresentInfo.swapchainCount = 1;
    PresentInfo.pSwapchains = &m_VulkanSwapchain;
    PresentInfo.pImageIndices = &SwapchainPresentIndx;
    PresentInfo.pResults = nullptr;

    // Do we wait for a semaphore before we present?
    if (!pWaitSemaphores.empty())
    {
        PresentInfo.waitSemaphoreCount = (uint32_t)pWaitSemaphores.size();
        PresentInfo.pWaitSemaphores = pWaitSemaphores.data();
    }

    retVal = vkQueuePresentKHR(m_VulkanQueues[eGraphicsQueue].Queue, &PresentInfo);
    if (retVal == VK_ERROR_OUT_OF_DATE_KHR)
    {
        // Swapchain is out of data.  This can happen if the window was
        // resized after it was created (which is NOT supported here)
        LOGE("Swapchain is out of date! Unable to get next image!");
        return false;
    }
    else if (retVal == VK_SUBOPTIMAL_KHR)
    {
        // This should not be spammed, print it 5 times and then be done!
        static int spamCount = 0;
        if (spamCount < 5)
        {
            ++spamCount;
            VkSurfaceCapabilitiesKHR optimalSurfaceCaps;
            if (QuerySurfaceCapabilities(optimalSurfaceCaps))
            {
                LOGE("Swapchain is not optimal! Should still be presented\n  Ideal swapchain %d x %d (actual %d x %d)", optimalSurfaceCaps.currentExtent.width, optimalSurfaceCaps.currentExtent.height, m_SurfaceWidth, m_SurfaceHeight);
            }
        }
    }
    else
    {
        if (!CheckVkError("vkQueuePresentKHR()", retVal))
        {
            return false;
        }
    }

    // Optinally wait for the queue to be idle.  Creates sync point between the CPU and the GPU.
    // Don't enable unless required for debugging.
    if(false)
    {
        WaitUntilIdle();
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::QueueWaitIdle(uint32_t QueueIndex) const
//-----------------------------------------------------------------------------
{
    assert(m_VulkanQueues[QueueIndex].Queue != VK_NULL_HANDLE);
    VkResult retVal = vkQueueWaitIdle(m_VulkanQueues[QueueIndex].Queue);
    return CheckVkError("vkQueueWaitIdle()", retVal);
}

//-----------------------------------------------------------------------------
bool Vulkan::WaitUntilIdle() const
//-----------------------------------------------------------------------------
{
    VkResult retVal = vkDeviceWaitIdle(m_VulkanDevice);
    return CheckVkError("vkDeviceWaitIdle()", retVal);
}

//-----------------------------------------------------------------------------
VkCommandBuffer Vulkan::StartSetupCommandBuffer()
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    // Make sure we are not leaking by starting without releasing!
    if (m_SetupCmdBuffer != VK_NULL_HANDLE)
    {
        LOGE("Setup CommandBuffer has NOT been freed!");
        FreeCommandBuffer(eGraphicsQueue, m_SetupCmdBuffer);
        m_SetupCmdBuffer = VK_NULL_HANDLE;
    }

    // Allocate the setup command buffer...
    if (!AllocateCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, eGraphicsQueue, &m_SetupCmdBuffer))
    {
        return VK_NULL_HANDLE;
    }
    SetDebugObjectName(m_SetupCmdBuffer, "Setup");

    // ... and start it up
    VkCommandBufferBeginInfo BeginInfo {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    BeginInfo.flags = 0;
    BeginInfo.pInheritanceInfo = nullptr;

    retVal = vkBeginCommandBuffer(m_SetupCmdBuffer, &BeginInfo);
    if (!CheckVkError("vkBeginCommandBuffer()", retVal))
    {
        return VK_NULL_HANDLE;
    }
    return m_SetupCmdBuffer;
}

//-----------------------------------------------------------------------------
void Vulkan::FinishSetupCommandBuffer(VkCommandBuffer setupCmdBuffer)
//-----------------------------------------------------------------------------
{
    VkResult retVal = VK_SUCCESS;

    // Make sure we are not out of state!
    if (m_SetupCmdBuffer == VK_NULL_HANDLE)
    {
        LOGE("Setup CommandBuffer has NOT been set started!");
        return;
    }
    if (m_SetupCmdBuffer != setupCmdBuffer)
    {
        LOGE("Setup CommandBuffer does not match he one passed in to FinishSetupCommandBuffer!");
    }

    // Stop recording the command buffer...
    retVal = vkEndCommandBuffer(m_SetupCmdBuffer);
    if (!CheckVkError("vkEndCommandBuffer()", retVal))
    {
        return;
    }

    // ... submit the command buffer ...
    QueueSubmit({ &m_SetupCmdBuffer,1 }, {}, {}, {}, Vulkan::eGraphicsQueue, VK_NULL_HANDLE);

    // ... wait for it to complete... (cpu wait)
    QueueWaitIdle();

    // ... and then clean up the command buffer
    FreeCommandBuffer(eGraphicsQueue, m_SetupCmdBuffer);
    m_SetupCmdBuffer = VK_NULL_HANDLE;
}

//-----------------------------------------------------------------------------
bool Vulkan::HasLoadedVulkanDeviceExtension( const std::string& extensionName ) const
//-----------------------------------------------------------------------------
{
    auto foundIt = m_DeviceExtensions.m_Extensions.find( extensionName );
    if (foundIt == m_DeviceExtensions.m_Extensions.end())
        return false;
    return foundIt->second->Status == VulkanExtensionStatus::eLoaded;
}

//-----------------------------------------------------------------------------
bool Vulkan::FillRenderPassTransformBeginInfoQCOM(VkRenderPassTransformBeginInfoQCOM& RPTransformBeginInfoQCOM) const
//-----------------------------------------------------------------------------
{
    if (!m_ExtRenderPassTransformEnabled)
        return false;

#ifndef VK_STRUCTURE_TYPE_RENDER_PASS_TRANSFORM_BEGIN_INFO_QCOM_LEGACY
#define VK_STRUCTURE_TYPE_RENDER_PASS_TRANSFORM_BEGIN_INFO_QCOM_LEGACY 1000282000
#endif // VK_STRUCTURE_TYPE_RENDER_PASS_TRANSFORM_BEGIN_INFO_QCOM_LEGACY

    RPTransformBeginInfoQCOM.transform = m_SwapchainPreTransform;
    RPTransformBeginInfoQCOM.sType = (VkStructureType)(m_ExtRenderPassTransformLegacy ? VK_STRUCTURE_TYPE_RENDER_PASS_TRANSFORM_BEGIN_INFO_QCOM_LEGACY : VK_STRUCTURE_TYPE_RENDER_PASS_TRANSFORM_BEGIN_INFO_QCOM);
    RPTransformBeginInfoQCOM.pNext = nullptr;
    return true;
}

//-----------------------------------------------------------------------------
bool Vulkan::FillCommandBufferInheritanceRenderPassTransformInfoQCOM(VkCommandBufferInheritanceRenderPassTransformInfoQCOM& InheritanceInfoRenderPassTransform) const
//-----------------------------------------------------------------------------
{
    if (!m_ExtRenderPassTransformEnabled)
        return false;

#ifndef VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDER_PASS_TRANSFORM_INFO_QCOM_LEGACY
#define VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDER_PASS_TRANSFORM_INFO_QCOM_LEGACY	1000282001
#endif // VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDER_PASS_TRANSFORM_INFO_QCOM_LEGACY

    InheritanceInfoRenderPassTransform.sType = (VkStructureType)(m_ExtRenderPassTransformLegacy ? VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDER_PASS_TRANSFORM_INFO_QCOM_LEGACY : VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDER_PASS_TRANSFORM_INFO_QCOM);
    InheritanceInfoRenderPassTransform.pNext = nullptr;
    InheritanceInfoRenderPassTransform.renderArea.offset = { 0,0 };
    InheritanceInfoRenderPassTransform.renderArea.extent = { m_SurfaceWidth, m_SurfaceHeight };
    InheritanceInfoRenderPassTransform.transform = m_SwapchainPreTransform;
    return true;
}

//=============================================================================
bool Vulkan::SetDebugObjectName(uint64_t object, VkObjectType objectType, const char *name)
//=============================================================================
{
#if defined(USES_VULKAN_DEBUG_LAYERS)
    if (gEnableValidation && name && *name)
    {
        return ((m_ExtDebugUtils && m_ExtDebugUtils->SetDebugUtilsObjectName( m_VulkanDevice, object, objectType, name )) ||
                (objectType <= VK_OBJECT_TYPE_COMMAND_POOL && m_ExtDebugMarker && m_ExtDebugMarker->DebugMarkerSetObjectName( m_VulkanDevice, object, (VkDebugReportObjectTypeEXT) objectType, name )));
    }
#endif // defined(USES_VULKAN_DEBUG_LAYERS)
    return true;
}

//-----------------------------------------------------------------------------
bool CheckVkError(const char* pPrefix, VkResult CheckVal)
//-----------------------------------------------------------------------------
{
    if (CheckVal == VK_SUCCESS)
        return true;

    switch (CheckVal)
    {
    case VK_NOT_READY:
        LOGE("Vulkan Error (VK_NOT_READY) from %s", pPrefix);
        break;
    case VK_TIMEOUT:
        LOGE("Vulkan Error (VK_TIMEOUT) from %s", pPrefix);
        break;
    case VK_EVENT_SET:
        LOGE("Vulkan Error (VK_EVENT_SET) from %s", pPrefix);
        break;
    case VK_EVENT_RESET:
        LOGE("Vulkan Error (VK_EVENT_RESET) from %s", pPrefix);
        break;
    case VK_INCOMPLETE:
        LOGE("Vulkan Error (VK_INCOMPLETE) from %s", pPrefix);
        break;
    case VK_ERROR_OUT_OF_HOST_MEMORY:
        LOGE("Vulkan Error (VK_ERROR_OUT_OF_HOST_MEMORY) from %s", pPrefix);
        break;
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        LOGE("Vulkan Error (VK_ERROR_OUT_OF_DEVICE_MEMORY) from %s", pPrefix);
        break;
    case VK_ERROR_INITIALIZATION_FAILED:
        LOGE("Vulkan Error (VK_ERROR_INITIALIZATION_FAILED) from %s", pPrefix);
        break;
    case VK_ERROR_DEVICE_LOST:
        LOGE("Vulkan Error (VK_ERROR_DEVICE_LOST) from %s", pPrefix);
        break;
    case VK_ERROR_MEMORY_MAP_FAILED:
        LOGE("Vulkan Error (VK_ERROR_MEMORY_MAP_FAILED) from %s", pPrefix);
        break;
    case VK_ERROR_LAYER_NOT_PRESENT:
        LOGE("Vulkan Error (VK_ERROR_LAYER_NOT_PRESENT) from %s", pPrefix);
        break;
    case VK_ERROR_EXTENSION_NOT_PRESENT:
        LOGE("Vulkan Error (VK_ERROR_EXTENSION_NOT_PRESENT) from %s", pPrefix);
        break;
    case VK_ERROR_FEATURE_NOT_PRESENT:
        LOGE("Vulkan Error (VK_ERROR_FEATURE_NOT_PRESENT) from %s", pPrefix);
        break;
    case VK_ERROR_INCOMPATIBLE_DRIVER:
        LOGE("Vulkan Error (VK_ERROR_INCOMPATIBLE_DRIVER) from %s", pPrefix);
        break;
    case VK_ERROR_TOO_MANY_OBJECTS:
        LOGE("Vulkan Error (VK_ERROR_TOO_MANY_OBJECTS) from %s", pPrefix);
        break;
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
        LOGE("Vulkan Error (VK_ERROR_FORMAT_NOT_SUPPORTED) from %s", pPrefix);
        break;
    case VK_ERROR_SURFACE_LOST_KHR:
        LOGE("Vulkan Error (VK_ERROR_SURFACE_LOST_KHR) from %s", pPrefix);
        break;
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
        LOGE("Vulkan Error (VK_ERROR_NATIVE_WINDOW_IN_USE_KHR) from %s", pPrefix);
        break;
    case VK_SUBOPTIMAL_KHR:
        LOGE("Vulkan Error (VK_SUBOPTIMAL_KHR) from %s", pPrefix);
        break;
    case VK_ERROR_OUT_OF_DATE_KHR:
        LOGE("Vulkan Error (VK_ERROR_OUT_OF_DATE_KHR) from %s", pPrefix);
        break;
    case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR:
        LOGE("Vulkan Error (VK_ERROR_INCOMPATIBLE_DISPLAY_KHR) from %s", pPrefix);
        break;
    case VK_ERROR_VALIDATION_FAILED_EXT:
        LOGE("Vulkan Error (VK_ERROR_VALIDATION_FAILED_EXT) from %s", pPrefix);
        break;

#if defined(OS_WINDOWS)
    case VK_ERROR_UNKNOWN:
        LOGE("Vulkan Error (VK_ERROR_UNKNOWN) from %s", pPrefix);
        break;
#endif // defined(OS_WINDOWS)

    default:
        LOGE("Vulkan Error (%d) from %s", CheckVal, pPrefix);
        break;
    }

    return false;
}

//-----------------------------------------------------------------------------
const char* Vulkan::VulkanFormatString(VkFormat WhichFormat)
//-----------------------------------------------------------------------------
{
    switch (WhichFormat)
    {
#define CASE_ENUM_RETURN_STRING(X) case X: return #X
        CASE_ENUM_RETURN_STRING(VK_FORMAT_UNDEFINED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R4G4_UNORM_PACK8);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R4G4B4A4_UNORM_PACK16);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B4G4R4A4_UNORM_PACK16);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R5G6B5_UNORM_PACK16);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B5G6R5_UNORM_PACK16);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R5G5B5A1_UNORM_PACK16);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B5G5R5A1_UNORM_PACK16);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A1R5G5B5_UNORM_PACK16);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8_UNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8_SNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8_USCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8_SSCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8_SRGB);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8_UNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8_SNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8_USCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8_SSCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8_SRGB);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8B8_UNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8B8_SNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8B8_USCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8B8_SSCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8B8_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8B8_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8B8_SRGB);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B8G8R8_UNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B8G8R8_SNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B8G8R8_USCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B8G8R8_SSCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B8G8R8_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B8G8R8_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B8G8R8_SRGB);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8B8A8_UNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8B8A8_SNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8B8A8_USCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8B8A8_SSCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8B8A8_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8B8A8_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R8G8B8A8_SRGB);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B8G8R8A8_UNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B8G8R8A8_SNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B8G8R8A8_USCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B8G8R8A8_SSCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B8G8R8A8_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B8G8R8A8_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B8G8R8A8_SRGB);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A8B8G8R8_UNORM_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A8B8G8R8_SNORM_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A8B8G8R8_USCALED_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A8B8G8R8_SSCALED_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A8B8G8R8_UINT_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A8B8G8R8_SINT_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A8B8G8R8_SRGB_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A2R10G10B10_UNORM_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A2R10G10B10_SNORM_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A2R10G10B10_USCALED_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A2R10G10B10_SSCALED_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A2R10G10B10_UINT_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A2R10G10B10_SINT_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A2B10G10R10_UNORM_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A2B10G10R10_SNORM_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A2B10G10R10_USCALED_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A2B10G10R10_SSCALED_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A2B10G10R10_UINT_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_A2B10G10R10_SINT_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16_UNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16_SNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16_USCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16_SSCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16_SFLOAT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16_UNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16_SNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16_USCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16_SSCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16_SFLOAT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16B16_UNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16B16_SNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16B16_USCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16B16_SSCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16B16_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16B16_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16B16_SFLOAT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16B16A16_UNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16B16A16_SNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16B16A16_USCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16B16A16_SSCALED);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16B16A16_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16B16A16_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R16G16B16A16_SFLOAT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R32_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R32_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R32_SFLOAT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R32G32_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R32G32_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R32G32_SFLOAT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R32G32B32_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R32G32B32_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R32G32B32_SFLOAT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R32G32B32A32_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R32G32B32A32_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R32G32B32A32_SFLOAT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R64_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R64_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R64_SFLOAT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R64G64_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R64G64_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R64G64_SFLOAT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R64G64B64_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R64G64B64_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R64G64B64_SFLOAT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R64G64B64A64_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R64G64B64A64_SINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_R64G64B64A64_SFLOAT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_B10G11R11_UFLOAT_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_D16_UNORM);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_X8_D24_UNORM_PACK32);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_D32_SFLOAT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_S8_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_D16_UNORM_S8_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_D24_UNORM_S8_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_D32_SFLOAT_S8_UINT);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_BC1_RGB_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_BC1_RGB_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_BC1_RGBA_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_BC1_RGBA_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_BC2_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_BC2_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_BC3_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_BC3_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_BC4_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_BC4_SNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_BC5_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_BC5_SNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_BC6H_UFLOAT_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_BC6H_SFLOAT_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_BC7_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_BC7_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_EAC_R11_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_EAC_R11_SNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_EAC_R11G11_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_EAC_R11G11_SNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_4x4_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_4x4_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_5x4_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_5x4_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_5x5_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_5x5_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_6x5_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_6x5_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_6x6_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_6x6_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_8x5_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_8x5_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_8x6_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_8x6_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_8x8_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_8x8_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_10x5_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_10x5_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_10x6_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_10x6_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_10x8_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_10x8_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_10x10_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_10x10_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_12x10_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_12x10_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_12x12_UNORM_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_ASTC_12x12_SRGB_BLOCK);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG);
        CASE_ENUM_RETURN_STRING(VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG);
#undef CASE_ENUM_RETURN_STRING
    default:
        return "Unknown!";
    }
}

//-----------------------------------------------------------------------------
const char* Vulkan::VulkanColorSpaceString(VkColorSpaceKHR ColorSpace)
//-----------------------------------------------------------------------------
{
    switch (ColorSpace)
    {
#define CASE_ENUM_RETURN_STRING(X) case X: return #X;
        CASE_ENUM_RETURN_STRING(VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);
        CASE_ENUM_RETURN_STRING(VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT);
        CASE_ENUM_RETURN_STRING(VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT);
        CASE_ENUM_RETURN_STRING(VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT);
        CASE_ENUM_RETURN_STRING(VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT);
        CASE_ENUM_RETURN_STRING(VK_COLOR_SPACE_BT709_LINEAR_EXT);
        CASE_ENUM_RETURN_STRING(VK_COLOR_SPACE_BT709_NONLINEAR_EXT);
        CASE_ENUM_RETURN_STRING(VK_COLOR_SPACE_BT2020_LINEAR_EXT);
        CASE_ENUM_RETURN_STRING(VK_COLOR_SPACE_HDR10_ST2084_EXT);
        CASE_ENUM_RETURN_STRING(VK_COLOR_SPACE_DOLBYVISION_EXT);
        CASE_ENUM_RETURN_STRING(VK_COLOR_SPACE_HDR10_HLG_EXT);
        CASE_ENUM_RETURN_STRING(VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT);
        CASE_ENUM_RETURN_STRING(VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT);
        CASE_ENUM_RETURN_STRING(VK_COLOR_SPACE_PASS_THROUGH_EXT);
        CASE_ENUM_RETURN_STRING(VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT);
        CASE_ENUM_RETURN_STRING(VK_COLOR_SPACE_DISPLAY_NATIVE_AMD);
#undef CASE_ENUM_RETURN_STRING
    default:
        return "Unknown!";
    }
}

//-----------------------------------------------------------------------------
void Vulkan::QueryPhysicalDeviceFeatures(VkPhysicalDevice physicalDevice, PhysicalDeviceFeatures& featuresOut)
//-----------------------------------------------------------------------------
{
    assert(featuresOut.Base.sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
    featuresOut.Base.pNext = nullptr;

    m_GetPhysicalDeviceFeaturesExtensions.PushExtensions(&featuresOut.Base);

    if (featuresOut.Base.pNext != nullptr)
    {
        if (m_ExtKhrGetPhysicalDeviceProperties2->Status == VulkanExtensionStatus::eLoaded)
        {
            m_ExtKhrGetPhysicalDeviceProperties2->m_vkGetPhysicalDeviceFeatures2KHR(physicalDevice, &featuresOut.Base);
        }
        else
        {
            LOGE("Vulkan version does not provide vkGetPhysicalDeviceFeatures2 which is required to get the requested feature information"); // if this is an issue we can implement alternate call to VK_KHR_get_physical_device_properties2
            vkGetPhysicalDeviceFeatures(physicalDevice, &featuresOut.Base.features);
        }
    }
    else
        vkGetPhysicalDeviceFeatures(physicalDevice, &featuresOut.Base.features);

    m_GetPhysicalDeviceFeaturesExtensions.PopExtensions(&featuresOut.Base);
}

//-----------------------------------------------------------------------------
void Vulkan::QueryPhysicalDeviceProperties( VkPhysicalDevice physicalDevice, const PhysicalDeviceFeatures& features, PhysicalDeviceProperties& propertiesOut )
//-----------------------------------------------------------------------------
{
    // **********************************
    // Properties
    // **********************************
    assert( propertiesOut.Base.sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 );
    assert( propertiesOut.Base.pNext == nullptr );

    m_GetPhysicalDevicePropertiesExtensions.PushExtensions( &propertiesOut.Base );

    if (m_ExtKhrGetPhysicalDeviceProperties2->Status == VulkanExtensionStatus::eLoaded)
    {
        m_ExtKhrGetPhysicalDeviceProperties2->m_vkGetPhysicalDeviceProperties2KHR( physicalDevice, &propertiesOut.Base );
    }
    else
    {
        assert( propertiesOut.Base.pNext == nullptr );  // we need fpGetPhysicalDeviceProperties2 if we want to populate a pNext chain of properties.
        vkGetPhysicalDeviceProperties( physicalDevice, &propertiesOut.Base.properties );
    }

    m_GetPhysicalDevicePropertiesExtensions.PopExtensions( &propertiesOut.Base );
}


//-----------------------------------------------------------------------------
size_t Vulkan::GetBestVulkanPhysicalDeviceId(const std::span<const PhysicalDeviceProperties> deviceProperties)
//-----------------------------------------------------------------------------
{
    // Prefer the first discrete GPU.
    for (size_t deviceIdx = 0; deviceIdx < deviceProperties.size(); ++deviceIdx)
    {
        if (deviceProperties[deviceIdx].Base.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            return deviceIdx;
    }
    // Otherwise use the first integrated GPU.
    for (size_t deviceIdx = 0; deviceIdx < deviceProperties.size(); ++deviceIdx)
    {
        if (deviceProperties[deviceIdx].Base.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            return deviceIdx;
    }
    // fallback to the first GPU.
    return 0;
}

//-----------------------------------------------------------------------------
void Vulkan::DumpDeviceInfo( const std::span<PhysicalDeviceFeatures> DeviceFeatures, const std::span<PhysicalDeviceProperties> DeviceProperties )
//-----------------------------------------------------------------------------
{
    for (uint32_t uiIndx = 0; uiIndx < (uint32_t) DeviceFeatures.size(); uiIndx++)
    {
        LOGI("******************************");
        LOGI("Vulkan Device %u", uiIndx);
        LOGI("******************************");

        // Can now query all sorts of information about the device
        // **********************************
        // Features
        // **********************************
        const PhysicalDeviceFeatures& Features = DeviceFeatures[uiIndx];

        LOGI("Features: ");

        LOGI("    robustBufferAccess: %s", Features.Base.features.robustBufferAccess ? "True" : "False");
        LOGI("    fullDrawIndexUint32: %s", Features.Base.features.fullDrawIndexUint32 ? "True" : "False");
        LOGI("    imageCubeArray: %s", Features.Base.features.imageCubeArray ? "True" : "False");
        LOGI("    independentBlend: %s", Features.Base.features.independentBlend ? "True" : "False");
        LOGI("    geometryShader: %s", Features.Base.features.geometryShader ? "True" : "False");
        LOGI("    tessellationShader: %s", Features.Base.features.tessellationShader ? "True" : "False");
        LOGI("    sampleRateShading: %s", Features.Base.features.sampleRateShading ? "True" : "False");
        LOGI("    dualSrcBlend: %s", Features.Base.features.dualSrcBlend ? "True" : "False");
        LOGI("    logicOp: %s", Features.Base.features.logicOp ? "True" : "False");
        LOGI("    multiDrawIndirect: %s", Features.Base.features.multiDrawIndirect ? "True" : "False");
        LOGI("    drawIndirectFirstInstance: %s", Features.Base.features.drawIndirectFirstInstance ? "True" : "False");
        LOGI("    depthClamp: %s", Features.Base.features.depthClamp ? "True" : "False");
        LOGI("    depthBiasClamp: %s", Features.Base.features.depthBiasClamp ? "True" : "False");
        LOGI("    fillModeNonSolid: %s", Features.Base.features.fillModeNonSolid ? "True" : "False");
        LOGI("    depthBounds: %s", Features.Base.features.depthBounds ? "True" : "False");
        LOGI("    wideLines: %s", Features.Base.features.wideLines ? "True" : "False");
        LOGI("    largePoints: %s", Features.Base.features.largePoints ? "True" : "False");
        LOGI("    alphaToOne: %s", Features.Base.features.alphaToOne ? "True" : "False");
        LOGI("    multiViewport: %s", Features.Base.features.multiViewport ? "True" : "False");
        LOGI("    samplerAnisotropy: %s", Features.Base.features.samplerAnisotropy ? "True" : "False");
        LOGI("    textureCompressionETC2: %s", Features.Base.features.textureCompressionETC2 ? "True" : "False");
        LOGI("    textureCompressionASTC_LDR: %s", Features.Base.features.textureCompressionASTC_LDR ? "True" : "False");
        LOGI("    textureCompressionBC: %s", Features.Base.features.textureCompressionBC ? "True" : "False");
        LOGI("    occlusionQueryPrecise: %s", Features.Base.features.occlusionQueryPrecise ? "True" : "False");
        LOGI("    pipelineStatisticsQuery: %s", Features.Base.features.pipelineStatisticsQuery ? "True" : "False");
        LOGI("    vertexPipelineStoresAndAtomics: %s", Features.Base.features.vertexPipelineStoresAndAtomics ? "True" : "False");
        LOGI("    fragmentStoresAndAtomics: %s", Features.Base.features.fragmentStoresAndAtomics ? "True" : "False");
        LOGI("    shaderTessellationAndGeometryPointSize: %s", Features.Base.features.shaderTessellationAndGeometryPointSize ? "True" : "False");
        LOGI("    shaderImageGatherExtended: %s", Features.Base.features.shaderImageGatherExtended ? "True" : "False");
        LOGI("    shaderStorageImageExtendedFormats: %s", Features.Base.features.shaderStorageImageExtendedFormats ? "True" : "False");
        LOGI("    shaderStorageImageMultisample: %s", Features.Base.features.shaderStorageImageMultisample ? "True" : "False");
        LOGI("    shaderStorageImageReadWithoutFormat: %s", Features.Base.features.shaderStorageImageReadWithoutFormat ? "True" : "False");
        LOGI("    shaderStorageImageWriteWithoutFormat: %s", Features.Base.features.shaderStorageImageWriteWithoutFormat ? "True" : "False");
        LOGI("    shaderUniformBufferArrayDynamicIndexing: %s", Features.Base.features.shaderUniformBufferArrayDynamicIndexing ? "True" : "False");
        LOGI("    shaderSampledImageArrayDynamicIndexing: %s", Features.Base.features.shaderSampledImageArrayDynamicIndexing ? "True" : "False");
        LOGI("    shaderStorageBufferArrayDynamicIndexing: %s", Features.Base.features.shaderStorageBufferArrayDynamicIndexing ? "True" : "False");
        LOGI("    shaderStorageImageArrayDynamicIndexing: %s", Features.Base.features.shaderStorageImageArrayDynamicIndexing ? "True" : "False");
        LOGI("    shaderClipDistance: %s", Features.Base.features.shaderClipDistance ? "True" : "False");
        LOGI("    shaderCullDistance: %s", Features.Base.features.shaderCullDistance ? "True" : "False");
        LOGI("    shaderFloat64: %s", Features.Base.features.shaderFloat64 ? "True" : "False");
        LOGI("    shaderInt64: %s", Features.Base.features.shaderInt64 ? "True" : "False");
        LOGI("    shaderInt16: %s", Features.Base.features.shaderInt16 ? "True" : "False");
        LOGI("    shaderResourceResidency: %s", Features.Base.features.shaderResourceResidency ? "True" : "False");
        LOGI("    shaderResourceMinLod: %s", Features.Base.features.shaderResourceMinLod ? "True" : "False");
        LOGI("    sparseBinding: %s", Features.Base.features.sparseBinding ? "True" : "False");
        LOGI("    sparseResidencyBuffer: %s", Features.Base.features.sparseResidencyBuffer ? "True" : "False");
        LOGI("    sparseResidencyImage2D: %s", Features.Base.features.sparseResidencyImage2D ? "True" : "False");
        LOGI("    sparseResidencyImage3D: %s", Features.Base.features.sparseResidencyImage3D ? "True" : "False");
        LOGI("    sparseResidency2Samples: %s", Features.Base.features.sparseResidency2Samples ? "True" : "False");
        LOGI("    sparseResidency4Samples: %s", Features.Base.features.sparseResidency4Samples ? "True" : "False");
        LOGI("    sparseResidency8Samples: %s", Features.Base.features.sparseResidency8Samples ? "True" : "False");
        LOGI("    sparseResidency16Samples: %s", Features.Base.features.sparseResidency16Samples ? "True" : "False");
        LOGI("    sparseResidencyAliased: %s", Features.Base.features.sparseResidencyAliased ? "True" : "False");
        LOGI("    variableMultisampleRate: %s", Features.Base.features.variableMultisampleRate ? "True" : "False");
        LOGI("    inheritedQueries: %s", Features.Base.features.inheritedQueries ? "True" : "False");

        // Display features from all the registered extensions
        {
            VulkanDeviceFeaturePrint registeredExtensionPrint;
            m_VulkanDeviceFeaturePrintExtensions.PushExtensions(&registeredExtensionPrint);
        }

        if (DeviceProperties.size() <= uiIndx)
            continue;

        const PhysicalDeviceProperties& Properties = DeviceProperties[uiIndx];

        // **********************************
        // Properties
        // **********************************

        LOGI("Properties: ");
        uint32_t MajorVersion = VK_VERSION_MAJOR( Properties.Base.properties.apiVersion);
        uint32_t MinorVersion = VK_VERSION_MINOR( Properties.Base.properties.apiVersion);
        uint32_t PatchVersion = VK_VERSION_PATCH( Properties.Base.properties.apiVersion);
        LOGI("    apiVersion: %d.%d.%d", MajorVersion, MinorVersion, PatchVersion);

        uint32_t DriverMajorVersion = VK_VERSION_MAJOR( Properties.Base.properties.driverVersion);
        uint32_t DriverMinorVersion = VK_VERSION_MINOR( Properties.Base.properties.driverVersion);
        uint32_t DriverPatchVersion = VK_VERSION_PATCH( Properties.Base.properties.driverVersion);
        LOGI("    driverVersion: %d.%d.%d", DriverMajorVersion, DriverMinorVersion, DriverPatchVersion);

        LOGI("    vendorID: %d", Properties.Base.properties.vendorID);
        LOGI("    deviceID: %d", Properties.Base.properties.deviceID);

        switch (Properties.Base.properties.deviceType)
        {
        case VK_PHYSICAL_DEVICE_TYPE_OTHER:
            LOGI("    deviceType: VK_PHYSICAL_DEVICE_TYPE_OTHER");
            break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            LOGI("    deviceType: VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU");
            break;
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            LOGI("    deviceType: VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU");
            break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            LOGI("    deviceType: VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU");
            break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            LOGI("    deviceType: VK_PHYSICAL_DEVICE_TYPE_CPU");
            break;

        default:
            LOGI("    deviceType: Unknown! (%d)", Properties.Base.properties.deviceType);
            break;
        }

        LOGI("    deviceName: %s", Properties.Base.properties.deviceName);

        LOGI("    limits:");
        LOGI("        maxImageDimension1D: %d", Properties.Base.properties.limits.maxImageDimension1D);
        LOGI("        maxImageDimension2D: %d", Properties.Base.properties.limits.maxImageDimension2D);
        LOGI("        maxImageDimension3D: %d", Properties.Base.properties.limits.maxImageDimension3D);
        LOGI("        maxImageDimensionCube: %d", Properties.Base.properties.limits.maxImageDimensionCube);
        LOGI("        maxImageArrayLayers: %d", Properties.Base.properties.limits.maxImageArrayLayers);
        LOGI("        maxTexelBufferElements: %d", Properties.Base.properties.limits.maxTexelBufferElements);
        LOGI("        maxUniformBufferRange: %d", Properties.Base.properties.limits.maxUniformBufferRange);
        LOGI("        maxStorageBufferRange: %d", Properties.Base.properties.limits.maxStorageBufferRange);
        LOGI("        maxPushConstantsSize: %d", Properties.Base.properties.limits.maxPushConstantsSize);
        LOGI("        maxMemoryAllocationCount: %d", Properties.Base.properties.limits.maxMemoryAllocationCount);
        LOGI("        maxSamplerAllocationCount: %d", Properties.Base.properties.limits.maxSamplerAllocationCount);
        LOGI("        bufferImageGranularity: %zu", Properties.Base.properties.limits.bufferImageGranularity);
        LOGI("        sparseAddressSpaceSize: %zu", Properties.Base.properties.limits.sparseAddressSpaceSize);
        LOGI("        maxBoundDescriptorSets: %d", Properties.Base.properties.limits.maxBoundDescriptorSets);
        LOGI("        maxPerStageDescriptorSamplers: %d", Properties.Base.properties.limits.maxPerStageDescriptorSamplers);
        LOGI("        maxPerStageDescriptorUniformBuffers: %d", Properties.Base.properties.limits.maxPerStageDescriptorUniformBuffers);
        LOGI("        maxPerStageDescriptorStorageBuffers: %d", Properties.Base.properties.limits.maxPerStageDescriptorStorageBuffers);
        LOGI("        maxPerStageDescriptorSampledImages: %d", Properties.Base.properties.limits.maxPerStageDescriptorSampledImages);
        LOGI("        maxPerStageDescriptorStorageImages: %d", Properties.Base.properties.limits.maxPerStageDescriptorStorageImages);
        LOGI("        maxPerStageDescriptorInputAttachments: %d", Properties.Base.properties.limits.maxPerStageDescriptorInputAttachments);
        LOGI("        maxPerStageResources: %d", Properties.Base.properties.limits.maxPerStageResources);
        LOGI("        maxDescriptorSetSamplers: %d", Properties.Base.properties.limits.maxDescriptorSetSamplers);
        LOGI("        maxDescriptorSetUniformBuffers: %d", Properties.Base.properties.limits.maxDescriptorSetUniformBuffers);
        LOGI("        maxDescriptorSetUniformBuffersDynamic: %d", Properties.Base.properties.limits.maxDescriptorSetUniformBuffersDynamic);
        LOGI("        maxDescriptorSetStorageBuffers: %d", Properties.Base.properties.limits.maxDescriptorSetStorageBuffers);
        LOGI("        maxDescriptorSetStorageBuffersDynamic: %d", Properties.Base.properties.limits.maxDescriptorSetStorageBuffersDynamic);
        LOGI("        maxDescriptorSetSampledImages: %d", Properties.Base.properties.limits.maxDescriptorSetSampledImages);
        LOGI("        maxDescriptorSetStorageImages: %d", Properties.Base.properties.limits.maxDescriptorSetStorageImages);
        LOGI("        maxDescriptorSetInputAttachments: %d", Properties.Base.properties.limits.maxDescriptorSetInputAttachments);
        LOGI("        maxVertexInputAttributes: %d", Properties.Base.properties.limits.maxVertexInputAttributes);
        LOGI("        maxVertexInputBindings: %d", Properties.Base.properties.limits.maxVertexInputBindings);
        LOGI("        maxVertexInputAttributeOffset: %d", Properties.Base.properties.limits.maxVertexInputAttributeOffset);
        LOGI("        maxVertexInputBindingStride: %d", Properties.Base.properties.limits.maxVertexInputBindingStride);
        LOGI("        maxVertexOutputComponents: %d", Properties.Base.properties.limits.maxVertexOutputComponents);
        LOGI("        maxTessellationGenerationLevel: %d", Properties.Base.properties.limits.maxTessellationGenerationLevel);
        LOGI("        maxTessellationPatchSize: %d", Properties.Base.properties.limits.maxTessellationPatchSize);
        LOGI("        maxTessellationControlPerVertexInputComponents: %d", Properties.Base.properties.limits.maxTessellationControlPerVertexInputComponents);
        LOGI("        maxTessellationControlPerVertexOutputComponents: %d", Properties.Base.properties.limits.maxTessellationControlPerVertexOutputComponents);
        LOGI("        maxTessellationControlPerPatchOutputComponents: %d", Properties.Base.properties.limits.maxTessellationControlPerPatchOutputComponents);
        LOGI("        maxTessellationControlTotalOutputComponents: %d", Properties.Base.properties.limits.maxTessellationControlTotalOutputComponents);
        LOGI("        maxTessellationEvaluationInputComponents: %d", Properties.Base.properties.limits.maxTessellationEvaluationInputComponents);
        LOGI("        maxTessellationEvaluationOutputComponents: %d", Properties.Base.properties.limits.maxTessellationEvaluationOutputComponents);
        LOGI("        maxGeometryShaderInvocations: %d", Properties.Base.properties.limits.maxGeometryShaderInvocations);
        LOGI("        maxGeometryInputComponents: %d", Properties.Base.properties.limits.maxGeometryInputComponents);
        LOGI("        maxGeometryOutputComponents: %d", Properties.Base.properties.limits.maxGeometryOutputComponents);
        LOGI("        maxGeometryOutputVertices: %d", Properties.Base.properties.limits.maxGeometryOutputVertices);
        LOGI("        maxGeometryTotalOutputComponents: %d", Properties.Base.properties.limits.maxGeometryTotalOutputComponents);
        LOGI("        maxFragmentInputComponents: %d", Properties.Base.properties.limits.maxFragmentInputComponents);
        LOGI("        maxFragmentOutputAttachments: %d", Properties.Base.properties.limits.maxFragmentOutputAttachments);
        LOGI("        maxFragmentDualSrcAttachments: %d", Properties.Base.properties.limits.maxFragmentDualSrcAttachments);
        LOGI("        maxFragmentCombinedOutputResources: %d", Properties.Base.properties.limits.maxFragmentCombinedOutputResources);
        LOGI("        maxComputeSharedMemorySize: %d", Properties.Base.properties.limits.maxComputeSharedMemorySize);
        LOGI("        maxComputeWorkGroupCount: [%d, %d, %d]", Properties.Base.properties.limits.maxComputeWorkGroupCount[0], Properties.Base.properties.limits.maxComputeWorkGroupCount[1], Properties.Base.properties.limits.maxComputeWorkGroupCount[2]);
        LOGI("        maxComputeWorkGroupInvocations: %d", Properties.Base.properties.limits.maxComputeWorkGroupInvocations);
        LOGI("        maxComputeWorkGroupSize: [%d, %d, %d]", Properties.Base.properties.limits.maxComputeWorkGroupSize[0], Properties.Base.properties.limits.maxComputeWorkGroupSize[1], Properties.Base.properties.limits.maxComputeWorkGroupSize[2]);
        LOGI("        subPixelPrecisionBits: %d", Properties.Base.properties.limits.subPixelPrecisionBits);
        LOGI("        subTexelPrecisionBits: %d", Properties.Base.properties.limits.subTexelPrecisionBits);
        LOGI("        mipmapPrecisionBits: %d", Properties.Base.properties.limits.mipmapPrecisionBits);
        LOGI("        maxDrawIndexedIndexValue: %d", Properties.Base.properties.limits.maxDrawIndexedIndexValue);
        LOGI("        maxDrawIndirectCount: %d", Properties.Base.properties.limits.maxDrawIndirectCount);
        LOGI("        maxSamplerLodBias: %f", Properties.Base.properties.limits.maxSamplerLodBias);
        LOGI("        maxSamplerAnisotropy: %f", Properties.Base.properties.limits.maxSamplerAnisotropy);
        LOGI("        maxViewports: %d", Properties.Base.properties.limits.maxViewports);
        LOGI("        maxViewportDimensions: [%d, %d]", Properties.Base.properties.limits.maxViewportDimensions[0], Properties.Base.properties.limits.maxViewportDimensions[1]);
        LOGI("        viewportBoundsRange: [%f, %f]", Properties.Base.properties.limits.viewportBoundsRange[0], Properties.Base.properties.limits.viewportBoundsRange[1]);
        LOGI("        viewportSubPixelBits: %d", Properties.Base.properties.limits.viewportSubPixelBits);
        LOGI("        minMemoryMapAlignment: %zu", Properties.Base.properties.limits.minMemoryMapAlignment);
        LOGI("        minTexelBufferOffsetAlignment: %zu", Properties.Base.properties.limits.minTexelBufferOffsetAlignment);
        LOGI("        minUniformBufferOffsetAlignment: %zu", Properties.Base.properties.limits.minUniformBufferOffsetAlignment);
        LOGI("        minStorageBufferOffsetAlignment: %zu", Properties.Base.properties.limits.minStorageBufferOffsetAlignment);
        LOGI("        minTexelOffset: %d", Properties.Base.properties.limits.minTexelOffset);
        LOGI("        maxTexelOffset: %d", Properties.Base.properties.limits.maxTexelOffset);
        LOGI("        minTexelGatherOffset: %d", Properties.Base.properties.limits.minTexelGatherOffset);
        LOGI("        maxTexelGatherOffset: %d", Properties.Base.properties.limits.maxTexelGatherOffset);
        LOGI("        minInterpolationOffset: %f", Properties.Base.properties.limits.minInterpolationOffset);
        LOGI("        maxInterpolationOffset: %f", Properties.Base.properties.limits.maxInterpolationOffset);
        LOGI("        subPixelInterpolationOffsetBits: %d", Properties.Base.properties.limits.subPixelInterpolationOffsetBits);
        LOGI("        maxFramebufferWidth: %d", Properties.Base.properties.limits.maxFramebufferWidth);
        LOGI("        maxFramebufferHeight: %d", Properties.Base.properties.limits.maxFramebufferHeight);
        LOGI("        maxFramebufferLayers: %d", Properties.Base.properties.limits.maxFramebufferLayers);
        LOGI("        framebufferColorSampleCounts: %d", Properties.Base.properties.limits.framebufferColorSampleCounts);
        LOGI("        framebufferDepthSampleCounts: %d", Properties.Base.properties.limits.framebufferDepthSampleCounts);
        LOGI("        framebufferStencilSampleCounts: %d", Properties.Base.properties.limits.framebufferStencilSampleCounts);
        LOGI("        framebufferNoAttachmentsSampleCounts: %d", Properties.Base.properties.limits.framebufferNoAttachmentsSampleCounts);
        LOGI("        maxColorAttachments: %d", Properties.Base.properties.limits.maxColorAttachments);
        LOGI("        sampledImageColorSampleCounts: %d", Properties.Base.properties.limits.sampledImageColorSampleCounts);
        LOGI("        sampledImageIntegerSampleCounts: %d", Properties.Base.properties.limits.sampledImageIntegerSampleCounts);
        LOGI("        sampledImageDepthSampleCounts: %d", Properties.Base.properties.limits.sampledImageDepthSampleCounts);
        LOGI("        sampledImageStencilSampleCounts: %d", Properties.Base.properties.limits.sampledImageStencilSampleCounts);
        LOGI("        storageImageSampleCounts: %d", Properties.Base.properties.limits.storageImageSampleCounts);
        LOGI("        maxSampleMaskWords: %d", Properties.Base.properties.limits.maxSampleMaskWords);
        LOGI("        timestampComputeAndGraphics: %s", Properties.Base.properties.limits.timestampComputeAndGraphics ? "True" : "False");
        LOGI("        timestampPeriod: %f", Properties.Base.properties.limits.timestampPeriod);
        LOGI("        maxClipDistances: %d", Properties.Base.properties.limits.maxClipDistances);
        LOGI("        maxCullDistances: %d", Properties.Base.properties.limits.maxCullDistances);
        LOGI("        maxCombinedClipAndCullDistances: %d", Properties.Base.properties.limits.maxCombinedClipAndCullDistances);
        LOGI("        discreteQueuePriorities: %d", Properties.Base.properties.limits.discreteQueuePriorities);
        LOGI("        pointSizeRange: [%f, %f]", Properties.Base.properties.limits.pointSizeRange[0], Properties.Base.properties.limits.pointSizeRange[1]);
        LOGI("        lineWidthRange: [%f, %f]", Properties.Base.properties.limits.lineWidthRange[0], Properties.Base.properties.limits.lineWidthRange[1]);
        LOGI("        pointSizeGranularity: %f", Properties.Base.properties.limits.pointSizeGranularity);
        LOGI("        lineWidthGranularity: %f", Properties.Base.properties.limits.lineWidthGranularity);
        LOGI("        strictLines: %s", Properties.Base.properties.limits.strictLines ? "True" : "False");
        LOGI("        standardSampleLocations: %s", Properties.Base.properties.limits.standardSampleLocations ? "True" : "False");
        LOGI("        optimalBufferCopyOffsetAlignment: %zu", Properties.Base.properties.limits.optimalBufferCopyOffsetAlignment);
        LOGI("        optimalBufferCopyRowPitchAlignment: %zu", Properties.Base.properties.limits.optimalBufferCopyRowPitchAlignment);
        LOGI("        nonCoherentAtomSize: %zu", Properties.Base.properties.limits.nonCoherentAtomSize);

        // Display features from all the registered extensions
        {
            m_VulkanDevicePropertiesPrintExtensions.PushExtensions( nullptr );
        }
    }
}

//-----------------------------------------------------------------------------
void Vulkan::ReportFramebufferProperties( VkFramebuffer vkFrameBuffer ) const
//-----------------------------------------------------------------------------
{
    if (m_ExtQcomTileProperties && m_ExtQcomTileProperties->Status == VulkanExtensionStatus::eLoaded)
    {
        uint32_t propertiesCount = 0;
        m_ExtQcomTileProperties->m_vkGetFramebufferTilePropertiesQCOM( m_VulkanDevice, vkFrameBuffer, &propertiesCount, nullptr );
        if (propertiesCount > 0)
        {
            std::vector< VkTilePropertiesQCOM> properties;
            properties.resize(propertiesCount, {VK_STRUCTURE_TYPE_TILE_PROPERTIES_QCOM});
            m_ExtQcomTileProperties->m_vkGetFramebufferTilePropertiesQCOM( m_VulkanDevice, vkFrameBuffer, &propertiesCount, properties.data() );
            for (const auto& prop : properties)
            {
                LOGI("Framebuffer properties: %ux%ux%u (origin %ux%u; apron %ux%u)", prop.tileSize.width, prop.tileSize.height, prop.tileSize.depth, prop.origin.x, prop.origin.y, prop.apronSize.width, prop.apronSize.height);
            }
        }
    }
}

