//============================================================================================================
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

//============================================================================================================
//
// Common types, helpers, and shared OpenCL/Vulkan state for all VK-CL interop modes.
//
//============================================================================================================

#pragma once

#include "main/applicationHelperBase.hpp"
#include "vulkan/vulkan.hpp"
#include "vulkan/commandBuffer.hpp"
#include <memory>
#include <array>
#include <vector>
#include <string>

#define OPENCL_ENABLE_FUNCTION_REMAP
#include "../open_cl_common/open_cl_utils.h"

class GraphicsApiBase;
class AssetManager;

enum INTEROP_MODE
{
    INTEROP_NONE = 0,
    INTEROP_BUFFER,
    INTEROP_LINEAR_IMAGE,
    INTEROP_OPTIMAL_IMAGE
};

// ---------------------------------------------------------------------------
// OpenCL error helpers
// ---------------------------------------------------------------------------
inline std::string ClErrorToString(cl_int err)
{
    switch (err)
    {
    case CL_SUCCESS:                  return "CL_SUCCESS";
    case CL_INVALID_VALUE:            return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE:           return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:          return "CL_INVALID_CONTEXT";
    case CL_INVALID_COMMAND_QUEUE:    return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_MEM_OBJECT:       return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_PROGRAM:          return "CL_INVALID_PROGRAM";
    case CL_INVALID_KERNEL:           return "CL_INVALID_KERNEL";
    case CL_INVALID_KERNEL_ARGS:      return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:   return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:  return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case CL_OUT_OF_RESOURCES:         return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:       return "CL_OUT_OF_HOST_MEMORY";
    case CL_BUILD_PROGRAM_FAILURE:    return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_BINARY:           return "CL_INVALID_BINARY";
    default:
        return "CL_UNKNOWN_ERROR(" + std::to_string(err) + ")";
    }
}

#define CL_CHECK(expr) \
{ \
    cl_int _cl_err = static_cast<cl_int>(expr); \
    if (_cl_err != CL_SUCCESS) \
    { \
        LOGE("CL error at %s:%d: %s", __FILE__, __LINE__, ClErrorToString(_cl_err).c_str()); \
        throw std::runtime_error(std::string("CL error: ") + ClErrorToString(_cl_err)); \
    } \
}

// ---------------------------------------------------------------------------
// Vulkan external memory/semaphore function pointers
// Loaded once by InitVkFunctionPointers(); shared by all interop modes.
// ---------------------------------------------------------------------------
extern PFN_vkGetMemoryFdKHR       fpGetMemoryFdKHR;
extern PFN_vkGetSemaphoreFdKHR    fpGetSemaphoreFdKHR;
extern PFN_vkImportSemaphoreFdKHR fpImportSemaphoreFdKHR;

/// Load VK external memory/semaphore function pointers from the given device.
/// Must be called once before using any interop mode.
bool InitVkFunctionPointers(VkDevice device);

// ---------------------------------------------------------------------------
// Vulkan helper functions
// ---------------------------------------------------------------------------

/// Find a Vulkan memory type index satisfying the given property flags.
uint32_t GetMemoryType(
    VkPhysicalDevice      physical_device,
    uint32_t              bits,
    VkMemoryPropertyFlags properties);

bool SupportsLinearFiltering(
    VkPhysicalDevice physical_device,
    VkFormat         format,
    VkImageTiling    tiling);

/// Insert a VkImageMemoryBarrier into the command buffer.
void ImageLayoutTransition(
    VkCommandBuffer                command_buffer,
    VkImage                        image,
    VkPipelineStageFlags           src_stage_mask,
    VkPipelineStageFlags           dst_stage_mask,
    VkAccessFlags                  src_access_mask,
    VkAccessFlags                  dst_access_mask,
    VkImageLayout                  old_layout,
    VkImageLayout                  new_layout,
    VkImageSubresourceRange const& subresource_range);

void BufferQueueFamilyOwnershipTransfer(
    VkCommandBuffer       command_buffer,
    VkBuffer              buffer,
    VkDeviceSize          offset,
    VkDeviceSize          size,
    uint32_t              src_queue_family,
    uint32_t              dst_queue_family,
    VkAccessFlags         src_access_mask,
    VkAccessFlags         dst_access_mask,
    VkPipelineStageFlags  src_stage_mask,
    VkPipelineStageFlags  dst_stage_mask);

void ImageQueueFamilyOwnershipTransfer(
    VkCommandBuffer                command_buffer,
    VkImage                        image,
    VkImageLayout                  old_layout,
    VkImageLayout                  new_layout,
    VkImageSubresourceRange const& subresource_range,
    uint32_t                       src_queue_family,
    uint32_t                       dst_queue_family,
    VkAccessFlags                  src_access_mask,
    VkAccessFlags                  dst_access_mask,
    VkPipelineStageFlags           src_stage_mask,
    VkPipelineStageFlags           dst_stage_mask);

VkExternalMemoryFeatureFlags GetExternalBufferMemoryFeatures(
    VkPhysicalDevice                     physical_device,
    VkBufferUsageFlags                   usage,
    VkExternalMemoryHandleTypeFlagBits   handle_type);

VkExternalMemoryFeatureFlags GetExternalImageMemoryFeatures(
    VkPhysicalDevice                     physical_device,
    VkFormat                             format,
    VkImageTiling                        tiling,
    VkImageUsageFlags                    usage,
    VkExternalMemoryHandleTypeFlagBits   handle_type);

// ---------------------------------------------------------------------------
// Shared OpenCL execution state + VK<->CL synchronization semaphores.
// Owned by Application; passed by reference to each interop context.
// ---------------------------------------------------------------------------
struct CLState
{
    // OpenCL execution state
    cl_device_id     device{nullptr};
    cl_context       context{nullptr};
    cl_command_queue queue{nullptr};

    // VK<->CL synchronization semaphores
    VkSemaphore      vk_sema_vk_to_cl{VK_NULL_HANDLE};  // VK signals → CL waits
    VkSemaphore      vk_sema_cl_to_vk{VK_NULL_HANDLE};  // CL signals → VK waits
    cl_semaphore_khr cl_sema_vk_to_cl{nullptr};
    cl_semaphore_khr cl_sema_cl_to_vk{nullptr};

    /// Select the OpenCL device matching the Vulkan device UUID,
    /// create cl_context + cl_command_queue, and create interop semaphores.
    bool Initialize(GraphicsApiBase& gfxApi, VkDevice vk_device);

    /// Release semaphores, queue, and context.
    void Release(VkDevice vk_device);

    /// Export VK semaphore as sync fd and re-import into CL semaphore.
    /// Call before CL dispatch (VK has finished rendering).
    void ExportVkSemaToCl(VkDevice device);

    /// Export CL semaphore as sync fd and import into VK semaphore.
    /// Call after CL dispatch (CL has finished processing).
    void ExportClSemaToVk(VkDevice device);

    /// Semaphore accessors for application.cpp
    VkSemaphore GetRenderingCompleteSemaphore()  { return vk_sema_vk_to_cl; }
    VkSemaphore GetProcessingCompleteSemaphore() { return vk_sema_cl_to_vk; }

    /// Enqueue wait on the VK→CL semaphore (call at the start of CL dispatch).
    void WaitForVkSignal();

    /// Enqueue signal on the CL→VK semaphore (call at the end of CL dispatch).
    void SignalVkWait();

    /// Flush the command queue (submit enqueued commands to the GPU).
    void Flush();
};
