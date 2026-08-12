//============================================================================================================
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

//============================================================================================================
//
// Shared VK-CL interop helpers: CLState, device selection, VK helpers.
//
//============================================================================================================

#include "interop_common.hpp"
#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// VK function pointer definitions
// ---------------------------------------------------------------------------
PFN_vkGetMemoryFdKHR       fpGetMemoryFdKHR       = nullptr;
PFN_vkGetSemaphoreFdKHR    fpGetSemaphoreFdKHR    = nullptr;
PFN_vkImportSemaphoreFdKHR fpImportSemaphoreFdKHR = nullptr;

bool InitVkFunctionPointers(VkDevice device)
{
    fpGetMemoryFdKHR = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
    fpGetSemaphoreFdKHR = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR"));
    fpImportSemaphoreFdKHR = reinterpret_cast<PFN_vkImportSemaphoreFdKHR>(
        vkGetDeviceProcAddr(device, "vkImportSemaphoreFdKHR"));

    if (!fpGetMemoryFdKHR)       { LOGE("Unable to get function pointer: vkGetMemoryFdKHR");       return false; }
    if (!fpGetSemaphoreFdKHR)    { LOGE("Unable to get function pointer: vkGetSemaphoreFdKHR");    return false; }
    if (!fpImportSemaphoreFdKHR) { LOGE("Unable to get function pointer: vkImportSemaphoreFdKHR"); return false; }
    return true;
}

// ---------------------------------------------------------------------------
// helper functions
// ---------------------------------------------------------------------------

uint32_t GetMemoryType(
    VkPhysicalDevice      physical_device,
    uint32_t              bits,
    VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
    {
        if ((bits & 1) == 1)
        {
            if ((memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }
        bits >>= 1;
    }

    throw std::runtime_error("Could not find a matching memory type");
}

bool SupportsLinearFiltering(
    VkPhysicalDevice physical_device,
    VkFormat         format,
    VkImageTiling    tiling)
{
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);

    const VkFormatFeatureFlags tiling_features =
        (tiling == VK_IMAGE_TILING_LINEAR)
            ? properties.linearTilingFeatures
            : properties.optimalTilingFeatures;

    return (tiling_features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
}

void ImageLayoutTransition(
    VkCommandBuffer               command_buffer,
    VkImage                       image,
    VkPipelineStageFlags          src_stage_mask,
    VkPipelineStageFlags          dst_stage_mask,
    VkAccessFlags                 src_access_mask,
    VkAccessFlags                 dst_access_mask,
    VkImageLayout                 old_layout,
    VkImageLayout                 new_layout,
    VkImageSubresourceRange const& subresource_range)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask       = src_access_mask;
    barrier.dstAccessMask       = dst_access_mask;
    barrier.oldLayout           = old_layout;
    barrier.newLayout           = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image;
    barrier.subresourceRange    = subresource_range;
    vkCmdPipelineBarrier(command_buffer, src_stage_mask, dst_stage_mask,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void BufferQueueFamilyOwnershipTransfer(
    VkCommandBuffer      command_buffer,
    VkBuffer             buffer,
    VkDeviceSize         offset,
    VkDeviceSize         size,
    uint32_t             src_queue_family,
    uint32_t             dst_queue_family,
    VkAccessFlags        src_access_mask,
    VkAccessFlags        dst_access_mask,
    VkPipelineStageFlags src_stage_mask,
    VkPipelineStageFlags dst_stage_mask)
{
    VkBufferMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask       = src_access_mask;
    barrier.dstAccessMask       = dst_access_mask;
    barrier.srcQueueFamilyIndex = src_queue_family;
    barrier.dstQueueFamilyIndex = dst_queue_family;
    barrier.buffer              = buffer;
    barrier.offset              = offset;
    barrier.size                = size;

    vkCmdPipelineBarrier(command_buffer, src_stage_mask, dst_stage_mask,
                         0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void ImageQueueFamilyOwnershipTransfer(
    VkCommandBuffer               command_buffer,
    VkImage                       image,
    VkImageLayout                 old_layout,
    VkImageLayout                 new_layout,
    VkImageSubresourceRange const& subresource_range,
    uint32_t                      src_queue_family,
    uint32_t                      dst_queue_family,
    VkAccessFlags                 src_access_mask,
    VkAccessFlags                 dst_access_mask,
    VkPipelineStageFlags          src_stage_mask,
    VkPipelineStageFlags          dst_stage_mask)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask       = src_access_mask;
    barrier.dstAccessMask       = dst_access_mask;
    barrier.oldLayout           = old_layout;
    barrier.newLayout           = new_layout;
    barrier.srcQueueFamilyIndex = src_queue_family;
    barrier.dstQueueFamilyIndex = dst_queue_family;
    barrier.image               = image;
    barrier.subresourceRange    = subresource_range;

    vkCmdPipelineBarrier(command_buffer, src_stage_mask, dst_stage_mask,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
}

VkExternalMemoryFeatureFlags GetExternalBufferMemoryFeatures(
    VkPhysicalDevice                   physical_device,
    VkBufferUsageFlags                 usage,
    VkExternalMemoryHandleTypeFlagBits handle_type)
{
    VkPhysicalDeviceExternalBufferInfo external_buffer_info{};
    external_buffer_info.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    external_buffer_info.handleType = handle_type;
    external_buffer_info.usage      = usage;

    VkExternalBufferProperties external_buffer_props{};
    external_buffer_props.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;

    vkGetPhysicalDeviceExternalBufferProperties(
        physical_device,
        &external_buffer_info,
        &external_buffer_props);

    return external_buffer_props.externalMemoryProperties.externalMemoryFeatures;
}

VkExternalMemoryFeatureFlags GetExternalImageMemoryFeatures(
    VkPhysicalDevice                   physical_device,
    VkFormat                           format,
    VkImageTiling                      tiling,
    VkImageUsageFlags                  usage,
    VkExternalMemoryHandleTypeFlagBits handle_type)
{
    VkPhysicalDeviceExternalImageFormatInfo external_image_format_info{};
    external_image_format_info.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
    external_image_format_info.handleType = handle_type;

    VkPhysicalDeviceImageFormatInfo2 image_format_info{};
    image_format_info.sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
    image_format_info.pNext  = &external_image_format_info;
    image_format_info.format = format;
    image_format_info.type   = VK_IMAGE_TYPE_2D;
    image_format_info.tiling = tiling;
    image_format_info.usage  = usage;

    VkExternalImageFormatProperties external_image_format_props{};
    external_image_format_props.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;

    VkImageFormatProperties2 image_format_props{};
    image_format_props.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
    image_format_props.pNext = &external_image_format_props;

    VkResult image_format_result = vkGetPhysicalDeviceImageFormatProperties2(
        physical_device,
        &image_format_info,
        &image_format_props);

    if (image_format_result != VK_SUCCESS)
        throw std::runtime_error("Vulkan image format does not support opaque fd external memory handle type");

    return external_image_format_props.externalMemoryProperties.externalMemoryFeatures;
}

// ---------------------------------------------------------------------------
// Capability validation functions
// ---------------------------------------------------------------------------

template <typename T>
static std::vector<T> GetClDeviceInfoList(
    cl_device_id  device,
    cl_device_info param_name,
    const char*   param_name_str)
{
    size_t value_size = 0;
    CL_CHECK(clGetDeviceInfo(device, param_name, 0, nullptr, &value_size));

    if ((value_size % sizeof(T)) != 0)
    {
        LOGE("Unexpected %s size: %zu", param_name_str, value_size);
        return {};
    }

    std::vector<T> values(value_size / sizeof(T));
    if (!values.empty())
    {
        CL_CHECK(clGetDeviceInfo(device, param_name, value_size, values.data(), nullptr));
    }

    return values;
}

template <typename T>
static bool ContainsValue(const std::vector<T>& values, T wanted)
{
    return std::find(values.begin(), values.end(), wanted) != values.end();
}

// Check Vulkan/OpenCL sync-fd semaphore import/export support
static bool ValidateSyncFdSemaphoreSupport(
    VkPhysicalDevice vk_physical_device,
    cl_device_id     cl_device)
{
    // Check Vulkan sync-fd semaphore import/export support
    VkPhysicalDeviceExternalSemaphoreInfo semaphore_info{};
    semaphore_info.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
    semaphore_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;

    VkExternalSemaphoreProperties semaphore_props{};
    semaphore_props.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;

    vkGetPhysicalDeviceExternalSemaphoreProperties(vk_physical_device,
                                                   &semaphore_info,
                                                   &semaphore_props);

    const VkExternalSemaphoreFeatureFlags vk_features = semaphore_props.externalSemaphoreFeatures;

    const bool vk_sync_fd_importable = (vk_features & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) != 0;
    if (!vk_sync_fd_importable)
    {
        LOGE("Verify failed: Vulkan sync fd semaphore is not importable.");
        return false;
    }

    const bool vk_sync_fd_exportable = (vk_features & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) != 0;
    if (!vk_sync_fd_exportable)
    {
        LOGE("Verify failed: Vulkan sync fd semaphore is not exportable.");
        return false;
    }

    // Check OpenCL sync-fd semaphore import/export handle types
    const auto cl_import_handle_types = GetClDeviceInfoList<cl_external_semaphore_handle_type_khr>(
        cl_device,
        CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR,
        "CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR");

    const bool cl_sync_fd_importable = ContainsValue<cl_external_semaphore_handle_type_khr>(cl_import_handle_types, CL_SEMAPHORE_HANDLE_SYNC_FD_KHR);
    if (!cl_sync_fd_importable)
    {
        LOGE("Verify failed: CL_SEMAPHORE_HANDLE_SYNC_FD_KHR is not in CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR.");
        return false;
    }

    const auto cl_export_handle_types = GetClDeviceInfoList<cl_external_semaphore_handle_type_khr>(
        cl_device,
        CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR,
        "CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR");

    const bool cl_sync_fd_exportable = ContainsValue<cl_external_semaphore_handle_type_khr>(cl_export_handle_types, CL_SEMAPHORE_HANDLE_SYNC_FD_KHR);
    if (!cl_sync_fd_exportable)
    {
        LOGE("Verify failed: CL_SEMAPHORE_HANDLE_SYNC_FD_KHR is not in CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR.");
        return false;
    }

    return true;
}

// Check OpenCL external memory import handle types used by this sample
static bool ValidateExternalMemoryImportHandleTypes(cl_device_id device)
{
    const auto handle_types = GetClDeviceInfoList<cl_external_memory_handle_type_khr>(
        device,
        CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR,
        "CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR");

    const bool has_opaque_fd = ContainsValue<cl_external_memory_handle_type_khr>(handle_types, CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR);
    if (!has_opaque_fd)
    {
        LOGE("Verify failed: CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR is not in CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR.");
        return false;
    }

    const bool has_qcom_vulkan_opaque_fd = ContainsValue<cl_external_memory_handle_type_khr>(handle_types, CL_EXTERNAL_MEMORY_HANDLE_VULKAN_OPAQUE_FD_QCOM);
    if (!has_qcom_vulkan_opaque_fd)
    {
        LOGE("Verify failed: CL_EXTERNAL_MEMORY_HANDLE_VULKAN_OPAQUE_FD_QCOM is not in CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR.");
        return false;
    }

    return true;
}

// Check OpenCL assume-linear image import handle type constraints
static bool ValidateAssumeLinearExternalMemoryHandleTypes(cl_device_id device)
{
    const auto handle_types = GetClDeviceInfoList<cl_external_memory_handle_type_khr>(
        device,
        CL_DEVICE_EXTERNAL_MEMORY_IMPORT_ASSUME_LINEAR_IMAGES_HANDLE_TYPES_KHR,
        "CL_DEVICE_EXTERNAL_MEMORY_IMPORT_ASSUME_LINEAR_IMAGES_HANDLE_TYPES_KHR");

    const bool has_opaque_fd = ContainsValue<cl_external_memory_handle_type_khr>(handle_types, CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR);
    if (!has_opaque_fd)
    {
        LOGE("Verify failed: CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR is not in CL_DEVICE_EXTERNAL_MEMORY_IMPORT_ASSUME_LINEAR_IMAGES_HANDLE_TYPES_KHR.");
        return false;
    }

    const bool has_qcom_vulkan_opaque_fd = ContainsValue<cl_external_memory_handle_type_khr>(handle_types, CL_EXTERNAL_MEMORY_HANDLE_VULKAN_OPAQUE_FD_QCOM);
    if (has_qcom_vulkan_opaque_fd)
    {
        LOGE("Verify failed: CL_EXTERNAL_MEMORY_HANDLE_VULKAN_OPAQUE_FD_QCOM is in CL_DEVICE_EXTERNAL_MEMORY_IMPORT_ASSUME_LINEAR_IMAGES_HANDLE_TYPES_KHR.");
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Internal: select OpenCL device matching Vulkan device UUID and driver UUID
// ---------------------------------------------------------------------------
static cl_device_id SelectCLDevice(GraphicsApiBase& gfxApi)
{
    auto& vulkan = static_cast<Vulkan&>(gfxApi);

    VkPhysicalDeviceIDPropertiesKHR id_props{};
    id_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
    props2.pNext = &id_props;

    vkGetPhysicalDeviceProperties2(vulkan.m_VulkanGpu, &props2);

    const std::vector<std::string> required_extensions{
        CL_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        CL_KHR_EXTERNAL_MEMORY_OPAQUE_FD_EXTENSION_NAME,
        "cl_qcom_external_memory_vulkan_opaque_fd",
        CL_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        CL_KHR_EXTERNAL_SEMAPHORE_SYNC_FD_EXTENSION_NAME,
        CL_KHR_DEVICE_UUID_EXTENSION_NAME,
    };

    cl_uint num_platforms = 0;
    clGetPlatformIDs(0, nullptr, &num_platforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    cl_platform_id selected_platform{nullptr};
    cl_device_id   selected_device{nullptr};

    for (auto& platform : platforms)
    {
        size_t ext_size = 0;
        clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, 0, nullptr, &ext_size);
        std::string ext_str(ext_size, '\0');
        clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, ext_size, &ext_str[0], nullptr);
        ext_str.erase(std::remove(ext_str.begin(), ext_str.end(), '\0'), ext_str.end());

        std::vector<std::string> avail_exts;
        std::istringstream iss(ext_str);
        for (std::string e; iss >> e;) avail_exts.push_back(e);

        bool all_present = std::all_of(required_extensions.begin(), required_extensions.end(),
            [&](const std::string& req)
            {
                return std::find(avail_exts.begin(), avail_exts.end(), req) != avail_exts.end();
            });
        if (!all_present) continue;

        cl_uint num_devices = 0;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);

        for (auto& dev : devices)
        {
            // Vulkan external memory opaque fd requires matching CL/VK device and driver UUIDs.
            // Sync fd semaphore interop does not explicitly require this, but uses the same CL device here.
            std::array<cl_uchar, CL_UUID_SIZE_KHR> device_uuid{};
            clGetDeviceInfo(dev, CL_DEVICE_UUID_KHR, device_uuid.size(), device_uuid.data(), nullptr);

            std::array<cl_uchar, CL_UUID_SIZE_KHR> driver_uuid{};
            clGetDeviceInfo(dev, CL_DRIVER_UUID_KHR, driver_uuid.size(), driver_uuid.data(), nullptr);

            const bool device_uuid_matches = std::equal(
                device_uuid.begin(),
                device_uuid.end(),
                std::begin(id_props.deviceUUID));

            const bool driver_uuid_matches = std::equal(
                driver_uuid.begin(),
                driver_uuid.end(),
                std::begin(id_props.driverUUID));

            if (device_uuid_matches && driver_uuid_matches)
            {
                selected_platform = platform;
                selected_device   = dev;
                break;
            }
        }
        if (selected_device) break;
    }

    if (!selected_platform || !selected_device)
        throw std::runtime_error(
            "Could not find an OpenCL platform + device matching Vulkan device UUID, driver UUID, and required extensions");

    return selected_device;
}

// ---------------------------------------------------------------------------
// CLState
// ---------------------------------------------------------------------------

bool CLState::Initialize(GraphicsApiBase& gfxApi, VkDevice vk_device)
{
    LOGI("******************************");
    LOGI("  Context...");
    LOGI("******************************");

    if (!load_opencl())
    {
        LOGE("Failed to load OpenCL.");
        return false;
    }

    auto& vulkan = static_cast<Vulkan&>(gfxApi);

    device = SelectCLDevice(gfxApi);

    // Validate required VK-CL external memory and semaphore capabilities.
    if (!ValidateSyncFdSemaphoreSupport(vulkan.m_VulkanGpu, device))
        return false;
    if (!ValidateExternalMemoryImportHandleTypes(device))
        return false;
    if (!ValidateAssumeLinearExternalMemoryHandleTypes(device))
        return false;

    cl_int err;

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);

    queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    CL_CHECK(err);

    // Create VK<->CL synchronization semaphores
    VkExportSemaphoreCreateInfo sema_export_info{};
    sema_export_info.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    sema_export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;

    VkSemaphoreCreateInfo import_sema_info{};
    import_sema_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkSemaphoreCreateInfo export_sema_info{};
    export_sema_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    export_sema_info.pNext = &sema_export_info;

    VkResult vk_result;
    vk_result = vkCreateSemaphore(vk_device, &import_sema_info, nullptr, &vk_sema_cl_to_vk);
    if (!CheckVkError("vkCreateSemaphore(cl_to_vk)", vk_result)) return false;

    vk_result = vkCreateSemaphore(vk_device, &export_sema_info, nullptr, &vk_sema_vk_to_cl);
    if (!CheckVkError("vkCreateSemaphore(vk_to_cl)", vk_result)) return false;

    // -1 is a special placeholder here, treated like a valid sync fd 
    // referring to an object that has already signaled
    int init_fd = -1;

    std::vector<cl_semaphore_properties_khr> import_props{
        (cl_semaphore_properties_khr)CL_SEMAPHORE_TYPE_KHR,
        (cl_semaphore_properties_khr)CL_SEMAPHORE_TYPE_BINARY_KHR,
        (cl_semaphore_properties_khr)CL_SEMAPHORE_HANDLE_SYNC_FD_KHR,
        (cl_semaphore_properties_khr)init_fd,
        (cl_semaphore_properties_khr)CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR,
        (cl_semaphore_properties_khr)device,
        (cl_semaphore_properties_khr)CL_SEMAPHORE_DEVICE_HANDLE_LIST_END_KHR,
        0
    };

    std::vector<cl_semaphore_properties_khr> export_props{
        (cl_semaphore_properties_khr)CL_SEMAPHORE_TYPE_KHR,
        (cl_semaphore_properties_khr)CL_SEMAPHORE_TYPE_BINARY_KHR,
        (cl_semaphore_properties_khr)CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR,
        (cl_semaphore_properties_khr)CL_SEMAPHORE_HANDLE_SYNC_FD_KHR,
        (cl_semaphore_properties_khr)CL_SEMAPHORE_EXPORT_HANDLE_TYPES_LIST_END_KHR,
        (cl_semaphore_properties_khr)CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR,
        (cl_semaphore_properties_khr)device,
        (cl_semaphore_properties_khr)CL_SEMAPHORE_DEVICE_HANDLE_LIST_END_KHR,
        0
    };

    cl_int cl_result;
    cl_sema_cl_to_vk = clCreateSemaphoreWithPropertiesKHR(context, export_props.data(), &cl_result);
    CL_CHECK(cl_result);
    cl_sema_vk_to_cl = clCreateSemaphoreWithPropertiesKHR(context, import_props.data(), &cl_result);
    CL_CHECK(cl_result);

    return true;
}

void CLState::Release(VkDevice vk_device)
{
    if (cl_sema_cl_to_vk) { clReleaseSemaphoreKHR(cl_sema_cl_to_vk); cl_sema_cl_to_vk = nullptr; }
    if (cl_sema_vk_to_cl) { clReleaseSemaphoreKHR(cl_sema_vk_to_cl); cl_sema_vk_to_cl = nullptr; }
    if (vk_sema_vk_to_cl != VK_NULL_HANDLE) { vkDestroySemaphore(vk_device, vk_sema_vk_to_cl, nullptr); vk_sema_vk_to_cl = VK_NULL_HANDLE; }
    if (vk_sema_cl_to_vk != VK_NULL_HANDLE) { vkDestroySemaphore(vk_device, vk_sema_cl_to_vk, nullptr); vk_sema_cl_to_vk = VK_NULL_HANDLE; }
    if (queue)   { clReleaseCommandQueue(queue); queue   = nullptr; }
    if (context) { clReleaseContext(context);    context = nullptr; }
    unload_opencl();
    device = nullptr;
}

void CLState::ExportVkSemaToCl(VkDevice vk_device)
{
    int fd;
    VkSemaphoreGetFdInfoKHR fd_info{};
    fd_info.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    fd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    fd_info.semaphore  = vk_sema_vk_to_cl;
    VkResult vk_result = fpGetSemaphoreFdKHR(vk_device, &fd_info, &fd);
    if (!CheckVkError("vkGetSemaphoreFdKHR()", vk_result))
        assert(vk_result == VK_SUCCESS);

    CL_CHECK(clReImportSemaphoreSyncFdKHR(cl_sema_vk_to_cl, nullptr, fd));
}

void CLState::ExportClSemaToVk(VkDevice vk_device)
{
    int fd;
    CL_CHECK(clGetSemaphoreHandleForTypeKHR(cl_sema_cl_to_vk, device, CL_SEMAPHORE_HANDLE_SYNC_FD_KHR, sizeof(int), &fd, nullptr));

    VkImportSemaphoreFdInfoKHR fd_info{};
    fd_info.sType      = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
    fd_info.semaphore  = vk_sema_cl_to_vk;
    fd_info.flags      = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
    fd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    fd_info.fd         = fd;
    VkResult vk_result = fpImportSemaphoreFdKHR(vk_device, &fd_info);
    if (!CheckVkError("vkImportSemaphoreFdKHR()", vk_result))
        assert(vk_result == VK_SUCCESS);
}

void CLState::WaitForVkSignal()
{
    CL_CHECK(clEnqueueWaitSemaphoresKHR(queue, 1, &cl_sema_vk_to_cl, nullptr, 0, nullptr, nullptr));
}

void CLState::SignalVkWait()
{
    CL_CHECK(clEnqueueSignalSemaphoresKHR(queue, 1, &cl_sema_cl_to_vk, nullptr, 0, nullptr, nullptr));
}

void CLState::Flush()
{
    clFlush(queue);
}
