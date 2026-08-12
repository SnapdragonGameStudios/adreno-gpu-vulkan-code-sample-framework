//============================================================================================================
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

//============================================================================================================
//
// Mode 2 VK-CL interop implementation (InteropImageLinear):
//   - Shares VkImage (linear tiling) with OpenCL via opaque fd
//   - VK copies scene color into the shared input VkImage (vkCmdCopyImage)
//   - OpenCL grayscale_image kernel reads from the shared input image,
//     writes to the shared output image
//   - VK copies the shared output image back to a TextureVulkan for the blit pass
//
//============================================================================================================

#include "interop_image_linear.hpp"
#include "vulkan/commandBuffer.hpp"
#include <cassert>
#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
InteropImageLinear::Context::Context()
{
}

// ---------------------------------------------------------------------------
bool InteropImageLinear::Context::Initialize(
    GraphicsApiBase& gfxApi,
    AssetManager&    assetManager,
    CLState&         cl_state,
    TextureVulkan*   color)
{
    auto& vulkan  = static_cast<Vulkan&>(gfxApi);
    m_cl_state    = &cl_state;

    const uint32_t width  = color->Width;
    const uint32_t height = color->Height;

    // For clarity, this sample keeps explicit Vulkan copies around the
    // shared image resources rather than implementing a full render-target
    // zero-copy path.

    ///////////////////////////
    // INPUT COLOR REFERENCE //
    ///////////////////////////
    // Keep a view of the scene color attachment so CopyToSharedImage knows the source.
    m_input_color_ref = std::make_unique<TextureVulkan>(
        std::move(CreateTextureObjectView(vulkan, *color, color->Format)));

    ////////////////////
    // OUTPUT TEXTURE //
    ////////////////////
    // Separate output texture that VK copies the shared output image into.
    m_scene_color_output = std::make_unique<TextureVulkan>(std::move(CreateTextureObject(
        vulkan,
        width, height,
        TextureFormat::R8G8B8A8_UNORM,
        TEXTURE_TYPE::TT_RENDER_TARGET_SAMPLED_TRANSFERDST,
        "scene color output (linear)")));

    ///////////////////
    // SHARED IMAGES //
    ///////////////////
    // Input: VK copies scene color into this, CL reads from it
    InitSharedImage(
        gfxApi,
        m_shared_color_input,
        width, height,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        CL_RGBA, CL_UNORM_INT8,
        CL_MEM_READ_ONLY);

    // Output: CL writes to this, VK copies from it
    InitSharedImage(
        gfxApi,
        m_shared_color_output,
        width, height,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        CL_RGBA, CL_UNORM_INT8,
        CL_MEM_WRITE_ONLY);

    //////////////////////
    // OPENCL RESOURCES //
    //////////////////////
    InitCLKernel(assetManager);

    return true;
}

// ---------------------------------------------------------------------------
void InteropImageLinear::Context::Release(GraphicsApiBase& gfxApi)
{
    auto& vulkan    = static_cast<Vulkan&>(gfxApi);
    VkDevice device = vulkan.m_VulkanDevice;

    // Release input color reference and output texture
    if (m_input_color_ref)
    {
        m_input_color_ref->Release(&gfxApi);
        m_input_color_ref.reset();
    }
    if (m_scene_color_output)
    {
        m_scene_color_output->Release(&gfxApi);
        m_scene_color_output.reset();
    }

    // Release shared VK/CL image resources
    auto ReleaseImage = [&](SharedImage& img)
    {
        if (img.cl_image)   { clReleaseMemObject(img.cl_image);                  img.cl_image   = nullptr; }
        if (img.vk_view)    { vkDestroyImageView(device, img.vk_view, nullptr);  img.vk_view    = VK_NULL_HANDLE; }
        if (img.vk_sampler) { vkDestroySampler(device, img.vk_sampler, nullptr); img.vk_sampler = VK_NULL_HANDLE; }
        if (img.vk_image)   { vkDestroyImage(device, img.vk_image, nullptr);     img.vk_image   = VK_NULL_HANDLE; }
        if (img.vk_memory)  { vkFreeMemory(device, img.vk_memory, nullptr);      img.vk_memory  = VK_NULL_HANDLE; }
        img = {};
    };
    ReleaseImage(m_shared_color_input);
    ReleaseImage(m_shared_color_output);

    // Release OpenCL resources
    if (m_grayscale_kernel)       { clReleaseKernel(m_grayscale_kernel);           m_grayscale_kernel       = nullptr; }
    if (m_cl_program)             { clReleaseProgram(m_cl_program);                m_cl_program             = nullptr; }

    m_cl_state = nullptr;
}

// ---------------------------------------------------------------------------
void InteropImageLinear::Context::Dispatch()
{
    cl_command_queue queue = m_cl_state->queue;
    std::array<cl_mem, 2> ext_objects = {m_shared_color_input.cl_image, m_shared_color_output.cl_image};

    CL_CHECK(clEnqueueAcquireExternalMemObjectsKHR(queue, static_cast<cl_uint>(ext_objects.size()), ext_objects.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueNDRangeKernel(queue, m_grayscale_kernel, 2, nullptr, m_grayscale_global, nullptr, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReleaseExternalMemObjectsKHR(queue, static_cast<cl_uint>(ext_objects.size()), ext_objects.data(), 0, nullptr, nullptr));
}

// ---------------------------------------------------------------------------
void InteropImageLinear::Context::CopyToSharedImage(
    Vulkan&            vulkan,
    CommandListVulkan& command_list)
{
    VkCommandBuffer cmd = command_list.m_VkCommandBuffer;

    // Transition scene color: COLOR_ATTACHMENT_OPTIMAL → TRANSFER_SRC_OPTIMAL
    ImageLayoutTransition(
        cmd,
        m_input_color_ref->GetVkImage(),
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        {},
        VK_ACCESS_TRANSFER_READ_BIT,
        m_input_color_ref->GetVkImageLayout(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    // Copy scene color → shared input image
    VkImageCopy copy_region{};
    copy_region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy_region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy_region.extent         = {m_shared_color_input.width, m_shared_color_input.height, 1};
    vkCmdCopyImage(cmd,
        m_input_color_ref->GetVkImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        m_shared_color_input.vk_image,   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &copy_region);

    // Transition scene color back
    ImageLayoutTransition(
        cmd,
        m_input_color_ref->GetVkImage(),
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        {},
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        m_input_color_ref->GetVkImageLayout(),
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

}

// ---------------------------------------------------------------------------
void InteropImageLinear::Context::CopyFromSharedImage(
    Vulkan&            vulkan,
    CommandListVulkan& command_list)
{
    VkCommandBuffer cmd           = command_list.m_VkCommandBuffer;
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageLayout orig_out_layout = m_scene_color_output->GetVkImageLayout();
    VkImage       dst_image       = m_scene_color_output->GetVkImage();
    uint32_t      width           = m_scene_color_output->Width;
    uint32_t      height          = m_scene_color_output->Height;

    // Transition output texture → TRANSFER_DST_OPTIMAL
    ImageLayoutTransition(
        cmd, dst_image,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        {},
        VK_ACCESS_TRANSFER_WRITE_BIT,
        orig_out_layout,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        range);

    // Copy shared output image → output texture
    VkImageCopy copy_region{};
    copy_region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy_region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy_region.extent         = {width, height, 1};
    vkCmdCopyImage(cmd,
        m_shared_color_output.vk_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dst_image,                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &copy_region);

    // Transition output texture back to original layout
    ImageLayoutTransition(
        cmd, dst_image,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        {},
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        orig_out_layout,
        range);
}

// ---------------------------------------------------------------------------
void InteropImageLinear::Context::ReleaseSharedResourcesToExternal(
    Vulkan&            vulkan,
    CommandListVulkan& command_list)
{
    VkCommandBuffer cmd = command_list.m_VkCommandBuffer;
    const uint32_t  vk_queue_family = static_cast<uint32_t>(vulkan.m_VulkanQueues[Vulkan::eGraphicsQueue].QueueFamilyIndex);
    const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    // Release ownership of the shared resource to the external queue family through pipeline barriers
    ImageQueueFamilyOwnershipTransfer(
        cmd,
        m_shared_color_input.vk_image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL,
        range,
        vk_queue_family,
        VK_QUEUE_FAMILY_EXTERNAL,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    ImageQueueFamilyOwnershipTransfer(
        cmd,
        m_shared_color_output.vk_image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL,
        range,
        vk_queue_family,
        VK_QUEUE_FAMILY_EXTERNAL,
        VK_ACCESS_TRANSFER_READ_BIT,
        0,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

// ---------------------------------------------------------------------------
void InteropImageLinear::Context::AcquireSharedResourcesFromExternal(
    Vulkan&            vulkan,
    CommandListVulkan& command_list)
{
    VkCommandBuffer cmd = command_list.m_VkCommandBuffer;
    const uint32_t  vk_queue_family = static_cast<uint32_t>(vulkan.m_VulkanQueues[Vulkan::eGraphicsQueue].QueueFamilyIndex);
    const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    // Aquire ownership of the shared resource to the external queue family through pipeline barriers
    ImageQueueFamilyOwnershipTransfer(
        cmd,
        m_shared_color_input.vk_image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        range,
        VK_QUEUE_FAMILY_EXTERNAL,
        vk_queue_family,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);

    ImageQueueFamilyOwnershipTransfer(
        cmd,
        m_shared_color_output.vk_image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        range,
        VK_QUEUE_FAMILY_EXTERNAL,
        vk_queue_family,
        0,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
}

// ---------------------------------------------------------------------------
void InteropImageLinear::Context::InitSharedImage(
    GraphicsApiBase&  gfxApi,
    SharedImage&      shared_image,
    uint32_t          width,
    uint32_t          height,
    VkFormat          format,
    VkImageUsageFlags usage,
    VkImageLayout     layout,
    cl_channel_order  channel_order,
    cl_channel_type   channel_type,
    cl_mem_flags      mem_flags)
{
    ////////////
    // VULKAN //
    ////////////

    auto& vulkan        = static_cast<Vulkan&>(gfxApi);
    auto  device_handle = vulkan.m_VulkanDevice;

    shared_image.width  = width;
    shared_image.height = height;

    VkExternalMemoryHandleTypeFlagBits external_handle_type =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

    // Check external memory support for this image format, usage, and handle type.
    const VkExternalMemoryFeatureFlags external_memory_features =
        GetExternalImageMemoryFeatures(vulkan.m_VulkanGpu,
                                       format,
                                       VK_IMAGE_TILING_LINEAR,
                                       usage,
                                       external_handle_type);

    if (!(external_memory_features & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT))
        throw std::runtime_error("Vulkan image memory is not exportable with opaque fd external memory handle type");

    const bool dedicated_only =
        (external_memory_features & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0;

    // Create exportable Vulkan image memory.
    VkExternalMemoryImageCreateInfo ext_img_info{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    ext_img_info.handleTypes = external_handle_type;

    VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.pNext         = &ext_img_info;
    image_info.imageType     = VK_IMAGE_TYPE_2D;
    image_info.format        = format;
    image_info.extent        = {width, height, 1};
    image_info.mipLevels     = 1;
    image_info.arrayLayers   = 1;
    image_info.samples       = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling        = VK_IMAGE_TILING_LINEAR;
    image_info.usage         = usage;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(device_handle, &image_info, nullptr, &shared_image.vk_image);

    // Allocate exportable memory.
    // Use dedicated allocation when required by the external memory properties.
    VkMemoryRequirements mem_req{};
    vkGetImageMemoryRequirements(device_handle, shared_image.vk_image, &mem_req);

    VkMemoryDedicatedAllocateInfo dedicated_info{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated_info.image  = shared_image.vk_image;
    dedicated_info.buffer = VK_NULL_HANDLE;

    VkExportMemoryAllocateInfoKHR export_info{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR};
    export_info.pNext       = dedicated_only ? &dedicated_info : nullptr;
    export_info.handleTypes = external_handle_type;

    VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc_info.pNext           = &export_info;
    alloc_info.allocationSize  = mem_req.size;
    alloc_info.memoryTypeIndex = GetMemoryType(vulkan.m_VulkanGpu, mem_req.memoryTypeBits, 0);

    vkAllocateMemory(device_handle, &alloc_info, nullptr, &shared_image.vk_memory);
    vkBindImageMemory(device_handle, shared_image.vk_image, shared_image.vk_memory, 0);

    // OpenCL needs Vulkan's row pitch for imported linear images
    VkImageSubresource subresource{};
    subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource.mipLevel   = 0;
    subresource.arrayLayer = 0;

    VkSubresourceLayout subresource_layout{};
    vkGetImageSubresourceLayout(device_handle, shared_image.vk_image, &subresource, &subresource_layout);

    // Create sampler
    const bool supports_linear_filtering =
        SupportsLinearFiltering(vulkan.m_VulkanGpu, format, VK_IMAGE_TILING_LINEAR);
    const VkFilter filter =
        supports_linear_filtering ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    const VkSamplerMipmapMode mipmap_mode =
        supports_linear_filtering ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;

    VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_info.magFilter   = filter;
    sampler_info.minFilter   = filter;
    sampler_info.mipmapMode  = mipmap_mode;
    sampler_info.maxLod      = 1.0f;
    sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    vkCreateSampler(device_handle, &sampler_info, nullptr, &shared_image.vk_sampler);

    // Create image view
    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    view_info.image            = shared_image.vk_image;
    view_info.format           = format;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(device_handle, &view_info, nullptr, &shared_image.vk_view);

    // Transition image to the requested layout
    VkCommandBuffer setup_cmd = vulkan.StartSetupCommandBuffer();
    vulkan.SetImageLayout(
        shared_image.vk_image,
        setup_cmd,
        VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        layout,
        (VkPipelineStageFlags)0,
        (VkPipelineStageFlags)0,
        0, 1, 0, 1);
    vulkan.FinishSetupCommandBuffer(setup_cmd);

    ////////////
    // OPENCL //
    ////////////

    // Import Vulkan memory into OpenCL.
    int fd = -1;
    VkMemoryGetFdInfoKHR fd_info{};
    fd_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    fd_info.memory     = shared_image.vk_memory;
    fpGetMemoryFdKHR(device_handle, &fd_info, &fd);

    std::vector<cl_mem_properties> props{
        (cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR,
        (cl_mem_properties)fd,
        (cl_mem_properties)CL_MEM_DEVICE_HANDLE_LIST_KHR,
        (cl_mem_properties)m_cl_state->device,
        (cl_mem_properties)CL_MEM_DEVICE_HANDLE_LIST_END_KHR,
        0
    };

    cl_image_format cl_fmt{};
    cl_fmt.image_channel_order     = channel_order;
    cl_fmt.image_channel_data_type = channel_type;

    cl_image_desc cl_desc{};
    cl_desc.image_type        = CL_MEM_OBJECT_IMAGE2D;
    cl_desc.image_width       = width;
    cl_desc.image_height      = height;
    cl_desc.image_depth       = 0; // only used for 3D image
    cl_desc.image_array_size  = 0; // only used for 1D/2D image array

    // If image_row_pitch is zero and the image is created from an external memory handle,
    // then the image row pitch is implemented-defined. Vulkan may pad each row, using 0 row
    // here would let OpenCL choose an implementation-defined pitch that may not match Vulkan's row pitch
    cl_desc.image_row_pitch   = static_cast<size_t>(subresource_layout.rowPitch);
    
    cl_desc.image_slice_pitch = 0; // subresource_layout.depthPitch for 3D image or 1D/2D image array
    cl_desc.num_mip_levels    = 0; // must be 0 unless cl_khr_mipmap_image is supported
    cl_desc.num_samples       = 0; // must be 0
    cl_desc.buffer            = nullptr;

    cl_int err;
    shared_image.cl_image = clCreateImageWithProperties(
        m_cl_state->context, props.data(), mem_flags,
        &cl_fmt, &cl_desc, nullptr, &err);
    CL_CHECK(err);
}

// ---------------------------------------------------------------------------
void InteropImageLinear::Context::InitCLKernel(AssetManager& assetManager)
{
    LOGI("*************************************");
    LOGI("Loading grayscale_image CL kernel...");
    LOGI("*************************************");

    const std::string kernelPath = std::string(KERNEL_DESTINATION_PATH) + "/grayscale_image.cl";
    std::string source;
    if (!assetManager.LoadFileIntoMemory(kernelPath, source))
    {
        LOGE("Failed to open kernel file: %s", kernelPath.c_str());
        throw std::runtime_error("Failed to open kernel file: " + kernelPath);
    }
    const char* src     = source.c_str();
    size_t      src_len = source.size();

    cl_int err;

    m_cl_program = clCreateProgramWithSource(m_cl_state->context, 1, &src, &src_len, &err);
    CL_CHECK(err);

    err = clBuildProgram(m_cl_program, 1, &m_cl_state->device, "", nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t log_size = 0;
        clGetProgramBuildInfo(m_cl_program, m_cl_state->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(m_cl_program, m_cl_state->device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
        LOGE("clBuildProgram failed: %s\nBuild log:\n%s", ClErrorToString(err).c_str(), log.c_str());
        throw std::runtime_error(std::string("clBuildProgram failed: ") + ClErrorToString(err));
    }

    m_grayscale_kernel = clCreateKernel(m_cl_program, "grayscale_image", &err);
    CL_CHECK(err);

    uint32_t width  = m_input_color_ref->Width;
    uint32_t height = m_input_color_ref->Height;

    cl_uint arg = 0;
    CL_CHECK(clSetKernelArg(m_grayscale_kernel, arg++, sizeof(cl_mem), &m_shared_color_input.cl_image));
    CL_CHECK(clSetKernelArg(m_grayscale_kernel, arg++, sizeof(cl_mem), &m_shared_color_output.cl_image));

    m_grayscale_global[0] = width;
    m_grayscale_global[1] = height;
}