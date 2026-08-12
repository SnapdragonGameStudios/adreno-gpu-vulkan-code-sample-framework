//============================================================================================================
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

//============================================================================================================
//
// Mode 1 VK-CL interop implementation (InteropBuffer):
//   - Shares VkBuffer with OpenCL via opaque fd
//   - VK copies scene color into the shared input VkBuffer
//   - OpenCL grayscale_buffer kernel reads from the shared input buffer,
//     writes to the shared output buffer
//   - VK copies the shared output buffer back to a TextureVulkan for the blit pass
//
//============================================================================================================

#include "interop_buffer.hpp"
#include "vulkan/commandBuffer.hpp"
#include <cassert>
#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
InteropBuffer::Context::Context()
{
}

// ---------------------------------------------------------------------------
bool InteropBuffer::Context::Initialize(
    GraphicsApiBase& gfxApi,
    AssetManager&    assetManager,
    CLState&         cl_state,
    TextureVulkan*   color)
{
    auto& vulkan  = static_cast<Vulkan&>(gfxApi);
    m_cl_state    = &cl_state;

    ///////////////////////////
    // INPUT COLOR REFERENCE //
    ///////////////////////////
    m_input_color_ref = std::make_unique<TextureVulkan>(std::move(CreateTextureObjectView(vulkan, *color, color->Format)));

    ////////////////////
    // OUTPUT TEXTURE //
    ////////////////////
    m_scene_color_output = std::make_unique<TextureVulkan>(std::move(CreateTextureObject(
        vulkan,
        color->Width,
        color->Height,
        TextureFormat::R8G8B8A8_UNORM,
        TEXTURE_TYPE::TT_RENDER_TARGET_SAMPLED_TRANSFERDST,
        "scene color output")));

    ////////////////////
    // SHARED BUFFERS //
    ////////////////////

    // Input: VK copies scene color into this, CL reads from it
    InitSharedBuffer(
        gfxApi,
        m_shared_color_input,
        m_input_color_ref->Width * m_input_color_ref->Height * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        CL_MEM_READ_ONLY);

    // Output: CL writes to this, VK copies from it
    InitSharedBuffer(
        gfxApi,
        m_shared_color_output,
        m_scene_color_output->Width * m_scene_color_output->Height * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        CL_MEM_READ_WRITE);

    //////////////////////
    // OPENCL RESOURCES //
    //////////////////////
    InitCLKernel(assetManager);

    return true;
}

// ---------------------------------------------------------------------------
void InteropBuffer::Context::Release(GraphicsApiBase& gfxApi)
{
    // Release shared VK/CL buffer resources
    auto& vulkan    = static_cast<Vulkan&>(gfxApi);
    VkDevice device = vulkan.m_VulkanDevice;

    auto ReleaseBuffer = [&](SharedBuffer& buf)
    {
        if (buf.cl_buffer) { clReleaseMemObject(buf.cl_buffer); buf.cl_buffer = nullptr; }
        if (buf.vk_buffer) { vkDestroyBuffer(device, buf.vk_buffer, nullptr); buf.vk_buffer = VK_NULL_HANDLE; }
        if (buf.vk_memory) { vkFreeMemory(device, buf.vk_memory, nullptr); buf.vk_memory = VK_NULL_HANDLE; }
        buf = {};
    };
    ReleaseBuffer(m_shared_color_input);
    ReleaseBuffer(m_shared_color_output);

    if (m_scene_color_output)
    {
        m_scene_color_output->Release(&gfxApi);
        m_scene_color_output.reset();
    }

    // Release OpenCL resources
    if (m_grayscale_kernel)       { clReleaseKernel(m_grayscale_kernel);          m_grayscale_kernel       = nullptr; }
    if (m_cl_program)             { clReleaseProgram(m_cl_program);               m_cl_program             = nullptr; }
    m_cl_state = nullptr;
}

// ---------------------------------------------------------------------------
void InteropBuffer::Context::Dispatch()
{
    cl_command_queue queue = m_cl_state->queue;
    std::array<cl_mem, 2> ext_objects = {m_shared_color_input.cl_buffer, m_shared_color_output.cl_buffer};

    CL_CHECK(clEnqueueAcquireExternalMemObjectsKHR(queue, static_cast<cl_uint>(ext_objects.size()), ext_objects.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueNDRangeKernel(queue, m_grayscale_kernel, 2, nullptr, m_grayscale_global, nullptr, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReleaseExternalMemObjectsKHR(queue, static_cast<cl_uint>(ext_objects.size()), ext_objects.data(), 0, nullptr, nullptr));
}

// ---------------------------------------------------------------------------
void InteropBuffer::Context::CopyToSharedBuffers(
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

    // Copy scene color → shared input buffer
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {m_input_color_ref->Width, m_input_color_ref->Height, 1};
    vkCmdCopyImageToBuffer(cmd,
        m_input_color_ref->GetVkImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        m_shared_color_input.vk_buffer, 1, &region);

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
void InteropBuffer::Context::CopyFromSharedBuffers(
    Vulkan&            vulkan,
    CommandListVulkan& command_list)
{
    VkCommandBuffer cmd           = command_list.m_VkCommandBuffer;
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageLayout orig_layout     = m_scene_color_output->GetVkImageLayout();
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
        orig_layout,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        range);

    // Copy shared output buffer → output texture
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {width, height, 1};
    vkCmdCopyBufferToImage(cmd,
        m_shared_color_output.vk_buffer, dst_image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition output texture back to original layout
    ImageLayoutTransition(
        cmd, dst_image,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        {},
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        orig_layout,
        range);
}

// ---------------------------------------------------------------------------
void InteropBuffer::Context::ReleaseSharedResourcesToExternal(
    Vulkan&            vulkan,
    CommandListVulkan& command_list)
{
    VkCommandBuffer cmd = command_list.m_VkCommandBuffer;
    const uint32_t  vk_queue_family = static_cast<uint32_t>(vulkan.m_VulkanQueues[Vulkan::eGraphicsQueue].QueueFamilyIndex);

    // Release ownership of the shared resource to the external queue family through pipeline barriers
    BufferQueueFamilyOwnershipTransfer(
        cmd,
        m_shared_color_input.vk_buffer,
        0,
        static_cast<VkDeviceSize>(m_shared_color_input.size),
        vk_queue_family,
        VK_QUEUE_FAMILY_EXTERNAL,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    BufferQueueFamilyOwnershipTransfer(
        cmd,
        m_shared_color_output.vk_buffer,
        0,
        static_cast<VkDeviceSize>(m_shared_color_output.size),
        vk_queue_family,
        VK_QUEUE_FAMILY_EXTERNAL,
        0,
        0,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
}

// ---------------------------------------------------------------------------
void InteropBuffer::Context::AcquireSharedResourcesFromExternal(
    Vulkan&            vulkan,
    CommandListVulkan& command_list)
{
    VkCommandBuffer cmd = command_list.m_VkCommandBuffer;
    const uint32_t  vk_queue_family = static_cast<uint32_t>(vulkan.m_VulkanQueues[Vulkan::eGraphicsQueue].QueueFamilyIndex);

    // Aquire ownership of the shared resource to the external queue family through pipeline barriers
    BufferQueueFamilyOwnershipTransfer(
        cmd,
        m_shared_color_input.vk_buffer,
        0,
        static_cast<VkDeviceSize>(m_shared_color_input.size),
        VK_QUEUE_FAMILY_EXTERNAL,
        vk_queue_family,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);

    BufferQueueFamilyOwnershipTransfer(
        cmd,
        m_shared_color_output.vk_buffer,
        0,
        static_cast<VkDeviceSize>(m_shared_color_output.size),
        VK_QUEUE_FAMILY_EXTERNAL,
        vk_queue_family,
        0,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
}

// ---------------------------------------------------------------------------
void InteropBuffer::Context::InitSharedBuffer(
    GraphicsApiBase&   gfxApi,
    SharedBuffer&      shared_buffer,
    size_t             size,
    VkBufferUsageFlags usage,
    cl_mem_flags       mem_flags)
{
    auto& vulkan = static_cast<Vulkan&>(gfxApi);
    auto  device = vulkan.m_VulkanDevice;

    shared_buffer.size = size;

    VkExternalMemoryHandleTypeFlagBits ext_handle_type =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

    // Check external memory support for this buffer usage and handle type.
    const VkExternalMemoryFeatureFlags external_memory_features =
        GetExternalBufferMemoryFeatures(vulkan.m_VulkanGpu, usage, ext_handle_type);

    if (!(external_memory_features & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT))
        throw std::runtime_error("Vulkan buffer memory is not exportable with opaque fd external memory handle type");

    const bool dedicated_only =
        (external_memory_features & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0;

    // Create exportable Vulkan buffer memory.
    VkExternalMemoryBufferCreateInfo ext_buf_info{};
    ext_buf_info.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext_buf_info.handleTypes = ext_handle_type;

    VkBufferCreateInfo buf_info{};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.pNext = &ext_buf_info;
    buf_info.size  = static_cast<VkDeviceSize>(size);
    buf_info.usage = usage;
    vkCreateBuffer(device, &buf_info, nullptr, &shared_buffer.vk_buffer);

    // Allocate exportable memory
    // Use dedicated allocation when required by the external memory properties.
    VkMemoryRequirements mem_req{};
    vkGetBufferMemoryRequirements(device, shared_buffer.vk_buffer, &mem_req);

    VkMemoryDedicatedAllocateInfo dedicated_info{};
    dedicated_info.sType  = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated_info.buffer = shared_buffer.vk_buffer;
    dedicated_info.image  = VK_NULL_HANDLE;

    VkExportMemoryAllocateInfoKHR export_info{};
    export_info.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    export_info.pNext       = dedicated_only ? &dedicated_info : nullptr;
    export_info.handleTypes = ext_handle_type;

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext           = &export_info;
    alloc_info.allocationSize  = mem_req.size;
    alloc_info.memoryTypeIndex = GetMemoryType(vulkan.m_VulkanGpu, mem_req.memoryTypeBits, 0);

    vkAllocateMemory(device, &alloc_info, nullptr, &shared_buffer.vk_memory);
    vkBindBufferMemory(device, shared_buffer.vk_buffer, shared_buffer.vk_memory, 0);

    // Import Vulkan memory into OpenCL.
    int fd = -1;
    VkMemoryGetFdInfoKHR fd_info{};
    fd_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    fd_info.memory     = shared_buffer.vk_memory;
    fpGetMemoryFdKHR(device, &fd_info, &fd);

    std::vector<cl_mem_properties> props{
        (cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR,
        (cl_mem_properties)fd,
        (cl_mem_properties)CL_MEM_DEVICE_HANDLE_LIST_KHR,
        (cl_mem_properties)m_cl_state->device,
        (cl_mem_properties)CL_MEM_DEVICE_HANDLE_LIST_END_KHR,
        0
    };

    cl_int err;
    shared_buffer.cl_buffer = clCreateBufferWithProperties(m_cl_state->context, props.data(), mem_flags, size, nullptr, &err);
    CL_CHECK(err);
}

// ---------------------------------------------------------------------------
void InteropBuffer::Context::InitCLKernel(AssetManager& assetManager)
{
    LOGI("*****************************");
    LOGI("Loading grayscale CL kernel...");
    LOGI("*****************************");

    const std::string kernelPath = std::string(KERNEL_DESTINATION_PATH) + "/grayscale_buffer.cl";
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

    m_grayscale_kernel = clCreateKernel(m_cl_program, "grayscale_buffer", &err);
    CL_CHECK(err);

    uint32_t width  = m_input_color_ref->Width;
    uint32_t height = m_input_color_ref->Height;

    cl_uint arg = 0;
    CL_CHECK(clSetKernelArg(m_grayscale_kernel, arg++, sizeof(cl_mem),  &m_shared_color_input.cl_buffer));
    CL_CHECK(clSetKernelArg(m_grayscale_kernel, arg++, sizeof(cl_mem),  &m_shared_color_output.cl_buffer));
    CL_CHECK(clSetKernelArg(m_grayscale_kernel, arg++, sizeof(cl_uint), &width));
    CL_CHECK(clSetKernelArg(m_grayscale_kernel, arg++, sizeof(cl_uint), &height));

    m_grayscale_global[0] = width;
    m_grayscale_global[1] = height;
}