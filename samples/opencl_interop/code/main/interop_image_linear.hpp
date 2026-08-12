//============================================================================================================
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

//============================================================================================================
//
// Mode 2: External Memory Linear Image interop
//   VK copies scene color into a shared VkImage (linear tiling),
//   CL processes it and writes to a shared output VkImage (linear tiling),
//   VK copies the output image back to a TextureVulkan for the blit pass.
//
//============================================================================================================

#pragma once

#include "interop_common.hpp"

namespace InteropImageLinear
{

/// Shared Vulkan image resources and imported OpenCL image for VK<->CL image interop.
struct SharedImage
{
    uint32_t       width{0};
    uint32_t       height{0};

    VkImage        vk_image{VK_NULL_HANDLE};
    VkDeviceMemory vk_memory{VK_NULL_HANDLE};
    VkSampler      vk_sampler{VK_NULL_HANDLE};
    VkImageView    vk_view{VK_NULL_HANDLE};

    cl_mem         cl_image{nullptr};
};

class Context
{
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
public:
    Context();

    bool Initialize(
        GraphicsApiBase&,
        AssetManager&,
        CLState&       cl_state,
        TextureVulkan* color);

    void Release(GraphicsApiBase&);

    /// Returns the processed output texture (for the blit pass).
    inline const TextureVulkan* GetSceneColorOutput() const
    {
        return m_scene_color_output.get();
    }

    void Dispatch();

    /// Copy scene color attachment → shared input image (call before CL dispatch).
    void CopyToSharedImage(
        Vulkan&            vulkan,
        CommandListVulkan& command_list);

    /// Copy shared output image → output texture (call after CL dispatch).
    void CopyFromSharedImage(
        Vulkan&            vulkan,
        CommandListVulkan& command_list);

    void ReleaseSharedResourcesToExternal(
        Vulkan&            vulkan,
        CommandListVulkan& command_list);

    void AcquireSharedResourcesFromExternal(
        Vulkan&            vulkan,
        CommandListVulkan& command_list);

protected:
    // Shared CL state (not owned — owned by Application)
    CLState*         m_cl_state{nullptr};

    // Reference to the scene color attachment (source for CopyToSharedImage)
    std::unique_ptr<TextureVulkan>      m_input_color_ref;

    // Shared input image: VK copies into it, CL reads from it
    SharedImage                         m_shared_color_input;

    // Shared output image: CL writes to it, VK copies from it
    SharedImage                         m_shared_color_output;

    // Output texture: VK copies shared output into this, then blits to screen
    std::unique_ptr<TextureVulkan>      m_scene_color_output;

    // Per-context CL program and kernel
    cl_program       m_cl_program{nullptr};
    cl_kernel        m_grayscale_kernel{nullptr};
    size_t           m_grayscale_global[2]{0, 0};

    // Internal helpers
    void InitCLKernel(AssetManager& assetManager);

    void InitSharedImage(
        GraphicsApiBase&   gfxApi,
        SharedImage&       shared_image,
        uint32_t           width,
        uint32_t           height,
        VkFormat           format,
        VkImageUsageFlags  usage,
        VkImageLayout      layout,
        cl_channel_order   channel_order,
        cl_channel_type    channel_type,
        cl_mem_flags       mem_flags);
};

} // namespace InteropImageLinear