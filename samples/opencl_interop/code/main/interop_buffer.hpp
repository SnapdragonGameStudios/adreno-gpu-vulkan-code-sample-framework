//============================================================================================================
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

//============================================================================================================
//
// Mode 1: External Memory Opaque FD Buffer interop + grayscale CL kernel.
//   VK copies scene color into a shared VkBuffer, CL processes it,
//   result is copied back from the shared VkBuffer into a VkImage.
//
//============================================================================================================

#pragma once

#include "interop_common.hpp"

namespace InteropBuffer
{

/// Shared Vulkan buffer resources and imported OpenCL buffer for VK<->CL buffer interop.
struct SharedBuffer
{
    size_t         size{0}; // Logical byte size shared by VkBuffer and cl_mem buffer.

    VkBuffer       vk_buffer{VK_NULL_HANDLE};
    VkDeviceMemory vk_memory{VK_NULL_HANDLE};

    cl_mem         cl_buffer{nullptr};
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

    /// Copy scene color attachment → shared input buffer (call before CL dispatch).
    void CopyToSharedBuffers(
        Vulkan&            vulkan,
        CommandListVulkan& command_list);

    /// Copy shared output buffer → output texture (call after CL dispatch).
    void CopyFromSharedBuffers(
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

    // Reference to the scene color attachment (source for CopyToSharedBuffers)
    std::unique_ptr<TextureVulkan>      m_input_color_ref;

    // Shared VK buffer <-> CL buffer (input color)
    SharedBuffer                        m_shared_color_input;

    // Shared VK buffer <-> CL buffer (output grayscale)
    SharedBuffer                        m_shared_color_output;

    // Output texture (same size as input, R8G8B8A8_UNORM)
    std::unique_ptr<TextureVulkan>      m_scene_color_output;

    // Per-context CL program and kernel
    cl_program       m_cl_program{nullptr};
    cl_kernel        m_grayscale_kernel{nullptr};
    size_t           m_grayscale_global[2]{0, 0};

    // Internal helpers
    void InitCLKernel(AssetManager& assetManager);

    void InitSharedBuffer(
        GraphicsApiBase&   gfxApi,
        SharedBuffer&      shared_buffer,
        size_t             size,
        VkBufferUsageFlags usage,
        cl_mem_flags       mem_flags);
};

} // namespace InteropBuffer