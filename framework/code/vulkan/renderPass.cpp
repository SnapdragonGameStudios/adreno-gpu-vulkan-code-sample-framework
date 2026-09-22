// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "renderPass.hpp"
#include "texture/textureFormat.hpp"
#include "vulkan/vulkan.hpp"

// Forward declarations
class Vulkan;


RenderPass<Vulkan>::RenderPass() noexcept
{
}

RenderPass<Vulkan>::~RenderPass()
{
}

RenderPass<Vulkan>::RenderPass(VkDevice device, VkRenderPass renderPass) noexcept
    : mRenderPass(device, renderPass)
{
}

RenderPass<Vulkan> CreateRenderPass( Vulkan& vulkan, std::span<const TextureFormat> ColorFormats, TextureFormat DepthFormat, Msaa Msaa, RenderPassInputUsage ColorInputUsage, RenderPassOutputUsage ColorOutputUsage, bool ShouldClearDepth, RenderPassOutputUsage DepthOutputUsage, std::span < const TextureFormat > ResolveFormats )
{
    RenderPass<Vulkan> renderPass{};
    if (!vulkan.CreateRenderPass(ColorFormats, DepthFormat, Msaa, ColorInputUsage, ColorOutputUsage, ShouldClearDepth, DepthOutputUsage, renderPass, ResolveFormats))
    {
        return RenderPass<Vulkan>{};
    }
    return renderPass;
}
