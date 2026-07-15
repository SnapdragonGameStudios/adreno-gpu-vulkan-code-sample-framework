//=============================================================================
//
//
//                  Copyright (c) 2020 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "shadowVulkan.hpp"
#include "vulkan/vulkan_support.hpp"
#include "vulkan/vulkan.hpp"

ShadowT<Vulkan>::ShadowT() : Shadow()
{
}

bool ShadowT<Vulkan>::Initialize(Vulkan& rGfxApi, uint32_t shadowMapWidth, uint32_t shadowMapHeight, bool addColorTarget)
{
    if (!Shadow::Initialize(shadowMapWidth, shadowMapHeight, addColorTarget))
        return false;

    m_Viewport.width = (float) shadowMapWidth;
    m_Viewport.height = (float) shadowMapHeight;
    m_Viewport.minDepth = 0.0f;
    m_Viewport.maxDepth = 1.0f;
    m_Viewport.x = 0.0f;
    m_Viewport.y = 0.0f;
    m_Scissor.offset.x = 0;
    m_Scissor.offset.y = 0;
    m_Scissor.extent.width = shadowMapWidth;
    m_Scissor.extent.height = shadowMapHeight;

    TextureFormat colorFormat = TextureFormat::R8G8B8A8_UNORM;
    TextureFormat depthFormat = TextureFormat::D32_SFLOAT;

    m_ShadowDepthBuffer = CreateTextureObject( rGfxApi, shadowMapWidth, shadowMapHeight, depthFormat, TEXTURE_TYPE::TT_DEPTH_TARGET, "Shadow Depth", Msaa::Samples1 );

    std::span<TextureVulkan> colorBufferSpan{};
    std::span<TextureFormat> colorFormatSpan{};
    if (addColorTarget)
    {
        m_ShadowColorBuffer = CreateTextureObject( rGfxApi, shadowMapWidth, shadowMapHeight, colorFormat, TEXTURE_TYPE::TT_RENDER_TARGET, "Shadow Depth", Msaa::Samples1 );
        colorBufferSpan ={ &m_ShadowColorBuffer, 1 };
        colorFormatSpan = {&colorFormat, 1};
    }

    RenderPass<Vulkan> shadowRenderPass;
    if (!rGfxApi.CreateRenderPass(
            colorFormatSpan,
            depthFormat,
            Msaa::Samples1,
            RenderPassInputUsage::Clear,            // color input
            RenderPassOutputUsage::StoreReadOnly,   // color output
            true,                                   // depth clear
            RenderPassOutputUsage::StoreReadOnly,   // depth output
            shadowRenderPass ))
    {
        return false;
    }

    Framebuffer<Vulkan> framebuffer;
    framebuffer.Initialize( rGfxApi, shadowRenderPass, colorBufferSpan, &m_ShadowDepthBuffer, "Shadow framebuffer" );
    m_ShadowMapRC = {std::move(shadowRenderPass), {}/*pipeline*/, framebuffer, "Shadow Context"};
    return true;
}

void ShadowT<Vulkan>::Release(Vulkan& vulkan)
{
    m_ShadowColorBuffer.Release(&vulkan);
    m_ShadowDepthBuffer.Release(&vulkan);
}
