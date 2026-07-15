//=============================================================================
//
//
//                  Copyright (c) 2020 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include "system/glm_common.hpp"
#include "vulkan/vulkan_support.hpp"
#include "vulkan/renderContext.hpp"
#include "vulkan/renderTarget.hpp"
#include "shadow.hpp"

/// Vulkan specialization implementation of Shadow.
template<>
class ShadowT<Vulkan> : public Shadow
{
public:
    ShadowT();

    bool Initialize(Vulkan& vulkan, uint32_t shadowMapWidth, uint32_t shadowMapHeight, bool addColorTarget);
    void Release(Vulkan& vulkan);

    const auto& GetViewport() const
    {
        return m_Viewport;
    }
    const auto& GetScissor() const
    {
        return m_Scissor;
    }

    const auto& GetRenderContext() const
    {
        return m_ShadowMapRC;
    }
    VkFramebuffer GetFramebuffer(uint32_t idx = 0) const
    {
        return m_ShadowMapRC.GetFramebuffer()->m_FrameBuffer;
    }
    void GetTargetSize(uint32_t& width, uint32_t& height) const
    {
        width = m_ShadowDepthBuffer.Width;
        height = m_ShadowDepthBuffer.Height;
    }

    auto GetDepthFormat(uint32_t frameBufferIndex = 0) const
    {
        return m_ShadowDepthBuffer.Format;
    }
    const auto& GetDepthTexture() const
    {
        return m_ShadowDepthBuffer;
    }
    const Texture<Vulkan>* GetColorTexture() const
    {
        return &m_ShadowColorBuffer;
    }

protected:
    VkViewport              m_Viewport = {};
    VkRect2D                m_Scissor = {};

    // Render targets (images)
    Texture<Vulkan>         m_ShadowDepthBuffer;
    Texture<Vulkan>         m_ShadowColorBuffer;

    // Render context (renderpass and framebuffer)
    RenderContext<Vulkan>   m_ShadowMapRC;
};

using ShadowVulkan = ShadowT<Vulkan>;
