//============================================================================================================
//
//
//                  Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include <string>
#include <variant>
#include "graphicsApi/renderContext.hpp"
#include "material/vulkan/pipeline.hpp" //TODO: move or refactor
#include "vulkan.hpp"
#include "framebuffer.hpp"
#include "renderPass.hpp"
#include "vulkan/renderTarget.hpp"
#include "texture/texture.hpp"
#include "texture/textureFormat.hpp"    // for msaa
#include "texture/vulkan/texture.hpp"
#include <string>

// Forward declarations
class Vulkan;

struct RenderingAttachmentInfo
{
    VkRenderingAttachmentInfo   renderingAttachmentInfo{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    std::optional<VkClearValue> clearValue;
    VkExtent2D                  renderArea = {0, 0};

    // this
    struct ColorInfo
    {
        RenderPassInputUsage  colorInputUsage  = RenderPassInputUsage::Load;
        RenderPassOutputUsage colorOutputUsage = RenderPassOutputUsage::Store;
    };

    // or this
    struct DepthInfo
    {
        bool                  clearDepth       = false;
        RenderPassOutputUsage depthOutputUsage = RenderPassOutputUsage::Store;
    };
    
    std::variant<std::monostate, ColorInfo, DepthInfo> v;

    RenderingAttachmentInfo() = default;
    inline RenderingAttachmentInfo(
        const Texture<Vulkan>&      _texture,
        std::optional<VkClearValue> _clear   = std::nullopt)
        : clearValue(_clear)
        , renderArea({ _texture.Width , _texture.Height })
    {
        renderingAttachmentInfo.imageView   = _texture.GetVkImageView();
        renderingAttachmentInfo.imageLayout = _texture.GetVkImageLayout();
        renderingAttachmentInfo.clearValue  = clearValue.value_or(_texture.GetVkClearValue());
    }

    static RenderingAttachmentInfo Color(
        const Texture<Vulkan>&      _texture,
        RenderPassInputUsage        _inputUsage,
        RenderPassOutputUsage       _outputUsage,
        std::optional<VkClearValue> _clear = std::nullopt)
    {
        RenderingAttachmentInfo renderingAttachmentInfo(_texture, _clear);
        renderingAttachmentInfo.v = ColorInfo{ _inputUsage , _outputUsage };
        return renderingAttachmentInfo;
    }

    static RenderingAttachmentInfo Depth(
        const Texture<Vulkan>&      _texture,
        bool                        _clearDepth,
        RenderPassOutputUsage       _outputUsage, 
        std::optional<VkClearValue> _clear = std::nullopt)
    {
        RenderingAttachmentInfo renderingAttachmentInfo(_texture, _clear);
        renderingAttachmentInfo.v = DepthInfo{ _clearDepth , _outputUsage };
        return renderingAttachmentInfo;
    }

    /*
    * Converts this struct into a Vulkan VkRenderingAttachmentInfo.
    */
    VkRenderingAttachmentInfo ToVkAttachmentInfo() const;
};

struct RenderingAttachmentInfoGroup
{
    std::array<RenderingAttachmentInfo, 16> colorAttachments;
    uint32_t                                colorCount = 0;
    std::optional<RenderingAttachmentInfo>  depthAttachment;

    /*
    * Constructs a RenderingAttachmentInfoGroup from a list of color attachments, an optional depth attachment, and an optional clear value override.
    */
    inline RenderingAttachmentInfoGroup(
        std::initializer_list<RenderingAttachmentInfo> colors,
        std::optional<RenderingAttachmentInfo>         depth            = std::nullopt,
        std::optional<RenderPassInputUsage>            colorInputUsage  = std::nullopt,
        std::optional<RenderPassOutputUsage>           colorOutputUsage = std::nullopt,
        std::optional<bool>                            clearDepth       = std::nullopt,
        std::optional<RenderPassOutputUsage>           depthOutputUsage = std::nullopt,
        std::optional<VkClearValue>                    clearOverride    = std::nullopt)
        : colorCount(static_cast<uint32_t>(colors.size()))
        , depthAttachment(depth)
    {
        assert(colorCount <= colorAttachments.size());

        std::copy(colors.begin(), colors.end(), colorAttachments.begin());

        if (clearOverride.has_value()) ApplyClearValueToAll(clearOverride.value());

                if (colorInputUsage || colorOutputUsage)
        {
            for (int i = 0; i < colorCount; i++)
            {
                colorAttachments[i].v = RenderingAttachmentInfo::ColorInfo();

                if(colorInputUsage)  std::get<RenderingAttachmentInfo::ColorInfo>(colorAttachments[i].v).colorInputUsage  = colorInputUsage.value();
                if(colorOutputUsage) std::get<RenderingAttachmentInfo::ColorInfo>(colorAttachments[i].v).colorOutputUsage = colorOutputUsage.value(); 
            }
        }
        
        if (depth && (clearDepth || depthOutputUsage))
        {
            depthAttachment    = RenderingAttachmentInfo(*depth);
            depthAttachment->v = RenderingAttachmentInfo::DepthInfo();
            if (clearDepth       && depthAttachment) std::get<RenderingAttachmentInfo::DepthInfo>(depthAttachment->v).clearDepth       = clearDepth.value();
            if (depthOutputUsage && depthAttachment) std::get<RenderingAttachmentInfo::DepthInfo>(depthAttachment->v).depthOutputUsage = depthOutputUsage.value();
        }
    }

    /*
    * Constructs a RenderingAttachmentInfoGroup directly from Texture<Vulkan> objects.
    */
    inline RenderingAttachmentInfoGroup(
        std::initializer_list<Texture<Vulkan>> textures,
        Texture<Vulkan>*                       depth            = nullptr,
        std::optional<RenderPassInputUsage>    colorInputUsage  = std::nullopt,
        std::optional<RenderPassOutputUsage>   colorOutputUsage = std::nullopt,
        std::optional<bool>                    clearDepth       = std::nullopt,
        std::optional<RenderPassOutputUsage>   depthOutputUsage = std::nullopt,
        std::optional<VkClearValue>            clearOverride    = std::nullopt)
    {
        assert(textures.size() <= colorAttachments.size());

        for (const auto& tex : textures) colorAttachments[colorCount++] = RenderingAttachmentInfo(tex);

        if (clearOverride.has_value()) ApplyClearValueToAll(clearOverride.value());

        if (colorInputUsage || colorOutputUsage)
        {
            for (int i = 0; i < colorCount; i++)
            {
                colorAttachments[i].v = RenderingAttachmentInfo::ColorInfo();

                if(colorInputUsage)  std::get<RenderingAttachmentInfo::ColorInfo>(colorAttachments[i].v).colorInputUsage  = colorInputUsage.value();
                if(colorOutputUsage) std::get<RenderingAttachmentInfo::ColorInfo>(colorAttachments[i].v).colorOutputUsage = colorOutputUsage.value(); 
            }
        }
        
        if (depth && (clearDepth || depthOutputUsage))
        {
            depthAttachment    = RenderingAttachmentInfo(*depth);
            depthAttachment->v = RenderingAttachmentInfo::DepthInfo();
            if (clearDepth       && depthAttachment) std::get<RenderingAttachmentInfo::DepthInfo>(depthAttachment->v).clearDepth       = clearDepth.value();
            if (depthOutputUsage && depthAttachment) std::get<RenderingAttachmentInfo::DepthInfo>(depthAttachment->v).depthOutputUsage = depthOutputUsage.value();
        }
    }

    /*
    * Constructs a RenderingAttachmentInfoGroup directly from a span of Texture<Vulkan> objects.
    */
    inline RenderingAttachmentInfoGroup(
        std::span<const Texture<Vulkan>>     textures,
        Texture<Vulkan>*                     depth            = nullptr,
        std::optional<RenderPassInputUsage>  colorInputUsage  = std::nullopt,
        std::optional<RenderPassOutputUsage> colorOutputUsage = std::nullopt,
        std::optional<bool>                  clearDepth       = std::nullopt,
        std::optional<RenderPassOutputUsage> depthOutputUsage = std::nullopt,
        std::optional<VkClearValue>          clearOverride    = std::nullopt)
    {
        assert(textures.size() <= colorAttachments.size());

        for (const auto& tex : textures) {RenderingAttachmentInfo info(tex); colorAttachments[colorCount++] = info;}

        if (clearOverride.has_value()) ApplyClearValueToAll(clearOverride.value());

        if (colorInputUsage || colorOutputUsage)
        {
            for (int i = 0; i < colorCount; i++)
            {
                colorAttachments[i].v = RenderingAttachmentInfo::ColorInfo();

                if(colorInputUsage)  std::get<RenderingAttachmentInfo::ColorInfo>(colorAttachments[i].v).colorInputUsage  = colorInputUsage.value();
                if(colorOutputUsage) std::get<RenderingAttachmentInfo::ColorInfo>(colorAttachments[i].v).colorOutputUsage = colorOutputUsage.value(); 
            }
        }
        
        if (depth && (clearDepth || depthOutputUsage))
        {
            depthAttachment    = RenderingAttachmentInfo(*depth);
            depthAttachment->v = RenderingAttachmentInfo::DepthInfo();
            if (clearDepth       && depthAttachment) std::get<RenderingAttachmentInfo::DepthInfo>(depthAttachment->v).clearDepth       = clearDepth.value();
            if (depthOutputUsage && depthAttachment) std::get<RenderingAttachmentInfo::DepthInfo>(depthAttachment->v).depthOutputUsage = depthOutputUsage.value();
        }
    }

    /*
    * Constructs a RenderingAttachmentInfoGroup directly from a vulkan render target.
    */
    inline RenderingAttachmentInfoGroup(
        const RenderTarget<Vulkan>&          renderTarget,
        std::optional<RenderPassInputUsage>  colorInputUsage  = std::nullopt,
        std::optional<RenderPassOutputUsage> colorOutputUsage = std::nullopt,
        std::optional<bool>                  clearDepth       = std::nullopt,
        std::optional<RenderPassOutputUsage> depthOutputUsage = std::nullopt,
        std::optional<VkClearValue>          clearOverride    = std::nullopt)
    {
        assert(renderTarget.GetColorAttachments().size() <= colorAttachments.size());

        for (const auto& attachment : renderTarget.GetColorAttachments()) {colorAttachments[colorCount++] = RenderingAttachmentInfo(attachment);}
        
        if (clearOverride.has_value()) ApplyClearValueToAll(clearOverride.value());

        if (colorInputUsage || colorOutputUsage)
        {
            for (int i = 0; i < colorCount; i++)
            {
                colorAttachments[i].v = RenderingAttachmentInfo::ColorInfo();

                if(colorInputUsage)  std::get<RenderingAttachmentInfo::ColorInfo>(colorAttachments[i].v).colorInputUsage  = colorInputUsage.value();
                if(colorOutputUsage) std::get<RenderingAttachmentInfo::ColorInfo>(colorAttachments[i].v).colorOutputUsage = colorOutputUsage.value(); 
            }
        }
        
        if (!renderTarget.GetDepthAttachment().IsEmpty() && (clearDepth || depthOutputUsage))
        {
            depthAttachment    = RenderingAttachmentInfo(renderTarget.GetDepthAttachment());
            depthAttachment->v = RenderingAttachmentInfo::DepthInfo();
            if (clearDepth       && depthAttachment) std::get<RenderingAttachmentInfo::DepthInfo>(depthAttachment->v).clearDepth       = clearDepth.value();
            if (depthOutputUsage && depthAttachment) std::get<RenderingAttachmentInfo::DepthInfo>(depthAttachment->v).depthOutputUsage = depthOutputUsage.value();
        }
    }

    inline bool AddColorAttachment(const RenderingAttachmentInfo& attachment)
    {
        if (colorCount >= colorAttachments.size())
        {
            return false;
        }
        colorAttachments[colorCount++] = attachment;
        return true;
    }

    inline void SetDepthAttachment(const RenderingAttachmentInfo& attachment)
    {
        depthAttachment = attachment;
    }

    inline void ApplyClearValueToAll(const VkClearValue& clear_value)
    {
        for (uint32_t i = 0; i < colorCount; ++i)
        {
            colorAttachments[i].clearValue = clear_value;
        }

        if (depthAttachment.has_value())
        {
            depthAttachment->clearValue = clear_value;
        }
    }

    std::tuple<std::pmr::vector<VkRenderingAttachmentInfo>, std::optional<VkRenderingAttachmentInfo>> ToVkAttachmentInfos() const;

    VkRect2D GetRenderArea() const;
};

/// Context for a single dynamic rendering context
template<>
class RenderContext<Vulkan> final
{
    using Texture = Texture<Vulkan>;

    RenderContext( const RenderContext<Vulkan>& ) noexcept = delete;
    RenderContext<Vulkan>& operator=( const RenderContext<Vulkan>& ) noexcept = delete;
public:
    RenderContext(RenderContext<Vulkan>&&) noexcept;
    RenderContext<Vulkan>& operator=(RenderContext<Vulkan>&&) noexcept;

    RenderContext() noexcept = default;
    RenderContext(RenderPass<Vulkan>, Pipeline<Vulkan>, Framebuffer<Vulkan>, std::string name) noexcept;
    RenderContext(RenderPass<Vulkan>, Framebuffer<Vulkan>, std::string name) noexcept;
    RenderContext(std::span<TextureFormat>, TextureFormat, TextureFormat, std::string name) noexcept;

    RenderContext<Vulkan> Copy() const;

    // this
    struct RenderPassContextData {
        RenderPassContextData( RenderPass<Vulkan>, Pipeline<Vulkan>, Framebuffer<Vulkan>, RenderPassClearData) noexcept;
        RenderPassContextData& operator=( RenderPassContextData&& other ) noexcept;
        RenderPassContextData( RenderPassContextData&& other ) noexcept = default;

        RenderPass<Vulkan>    renderPass{};
        Pipeline<Vulkan>      overridePipeline{};
        Framebuffer<Vulkan>   framebuffer{};
        RenderPassClearData   renderPassClearData{};
    };

    uint32_t              viewMask = 0;
    uint32_t              subPass = 0;
    Msaa                  msaa = Msaa::Samples1;

    // or this
    struct DynamicRenderContextData {
        std::vector<VkFormat> colorAttachmentFormats;
        VkFormat              depthAttachmentFormat = VK_FORMAT_UNDEFINED;
        VkFormat              stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
    };

    std::variant<std::monostate, RenderPassContextData, DynamicRenderContextData> v;

    bool                        IsDynamic() const { return std::holds_alternative<DynamicRenderContextData>(v); }
    fvk::VkRenderPassBeginInfo  GetRenderPassBeginInfo() const;                                                                       // render pass rendering
    fvk::VkRenderingInfo        GetRenderingInfo(const RenderingAttachmentInfoGroup&, std::optional< VkRect2D> = std::nullopt) const; // dynamic rendering

    Pipeline<Vulkan>            GetOverridePipeline() const;
    RenderPass<Vulkan>          GetRenderPass() const;
    const Framebuffer<Vulkan>*  GetFramebuffer() const;
    size_t                      GetNumColorAttachmentFormats() const;
    fvk::VkPipelineRenderingCreateInfo GetPipelineRenderingCreateInfo() const;
    RenderPassClearData GetRenderPassClearData() const;

    std::string           name {};
};


// /// Dynamic Rendering Context
// struct RenderingContext
// {
//     RenderingContext() = default;
//     RenderingContext(const RenderingPassContext& passContext)
//         : passContexts({ passContext })
//     {
//     }

//     std::vector< VkSampleCountFlagBits> passMultisample;
//     std::vector< RenderingPassContext>  passContexts;
// };
