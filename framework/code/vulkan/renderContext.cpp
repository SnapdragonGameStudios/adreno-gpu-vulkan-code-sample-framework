#include "renderContext.hpp"
#include "allocator/threadBufferResource.hpp"

namespace
{
    // TODO: Make the same function from vulkan.cpp public accessible and replace this with it
    static constexpr VkAttachmentLoadOp InputUsageToVkAttachmentLoadOp(const RenderPassInputUsage t)
    {
        constexpr VkAttachmentLoadOp cRenderPassInputUsageToVk[]{
            VK_ATTACHMENT_LOAD_OP_CLEAR,
            VK_ATTACHMENT_LOAD_OP_LOAD,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE
        };
        return cRenderPassInputUsageToVk[(int)t];
    }
}

//-----------------------------------------------------------------------------
VkRenderingAttachmentInfo RenderingAttachmentInfo::ToVkAttachmentInfo() const
//-----------------------------------------------------------------------------
{
    VkRenderingAttachmentInfo attachmentInfo = renderingAttachmentInfo;
    attachmentInfo.resolveMode = VK_RESOLVE_MODE_NONE;
    attachmentInfo.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    // Color attachment
    if (std::holds_alternative<ColorInfo>(v))
    {
        const auto& info           = std::get<ColorInfo>(v);
        attachmentInfo.loadOp      = InputUsageToVkAttachmentLoadOp(info.colorInputUsage);
        attachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        switch (info.colorOutputUsage) {
        case RenderPassOutputUsage::Discard:
            attachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            break;
        case RenderPassOutputUsage::Store:
        case RenderPassOutputUsage::StoreReadOnly:
        case RenderPassOutputUsage::StoreTransferSrc:
            attachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            break;
        case RenderPassOutputUsage::Clear:
            assert(0); // currently unsupported
            break;
        case RenderPassOutputUsage::Present:
            break;
        }
    }
    // Depth attachment
    else if (std::holds_alternative<DepthInfo>(v))
    {
        const auto& info           = std::get<DepthInfo>(v);
        attachmentInfo.loadOp      = InputUsageToVkAttachmentLoadOp(info.clearDepth ? RenderPassInputUsage::Clear : RenderPassInputUsage::Load);
        attachmentInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        switch (info.depthOutputUsage) {
        case RenderPassOutputUsage::Discard:
            attachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            break;
        case RenderPassOutputUsage::Store:
        case RenderPassOutputUsage::StoreReadOnly:
        case RenderPassOutputUsage::StoreTransferSrc:
            attachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            break;
        case RenderPassOutputUsage::Clear:
            assert(0); // currently unsupported
            break;
        case RenderPassOutputUsage::Present:
            break;
        }
    }
    // Not-specified, go with default values (might be expensive and not work well with local load/read)
    else
    {
        attachmentInfo.loadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        if (renderingAttachmentInfo.imageLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        {
            attachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        else if (renderingAttachmentInfo.imageLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            || renderingAttachmentInfo.imageLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        {
            attachmentInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }
        else
        {
            // Cannot decide if color or depth
            assert(false);
        }
    }

    // Rendering local read override (check for original layout)
    // Note: From the spec, VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR can be used as color/depth
    // attachment, storage and input operations (barriers are need to ensure memory visibility)
    if (renderingAttachmentInfo.imageLayout == VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR)
    {
        attachmentInfo.imageLayout = VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR;
    }

    attachmentInfo.clearValue = clearValue.value_or(attachmentInfo.clearValue);

    return attachmentInfo;
}

//-----------------------------------------------------------------------------
std::tuple<std::pmr::vector<VkRenderingAttachmentInfo>, std::optional<VkRenderingAttachmentInfo>> RenderingAttachmentInfoGroup::ToVkAttachmentInfos() const
//-----------------------------------------------------------------------------
{
    core::ThreadAutomaticMonotonicMemoryResource threadMemoryResource;
    std::pmr::vector<VkRenderingAttachmentInfo> colorInfos(threadMemoryResource);
    colorInfos.reserve(colorCount);

    for (uint32_t i = 0; i < colorCount; ++i)
    {
        colorInfos.push_back(colorAttachments[i].ToVkAttachmentInfo());
    }

    std::optional<VkRenderingAttachmentInfo> depth_info;
    if (depthAttachment.has_value())
    {
        depth_info = depthAttachment->ToVkAttachmentInfo();
    }

    return { std::move(colorInfos), depth_info };
}

//-----------------------------------------------------------------------------
VkRect2D RenderingAttachmentInfoGroup::GetRenderArea() const
//-----------------------------------------------------------------------------
{
    VkRect2D renderArea = VkRect2D{ {0, 0}, {0, 0} };

    for (int i = 0; i < colorCount; i++)
    {
        renderArea.extent.width = std::max(renderArea.extent.width, colorAttachments[i].renderArea.width);
        renderArea.extent.height = std::max(renderArea.extent.height, colorAttachments[i].renderArea.height);
    }

    if (depthAttachment)
    {
        renderArea.extent.width = std::max(renderArea.extent.width, depthAttachment->renderArea.width);
        renderArea.extent.height = std::max(renderArea.extent.height, depthAttachment->renderArea.height);
    }

    return renderArea;
}

//-----------------------------------------------------------------------------
RenderContext<Vulkan>::RenderContext( RenderContext<Vulkan>&& other ) noexcept = default;
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
RenderContext<Vulkan>& RenderContext<Vulkan>::operator=( RenderContext<Vulkan>&& other ) noexcept
//-----------------------------------------------------------------------------
{
    if (this != &other)
    {
        v = std::move( other.v );

        viewMask = other.viewMask;
        other.viewMask = {};
        subPass = other.subPass;
        other.subPass = 0;
        msaa = other.msaa;
        other.msaa = Msaa::Samples1;
        name = std::move( other.name );
    }

    return *this;
}

//-----------------------------------------------------------------------------
RenderContext<Vulkan>::RenderPassContextData& RenderContext<Vulkan>::RenderPassContextData::operator=( RenderContext<Vulkan>::RenderPassContextData&& other ) noexcept
//-----------------------------------------------------------------------------
{
    if (this != &other)
    {
        renderPass = std::move( other.renderPass );
        overridePipeline = std::move( other.overridePipeline);
        framebuffer = std::move( other.framebuffer );
        renderPassClearData = std::move( other.renderPassClearData);
    }
    return *this;
}

//-----------------------------------------------------------------------------
RenderContext<Vulkan>::RenderContext( RenderPass<Vulkan> _renderPass, Pipeline<Vulkan> _pipeline, Framebuffer<Vulkan> _framebuffer, std::string _name ) noexcept
//-----------------------------------------------------------------------------
    : v{std::move( RenderPassContextData {
            std::move( _renderPass ), std::move( _pipeline ), _framebuffer, _framebuffer.GetRenderPassClearData().Copy()
        } )}
    , name{std::move( _name )}
{
}

//-----------------------------------------------------------------------------
RenderContext<Vulkan>::RenderContext( RenderPass<Vulkan> _renderPass, Framebuffer<Vulkan> _framebuffer, std::string _name ) noexcept
//-----------------------------------------------------------------------------
    : v{std::move( RenderPassContextData {
            std::move( _renderPass ), Pipeline<Vulkan>(), std::move(_framebuffer), _framebuffer.GetRenderPassClearData().Copy()
        } )}
    , name{std::move( _name )}
{
}

//-----------------------------------------------------------------------------
RenderContext<Vulkan>::RenderContext(std::span<TextureFormat> _colorAttachmentFormats, TextureFormat _depthAttachmentFormat, TextureFormat _stencilAttachmentFormat, std::string _name ) noexcept
//-----------------------------------------------------------------------------
    : v{std::move(DynamicRenderContextData {
            [] (std::span<TextureFormat> formats) -> std::vector<VkFormat> {
                std::vector<VkFormat> vkFormats;
                vkFormats.reserve(formats.size());
                for (const auto& format : formats)
                {
                    vkFormats.push_back(TextureFormatToVk(format));
                }
                return vkFormats;
            }(_colorAttachmentFormats),
            TextureFormatToVk(_depthAttachmentFormat),
            TextureFormatToVk(_stencilAttachmentFormat)
        } )}
    , name{std::move(_name)}
{
}


//-----------------------------------------------------------------------------
RenderContext<Vulkan>::RenderPassContextData::RenderPassContextData( RenderPass<Vulkan> _renderPass, Pipeline<Vulkan> _pipeline, Framebuffer<Vulkan> _framebuffer, RenderPassClearData _renderPassClearData) noexcept
    : renderPass{std::move( _renderPass )}
    , overridePipeline{std::move( _pipeline )}
    , framebuffer{std::move( _framebuffer )}
    , renderPassClearData{std::move(_renderPassClearData)}
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
fvk::VkRenderPassBeginInfo RenderContext<Vulkan>::GetRenderPassBeginInfo() const
//-----------------------------------------------------------------------------
{
    const RenderPassContextData& context = std::get<RenderPassContextData>( v );
    fvk::VkRenderPassBeginInfo renderPassBeginInfo{VkRenderPassBeginInfo{
        .renderPass = context.renderPass.mRenderPass,
        .framebuffer = context.framebuffer,
        .renderArea = context.renderPassClearData.scissor,
    }};
    auto infoClearValues = renderPassBeginInfo.AddMemberArray<VkClearValue, offsetof(::VkRenderPassBeginInfo, pClearValues), offsetof( ::VkRenderPassBeginInfo, clearValueCount)>(context.renderPassClearData.clearValues.size());
    std::copy( context.renderPassClearData.clearValues.begin(), context.renderPassClearData.clearValues.end(), infoClearValues.begin() );

    return renderPassBeginInfo;
}

//-----------------------------------------------------------------------------
fvk::VkRenderingInfo RenderContext<Vulkan>::GetRenderingInfo(const RenderingAttachmentInfoGroup& renderingAttachmentInfoGroup, std::optional< VkRect2D> renderArea) const
//-----------------------------------------------------------------------------
{
    auto [colorAttachmentInfo, depthAttachmentInfo] = renderingAttachmentInfoGroup.ToVkAttachmentInfos();

    const DynamicRenderContextData& context = std::get<DynamicRenderContextData>( v );
    fvk::VkRenderingInfo renderingInfo{VkRenderingInfo{
        .renderArea = renderArea ? renderArea.value() : renderingAttachmentInfoGroup.GetRenderArea(),
        .layerCount = 1,
        .pStencilAttachment = VK_NULL_HANDLE,
    }};

    auto colorAttachmentValues = renderingInfo.AddMemberArray<VkRenderingAttachmentInfo, offsetof(::VkRenderingInfo, pColorAttachments), offsetof(::VkRenderingInfo, colorAttachmentCount)>(colorAttachmentInfo.size());
    std::copy(colorAttachmentInfo.begin(), colorAttachmentInfo.end(), colorAttachmentValues.begin());

    if (depthAttachmentInfo)
    {
        auto depthAttachmentValue = renderingInfo.AddMemberArray<VkRenderingAttachmentInfo, offsetof(::VkRenderingInfo, pDepthAttachment), 0>(1);
        depthAttachmentValue[0] = depthAttachmentInfo.value();
    }

    return renderingInfo;
}

//-----------------------------------------------------------------------------
Pipeline<Vulkan> RenderContext<Vulkan>::GetOverridePipeline() const
//-----------------------------------------------------------------------------
{
    if (!IsDynamic())
    {
        return std::get<RenderPassContextData>( v ).overridePipeline.Copy();
    }
    else
        return {};
}

//-----------------------------------------------------------------------------
RenderPassClearData RenderContext<Vulkan>::GetRenderPassClearData() const
//-----------------------------------------------------------------------------
{
    if (!IsDynamic())
    {
        return std::get<RenderPassContextData>( v ).renderPassClearData.Copy();
    }
    else
        return {};
}

//-----------------------------------------------------------------------------
RenderPass<Vulkan> RenderContext<Vulkan>::GetRenderPass() const
//-----------------------------------------------------------------------------
{
    if (!IsDynamic())
    {
        return std::get<RenderPassContextData>( v ).renderPass.Copy();
    }
    else
        return {};
}

//-----------------------------------------------------------------------------
const Framebuffer<Vulkan>* RenderContext<Vulkan>::GetFramebuffer() const
//-----------------------------------------------------------------------------
{
    if (!IsDynamic())
    {
        return &std::get<RenderPassContextData>( v ).framebuffer;
    }
    else
        return nullptr;
}

//-----------------------------------------------------------------------------
size_t  RenderContext<Vulkan>::GetNumColorAttachmentFormats() const
//-----------------------------------------------------------------------------
{
    if (IsDynamic())
    {
        return std::get<DynamicRenderContextData>(v).colorAttachmentFormats.size();
    }
    else
        return 0;
}

//-----------------------------------------------------------------------------
fvk::VkPipelineRenderingCreateInfo RenderContext<Vulkan>::GetPipelineRenderingCreateInfo() const
//-----------------------------------------------------------------------------
{
    const DynamicRenderContextData& context = std::get<DynamicRenderContextData>( v );
    fvk::VkPipelineRenderingCreateInfo info{VkPipelineRenderingCreateInfo{
        .viewMask = viewMask,
        .depthAttachmentFormat = context.depthAttachmentFormat,
        .stencilAttachmentFormat = context.stencilAttachmentFormat
    }};
    auto infoColorAttachments = info.AddMemberArray<VkFormat, offsetof( VkPipelineRenderingCreateInfo, pColorAttachmentFormats ), offsetof( VkPipelineRenderingCreateInfo, colorAttachmentCount )>( context.colorAttachmentFormats.size() );
    std::copy( context.colorAttachmentFormats.begin(), context.colorAttachmentFormats.end(), infoColorAttachments.begin() );

    return info;
}

