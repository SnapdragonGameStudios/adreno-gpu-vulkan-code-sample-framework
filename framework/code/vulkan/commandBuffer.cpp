//============================================================================================================
//
//
//                  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "commandBuffer.hpp"
#include "framebuffer.hpp"
#include "renderContext.hpp"
#include "renderTarget.hpp"
#include "timerPool.hpp"
#include "vulkan/extensionLib.hpp"
#include "vulkan_support.hpp"

//-----------------------------------------------------------------------------
CommandList<Vulkan>::CommandList() noexcept
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
CommandList<Vulkan>::~CommandList()
//-----------------------------------------------------------------------------
{
    Release();
}

//-----------------------------------------------------------------------------
CommandList<Vulkan>::CommandList(CommandList<Vulkan>&& other) noexcept
//---------------------------------------------------------------------------
{
    assert(0);  // Currently move is not implemented for this class, but std::vector will not compile if a move (or copy) is not provided, we currently expect that the move will not get called (ie vector of CommandList<Vulkan> will not resize)
}

//-----------------------------------------------------------------------------
CommandList<Vulkan>& CommandList<Vulkan>::operator=(CommandList<Vulkan> && other) noexcept
//-----------------------------------------------------------------------------
{
    assert(0);  // Currently move is not implemented for this class, but std::vector will not compile if a move (or copy) is not provided, we currently expect that the move will not get called (ie vector of CommandList<Vulkan> will not resize)
    return *this;
}


//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::Initialize(Vulkan* pVulkan, const std::string& Name, CommandListBase::Type CmdBuffType, uint32_t QueueIndex, TimerPoolBase* pGpuTimerPool)
//-----------------------------------------------------------------------------
{
    m_Name = Name;

    // Reset draw counts
    m_NumDrawCalls = 0;
    m_NumTriangles = 0;

    // Need Vulkan objects to release ourselves
    m_pVulkan = pVulkan;

    // Timer queries
    m_GpuTimerPool = pGpuTimerPool;
    assert(m_GpuTimerQueries.empty());

    // What type of command buffer are we dealing with
    VkCommandBufferLevel CmdBuffLevel = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    m_Type = Type::Primary;
    switch (CmdBuffType) {
        case Type::Primary:
            break;
        case Type::Secondary:
            m_Type = Type::Secondary;
            CmdBuffLevel = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
            break;
        default:
            LOGE( "Command Buffer initialization with unsupported type! (%d)", CmdBuffType );
            break;
    }

    // Store which queue this command list is using (index in to Vulkan::m_VulkanQueues) so we can submit to the associated device queue.
    assert(pVulkan->m_VulkanQueues[QueueIndex].Queue != VK_NULL_HANDLE);
    m_QueueIndex = QueueIndex;
    pVulkan->AllocateCommandBuffer(CmdBuffLevel, QueueIndex, &m_VkCommandBuffer);
    pVulkan->SetDebugObjectName(m_VkCommandBuffer, m_Name.c_str());
    return true;
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::Reset()
//-----------------------------------------------------------------------------
{
    VkResult RetVal;

    if (m_VkCommandBuffer == VK_NULL_HANDLE)
    {
        LOGE("Error! Trying to reset command buffer before it has been initialized: %s", m_Name.c_str());
        return false;
    }

    // vkBeginCommandBuffer should reset the command buffer, but Reset can be called
    // to make it more explicit.
    RetVal = vkResetCommandBuffer(m_VkCommandBuffer, 0);
    if (!CheckVkError("vkResetCommandBuffer()", RetVal))
    {
        LOGE("Unable to reset command buffer: %s", m_Name.c_str());
        return false;
    }

    // Reset draw counts
    m_NumDrawCalls = 0;
    m_NumTriangles = 0;

    // Reset queries
    m_GpuTimerPool->TimerCommandBufferReset(m_GpuTimerQueries);
    m_GpuTimerQueries.clear();

    return true;
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::Begin( VkCommandBufferUsageFlags CmdBuffUsage )
//-----------------------------------------------------------------------------
{
    assert(m_Type == Type::Primary);
    if (m_Type != Type::Primary)
    {
        LOGE("Error! Trying to begin secondary command buffer without Framebuffer or RenderPass: %s", m_Name.c_str());
        return false;
    }

    const VkCommandBufferBeginInfo CmdBeginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = CmdBuffUsage,
    };
    return Begin( CmdBeginInfo );
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::Begin( VkFramebuffer FrameBuffer, VkRenderPass RenderPass, bool IsSwapChainRenderPass, uint32_t SubPass, VkCommandBufferUsageFlags CmdBuffUsage )
//-----------------------------------------------------------------------------
{
    if (m_VkCommandBuffer == VK_NULL_HANDLE)
    {
        LOGE( "Error! Trying to begin command buffer before it has been initialized: %s", m_Name.c_str() );
        return false;
    }

    assert( m_Type == Type::Secondary );
    if (m_Type != Type::Secondary)
    {
        LOGE( "Error! Trying to begin primary command buffer with inheritance info (%s)", m_Name.c_str() );
        return false;
    }

    VkCommandBufferInheritanceInfo InheritanceInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
        .framebuffer = FrameBuffer,
        .occlusionQueryEnable = VK_FALSE,
        .queryFlags = 0,
        .pipelineStatistics = 0,
    };
    VkCommandBufferBeginInfo CmdBeginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = CmdBuffUsage,
        .pInheritanceInfo = &InheritanceInfo,
    };

    VkCommandBufferInheritanceRenderPassTransformInfoQCOM InheritanceInfoRenderPassTransform = {};

    // Secondary command buffer MAY BE inside another render pass (compute can be in a secondary buffer and not have/inherit a renderder pass)
    if (RenderPass != VK_NULL_HANDLE)
    {
        CmdBeginInfo.flags |= VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;

        InheritanceInfo.renderPass = RenderPass;
        InheritanceInfo.subpass = SubPass;

        // We may also have to pass the inherited render pass transform down to the secondary buffer
        if (IsSwapChainRenderPass && m_pVulkan->FillCommandBufferInheritanceRenderPassTransformInfoQCOM( InheritanceInfoRenderPassTransform ))
        {
            LOGI( "VkCommandBufferInheritanceRenderPassTransformInfoQCOM (%s): Extents = (%d x %d)", m_Name.c_str(), InheritanceInfoRenderPassTransform.renderArea.extent.width, InheritanceInfoRenderPassTransform.renderArea.extent.height );
            InheritanceInfo.pNext = &InheritanceInfoRenderPassTransform;
        }
    }

    return Begin( CmdBeginInfo );
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::Begin( const VkCommandBufferBeginInfo& CmdBeginInfo )
//-----------------------------------------------------------------------------
{
    // By calling vkBeginCommandBuffer, cmdBuffer is put into the recording state.
    VkResult RetVal = vkBeginCommandBuffer(m_VkCommandBuffer, &CmdBeginInfo );
    if (!CheckVkError("vkBeginCommandBuffer()", RetVal))
    {
        LOGE("Unable to begin command buffer: %s", m_Name.c_str());
        return false;
    }

    // Reset draw counts
    m_NumDrawCalls = 0;
    m_NumTriangles = 0;

    // Reset queries
    m_GpuTimerPool->TimerCommandBufferReset(m_GpuTimerQueries);
    m_GpuTimerQueries.clear();

    return true;
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::Begin( VkFramebuffer FrameBuffer, const RenderPass<Vulkan>& RenderPass, bool IsSwapChainRenderPass, uint32_t SubPass, VkCommandBufferUsageFlags CmdBuffUsage )
//-----------------------------------------------------------------------------
{
    return Begin( FrameBuffer, RenderPass.mRenderPass, IsSwapChainRenderPass, SubPass, CmdBuffUsage );
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::Begin( const Framebuffer<Vulkan>& FrameBuffer, const RenderPass<Vulkan>& RenderPass, bool IsSwapChainRenderPass, uint32_t SubPass, VkCommandBufferUsageFlags CmdBuffUsage )
//-----------------------------------------------------------------------------
{
    return Begin( FrameBuffer.m_FrameBuffer, RenderPass.mRenderPass, IsSwapChainRenderPass, SubPass, CmdBuffUsage );
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::Begin(const VkCommandBufferInheritanceRenderingInfo& DynamicRenderingInheritanceInfo, VkCommandBufferUsageFlags CmdBuffUsage)
//-----------------------------------------------------------------------------
{
    assert(m_Type == Type::Secondary);
    if (m_Type != Type::Secondary)
    {
        LOGE("Error! Trying to begin primary command buffer with inheritance info (%s)", m_Name.c_str());
        return false;
    }

    VkCommandBufferInheritanceInfo InheritanceInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
        .pNext = &DynamicRenderingInheritanceInfo
    };

    VkCommandBufferBeginInfo CmdBeginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = CmdBuffUsage | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
        .pInheritanceInfo = &InheritanceInfo,
    };

    return Begin(CmdBeginInfo);
}


//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::Begin(const RenderContext<Vulkan>& renderContext, VkCommandBufferUsageFlags cmdBuffUsage)
//-----------------------------------------------------------------------------
{
    assert( m_Type == Type::Secondary );
    if (m_Type != Type::Secondary)
    {
        LOGE( "Error! Trying to begin primary command buffer with inheritance info (%s)", m_Name.c_str() );
        return false;
    }

    if (renderContext.IsDynamic())
    {
        // dynamic rendering
        const auto& dynamicRenderContext = std::get<RenderContext<Vulkan>::DynamicRenderContextData>( renderContext.v );
        VkCommandBufferInheritanceRenderingInfo dynamicRenderingInheritanceInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO,
            .flags = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT,
            .colorAttachmentCount = (uint32_t)dynamicRenderContext.colorAttachmentFormats.size(),
            .pColorAttachmentFormats = dynamicRenderContext.colorAttachmentFormats.data(),
            .depthAttachmentFormat = dynamicRenderContext.depthAttachmentFormat,
            .stencilAttachmentFormat = dynamicRenderContext.stencilAttachmentFormat,
            .rasterizationSamples = EnumToVk( renderContext.msaa )
        };
        return Begin( dynamicRenderingInheritanceInfo, cmdBuffUsage );
    }
    else
    {
        const auto& renderPassContext = std::get<RenderContext<Vulkan>::RenderPassContextData>(renderContext.v);

        return Begin( renderPassContext.framebuffer, renderPassContext.renderPass, false, renderContext.subPass, cmdBuffUsage );
    }
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::BeginRenderPass(VkRect2D RenderExtent, float MinDepth, float MaxDepth, const std::span<const VkClearColorValue> ClearColors, uint32_t NumColorBuffers, bool HasDepth, VkRenderPass RenderPass, bool IsSwapChainRenderPass, VkFramebuffer FrameBuffer, VkSubpassContents SubContents)
//-----------------------------------------------------------------------------
{
    VkResult RetVal = VK_SUCCESS;

    // ... set Viewport and Scissor for this pass ...
    const VkViewport Viewport{
        .width = (float)RenderExtent.extent.width,
        .height = (float)RenderExtent.extent.height,
        .minDepth = MinDepth,
        .maxDepth = MaxDepth
    };
    vkCmdSetViewport(m_VkCommandBuffer, 0, 1, &Viewport);
    vkCmdSetScissor (m_VkCommandBuffer, 0, 1, &RenderExtent);

    // ... set clear values ...
    std::array<VkClearValue, 9> ClearValues;
    uint32_t ClearValuesCount = 0;

    assert(NumColorBuffers <= (ClearValues.size() - (HasDepth?1:0)));
    for (uint32_t i=0; i< NumColorBuffers; ++i)
    {
        ClearValues[ClearValuesCount].color = ClearColors[ClearColors.size() > NumColorBuffers ? ClearValuesCount : 0];
        ++ClearValuesCount;
    }
    if (HasDepth)
    {
        ClearValues[ClearValuesCount].depthStencil.depth = MaxDepth;
        ClearValues[ClearValuesCount].depthStencil.stencil = 0;
        ++ClearValuesCount;
    }

    // ... begin render pass ...
    fvk::VkRenderPassBeginInfo RPBeginInfo{{
        .renderPass = RenderPass,
        .framebuffer = FrameBuffer,
        .renderArea = RenderExtent,
        .clearValueCount = ClearValuesCount,
        .pClearValues = ClearValues.data()
    }};

    fvk::VkRenderPassTransformBeginInfoQCOM RPTransformBeginInfoQCOM = {};
    if (IsSwapChainRenderPass && m_pVulkan->FillRenderPassTransformBeginInfoQCOM(*RPTransformBeginInfoQCOM))
    {
        //LOGI("RPTransformBeginInfoQCOM(%s): Pre swap Extents = (%d x %d) transform = %d", m_Name.c_str(), RPBeginInfo.renderArea.extent.width, RPBeginInfo.renderArea.extent.height, RPTransformBeginInfoQCOM.transform);
        RPBeginInfo.Add( RPTransformBeginInfoQCOM );
    }

    //LOGI("BeginRenderPass(%s): Extents = (%d x %d) %s", m_Name.c_str(), RPBeginInfo.renderArea.extent.width, RPBeginInfo.renderArea.extent.height, (RPBeginInfo.pNext == &RPTransformBeginInfoQCOM) ? " (transformed)" : "");

    return BeginRenderPass(*RPBeginInfo, SubContents);
}


//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::BeginRenderPass(VkRect2D RenderExtent, float MinDepth, float MaxDepth, const std::span<const VkClearColorValue> ClearColors, uint32_t NumColorBuffers, bool HasDepth, const RenderPass<Vulkan>& RenderPass, bool IsSwapChainRenderPass, VkFramebuffer FrameBuffer, VkSubpassContents SubContents)
//-----------------------------------------------------------------------------
{
    return BeginRenderPass( RenderExtent, MinDepth, MaxDepth, ClearColors, NumColorBuffers, HasDepth, RenderPass.mRenderPass, IsSwapChainRenderPass, FrameBuffer, SubContents );
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::BeginRenderPass(const RenderTarget<Vulkan>& RenderTarget, VkRenderPass RenderPass, VkSubpassContents SubContents)
//-----------------------------------------------------------------------------
{
    VkRect2D Scissor = {};
    Scissor.offset.x = 0;
    Scissor.offset.y = 0;
    Scissor.extent.width = RenderTarget.m_Width;
    Scissor.extent.height = RenderTarget.m_Height;

    return BeginRenderPass(Scissor, 0.0f, 1.0f, { RenderTarget.m_ClearColorValues.data(), RenderTarget.m_ClearColorValues.size() }, RenderTarget.GetNumColorLayers(), RenderTarget.m_DepthFormat != TextureFormat::UNDEFINED, RenderPass, false, RenderTarget.m_FrameBuffer, SubContents);
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::BeginRenderPass( const RenderTarget<Vulkan>& RenderTarget, const RenderPass<Vulkan>& RenderPass, VkSubpassContents SubContents )
//-----------------------------------------------------------------------------
{
    return BeginRenderPass(RenderTarget, RenderPass.mRenderPass, SubContents);
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::BeginRenderPass( const Framebuffer<Vulkan>& Framebuffer, const RenderPass<Vulkan>& RenderPass, const RenderPassClearData& RenderPassClearData, VkSubpassContents SubContents )
//-----------------------------------------------------------------------------
{
    VkResult RetVal = VK_SUCCESS;

    vkCmdSetViewport( m_VkCommandBuffer, 0, 1, &RenderPassClearData.viewport );
    vkCmdSetScissor( m_VkCommandBuffer, 0, 1, &RenderPassClearData.scissor);

    // ... begin render pass ...
    fvk::VkRenderPassBeginInfo RPBeginInfo{{
        .renderPass = RenderPass.mRenderPass,
        .framebuffer = Framebuffer.m_FrameBuffer,
        .renderArea = RenderPassClearData.scissor,
        .clearValueCount = (uint32_t)RenderPassClearData.clearValues.size(),
        .pClearValues = RenderPassClearData.clearValues.data()
    }};

    //fvk::VkRenderPassTransformBeginInfoQCOM RPTransformBeginInfoQCOM = {};
    //if (IsSwapChainRenderPass && m_pVulkan->FillRenderPassTransformBeginInfoQCOM( *RPTransformBeginInfoQCOM ))
    //{
    //    //LOGI("RPTransformBeginInfoQCOM(%s): Pre swap Extents = (%d x %d) transform = %d", m_Name.c_str(), RPBeginInfo.renderArea.extent.width, RPBeginInfo.renderArea.extent.height, RPTransformBeginInfoQCOM.transform);
    //    RPBeginInfo.Add( RPTransformBeginInfoQCOM );
    //}

    //LOGI("BeginRenderPass(%s): Extents = (%d x %d) %s", m_Name.c_str(), RPBeginInfo.renderArea.extent.width, RPBeginInfo.renderArea.extent.height, (RPBeginInfo.pNext == &RPTransformBeginInfoQCOM) ? " (transformed)" : "");

    return BeginRenderPass( *RPBeginInfo, SubContents );
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::BeginSwapchainRenderPass( uint32_t SwapchainImageIndex, const RenderPass<Vulkan>& RenderPass, const RenderPassClearData& RenderPassClearData, VkSubpassContents SubContents )
//-----------------------------------------------------------------------------
{
    VkResult RetVal = VK_SUCCESS;

    vkCmdSetViewport( m_VkCommandBuffer, 0, 1, &RenderPassClearData.viewport );
    vkCmdSetScissor( m_VkCommandBuffer, 0, 1, &RenderPassClearData.scissor );

    // ... begin render pass ...
    fvk::VkRenderPassBeginInfo RPBeginInfo{{
        .renderPass = RenderPass.mRenderPass,
        .framebuffer = m_pVulkan->m_SwapchainBuffers[SwapchainImageIndex].framebuffer.m_FrameBuffer,
        .renderArea = RenderPassClearData.scissor,
        .clearValueCount = (uint32_t)RenderPassClearData.clearValues.size(),
        .pClearValues = RenderPassClearData.clearValues.data()
    }};

    fvk::VkRenderPassTransformBeginInfoQCOM RPTransformBeginInfoQCOM = {};
    if (m_pVulkan->FillRenderPassTransformBeginInfoQCOM( *RPTransformBeginInfoQCOM ))
    {
        //LOGI("RPTransformBeginInfoQCOM(%s): Pre swap Extents = (%d x %d) transform = %d", m_Name.c_str(), RPBeginInfo.renderArea.extent.width, RPBeginInfo.renderArea.extent.height, RPTransformBeginInfoQCOM.transform);
        RPBeginInfo.Add( RPTransformBeginInfoQCOM );
    }

    //LOGI("BeginSwapchainRenderPass(%s): Extents = (%d x %d) %s", m_Name.c_str(), RPBeginInfo.renderArea.extent.width, RPBeginInfo.renderArea.extent.height, (RPBeginInfo.pNext == &RPTransformBeginInfoQCOM) ? " (transformed)" : "");

    return BeginRenderPass( *RPBeginInfo, SubContents );
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::BeginRenderPass( const RenderContext<Vulkan>& renderContext, VkSubpassContents SubContents )
//-----------------------------------------------------------------------------
{
    // ... begin render pass ...
    fvk::VkRenderPassBeginInfo RPBeginInfo{ renderContext.GetRenderPassBeginInfo() };
    return BeginRenderPass( *RPBeginInfo, SubContents );
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::BeginRenderPass( const VkRenderPassBeginInfo& RPBeginInfo, VkSubpassContents SubContents )
//-----------------------------------------------------------------------------
{
    vkCmdBeginRenderPass(m_VkCommandBuffer, &RPBeginInfo, SubContents);
    return true;
}

//-----------------------------------------------------------------------------
void CommandList<Vulkan>::BeginRenderPass( const VkRenderingInfo& renderingInfo )
//-----------------------------------------------------------------------------
{
    auto* dynamicRenderingExt = m_pVulkan->GetExtension<ExtensionLib::Ext_VK_KHR_dynamic_rendering>();
    assert((dynamicRenderingExt != nullptr) && "dynamic rendering overload of BeginRenderPass(...) called without dynamic rendering extension available/enabled");
    dynamicRenderingExt->m_vkCmdBeginRenderingKHR( m_VkCommandBuffer, &renderingInfo );
}

//-----------------------------------------------------------------------------
void CommandList<Vulkan>::EndRendering()
//-----------------------------------------------------------------------------
{
    auto* dynamicRenderingExt = m_pVulkan->GetExtension<ExtensionLib::Ext_VK_KHR_dynamic_rendering>();
    assert((dynamicRenderingExt != nullptr) && "dynamic rendering function EndRendering() called without dynamic rendering extension available/enabled");
    dynamicRenderingExt->m_vkCmdEndRenderingKHR( m_VkCommandBuffer );
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::NextSubpass( VkSubpassContents SubContents )
//-----------------------------------------------------------------------------
{
    vkCmdNextSubpass(m_VkCommandBuffer, SubContents);
    return true;
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::EndRenderPass()
//-----------------------------------------------------------------------------
{
    vkCmdEndRenderPass(m_VkCommandBuffer);
    return true;
}

//-----------------------------------------------------------------------------
int CommandList<Vulkan>::StartGpuTimer(const std::string_view& timerName)
//-----------------------------------------------------------------------------
{
    if (!m_GpuTimerPool)
        return -1;
    int deviceQueueFamilyIndex = m_pVulkan->m_VulkanQueues[m_QueueIndex].QueueFamilyIndex;
    assert(deviceQueueFamilyIndex >= 0);
    const auto timerId = m_GpuTimerPool->StartGpuTimer(m_VkCommandBuffer, timerName, deviceQueueFamilyIndex);
    if (timerId>=0)
        m_GpuTimerQueries.push_back(timerId);
    return timerId;
}

//-----------------------------------------------------------------------------
void CommandList<Vulkan>::StopGpuTimer(int timerId)
//-----------------------------------------------------------------------------
{
    if (timerId < 0)
        return;
    // assume timerId is in m_GpuTimerQueries (we are in trouble if it is not, likely didnt call StartGpuTimer)
    assert(m_GpuTimerPool);
    m_GpuTimerPool->StopGpuTimer(m_VkCommandBuffer, timerId);
}

//-----------------------------------------------------------------------------
void CommandList<Vulkan>::ExecuteCommands( const CommandList<Vulkan>& secondaryCommands )
//-----------------------------------------------------------------------------
{
    vkCmdExecuteCommands( m_VkCommandBuffer, 1, &secondaryCommands.m_VkCommandBuffer );
}

//-----------------------------------------------------------------------------
void CommandList<Vulkan>::ExecuteCommands( VkCommandBuffer vkCommandBuffer )
//-----------------------------------------------------------------------------
{
    vkCmdExecuteCommands( m_VkCommandBuffer, 1, &vkCommandBuffer );
}

//-----------------------------------------------------------------------------
bool CommandList<Vulkan>::End()
//-----------------------------------------------------------------------------
{
    VkResult RetVal;

    if (m_VkCommandBuffer == VK_NULL_HANDLE)
    {
        LOGE("Error! Trying to end command buffer before it has been initialized: %s", m_Name.c_str());
        return false;
    }

    // By ending the command buffer, it is put out of record mode.
    RetVal = vkEndCommandBuffer(m_VkCommandBuffer);
    if (!CheckVkError("vkEndCommandBuffer()", RetVal))
    {
        LOGE("Unable to end command buffer: %s", m_Name.c_str());
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
void CommandList<Vulkan>::QueueSubmit( const std::span<const VkSemaphore> WaitSemaphores, const std::span<const VkPipelineStageFlags> WaitDstStageMasks, const std::span<const VkSemaphore> SignalSemaphores, VkFence CompletionFence ) const
//-----------------------------------------------------------------------------
{
    assert( m_VkCommandBuffer != VK_NULL_HANDLE );
    m_pVulkan->QueueSubmit( { &m_VkCommandBuffer, 1 }, WaitSemaphores, WaitDstStageMasks, SignalSemaphores, m_QueueIndex, CompletionFence );
}

//-----------------------------------------------------------------------------
void CommandList<Vulkan>::QueueSubmit(const std::span<const SemaphoreWait> WaitSemaphores, const std::span<const SemaphoreSignal> SignalSemaphores, VkFence CompletionFence) const
//-----------------------------------------------------------------------------
{
    constexpr uint32_t cMaxSemaphores = 8;
    std::array<VkSemaphoreSubmitInfoKHR, cMaxSemaphores> SemaphoreInfos;
    assert(WaitSemaphores.size() + SignalSemaphores.size() < SemaphoreInfos.size());
    uint32_t SemaphoreCount = 0, WaitSemaphoreInfosCount = 0;
    for (WaitSemaphoreInfosCount = 0; SemaphoreCount < SemaphoreInfos.size() && SemaphoreCount < WaitSemaphores.size(); ++SemaphoreCount, ++WaitSemaphoreInfosCount)
    {
        SemaphoreInfos[SemaphoreCount] = { .sType = (VkStructureType) VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
                                           .semaphore = WaitSemaphores[WaitSemaphoreInfosCount].semaphore.GetVkSemaphore(),
                                           .value = WaitSemaphores[WaitSemaphoreInfosCount].value,
                                           .stageMask = WaitSemaphores[WaitSemaphoreInfosCount].stageMask,
                                           .deviceIndex = 0 };
    }

    uint32_t SignalSemaphoreInfosCount = 0;
    for (SignalSemaphoreInfosCount = 0; SemaphoreCount < SemaphoreInfos.size() && SignalSemaphoreInfosCount < SignalSemaphores.size(); ++SemaphoreCount, ++SignalSemaphoreInfosCount)
    {
        SemaphoreInfos[SemaphoreCount] = { .sType = (VkStructureType) VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
                                           .semaphore = SignalSemaphores[SignalSemaphoreInfosCount].semaphore.GetVkSemaphore(),
                                           .value = SignalSemaphores[SignalSemaphoreInfosCount].value,
                                           .stageMask = SignalSemaphores[SignalSemaphoreInfosCount].stageMask,
                                           .deviceIndex = 0 };
    }

    std::array<VkCommandBufferSubmitInfo, 1> CommandBufferInfos;
    CommandBufferInfos[0] = {.sType = (VkStructureType) VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR,
                             .commandBuffer = m_VkCommandBuffer };

    VkSubmitInfo2KHR SubmitInfo{ .sType = (VkStructureType) VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,
                                 .waitSemaphoreInfoCount = WaitSemaphoreInfosCount,
                                 .pWaitSemaphoreInfos = SemaphoreInfos.data(),
                                 .commandBufferInfoCount = (uint32_t) CommandBufferInfos.size(),
                                 .pCommandBufferInfos = CommandBufferInfos.data(),
                                 .signalSemaphoreInfoCount = SignalSemaphoreInfosCount,
                                 .pSignalSemaphoreInfos = SemaphoreInfos.data() + WaitSemaphoreInfosCount };

    m_pVulkan->QueueSubmit({ &SubmitInfo,1}, m_QueueIndex, CompletionFence);
}

//-----------------------------------------------------------------------------
void CommandList<Vulkan>::QueueSubmit( const Vulkan::BufferIndexAndFence& CurrentVulkanBuffer, VkSemaphore renderCompleteSemaphore ) const
//-----------------------------------------------------------------------------
{
    assert( m_VkCommandBuffer != VK_NULL_HANDLE );
    QueueSubmit( CurrentVulkanBuffer.semaphore, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, renderCompleteSemaphore, CurrentVulkanBuffer.fence );
}


//-----------------------------------------------------------------------------
void CommandList<Vulkan>::Release()
//-----------------------------------------------------------------------------
{
    CommandListBase::Release();

    if (m_VkCommandBuffer != VK_NULL_HANDLE)
    {
        // Do not need to worry about the device or pool being NULL since we could 
        // not have created the command buffer!
        m_pVulkan->FreeCommandBuffer(m_QueueIndex, m_VkCommandBuffer);
        m_VkCommandBuffer = nullptr;
    }

    m_QueueIndex = 0;
    m_GpuTimerPool = nullptr;
    m_GpuTimerQueries.clear();
    m_pVulkan = nullptr;
}
