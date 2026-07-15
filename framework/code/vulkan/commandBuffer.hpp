//============================================================================================================
//
//
//                  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include <string>

// Need the vulkan wrapper
#include "vulkan.hpp"
#include "graphicsApi/commandList.hpp"

// Forward declarations
class Vulkan;
class RenderPassClearData;
class TimerPoolBase;
class SemaphoreWait;
class SemaphoreSignal;

template<class T_GFXAPI> class CommandList;
using CommandListVulkan = CommandList<Vulkan>;
template<class T_GFXAPI> class Framebuffer;
template<class T_GFXAPI> class RenderTarget;
template<class T_GFXAPI> class RenderContext;


///
/// Wrapper around VkCommandBuffer.
/// Simplifies creation, use and destruction of VkCommandBuffer.
/// 
template<>
class CommandList<Vulkan> final : public CommandListBase
{
    CommandList(const CommandList<Vulkan>&) = delete;
    CommandList<Vulkan>& operator=(const CommandList<Vulkan>&) = delete;
    // Functions
public:
    CommandList() noexcept;
    ~CommandList();
    CommandList(CommandList<Vulkan>&&) noexcept;
    CommandList<Vulkan>& operator=(CommandList<Vulkan>&&) noexcept;
    
    // Curtesy
    operator VkCommandBuffer() const { return m_VkCommandBuffer; }

    bool Initialize(Vulkan* pVulkan, const std::string& Name = {}, CommandListBase::Type CmdBuffType = Type::Primary, uint32_t QueueIndex = Vulkan::eGraphicsQueue, TimerPoolBase* pTimerPool = nullptr);
    //bool Initialize(Vulkan* pVulkan, const std::string& Name, VkCommandBufferLevel CmdBuffLevel, std::same_as<bool> auto QueueIndex, TimerPoolBase* pTimerPool = nullptr) = delete;

    // Begin primary command buffer
    bool Begin(VkCommandBufferUsageFlags CmdBuffUsage = 0);
    // Begin command buffer
    bool Begin(const VkCommandBufferBeginInfo& CmdBeginInfo);

    // Begin secondary command buffer with inheritance
    bool Begin(VkFramebuffer FrameBuffer, VkRenderPass RenderPass, bool IsSwapChainRenderPass = false, uint32_t SubPass = 0, VkCommandBufferUsageFlags CmdBuffUsage = 0);
    // Begin secondary command buffer with inheritance
    bool Begin(VkFramebuffer FrameBuffer, const RenderPass<Vulkan>& RenderPass, bool IsSwapChainRenderPass = false, uint32_t SubPass = 0, VkCommandBufferUsageFlags CmdBuffUsage = 0);
    // Begin secondary command buffer with inheritance
    bool Begin(const Framebuffer<Vulkan>& FrameBuffer, const RenderPass<Vulkan>& RenderPass, bool IsSwapChainRenderPass = false, uint32_t SubPass = 0, VkCommandBufferUsageFlags CmdBuffUsage = 0);
    // Begin secondary command buffer (dynamic rendering)
    bool Begin(const VkCommandBufferInheritanceRenderingInfo& DynamicRenderingInheritanceInfo, VkCommandBufferUsageFlags CmdBuffUsage);
    // Begin secondary command buffer with inheritance
    bool Begin(const RenderContext<Vulkan>& RenderContext, VkCommandBufferUsageFlags CmdBuffUsage = 0 );

    bool BeginRenderPass(VkRect2D RenderExtent, float MinDepth, float MaxDepth, const std::span<const VkClearColorValue> ClearColors, uint32_t NumColorBuffers, bool HasDepth, VkRenderPass RenderPass, bool IsSwapChainRenderPass, VkFramebuffer FrameBuffer, VkSubpassContents SubContents = VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
    bool BeginRenderPass(VkRect2D RenderExtent, float MinDepth, float MaxDepth, const std::span<const VkClearColorValue> ClearColors, uint32_t NumColorBuffers, bool HasDepth, const RenderPass<Vulkan>& RenderPass, bool IsSwapChainRenderPass, VkFramebuffer FrameBuffer, VkSubpassContents SubContents = VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
    bool BeginRenderPass(const RenderTarget<Vulkan>& renderTarget, VkRenderPass RenderPass, VkSubpassContents SubContents);
    bool BeginRenderPass(const RenderTarget<Vulkan>& renderTarget, const RenderPass<Vulkan>& RenderPass, VkSubpassContents SubContents);
    bool BeginRenderPass(const Framebuffer<Vulkan>& Framebuffer, const RenderPass<Vulkan>& RenderPass, const RenderPassClearData& RenderPassClearData, VkSubpassContents SubContents);
    bool BeginRenderPass(const VkRenderPassBeginInfo&, VkSubpassContents SubContents);
    bool BeginSwapchainRenderPass(uint32_t SwapchainImageIndex, const RenderPass<Vulkan>& RenderPass, const RenderPassClearData& RenderPassClearData, VkSubpassContents SubContents);
    bool BeginRenderPass(const RenderContext<Vulkan>& renderContext, VkSubpassContents SubContents = VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
    void BeginRenderPass(const VkRenderingInfo& renderingInfo); // dynamic rendering
    void EndRendering(); // dynamic rendering
    bool NextSubpass(VkSubpassContents SubContents = VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
    bool EndRenderPass();

    int StartGpuTimer(const std::string_view& timerName);
    void StopGpuTimer(int timerId);

    /// Execute contents of the supplied command buffer (must be a secondary command buffer)
    void ExecuteCommands( const CommandList<Vulkan>& secondaryCommands );
    void ExecuteCommands( VkCommandBuffer );

    bool End();

    /// @brief Get command list ready for re-recording
    bool Reset();

    /// @brief Call vkQueueSubmit for this command buffer.
    /// Will submit to the compute or graphics queue (depending on m_QueueIndex).
    /// @param WaitSemaphores semaphores to wait upon before the gpu can execute this command buffer
    /// @param WaitDstStageMasks pipeline stages that the wait occurs on.  MUST have the same number of entries as the WaitSemaphores parameter.
    /// @param SignalSemaphores semaphores to signal when this commandlist finishes
    /// @param CompletionFence fence to set when the commandlist finishes (for CPU synchronization)
    void QueueSubmit(const std::span<const VkSemaphore> WaitSemaphores, const std::span<const VkPipelineStageFlags> WaitDstStageMasks, const std::span<const VkSemaphore> SignalSemaphores, VkFence CompletionFence = VK_NULL_HANDLE) const;
    /// @brief Simplified QueueSubmit for cases where we are only waiting and signalling one semaphore.
    void QueueSubmit( VkSemaphore WaitSemaphore, VkPipelineStageFlags WaitDstStageMask, VkSemaphore SignalSemaphore, VkFence CompletionFence = VK_NULL_HANDLE ) const
    {
        typedef std::span<VkSemaphore> tSemSpan;
        tSemSpan WaitSemaphores = (WaitSemaphore != VK_NULL_HANDLE) ? tSemSpan{ &WaitSemaphore, 1 } : tSemSpan{};
        tSemSpan SignalSemaphores = (SignalSemaphore != VK_NULL_HANDLE) ? tSemSpan{ &SignalSemaphore, 1 } : tSemSpan{};
        QueueSubmit( WaitSemaphores, std::span<const VkPipelineStageFlags>( &WaitDstStageMask, WaitSemaphores.size() ), SignalSemaphores, CompletionFence );
    }
    template<typename T_SEMA_CONTAINER, typename T_STAGEFLAGS_CONTAINER>
    void QueueSubmitT(T_SEMA_CONTAINER WaitSemaphores, T_STAGEFLAGS_CONTAINER WaitDstStageMasks, T_SEMA_CONTAINER SignalSemaphores, VkFence CompletionFence = VK_NULL_HANDLE)
    {
        QueueSubmit( {WaitSemaphores}, {WaitDstStageMasks}, {SignalSemaphores}, CompletionFence );
    }
    /// Call vkQueueSubmit for this command buffer.
    /// Will submit to the compute or graphics queue (depending on m_QueueIndex).
    /// This variant uses the Wrap_VkSemaphore class and its SemaphoreWait / SemaphoreSignal helpers.
    /// @param WaitSemaphores semaphores to wait upon before the gpu can execute this command buffer
    /// @param SignalSemaphores semaphores to signal when this commandlist finishes
    /// @param CompletionFence fence to set when the commandlist finishes (for CPU synchronization)
    void QueueSubmit(const std::span<const SemaphoreWait> WaitSemaphores, const std::span<const SemaphoreSignal> SignalSemaphores, VkFence CompletionFence = VK_NULL_HANDLE) const;

    /// @brief Call vkQueueSubmit and vkQueuePresentKHR for the given 'frame'
    /// Call when this is the last (or only) commandlist to be submitted for the current frame.
    /// @param CurrentVulkanBuffer
    void QueueSubmit( const Vulkan::BufferIndexAndFence& CurrentVulkanBuffer, VkSemaphore renderCompleteSemaphore ) const;

    /// @brief Release the Vulkan resources used by this wrapper and cleanup.
    void Release() override;

private:

    // Attributes
public:
    uint32_t            m_QueueIndex = 0;
    VkCommandBuffer     m_VkCommandBuffer = VK_NULL_HANDLE;
    TimerPoolBase*      m_GpuTimerPool = nullptr;
    std::vector<int>    m_GpuTimerQueries;  // Vector of the timerId's that were added to this command buffer, needed so we can mark them as no longer 'inflight'

private:
    Vulkan*             m_pVulkan = nullptr;
};
