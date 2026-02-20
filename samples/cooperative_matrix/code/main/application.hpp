//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include "main/applicationHelperBase.hpp"
#include "memory/vulkan/uniform.hpp"
#include "vulkan/commandBuffer.hpp"
#include "cooperative_matrix_tester.hpp"
#include "vulkan/renderPass.hpp"
#include <unordered_map>

enum RENDER_PASS
{
    RP_HUD = 0,
    RP_BLIT,
    NUM_RENDER_PASSES
};

// **********************
// Render Pass
// **********************
struct PassSetupInfo
{
    RenderPassInputUsage    ColorInputUsage;
    bool                    ClearDepthRenderPass;
    RenderPassOutputUsage   ColorOutputUsage;
    RenderPassOutputUsage   DepthOutputUsage;
    glm::vec4               ClearColor;
};

struct PassData
{
    // Pass internal data
    PassSetupInfo PassSetup;
    std::vector<RenderContext<Vulkan>>  RenderContext;  // context per framebuffer (some passes might all point to the same framebuffers)

    // Recorded objects that are set to be drawn on this pass
    std::vector< CommandListVulkan> ObjectsCmdBuffer;

    // Command buffer used to dispatch the render pass
    std::vector< CommandListVulkan> PassCmdBuffer;

    // Indicates the completing of the underlying render pass
    VkSemaphore PassCompleteSemaphore = VK_NULL_HANDLE;

    // Render targed used by the underlying render pass
    // note: The blit pass uses the backbuffer directly instead this RT
    RenderTarget<Vulkan> RenderTarget;
};

// **********************
// Application
// **********************
class Application : public ApplicationHelperBase
{
public:
    Application();
    ~Application() override;

    // ApplicationHelperBase
    virtual void PreInitializeSetVulkanConfiguration(Vulkan::AppConfiguration& config) override;
    virtual bool Initialize(uintptr_t windowHandle, uintptr_t hInstance) override;
    virtual void Destroy() override;
    virtual void Render(float fltDiffTime) override;

private:

    // Application - Initialization
    bool InitializeCamera();
    bool LoadShaders();
    bool CreateRenderTargets();
    bool InitAllRenderPasses();
    bool InitGui(uintptr_t windowHandle);
    bool LoadMeshObjects();
    bool InitCommandBuffers();
    bool InitLocalSemaphores();
    bool BuildCmdBuffers();

private:

    // Application - Frame
    void BeginRenderPass(uint32_t WhichBuffer, RENDER_PASS WhichPass, uint32_t WhichSwapchainImage);
    void AddPassCommandBuffer(uint32_t WhichBuffer, RENDER_PASS WhichPass);
    void EndRenderPass(uint32_t WhichBuffer, RENDER_PASS WhichPass);
    void SubmitRenderPass(uint32_t WhichBuffer, RENDER_PASS WhichPass, const std::span<const VkSemaphore> WaitSemaphores, const std::span<const VkPipelineStageFlags> WaitDstStageMasks, std::span<VkSemaphore> SignalSemaphores, VkFence CompletionFence = (VkFence)nullptr);
    void UpdateGui();

private:

    // Render passes
    std::array< PassData, NUM_RENDER_PASSES> m_RenderPassData;

    // Drawables
    std::unique_ptr<Drawable> m_BlitQuadDrawable;

    // Shaders
    std::unique_ptr<ShaderManager> m_ShaderManager;

    // Materials
    std::unique_ptr<MaterialManager> m_MaterialManager;

    std::unique_ptr<CooperativeMatrixRunner> m_cooperative_matrix_runner;
};
