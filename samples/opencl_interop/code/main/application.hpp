//============================================================================================================
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#pragma once

#include "main/applicationHelperBase.hpp"
#include "vulkan/commandBuffer.hpp"
#include "vulkan/renderPass.hpp"
#include "material/vulkan/drawable.hpp"
#include "material/vulkan/materialManager.hpp"
#include "material/vulkan/shaderManager.hpp"
#include "interop_common.hpp"

// Forward declarations for interop contexts
namespace InteropBuffer       { class Context; }  // Mode 1: buffer interop
namespace InteropImageLinear  { class Context; }  // Mode 2: linear image interop
namespace InteropImageOptimal { class Context; }  // Mode 3: optimal image interop

enum RENDER_PASS
{
    RP_SCENE = 0,
    RP_HUD,
    RP_BLIT,
    NUM_RENDER_PASSES
};

// ======================
// Render Pass
// ======================
struct RenderPassSetupInfo
{
    RenderPassInputUsage    ColorInputUsage;
    bool                    ClearDepthRenderPass;
    RenderPassOutputUsage   ColorOutputUsage;
    RenderPassOutputUsage   DepthOutputUsage;
    glm::vec4               ClearColor;
};

struct RenderPassData
{
    // Pass internal data
    RenderPassSetupInfo                 RenderPassSetup;
    std::vector<RenderContext<Vulkan>>  RenderContext;

    // Render target used by the underlying render pass
    // note: The blit pass uses the backbuffer directly instead of this RT
    RenderTarget<Vulkan>                RenderTarget;
};

// ======================
// Application
// ======================
class Application : public ApplicationHelperBase
{
public:
    Application();
    ~Application() override;

    // ApplicationHelperBase
    virtual void PreInitializeSetVulkanConfiguration(Vulkan::AppConfiguration&) override;
    virtual bool Initialize(uintptr_t windowHandle, uintptr_t hInstance) override;
    virtual void Destroy() override;
    virtual void Render(float fltDiffTime) override;

private:

    // Application - Initialization
    bool InitializeCamera();
    bool LoadShaders();
    bool CreateRenderTargets();
    bool InitUniforms();
    bool InitAllRenderPasses();
    bool InitInteropContext();
    bool InitGui(uintptr_t windowHandle);
    bool LoadMeshObjects();
    bool InitCommandBuffers();
    bool InitLocalSemaphores();

private:

    // Application - Frame
    void UpdateGui();
    bool UpdateUniforms(uint32_t WhichBuffer);

private:

    // Shared OpenCL execution state (device + context + queue)
    CLState m_cl_state;

    // Interop contexts (one per mode)
    std::unique_ptr<InteropBuffer::Context>       m_interop_buffer_context;
    std::unique_ptr<InteropImageLinear::Context>  m_interop_linear_context;
    std::unique_ptr<InteropImageOptimal::Context> m_interop_optimal_context;

    // Render passes
    std::array<RenderPassData, NUM_RENDER_PASSES>     m_RenderPassData;

    // Command lists for Vulkan work before/after OpenCL interop
    std::array<CommandListVulkan, NUM_VULKAN_BUFFERS> m_SceneCommandList;
    std::array<CommandListVulkan, NUM_VULKAN_BUFFERS> m_BlitCommandList;

    // Semaphores for Vulkan submit boundaries
    VkSemaphore m_SceneCompleteSemaphore = VK_NULL_HANDLE;
    VkSemaphore m_BlitCompleteSemaphore  = VK_NULL_HANDLE;

    // Drawables
    std::unique_ptr<Drawable> m_SceneQuadDrawable;
    std::unique_ptr<Drawable> m_BlitQuadDrawable;

    // Shaders
    std::unique_ptr<ShaderManager>  m_ShaderManager;

    // Materials
    std::unique_ptr<MaterialManager> m_MaterialManager;
};