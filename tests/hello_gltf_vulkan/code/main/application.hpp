//============================================================================================================
//
//
//                  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

///
/// Sample app demonstrating the loading of a .gltf file (hello world)
///
#pragma once

#include "main/applicationHelperBase.hpp"
#include "vulkan/renderContext.hpp"
#include "vulkan/renderTarget.hpp"
#include "memory/vulkan/uniform.hpp"
#include "vulkan/commandBuffer.hpp"

class MaterialManagerBase;
class ShaderManagerBase;
using CommandListVulkan = CommandList<Vulkan>;

enum RENDER_PASS
{
    RP_SCENE = 0,
    RP_HUD,
    RP_BLIT,
    NUM_RENDER_PASSES
};

// **********************
// Uniform Buffers
// **********************
struct ObjectVertUB
{
    glm::mat4   MVPMatrix;
    glm::mat4   ModelMatrix;
    glm::mat4   ShadowMatrix;
};

struct ObjectFragUB
{
    glm::vec4   Color;
    glm::vec4   NormalHeight;
};

struct LightUB
{
    glm::mat4 ProjectionInv;
    glm::mat4 ViewInv;
    glm::mat4 ViewProjectionInv; // ViewInv * ProjectionInv
    glm::vec4 ProjectionInvW;    // w components of ProjectionInv
    glm::vec4 CameraPos;

    glm::vec4 LightDirection = glm::vec4(-0.022f, 1.0f, -0.17f, 0.0f);
    glm::vec4 LightColor     = glm::vec4(0.8f, 0.8f, 1.0f, 4.0f/*intensity*/);
    glm::vec4 AmbientColor   = glm::vec4(0.32f, 0.28f, 0.1f, 0.0f);
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
    PassSetupInfo                   PassSetup;
    RenderContext<Vulkan>           RenderPass;

    // Recorded objects that are set to be drawn on this pass
    std::vector< CommandListVulkan> ObjectsCmdBuffer;

    // Indicates the completing of the underlying render pass
    VkSemaphore                     PassCompleteSemaphore = VK_NULL_HANDLE;

    //// Render target used by the underlying render pass
    //// note: The blit pass uses the backbuffer directly instead this RT
    //RenderTarget<Vulkan>            RenderTarget;
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
    virtual bool Initialize(uintptr_t windowHandle, uintptr_t hInstance) override;
    virtual void Destroy() override;
    virtual void Render(float fltDiffTime) override;

private:

    // Application - Initialization
    void PreInitializeSetVulkanConfiguration(Vulkan::AppConfiguration& appConfig) override;
    bool InitializeCamera();
    bool LoadShaders();
    bool CreateRenderTargets();
    bool InitUniforms();
    bool InitAllRenderPasses();
    bool InitGui(uintptr_t windowHandle);
    bool LoadMeshObjects();
    bool InitCommandBuffers();
    bool InitLocalSemaphores();
    bool BuildCmdBuffers();

private:

    // Application - Frame
    void UpdateGui();
    bool UpdateUniforms(uint32_t WhichBuffer);

private:

    // Rendertarget images
    TextureVulkan                   m_ColorBuffer;
    TextureVulkan                   m_DepthBuffer;
    TextureVulkan                   m_HudColorBuffer;

    // Dynamic rendering contexts
    RenderContext                   m_SceneRenderContext;
    RenderContext                   m_HudRenderContext;
    RenderContext                   m_BlitRenderContext;

    // UBOs
    UniformArrayT<ObjectVertUB, NUM_VULKAN_BUFFERS> m_ObjectVertUniform;
    ObjectVertUB                    m_ObjectVertUniformData{};
    UniformArrayT<ObjectFragUB, NUM_VULKAN_BUFFERS> m_ObjectFragUniform;
    ObjectFragUB                    m_ObjectFragUniformData{};
    UniformArrayT<LightUB, NUM_VULKAN_BUFFERS> m_LightUniform;
    LightUB                         m_LightUniformData{};

    // Drawables
    std::vector<Drawable>           m_SceneDrawables;
    std::unique_ptr<Drawable>       m_BlitQuadDrawable;

    // Materials

    // Command buffers
    std::vector<CommandListVulkan> m_PrimaryCommandLists;           // one per frame
    std::vector<CommandListVulkan> m_SecondaryObjectCommandLists;   // one per frame

    // Explicit image layout tracking for dynamic rendering.
    VkImageLayout                  m_ColorBufferLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkImageLayout                  m_DepthBufferLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    VkImageLayout                  m_HudColorBufferLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    std::vector<bool>              m_SwapchainImageInitialized;
};
