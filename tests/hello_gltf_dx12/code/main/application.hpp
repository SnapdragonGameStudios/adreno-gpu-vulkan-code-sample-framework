//============================================================================================================
//
//
//                  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

///
/// Sample app demonstrating the loading of a .gltf file (hello world)
///
#pragma once


#define VULKAN (0)
#if VULKAN
#include "main/applicationHelperBase.hpp"
using tGfxApi = Vulkan;
#include "memory/vulkan/uniform.hpp"
#include "vulkan/commandBuffer.hpp"
#include "vulkan/renderPass.hpp"
#include "vulkan/renderTarget.hpp"
#define NUM_SWAPCHAIN_BUFFERS NUM_VULKAN_BUFFERS
#else
#include "main/applicationHelperBaseDx12.hpp"
using tGfxApi = Dx12;
#include "memory/dx12/uniform.hpp"
#include "dx12/commandList.hpp"
#include "dx12/renderPass.hpp"
#include "dx12/renderTarget.hpp"
#endif

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
    glm::vec4 LightColor     = glm::vec4(1.0f, 1.0f, 1.0f, 2.0f);

    glm::vec4 AmbientColor   = glm::vec4(0.3f, 0.3f, 0.3f, 0.0f);

    int Width;
    int Height;
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
    
    RenderPass<tGfxApi> RenderPass;

    // Recorded objects that are set to be drawn on this pass
    std::vector< CommandList<tGfxApi>> ObjectsCmdBuffer;

    // Indicates the completing of the underlying render pass
    //VkSemaphore PassCompleteSemaphore = VK_NULL_HANDLE;

    // Render targed used by the underlying render pass
    // note: The blit pass uses the backbuffer directly instead this RT
    RenderTarget<tGfxApi> RenderTarget;
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
    virtual void PreInitializeSetVulkanConfiguration( AppConfiguration& ) override;
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
    bool InitGui(uintptr_t windowHandle);
    bool LoadMeshObjects();
    bool InitCommandBuffers();
    bool BuildCmdBuffers();

private:

    // Application - Frame
    void UpdateGui();
    bool UpdateUniforms(uint32_t WhichBuffer);

private:

    // Render passes
    std::array< PassData, NUM_RENDER_PASSES> m_RenderPassData;

    // UBOs
    UniformArrayT<ObjectVertUB, NUM_SWAPCHAIN_BUFFERS>  m_ObjectVertUniform;
    ObjectVertUB                m_ObjectVertUniformData{};
    UniformArrayT<ObjectFragUB, NUM_SWAPCHAIN_BUFFERS>  m_ObjectFragUniform;
    ObjectFragUB                m_ObjectFragUniformData{};
    UniformArrayT<LightUB, NUM_SWAPCHAIN_BUFFERS>   m_LightUniform;
    LightUB                     m_LightUniformData{};

    // Drawables
    std::vector<Drawable>       m_SceneDrawables;
    std::unique_ptr<Drawable>   m_BlitQuadDrawable;
    std::unique_ptr<ComputableBase> m_ColorGradingComputable;

    // Per Frame Command lists
    std::array<CommandList, NUM_SWAPCHAIN_BUFFERS> m_CommandLists;
    std::array<CommandList, NUM_SWAPCHAIN_BUFFERS> m_ComputeCommandLists;
};
