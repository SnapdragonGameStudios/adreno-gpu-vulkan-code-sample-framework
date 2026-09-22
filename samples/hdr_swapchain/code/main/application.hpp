// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2020 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

///
/// @file hdrSwapchain.hpp
/// @brief Application showing how to change between different swap-chain (HDR) formats.
/// Provides a user selectable dropdown populated with populated with available formats (for the current display device).
/// Also demonstrates the use of VK_QCOM_render_pass_transform Vulkan extension to rotate the buffers to the device native orientation.
/// 

#include "main/applicationHelperBase.hpp"
#include "materials.hpp"
#include "vulkan/commandBuffer.hpp"
#include "vulkan/MeshObject.h"
#include "vulkan/vulkan_support.hpp"
#include "vulkan/TextureFuncts.h"
#include "camera/camera.hpp"
#include "memory/vulkan/uniform.hpp"
#include "shadow/shadowVulkan.hpp"
#include "system/glm_common.hpp"

#include <memory>
#include <map>
#include <string>
#include <span>

// Forward declarations
class ShaderManagerBase;
template<typename T_GFXAPI> class MaterialManager;
class MaterialPassBase;
class MaterialBase;
class ComputableBase;
class Gui;

enum RENDER_PASS
{
    RP_GBUFFER = 0,
    RP_SHADOW,
    RP_LIGHT,
    RP_HUD,
    RP_BLIT,
    NUM_RENDER_PASSES
};

// Class
class Application : public ApplicationHelperBase
{
public:
    Application();
    ~Application() override;

    // Override FrameworkApplicationBase
    int     PreInitializeSelectSurfaceFormat(std::span<const SurfaceFormat>) override;
    bool    Initialize(uintptr_t windowHandle, uintptr_t instanceHandle) override;
    void    Destroy() override;
    void    Render(float fltDiffTime) override;

    // Application
    bool    LoadMeshObjects();
    bool    LoadTextures();
    bool    CreateRenderTargets();
    bool    InitShadowMap();
    bool    InitLighting();
    bool    LoadShaders();
    bool    InitUniforms();
    bool    UpdateUniforms(uint32_t WhichBuffer);
    bool    InitMaterials();
    bool    InitCommandBuffers();
    bool    InitAllRenderPasses();
    bool    InitDrawables();
    bool    InitHdr();
    bool    InitGui(uintptr_t windowHandle);

    bool    BuildCmdBuffers();
    bool    InitLocalSemaphores();

    const char* GetPassName(uint32_t WhichPass);

    void    UpdateGui();
    void    UpdateCamera(float ElapsedTime);
    void    UpdateLighting(float ElapsedTime);
    void    UpdateShadowMap(float ElapsedTime);

    bool    ChangeSurfaceFormat(SurfaceFormat newSurfaceFormat);

    void    BeginRenderPass(RENDER_PASS WhichPass);
    void    AddPassCommandBuffers(RENDER_PASS WhichPass);
    void    AddPassCommandBuffers(RENDER_PASS WhichPass, std::span<VkCommandBuffer> SubCommandBuffers);
    void    AddPassCommandBuffer(RENDER_PASS WhichPass, CommandListVulkan& SubCommandBuffer)
    {
        AddPassCommandBuffers(WhichPass, {&SubCommandBuffer.m_VkCommandBuffer, 1});
    }
    void    EndRenderPass(RENDER_PASS WhichPass);
    void    SubmitRenderPass(RENDER_PASS WhichPass, const std::span<const VkSemaphore> WaitSemaphores, const std::span<const VkPipelineStageFlags> WaitDstStageMasks, const std::span<const VkSemaphore> SignalSemaphores, VkFence CompletionFence = VK_NULL_HANDLE);

private:
    // Private variable to hold which "swapchain" we are on
    uint32_t            m_CurrentVulkanBuffer;

    // Metrics
    uint32_t            m_TotalDrawCalls;
    uint32_t            m_TotalTriangles;

    // Requested surface format (for changing formats)
    SurfaceFormat       m_RequestedSurfaceFormat;
    // sRGB output (done in blit shader) on/off
    bool                m_bEncodeSRGB;

    // Drawables
    std::vector<Drawable>       m_SceneObject;
    std::unique_ptr<Drawable>   m_LightDrawable;
    std::unique_ptr<Drawable>   m_BlitDrawable;
    // Computables
    std::unique_ptr<ComputableBase> m_VsmComputable;
    std::unique_ptr<ComputableBase> m_ComputableTest;
    std::unique_ptr<ComputableBase> m_BloomComputable;
    std::unique_ptr<ComputableBase> m_NNAOComputable;

    // Textures
    const TextureVulkan*           m_TexWhite;
    const TextureVulkan*           m_DefaultNormal;

    // Light Stuff
    glm::vec4               m_LightColor;

    // Shadow Map stuff
    static const int        cNumShadows = 1;
    std::array<ShadowVulkan, cNumShadows> m_Shadows;

    // **********************
    // The Object
    // **********************
    float                   m_ObjectScale;
    glm::vec3               m_ObjectWorldPos;

    UniformT<ObjectVertUB>  m_ObjectVertUniform[NUM_RENDER_PASSES][NUM_VULKAN_BUFFERS];
    ObjectVertUB            m_ObjectVertUniformData;
    UniformT<ObjectFragUB>  m_ObjectFragUniform[NUM_RENDER_PASSES][NUM_VULKAN_BUFFERS];
    ObjectFragUB            m_ObjectFragUniformData;

    CommandListVulkan    m_ObjectCmdBuffer[NUM_VULKAN_BUFFERS][NUM_RENDER_PASSES];

    // **********************
    // The Skybox
    // **********************
    float                   m_SkyboxScale;
    glm::vec3               m_SkyboxWorldPos;

    UniformVulkan           m_SkyboxVertUniform[NUM_RENDER_PASSES][NUM_VULKAN_BUFFERS];
    SkyboxVertUB            m_SkyboxVertUniformData;

    // **********************
    // Deferred Lighting
    // **********************
    CommandListVulkan    m_LightCmdBuffer[NUM_VULKAN_BUFFERS];
    struct LightFragCtrl {
        static const int cNUM_LIGHTS = 8;
        glm::mat4 ProjectionInv;
        glm::mat4 ViewInv;
        glm::mat4 ViewProjectionInv; // ViewInv * ProjectionInv
        glm::mat4 WorldToShadow;
        glm::vec4 ProjectionInvW; // W components of ProjectionInv
        glm::vec4 CameraPos;

        glm::vec4 LightPositions[cNUM_LIGHTS];  // w is intensity (inv square falloff)
        glm::vec4 LightColors[cNUM_LIGHTS];

        glm::vec4 AmbientColor;

        float AmbientOcclusionScale;
        int Width;
        int Height;

    } m_LightFragUniformData;
    UniformArrayT<LightFragCtrl, NUM_VULKAN_BUFFERS> m_LightFragUniform;

    // **********************
    // Post/Blit
    // **********************
    struct BlitFragCtrl {
        float Bloom = 2.0f;
        float Diffuse = 1.0f;   // 0 to 2 range (dark to white)
        int sRGB = 0;           // 1 - apply srgb conversion in output blit shader, 0 passthrough color
    } m_BlitFragUniformData;
    UniformArrayT<BlitFragCtrl, NUM_VULKAN_BUFFERS> m_BlitFragUniform;
    CommandListVulkan    m_BlitCmdBuffer[NUM_VULKAN_BUFFERS];

    // **********************
    // Compute
    // **********************
    struct ComputeCtrl {
        int width, height;
    } m_ComputeCtrl;
    struct NNAOCtrl {
        // Output texture size
        int Width;
        int Height;
        int _align[2];
        // X: Screen Width
        // Y: Screen Height
        // Z: One Width Pixel
        // W: One Height Pixel
        glm::vec4   ScreenSize;
        // Camera projection
        glm::mat4   CameraProjection;
        // Camera inverse projection
        glm::mat4   CameraInvProjection;
    } m_NNAOCtrl;
    UniformT<ComputeCtrl>   m_ComputeCtrlUniform;
    UniformT<ComputeCtrl>   m_ComputeCtrlUniformHalf;
    UniformT<ComputeCtrl>   m_ComputeCtrlUniformQuarter;
    UniformArrayT<NNAOCtrl, NUM_VULKAN_BUFFERS> m_NNAOCtrlUniform;
    TextureVulkan           m_VsmTarget;
    TextureVulkan           m_ComputeIntermediateHalfTarget;
    TextureVulkan           m_ComputeIntermediateHalf2Target;
    TextureVulkan           m_ComputeIntermediateQuarterTarget;
    TextureVulkan           m_ComputeIntermediateQuarter2Target;
    TextureVulkan           m_ComputeRenderTarget;
    TextureVulkan           m_BloomRenderTarget;
    TextureVulkan           m_NNAORenderTarget;
    TextureVulkan           m_NNAOTempTarget;

    CommandListVulkan    m_VsmComputeCmdBuffer[NUM_VULKAN_BUFFERS];      // Command buffer to run VSM compute commands on the (regular) graphics queue 
    CommandListVulkan    m_VsmAsyncComputeCmdBuffer[NUM_VULKAN_BUFFERS]; // Command buffer for VSM on Async Compute queue (needs separate command buffer since command pool is tied to the destination queue)
    CommandListVulkan    m_DiffractionDownsampleCmdBuffer[NUM_VULKAN_BUFFERS];
    CommandListVulkan    m_BloomComputeCmdBuffer[NUM_VULKAN_BUFFERS];
    CommandListVulkan    m_BloomAsyncComputeCmdBuffer[NUM_VULKAN_BUFFERS];
    CommandListVulkan    m_NNAOComputeCmdBuffer[NUM_VULKAN_BUFFERS];
    CommandListVulkan    m_NNAOAsyncComputeCmdBuffer[NUM_VULKAN_BUFFERS];


    // **********************
    // Pass Stuff
    // **********************
    struct PassSetup {

        std::vector<TextureFormat> ColorFormats;
        TextureFormat           DepthFormat;
        RenderPassInputUsage    ColorInputUsage;
        bool                    ClearDepthRenderPass;
        RenderPassOutputUsage   ColorOutputUsage;
        RenderPassOutputUsage   DepthOutputUsage;
        glm::vec4               ClearColor;
        uint32_t                TargetWidth;
        uint32_t                TargetHeight;

    } m_PassSetup[NUM_RENDER_PASSES];

    CommandListVulkan    m_PassCmdBuffer[NUM_VULKAN_BUFFERS][NUM_RENDER_PASSES];
    std::array<VkSemaphore, NUM_RENDER_PASSES>  m_PassCompleteSemaphore;
    VkSemaphore             m_VsmAsyncComputeCanStartSemaphore;
    VkSemaphore             m_VsmAsyncComputeCompleteSemaphore;
    VkSemaphore             m_BloomAsyncComputeCanStartSemaphore;
    VkSemaphore             m_BloomAsyncComputeCompleteSemaphore;
    VkSemaphore             m_NNAOAsyncComputeCanStartSemaphore;
    VkSemaphore             m_NNAOAsyncComputeCompleteSemaphore;

    // Don't actually need a render pass for each Vulkan Buffer, just one per pass
    // These are NOT the same vkrenderpass's as in the RenderTargetArray's but should be compatible with them (are allowed dfferent load/clear parameters on the atachments)
    std::array<RenderPass, NUM_RENDER_PASSES>   m_RenderPass;

    // Render targets for each pass
    RenderTarget  m_GBufferRT;
    RenderTarget  m_MainRT;
    RenderTarget  m_HudRT;
};
