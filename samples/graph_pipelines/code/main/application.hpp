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
#include <unordered_map>

#include "ml/GraphPipelineTypes.hpp"
#include "ml/DataGraphPipeline.hpp"
#include "ml/GraphDispatch.hpp"
#include "ml/TensorResources.hpp"
#include "ml/QcomDataGraphModel.hpp"

#define NUM_SPOT_LIGHTS 4

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
    glm::vec4   ORM;
};

struct BlitFragUB
{
    bool IsUpscalingActive;
};

struct LightUB
{
    glm::mat4 ProjectionInv;
    glm::mat4 ViewInv;
    glm::mat4 ViewProjectionInv; // ViewInv * ProjectionInv
    glm::vec4 ProjectionInvW;    // w components of ProjectionInv
    glm::vec4 CameraPos;

    glm::vec4 LightDirection = glm::vec4(-0.564000f, 0.826000f, 0.000000f, 0.0f);
    glm::vec4 LightColor = glm::vec4(1.000000f, 1.000000f, 1.000000f, 1.000000);

    glm::vec4 SpotLights_pos[NUM_SPOT_LIGHTS];
    glm::vec4 SpotLights_dir[NUM_SPOT_LIGHTS];
    glm::vec4 SpotLights_color[NUM_SPOT_LIGHTS];

    glm::vec4 AmbientColor = glm::vec4(0.340000f, 0.340000f, 0.340000f, 0.0f);

    int Width;
    int Height;
};

struct GraphPipelineInstance
{
    VkPipelineLayout      pipelineLayout      = VK_NULL_HANDLE;

    VkPipeline                    graphPipeline = VK_NULL_HANDLE;
    VkPipelineCache               pipelineCache = VK_NULL_HANDLE;
    VkDataGraphPipelineSessionARM graphSession  = VK_NULL_HANDLE;
    std::vector<VkDeviceMemory>   sessionMemory;
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
    PassSetupInfo RenderPassSetup;
    std::vector<RenderContext<Vulkan>> RenderContext;  // context per framebuffer (some passes might all point to the same framebuffers)

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
    struct ObjectMaterialParameters
    {
        UniformT<ObjectFragUB>  objectFragUniform;
        ObjectFragUB            objectFragUniformData;

        std::size_t GetHash() const
        {
            auto hash_combine = [](std::size_t seed, const float& v) -> std::size_t
            {
                std::hash<float> hasher;
                seed ^= hasher(v) + 0x9e3228b9 + (seed << 6) + (seed >> 2);
                return seed;
            };

            std::size_t result = 0;
            result = hash_combine(result, objectFragUniformData.Color.x);
            result = hash_combine(result, objectFragUniformData.Color.y);
            result = hash_combine(result, objectFragUniformData.Color.z);
            result = hash_combine(result, objectFragUniformData.Color.w);
            result = hash_combine(result, objectFragUniformData.ORM.r);
            result = hash_combine(result, objectFragUniformData.ORM.g);
            result = hash_combine(result, objectFragUniformData.ORM.b);
            result = hash_combine(result, objectFragUniformData.ORM.a);

            return result;
        };
    };

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
    bool InitializeLights();
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

    bool CreateTensors();
    bool CreateGraphPipeline();

private:

    // Application - Frame
    void BeginRenderPass(uint32_t WhichBuffer, RENDER_PASS WhichPass, uint32_t WhichSwapchainImage, bool beginCmdBuffer = true);
    void AddPassCommandBuffer(uint32_t WhichBuffer, RENDER_PASS WhichPass);
    void EndRenderPass(uint32_t WhichBuffer, RENDER_PASS WhichPass);
    void SubmitRenderPass(uint32_t WhichBuffer, RENDER_PASS WhichPass, const std::span<const VkSemaphore> WaitSemaphores, const std::span<const VkPipelineStageFlags> WaitDstStageMasks, std::span<VkSemaphore> SignalSemaphores, VkFence CompletionFence = (VkFence)nullptr);
    void UpdateGui();
    bool UpdateUniforms(uint32_t WhichBuffer);

    void CopyImageToTensor(
        CommandListVulkan&         cmdList,
        const TextureVulkan&       srcImage,
        VkImageLayout              currentLayout,
        const Ml::GraphPipelineTensor& tensorBinding);

    void CopyTensorToImage(
        CommandListVulkan&         cmdList,
        const TextureVulkan&       dstImage,
        VkImageLayout              currentLayout,
        const Ml::GraphPipelineTensor& tensorBinding);

    void CopyImageToImageBlit(
        CommandListVulkan& cmdList,
        const TextureVulkan& srcImage,
        VkImageLayout        srcLayout,
        const TextureVulkan& dstImage,
        VkImageLayout        dstFinalLayout);

private:

    // Render passes
    std::array< PassData, NUM_RENDER_PASSES> m_RenderPassData;

    // UBOs
    UniformT<ObjectVertUB>                                    m_ObjectVertUniform;
    ObjectVertUB                                              m_ObjectVertUniformData;
    UniformT<LightUB>                                         m_LightUniform;
    LightUB                                                   m_LightUniformData;
    UniformT<BlitFragUB>                                      m_BlitFragUniform;
    BlitFragUB                                                m_BlitFragUniformData;
    std::unordered_map<std::size_t, ObjectMaterialParameters> m_ObjectFragUniforms;

    // Drawables
    std::vector<Drawable> m_SceneDrawables;
    std::unique_ptr<Drawable> m_BlitQuadDrawable;

    // Shaders
    std::unique_ptr<ShaderManager> m_ShaderManager;

    // Materials
    std::unique_ptr<MaterialManager> m_MaterialManager;

    // Upscaling
    bool       m_ShouldUpscale = false; // When m_IsGraphPipelinesSupported is true, controls if we run the model or just blit
    glm::ivec2 m_RenderResolution;
    glm::ivec2 m_UpscaledResolution;

    // ML types
    Ml::DataGraphPipeline  m_data_graph_pipeline;
    Ml::GraphDispatch      m_data_graph_dispatch;
    Ml::TensorResources    m_tensor_resources;
    Ml::QcomDataGraphModel m_QCOM_data_graph_model;

    // Graph Pipelines
    bool                            m_IsGraphPipelinesSupported = false; // Enables/disable the whole graph pipeline functionality
    Ml::GraphPipelineTensor         m_InputTensor;
    Ml::GraphPipelineTensor         m_OutputTensor;
    GraphPipelineInstance           m_GraphPipelineInstance;
    std::vector< CommandListVulkan> m_GraphPipelineCommandLists; // Cmd buffer allocated from the Data Graph queue
    VkSemaphore                     m_GraphPipelinePassCompleteSemaphore = VK_NULL_HANDLE;
    TextureVulkan                   m_UpscaledImageResult; // Receives the model output, or scene color as a blit when upscaling is disabled

    // Hardcoded QNN constants -> These eventually will be infered from the binary model file
    uint32_t m_QNNInputPortBinding  = 0;
    uint32_t m_QNNOutputPortBinding = 1;
    uint32_t m_QNNMaxPortIndex      = 2;
};
