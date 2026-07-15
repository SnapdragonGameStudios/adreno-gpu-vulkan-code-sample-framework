//============================================================================================================
//
//
//                  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include "main/applicationHelperBase.hpp"
#include "memory/vulkan/uniform.hpp"
#include "memory/buffer.hpp"
#include "memory/vulkan/bufferObject.hpp"
#include "memory/vulkan/indexBufferObject.hpp"
#include "memory/vulkan/vertexBufferObject.hpp"
#include "memory/vulkan/drawIndirectBufferObject.hpp"
#include "vulkan/commandBuffer.hpp"
#include "vulkan/renderPass.hpp"
#include <unordered_map>

#define MAX_LIGHT_COUNT 2048
#define MAX_CLUSTER_MATRIX_DIMENSION_SIZE 8
#define MAX_CLUSTER_MATRIX_SIZE (MAX_CLUSTER_MATRIX_DIMENSION_SIZE * MAX_CLUSTER_MATRIX_DIMENSION_SIZE)

// Forward declarations
class MaterialManagerBase;
class ShaderManagerBase;
struct TileMemoryHeapExtension;

enum RENDER_PASS
{
    RP_SCENE = 0,
    RP_DEFERRED_LIGHT,  
    RP_HUD,
    RP_BLIT,
    NUM_RENDER_PASSES
};

enum COMPUTE_PASS
{
    CP_LIGHT_CULLING_LIST = 0,
    NUM_COMPUTE_PASSES
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

struct LightUB
{
    glm::vec4 lightPosAndRadius; // xyz = pos | w = radius
    glm::vec4 lightColor;
};

struct alignas(16) SceneInfoUB
{
    glm::mat4 projectionInv;
    glm::mat4 view;
    glm::mat4 viewInv;
    glm::mat4 viewProjection;
    glm::mat4 viewProjectionInv; // ViewInv * ProjectionInv
    glm::vec4 projectionInvW;    // w components of ProjectionInv
    glm::vec4 cameraPos;

    glm::vec4 skyLightDirection = glm::vec4(-0.564000f, 0.826000f, 0.000000f, 0.0f);
    glm::vec4 skyLightColor = glm::vec4(1.000000f, 1.000000f, 1.000000f, 1.000000);

    glm::vec4 ambientColor = glm::vec4(0.340000f, 0.340000f, 0.340000f, 0.0f);

    uint32_t viewportWidth;
    uint32_t viewportHeight;
    uint32_t clusterCountX    = MAX_CLUSTER_MATRIX_DIMENSION_SIZE; // How many clusters on the x coordinate, inside each bin
    uint32_t clusterCountY    = MAX_CLUSTER_MATRIX_DIMENSION_SIZE; // How many clusters on the y coordinate, inside each bin
    int32_t lightCount        = MAX_LIGHT_COUNT / 4;
    int32_t debugShaders      = false;
    int32_t ignoreLightTiles  = false;
    int32_t tileMemoryEnabled = false;
};

struct BasePassData
{
};

struct RenderPassData : public BasePassData
{
    ::RenderContext<Vulkan> renderContext;
};

struct ComputePassData : public BasePassData
{
    std::unique_ptr< Computable<Vulkan>> passComputable;
};

struct RenderTargetGroup
{
    std::vector< Texture<Vulkan>> color;
    std::vector< Texture<Vulkan>> colorTileMemory;
    Texture<Vulkan> depth;
    Texture<Vulkan> depthTileMemory;
};

struct PassAttachmentInfo
{
    PassAttachmentInfo() = default;
    PassAttachmentInfo(
        const RenderTargetGroup& inRenderTarget,
        RenderPassInputUsage inInputUsage,
        RenderPassOutputUsage inOutputUsage,
        std::vector<bool> inRenderTargetLocalReadColorAttachments = {},
        bool inRenderTargetLocalReadDepthStencilAttachment = false)
        : renderTarget(&inRenderTarget)
        , colorInputUsage(inInputUsage)
        , depthInputUsage(inInputUsage)
        , colorOutputUsage(inOutputUsage)
        , depthOutputUsage(inOutputUsage)
        , renderTargetLocalReadColorAttachments(std::move(inRenderTargetLocalReadColorAttachments))
        , renderTargetLocalReadDepthStencilAttachment(inRenderTargetLocalReadDepthStencilAttachment)
    {
    }
    
    PassAttachmentInfo(
        const RenderTargetGroup& inRenderTarget,
        RenderPassInputUsage inColorInputUsage,
        RenderPassOutputUsage inColorOutputUsage,
        RenderPassInputUsage inDepthInputUsage,
        RenderPassOutputUsage inDepthOutputUsage, 
        std::vector<bool> inRenderTargetLocalReadColorAttachments = {},
        bool inRenderTargetLocalReadDepthStencilAttachment = false)
        : renderTarget(&inRenderTarget)
        , colorInputUsage(inColorInputUsage)
        , colorOutputUsage(inColorOutputUsage)
        , depthInputUsage(inDepthInputUsage)
        , depthOutputUsage(inDepthOutputUsage)
        , renderTargetLocalReadColorAttachments(std::move(inRenderTargetLocalReadColorAttachments))
        , renderTargetLocalReadDepthStencilAttachment(inRenderTargetLocalReadDepthStencilAttachment)
    {
    }

    const RenderTargetGroup* renderTarget;
    std::vector<bool>        renderTargetLocalReadColorAttachments;
    bool                     renderTargetLocalReadDepthStencilAttachment;
    RenderPassInputUsage     colorInputUsage;
    RenderPassInputUsage     depthInputUsage;
    RenderPassOutputUsage    colorOutputUsage;
    RenderPassOutputUsage    depthOutputUsage;
};

struct PassDataProcessor
{
    PassDataProcessor() = default;
    PassDataProcessor(std::function<void(uint32_t, CommandListVulkan&)> inDrawCallback)
    : drawCallback(inDrawCallback)
    {
    }
    
    PassDataProcessor(
        std::function<void(uint32_t, CommandListVulkan&)> inPreDrawCallback,
        std::function<void(uint32_t, CommandListVulkan&)> inDrawCallback,
        std::function<void(uint32_t, CommandListVulkan&)> inPostDrawCallback)
    : preDrawCallback(inPreDrawCallback)
    , drawCallback(inDrawCallback)
    , postDrawCallback(inPostDrawCallback)
    {
    }

    std::function<void(uint32_t, CommandListVulkan&)> preDrawCallback;
    std::function<void(uint32_t, CommandListVulkan&)> drawCallback;
    std::function<void(uint32_t, CommandListVulkan&)> postDrawCallback;
};

using PassRecordData = std::pair<std::reference_wrapper<BasePassData>, PassDataProcessor>;

struct TileMemoryHeapInfo
{
    VkDeviceMemory deviceMemoryBlob;
    VkDeviceSize allocationSize;
    VkDeviceSize usedSize;
    uint32_t heapIndex;
    uint32_t memoryType;
    VkMemoryHeap heap;

    VkDeviceSize GetAvailableSize() const
    {
        return allocationSize - usedSize;
    }
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
    bool InitBuffers();
    bool InitGui(uintptr_t windowHandle);
    bool LoadMeshObjects();
    bool CreateComputables();
    bool InitCommandBuffers();
    bool InitLocalSemaphores();
    bool SetupRenderContext();
    
private:

    void UpdateGui();
    bool UpdateUniforms(uint32_t WhichBuffer);
    bool UpdateDescriptors(uint32_t WhichBuffer);

private:

    // Passes
    std::array<RenderPassData, NUM_RENDER_PASSES>   m_RenderPassData;
    std::array<ComputePassData, NUM_COMPUTE_PASSES> m_ComputePassData;

    std::array<RenderTargetGroup, NUM_RENDER_PASSES> m_RenderTargets;

    std::array< CommandListVulkan, NUM_VULKAN_BUFFERS> m_SwapchainCopyCmdBuffer;
    VkSemaphore m_SwapchhainCopySemaphore;

    std::array< CommandListVulkan, NUM_VULKAN_BUFFERS> m_CmdBuffer;
    VkSemaphore                                        m_Semaphore;

    // UBOs
    UniformT<ObjectVertUB>                                    m_ObjectVertUniform;
    ObjectVertUB                                              m_ObjectVertUniformData;
    Uniform                                                   m_LightUniform;
    std::array<LightUB, MAX_LIGHT_COUNT>                      m_LightUniformData;
    UniformT<SceneInfoUB>                                     m_SceneInfoUniform;
    SceneInfoUB                                               m_SceneInfoUniformData;

    BufferVulkan m_LightIndicesBuffer;
    BufferVulkan m_LightCountsBuffer;

    std::unordered_map<std::size_t, ObjectMaterialParameters> m_ObjectFragUniforms;

    // Drawables
    std::vector<Drawable> m_SceneDrawables;
    std::unique_ptr<Drawable> m_DeferredLightQuadDrawable;
    std::unique_ptr<Drawable> m_BlitQuadDrawable;

    // Shaders
    std::unique_ptr<ShaderManager> m_ShaderManager;

    // Materials
    std::unique_ptr<MaterialManager> m_MaterialManager;

    // Tile memory heap specific
    bool m_IsTileMemoryHeapSupported  = true;
    bool m_RenderTargetUsesTileMemory = false;
    // std::optional<TileMemoryHeapInfo> m_tileMemoryHeapInfo;

    // Tile memory extension
    const TileMemoryHeapExtension* m_TileMemoryHeapExtension = nullptr;
    MemoryPool<Vulkan>             m_TileMemoryPool;
    Texture<Vulkan>                m_AlbedoColorBufferCopy;
};
