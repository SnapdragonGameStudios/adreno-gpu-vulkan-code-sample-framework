//============================================================================================================
//
//
//                  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#include "application.hpp"
#include "main/applicationEntrypoint.hpp"
#include "camera/cameraController.hpp"
#include "camera/cameraControllerTouch.hpp"
#include "camera/cameraData.hpp"
#include "camera/cameraGltfLoader.hpp"
#include "gui/imguiVulkan.hpp"
#include "material/drawable.hpp"
#include "material/vulkan/shaderModule.hpp"
#include "material/computable.hpp"
#include "material/materialManager.hpp"
#include "material/shaderManagerT.hpp"
#include "material/vulkan/specializationConstantsLayout.hpp"
#include "mesh/meshHelper.hpp"
#include "mesh/meshLoader.hpp"
#include "system/math_common.hpp"
#include "texture/textureManager.hpp"
#include "imgui.h"
#include "vulkan/extensionHelpers.hpp"
#include "material/drawableLoader.hpp"
#include "mesh/mesh.hpp"
#include "vulkan/extensionLib.hpp"
#include "material/vulkan/computable.hpp"
#include "material/vulkan/drawable.hpp"

#include <random>
#include <iostream>
#include <filesystem>

/*
============================================================================================================
* TILE MEMORY HEAP SAMPLE
============================================================================================================
*
* This sample demonstrates a light clustering algorithm using Vulkan, with specific support for the
* VK_QCOM_tile_memory_heap extension. This extension allows the application to allocate and manage tile memory,
* which is used for efficient memory management within a command buffer submission batch.
*
* Reference points:
*
* 1. PreInitializeSetVulkanConfiguration
*    - Sets up the Vulkan configuration, including required and optional extensions.
*    - config.OptionalExtension<ExtensionLib::Ext_VK_QCOM_tile_memory_heap>();
*    - The application checks for the VK_QCOM_tile_memory_heap extension and sets up the Vulkan configuration accordingly.
*    - Memory objects are created, including setting up the tile memory heap if supported.
* 
* 2. CreateMemoryObjects
*    - Creates memory objects, including tile memory if supported.
* 
* 3. CreateRenderTargets
*    - Render targets are created with the option to use tile memory. This involves setting the appropriate usage flags and binding the memory.
* 
* 4. Render
*    - The main rendering function binds the tile memory heap to the command buffer if tile memory is enabled.
*    - The rendering process includes multiple passes:
*      - GBuffer Pass: Outputs albedo, normal, and depth.
*      - Clustering Pass: Divides point lights into clusters and performs light culling.
*      - Lighting Pass: Applies the corresponding pixel light cluster to a screen quad.
*
* SPEC: https://registry.khronos.org/vulkan/specs/latest/man/html/VK_QCOM_tile_memory_heap.html
============================================================================================================
*/

namespace
{
    static constexpr std::array<const char*, NUM_RENDER_PASSES> sRenderPassNames = { "RP_SCENE", "RP_DEFERRED_LIGHT", "RP_HUD", "RP_BLIT" };

    glm::vec3 gCameraStartPos = glm::vec3(0.0f, 3.5f, 0.0f);
    glm::vec3 gCameraStartRot = glm::vec3(0.0f, 0.0f, 0.0f);

    float   gFOV = PI_DIV_4;
    float   gNearPlane = 1.0f;
    float   gFarPlane = 1800.0f;
    float   gNormalAmount = 0.3f;
    float   gNormalMirrorReflectAmount = 0.05f;

    const char* gSceneAssetModel = "SteamPunkSauna.gltf";

    const TextureFormat SceneColorType[]         = { TextureFormat::R8G8B8A8_SRGB, TextureFormat::R8G8B8A8_SRGB };
    const TextureFormat DeferredLightColorType[] = { TextureFormat::R8G8B8A8_SRGB};
    const TextureFormat HudColorType[]           = { TextureFormat::R8G8B8A8_SRGB };
    const TextureFormat FinalColorType[]         = { TextureFormat::R8G8B8A8_SRGB };
}

//-----------------------------------------------------------------------------
// Patch the ExtensionLib::Ext_VK_QCOM_tile_memory_heap extension so we can intercept the extension
// being registered with VkDeviceCreate and (potential) request tile memory.
struct TileMemoryHeapExtension : public ExtensionLib::Ext_VK_QCOM_tile_memory_heap
{
    TileMemoryHeapExtension(VulkanExtensionStatus status, Vulkan& vulkan)
        : ExtensionLib::Ext_VK_QCOM_tile_memory_heap(status)
        , m_Vulkan( vulkan )
    {}
    void PostLoad() override
    {
        assert( m_QcomTileMemoryTypeIndex == -1 );

        // Memory manager not available yet so can't use MemoryManager<Vulkan>::GetHeapWithHeapFlags
        // Go directly to Vulkan api for memory heap data
        VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(m_Vulkan.m_VulkanGpu, &physicalDeviceMemoryProperties );
        for (uint32_t typeIndex = 0; typeIndex < physicalDeviceMemoryProperties.memoryTypeCount; ++typeIndex)
        {
            auto heapIndex = physicalDeviceMemoryProperties.memoryTypes[typeIndex].heapIndex;
            if ((physicalDeviceMemoryProperties.memoryHeaps[heapIndex].flags & VkMemoryHeapFlagBits( VK_MEMORY_HEAP_TILE_MEMORY_BIT_QCOM )) != 0)
            {
                m_tileMemoryHeapSize = physicalDeviceMemoryProperties.memoryHeaps[heapIndex].size;
                m_QcomTileMemoryTypeIndex = typeIndex;
                LOGI( "Qcom tile memory typeIndex: %d  heap index: %d  Heap size: %zu", m_QcomTileMemoryTypeIndex, heapIndex, m_tileMemoryHeapSize );
                break;
            }
        }
        if (m_QcomTileMemoryTypeIndex == -1)
        {
            LOGI( "Qcom tile memory heap NOT FOUND (despite VK_QCOM_tile_memory_heap extension being loaded)" );
        }
    }

    MemoryPool<Vulkan> CreateTileMemoryPool() const
    {
        if (m_QcomTileMemoryTypeIndex < 0)
            return {};
        return m_Vulkan.GetMemoryManager().CreateCustomPool( m_QcomTileMemoryTypeIndex, m_tileMemoryHeapSize, 1, VK_BUFFER_USAGE_TILE_MEMORY_BIT_QCOM, VK_IMAGE_USAGE_TILE_MEMORY_BIT_QCOM );
    }

    Vulkan& m_Vulkan;
    int     m_QcomTileMemoryTypeIndex = -1;
    size_t  m_tileMemoryHeapSize = 0;
};


///
/// @brief Implementation of the Application entrypoint (called by the framework)
/// @return Pointer to Application (derived from @FrameworkApplicationBase).
/// Creates the Application class.  Ownership is passed to the calling (framework) function.
/// 
FrameworkApplicationBase* Application_ConstructApplication()
{
    return new Application();
}

Application::Application() : ApplicationHelperBase()
{
}

Application::~Application()
{
}

//-----------------------------------------------------------------------------
void Application::PreInitializeSetVulkanConfiguration(Vulkan::AppConfiguration& config)
//-----------------------------------------------------------------------------
{
    ApplicationHelperBase::PreInitializeSetVulkanConfiguration(config);
    config.RequiredExtension<ExtensionLib::Ext_VK_KHR_synchronization2>();
    config.RequiredExtension<ExtensionLib::Ext_VK_KHR_create_renderpass2>();
    config.RequiredExtension<ExtensionLib::Ext_VK_KHR_depth_stencil_resolve>();
    config.RequiredExtension<ExtensionLib::Ext_VK_KHR_get_memory_requirements2>();
    config.RequiredExtension<ExtensionLib::Ext_VK_KHR_dynamic_rendering>();
    config.OptionalExtension<ExtensionLib::Ext_VK_KHR_dynamic_rendering_local_read>();

    m_TileMemoryHeapExtension = config.OptionalExtension<TileMemoryHeapExtension>(*GetVulkan());
}

//-----------------------------------------------------------------------------
bool Application::Initialize(uintptr_t windowHandle, uintptr_t hInstance)
//-----------------------------------------------------------------------------
{
    if (!ApplicationHelperBase::Initialize( windowHandle, hInstance ))
    {
        return false;
    }

    m_IsTileMemoryHeapSupported &= GetVulkan()->HasLoadedVulkanDeviceExtension(VK_QCOM_TILE_MEMORY_HEAP_EXTENSION_NAME);

    if (m_IsTileMemoryHeapSupported && m_TileMemoryHeapExtension)
    {
        m_TileMemoryPool = m_TileMemoryHeapExtension->CreateTileMemoryPool();
    }

    if (!InitializeCamera())
    {
        return false;
    }
    
    if (!InitializeLights())
    {
        return false;
    }
        
    if (!LoadShaders())
    {
        return false;
    }

    if (!InitBuffers())
    {
        return false;
    }

    if (!CreateRenderTargets())
    {
        return false;
    }

    if (!SetupRenderContext())
    {
        return false;
    }

    if (!InitGui(windowHandle))
    {
        return false;
    }

    if (!LoadMeshObjects())
    {
        return false;
    }
    
    if (!CreateComputables())
    {
        return false;
    }

    if (!InitCommandBuffers())
    {
        return false;
    }

    if (!InitLocalSemaphores())
    {
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
void Application::Destroy()
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();

    // Uniform Buffers
    ReleaseUniformBuffer(pVulkan, &m_ObjectVertUniform);
    ReleaseUniformBuffer(pVulkan, &m_LightUniform);
     
    for (auto& [hash, objectUniform] : m_ObjectFragUniforms)
    {
        ReleaseUniformBuffer(pVulkan, &objectUniform.objectFragUniform);
    }

    // Cmd buffers
    for (int whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        m_CmdBuffer[whichPass].Release();
        
        for(auto& colorTexture : m_RenderTargets[whichPass].color)
        {
            colorTexture.Release(pVulkan);
        }

        m_RenderTargets[whichPass].depth.Release(pVulkan);
    }

    for(auto& swapchainCopyCmdBuffer : m_SwapchainCopyCmdBuffer)
    {
        swapchainCopyCmdBuffer.Release();
    }

    // Semaphores
    vkDestroySemaphore(pVulkan->m_VulkanDevice, m_Semaphore, nullptr);

    m_AlbedoColorBufferCopy.Release(pVulkan);

    // Drawables
    m_SceneDrawables.clear();
    m_BlitQuadDrawable.reset();

    // Internal
    m_ShaderManager.reset();
    m_MaterialManager.reset();
    m_CameraController.reset();
    m_AssetManager.reset();

    ApplicationHelperBase::Destroy();
}

//-----------------------------------------------------------------------------
bool Application::InitializeLights()
//-----------------------------------------------------------------------------
{
    // NOTE: Hardcoded values for the Museum asset
    const float sceneRadius    = 67.0f;
    const float sceneMinHeight = 3.0f;
    const float sceneMaxHeight = 25.0f;

    const glm::vec2 lightScaleRange = glm::vec2(3.5f, 5.0f);

    std::random_device rd;
    std::mt19937 gen(rd());

    // Distributions for position, scale, and color
    std::uniform_real_distribution<float> sceneRadiusDist(-sceneRadius, sceneRadius);
    std::uniform_real_distribution<float> sceneHeightDist(sceneMinHeight, sceneMaxHeight);
    std::uniform_real_distribution<float> lightRadiusDist(lightScaleRange.x, lightScaleRange.y);
    std::uniform_real_distribution<float> lightColorDist(0.0f, 1.0f);

    for(auto& lightData : m_LightUniformData)
    {
        const glm::vec2 direction     = glm::normalize(glm::vec2(sceneRadiusDist(gen), sceneRadiusDist(gen)));
        const glm::vec2 planePosition = direction * sceneRadiusDist(gen);
        const glm::vec3 lightPosition = glm::vec3(planePosition.x, sceneHeightDist(gen), planePosition.y);

        lightData.lightPosAndRadius = glm::vec4(lightPosition, lightRadiusDist(gen));
        lightData.lightColor        = glm::vec4(lightColorDist(gen), lightColorDist(gen), lightColorDist(gen), 1.0f);
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitializeCamera()
//-----------------------------------------------------------------------------
{
    LOGI("******************************");
    LOGI("Initializing Camera...");
    LOGI("******************************");

    m_Camera.SetPosition(gCameraStartPos, glm::quat(gCameraStartRot * TO_RADIANS));
    m_Camera.SetAspect(float(gRenderWidth) / float(gRenderHeight));
    m_Camera.SetFov(gFOV);
    m_Camera.SetClipPlanes(gNearPlane, gFarPlane);

    // Camera Controller //

#if defined(OS_ANDROID)
    typedef CameraControllerTouch           tCameraController;
#else
    typedef CameraController                tCameraController;
#endif

    auto cameraController = std::make_unique<tCameraController>();
    if (!cameraController->Initialize(gRenderWidth, gRenderHeight))
    {
        return false;
    }

    m_CameraController = std::move(cameraController);
    m_CameraController->SetMoveSpeed(10.0f);

    return true;
}

//-----------------------------------------------------------------------------
bool Application::LoadShaders()
//-----------------------------------------------------------------------------
{
    m_ShaderManager = std::make_unique<ShaderManager>(*GetVulkan());
    m_ShaderManager->RegisterRenderPassNames(sRenderPassNames);

    m_MaterialManager = std::make_unique<MaterialManager>(*GetVulkan());

    LOGI("******************************");
    LOGI("Loading Shaders...");
    LOGI("******************************");

    typedef std::pair<std::string, std::string> tIdAndFilename;
    for (const tIdAndFilename& i :
            { tIdAndFilename { "Blit",  "Blit.json" },
              tIdAndFilename { "SceneOpaque", "SceneOpaque.json" },
              tIdAndFilename { "SceneTransparent", "SceneTransparent.json" },
              tIdAndFilename { "LightCulling", "LightCulling.json" },
              tIdAndFilename { "DeferredLight", "DeferredLight.json" }
            })
    {
        if (!m_ShaderManager->AddShader(*m_AssetManager, i.first, i.second, SHADER_DESTINATION_PATH))
        {
            LOGE("Error Loading shader %s from %s", i.first.c_str(), i.second.c_str());
            LOGI("Please verify if you have all required assets on the sample media folder");
            LOGI("If you are running on Android, don't forget to run the `02_CopyMediaToDevice.bat` script to copy all media files into the device memory");
            return false;
        }
    }

    return true;
}

static Texture<Vulkan> TileMemoryAllocateTry(
    Vulkan&              		vulkan,
    CreateTexObjectInfo&        createInfo,
    MemoryPool<Vulkan>*const    tileMemoryPoolPtr,
    const std::string&         	textureName)
{
    assert(tileMemoryPoolPtr);
    
    const auto tileMemoryTextureName = textureName + " TILEMEM";
    createInfo.pName = tileMemoryTextureName.c_str();

    auto newTexture = CreateTextureObject(vulkan, createInfo, tileMemoryPoolPtr);
    if (!newTexture.IsEmpty())
    {
        LOGI("%s was allocated in tile memory", createInfo.pName);
    }
    return newTexture;
}

static TEXTURE_TYPE TextureTypeGet(const std::string& textureName, const TEXTURE_TYPE textureTypeDefault)
{
    return textureName == "BLIT" ? TEXTURE_TYPE::TT_RENDER_TARGET_TRANSFERSRC : textureTypeDefault;
}


//-----------------------------------------------------------------------------
bool Application::CreateRenderTargets()
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();

    LOGI("**************************");
    LOGI("Creating Render Targets...");
    LOGI("**************************");

    const TextureFormat desiredDepthFormat = pVulkan->GetBestSurfaceDepthFormat();

    auto CreateRenderTarget = [&](
        auto& renderTargetGroup, 
        std::string_view name, 
        bool tileMemory,
        uint32_t width,
        uint32_t height,
        const std::span<const TextureFormat> colorFormats, 
        std::optional< TextureFormat> depthFormat = std::nullopt)
    {
        auto textureName = std::string(name);

        for (size_t colorFormatIndex=0; colorFormatIndex < colorFormats.size(); ++colorFormatIndex)
        {
            const auto& colorFormat = colorFormats[colorFormatIndex];
            auto textureNameColor = textureName + std::string(" RENDER_TARGET_") + std::to_string(colorFormatIndex);
            CreateTexObjectInfo createInfo{
                .uiWidth = width,
                .uiHeight = height,
                .Format = colorFormat,
                .TexType = TextureTypeGet(textureName, TEXTURE_TYPE::TT_RENDER_TARGET),
                .pName = textureNameColor.c_str(),
            };
            renderTargetGroup.color.push_back(CreateTextureObject(*pVulkan, createInfo));

            if (tileMemory && m_TileMemoryPool)
            {
                auto tileTexture = TileMemoryAllocateTry(*pVulkan, createInfo, &m_TileMemoryPool, textureNameColor);
                if (!tileTexture.IsEmpty())
                {
                    renderTargetGroup.colorTileMemory.push_back(std::move(tileTexture));
                    continue;
                }
            }

            renderTargetGroup.colorTileMemory.push_back(CreateTextureObject(*pVulkan, createInfo));
        }

        if (depthFormat)
        {
            auto textureNameDepth = textureName + std::string(" DEPTH");
            CreateTexObjectInfo createInfo{
                .uiWidth = width,
                .uiHeight = height,
                .Format = depthFormat.value(),
                .TexType = TextureTypeGet(textureName, TEXTURE_TYPE::TT_DEPTH_TARGET),
                .pName = textureNameDepth.c_str(),
            };
            renderTargetGroup.depth = CreateTextureObject(*pVulkan, createInfo);

            if (tileMemory && m_TileMemoryPool)
            {
				auto tileTexture = TileMemoryAllocateTry(*pVulkan, createInfo, &m_TileMemoryPool, textureNameDepth);
                if (!tileTexture.IsEmpty())
                {
                    renderTargetGroup.depthTileMemory = std::move(tileTexture);
                    return;
                }
            }

            renderTargetGroup.depthTileMemory = CreateTextureObject(*pVulkan, createInfo);
        }
    };
    
    CreateRenderTarget(m_RenderTargets[RP_SCENE], "SCENE", true, gRenderWidth, gRenderHeight, SceneColorType, desiredDepthFormat);
    CreateRenderTarget(m_RenderTargets[RP_DEFERRED_LIGHT], "DEFERRED LIGHT", false, gRenderWidth, gRenderHeight, DeferredLightColorType);
    CreateRenderTarget(m_RenderTargets[RP_HUD], "HUD", false, gSurfaceWidth, gSurfaceHeight, HudColorType);
    CreateRenderTarget(m_RenderTargets[RP_BLIT], "BLIT", false, gSurfaceWidth, gSurfaceHeight, FinalColorType);

    {
        CreateTexObjectInfo createInfo{
            .uiWidth  = gRenderWidth,
            .uiHeight = gRenderHeight,
            .Format   = SceneColorType[0],
            .TexType  = TEXTURE_TYPE::TT_SAMPLED_TRANSFERDST,
            .pName    = "AlbedoColorBufferCopy",
        };
        m_AlbedoColorBufferCopy = CreateTextureObject(*pVulkan, createInfo);
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::SetupRenderContext()
//-----------------------------------------------------------------------------
{
    auto& vulkan = *GetVulkan();

    LOGI("******************************");
    LOGI("Initializing Render Passes... ");
    LOGI("******************************");

    for (uint32_t whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        std::vector< TextureFormat> colorFormats;
        for (auto& colorAttachment : m_RenderTargets[whichPass].color)
        {
            colorFormats.push_back(colorAttachment.Format);
        }

        m_RenderPassData[whichPass].renderContext =
        { 
            colorFormats, 
            m_RenderTargets[whichPass].depth.Format,
            TextureFormat::UNDEFINED, 
            sRenderPassNames[whichPass]
        };
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitBuffers()
//-----------------------------------------------------------------------------
{
    LOGI("******************************");
    LOGI("Initializing Buffers...");
    LOGI("******************************");

    Vulkan* const pVulkan = GetVulkan();

    if (!CreateUniformBuffer(pVulkan, m_ObjectVertUniform))
    {
        return false;
    }

    if (!CreateUniformBuffer(pVulkan, &m_LightUniform, sizeof(LightUB) * m_LightUniformData.size(), m_LightUniformData.data()))
    {
        return false;
    }

    if (!CreateUniformBuffer(pVulkan, m_SceneInfoUniform))
    {
        return false;
    }

    // Worst case = every light is present on every tile
    // NOTE: int32_t can totally be swapped by a smaller type
    const size_t lightIndicesBufferSize = sizeof(int32_t) * MAX_CLUSTER_MATRIX_SIZE * MAX_LIGHT_COUNT;

    if (!m_LightCountsBuffer.Initialize(
        &pVulkan->GetMemoryManager(),
        sizeof(int32_t) * MAX_CLUSTER_MATRIX_SIZE,
        BufferUsageFlags::Storage,
        MemoryUsage::GpuExclusive))
    {
        return false;
    }
    
    if (!m_LightIndicesBuffer.Initialize(
        &pVulkan->GetMemoryManager(),
        lightIndicesBufferSize,
        BufferUsageFlags::Storage,
        MemoryUsage::GpuExclusive))
    {
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitGui(uintptr_t windowHandle)
//-----------------------------------------------------------------------------
{
    const auto& hudRenderTarget = m_RenderTargets[RP_HUD];
    m_Gui = std::make_unique<GuiImguiGfx>(*GetVulkan());
    if (!m_Gui->Initialize(
        windowHandle, 
        hudRenderTarget.color[0].Format, 
        hudRenderTarget.color[0].Width,
        hudRenderTarget.color[0].Height))
    {
        return false;
    }
    
    return true;
}

//-----------------------------------------------------------------------------
bool Application::LoadMeshObjects()
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();

    LOGI("***********************");
    LOGI("Initializing Meshes... ");
    LOGI("***********************");

    const auto* pSceneOpaqueShader             = m_ShaderManager->GetShader("SceneOpaque");
    const auto* pSceneTransparentShader        = m_ShaderManager->GetShader("SceneTransparent");
    const auto* deferredLightShader            = m_ShaderManager->GetShader("DeferredLight");
    const auto* pBlitQuadShader                = m_ShaderManager->GetShader("Blit");
    if (!pSceneOpaqueShader || !pSceneTransparentShader || !pBlitQuadShader || !deferredLightShader)
    {
        return false;
    }
    
    LOGI("***********************************");
    LOGI("Loading and preparing the museum...");
    LOGI("***********************************");

    const PathManipulator_PrefixDirectory prefixTextureDir{ TEXTURE_DESTINATION_PATH };
    auto* whiteTexture = m_TextureManager->GetOrLoadTexture("white_d.ktx", m_SamplerRepeat, prefixTextureDir);
    auto* blackTexture = m_TextureManager->GetOrLoadTexture("black_d.ktx", m_SamplerRepeat, prefixTextureDir);
    auto* normalDefaultTexture = m_TextureManager->GetOrLoadTexture("normal_default.ktx", m_SamplerRepeat, prefixTextureDir);

    if (!whiteTexture || !blackTexture || !normalDefaultTexture)
    {
        LOGE("Failed to load supporting textures");
        return false;
    }

    auto UniformBufferLoader = [&](const ObjectMaterialParameters& objectMaterialParameters) -> ObjectMaterialParameters&
    {
        auto hash = objectMaterialParameters.GetHash();

        auto iter = m_ObjectFragUniforms.try_emplace(hash, ObjectMaterialParameters());
        if (iter.second)
        {
            iter.first->second.objectFragUniformData = objectMaterialParameters.objectFragUniformData;
            if (!CreateUniformBuffer(pVulkan, iter.first->second.objectFragUniform))
            {
                LOGE("Failed to create object uniform buffer");
            }
        }

        return iter.first->second;
    };

    auto MaterialLoader = [&](const MeshObjectIntermediate::MaterialDef& materialDef)->std::optional<Material>
    {
        const PathManipulator_PrefixDirectory prefixTextureDir{ TEXTURE_DESTINATION_PATH };
        const PathManipulator_ChangeExtension changeTextureExt{ ".ktx" };

        auto* diffuseTexture = m_TextureManager->GetOrLoadTexture(materialDef.diffuseFilename, m_SamplerRepeat, prefixTextureDir, changeTextureExt);
        auto* normalTexture = m_TextureManager->GetOrLoadTexture(materialDef.bumpFilename, m_SamplerRepeat, prefixTextureDir, changeTextureExt);
        auto* emissiveTexture = m_TextureManager->GetOrLoadTexture(materialDef.emissiveFilename, m_SamplerRepeat, prefixTextureDir, changeTextureExt);
        auto* metallicRoughnessTexture = m_TextureManager->GetOrLoadTexture(materialDef.specMapFilename, m_SamplerRepeat, prefixTextureDir, changeTextureExt);
        bool alphaCutout = materialDef.alphaCutout;
        bool transparent = materialDef.transparent;

        const auto* targetShader = transparent ? pSceneTransparentShader : pSceneOpaqueShader;

        ObjectMaterialParameters objectMaterial;
        objectMaterial.objectFragUniformData.Color.r = static_cast<float>(materialDef.baseColorFactor[0]);
        objectMaterial.objectFragUniformData.Color.g = static_cast<float>(materialDef.baseColorFactor[1]);
        objectMaterial.objectFragUniformData.Color.b = static_cast<float>(materialDef.baseColorFactor[2]);
        objectMaterial.objectFragUniformData.Color.a = static_cast<float>(materialDef.baseColorFactor[3]);
        objectMaterial.objectFragUniformData.ORM.b   = static_cast<float>(materialDef.metallicFactor);
        objectMaterial.objectFragUniformData.ORM.g   = static_cast<float>(materialDef.roughnessFactor);

        if (diffuseTexture == nullptr || normalTexture == nullptr)
        {
            return std::nullopt;
        }

        auto shaderMaterial = m_MaterialManager->CreateMaterial(*targetShader, NUM_VULKAN_BUFFERS,
            [&](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
            {
                if (texName == "Diffuse")
                {
                    return { diffuseTexture ? diffuseTexture : whiteTexture };
                }
                if (texName == "Normal")
                {
                    return { normalTexture ? normalTexture : normalDefaultTexture };
                }
                if (texName == "Emissive")
                {
                    return { emissiveTexture ? emissiveTexture : blackTexture };
                }
                if (texName == "MetallicRoughness")
                {
                    return { metallicRoughnessTexture ? metallicRoughnessTexture : blackTexture };
                }

                return {};
            },
            [&](const std::string& bufferName) -> PerFrameBufferVulkan
            {
                if (bufferName == "Vert")
                {
                    return { m_ObjectVertUniform.buf.GetVkBuffer() };
                }
                else if (bufferName == "Frag")
                {
                    return { UniformBufferLoader(objectMaterial).objectFragUniform.buf.GetVkBuffer() };
                }

                return {};
            }
            );

        return shaderMaterial;
    };


    const uint32_t loaderFlags = 0; // No instancing
    const bool ignoreTransforms = (loaderFlags & DrawableLoader::LoaderFlags::IgnoreHierarchy) != 0;

    const auto sceneAssetPath = std::filesystem::path(MESH_DESTINATION_PATH).append(gSceneAssetModel).string();
    MeshLoaderModelSceneSanityCheck meshSanityCheckProcessor(sceneAssetPath);
    MeshObjectIntermediateGltfProcessor meshObjectProcessor(sceneAssetPath, ignoreTransforms, glm::vec3(1.0f,1.0f,1.0f));
    CameraGltfProcessor meshCameraProcessor{};

    if (!MeshLoader::LoadGltf(*m_AssetManager, sceneAssetPath, meshSanityCheckProcessor, meshObjectProcessor, meshCameraProcessor) ||
        !DrawableLoader::CreateDrawables(*pVulkan,
            std::move(meshObjectProcessor.m_meshObjects),
            m_RenderPassData[RP_SCENE].renderContext,
            MaterialLoader,
            m_SceneDrawables,
            loaderFlags))
    {
        LOGE("Error Loading the museum gltf file");
        LOGI("Please verify if you have all required assets on the sample media folder");
        LOGI("If you are running on Android, don't forget to run the `02_CopyMediaToDevice.bat` script to copy all media files into the device memory");
        return false;
    }

    if (!meshCameraProcessor.m_cameras.empty())
    {
        const auto& camera = meshCameraProcessor.m_cameras[0];
        m_Camera.SetPosition(camera.Position, camera.Orientation);
    }

    LOGI("*********************");
    LOGI("Creating Quad mesh...");
    LOGI("*********************");
    {
        Mesh blitQuadMesh;
        MeshHelper::CreateScreenSpaceMesh(pVulkan->GetMemoryManager(), 0, &blitQuadMesh);

        // Blit Material
        auto blitQuadShaderMaterial = m_MaterialManager->CreateMaterial(*pBlitQuadShader, pVulkan->m_SwapchainImageCount,
            [this](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
            {
                if (texName == "Scene")
                {
                    return { &m_RenderTargets[RP_DEFERRED_LIGHT].color[0] };
                }
                else if (texName == "Overlay")
                {
                    return { &m_RenderTargets[RP_HUD].color[0] };
                }
                return {};
            },
            [this](const std::string& bufferName) -> PerFrameBufferVulkan
            {
                return {};
            }
            );

        m_BlitQuadDrawable = std::make_unique<Drawable>(*pVulkan, std::move(blitQuadShaderMaterial));
        if (!m_BlitQuadDrawable->Init(m_RenderPassData[RP_BLIT].renderContext, std::move(blitQuadMesh)))
        {
            return false;
        }
    }

    LOGI("************************************");
    LOGI("Creating Deferred Light Quad mesh...");
    LOGI("************************************");
    {
        Mesh blitQuadMesh;
        MeshHelper::CreateScreenSpaceMesh(pVulkan->GetMemoryManager(), 0, &blitQuadMesh);

        auto deferredLightQuadShaderMaterial = m_MaterialManager->CreateMaterial(*deferredLightShader, NUM_VULKAN_BUFFERS,
            [this](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
            {
                if (texName == "SceneColor")
                {
                    return { &m_RenderTargets[RP_SCENE].color[0] };
                }
                else if (texName == "SceneNormal")
                {
                    return { &m_RenderTargets[RP_SCENE].color[1] };
                }
                else if (texName == "SceneDepth")
                {
                    return { &m_RenderTargets[RP_SCENE].depth };
                }
                return {};
            },
            [&](const std::string& bufferName) -> PerFrameBufferVulkan
            {
                if (bufferName == "SceneInfo")
                {
                    return { m_SceneInfoUniform.buf.GetVkBuffer() };
                }
                else if (bufferName == "LightInfo")
                {
                    return { m_LightUniform.buf.GetVkBuffer() };
                }
                else if (bufferName == "LightIndices")
                {
                    return { m_LightIndicesBuffer.GetVkBuffer() };
                }
                else if (bufferName == "LightCounts")
                {
                    return { m_LightCountsBuffer.GetVkBuffer() };
                }
                assert(0);
                return {};
            }
        );

        m_DeferredLightQuadDrawable = std::make_unique<Drawable>(*pVulkan, std::move(deferredLightQuadShaderMaterial));
        if (!m_DeferredLightQuadDrawable->Init(m_RenderPassData[RP_DEFERRED_LIGHT].renderContext, std::move(blitQuadMesh)))
        {
            return false;
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::CreateComputables()
//-----------------------------------------------------------------------------
{
    auto pVulkan = GetVulkan();

    const auto* lightCullingShader = m_ShaderManager->GetShader("LightCulling");
    if (!lightCullingShader)
    {
        return false;
    }

    auto material = m_MaterialManager->CreateMaterial(
        *lightCullingShader,
        NUM_VULKAN_BUFFERS,
        [&](const std::string& texName) -> MaterialManager::tPerFrameTexInfo 
        {
            if (texName == "SceneDepth")
            {
                return { &m_RenderTargets[RP_SCENE].depth };
            }
            assert(0);
            return {};
        }, 
        [&](const std::string& bufferName) -> PerFrameBufferVulkan
        {
            if (bufferName == "SceneInfo")
            {
                return { m_SceneInfoUniform.buf.GetVkBuffer() };
            }
            else if (bufferName == "LightInfo")
            {
                return { m_LightUniform.buf.GetVkBuffer() };
            }
            else if (bufferName == "LightIndices")
            {
                return { m_LightIndicesBuffer.GetVkBuffer() };
            }
            else if (bufferName == "LightCounts")
            {
                return { m_LightCountsBuffer.GetVkBuffer() };
            }
            assert(0);
            return {};
        });

    m_ComputePassData[CP_LIGHT_CULLING_LIST].passComputable = std::make_unique<Computable>(*pVulkan, std::move(material));
    if (!m_ComputePassData[CP_LIGHT_CULLING_LIST].passComputable->Init())
    {
        LOGE("Error Creating SGSR computables...");
    }
    
    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitCommandBuffers()
//-----------------------------------------------------------------------------
{
    LOGI("*******************************");
    LOGI("Initializing Command Buffers...");
    LOGI("*******************************");

    Vulkan* const pVulkan = GetVulkan();

    auto GetPassName = [](uint32_t whichPass)
    {
        if (whichPass >= sRenderPassNames.size())
        {
            LOGE("GetPassName() called with unknown pass (%d)!", whichPass);
            return "RP_UNKNOWN";
        }

        return sRenderPassNames[whichPass];
    };

    char szName[256];
    for (uint32_t whichBuffer = 0; whichBuffer < m_CmdBuffer.size(); whichBuffer++)
    {
        // The Pass Command Buffer => Primary
        sprintf(szName, "Primary (Buffer %d of %d)", whichBuffer + 1, NUM_VULKAN_BUFFERS);
        if (!m_CmdBuffer[whichBuffer].Initialize(pVulkan, szName, CommandListBase::Type::Primary))
        {
            return false;
        }
    }

    for (int i=0; i< m_SwapchainCopyCmdBuffer.size(); i++)
    {
        auto& swapchainCopyCmdBuffer = m_SwapchainCopyCmdBuffer[i];

        // The Pass Command Buffer => Primary
        sprintf(szName, "Swapchain Copy CMD Buffer (Buffer %d of %d)", i + 1, NUM_VULKAN_BUFFERS);
        if (!swapchainCopyCmdBuffer.Initialize(pVulkan, szName, CommandListBase::Type::Primary))
        {
            return false;
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitLocalSemaphores()
//-----------------------------------------------------------------------------
{
    LOGI("********************************");
    LOGI("Initializing Local Semaphores...");
    LOGI("********************************");

    const VkSemaphoreCreateInfo SemaphoreInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

    VkResult retVal = vkCreateSemaphore(GetVulkan()->m_VulkanDevice, &SemaphoreInfo, NULL, &m_Semaphore);
    if (!CheckVkError("vkCreateSemaphore()", retVal))
    {
        return false;
    }

    retVal = vkCreateSemaphore(GetVulkan()->m_VulkanDevice, &SemaphoreInfo, NULL, &m_SwapchhainCopySemaphore);
    if (!CheckVkError("vkCreateSemaphore()", retVal))
    {
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
void Application::UpdateGui()
//-----------------------------------------------------------------------------
{
    if (m_Gui)
    {
        m_Gui->Update();
        ImGuiIO& io = ImGui::GetIO();

        ImGui::SetNextWindowPos(ImVec2(200, 100), ImGuiCond_Once);
        if (ImGui::Begin("FPS", (bool*)nullptr, ImGuiWindowFlags_NoTitleBar))
        {
            if(ImGui::CollapsingHeader("Scene Details"))
            {
                ImGui::Text("FPS: %.1f", m_CurrentFPS);
                ImGui::Text("Camera [%f, %f, %f]", m_Camera.Position().x, m_Camera.Position().y, m_Camera.Position().z);
            }
            
            // Sun light part isn't really doing anything right now
#if 0
            if(ImGui::CollapsingHeader("Sun Light"))
            {
                ImGui::DragFloat3("Sun Dir", &m_SceneInfoUniformData.skyLightDirection.x, 0.01f, -1.0f, 1.0f);
                ImGui::DragFloat3("Sun Color", &m_SceneInfoUniformData.skyLightColor.x, 0.01f, 0.0f, 1.0f);
                ImGui::DragFloat("Sun Intensity", &m_SceneInfoUniformData.skyLightColor.w, 0.1f, 0.0f, 100.0f);
                ImGui::DragFloat3("Ambient Color", &m_SceneInfoUniformData.ambientColor.x, 0.01f, 0.0f, 1.0f);
            }
#endif

            if(ImGui::CollapsingHeader("Clustered Deferred Light", ImGuiTreeNodeFlags_DefaultOpen))
            {
                bool ignoreLightTiles = static_cast<bool>(m_SceneInfoUniformData.ignoreLightTiles);
                ImGui::Checkbox("Deactivate Clustering", &ignoreLightTiles);
                m_SceneInfoUniformData.ignoreLightTiles = static_cast<int32_t>(ignoreLightTiles);

                bool debugShaders = static_cast<bool>(m_SceneInfoUniformData.debugShaders);
                ImGui::Checkbox("Visualize Clusters", &debugShaders);
                m_SceneInfoUniformData.debugShaders = static_cast<int32_t>(debugShaders);

                ImGui::Separator();

                ImGui::DragInt("Light Count", &m_SceneInfoUniformData.lightCount, 1.0f, 0, MAX_LIGHT_COUNT);

                ImGui::Separator();

                std::array<int32_t, 2> tileCount
                { 
                    static_cast<int32_t>(m_SceneInfoUniformData.clusterCountX) ,
                    static_cast<int32_t>(m_SceneInfoUniformData.clusterCountY)
                };

                ImGui::DragInt2("Cluster Count", tileCount.data(), 1.0f, 1, MAX_CLUSTER_MATRIX_DIMENSION_SIZE);
                m_SceneInfoUniformData.clusterCountX = std::min(static_cast<uint32_t>(MAX_CLUSTER_MATRIX_DIMENSION_SIZE), static_cast<uint32_t>(tileCount[0]));
                m_SceneInfoUniformData.clusterCountY = std::min(static_cast<uint32_t>(MAX_CLUSTER_MATRIX_DIMENSION_SIZE), static_cast<uint32_t>(tileCount[1]));

                ImGui::Text(
                    "Total Clusters: [%d, %d] -> %d", 
                    m_SceneInfoUniformData.clusterCountX,
                    m_SceneInfoUniformData.clusterCountY, 
                    m_SceneInfoUniformData.clusterCountX * m_SceneInfoUniformData.clusterCountY);
            }

            if (ImGui::CollapsingHeader("Tile Memory", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::BeginDisabled(!m_IsTileMemoryHeapSupported);
                ImGui::Checkbox("Tile Memory: Render Targets", &m_RenderTargetUsesTileMemory);
                ImGui::EndDisabled();

                m_SceneInfoUniformData.tileMemoryEnabled = static_cast<int32_t>(m_RenderTargetUsesTileMemory);
            }

            glm::vec3 LightDirNotNormalized   = m_SceneInfoUniformData.skyLightDirection;
            LightDirNotNormalized             = glm::normalize(LightDirNotNormalized);
            m_SceneInfoUniformData.skyLightDirection = glm::vec4(LightDirNotNormalized, 0.0f);
        }
        ImGui::End();

        return;
    }
}

//-----------------------------------------------------------------------------
bool Application::UpdateUniforms(uint32_t whichBuffer)
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();

    // Vert data
    {
        glm::mat4 LocalModel = glm::mat4(1.0f);
        LocalModel           = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
        LocalModel           = glm::scale(LocalModel, glm::vec3(1.0f));
        glm::mat4 LocalMVP   = m_Camera.ProjectionMatrix() * m_Camera.ViewMatrix() * LocalModel;

        m_ObjectVertUniformData.MVPMatrix   = LocalMVP;
        m_ObjectVertUniformData.ModelMatrix = LocalModel;
        UpdateUniformBuffer(pVulkan, m_ObjectVertUniform, m_ObjectVertUniformData);
    }

    // Frag data
    for (auto& [hash, objectUniform] : m_ObjectFragUniforms)
    {
        UpdateUniformBuffer(pVulkan, objectUniform.objectFragUniform, objectUniform.objectFragUniformData);
    }

    // Scene data
    {
        glm::mat4 CameraViewInv       = glm::inverse(m_Camera.ViewMatrix());
        glm::mat4 CameraProjection    = m_Camera.ProjectionMatrix();
        glm::mat4 CameraProjectionInv = glm::inverse(CameraProjection);

        m_SceneInfoUniformData.projectionInv     = CameraProjectionInv;
        m_SceneInfoUniformData.viewInv           = CameraViewInv;
        m_SceneInfoUniformData.view              = m_Camera.ViewMatrix();
        m_SceneInfoUniformData.viewInv[3][0] *= 0.5f;
        m_SceneInfoUniformData.viewInv[3][1] *= 0.5f;
        m_SceneInfoUniformData.viewInv[3][2] *= 0.5f;
        m_SceneInfoUniformData.viewProjection    = CameraProjection * m_Camera.ViewMatrix();
        m_SceneInfoUniformData.viewProjectionInv = glm::inverse(m_SceneInfoUniformData.viewProjection);
        m_SceneInfoUniformData.projectionInvW    = glm::vec4(CameraProjectionInv[0].w, CameraProjectionInv[1].w, CameraProjectionInv[2].w, CameraProjectionInv[3].w);
        m_SceneInfoUniformData.cameraPos         = glm::vec4(m_Camera.Position(), 0.0f);

        m_SceneInfoUniformData.viewportWidth = gRenderWidth;
        m_SceneInfoUniformData.viewportHeight = gRenderHeight;

        UpdateUniformBuffer(pVulkan, m_SceneInfoUniform, m_SceneInfoUniformData);
    }

    // Light data (this doesn't change, no need to update every frame)
    {
        // UpdateUniformBuffer(pVulkan, m_LightUniform, m_LightUniformData);
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::UpdateDescriptors(uint32_t whichBuffer)
//-----------------------------------------------------------------------------
{
    // NOTE: Commented sections are because transient buffers depend on the tile memory extension

    bool success = true;

    for(auto& computablePasses : m_ComputePassData[CP_LIGHT_CULLING_LIST].passComputable->GetPasses())
    {
        auto& vkComputablePass = apiCast<Vulkan>(computablePasses);
        auto& computableMaterialPass = vkComputablePass.GetMaterialPass();

        success &= computableMaterialPass.UpdateDescriptorSetBinding(whichBuffer, "SceneDepth",
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_SCENE].depthTileMemory : m_RenderTargets[RP_SCENE].depth);
    }

    success &= m_BlitQuadDrawable->GetMaterial().UpdateDescriptorSetBinding(whichBuffer, "Scene",
        m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_DEFERRED_LIGHT].colorTileMemory[0] : m_RenderTargets[RP_DEFERRED_LIGHT].color[0]);
    success &= m_BlitQuadDrawable->GetMaterial().UpdateDescriptorSetBinding(whichBuffer, "Overlay",
        m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_HUD].colorTileMemory[0] : m_RenderTargets[RP_HUD].color[0]);
        
    success &= m_DeferredLightQuadDrawable->GetMaterial().UpdateDescriptorSetBinding(whichBuffer, "SceneColor",
        m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_SCENE].colorTileMemory[0] : m_RenderTargets[RP_SCENE].color[0]);
    success &= m_DeferredLightQuadDrawable->GetMaterial().UpdateDescriptorSetBinding(whichBuffer, "SceneNormal",
        m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_SCENE].colorTileMemory[1] : m_RenderTargets[RP_SCENE].color[1]);
    success &= m_DeferredLightQuadDrawable->GetMaterial().UpdateDescriptorSetBinding(whichBuffer, "SceneDepth",
        m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_SCENE].depthTileMemory : m_RenderTargets[RP_SCENE].depth);

    return success;
}

//-----------------------------------------------------------------------------
void Application::Render(float fltDiffTime)
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();

    // Obtain the next swap chain image for the next frame.
    auto currentVulkanBuffer = pVulkan->SetNextBackBuffer();
    uint32_t whichBuffer     = currentVulkanBuffer.idx;

    // ********************************
    // Application Draw() - Begin
    // ********************************

    UpdateGui();

    // Update camera
    m_Camera.UpdateController(fltDiffTime, *m_CameraController);
    m_Camera.UpdateMatrices();

    // Update uniform buffers with latest data
    if (!UpdateUniforms(whichBuffer))
    {
        assert(false);
        return;
    }

    // Update descriptors
    if(!UpdateDescriptors(whichBuffer))
    {
        assert(false);
        return;
    }

    // const VkPipelineStageFlags defaultGfxWaitDstStageMasks[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    const VkPipelineStageFlags defaultGfxWaitDstStageMasks[] = { VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };

    // First time through, wait for the back buffer to be ready
    std::span<const VkSemaphore> pWaitSemaphores = { &currentVulkanBuffer.semaphore, 1 };

    auto RecordPassData = [this, whichBuffer, defaultGfxWaitDstStageMasks, &pWaitSemaphores](
        std::optional<std::pair<RENDER_PASS, RenderingAttachmentInfoGroup>> renderingInfo,
        PassRecordData                                                      passRecordInfo,
        VkFence                                                             completionFence = VK_NULL_HANDLE)
    {
        auto& [passData, passDataProcessor] = passRecordInfo;

        if (passDataProcessor.preDrawCallback)
        {
            passDataProcessor.preDrawCallback(whichBuffer, m_CmdBuffer[whichBuffer]);
        }

        if (renderingInfo)
        {
            const auto& [renderPassIndex, renderingAttachmentGroup] = renderingInfo.value();
            auto& renderPassInfo = m_RenderPassData[renderPassIndex].renderContext.GetRenderingInfo(renderingAttachmentGroup).get();
            vkCmdBeginRenderingKHR(m_CmdBuffer[whichBuffer].m_VkCommandBuffer, &renderPassInfo);
            
            VkViewport viewport;
            viewport.x = renderPassInfo.renderArea.offset.x;
            viewport.y = renderPassInfo.renderArea.offset.y;
            viewport.width = renderPassInfo.renderArea.extent.width;
            viewport.height = renderPassInfo.renderArea.extent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(m_CmdBuffer[whichBuffer].m_VkCommandBuffer, 0, 1, &viewport);

            VkRect2D scissor;
            scissor.offset.x = viewport.x;
            scissor.offset.y = viewport.y;
            scissor.extent.width = viewport.width;
            scissor.extent.height = viewport.height;
            vkCmdSetScissor(m_CmdBuffer[whichBuffer].m_VkCommandBuffer, 0, 1, &scissor);
        }

        if (passDataProcessor.drawCallback)
        {
            passDataProcessor.drawCallback(whichBuffer, m_CmdBuffer[whichBuffer]);
        }
 
        if (renderingInfo)
        {
            vkCmdEndRenderingKHR(m_CmdBuffer[whichBuffer].m_VkCommandBuffer);
        }

        if (passDataProcessor.postDrawCallback)
        {
            passDataProcessor.postDrawCallback(whichBuffer, m_CmdBuffer[whichBuffer]);
        }
    };

    auto SimpleBufferBarrier = [this](
        CommandListVulkan&   commandList,
        VkBuffer             targetBuffer, 
        VkPipelineStageFlags srcStageMask,
        VkPipelineStageFlags dstStageMask,
        VkAccessFlags        srcAccessMask,
        VkAccessFlags        dstAccessMask)
    {
        VkBufferMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
        memoryBarrier.srcAccessMask = srcAccessMask;
        memoryBarrier.dstAccessMask = dstAccessMask;
        memoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memoryBarrier.buffer = targetBuffer;
        memoryBarrier.offset = 0;
        memoryBarrier.size = VK_WHOLE_SIZE;

        vkCmdPipelineBarrier(
            commandList.m_VkCommandBuffer,
            srcStageMask,
            dstStageMask,
            0,
            0,
            nullptr,
            1,
            &memoryBarrier,
            0,
            nullptr);
    };
    
    auto SimpleImageBarrier = [this](
        CommandListVulkan&           commandList, 
        VkImage                      targetImage,
        VkImageAspectFlags           aspectFlags,
        VkPipelineStageFlags         srcStageMask,
        VkPipelineStageFlags         dstStageMask,
        VkAccessFlags                srcAccessMask,
        VkAccessFlags                dstAccessMask,
        VkImageLayout                fromLayout,
        std::optional<VkImageLayout> toLayout = std::nullopt)
    {
        VkImageMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        memoryBarrier.oldLayout = fromLayout;
        memoryBarrier.newLayout = toLayout ? toLayout.value() : fromLayout;
        memoryBarrier.image = targetImage;
        memoryBarrier.subresourceRange.aspectMask = aspectFlags;
        memoryBarrier.subresourceRange.baseMipLevel = 0;
        memoryBarrier.subresourceRange.levelCount = 1;
        memoryBarrier.subresourceRange.baseArrayLayer = 0;
        memoryBarrier.subresourceRange.layerCount = 1;
        memoryBarrier.srcAccessMask = srcAccessMask;
        memoryBarrier.dstAccessMask = dstAccessMask;
        memoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        vkCmdPipelineBarrier(
            commandList.m_VkCommandBuffer,
            srcStageMask,
            dstStageMask, 
            0, 
            0,
            nullptr,
            0,
            nullptr,
            1,
            &memoryBarrier);
    };
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // DATA PROCESSOR: RP_SCENE //
    ///////////////////////////////////////////////////////////////////////////////////////
    auto sceneDataProcessor = PassDataProcessor(
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        SimpleImageBarrier(
            commandList, 
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_SCENE].colorTileMemory[0].GetVkImage() : m_RenderTargets[RP_SCENE].color[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, 
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        SimpleImageBarrier(
            commandList, 
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_SCENE].colorTileMemory[1].GetVkImage() : m_RenderTargets[RP_SCENE].color[1].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        SimpleImageBarrier(
            commandList, 
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_SCENE].depthTileMemory.GetVkImage() : m_RenderTargets[RP_SCENE].depth.GetVkImage(),
            VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }, 
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        // Scene drawables
        for (const auto& sceneDrawable : m_SceneDrawables)
        {
            AddDrawableToCmdBuffer(sceneDrawable, commandList, whichBuffer);
        }
    }, 
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        const auto& albedoColorTexture = m_RenderTargetUsesTileMemory
            ? m_RenderTargets[RP_SCENE].colorTileMemory[0]
            : m_RenderTargets[RP_SCENE].color[0];

        // Transition image to TRANSFER_SRC -- even when image resides in tile memory
        SimpleImageBarrier(
            commandList,
            albedoColorTexture.GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        SimpleImageBarrier(
            commandList, 
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_SCENE].colorTileMemory[1].GetVkImage() : m_RenderTargets[RP_SCENE].color[1].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        SimpleImageBarrier(
            commandList,
            m_AlbedoColorBufferCopy.GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        {
            VkImageCopy copyRegion{};
            copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.srcOffset      = { 0, 0, 0 };
            copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.dstOffset      = { 0, 0, 0 };
            copyRegion.extent         = { albedoColorTexture.Width, albedoColorTexture.Height, 1 };

            vkCmdCopyImage(
                commandList.m_VkCommandBuffer,
                albedoColorTexture.GetVkImage(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_AlbedoColorBufferCopy.GetVkImage(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &copyRegion);
        }

        // Transition albedo buffer back to SHADER_READ_ONLY so the deferred light pass can sample it
        SimpleImageBarrier(
            commandList,
            albedoColorTexture.GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // Transition m_AlbedoColorBufferCopy so it can be sampled by a future decal rendering pass (unimplemented)
        SimpleImageBarrier(
            commandList,
            m_AlbedoColorBufferCopy.GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        SimpleImageBarrier(
            commandList,
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_SCENE].depthTileMemory.GetVkImage() : m_RenderTargets[RP_SCENE].depth.GetVkImage(),
            VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    });
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // DATA PROCESSOR: CP_LIGHT_CULLING_LIST //
    ///////////////////////////////////////////////////////////////////////////////////////
    auto lightCullingListDataProcessor = PassDataProcessor(
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        SimpleBufferBarrier(
            commandList,
            m_LightIndicesBuffer.GetVkBuffer(),
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT);
        SimpleBufferBarrier(
            commandList,
            m_LightCountsBuffer.GetVkBuffer(),
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT);
    }, 
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        auto& computePassData = m_ComputePassData[CP_LIGHT_CULLING_LIST];

        for (auto& computablePass : computePassData.passComputable->GetPasses())
        {
            computePassData.passComputable->SetDispatchGroupCount(0, 
            { 
                // Dispatch all tiles at once
                m_SceneInfoUniformData.clusterCountX,
                m_SceneInfoUniformData.clusterCountY,
                1 
            });

            computePassData.passComputable->DispatchPass(
                commandList,
                computablePass,
                whichBuffer);
        }
    }, 
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        SimpleBufferBarrier(
            commandList, 
            m_LightIndicesBuffer.GetVkBuffer(),
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 
            VK_ACCESS_MEMORY_WRITE_BIT, 
            VK_ACCESS_MEMORY_READ_BIT);
        SimpleBufferBarrier(
            commandList, 
            m_LightCountsBuffer.GetVkBuffer(),
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT);
        SimpleImageBarrier(
            commandList,
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_SCENE].depthTileMemory.GetVkImage() : m_RenderTargets[RP_SCENE].depth.GetVkImage(),
            VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, // VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    });

    ///////////////////////////////////////////////////////////////////////////////////////
    // DATA PROCESSOR: RP_DEFERRED_LIGHT //
    ///////////////////////////////////////////////////////////////////////////////////////
    auto deferredLightDataProcessor = PassDataProcessor(
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        SimpleImageBarrier(
            commandList, 
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_DEFERRED_LIGHT].colorTileMemory[0].GetVkImage() : m_RenderTargets[RP_DEFERRED_LIGHT].color[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, 
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }, 
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        AddDrawableToCmdBuffer(*m_DeferredLightQuadDrawable, commandList, whichBuffer);
    }, 
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        SimpleImageBarrier(
            commandList,
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_DEFERRED_LIGHT].colorTileMemory[0].GetVkImage() : m_RenderTargets[RP_DEFERRED_LIGHT].color[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    ///////////////////////////////////////////////////////////////////////////////////////
    // DATA PROCESSOR: RP_HUD //
    ///////////////////////////////////////////////////////////////////////////////////////
    auto hudDataProcessor = PassDataProcessor(
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        SimpleImageBarrier(
            commandList, 
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_HUD].colorTileMemory[0].GetVkImage() : m_RenderTargets[RP_HUD].color[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }, 
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        GetGui()->Render(commandList.m_VkCommandBuffer);
    }, 
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        SimpleImageBarrier(
            commandList,
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_HUD].colorTileMemory[0].GetVkImage() : m_RenderTargets[RP_HUD].color[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    ///////////////////////////////////////////////////////////////////////////////////////
    // DATA PROCESSOR: RP_BLIT //
    ///////////////////////////////////////////////////////////////////////////////////////
    auto blitDataProcessor = PassDataProcessor(
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        SimpleImageBarrier(
            commandList, 
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_BLIT].colorTileMemory[0].GetVkImage() : m_RenderTargets[RP_BLIT].color[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }, 
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        AddDrawableToCmdBuffer(*m_BlitQuadDrawable.get(), commandList, whichBuffer);
    }, 
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        SimpleImageBarrier(
            commandList,
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_BLIT].colorTileMemory[0].GetVkImage() : m_RenderTargets[RP_BLIT].color[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    });

    ///////////////////////////////////////////////////////////////////////////////////////
    // RECORD COMMANDS //
    ///////////////////////////////////////////////////////////////////////////////////////

    if (!m_CmdBuffer[whichBuffer].Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
    {
        return;
    }

    /*
    * Here we bind the tile memory heap object, ensuring it will be used for any subsequent command
    * Since all tile memory comes from a single pool, and we are not micro-managing this pool (we allocate 
    * attachments until nothing else fits), we can grab the memory allocation from any of the attachments,
    * here we choose the scene color one since it's the first we attempt to allocate on tile memory (and
    * consequentially, the one that will always successfully be placed there)
    */
    if (m_IsTileMemoryHeapSupported && m_RenderTargetUsesTileMemory && !m_RenderTargets[RP_SCENE].colorTileMemory.empty())
    {
        const auto& imageAllocation = m_RenderTargets[RP_SCENE].colorTileMemory[0].Image.GetMemoryAllocation();
        VkTileMemoryBindInfoQCOM tileMemoryBindInfo{ VK_STRUCTURE_TYPE_TILE_MEMORY_BIND_INFO_QCOM };
        tileMemoryBindInfo.memory = pVulkan->GetMemoryManager().GetVkDeviceMemory(imageAllocation);

        vkCmdBindTileMemoryQCOM(m_CmdBuffer[whichBuffer].m_VkCommandBuffer, &tileMemoryBindInfo);
    }

    RecordPassData(
        std::optional<std::pair<RENDER_PASS, RenderingAttachmentInfoGroup>>({ RP_SCENE , RenderingAttachmentInfoGroup(
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_SCENE].colorTileMemory : m_RenderTargets[RP_SCENE].color,
            m_RenderTargetUsesTileMemory ? &m_RenderTargets[RP_SCENE].depthTileMemory : &m_RenderTargets[RP_SCENE].depth,
            RenderPassInputUsage::Clear,
            RenderPassOutputUsage::Store,
            true,
            RenderPassOutputUsage::Store) 
        }),
        PassRecordData(m_RenderPassData[RP_SCENE], std::move(sceneDataProcessor)));

    RecordPassData(
        std::nullopt,
        PassRecordData(m_ComputePassData[CP_LIGHT_CULLING_LIST], std::move(lightCullingListDataProcessor)));

    RecordPassData(
        std::optional<std::pair<RENDER_PASS, RenderingAttachmentInfoGroup>>({ RP_DEFERRED_LIGHT , RenderingAttachmentInfoGroup(
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_DEFERRED_LIGHT].colorTileMemory : m_RenderTargets[RP_DEFERRED_LIGHT].color,
            nullptr,
            RenderPassInputUsage::Clear,
            RenderPassOutputUsage::Store) 
        }),
        PassRecordData(m_RenderPassData[RP_DEFERRED_LIGHT], std::move(deferredLightDataProcessor)));

    RecordPassData(
        std::optional<std::pair<RENDER_PASS, RenderingAttachmentInfoGroup>>({ RP_HUD , RenderingAttachmentInfoGroup(
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_HUD].colorTileMemory : m_RenderTargets[RP_HUD].color,
            nullptr,
            RenderPassInputUsage::Clear,
            RenderPassOutputUsage::Store)
            }),
        PassRecordData(m_RenderPassData[RP_HUD], std::move(hudDataProcessor)));

    RecordPassData(
        std::optional<std::pair<RENDER_PASS, RenderingAttachmentInfoGroup>>({ RP_BLIT , RenderingAttachmentInfoGroup(
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_BLIT].colorTileMemory : m_RenderTargets[RP_BLIT].color,
            nullptr,
            RenderPassInputUsage::Clear,
            RenderPassOutputUsage::Store)
            }),
        PassRecordData(m_RenderPassData[RP_BLIT], std::move(blitDataProcessor)));

    m_CmdBuffer[whichBuffer].End();

    m_CmdBuffer[whichBuffer].QueueSubmit(
        pWaitSemaphores,
        defaultGfxWaitDstStageMasks,
        { &m_Semaphore,1 });

    pWaitSemaphores = { &m_Semaphore,1 };

    ///////////////////////////////////////////////////////////////////////////////////////
    // SWAPCHAIN COPY //
    ///////////////////////////////////////////////////////////////////////////////////////
    {
        const auto& sourceImage = m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_BLIT].colorTileMemory[0] : m_RenderTargets[RP_BLIT].color[0];
        const auto swapchainImage = pVulkan->GetSwapchainImage(currentVulkanBuffer.swapchainPresentIdx);
        
        m_SwapchainCopyCmdBuffer[whichBuffer].Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

        ////////////////////////////////////
        // PREPARE SWAPCHAIN FOR TRANSFER //
        ////////////////////////////////////
        {
            SimpleImageBarrier(
                m_SwapchainCopyCmdBuffer[whichBuffer],
                swapchainImage,
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        }

        /////////////////////////
        // BLIT INTO SWAPCHAIN //
        /////////////////////////

        VkImageBlit blitRegion{};
        blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blitRegion.srcSubresource.baseArrayLayer = 0;
        blitRegion.srcSubresource.layerCount = 1;
        blitRegion.srcSubresource.mipLevel = 0;
        blitRegion.srcOffsets[1].x = sourceImage.Width;
        blitRegion.srcOffsets[1].y = sourceImage.Height;
        blitRegion.srcOffsets[1].z = 1;
        blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blitRegion.dstSubresource.baseArrayLayer = 0;
        blitRegion.dstSubresource.layerCount = 1;
        blitRegion.dstSubresource.mipLevel = 0;
        blitRegion.dstOffsets[1].x = pVulkan->m_SurfaceWidth;
        blitRegion.dstOffsets[1].y = pVulkan->m_SurfaceHeight;
        blitRegion.dstOffsets[1].z = 1;
        
        vkCmdBlitImage(
            m_SwapchainCopyCmdBuffer[whichBuffer].m_VkCommandBuffer,
            m_RenderTargetUsesTileMemory ? m_RenderTargets[RP_BLIT].colorTileMemory[0].GetVkImage() : m_RenderTargets[RP_BLIT].color[0].GetVkImage(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            swapchainImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &blitRegion,
            VK_FILTER_NEAREST);

        ///////////////////////////////////
        // PREPARE SWAPCHAIN FOR PRESENT //
        ///////////////////////////////////
        {
            SimpleImageBarrier(
                m_SwapchainCopyCmdBuffer[whichBuffer],
                swapchainImage,
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_MEMORY_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        }

        m_SwapchainCopyCmdBuffer[whichBuffer].End();

        ///////////////////////////
        // SUBMIT SWAPCHAIN COPY //
        ///////////////////////////

        m_SwapchainCopyCmdBuffer[whichBuffer].QueueSubmit(
            pWaitSemaphores,
            defaultGfxWaitDstStageMasks,
            { &m_SwapchhainCopySemaphore,1 },
            currentVulkanBuffer.fence);

        pWaitSemaphores = { &m_SwapchhainCopySemaphore,1 };
    }

    // Queue is loaded up, tell the driver to start processing
    pVulkan->PresentQueue(pWaitSemaphores, currentVulkanBuffer.swapchainPresentIdx);
}
