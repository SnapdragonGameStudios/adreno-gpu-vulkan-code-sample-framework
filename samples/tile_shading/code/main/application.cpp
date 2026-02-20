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
* TILE SHADING SAMPLE
============================================================================================================
*
* This sample demonstrates a light clustering algorithm using Vulkan, with specific support for the
* VK_QCOM_tile_shading extension. This extension allows the application to leverage tile-based deferred rendering
* (TBDR) for efficient memory management and rendering performance.
*
* Reference points:
*
* 1. PreInitializeSetVulkanConfiguration
*    - Sets up the Vulkan configuration, including required and optional extensions.
*    - config.OptionalExtension<ExtensionLib::Ext_VK_QCOM_tile_shading>();
*    - The application checks for the VK_QCOM_tile_shading extension and sets up the Vulkan configuration accordingly.
*    - If supported, tile shading is enabled at startup.
* 
* 2. Initialize
*    - Initializes various components of the application, including camera, lights, shaders, buffers, render targets, GUI, and mesh objects.
*    - m_IsTileShadingSupported &= GetVulkan()->HasLoadedVulkanDeviceExtension(VK_QCOM_TILE_SHADING_EXTENSION_NAME);
*    - m_IsTileShadingActive = m_IsTileShadingSupported;
* 
* 3. CreateComputables
*    - Creates computable objects for light culling, including tile shading variants.
*    - m_TileShadingPassData.passComputable->DispatchPass(...);
* 
* 4. Render
*    - The main rendering function binds the tile shading extension to the command buffer if tile shading is enabled.
*    - The rendering process includes multiple passes:
*      - Scene Pass: Outputs albedo, normal, and depth.
*      - Light Culling Pass: Divides point lights into clusters and performs light culling.
*      - Deferred Light Pass: Applies the corresponding pixel light cluster to a screen quad.
*    - vkCmdBeginPerTileExecutionQCOM and vkCmdEndPerTileExecutionQCOM are used for the compute dispatch.
*
* SPEC: https://registry.khronos.org/vulkan/specs/latest/man/html/VK_QCOM_tile_shading.html
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
}

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
    config.OptionalExtension<ExtensionLib::Ext_VK_QCOM_tile_shading>();
}

//-----------------------------------------------------------------------------
bool Application::Initialize(uintptr_t windowHandle, uintptr_t hInstance)
//-----------------------------------------------------------------------------
{
    if (!ApplicationHelperBase::Initialize( windowHandle, hInstance ))
    {
        return false;
    }

    m_IsTilePropertiesSupported &= GetVulkan()->HasLoadedVulkanDeviceExtension(VK_QCOM_TILE_PROPERTIES_EXTENSION_NAME);
    m_IsTileShadingSupported &= GetVulkan()->HasLoadedVulkanDeviceExtension(VK_QCOM_TILE_SHADING_EXTENSION_NAME);

    // Begin with tile shading active (so we can gather all the necessary tile info at startup), if supported
    m_IsTileShadingActive = m_IsTileShadingSupported;

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
        for (auto& cmdBuffer : m_RenderPassData[whichPass].passCmdBuffer)
        {
            cmdBuffer.Release();
        }
        
        for (auto& cmdBuffer : m_TileShadingPassData.passCmdBuffer)
        {
            cmdBuffer.Release();
        }

        m_DefaultRenderTargets[whichPass].Release();
    }
    m_TileShadingSceneRenderTarget.Release();

    for(auto& swapchainCopyCmdBuffer : m_SwapchainCopyCmdBuffer)
    {
        swapchainCopyCmdBuffer.Release();
    }

    // Render passes / Semaphores
    for (int whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        vkDestroySemaphore(pVulkan->m_VulkanDevice, m_RenderPassData[whichPass].passSemaphore, nullptr);
    }
    vkDestroySemaphore(pVulkan->m_VulkanDevice, m_TileShadingPassData.passSemaphore, nullptr);

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

    for(int i=0; i< m_LightUniformData.size(); i++)
    {
        auto& lightData              = m_LightUniformData[i];
        const float lightIndexFactor = static_cast<float>(i) / static_cast<float>(m_LightUniformData.size()) * 0.8f + 0.2f;

        const glm::vec2 direction     = glm::normalize(glm::vec2(sceneRadiusDist(gen), sceneRadiusDist(gen)));
        const glm::vec2 planePosition = direction * sceneRadiusDist(gen) * lightIndexFactor;
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

    auto LoadShader = [&](const std::string& id, const std::string& filename) -> bool
    {
        if (!m_ShaderManager->AddShader(*m_AssetManager, id, filename, SHADER_DESTINATION_PATH))
        {
            LOGE("Error Loading shader %s from %s", id.c_str(), filename.c_str());
            LOGI("Please verify if you have all required assets on the sample media folder");
            LOGI("If you are running on Android, don't forget to run the `02_CopyMediaToDevice.bat` script to copy all media files into the device memory");
            return false;
        }

        return true;
    };

    typedef std::pair<std::string, std::string> tIdAndFilename;
    for (const tIdAndFilename& i :
            { tIdAndFilename { "Blit",  "Blit.json" },
              tIdAndFilename { "SceneOpaque", "SceneOpaque.json" },
              tIdAndFilename { "SceneTransparent", "SceneTransparent.json" },
              tIdAndFilename { "LightCulling", "LightCulling.json" },
              tIdAndFilename { "DeferredLight", "DeferredLight.json" }
            })
    {
        if (!LoadShader(i.first, i.second))
        {
            return false;
        }
    }

    if (m_IsTileShadingSupported)
    {
        if (!LoadShader("LightCullingTileShading", "LightCullingTileShading.json") 
        || !LoadShader("LightCullingTileShadingAreaDispatch", "LightCullingTileShadingAreaDispatch.json") 
        || !LoadShader("DeferredLightTileShading", "DeferredLightTileShading.json"))
        {
            return false;
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::CreateRenderTargets()
//-----------------------------------------------------------------------------
{
    auto pVulkan = GetVulkan();

    LOGI("**************************");
    LOGI("Creating Render Targets...");
    LOGI("**************************");

    TextureFormat vkDesiredDepthFormat = pVulkan->GetBestSurfaceDepthFormat();
    TextureFormat desiredDepthFormat = vkDesiredDepthFormat;

    const TextureFormat SceneColorType[] = { TextureFormat::R8G8B8A8_SRGB, TextureFormat::R8G8B8A8_SRGB };
    const TextureFormat DeferredLightColorType[] = { TextureFormat::R8G8B8A8_SRGB};
    const TextureFormat HudColorType[]  = { TextureFormat::R8G8B8A8_SRGB };
    const TextureFormat FinalColorType[]  = { TextureFormat::R8G8B8A8_SRGB };
    const TextureFormat TileShadingSceneColorType[] = 
    { 
        TextureFormat::R8G8B8A8_SRGB, // Diffuse
        TextureFormat::R8G8B8A8_SRGB, // Normal
        TextureFormat::R8G8B8A8_SRGB, // Scene
    };

    if (!m_DefaultRenderTargets[RP_SCENE].Initialize(
        pVulkan, 
        RenderTargetInitializeInfo
        {
            gRenderWidth,
            gRenderHeight,
            SceneColorType,
            desiredDepthFormat,
            nullptr,
            {}, // std::span<const TEXTURE_TYPE>
        },
        "Scene RT"))
    {
        LOGE("Unable to create scene render target");
        return false;
    }
    
    if (!m_DefaultRenderTargets[RP_DEFERRED_LIGHT].Initialize(
        pVulkan, 
        RenderTargetInitializeInfo
        {
            gRenderWidth,
            gRenderHeight,
            DeferredLightColorType,
            TextureFormat::UNDEFINED,
            nullptr,
            {}, // std::span<const TEXTURE_TYPE>
        },
        "Particles RT"))
    {
        LOGE("Unable to create the deferred light render target");
        return false;
    }

    if (!m_DefaultRenderTargets[RP_HUD].Initialize(
        pVulkan, 
        RenderTargetInitializeInfo
        {
            gSurfaceWidth,
            gSurfaceHeight,
            HudColorType,
            TextureFormat::UNDEFINED,
            nullptr,
            {}, // std::span<const TEXTURE_TYPE>
        },
        "HUD RT"))
    {
        LOGE("Unable to create hud render target");
        return false;
    }

    std::array<TEXTURE_TYPE, 1> blitColorTypes;
    blitColorTypes[0] = TT_RENDER_TARGET_TRANSFERSRC;
    
    if (!m_DefaultRenderTargets[RP_BLIT].Initialize(
        pVulkan, 
        RenderTargetInitializeInfo
        {
            gSurfaceWidth,
            gSurfaceHeight,
            FinalColorType,
            TextureFormat::UNDEFINED,
            nullptr,
            blitColorTypes,
        },
        "BLIT RT"))
    {
        LOGE("Unable to create blit render target");
        return false;
    }

    if (m_IsTileShadingSupported)
    {
        std::array<TEXTURE_TYPE, 3> tileShadingSceneColorTypes{ TT_RENDER_TARGET_LOCAL_READ_TRANSIENT , TT_RENDER_TARGET_LOCAL_READ_TRANSIENT , TT_RENDER_TARGET };
        if (!m_TileShadingSceneRenderTarget.Initialize(
            pVulkan, 
            RenderTargetInitializeInfo
            {
                gRenderWidth,
                gRenderHeight,
                TileShadingSceneColorType,
                desiredDepthFormat,
                nullptr,
                tileShadingSceneColorTypes, 
                TT_DEPTH_TARGET_LOCAL_READ,
            },
            "Tile Shading Scene RT"))
        {
            LOGE("Unable to create tile shading scene render target");
            return false;
        }
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
        for (auto& colorAttachment : m_DefaultRenderTargets[whichPass].GetColorAttachments())
        {
            colorFormats.push_back(colorAttachment.Format);
        }

        m_RenderPassData[whichPass].renderContext =
        { 
            colorFormats, 
            m_DefaultRenderTargets[whichPass].GetDepthFormat(),
            TextureFormat::UNDEFINED, 
            sRenderPassNames[whichPass]
        };
    }

    {
        std::vector< TextureFormat> colorFormats;
        for (auto& colorAttachment : m_TileShadingSceneRenderTarget.GetColorAttachments())
        {
            colorFormats.push_back(colorAttachment.Format);
        }

        m_TileShadingPassData.renderContext =
        {
            colorFormats,
            m_TileShadingSceneRenderTarget.GetDepthFormat(),
            TextureFormat::UNDEFINED,
            "RP_SCENE"
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
    const auto& hudRenderTarget = m_DefaultRenderTargets[RP_HUD];
    m_Gui = std::make_unique<GuiImguiGfx>(*GetVulkan());
    if (!m_Gui->Initialize(
        windowHandle,
        hudRenderTarget.m_ColorAttachments[0].Format,
        hudRenderTarget.m_ColorAttachments[0].Width,
        hudRenderTarget.m_ColorAttachments[0].Height))
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
    const auto* deferredLightTileShadingShader = m_ShaderManager->GetShader("DeferredLightTileShading");
    const auto* pBlitQuadShader                = m_ShaderManager->GetShader("Blit");
    if (!pSceneOpaqueShader || !pSceneTransparentShader || !pBlitQuadShader || !deferredLightShader || (m_IsTileShadingSupported && !deferredLightTileShadingShader))
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


    const auto loaderFlags = 0; // No instancing
    const bool ignoreTransforms = (loaderFlags & DrawableLoader::LoaderFlags::IgnoreHierarchy) != 0;

    const auto sceneAssetPath = std::filesystem::path(MESH_DESTINATION_PATH).append(gSceneAssetModel).string();
    MeshLoaderModelSceneSanityCheck meshSanityCheckProcessor(sceneAssetPath);
    MeshObjectIntermediateGltfProcessor meshObjectProcessor(sceneAssetPath, ignoreTransforms, glm::vec3(1.0f, 1.0f, 1.0f));
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
                    return { &m_DefaultRenderTargets[RP_DEFERRED_LIGHT].GetColorAttachments()[0] };
                }
                else if (texName == "Overlay")
                {
                    return { &m_DefaultRenderTargets[RP_HUD].GetColorAttachments()[0] };
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
                    return { &m_DefaultRenderTargets[RP_SCENE].GetColorAttachments()[0] };
                }
                else if (texName == "SceneNormal")
                {
                    return { &m_DefaultRenderTargets[RP_SCENE].GetColorAttachments()[1] };
                }
                else if (texName == "SceneDepth")
                {
                    return { &m_DefaultRenderTargets[RP_SCENE].GetDepthAttachment() };
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

        if (m_IsTileShadingSupported)
        {
            auto deferredLightTileShadingQuadShaderMaterial = m_MaterialManager->CreateMaterial(*deferredLightTileShadingShader, NUM_VULKAN_BUFFERS,
                [this](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
                {
                    if (texName == "SceneColor")
                    {
                        return { &m_TileShadingSceneRenderTarget.GetColorAttachments()[0] };
                    }
                    else if (texName == "SceneNormal")
                    {
                        return { &m_TileShadingSceneRenderTarget.GetColorAttachments()[1] };
                    }
                    else if (texName == "SceneDepth")
                    {
                        return { &m_TileShadingSceneRenderTarget.GetDepthAttachment() };
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
        
            Mesh tileShadingBlitQuadMesh;
            MeshHelper::CreateScreenSpaceMesh(pVulkan->GetMemoryManager(), 0, &tileShadingBlitQuadMesh);

            m_DeferredLightTileShadingQuadDrawable = std::make_unique<Drawable>(*pVulkan, std::move(deferredLightTileShadingQuadShaderMaterial));
            if (!m_DeferredLightTileShadingQuadDrawable->Init(m_TileShadingPassData.renderContext, std::move(tileShadingBlitQuadMesh)))
            {
                return false;
            }
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::CreateComputables()
//-----------------------------------------------------------------------------
{
    auto pVulkan = GetVulkan();

    const auto lightCullingShader                        = m_ShaderManager->GetShader("LightCulling");
    const auto lightCullingTileShadingShader             = m_ShaderManager->GetShader("LightCullingTileShading");
    const auto lightCullingTileShadingShaderAreaDispatch = m_ShaderManager->GetShader("LightCullingTileShadingAreaDispatch");
    if (!lightCullingShader || (m_IsTileShadingSupported && (!lightCullingTileShadingShader || !lightCullingTileShadingShaderAreaDispatch)))
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
                return { &m_DefaultRenderTargets[RP_SCENE].GetDepthAttachment() };
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
    
    if (m_IsTileShadingSupported)
    {
        auto materialTileShading = m_MaterialManager->CreateMaterial(
            *lightCullingTileShadingShader,
            NUM_VULKAN_BUFFERS,
            [&](const std::string& texName) -> MaterialManager::tPerFrameTexInfo 
            {
                if (texName == "SceneDepth")
                {
                    return { &m_TileShadingSceneRenderTarget.GetDepthAttachment() };
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

        m_TileShadingPassData.passComputable = std::make_unique<Computable>(*pVulkan, std::move(materialTileShading));
        if (!m_TileShadingPassData.passComputable->Init())
        {
            LOGE("Error Creating SGSR computables...");
        }

        auto materialTileShadingAreaDispatch = m_MaterialManager->CreateMaterial(
            *lightCullingTileShadingShaderAreaDispatch,
            NUM_VULKAN_BUFFERS,
            [&](const std::string& texName) -> MaterialManager::tPerFrameTexInfo 
            {
                if (texName == "SceneDepth")
                {
                    return { &m_TileShadingSceneRenderTarget.GetDepthAttachment() };
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

        m_TileShadingPassData.passComputableAreaDispatch = std::make_unique<Computable>(*pVulkan, std::move(materialTileShadingAreaDispatch));
        if (!m_TileShadingPassData.passComputableAreaDispatch->Init())
        {
            LOGE("Error Creating SGSR computables...");
        }
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
    for (uint32_t whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        for (uint32_t whichBuffer = 0; whichBuffer < m_RenderPassData[whichPass].passCmdBuffer.size(); whichBuffer++)
        {
            // The Pass Command Buffer => Primary
            sprintf(szName, "Primary (%s; Buffer %d of %d)", GetPassName(whichPass), whichBuffer + 1, NUM_VULKAN_BUFFERS);
            if (!m_RenderPassData[whichPass].passCmdBuffer[whichBuffer].Initialize(pVulkan, szName, CommandListBase::Type::Primary))
            {
                return false;
            }
        }
    }

    for (uint32_t whichBuffer = 0; whichBuffer < m_TileShadingPassData.passCmdBuffer.size(); whichBuffer++)
    {
        // The Pass Command Buffer => Primary
        sprintf(szName, "Primary (Buffer %d of %d)", whichBuffer + 1, NUM_VULKAN_BUFFERS);
        if (!m_TileShadingPassData.passCmdBuffer[whichBuffer].Initialize(pVulkan, szName, CommandListBase::Type::Primary))
        {
            return false;
        }
    }

    for (uint32_t whichPass = 0; whichPass < NUM_COMPUTE_PASSES; whichPass++)
    {
        for (uint32_t whichBuffer = 0; whichBuffer < m_ComputePassData[whichPass].passCmdBuffer.size(); whichBuffer++)
        {
            // The Pass Command Buffer => Primary
            sprintf(szName, "Primary (%s; Buffer %d of %d)", GetPassName(whichPass), whichBuffer + 1, NUM_VULKAN_BUFFERS);
            if (!m_ComputePassData[whichPass].passCmdBuffer[whichBuffer].Initialize(pVulkan, szName, CommandListBase::Type::Primary))
            {
                return false;
            }
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

    for (uint32_t whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        VkResult retVal = vkCreateSemaphore(GetVulkan()->m_VulkanDevice, &SemaphoreInfo, NULL, &m_RenderPassData[whichPass].passSemaphore);
        if (!CheckVkError("vkCreateSemaphore()", retVal))
        {
            return false;
        }
    }
    
    {
        VkResult retVal = vkCreateSemaphore(GetVulkan()->m_VulkanDevice, &SemaphoreInfo, NULL, &m_TileShadingPassData.passSemaphore);
        if (!CheckVkError("vkCreateSemaphore()", retVal))
        {
            return false;
        }
    }
    
    for (uint32_t whichPass = 0; whichPass < NUM_COMPUTE_PASSES; whichPass++)
    {
        VkResult retVal = vkCreateSemaphore(GetVulkan()->m_VulkanDevice, &SemaphoreInfo, NULL, &m_ComputePassData[whichPass].passSemaphore);
        if (!CheckVkError("vkCreateSemaphore()", retVal))
        {
            return false;
        }
    }

    VkResult retVal = vkCreateSemaphore(GetVulkan()->m_VulkanDevice, &SemaphoreInfo, NULL, &m_SwapchhainCopySemaphore);
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

        if (ImGui::Begin("FPS", (bool*)nullptr, ImGuiWindowFlags_NoTitleBar))
        {
            if(ImGui::CollapsingHeader("Scene Details"))
            {
                ImGui::Text("FPS: %.1f", m_CurrentFPS);
                ImGui::Text("Camera [%f, %f, %f]", m_Camera.Position().x, m_Camera.Position().y, m_Camera.Position().z);
                ImGui::Checkbox("Limit Values to Adreno Values", &m_CapMaxValues);
            }
            
            if(ImGui::CollapsingHeader("Clustered Deferred Light", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::BeginDisabled(!m_IsTileShadingSupported);
                ImGui::Checkbox("Tile Shading Enabled", &m_IsTileShadingActive);
                ImGui::EndDisabled();

                ImGui::BeginDisabled(!m_IsTileShadingSupported || !m_IsTileShadingActive || true);
                ImGui::Checkbox("Area Dispatch", &m_IsAreaDispatchActive);
                ImGui::EndDisabled();

                ImGui::Separator();

                bool ignoreLightTiles = static_cast<bool>(m_SceneInfoUniformData.ignoreLightTiles);
                ImGui::Checkbox("Deactivate Clustering", &ignoreLightTiles);
                m_SceneInfoUniformData.ignoreLightTiles = static_cast<int32_t>(ignoreLightTiles);

                bool debugShaders = static_cast<bool>(m_SceneInfoUniformData.debugShaders);
                ImGui::Checkbox("Visualize Clusters", &debugShaders);
                m_SceneInfoUniformData.debugShaders = static_cast<int32_t>(debugShaders);

                ImGui::Separator();

                ImGui::DragInt("Light Count", &m_SceneInfoUniformData.lightCount, 1.0f, 0, m_CapMaxValues ? 64 : MAX_LIGHT_COUNT);
                m_SceneInfoUniformData.lightCount = std::min(m_SceneInfoUniformData.lightCount, m_CapMaxValues ? 64 : MAX_LIGHT_COUNT);

                ImGui::Separator();

                std::array<int32_t, 1> clusterSize
                {
                    static_cast<int32_t>(m_SceneInfoUniformData.clusterSizeX)
                };

                ImGui::DragInt("Cluster Size", clusterSize.data(), 8.0f, 32, m_CapMaxValues ? 96 : 192);
                m_SceneInfoUniformData.clusterSizeX = static_cast<uint32_t>(clusterSize[0]);
                m_SceneInfoUniformData.clusterSizeY = static_cast<uint32_t>(clusterSize[0]);

                // Calculate the cluster count
                if(m_IsTileShadingActive)
                {
                    m_SceneInfoUniformData.binClusterCountX    = static_cast<uint32_t>(std::ceil((static_cast<float>(m_BinSize.x) / static_cast<float>(m_SceneInfoUniformData.clusterSizeX))));
                    m_SceneInfoUniformData.binClusterCountY    = static_cast<uint32_t>(std::ceil((static_cast<float>(m_BinSize.y) / static_cast<float>(m_SceneInfoUniformData.clusterSizeY))));
                    m_SceneInfoUniformData.globalClusterCountX = m_SceneInfoUniformData.binClusterCountX * m_SceneInfoUniformData.binCountX;
                    m_SceneInfoUniformData.globalClusterCountY = m_SceneInfoUniformData.binClusterCountY * m_SceneInfoUniformData.binCountY;
                }
                else
                {
                    if (m_IsTileShadingSupported)
                    {
                        /*
                        * Use the same amount of clusters as the tile shading variant, so comparissions are fully compatible (aples to aples), otherwise
                        * the non-tile path could potentially trigger less dispatches than the tile shading one, resulting in better performance. The 
                        * reason this is done is: Using the cluster size as the base variable hinders our ability of fitting clusters perfectly according
                        * to the binning/screen area, ideally we should use bin count (which is then used to determine the optimal cluster size). This is
                        * a limitation on how the sample was prepared and not related to the extension itself.
                        */
                        m_SceneInfoUniformData.globalClusterCountX = static_cast<uint32_t>(std::ceil((static_cast<float>(m_BinSize.x) / static_cast<float>(m_SceneInfoUniformData.clusterSizeX)))) * m_SceneInfoUniformData.binCountX;
                        m_SceneInfoUniformData.globalClusterCountY = static_cast<uint32_t>(std::ceil((static_cast<float>(m_BinSize.y) / static_cast<float>(m_SceneInfoUniformData.clusterSizeY)))) * m_SceneInfoUniformData.binCountY;
                    }
                    else
                    {
                        m_SceneInfoUniformData.globalClusterCountX = (m_SceneInfoUniformData.viewportWidth + (m_SceneInfoUniformData.clusterSizeX - 1)) / m_SceneInfoUniformData.clusterSizeX;
                        m_SceneInfoUniformData.globalClusterCountY = (m_SceneInfoUniformData.viewportHeight + (m_SceneInfoUniformData.clusterSizeY - 1)) / m_SceneInfoUniformData.clusterSizeY;
                    }

                    m_SceneInfoUniformData.binClusterCountX = m_SceneInfoUniformData.globalClusterCountX;
                    m_SceneInfoUniformData.binClusterCountY = m_SceneInfoUniformData.globalClusterCountY;
                }

                // Disable tile memory if number of bins is too high and tile shading isn't active
                // If tile shading is active, we don't really care about the number of bins, because each time one
                // executed via tile shading dispatch, it will fill the same memory buffer area, because of that
                // we only need a buffer that fits the light cluster total, which is pretty much a guarantee on
                // any device that supports tile memory
                if (m_SceneInfoUniformData.binCountX * m_SceneInfoUniformData.binCountY > MAX_TILE_MEMORY_SUPPORTED_TILES && !m_IsTileShadingActive)
                {
                    m_SceneInfoUniformData.tileMemoryEnabled = false;
                }

                ImGui::Text("Bin Count: [%d, %d] -> T %d", 
                    m_IsTileShadingActive ? m_SceneInfoUniformData.binCountX : 1, 
                    m_IsTileShadingActive ? m_SceneInfoUniformData.binCountY : 1,
                    m_IsTileShadingActive ? m_SceneInfoUniformData.binCountX * m_SceneInfoUniformData.binCountY : 1);

                ImGui::Text(
                    "Total Clusters Per Bin: [%d, %d] -> T %d",
                    m_SceneInfoUniformData.binClusterCountX,
                    m_SceneInfoUniformData.binClusterCountY,
                    m_SceneInfoUniformData.binClusterCountX * m_SceneInfoUniformData.binClusterCountY);

                ImGui::Text(
                    "Total Clusters: [%d, %d] -> T %d",
                    m_SceneInfoUniformData.globalClusterCountX,
                    m_SceneInfoUniformData.globalClusterCountY,
                    m_SceneInfoUniformData.globalClusterCountX * m_SceneInfoUniformData.globalClusterCountY);
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
    bool success = true;

    // Select which render target should be used for the screen blit operation depending if tile shading 
    // is or isn't active
    success &= m_BlitQuadDrawable->GetMaterial().UpdateDescriptorSetBinding(whichBuffer, "Scene", 
        m_IsTileShadingActive 
            ? m_TileShadingSceneRenderTarget.GetColorAttachments()[2]
            : m_DefaultRenderTargets[RP_DEFERRED_LIGHT].GetColorAttachments()[0]);

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

    m_SceneInfoUniformData.viewportWidth  = gRenderWidth;
    m_SceneInfoUniformData.viewportHeight = gRenderHeight;

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

    auto RefreshBinInformation = [&](const VkRenderingInfo* renderingInfo)
    {
        if (m_IsTilePropertiesSupported)
        {
            VkTilePropertiesQCOM tileProperties{ VK_STRUCTURE_TYPE_TILE_PROPERTIES_QCOM };
            if (vkGetDynamicRenderingTilePropertiesQCOM(GetVulkan()->m_VulkanDevice, renderingInfo, &tileProperties) == VK_SUCCESS)
            {
                auto CalculateTileCount = [&tileProperties](VkExtent2D renderArea) -> std::pair<uint32_t, uint32_t>
                {
                        uint32_t tilesX = static_cast<uint32_t>(std::ceil(static_cast<float>(renderArea.width) / tileProperties.tileSize.width));
                        uint32_t tilesY = static_cast<uint32_t>(std::ceil(static_cast<float>(renderArea.height) / tileProperties.tileSize.height));

                        return std::make_pair(tilesX, tilesY);
                };
                
                const auto [binCountX, binCountY] = CalculateTileCount(renderingInfo->renderArea.extent);

                m_SceneInfoUniformData.binCountX = binCountX;
                m_SceneInfoUniformData.binCountY = binCountY;
                m_BinSize.x = tileProperties.tileSize.width;
                m_BinSize.y = tileProperties.tileSize.height;
            }
        }
    };

    using RecordRenderingInfo = std::optional<std::pair<::RenderContext<Vulkan>*, RenderingAttachmentInfoGroup>>;
    auto RecordPassData = [this, whichBuffer, defaultGfxWaitDstStageMasks, &pWaitSemaphores, &RefreshBinInformation](
        RecordRenderingInfo renderingInfo,
        PassRecordData      passRecordInfo,
        bool                isTileShadingPass = false,
        VkFence             completionFence   = VK_NULL_HANDLE)
    {
        auto& [passData, passDataProcessor] = passRecordInfo;
        auto& commandBuffer = passData.get().passCmdBuffer[whichBuffer];

        if (!commandBuffer.Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        {
            return;
        }

        if (passDataProcessor.preDrawCallback)
        {
            passDataProcessor.preDrawCallback(whichBuffer, commandBuffer);
        }

        if (renderingInfo)
        {
            const auto& [renderContext, renderingAttachmentGroup] = renderingInfo.value();
            const auto renderPassInfoRef = renderContext->GetRenderingInfo(renderingAttachmentGroup);
            VkRenderingInfo renderPassInfo = renderPassInfoRef.get();
            const auto renderArea = renderPassInfo.renderArea;

            VkRenderPassTileShadingCreateInfoQCOM tileShadingCreateInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_TILE_SHADING_CREATE_INFO_QCOM };
            tileShadingCreateInfo.flags = VK_TILE_SHADING_RENDER_PASS_ENABLE_BIT_QCOM /* | VK_TILE_SHADING_RENDER_PASS_PER_TILE_EXECUTION_BIT_QCOM*/;
            tileShadingCreateInfo.tileApronSize = VkExtent2D{0, 0};

            if (isTileShadingPass)
            {
                renderPassInfo.pNext = &tileShadingCreateInfo;
            }

            vkCmdBeginRenderingKHR(commandBuffer, &renderPassInfo);

            // For simplicity, since refresh the bin information after beginning a tile-shading rendering context
            // (ideally this should be done before starting the rendering context, as the information will lag
            // one frame, but for a sample it's good enough)
            if (isTileShadingPass)
            {
                RefreshBinInformation(&renderPassInfo);
            }
            
            VkViewport viewport;
            viewport.x = renderPassInfo.renderArea.offset.x;
            viewport.y = renderPassInfo.renderArea.offset.y;
            viewport.width = renderPassInfo.renderArea.extent.width;
            viewport.height = renderPassInfo.renderArea.extent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            VkRect2D scissor;
            scissor.offset.x = viewport.x;
            scissor.offset.y = viewport.y;
            scissor.extent.width = viewport.width;
            scissor.extent.height = viewport.height;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        }

        if (passDataProcessor.drawCallback)
        {
            passDataProcessor.drawCallback(whichBuffer, commandBuffer);
        }
 
        if (renderingInfo)
        {
            vkCmdEndRenderingKHR(commandBuffer);
        }

        if (passDataProcessor.postDrawCallback)
        {
            passDataProcessor.postDrawCallback(whichBuffer, commandBuffer);
        }

        commandBuffer.End();

        commandBuffer.QueueSubmit(
            pWaitSemaphores,
            defaultGfxWaitDstStageMasks,
            { &passData.get().passSemaphore,1 },
            completionFence);

        pWaitSemaphores = { &passData.get().passSemaphore,1 };
    };

    auto SimpleBufferBarrier = [this](
        CommandListVulkan& commandList,
        VkBuffer           targetBuffer, 
        VkPipelineStageFlags2 srcStageMask,
        VkPipelineStageFlags2 dstStageMask,
        VkAccessFlags2        srcAccessMask,
        VkAccessFlags2        dstAccessMask)
    {
        VkBufferMemoryBarrier2 memoryBarrier = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
        memoryBarrier.srcAccessMask = srcAccessMask;
        memoryBarrier.dstAccessMask = dstAccessMask;
        memoryBarrier.srcStageMask = srcStageMask;
        memoryBarrier.dstStageMask = dstStageMask;
        memoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memoryBarrier.buffer = targetBuffer;
        memoryBarrier.offset = 0;
        memoryBarrier.size = VK_WHOLE_SIZE;

        VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dependencyInfo.bufferMemoryBarrierCount = 1;
        dependencyInfo.pBufferMemoryBarriers = &memoryBarrier;

        vkCmdPipelineBarrier2KHR(commandList.m_VkCommandBuffer, &dependencyInfo);
    };
    
    auto SimpleImageBarrier = [this](
        CommandListVulkan&           commandList, 
        VkImage                      targetImage,
        VkImageAspectFlags           aspectFlags,
        VkPipelineStageFlags2        srcStageMask,
        VkPipelineStageFlags2        dstStageMask,
        VkAccessFlags2               srcAccessMask,
        VkAccessFlags2               dstAccessMask,
        VkImageLayout                fromLayout,
        std::optional<VkImageLayout> toLayout = std::nullopt)
    {
        VkImageMemoryBarrier2 memoryBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
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
        memoryBarrier.srcStageMask = srcStageMask;
        memoryBarrier.dstStageMask = dstStageMask;
        memoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        VkDependencyInfo dependencyInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dependencyInfo.imageMemoryBarrierCount = 1;
        dependencyInfo.pImageMemoryBarriers = &memoryBarrier;

        vkCmdPipelineBarrier2KHR(commandList.m_VkCommandBuffer, &dependencyInfo);
    };
    
    ///////////////////////////////////////////////////////////////////////////////////////
    // DATA PROCESSOR: RP_SCENE //
    ///////////////////////////////////////////////////////////////////////////////////////
    auto sceneDataProcessor = PassDataProcessor(
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        SimpleImageBarrier(
            commandList, 
            m_DefaultRenderTargets[RP_SCENE].GetColorAttachments()[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, 
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        SimpleImageBarrier(
            commandList, 
            m_DefaultRenderTargets[RP_SCENE].GetColorAttachments()[1].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        SimpleImageBarrier(
            commandList, 
            m_DefaultRenderTargets[RP_SCENE].GetDepthAttachment().GetVkImage(),
            VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
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
        SimpleImageBarrier(
            commandList, 
            m_DefaultRenderTargets[RP_SCENE].GetColorAttachments()[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        SimpleImageBarrier(
            commandList, 
            m_DefaultRenderTargets[RP_SCENE].GetColorAttachments()[1].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        SimpleImageBarrier(
            commandList,
            m_DefaultRenderTargets[RP_SCENE].GetDepthAttachment().GetVkImage(),
            VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL); // VK_IMAGE_LAYOUT_GENERAL);
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
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT);
        SimpleBufferBarrier(
            commandList,
            m_LightCountsBuffer.GetVkBuffer(),
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT);
    }, 
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        auto& computePassData = m_ComputePassData[CP_LIGHT_CULLING_LIST];

        for (auto& computablePass : computePassData.passComputable->GetPasses())
        {
            computePassData.passComputable->SetDispatchGroupCount(0, 
            { 
                // Dispatch all tiles at once
                m_SceneInfoUniformData.globalClusterCountX,
                m_SceneInfoUniformData.globalClusterCountY,
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
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 
            VK_ACCESS_2_MEMORY_WRITE_BIT, 
            VK_ACCESS_2_MEMORY_READ_BIT);
        SimpleBufferBarrier(
            commandList, 
            m_LightCountsBuffer.GetVkBuffer(),
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT);
        SimpleImageBarrier(
            commandList,
            m_DefaultRenderTargets[RP_SCENE].GetDepthAttachment().GetVkImage(),
            VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT,
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
            m_DefaultRenderTargets[RP_DEFERRED_LIGHT].GetColorAttachments()[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
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
            m_DefaultRenderTargets[RP_DEFERRED_LIGHT].GetColorAttachments()[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT,
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
            m_DefaultRenderTargets[RP_HUD].GetColorAttachments()[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
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
            m_DefaultRenderTargets[RP_HUD].GetColorAttachments()[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT,
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
            m_DefaultRenderTargets[RP_BLIT].GetColorAttachments()[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
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
            m_DefaultRenderTargets[RP_BLIT].GetColorAttachments()[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    });

    ///////////////////////////////////////////////////////////////////////////////////////
    // DATA PROCESSOR: TILE SHADING SCENE //
    ///////////////////////////////////////////////////////////////////////////////////////
    auto tileShadingSceneDataProcessor = PassDataProcessor(
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        SimpleImageBarrier(
            commandList, 
            m_TileShadingSceneRenderTarget.GetColorAttachments()[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        SimpleImageBarrier(
            commandList, 
            m_TileShadingSceneRenderTarget.GetColorAttachments()[1].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        SimpleImageBarrier(
            commandList, 
            m_TileShadingSceneRenderTarget.GetColorAttachments()[2].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        SimpleImageBarrier(
            commandList, 
            m_TileShadingSceneRenderTarget.GetDepthAttachment().GetVkImage(),
            VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
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

        SimpleImageBarrier(
            commandList,
            m_TileShadingSceneRenderTarget.GetColorAttachments()[0].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR);
        SimpleImageBarrier(
            commandList,
            m_TileShadingSceneRenderTarget.GetColorAttachments()[1].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR);
        SimpleImageBarrier(
            commandList,
            m_TileShadingSceneRenderTarget.GetDepthAttachment().GetVkImage(),
            VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_2_SHADER_TILE_ATTACHMENT_READ_BIT_QCOM,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR); // Works with storage image, otherwise it would need to be VK_IMAGE_LAYOUT_GENERAL

        SimpleBufferBarrier(
            commandList,
            m_LightIndicesBuffer.GetVkBuffer(),
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT);
        SimpleBufferBarrier(
            commandList,
            m_LightCountsBuffer.GetVkBuffer(),
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT);

        VkPerTileBeginInfoQCOM per_tile_beging_info{ VK_STRUCTURE_TYPE_PER_TILE_BEGIN_INFO_QCOM };
        vkCmdBeginPerTileExecutionQCOM(commandList.m_VkCommandBuffer, &per_tile_beging_info);

        if (m_IsAreaDispatchActive)
        {
            auto& passes = m_TileShadingPassData.passComputableAreaDispatch->GetPasses();
            for (int i=0; i< passes.size(); ++i)
            {
                m_TileShadingPassData.passComputableAreaDispatch->DispatchPass(
                    commandList,
                    passes[i],
                    whichBuffer);
            }
        }
        else
        {
            auto& passes = m_TileShadingPassData.passComputable->GetPasses();
            for (int i = 0; i < passes.size(); ++i)
            {
                m_TileShadingPassData.passComputable->SetDispatchGroupCount(i,
                {
                    m_SceneInfoUniformData.binClusterCountX,
                    m_SceneInfoUniformData.binClusterCountY,
                    1
                });

                m_TileShadingPassData.passComputable->DispatchPass(
                    commandList,
                    passes[i],
                    whichBuffer);
            }
        }
        
        VkPerTileEndInfoQCOM per_tile_end_info{ VK_STRUCTURE_TYPE_PER_TILE_END_INFO_QCOM };
        vkCmdEndPerTileExecutionQCOM(commandList.m_VkCommandBuffer, &per_tile_end_info);

        SimpleBufferBarrier(
            commandList,
            m_LightIndicesBuffer.GetVkBuffer(),
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT);
        SimpleBufferBarrier(
            commandList,
            m_LightCountsBuffer.GetVkBuffer(),
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT);
        SimpleImageBarrier(
            commandList,
            m_TileShadingSceneRenderTarget.GetDepthAttachment().GetVkImage(),
            VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_TILE_ATTACHMENT_READ_BIT_QCOM, // VK_ACCESS_2_MEMORY_READ_BIT
            VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT, // VK_ACCESS_2_MEMORY_READ_BIT
            VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR,
            VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR);

        std::array<uint32_t, 2> inputColorAttachmentIndices = { 0, 1 };
        uint32_t                inputDepthAttachmentIndex = 2;
        std::array<uint32_t, 1> colorAttachmentLocations = { 2 };

        VkRenderingInputAttachmentIndexInfo renderingInputAttachmentIndexInfo{ VK_STRUCTURE_TYPE_RENDERING_INPUT_ATTACHMENT_INDEX_INFO };
        renderingInputAttachmentIndexInfo.colorAttachmentCount = inputColorAttachmentIndices.size();
        renderingInputAttachmentIndexInfo.pColorAttachmentInputIndices = inputColorAttachmentIndices.data();
        renderingInputAttachmentIndexInfo.pDepthInputAttachmentIndex = &inputDepthAttachmentIndex;

        VkRenderingAttachmentLocationInfo renderingAttachmentLocationInfo{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO };
        renderingAttachmentLocationInfo.colorAttachmentCount = colorAttachmentLocations.size();
        renderingAttachmentLocationInfo.pColorAttachmentLocations = colorAttachmentLocations.data();

        vkCmdSetRenderingInputAttachmentIndices(commandList, &renderingInputAttachmentIndexInfo);
        vkCmdSetRenderingAttachmentLocations(commandList, &renderingAttachmentLocationInfo);

        AddDrawableToCmdBuffer(*m_DeferredLightTileShadingQuadDrawable, commandList, whichBuffer);
    }, 
    [&](uint32_t whichBuffer, CommandListVulkan& commandList)
    {
        SimpleImageBarrier(
            commandList,
            m_TileShadingSceneRenderTarget.GetColorAttachments()[2].GetVkImage(),
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, // VK_ACCESS_2_MEMORY_WRITE_BIT
            VK_ACCESS_2_MEMORY_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    

    ///////////////////////////////////////////////////////////////////////////////////////
    // RECORD COMMANDS //
    ///////////////////////////////////////////////////////////////////////////////////////
    if (m_IsTileShadingActive)
    {
        RecordPassData(
            RecordRenderingInfo({ &m_TileShadingPassData.renderContext , RenderingAttachmentInfoGroup(
                {
                    RenderingAttachmentInfo::Color(m_TileShadingSceneRenderTarget.GetColorAttachments()[0], RenderPassInputUsage::Clear, RenderPassOutputUsage::Store), // RenderPassOutputUsage::Discard
                    RenderingAttachmentInfo::Color(m_TileShadingSceneRenderTarget.GetColorAttachments()[1], RenderPassInputUsage::Clear, RenderPassOutputUsage::Store), // RenderPassOutputUsage::Discard
                    RenderingAttachmentInfo::Color(m_TileShadingSceneRenderTarget.GetColorAttachments()[2], RenderPassInputUsage::Clear, RenderPassOutputUsage::Store),
                },
                    RenderingAttachmentInfo::Depth(m_TileShadingSceneRenderTarget.GetDepthAttachment(),     true,                        RenderPassOutputUsage::Store) // RenderPassOutputUsage::Discard
                )}),
            PassRecordData(m_TileShadingPassData, std::move(tileShadingSceneDataProcessor)),
            true);
        RecordPassData(
            RecordRenderingInfo({ &m_TileShadingPassData.renderContext , RenderingAttachmentInfoGroup(
                m_TileShadingSceneRenderTarget,
                RenderPassInputUsage::Clear,
                RenderPassOutputUsage::Store,
                true,
                RenderPassOutputUsage::Store)
                }),
            PassRecordData(m_TileShadingPassData, std::move(tileShadingSceneDataProcessor)),
            true);
    }
    else
    {
        RecordPassData(
            RecordRenderingInfo({ &m_RenderPassData[RP_SCENE].renderContext , RenderingAttachmentInfoGroup(
                m_DefaultRenderTargets[RP_SCENE],
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
            RecordRenderingInfo({ &m_RenderPassData[RP_DEFERRED_LIGHT].renderContext , RenderingAttachmentInfoGroup(
                m_DefaultRenderTargets[RP_DEFERRED_LIGHT],
                RenderPassInputUsage::Clear,
                RenderPassOutputUsage::Store)
                }),
            PassRecordData(m_RenderPassData[RP_DEFERRED_LIGHT], std::move(deferredLightDataProcessor)));
    }

    RecordPassData(
        RecordRenderingInfo({ &m_RenderPassData[RP_HUD].renderContext , RenderingAttachmentInfoGroup(
            m_DefaultRenderTargets[RP_HUD],
            RenderPassInputUsage::Clear,
            RenderPassOutputUsage::Store)
            }),
        PassRecordData(m_RenderPassData[RP_HUD], std::move(hudDataProcessor)));
        
    RecordPassData(
        RecordRenderingInfo({ &m_RenderPassData[RP_BLIT].renderContext , RenderingAttachmentInfoGroup(
            m_DefaultRenderTargets[RP_BLIT],
            RenderPassInputUsage::Clear,
            RenderPassOutputUsage::Store)
            }),
        PassRecordData(m_RenderPassData[RP_BLIT], std::move(blitDataProcessor)));

    ///////////////////////////////////////////////////////////////////////////////////////
    // SWAPCHAIN COPY //
    ///////////////////////////////////////////////////////////////////////////////////////
    {
        const auto& sourceImage = m_DefaultRenderTargets[RP_BLIT].GetColorAttachments()[0];
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
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
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
            m_DefaultRenderTargets[RP_BLIT].GetColorAttachments()[0].GetVkImage(),
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
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_ACCESS_2_MEMORY_READ_BIT,
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