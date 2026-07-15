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

#include "application.hpp"
#include "main/applicationEntrypoint.hpp"
#include "camera/cameraController.hpp"
#include "camera/cameraControllerTouch.hpp"
#include "camera/cameraData.hpp"
#include "camera/cameraGltfLoader.hpp"
#include "gui/imguiVulkan.hpp"
#include "material/vulkan/descriptorSetLayout.hpp"
#include "material/vulkan/drawable.hpp"
#include "material/vulkan/material.hpp"
#include "material/vulkan/materialPass.hpp"
#include "material/vulkan/materialManager.hpp"
#include "material/vulkan/shader.hpp"
#include "material/vulkan/specializationConstantsLayout.hpp"
#include "material/shaderManagerT.hpp"
#include "mesh/meshHelper.hpp"
#include "mesh/meshLoader.hpp"
#include "system/math_common.hpp"
#include "texture/textureManager.hpp"
#include "imgui/imgui.h"
#include "vulkan/extensionLib.hpp"

#include <random>
#include <iostream>
#include <filesystem>

namespace
{
    static constexpr std::array<const char* const, NUM_RENDER_PASSES> sRenderPassNames = { "RP_SCENE", "RP_HUD", "RP_BLIT" };

    glm::vec3   gCameraStartPos    = glm::vec3(-2.1f, 4.0f, 6.0f);
    glm::vec3   gCameraStartRot    = glm::vec3(-10.0f, -20.0f, 0.0f);
    VAR(float,  gCameraRotateSpeed,  0.25f, kVariableNonpersistent);
    VAR(float,  gCameraMoveSpeed,    1.0f,  kVariableNonpersistent);

    float   gFOV = PI_DIV_4;
    float   gNearPlane = 1.0f;
    float   gFarPlane = 1800.0f;
    float   gNormalAmount = 1.0f;
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
void Application::PreInitializeSetVulkanConfiguration(Vulkan::AppConfiguration& appConfig)
//-----------------------------------------------------------------------------
{
    ApplicationHelperBase::PreInitializeSetVulkanConfiguration( appConfig );
    appConfig.RequiredExtension<ExtensionLib::Ext_VK_KHR_dynamic_rendering>();

    // Fixes a crash when trying to debug with renderdoc...
    appConfig.AddExtension(std::make_unique<VulkanExtension<VulkanExtensionType::eInstance>>("VK_KHR_portability_enumeration", VulkanExtensionStatus::eOptional));
}

//-----------------------------------------------------------------------------
bool Application::Initialize(uintptr_t windowHandle, uintptr_t hInstance)
//-----------------------------------------------------------------------------
{
    if (!ApplicationHelperBase::Initialize( windowHandle, hInstance ))
    {
        return false;
    }

    if (!InitializeCamera())
    {
        return false;
    }

    if (!LoadShaders())
    {
        return false;
    }

    if (!InitUniforms())
    {
        return false;
    }

    if (!CreateRenderTargets())
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

    if (!InitCommandBuffers())
    {
        return false;
    }

    if (!BuildCmdBuffers())
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
    pVulkan->WaitUntilIdle();

    // Uniform Buffers
    ReleaseUniformBuffer(pVulkan, m_ObjectVertUniform);
    ReleaseUniformBuffer(pVulkan, m_ObjectFragUniform);
    ReleaseUniformBuffer(pVulkan, m_LightUniform);

    // Cmd buffers
    for (auto& cmdBuffer : m_PrimaryCommandLists)
    {
        cmdBuffer.Release();
    }
    for (auto& cmdBuffer : m_SecondaryObjectCommandLists)
    {
        cmdBuffer.Release();
    }

    // Textures
    m_ColorBuffer.Release(pVulkan);
    m_DepthBuffer.Release(pVulkan);
    m_HudColorBuffer.Release(pVulkan);

    // Render passes

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

    cameraController->SetRotateSpeed(gCameraRotateSpeed);
    cameraController->SetMoveSpeed(gCameraMoveSpeed);

    m_CameraController = std::move(cameraController);

    return true;
}

//-----------------------------------------------------------------------------
bool Application::LoadShaders()
//-----------------------------------------------------------------------------
{
    m_ShaderManager->RegisterRenderPassNames(sRenderPassNames);

    LOGI("******************************");
    LOGI("Loading Shaders...");
    LOGI("******************************");

    typedef std::pair<std::string, std::string> tIdAndFilename;
    for (const tIdAndFilename& i :
            { tIdAndFilename { "Blit",  "Blit.json" },
              tIdAndFilename { "Scene", "Scene.json" }
            })
    {

        if (!m_ShaderManager->AddShader(*m_AssetManager, i.first, i.second, SHADER_DESTINATION_PATH))
        {
            LOGE("Error Loading shader %s from %s", i.first.c_str(), i.second.c_str());
            LOGI("Please verify if you have all required assets on the sample media folder (%s)", SHADER_DESTINATION_PATH);
            return false;
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::CreateRenderTargets()
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();

    LOGI("**************************");
    LOGI("Creating Render Targets...");
    LOGI("**************************");

    TextureFormat vkDesiredDepthFormat = pVulkan->GetBestSurfaceDepthFormat();
    TextureFormat desiredDepthFormat = vkDesiredDepthFormat;

    TextureFormat mainColorFormat[] = { TextureFormat::R8G8B8A8_SRGB };
    TextureFormat hudColorFormat[]  = { TextureFormat::R8G8B8A8_SRGB };
    TextureFormat swapchainColorFormat[] = { pVulkan->GetSwapchainFormat() };
    const Msaa msaa = Msaa::Samples1;

    m_ColorBuffer    = CreateTextureObject( *pVulkan, gRenderWidth, gRenderHeight, mainColorFormat[0], TEXTURE_TYPE::TT_RENDER_TARGET, "Color", msaa);
    m_DepthBuffer    = CreateTextureObject( *pVulkan, gRenderWidth, gRenderHeight, desiredDepthFormat, TEXTURE_TYPE::TT_DEPTH_TARGET, "Depth", msaa );
    m_HudColorBuffer = CreateTextureObject( *pVulkan, gRenderWidth, gRenderHeight, hudColorFormat[0], TEXTURE_TYPE::TT_RENDER_TARGET, "HudColor", msaa );

    m_SceneRenderContext = RenderContext{ std::span<TextureFormat>(mainColorFormat, 1), desiredDepthFormat, TextureFormat::UNDEFINED, "RP_SCENE" };
    m_HudRenderContext   = RenderContext{ std::span<TextureFormat>(hudColorFormat, 1), TextureFormat::UNDEFINED, TextureFormat::UNDEFINED, "RP_HUD" };
    m_BlitRenderContext  = RenderContext{ std::span<TextureFormat>(swapchainColorFormat, 1), TextureFormat::UNDEFINED, TextureFormat::UNDEFINED, "RP_BLIT" };

    // CreateTextureObject transitions render targets to attachment layouts, but keeps descriptor
    // layouts set to shader-read for later sampling. Track the real current layouts explicitly.
    m_ColorBufferLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    m_DepthBufferLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    m_HudColorBufferLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    m_SwapchainImageInitialized.assign(pVulkan->GetSwapchainBufferCount(), false);

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitUniforms()
//-----------------------------------------------------------------------------
{
    LOGI("******************************");
    LOGI("Initializing Uniforms...");
    LOGI("******************************");

    Vulkan* const pVulkan = GetVulkan();

    if (!CreateUniformBuffer(pVulkan, m_ObjectVertUniform))
    {
        return false;
    }

    if (!CreateUniformBuffer(pVulkan, m_ObjectFragUniform))
    {
        return false;
    }

    if (!CreateUniformBuffer(pVulkan, m_LightUniform))
    {
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitAllRenderPasses()
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();

    TextureFormat surfaceFormat = pVulkan->m_SurfaceFormat;
    auto swapChainColorFormat = std::span<const TextureFormat>({ &surfaceFormat, 1 });
    auto swapChainDepthFormat = pVulkan->m_SwapchainDepth.format;


    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitGui(uintptr_t windowHandle)
//-----------------------------------------------------------------------------
{
    m_Gui = std::make_unique<GuiImguiGfx>(*GetGfxApi());
    if (!m_Gui->Initialize(windowHandle, m_HudColorBuffer.Format, m_HudColorBuffer.Width, m_HudColorBuffer.Height))
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

    const auto* pSceneShader    = m_ShaderManager->GetShader("Scene");
    const auto* pBlitQuadShader = m_ShaderManager->GetShader("Blit");
    if (!pSceneShader || !pBlitQuadShader)
    {
        return false;
    }
    
    LOGI("******************************");
    LOGI("Loading and preparing scene...");
    LOGI("******************************");

    m_TextureManager->SetDefaultFilenameManipulators(PathManipulator_PrefixDirectory{TEXTURE_DESTINATION_PATH}, PathManipulator_ChangeExtension{".ktx"});

    auto MaterialLoader = [&](const MeshObjectIntermediate::MaterialDef& materialDef) ->std::optional<Material>
    {
        auto* diffuseTexture = m_TextureManager->GetOrLoadTexture(materialDef.diffuseFilename, m_SamplerRepeat);
        auto* normalTexture = m_TextureManager->GetOrLoadTexture(materialDef.bumpFilename, m_SamplerRepeat);

        if (diffuseTexture == nullptr || normalTexture == nullptr)
        {
            return std::nullopt;
        }

        auto shaderMaterial = m_MaterialManager->CreateMaterial(*pSceneShader, NUM_VULKAN_BUFFERS,
            [&](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
            {
                if (texName == "Diffuse")
                {
                    return { diffuseTexture };
                }
                if (texName == "Normal")
                {
                    return { normalTexture };
                }

                return {};
            },
            [this](const std::string& bufferName) -> PerFrameBuffer
            {
                if (bufferName == "Vert")
                {
                    return { m_ObjectVertUniform.buf[0].GetVkBuffer() };
                }
                else if (bufferName == "Frag")
                {
                    return { m_ObjectFragUniform.buf[0].GetVkBuffer() };
                }
                else if (bufferName == "Light")
                {
                    return { m_LightUniform.buf[0].GetVkBuffer() };
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
    MeshObjectIntermediateGltfProcessor meshObjectProcessor(sceneAssetPath, ignoreTransforms, glm::vec3(1.0f,1.0f,1.0f));
    CameraGltfProcessor meshCameraProcessor{};

    if (!MeshLoader::LoadGltf(*m_AssetManager, sceneAssetPath, meshSanityCheckProcessor, meshObjectProcessor, meshCameraProcessor) ||
        !DrawableLoader::CreateDrawables(*pVulkan,
                                        std::move(meshObjectProcessor.m_meshObjects),
                                        m_SceneRenderContext,
                                        MaterialLoader,
                                        m_SceneDrawables,
                                        loaderFlags))
    {
        LOGE("Error Loading the scene gltf file");
        LOGI("Please verify if you have all required assets on the sample media folder");
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

    Mesh blitQuadMesh;
    MeshHelper::CreateScreenSpaceMesh(pVulkan->GetMemoryManager(), 0, &blitQuadMesh);

    // Blit MaterialBase
    auto blitQuadShaderMaterial = m_MaterialManager->CreateMaterial(*pBlitQuadShader, pVulkan->m_SwapchainImageCount,
        [this](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
        {
            if (texName == "Diffuse")
            {
                return { &m_ColorBuffer};
            }
            else if (texName == "Overlay")
            {
                return { &m_HudColorBuffer};
            }
            return {};
        },
        [this](const std::string& bufferName) -> PerFrameBuffer
        {
            return {};
        }
        );

    m_BlitQuadDrawable = std::make_unique<Drawable>(*pVulkan, std::move(blitQuadShaderMaterial));
    if (!m_BlitQuadDrawable->Init(m_BlitRenderContext, std::move(blitQuadMesh)))
    {
        return false;
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

    const size_t numSwapchainBuffers = pVulkan->GetSwapchainBufferCount();
    char szName[256];
    m_PrimaryCommandLists.resize( numSwapchainBuffers );
    m_SecondaryObjectCommandLists.resize( numSwapchainBuffers );

    for (uint32_t whichBuffer = 0; whichBuffer < numSwapchainBuffers; whichBuffer++)
    {
        sprintf( szName, "Primary (Buffer %d)", whichBuffer );
        m_PrimaryCommandLists[whichBuffer].Initialize( pVulkan, szName, CommandList::Type::Primary );

        sprintf( szName, "Secondary Objects (Buffer %d)", whichBuffer );
        m_SecondaryObjectCommandLists[whichBuffer].Initialize( pVulkan, szName, CommandList::Type::Secondary );
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::BuildCmdBuffers()
//-----------------------------------------------------------------------------
{
    LOGI("***************************");
    LOGI("Building Command Buffers...");
    LOGI("****************************");

    Vulkan* const pVulkan = GetVulkan();

    // Begin recording secondary
    for (auto& cmdBuffer: m_SecondaryObjectCommandLists)
    {
        uint32_t targetWidth = m_ColorBuffer.Width;
        uint32_t targetHeight = m_ColorBuffer.Height;
        const VkViewport viewport = {.width = (float)targetWidth, .height = (float)targetHeight, .minDepth = 0.0f, .maxDepth = 1.0f};
        const VkRect2D scissor = {.extent {.width = targetWidth, .height = targetHeight }};

        if (!cmdBuffer.Begin( m_SceneRenderContext ))
        {
            return false;
        }
        vkCmdSetViewport( cmdBuffer, 0, 1, &viewport );
        vkCmdSetScissor( cmdBuffer, 0, 1, &scissor );
    }

    // Scene drawables
    for (const auto& sceneDrawable : m_SceneDrawables)
    {
        AddDrawableToCmdBuffers( sceneDrawable, m_SecondaryObjectCommandLists.data(), 1, (uint32_t)m_SecondaryObjectCommandLists.size() );
    }

    // End recording
    for (auto& cmdBuffer : m_SecondaryObjectCommandLists)
    {
        cmdBuffer.End();
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

        if (ImGui::Begin("FPS", (bool*)nullptr, ImGuiWindowFlags_NoTitleBar))
        {
            ImGui::Text("FPS: %.1f", m_CurrentFPS);
            ImGui::Text("Camera [%f, %f, %f]", m_Camera.Position().x, m_Camera.Position().y, m_Camera.Position().z);
            ImGui::DragFloat3("Light Dir", &m_LightUniformData.LightDirection.x, 0.01f, -1.0f, 1.0f);
            ImGui::DragFloat3("Light Color", &m_LightUniformData.LightColor.x, 0.01f, 0.0f, 1.0f);
            ImGui::DragFloat("Light Intensity", &m_LightUniformData.LightColor.w, 0.1f, 0.0f, 100.0f);
            ImGui::DragFloat3("Ambient Color", &m_LightUniformData.AmbientColor.x, 0.01f, 0.0f, 1.0f);

            glm::vec3 LightDirNotNormalized   = m_LightUniformData.LightDirection;
            LightDirNotNormalized             = glm::normalize(LightDirNotNormalized);
            m_LightUniformData.LightDirection = glm::vec4(LightDirNotNormalized, 0.0f);
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
        UpdateUniformBuffer(pVulkan, m_ObjectVertUniform, m_ObjectVertUniformData, whichBuffer);
    }

    // Frag data
    {
        m_ObjectFragUniformData.Color        = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
        m_ObjectFragUniformData.NormalHeight = glm::vec4(gNormalAmount, gNormalMirrorReflectAmount, 0.0f, 0.0f);

        UpdateUniformBuffer(pVulkan, m_ObjectFragUniform, m_ObjectFragUniformData, whichBuffer);
    }

    // Light data
    {
        glm::mat4 CameraViewInv       = glm::inverse(m_Camera.ViewMatrix());
        glm::mat4 CameraProjection    = m_Camera.ProjectionMatrix();
        glm::mat4 CameraProjectionInv = glm::inverse(CameraProjection);

        m_LightUniformData.ProjectionInv     = CameraProjectionInv;
        m_LightUniformData.ViewInv           = CameraViewInv;
        m_LightUniformData.ViewProjectionInv = CameraViewInv * CameraProjectionInv;
        m_LightUniformData.ProjectionInvW    = glm::vec4(CameraProjectionInv[0].w, CameraProjectionInv[1].w, CameraProjectionInv[2].w, CameraProjectionInv[3].w);
        m_LightUniformData.CameraPos         = glm::vec4(m_Camera.Position(), 0.0f);

        UpdateUniformBuffer(pVulkan, m_LightUniform, m_LightUniformData, whichBuffer);
    }

    return true;
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
    UpdateUniforms(whichBuffer);

    auto& cmdBuffer = m_PrimaryCommandLists[whichBuffer];
    cmdBuffer.Reset();
    cmdBuffer.Begin();

    auto transitionImage = [pVulkan, &cmdBuffer](VkImage image,
                                                 VkImageAspectFlags aspectMask,
                                                 VkImageLayout& currentLayout,
                                                 VkImageLayout newLayout)
    {
        if (currentLayout != newLayout)
        {
            pVulkan->SetImageLayout(image, cmdBuffer, aspectMask, currentLayout, newLayout);
            currentLayout = newLayout;
        }
    };

    // RP_SCENE
    {
        transitionImage(m_ColorBuffer.GetVkImage(), VK_IMAGE_ASPECT_COLOR_BIT, m_ColorBufferLayout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        transitionImage(m_DepthBuffer.GetVkImage(), VK_IMAGE_ASPECT_DEPTH_BIT, m_DepthBufferLayout, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        RenderingAttachmentInfoGroup sceneAttachments(
            { RenderingAttachmentInfo::Color(m_ColorBuffer, RenderPassInputUsage::Clear, RenderPassOutputUsage::StoreReadOnly) },
            RenderingAttachmentInfo::Depth(m_DepthBuffer, true, RenderPassOutputUsage::Store));

        auto renderingInfo = m_SceneRenderContext.GetRenderingInfo(sceneAttachments);
        renderingInfo->flags = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT;

        cmdBuffer.BeginRenderPass(renderingInfo.get());
        vkCmdExecuteCommands(cmdBuffer, 1, &m_SecondaryObjectCommandLists[whichBuffer].m_VkCommandBuffer);
        cmdBuffer.EndRendering();

        transitionImage(m_ColorBuffer.GetVkImage(), VK_IMAGE_ASPECT_COLOR_BIT, m_ColorBufferLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    // RP_HUD
    if (m_Gui)
    {
        transitionImage(m_HudColorBuffer.GetVkImage(), VK_IMAGE_ASPECT_COLOR_BIT, m_HudColorBufferLayout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        RenderingAttachmentInfoGroup hudAttachments(
            { RenderingAttachmentInfo::Color(m_HudColorBuffer, RenderPassInputUsage::Clear, RenderPassOutputUsage::StoreReadOnly) });

        auto renderingInfo = m_HudRenderContext.GetRenderingInfo(hudAttachments);
        cmdBuffer.BeginRenderPass(renderingInfo.get());
        GetGui()->Render(cmdBuffer);
        cmdBuffer.EndRendering();

        transitionImage(m_HudColorBuffer.GetVkImage(), VK_IMAGE_ASPECT_COLOR_BIT, m_HudColorBufferLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    // RP_BLIT
    {
        const uint32_t swapchainImageIndex = currentVulkanBuffer.swapchainPresentIdx;
        VkImageLayout swapchainOldLayout = m_SwapchainImageInitialized[swapchainImageIndex] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED;

        pVulkan->SetImageLayout(pVulkan->GetSwapchainImage(swapchainImageIndex),
                                cmdBuffer,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                swapchainOldLayout,
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        m_SwapchainImageInitialized[swapchainImageIndex] = true;

        VkRenderingAttachmentInfo swapchainColorAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        swapchainColorAttachment.imageView = pVulkan->GetSwapchainImageView(swapchainImageIndex);
        swapchainColorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        swapchainColorAttachment.resolveMode = VK_RESOLVE_MODE_NONE;
        swapchainColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        swapchainColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        const VkRect2D renderArea{ .extent{ .width = pVulkan->GetSurfaceWidth(), .height = pVulkan->GetSurfaceHeight() } };
        fvk::VkRenderingInfo renderingInfo{ VkRenderingInfo{
            .renderArea = renderArea,
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &swapchainColorAttachment,
        } };

        cmdBuffer.BeginRenderPass(renderingInfo.get());

        const VkViewport viewport = {
            .width = static_cast<float>(pVulkan->GetSurfaceWidth()),
            .height = static_cast<float>(pVulkan->GetSurfaceHeight()),
            .minDepth = 0.0f,
            .maxDepth = 1.0f
        };
        vkCmdSetScissor(cmdBuffer, 0, 1, &renderArea);
        vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

        // Blit quad drawable
        AddDrawableToCmdBuffers( *m_BlitQuadDrawable.get(), &cmdBuffer, 1, 1, whichBuffer );

        cmdBuffer.EndRendering();

        VkImageLayout swapchainCurrentLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        transitionImage(pVulkan->GetSwapchainImage(swapchainImageIndex),
                        VK_IMAGE_ASPECT_COLOR_BIT,
                        swapchainCurrentLayout,
                        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    }

    // Done adding commands to primary buffer
    cmdBuffer.End();

    // Send the primary render pass to the gpu
    cmdBuffer.QueueSubmit( currentVulkanBuffer, currentVulkanBuffer.renderCompleteSemaphore );

    // Queue is loaded up, tell the driver to start processing
    pVulkan->PresentQueue( currentVulkanBuffer.renderCompleteSemaphore, currentVulkanBuffer.swapchainPresentIdx );

    // ********************************
    // Application Draw() - End
    // ********************************
}
