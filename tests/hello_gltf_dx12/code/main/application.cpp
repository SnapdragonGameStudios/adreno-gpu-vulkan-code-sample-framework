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

#if VULKAN
#include "vulkan/vulkan.hpp"
#include "vulkan/commandBuffer.hpp"
#include "gui/imguiVulkan.hpp"
#include "material/vulkan/computable.hpp"
#include "material/vulkan/drawable.hpp"
#include "material/vulkan/material.hpp"
#include "material/vulkan/materialManager.hpp"
#include "material/vulkan/pipeline.hpp"
#include "material/vulkan/shaderModule.hpp"
#include "texture/vulkan/textureManager.hpp"
#else
#include "dx12/dx12.hpp"
#include "dx12/commandList.hpp"
#include "gui/imguiDx12.hpp"
#include "material/dx12/computable.hpp"
#include "material/dx12/drawableDx12.hpp"
#include "material/dx12/material.hpp"
#include "material/dx12/materialManager.hpp"
#include "texture/dx12/textureManager.hpp"
#endif
#include "material/drawableLoader.hpp"
#include "material/shaderManagerT.hpp"
#include "mesh/meshHelper.hpp"
#include "system/math_common.hpp"
#include "imgui.h"

#include <random>
#include <iostream>
#include <filesystem>
using namespace std::string_literals;


namespace
{
    static constexpr std::array<const char*, NUM_RENDER_PASSES> sRenderPassNames = { "RP_SCENE", "RP_HUD", "RP_BLIT" };

    glm::vec3 gCameraStartPos = glm::vec3(20.48f, 23.0f, -7.21f);
    glm::vec3 gCameraStartRot = glm::vec3(0.0f, -70.0f, 0.0f);

    float   gFOV = PI_DIV_4;
    float   gNearPlane = 1.0f;
    float   gFarPlane = 1800.0f;
    float   gNormalAmount = 2.0f;
    float   gNormalMirrorReflectAmount = 0.05f;

    const char* gMuseumAssetsPath = "Media\\Meshes\\Museum.gltf";
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

//-----------------------------------------------------------------------------
Application::Application() : ApplicationHelperBase()
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
Application::~Application()
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
void Application::PreInitializeSetVulkanConfiguration( AppConfiguration& config)
//-----------------------------------------------------------------------------
{
    config.gfx.SwapchainDepthFormat = TextureFormat::UNDEFINED;
}

//-----------------------------------------------------------------------------
bool Application::Initialize(uintptr_t windowHandle, uintptr_t hInstance)
//-----------------------------------------------------------------------------
{
    if (!ApplicationHelperBase::Initialize(windowHandle, hInstance))
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

    if (!InitAllRenderPasses())
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
    auto* const pGfxApi = GetGfxApi();

    // Uniform Buffers
    ReleaseUniformBuffer(pGfxApi, m_ObjectVertUniform);
    ReleaseUniformBuffer(pGfxApi, m_ObjectFragUniform);
    ReleaseUniformBuffer(pGfxApi, m_LightUniform);

    // Cmd buffers
    for (int whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        for (auto& cmdBuffer : m_RenderPassData[whichPass].ObjectsCmdBuffer)
        {
            cmdBuffer.Release();
        }

        //m_RenderPassData[whichPass].RenderTarget.Release();
    }

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
#if VULKAN
            { tIdAndFilename { "Blit"s,  "Media\\Shaders\\Blit.json"s },
              tIdAndFilename { "Scene"s, "Media\\Shaders\\Scene.json"s }
            })
#else
            { tIdAndFilename { "Blit"s,  "Media\\Shaders\\BlitDx.json"s },
              tIdAndFilename { "Scene"s, "Media\\Shaders\\SceneDx.json"s },
              tIdAndFilename { "ColorGrading"s, "Media\\Shaders\\ColorGradingDx.json"s }
            })
#endif
    {
        if (!m_ShaderManager->AddShader(*m_AssetManager, i.first, i.second))
        {
            LOGE("Error Loading shader %s from %s", i.first.c_str(), i.second.c_str());
            LOGI("Please verify if you have all required assets on the sample media folder");
            return false;
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::CreateRenderTargets()
//-----------------------------------------------------------------------------
{
    tGfxApi* const pGfxApi = GetGfxApi();

    LOGI("**************************");
    LOGI("Creating Render Targets...");
    LOGI("**************************");

    TextureFormat desiredDepthFormat = pGfxApi->GetBestSurfaceDepthFormat();

    const TextureFormat MainColorType[] = { TextureFormat::R8G8B8A8_UNORM };
    const TextureFormat HudColorType[]  = { TextureFormat::R8G8B8A8_UNORM };

    if (!m_RenderPassData[RP_SCENE].RenderTarget.Initialize(pGfxApi, gRenderWidth, gRenderHeight, MainColorType, desiredDepthFormat, Msaa::Samples1, "Scene RT"))
    {
        LOGE("Unable to create scene render target");
        return false;
    }

    // Notice no depth on the HUD RT
    if (!m_RenderPassData[RP_HUD].RenderTarget.Initialize(pGfxApi, gSurfaceWidth, gSurfaceHeight, HudColorType, TextureFormat::UNDEFINED, Msaa::Samples1, "HUD RT"))
    {
        LOGE("Unable to create hud render target");
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitUniforms()
//-----------------------------------------------------------------------------
{
    LOGI("******************************");
    LOGI("Initializing Uniforms...");
    LOGI("******************************");

    tGfxApi* const pGfxApi = GetGfxApi();

    if (!CreateUniformBuffer(pGfxApi, m_ObjectVertUniform))
    {
        return false;
    }

    if (!CreateUniformBuffer(pGfxApi, m_ObjectFragUniform))
    {
        return false;
    }

    if (!CreateUniformBuffer(pGfxApi, m_LightUniform))
    {
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitAllRenderPasses()
//-----------------------------------------------------------------------------
{
    tGfxApi* const pGfxApi = GetGfxApi();

    //                                       ColorInputUsage |               ClearDepthRenderPass | ColorOutputUsage |                     DepthOutputUsage |              ClearColor
    m_RenderPassData[RP_SCENE].PassSetup = { RenderPassInputUsage::Clear,    true,                  RenderPassOutputUsage::StoreReadOnly,  RenderPassOutputUsage::Store,   {}};
    m_RenderPassData[RP_HUD].PassSetup   = { RenderPassInputUsage::Clear,    false,                 RenderPassOutputUsage::StoreReadOnly,  RenderPassOutputUsage::Discard, {}};
    m_RenderPassData[RP_BLIT].PassSetup  = { RenderPassInputUsage::DontCare, false,                 RenderPassOutputUsage::Present,        RenderPassOutputUsage::Discard, {}};

    TextureFormat swapChainColorFormat = pGfxApi->GetSwapchainFormat();
    TextureFormat swapChainDepthFormat = TextureFormat::UNDEFINED;
    auto swapChainColorFormats = std::span<const TextureFormat>({ &swapChainColorFormat, 1 });

    LOGI("******************************");
    LOGI("Initializing Render Passes... ");
    LOGI("******************************");

#if 1
    for (uint32_t whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        bool isSwapChainRenderPass = whichPass == RP_BLIT;

        std::span<const TextureFormat> colorFormats = isSwapChainRenderPass ? swapChainColorFormats : m_RenderPassData[whichPass].RenderTarget.m_pLayerFormats;
        TextureFormat                  depthFormat  = isSwapChainRenderPass ? swapChainDepthFormat : m_RenderPassData[whichPass].RenderTarget.m_DepthFormat;

        const auto& passSetup = m_RenderPassData[whichPass].PassSetup;
        
        auto renderPass = ::CreateRenderPass(*pGfxApi,
                                             { colorFormats },
                                             depthFormat,
                                             Msaa::Samples1,
                                             passSetup.ColorInputUsage,
                                             passSetup.ColorOutputUsage,
                                             passSetup.ClearDepthRenderPass,
                                             passSetup.DepthOutputUsage,
                                             {});
        if (!renderPass)
            return false;
        m_RenderPassData[whichPass].RenderPass = std::move(renderPass);
    }
#endif
#if VULKAN
    m_RenderPassData[RP_SCENE].RenderTarget.InitializeFrameBuffer(pGfxApi, m_RenderPassData[RP_SCENE].RenderPass);
    m_RenderPassData[RP_HUD].RenderTarget.InitializeFrameBuffer(pGfxApi, m_RenderPassData[RP_HUD].RenderPass);
#endif // VULKAN

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitGui(uintptr_t windowHandle)
//-----------------------------------------------------------------------------
{
    const auto& hudRenderTarget = m_RenderPassData[RP_HUD].RenderTarget;
    m_Gui = std::make_unique<GuiImguiGfx>(*GetGfxApi(), m_RenderPassData[RP_HUD].RenderPass);
    if (!m_Gui->Initialize(windowHandle, hudRenderTarget.m_pLayerFormats[0], hudRenderTarget.m_Width, hudRenderTarget.m_Height))
    {
        return false;
    }
    
    return true;
}

//-----------------------------------------------------------------------------
bool Application::LoadMeshObjects()
//-----------------------------------------------------------------------------
{
    tGfxApi* const pGfxApi = GetGfxApi();

    LOGI("***********************");
    LOGI("Initializing Meshes... ");
    LOGI("***********************");

    const auto* pSceneShader    = m_ShaderManager->GetShader("Scene");
    const auto* pBlitQuadShader = m_ShaderManager->GetShader("Blit");
    const auto* pColorGradingShader = m_ShaderManager->GetShader("ColorGrading");
    if (!pSceneShader || !pBlitQuadShader || !pColorGradingShader)
    {
        return false;
    }

#if 1
    LOGI("***********************************");
    LOGI("Loading and preparing the museum...");
    LOGI("***********************************");

    m_TextureManager->SetDefaultFilenameManipulators(PathManipulator_PrefixDirectory{"Media\\Textures\\"}, PathManipulator_ChangeExtension{".ktx"});

    auto MaterialLoader = [&](const MeshObjectIntermediate::MaterialDef& materialDef)->std::unique_ptr<MaterialBase>
    {
        auto* diffuseTexture = m_TextureManager->GetOrLoadTexture(materialDef.diffuseFilename, m_SamplerEdgeClamp);
        auto* normalTexture = m_TextureManager->GetOrLoadTexture(materialDef.bumpFilename, m_SamplerEdgeClamp);

        if (diffuseTexture == nullptr || normalTexture == nullptr)
        {
            return {};
        }

        auto shaderMaterial = m_MaterialManager->CreateMaterial(*pSceneShader, NUM_SWAPCHAIN_BUFFERS,
            [&](const std::string& texName, MaterialManagerBase::tPerFrameTexInfo& texInfo)
            {
                if (texName == "Diffuse")
                {
                    texInfo = { diffuseTexture };
                }
                else if (texName == "Normal")
                {
                    texInfo = { normalTexture };
                }
                return;
            },
            [this](const std::string& bufferName, PerFrameBufferBase& buffers)
            {
                if (bufferName == "Vert")
                {
                    buffers = { m_ObjectVertUniform.bufferHandles };
                }
                else if (bufferName == "Frag")
                {
                    buffers = { m_ObjectFragUniform.bufferHandles};
                }
                else if (bufferName == "Light")
                {
                    buffers = { m_LightUniform.bufferHandles};
                }
                return;
            }
            );

        return shaderMaterial;
    };

    bool sceneMeshResult = DrawableLoader::LoadDrawables(
        *pGfxApi, 
        *m_AssetManager, 
        m_RenderPassData[RP_SCENE].RenderPass,
        sRenderPassNames[RP_SCENE],
        gMuseumAssetsPath,
        MaterialLoader,
        m_SceneDrawables,
        Msaa::Samples1,
        false, // UseInstancing
        {});   // RenderPassSubpasses
    if (!sceneMeshResult)
    {
        LOGE("Error Loading the museum gltf file");
        LOGI("Please verify if you have all required assets on the sample media folder");
        LOGI("If you are running on Android, don't forget to run the `02_CopyMediaToDevice.bat` script to copy all media files into the device memory");
        return false;
    }
#endif

    LOGI("*********************");
    LOGI("Creating Quad mesh...");
    LOGI("*********************");

    Mesh blitQuadMesh;
    MeshHelper::CreateScreenSpaceMesh(pGfxApi->GetMemoryManager(), 0, &blitQuadMesh);

    // Blit MaterialBase
    auto blitQuadShaderMaterial = m_MaterialManager->CreateMaterial(*pBlitQuadShader, pGfxApi->GetSwapchainBufferCount(),
        [this](const std::string& texName, MaterialManagerBase::tPerFrameTexInfo& texInfo)
        {
            if (texName == "Diffuse")
            {
                texInfo = { &m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[0] };
            }
            else if (texName == "Overlay")
            {
                texInfo = { &m_RenderPassData[RP_HUD].RenderTarget.m_ColorAttachments[0] };
            }
            return;
        },
        [this](const std::string& bufferName, PerFrameBufferBase&)
        {
            assert(0);
            return;
        }
        );

    m_BlitQuadDrawable = std::make_unique<Drawable>(*pGfxApi, std::move( blitQuadShaderMaterial ));
    if (!m_BlitQuadDrawable->Init(m_RenderPassData[RP_BLIT].RenderPass, {}, sRenderPassNames[RP_BLIT], std::move(blitQuadMesh)))
    {
        return false;
    }

    auto colorGradingMaterial = m_MaterialManager->CreateMaterial( *pColorGradingShader, 1,
        [this](const std::string& texName, MaterialManagerBase::tPerFrameTexInfo& texInfo)
        {
            if (texName == "Input")
            {
                texInfo = { &m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[0] };
            }
            return;
        },
        nullptr,
        [this]( const std::string& texName, ImageInfoBase& imageInfo )
        {
            if (texName == "Output")
            {
                //imageInfo = {&m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[1]};
            }
            return;
        });

    m_ColorGradingComputable = std::make_unique<Computable>(*pGfxApi, std::move(colorGradingMaterial));


    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitCommandBuffers()
//-----------------------------------------------------------------------------
{
    LOGI("*******************************");
    LOGI("Initializing Command Buffers...");
    LOGI("*******************************");

    tGfxApi* const pGfxApi = GetGfxApi();

    auto GetPassName = [](uint32_t whichPass)
    {
        if (whichPass >= sRenderPassNames.size())
        {
            LOGE("GetPassName() called with unknown pass (%d)!", whichPass);
            return "RP_UNKNOWN";
        }

        return sRenderPassNames[whichPass];
    };

    const uint32_t NumSwapchainBuffers = pGfxApi->GetSwapchainBufferCount();
    m_RenderPassData[RP_SCENE].ObjectsCmdBuffer.resize(NumSwapchainBuffers);
    m_RenderPassData[RP_HUD].ObjectsCmdBuffer.resize(NumSwapchainBuffers);
    m_RenderPassData[RP_BLIT].ObjectsCmdBuffer.resize(NumSwapchainBuffers);

    char szName[256];
    for (uint32_t whichBuffer = 0; whichBuffer < NumSwapchainBuffers; whichBuffer++)
    {
        for (uint32_t whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
        {
            // Model => Secondary
            sprintf(szName, "Model (%s; List %d of %d)", GetPassName(whichPass), whichBuffer + 1, NUM_SWAPCHAIN_BUFFERS);
            if (!m_RenderPassData[whichPass].ObjectsCmdBuffer[whichBuffer].Initialize(pGfxApi, szName, CommandListBase::Type::Bundle))
            {
                return false;
            }
        }
        // Per frame command lists
        sprintf(szName, "Render (List %d of %d)", whichBuffer + 1, NUM_SWAPCHAIN_BUFFERS);
        m_CommandLists[whichBuffer].Initialize(pGfxApi, szName);

        sprintf( szName, "Compute (List %d of %d)", whichBuffer + 1, NUM_SWAPCHAIN_BUFFERS );
        m_ComputeCommandLists[whichBuffer].Initialize( pGfxApi, szName, CommandListBase::Type::Compute );
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

    tGfxApi* const pGfxApi = GetGfxApi();

    // Begin recording
    for (uint32_t whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        auto& renderPassData         = m_RenderPassData[whichPass];
        bool  bisSwapChainRenderPass = whichPass == RP_BLIT;

        for (uint32_t whichBuffer = 0; whichBuffer < renderPassData.ObjectsCmdBuffer.size(); whichBuffer++)
        {
            auto& cmdBufer = renderPassData.ObjectsCmdBuffer[whichBuffer];

            uint32_t targetWidth  = bisSwapChainRenderPass ? pGfxApi->GetSurfaceWidth() : renderPassData.RenderTarget.m_Width;
            uint32_t targetHeight = bisSwapChainRenderPass ? pGfxApi->GetSurfaceHeight() : renderPassData.RenderTarget.m_Height;

#if VULKAN
            VkViewport viewport = {};
            viewport.x          = 0.0f;
            viewport.y          = 0.0f;
            viewport.width      = (float)targetWidth;
            viewport.height     = (float)targetHeight;
            viewport.minDepth   = 0.0f;
            viewport.maxDepth   = 1.0f;

            VkRect2D scissor      = {};
            scissor.offset.x      = 0;
            scissor.offset.y      = 0;
            scissor.extent.width  = targetWidth;
            scissor.extent.height = targetHeight;

            // Set up some values that change based on render pass
            const auto&  whichRenderPass  = renderPassData.RenderPass;
            VkFramebuffer whichFramebuffer = bisSwapChainRenderPass ? pGfxApi->GetSwapchainFramebuffer(WhichBuffer).m_FrameBuffer : renderPassData.RenderTarget.m_FrameBuffer;

            if (!cmdBufer.Begin(whichFramebuffer, whichRenderPass, bisSwapChainRenderPass))
            {
                return false;
            }
            vkCmdSetViewport(cmdBufer.m_VkCommandBuffer, 0, 1, &viewport);
            vkCmdSetScissor(cmdBufer.m_VkCommandBuffer, 0, 1, &scissor);
#else  // VULKAN
            if (!cmdBufer.Begin())
            {
                return false;
            }
#endif // VULKAN
        }
    }
    
    // Scene drawables
    for (const auto& sceneDrawable : m_SceneDrawables)
    {
        AddDrawableToCmdBuffers(sceneDrawable, m_RenderPassData[RP_SCENE].ObjectsCmdBuffer.data(), 1, static_cast<uint32_t>(m_RenderPassData[RP_SCENE].ObjectsCmdBuffer.size()));
    }

    // Blit quad drawable
    AddDrawableToCmdBuffers(*m_BlitQuadDrawable.get(), m_RenderPassData[RP_BLIT].ObjectsCmdBuffer.data(), 1, static_cast<uint32_t>(m_RenderPassData[RP_BLIT].ObjectsCmdBuffer.size()));

    // End recording
    for (uint32_t whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        auto& renderPassData = m_RenderPassData[whichPass];

        for (uint32_t whichBuffer = 0; whichBuffer < renderPassData.ObjectsCmdBuffer.size(); whichBuffer++)
        {
            auto& cmdBufer = renderPassData.ObjectsCmdBuffer[whichBuffer];
            if (!cmdBufer.End())
            {
                return false;
            }
        }
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
    tGfxApi* const pGfxApi = GetGfxApi();

    // Vert data
    {
        glm::mat4 LocalModel = glm::mat4(1.0f);
        LocalModel           = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
        LocalModel           = glm::scale(LocalModel, glm::vec3(1.0f));
        glm::mat4 LocalMVP   = m_Camera.ProjectionMatrix() * m_Camera.ViewMatrix() * LocalModel;

        //m_ObjectVertUniformData.MVPMatrix = glm::transpose( LocalMVP );
        m_ObjectVertUniformData.MVPMatrix = LocalMVP;
        m_ObjectVertUniformData.ModelMatrix = LocalModel;
        UpdateUniformBuffer(pGfxApi, m_ObjectVertUniform, m_ObjectVertUniformData, whichBuffer);
    }

    // Frag data
    {
        m_ObjectFragUniformData.Color        = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
        m_ObjectFragUniformData.NormalHeight = glm::vec4(gNormalAmount, gNormalMirrorReflectAmount, 0.0f, 0.0f);

        UpdateUniformBuffer(pGfxApi, m_ObjectFragUniform, m_ObjectFragUniformData, whichBuffer);
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

        UpdateUniformBuffer(pGfxApi, m_LightUniform, m_LightUniformData, whichBuffer);
    }

    return true;
}

//-----------------------------------------------------------------------------
void Application::Render( float fltDiffTime )
//-----------------------------------------------------------------------------
{
    tGfxApi* const pGfxApi = GetGfxApi();

    // ********************************
    // Application Draw() - Begin
    // ********************************

    UpdateGui();

    // Update camera
    m_Camera.UpdateController( fltDiffTime, *m_CameraController );
    m_Camera.UpdateMatrices();

    // Obtain the next swap chain image for the next frame.
    auto currentBackBuffer = pGfxApi->SetNextBackBuffer();
    uint32_t whichBuffer = currentBackBuffer.idx;

    // Update uniform buffers with latest data
    UpdateUniforms( whichBuffer );

#if VULKAN
    // Open the command list for recording commands
    auto& commandList = m_CommandLists[whichBuffer];
    commandList.Begin();

    // RP_SCENE
    {
        const auto& renderPassData = m_RenderPassData[RP_SCENE];

        VkRect2D passArea = {};
        passArea.extent.width = renderPassData.RenderTarget.m_Width;
        passArea.extent.height = renderPassData.RenderTarget.m_Height;
        VkClearColorValue clearColor = {renderPassData.PassSetup.ClearColor[0], renderPassData.PassSetup.ClearColor[1], renderPassData.PassSetup.ClearColor[2], renderPassData.PassSetup.ClearColor[3]};

        commandList.BeginRenderPass(
            passArea,
            0.0f,
            1.0f,
            {&clearColor, 1},
            1,
            true,
            renderPassData.RenderPass,
            false,
            renderPassData.RenderTarget.m_FrameBuffer,
            VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS );
//        commandList.BeginRenderPass( renderPassData.RenderTarget, renderPassData.RenderPass, VK_SUBPASS_CONTENTS_INLINE );
        vkCmdExecuteCommands( commandList.m_VkCommandBuffer, 1, &renderPassData.ObjectsCmdBuffer[whichBuffer].m_VkCommandBuffer );
        commandList.EndRenderPass();
    }
    // RP_HUD
    if (m_Gui)
    {
        const auto& renderPassData = m_RenderPassData[RP_HUD];

        // Render gui into main command list
        commandList.BeginRenderPass( renderPassData.RenderTarget, renderPassData.RenderPass, VK_SUBPASS_CONTENTS_INLINE );
        GetGui()->Render( commandList.m_VkCommandBuffer );
        commandList.EndRenderPass();
    }
    // RP_BLIT
    {
        const auto& renderPassData = m_RenderPassData[RP_BLIT];

        VkRect2D passArea = {};
        passArea.extent.width = pGfxApi->GetSurfaceWidth();
        passArea.extent.height = pGfxApi->GetSurfaceHeight();
        VkClearColorValue clearColor = {renderPassData.PassSetup.ClearColor[0], renderPassData.PassSetup.ClearColor[1], renderPassData.PassSetup.ClearColor[2], renderPassData.PassSetup.ClearColor[3]};

        commandList.BeginRenderPass(
            passArea,
            0.0f,
            1.0f,
            {&clearColor, 1},
            1,
            false,
            renderPassData.RenderPass,
            true,
            pGfxApi->m_SwapchainBuffers[currentBackBuffer.swapchainPresentIdx].framebuffer,
            VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS );
        vkCmdExecuteCommands( commandList.m_VkCommandBuffer, 1, &m_RenderPassData[RP_BLIT].ObjectsCmdBuffer[whichBuffer].m_VkCommandBuffer );
        commandList.EndRenderPass();
    }

    // Submit the commands to the queue.
    commandList.End();
    commandList.QueueSubmit(currentBackBuffer, pGfxApi->m_RenderCompleteSemaphore);

    // Queue is loaded up, tell the driver to start processing
    pGfxApi->PresentQueue(pGfxApi->m_RenderCompleteSemaphore, currentBackBuffer.swapchainPresentIdx);

#else //VULKAN
    // Open the command list for recording commands
    auto& commandList = m_CommandLists[whichBuffer];
    commandList.Begin();

    pGfxApi->SetDescriptorHeaps(commandList.Get());

#if 0
    // RP_SCENE
    m_RenderPassData[RP_SCENE].RenderTarget.SetRenderTarget( commandList );
    commandList->ExecuteBundle( m_RenderPassData[RP_SCENE].ObjectsCmdBuffer[whichBuffer].Get() );

    // RP_HUD
    if (m_Gui)
    {
        // Render gui (has its own command lists, optionally returns nullptr if not rendering anything)
        m_RenderPassData[RP_HUD].RenderTarget.SetRenderTarget( commandList );
        GetGui()->Render(commandList.Get());
        pGfxApi->SetDescriptorHeaps( commandList.Get() );
    }

    // Blit Results to the screen
    pGfxApi->BackbufferRenderSetup( whichBuffer, *commandList );
    commandList->ExecuteBundle( m_RenderPassData[RP_BLIT].ObjectsCmdBuffer[whichBuffer].Get() );
#else
    // Add everything inline

    // Scene drawables
    m_RenderPassData[RP_SCENE].RenderTarget.SetRenderTarget( commandList );
    for (const auto& sceneDrawable : m_SceneDrawables)
    {
        AddDrawableToCmdBuffers( sceneDrawable, &commandList, 1, 1 );
    }

    // RP_HUD
    if (m_Gui)
    {
        // Render gui (has its own command lists, optionally returns nullptr if not rendering anything)
        m_RenderPassData[RP_HUD].RenderTarget.SetRenderTarget( commandList );
        GetGui()->Render( commandList.Get() );
        pGfxApi->SetDescriptorHeaps( commandList.Get() );
    }

    // Blit quad drawable
    pGfxApi->BackbufferRenderSetup( whichBuffer, *commandList );
    AddDrawableToCmdBuffers( *m_BlitQuadDrawable.get(), &commandList, 1, 1 );
#endif

    // Transition backbuffer to be 'presentable'
    pGfxApi->BackbufferPresentSetup( whichBuffer, commandList.Get() );

    // Close the command list
    commandList.End();

    // Submit the commands to the queue.
    pGfxApi->CommandListExecute( *commandList );

    {
        // Open the compute command list for recording commands
        auto& commandList = m_ComputeCommandLists[whichBuffer];
        commandList.Begin();
        commandList.End();

        // Submit the commands to the queue.
        pGfxApi->ComputeCommandListExecute( *commandList );
    }

    //m_RenderPassData[RP_BLIT].PassCmdBuffer[whichBuffer].End();
        //m_RenderPassData[RP_BLIT].PassCmdBuffer[whichBuffer].QueueSubmit(WaitSemaphores, WaitDstStageMasks, SignalSemaphores, CompletionFence);
        
        
        //      pWaitSemaphores = { &m_RenderPassData[RP_BLIT].PassCompleteSemaphore,1 };

    // Queue is loaded up, tell the driver to start processing
    pGfxApi->PresentSwapchain();
#endif // VULKAN

    // ********************************
    // Application Draw() - End
    // ********************************
}
