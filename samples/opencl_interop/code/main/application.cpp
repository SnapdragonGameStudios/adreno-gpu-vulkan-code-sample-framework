//============================================================================================================
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "application.hpp"
#include "main/applicationEntrypoint.hpp"
#include "camera/cameraController.hpp"
#include "camera/cameraControllerTouch.hpp"
#include "camera/cameraData.hpp"
#include "gui/imguiVulkan.hpp"
#include "material/vulkan/drawable.hpp"
#include "material/vulkan/materialManager.hpp"
#include "material/vulkan/shaderManager.hpp"
#include "material/vulkan/shaderModule.hpp"
#include "material/vulkan/specializationConstantsLayout.hpp"
#include "mesh/meshHelper.hpp"
#include "system/math_common.hpp"
#include "texture/vulkan/textureManager.hpp"
#include "imgui.h"
#include "interop_common.hpp"
#include "interop_buffer.hpp"
#include "interop_image_linear.hpp"
#include "interop_image_optimal.hpp"

#include <iostream>

VAR( int, gInteropMode, INTEROP_NONE, kVariableNonpersistent );

namespace
{
    static constexpr std::array<const char*, NUM_RENDER_PASSES> sRenderPassNames = { "RP_SCENE", "RP_HUD", "RP_BLIT" };
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

Application::Application()
    : ApplicationHelperBase()
{
}

Application::~Application()
{
}

//-----------------------------------------------------------------------------
void Application::PreInitializeSetVulkanConfiguration(Vulkan::AppConfiguration& appConfig)
//-----------------------------------------------------------------------------
{
    appConfig.SwapchainDepthFormat = TextureFormat::UNDEFINED;
    appConfig.RequiredExtension(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
    appConfig.RequiredExtension(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
    appConfig.RequiredExtension(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    appConfig.RequiredExtension(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
}

//-----------------------------------------------------------------------------
bool Application::Initialize(uintptr_t windowHandle, uintptr_t hInstance)
//-----------------------------------------------------------------------------
{
    if (!ApplicationHelperBase::Initialize(windowHandle, hInstance))
        return false;

    gRenderWidth  = gSurfaceWidth;
    gRenderHeight = gSurfaceHeight;

    if (!InitializeCamera())    return false;
    if (!LoadShaders())         return false;
    if (!InitUniforms())        return false;
    if (!CreateRenderTargets()) return false;
    if (!InitAllRenderPasses()) return false;
    if (!InitGui(windowHandle)) return false;
    if (!InitInteropContext())  return false;
    if (!LoadMeshObjects())     return false;
    if (!InitCommandBuffers())  return false;
    if (!InitLocalSemaphores()) return false;

    return true;
}

//-----------------------------------------------------------------------------
void Application::Destroy()
//-----------------------------------------------------------------------------
{
    auto* const pVulkan = GetVulkan();

    // Release interop contexts first (they reference CLState)
    if (m_interop_buffer_context)  { m_interop_buffer_context->Release(*GetVulkan());  m_interop_buffer_context.reset();  }
    if (m_interop_linear_context)  { m_interop_linear_context->Release(*GetVulkan());  m_interop_linear_context.reset();  }
    if (m_interop_optimal_context) { m_interop_optimal_context->Release(*GetVulkan()); m_interop_optimal_context.reset(); }
    // Release shared CL state last
    m_cl_state.Release(pVulkan->m_VulkanDevice);

    // Cmd buffers
    for (auto& cmdBuffer : m_SceneCommandList)
        cmdBuffer.Release();
    for (auto& cmdBuffer : m_BlitCommandList)
        cmdBuffer.Release();

    for (int whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        m_RenderPassData[whichPass].RenderTarget.Release();
    }

    // Semaphores
    vkDestroySemaphore(pVulkan->m_VulkanDevice, m_SceneCompleteSemaphore, nullptr);
    vkDestroySemaphore(pVulkan->m_VulkanDevice, m_BlitCompleteSemaphore, nullptr);
    m_SceneCompleteSemaphore = VK_NULL_HANDLE;
    m_BlitCompleteSemaphore = VK_NULL_HANDLE;

    // Drawables
    m_SceneQuadDrawable.reset();
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

    m_Camera.SetPosition(glm::vec3(0.0f, 0.0f, 3.0f), glm::quat(glm::vec3(0.0f, 0.0f, 0.0f)));
    m_Camera.SetAspect(float(gRenderWidth) / float(gRenderHeight));
    m_Camera.SetFov(glm::radians(45.0f));
    m_Camera.SetClipPlanes(0.1f, 100.0f);

    // This sample is only for Android
    typedef CameraControllerTouch tCameraController;

    auto cameraController = std::make_unique<tCameraController>();
    if (!cameraController->Initialize(gSurfaceWidth, gSurfaceHeight))
        return false;

    m_CameraController = std::move(cameraController);
    m_CameraController->SetMoveSpeed(0.4f);

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
        { tIdAndFilename{ "Blit",      "Blit.json"      },
          tIdAndFilename{ "SceneQuad", "SceneQuad.json" },
        })
    {
        if (!m_ShaderManager->AddShader(*m_AssetManager, i.first, i.second, SHADER_DESTINATION_PATH))
        {
            LOGE("Error Loading shader %s from %s", i.first.c_str(), i.second.c_str());
            LOGI("Please verify if you have all required shaders/assets packaged by CMake/Gradle");
            return false;
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::CreateRenderTargets()
//-----------------------------------------------------------------------------
{
    auto* const pVulkan = GetVulkan();

    LOGI("**************************");
    LOGI("Creating Render Targets...");
    LOGI("**************************");

    const TextureFormat SceneColorType[] = { TextureFormat::R8G8B8A8_UNORM };
    const TextureFormat HudColorType[]   = { TextureFormat::R8G8B8A8_SRGB  };
    const TEXTURE_TYPE  SceneColorTypeUsage[] = { TT_RENDER_TARGET_SAMPLED_TRANSFERSRC };
    const Msaa          SceneMsaa[] = { Msaa::Samples1 };

    if (!m_RenderPassData[RP_SCENE].RenderTarget.Initialize(pVulkan, gRenderWidth, gRenderHeight, SceneColorType, TextureFormat::UNDEFINED, "Scene RT", SceneColorTypeUsage, SceneMsaa))
    {
        LOGE("Unable to create scene render target");
        return false;
    }

    if (!m_RenderPassData[RP_HUD].RenderTarget.Initialize(pVulkan, gSurfaceWidth, gSurfaceHeight, HudColorType, TextureFormat::UNDEFINED, Msaa::Samples1, "HUD RT"))
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
    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitAllRenderPasses()
//-----------------------------------------------------------------------------
{
    auto& vulkan = *GetVulkan();

    //                                             ColorInputUsage |               ClearDepth | ColorOutputUsage |                    DepthOutputUsage |                   ClearColor
    m_RenderPassData[RP_SCENE].RenderPassSetup = { RenderPassInputUsage::Clear,    false,       RenderPassOutputUsage::StoreReadOnly, RenderPassOutputUsage::Discard,     {}};
    m_RenderPassData[RP_HUD].RenderPassSetup   = { RenderPassInputUsage::Clear,    false,       RenderPassOutputUsage::StoreReadOnly, RenderPassOutputUsage::Discard,     {}};
    m_RenderPassData[RP_BLIT].RenderPassSetup  = { RenderPassInputUsage::DontCare, true,        RenderPassOutputUsage::Present,       RenderPassOutputUsage::Discard,     {}};

    TextureFormat surfaceFormat       = vulkan.m_SurfaceFormat;
    auto          swapChainColorFmt   = std::span<const TextureFormat>({ &surfaceFormat, 1 });
    TextureFormat swapChainDepthFmt   = vulkan.m_SwapchainDepth.format;

    LOGI("******************************");
    LOGI("Initializing Render Passes... ");
    LOGI("******************************");

    for (uint32_t whichPass = 0; whichPass < RP_BLIT; whichPass++)
    {
        std::span<const TextureFormat> colorFormats = m_RenderPassData[whichPass].RenderTarget.m_pLayerFormats;
        TextureFormat                  depthFormat  = m_RenderPassData[whichPass].RenderTarget.m_DepthFormat;

        const auto& setup = m_RenderPassData[whichPass].RenderPassSetup;
        auto&       passData = m_RenderPassData[whichPass];

        RenderPass renderPass;
        if (!vulkan.CreateRenderPass(
                { colorFormats },
                depthFormat,
                Msaa::Samples1,
                setup.ColorInputUsage,
                setup.ColorOutputUsage,
                setup.ClearDepthRenderPass,
                setup.DepthOutputUsage,
                renderPass))
        {
            return false;
        }

        Framebuffer<Vulkan> framebuffer;
        framebuffer.Initialize(
            vulkan,
            renderPass,
            passData.RenderTarget.m_ColorAttachments,
            &passData.RenderTarget.m_DepthAttachment,
            sRenderPassNames[whichPass]);

        passData.RenderContext.push_back({ std::move(renderPass), {}, std::move(framebuffer), sRenderPassNames[whichPass] });
    }

    for (auto whichBuffer = 0; whichBuffer < vulkan.GetSwapchainBufferCount(); ++whichBuffer)
    {
        m_RenderPassData[RP_BLIT].RenderContext.push_back({ vulkan.m_SwapchainRenderPass.Copy(), {}, vulkan.GetSwapchainFramebuffer(whichBuffer), "RP_BLIT" });
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitGui(uintptr_t windowHandle)
//-----------------------------------------------------------------------------
{
    const auto& hudRT = m_RenderPassData[RP_HUD].RenderTarget;
    m_Gui = std::make_unique<GuiImguiGfx>(*GetVulkan(), m_RenderPassData[RP_HUD].RenderContext[0].GetRenderPass().Copy());
    if (!m_Gui->Initialize(windowHandle, TextureFormat::R8G8B8A8_UNORM, hudRT.m_Width, hudRT.m_Height))
        return false;
    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitInteropContext()
//-----------------------------------------------------------------------------
{
    auto& vulkan = *GetVulkan();

    // Load VK external memory/semaphore function pointers
    if (!InitVkFunctionPointers(vulkan.m_VulkanDevice))
        return false;

    // Initialize shared CL state (device + context + queue + semaphores)
    if (!m_cl_state.Initialize(vulkan, vulkan.m_VulkanDevice))
        return false;

    TextureVulkan* sceneColor = &m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[0];

    // Buffer interop using cl_external_memory_opaque_fd
    m_interop_buffer_context = std::make_unique<InteropBuffer::Context>();
    if (!m_interop_buffer_context->Initialize(vulkan, *m_AssetManager, m_cl_state, sceneColor))
        return false;

    // Linear-tiling image interop using cl_external_memory_opaque_fd
    m_interop_linear_context = std::make_unique<InteropImageLinear::Context>();
    if (!m_interop_linear_context->Initialize(vulkan, *m_AssetManager, m_cl_state, sceneColor))
        return false;

    // Optimal-tiling image interop using cl_qcom_external_memory_vulkan_opaque_fd
    m_interop_optimal_context = std::make_unique<InteropImageOptimal::Context>();
    if (!m_interop_optimal_context->Initialize(vulkan, *m_AssetManager, m_cl_state, sceneColor))
        return false;

    return true;
}

//-----------------------------------------------------------------------------
bool Application::LoadMeshObjects()
//-----------------------------------------------------------------------------
{
    auto& vulkan = *GetVulkan();

    LOGI("***********************");
    LOGI("Initializing Meshes... ");
    LOGI("***********************");

    const auto* pSceneQuadShader = m_ShaderManager->GetShader("SceneQuad");
    const auto* pBlitQuadShader  = m_ShaderManager->GetShader("Blit");
    if (!pSceneQuadShader || !pBlitQuadShader)
        return false;

    // -------------------------
    // Scene quad (fullscreen)
    // -------------------------
    LOGI("***************************");
    LOGI("Creating Scene Quad mesh...");
    LOGI("***************************");

    Mesh sceneQuadMesh;
    if (!MeshHelper::CreateMesh<Vulkan>(
            vulkan.GetMemoryManager(),
            MeshObjectIntermediate::CreateScreenSpaceMesh(),
            0, pSceneQuadShader->m_shaderDescription->m_vertexFormats,
            &sceneQuadMesh))
        return false;

    auto sceneQuadMaterial = m_MaterialManager->CreateMaterial(*pSceneQuadShader, NUM_VULKAN_BUFFERS,
        [](const std::string&) -> const MaterialManager::tPerFrameTexInfo { return {}; },
        [](const std::string&) -> PerFrameBuffer { return {}; });

    m_SceneQuadDrawable = std::make_unique<Drawable>(vulkan, std::move(sceneQuadMaterial));
    if (!m_SceneQuadDrawable->Init(m_RenderPassData[RP_SCENE].RenderContext[0], std::move(sceneQuadMesh)))
        return false;

    // -------------------------
    // Blit quad (final output)
    // -------------------------
    LOGI("**************************");
    LOGI("Creating Blit Quad mesh...");
    LOGI("**************************");

    Mesh blitQuadMesh;
    if (!MeshHelper::CreateMesh<Vulkan>(
            vulkan.GetMemoryManager(),
            MeshObjectIntermediate::CreateScreenSpaceMesh(),
            0, pBlitQuadShader->m_shaderDescription->m_vertexFormats,
            &blitQuadMesh))
        return false;

    auto blitMaterial = m_MaterialManager->CreateMaterial(*pBlitQuadShader, 2,
        [this](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
        {
            if (texName == "Diffuse")
            {
                return { &m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[0] };
            }
            if (texName == "Overlay")
                return { &m_RenderPassData[RP_HUD].RenderTarget.m_ColorAttachments[0] };
            return {};
        },
        [](const std::string&) -> PerFrameBuffer { return {}; });

    m_BlitQuadDrawable = std::make_unique<Drawable>(vulkan, std::move(blitMaterial));
    if (!m_BlitQuadDrawable->Init(m_RenderPassData[RP_BLIT].RenderContext[0], std::move(blitQuadMesh)))
        return false;

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitCommandBuffers()
//-----------------------------------------------------------------------------
{
    LOGI("*******************************");
    LOGI("Initializing Command Buffers...");
    LOGI("*******************************");

    auto* const pVulkan = GetVulkan();

    for (auto& cmd : m_SceneCommandList)
    {
        if (!cmd.Initialize(pVulkan, "Scene + HUD Command List", CommandListBase::Type::Primary))
            return false;
    }

    for (auto& cmd : m_BlitCommandList)
    {
        if (!cmd.Initialize(pVulkan, "Interop + Blit Command List", CommandListBase::Type::Primary))
            return false;
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

    VkResult retVal = vkCreateSemaphore(GetVulkan()->m_VulkanDevice, &SemaphoreInfo, nullptr, &m_SceneCompleteSemaphore);
    if (!CheckVkError("vkCreateSemaphore(m_SceneCompleteSemaphore)", retVal))
        return false;

    retVal = vkCreateSemaphore(GetVulkan()->m_VulkanDevice, &SemaphoreInfo, nullptr, &m_BlitCompleteSemaphore);
    if (!CheckVkError("vkCreateSemaphore(m_BlitCompleteSemaphore)", retVal))
        return false;

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

        ImGui::SetNextWindowSize(ImVec2(1250.0f, 375.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("FPS", (bool*)nullptr, ImGuiWindowFlags_NoTitleBar))
        {
            ImGui::Text("FPS: %.1f", m_CurrentFPS);
            ImGui::Text("Frame Time: %.3f ms", 1000.0f / m_CurrentFPS);
            static std::array<const char* const, 4> sModeNames{
                "no interop",
                "buffer interop",
                "linear tiling image interop",
                "optimal tiling image interop"
            };
            ImGui::ListBox("Interop Mode", &gInteropMode, sModeNames.data(), (int)sModeNames.size());
        }
        ImGui::End();
    }
}

//-----------------------------------------------------------------------------
bool Application::UpdateUniforms(uint32_t whichBuffer)
//-----------------------------------------------------------------------------
{
    return true;
}

//-----------------------------------------------------------------------------
void Application::Render(float fltDiffTime)
//-----------------------------------------------------------------------------
{
    auto& vulkan = *GetVulkan();

    auto currentVulkanBuffer = vulkan.SetNextBackBuffer();
    uint32_t whichBuffer = currentVulkanBuffer.idx;
    uint32_t swapchainImage = currentVulkanBuffer.swapchainPresentIdx;

    UpdateGui();

    m_Camera.UpdateController(fltDiffTime, *m_CameraController);
    m_Camera.UpdateMatrices();
    UpdateUniforms(whichBuffer);

    const VkPipelineStageFlags allCommandsWaitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    // Stage 1: Vulkan scene + optional HUD before OpenCL.
    {
        auto& commandBuffer = m_SceneCommandList[whichBuffer];
        commandBuffer.Reset();
        commandBuffer.Begin();

        // RP_SCENE
        {
            const auto& renderContext = m_RenderPassData[RP_SCENE].RenderContext[0];
            const fvk::VkRenderPassBeginInfo RPBeginInfo{ renderContext.GetRenderPassBeginInfo() };
            vkCmdBeginRenderPass(commandBuffer, &RPBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

            const auto scissor = renderContext.GetRenderPassClearData().scissor;
            const auto viewport = renderContext.GetRenderPassClearData().viewport;
            vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
            vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

            AddDrawableToCmdBuffers(*m_SceneQuadDrawable, &commandBuffer, 1, 1);

            vkCmdEndRenderPass(commandBuffer);
        }

        // RP_HUD
        VkCommandBuffer guiCommandBuffer = VK_NULL_HANDLE;
        if (m_Gui)
        {
            guiCommandBuffer = GetGui()->Render(whichBuffer, m_RenderPassData[RP_HUD].RenderTarget.m_FrameBuffer);
        }

        if (guiCommandBuffer != VK_NULL_HANDLE)
        {
            const auto& renderContext = m_RenderPassData[RP_HUD].RenderContext[0];
            const fvk::VkRenderPassBeginInfo RPBeginInfo{ renderContext.GetRenderPassBeginInfo() };
            vkCmdBeginRenderPass(commandBuffer, &RPBeginInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
            vkCmdExecuteCommands(commandBuffer, 1, &guiCommandBuffer);
            vkCmdEndRenderPass(commandBuffer);
        }

        // Copy scene color to shared OpenCL resource. INTEROP_NONE skips the OpenCL path entirely.
        if (gInteropMode != INTEROP_NONE)
        {
            switch (gInteropMode)
            {
            case INTEROP_BUFFER:
                m_interop_buffer_context->CopyToSharedBuffers(vulkan, commandBuffer);
                m_interop_buffer_context->ReleaseSharedResourcesToExternal(vulkan, commandBuffer);
                break;
            case INTEROP_LINEAR_IMAGE:
                m_interop_linear_context->CopyToSharedImage(vulkan, commandBuffer);
                m_interop_linear_context->ReleaseSharedResourcesToExternal(vulkan, commandBuffer);
                break;
            case INTEROP_OPTIMAL_IMAGE:
                m_interop_optimal_context->CopyToSharedImage(vulkan, commandBuffer);
                m_interop_optimal_context->ReleaseSharedResourcesToExternal(vulkan, commandBuffer);
                break;
            default:
                break;
            }
        }

        commandBuffer.End();

        std::vector<VkSemaphore> signalSemaphores{ m_SceneCompleteSemaphore };
        if (gInteropMode != INTEROP_NONE)
        {
            signalSemaphores.push_back(m_cl_state.GetRenderingCompleteSemaphore());
        }

        commandBuffer.QueueSubmit(
            { &currentVulkanBuffer.semaphore, 1 },
            { &allCommandsWaitStage, 1 },
            std::span<const VkSemaphore>(signalSemaphores),
            VK_NULL_HANDLE);
    }

    // Stage 2: OpenCL process scene color only.
    if (gInteropMode != INTEROP_NONE)
    {
        m_cl_state.ExportVkSemaToCl(vulkan.m_VulkanDevice);
        m_cl_state.WaitForVkSignal();

        switch (gInteropMode)
        {
        case INTEROP_BUFFER:
            m_interop_buffer_context->Dispatch();
            break;
        case INTEROP_LINEAR_IMAGE:
            m_interop_linear_context->Dispatch();
            break;
        case INTEROP_OPTIMAL_IMAGE:
            m_interop_optimal_context->Dispatch();
            break;
        default:
            break;
        }

        m_cl_state.SignalVkWait();
        m_cl_state.Flush();
        m_cl_state.ExportClSemaToVk(vulkan.m_VulkanDevice);
    }

    // Stage 3: Vulkan final blit after OpenCL.
    {
        auto& commandBuffer = m_BlitCommandList[whichBuffer];
        commandBuffer.Reset();
        commandBuffer.Begin();

        if (gInteropMode != INTEROP_NONE)
        {
            switch (gInteropMode)
            {
            case INTEROP_BUFFER:
                m_interop_buffer_context->AcquireSharedResourcesFromExternal(vulkan, commandBuffer);
                m_interop_buffer_context->CopyFromSharedBuffers(vulkan, commandBuffer);
                break;
            case INTEROP_LINEAR_IMAGE:
                m_interop_linear_context->AcquireSharedResourcesFromExternal(vulkan, commandBuffer);
                m_interop_linear_context->CopyFromSharedImage(vulkan, commandBuffer);
                break;
            case INTEROP_OPTIMAL_IMAGE:
                m_interop_optimal_context->AcquireSharedResourcesFromExternal(vulkan, commandBuffer);
                m_interop_optimal_context->CopyFromSharedImage(vulkan, commandBuffer);
                break;
            default:
                break;
            }
        }

        const auto& renderContext = m_RenderPassData[RP_BLIT].RenderContext[swapchainImage];
        const fvk::VkRenderPassBeginInfo RPBeginInfo{ renderContext.GetRenderPassBeginInfo() };
        vkCmdBeginRenderPass(commandBuffer, &RPBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        const auto scissor = renderContext.GetRenderPassClearData().scissor;
        const auto viewport = renderContext.GetRenderPassClearData().viewport;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        const TextureVulkan* pOutputBuffer = nullptr;
        switch (gInteropMode)
        {
        case INTEROP_NONE:
        default:
            pOutputBuffer = &m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[0];
            break;
        case INTEROP_BUFFER:
            pOutputBuffer = m_interop_buffer_context->GetSceneColorOutput();
            break;
        case INTEROP_LINEAR_IMAGE:
            pOutputBuffer = m_interop_linear_context->GetSceneColorOutput();
            break;
        case INTEROP_OPTIMAL_IMAGE:
            pOutputBuffer = m_interop_optimal_context->GetSceneColorOutput();
            break;
        }

        static uint32_t outputBufferFlipFlop = 0;
        outputBufferFlipFlop ^= 1;
        m_BlitQuadDrawable->GetMaterial().UpdateDescriptorSetBinding(outputBufferFlipFlop, "Diffuse", *pOutputBuffer);
        AddDrawableToCmdBuffers(*m_BlitQuadDrawable, &commandBuffer, 1, 1, outputBufferFlipFlop);

        vkCmdEndRenderPass(commandBuffer);
        commandBuffer.End();

        std::vector<VkSemaphore> waitSemaphores{ m_SceneCompleteSemaphore };
        std::vector<VkPipelineStageFlags> waitStages{ allCommandsWaitStage };

        if (gInteropMode != INTEROP_NONE)
        {
            waitSemaphores.push_back(m_cl_state.GetProcessingCompleteSemaphore());
            waitStages.push_back(allCommandsWaitStage);
        }

        commandBuffer.QueueSubmit(
            std::span<const VkSemaphore>(waitSemaphores),
            std::span<const VkPipelineStageFlags>(waitStages),
            { &m_BlitCompleteSemaphore, 1 },
            currentVulkanBuffer.fence);
    }

    vulkan.PresentQueue(m_BlitCompleteSemaphore, swapchainImage);

    // This sample keeps a single set of shared VK/CL resources and external semaphores.
    // For multiple frames in flight, use per-frame shared resources and per-frame
    // external semaphores instead.
    vkWaitForFences(vulkan.m_VulkanDevice, 1, &currentVulkanBuffer.fence, VK_TRUE, UINT64_MAX);
}
