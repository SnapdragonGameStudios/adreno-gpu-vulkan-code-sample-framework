//============================================================================================================
//
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
#include "camera/cameraGltfLoader.hpp"
#include "gui/imguiVulkan.hpp"
#include "material/vulkan/computable.hpp"
#include "material/vulkan/drawable.hpp"
#include "material/drawableLoader.hpp"
#include "material/vulkan/materialManager.hpp"
#include "material/vulkan/shaderModule.hpp"
#include "material/vulkan/shaderManager.hpp"
#include "material/vulkan/specializationConstantsLayout.hpp"
#include "mesh/meshHelper.hpp"
#include "mesh/meshLoader.hpp"
#include "system/math_common.hpp"
#include "texture/vulkan/textureManager.hpp"
#include "vulkan/extensionHelpers.hpp"
#include "vulkan/extensionLib.hpp"
#include "imgui.h"

#include <random>
#include <iostream>
#include <filesystem>

namespace
{
    static constexpr std::array<const char*, NUM_RENDER_PASSES> sRenderPassNames = { "RP_HUD", "RP_BLIT" };

    glm::vec3 gCameraStartPos = glm::vec3(26.48f, 20.0f, -5.21f);
    glm::vec3 gCameraStartRot = glm::vec3(0.0f, 110.0f, 0.0f);

    float   gFOV = PI_DIV_4;
    float   gNearPlane = 1.0f;
    float   gFarPlane = 1800.0f;

    // ---- VK_QCOM_cooperative_matrix_conversion, defined locally ----------------
    // The framework builds against a vendored Vulkan-Headers submodule that
    // predates this extension, so neither VkPhysicalDeviceCooperativeMatrixConversionFeaturesQCOM
    // nor its VK_STRUCTURE_TYPE_* enum exist there. The SPIR-V capability
    // CooperativeMatrixConversionQCOM emitted by the coopmat shaders is only legal
    // when this extension's FEATURE bit is enabled at vkCreateDevice (enabling the
    // extension string alone is not sufficient — vkCreateShaderModule rejects the
    // capability otherwise). So we declare the feature struct + its sType value
    // (from the Vulkan registry; ABI-stable) locally and chain it into the device
    // pNext via the framework's VulkanDeviceFeaturesExtensionHelper. The helper
    // queries AvailableFeatures through vkGetPhysicalDeviceFeatures2 and, by
    // default, requests exactly what is available — so the feature is enabled iff
    // the driver actually supports it.
    struct QcomCoopMatConversionFeatures
    {
        VkStructureType sType;
        void*           pNext;
        VkBool32        cooperativeMatrixConversion;
    };
    // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_CONVERSION_FEATURES_QCOM
    static constexpr VkStructureType kSTypeQcomCoopMatConvFeatures =
        static_cast<VkStructureType>(1000172000);

    class QcomCoopMatConversionExtension final
        : public VulkanDeviceFeaturesExtensionHelper<QcomCoopMatConversionFeatures, kSTypeQcomCoopMatConvFeatures>
    {
    public:
        explicit QcomCoopMatConversionExtension(VulkanExtensionStatus status)
            : VulkanDeviceFeaturesExtensionHelper("VK_QCOM_cooperative_matrix_conversion", status) {}
        void PrintFeatures() const override
        {
            LOGI("VK_QCOM_cooperative_matrix_conversion: cooperativeMatrixConversion = %s",
                 AvailableFeatures.cooperativeMatrixConversion ? "True" : "False");
        }
    };
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
    // We deliberately do NOT force a higher instance API version. The runtime
    // shader compiler targets Vulkan 1.2 / SPIR-V 1.5 and the coopmat shaders use
    // a literal workgroup size (no LocalSizeId), so nothing here needs Vulkan 1.3
    // or the maintenance4 feature. Forcing 1.3 previously destabilized normal
    // frame rendering on some drivers (intermittent VK_ERROR_INITIALIZATION_FAILED).
    config.RequiredExtension<ExtensionLib::Ext_VK_KHR_get_physical_device_properties2>();
    config.RequiredExtension<ExtensionLib::Ext_VK_KHR_synchronization2>();
    config.RequiredExtension<ExtensionLib::Ext_VK_KHR_create_renderpass2>();
    config.RequiredExtension<ExtensionLib::Ext_VK_KHR_buffer_device_address>();
    config.RequiredExtension<ExtensionLib::Ext_VK_KHR_8bit_storage>();
    // fp16 storage + arithmetic are needed by all the MLP shaders.
    config.RequiredExtension<ExtensionLib::Ext_VK_KHR_shader_float16_int8>();
    // vulkanMemoryModel is required by the cooperative-matrix / memory-scope
    // shaders (they declare the VulkanMemoryModel SPIR-V capability). Enable the
    // FEATURE (not just the extension) so those modules validate. Harmless for
    // the ALU path.
    config.RequiredExtension<ExtensionLib::Ext_VK_KHR_vulkan_memory_model>();
    // Coopmat mode is optional; if not present the runner disables coopmat mode on the UI.
    config.OptionalExtension<ExtensionLib::Ext_VK_KHR_cooperative_matrix>();
    // Register VK_QCOM_cooperative_matrix_conversion as OPTIONAL. All coopmat
    // strategies use the QCOM vector<->coopmat conversion ops for their output
    // store, and the SPIR-V CooperativeMatrixConversionQCOM capability those emit
    // is only legal when this extension's FEATURE bit is enabled at vkCreateDevice
    // (vkCreateShaderModule rejects the capability otherwise). Registering the
    // extension string alone is NOT enough — the feature struct must be chained
    // into the device pNext. We do that via QcomCoopMatConversionExtension (a
    // VulkanDeviceFeaturesExtensionHelper with a locally-declared feature struct,
    // since the vendored Vulkan-Headers predate this extension). Optional means it
    // is enabled iff the driver advertises it and NEVER fails vkCreateDevice when
    // absent; the runner (FusedMlpRunner::InitializeRunner) greys out coopmat mode
    // when it is not loaded, so the QCOM ops are never compiled on an unsupported
    // device.
    config.OptionalExtension<QcomCoopMatConversionExtension>();
}

//-----------------------------------------------------------------------------
bool Application::Initialize(uintptr_t windowHandle, uintptr_t hInstance)
//-----------------------------------------------------------------------------
{
    if (!ApplicationHelperBase::Initialize( windowHandle, hInstance ))
    {
        return false;
    }

    // Always construct the runner.  It detects which extensions are loaded and
    // greys-out the modes/strategies that are unavailable on this device.
    GetVulkan()->WaitUntilIdle();
    m_fused_mlp_runner = std::make_unique<FusedMlpRunner>(*GetVulkan());
    LOGI("Initializing fused MLP runner");
    if (!m_fused_mlp_runner->InitializeRunner())
    {
        return false;
    }
    LOGI("Fused MLP runner initialized!");

    if (!InitializeCamera())
    {
        return false;
    }

    if (!LoadShaders())
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

    if (!InitLocalSemaphores())
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

    // Runner owns Vulkan resources; release before tearing down the device.
    m_fused_mlp_runner.reset();

    // Cmd buffers
    for (int whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        for (auto& cmdBuffer : m_RenderPassData[whichPass].PassCmdBuffer)
        {
            cmdBuffer.Release();
        }

        for (auto& cmdBuffer : m_RenderPassData[whichPass].ObjectsCmdBuffer)
        {
            cmdBuffer.Release();
        }

        m_RenderPassData[whichPass].RenderTarget.Release();
    }

    // Render passes / Context / Semaphores
    for (int whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        vkDestroySemaphore(pVulkan->m_VulkanDevice, m_RenderPassData[whichPass].PassCompleteSemaphore, nullptr);
        m_RenderPassData[whichPass].RenderContext.clear();
    }

    // Drawables
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
    m_ShaderManager = std::make_unique<ShaderManager>(*GetVulkan());
    m_ShaderManager->RegisterRenderPassNames(sRenderPassNames);

    m_MaterialManager = std::make_unique<MaterialManager>(*GetVulkan());

    LOGI("******************************");
    LOGI("Loading Shaders...");
    LOGI("******************************");

    typedef std::pair<std::string, std::string> tIdAndFilename;
    for (const tIdAndFilename& i :
            { tIdAndFilename { "Blit",  "Blit.json" }
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

//-----------------------------------------------------------------------------
bool Application::CreateRenderTargets()
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();

    LOGI("**************************");
    LOGI("Creating Render Targets...");
    LOGI("**************************");

    const TextureFormat HudColorType[]  = { TextureFormat::R8G8B8A8_SRGB };

    // Notice no depth on the HUD RT
    if (!m_RenderPassData[RP_HUD].RenderTarget.Initialize(pVulkan, gSurfaceWidth, gSurfaceHeight, HudColorType, TextureFormat::UNDEFINED, Msaa::Samples1, "HUD RT"))
    {
        LOGE("Unable to create hud render target");
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitAllRenderPasses()
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();

    //                                       ColorInputUsage |               ClearDepthRenderPass | ColorOutputUsage |                     DepthOutputUsage |              ClearColor
    m_RenderPassData[RP_HUD].PassSetup   = { RenderPassInputUsage::Clear,    false,                 RenderPassOutputUsage::StoreReadOnly,  RenderPassOutputUsage::Discard, {}};
    m_RenderPassData[RP_BLIT].PassSetup  = { RenderPassInputUsage::DontCare, true,                  RenderPassOutputUsage::Present,        RenderPassOutputUsage::Discard, {}};

    TextureFormat surfaceFormat = pVulkan->m_SurfaceFormat;
    auto swapChainColorFormat = std::span<const TextureFormat>({ &surfaceFormat, 1 });
    auto swapChainDepthFormat = pVulkan->m_SwapchainDepth.format;

    LOGI("******************************");
    LOGI("Initializing Render Passes... ");
    LOGI("******************************");

    for (uint32_t whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        bool isSwapChainRenderPass = whichPass == RP_BLIT;

        std::span<const TextureFormat> colorFormats = isSwapChainRenderPass ? swapChainColorFormat : m_RenderPassData[whichPass].RenderTarget.m_pLayerFormats;
        TextureFormat                  depthFormat = isSwapChainRenderPass ? swapChainDepthFormat : m_RenderPassData[whichPass].RenderTarget.m_DepthFormat;

        const auto& passSetup = m_RenderPassData[whichPass].PassSetup;
        auto& passData = m_RenderPassData[whichPass];

        RenderPass renderPass;
        if (!pVulkan->CreateRenderPass(
            { colorFormats },
            depthFormat,
            Msaa::Samples1,
            passSetup.ColorInputUsage,
            passSetup.ColorOutputUsage,
            passSetup.ClearDepthRenderPass,
            passSetup.DepthOutputUsage,
            renderPass))
        {
            return false;
        }

        Framebuffer<Vulkan> framebuffer;
        if (!isSwapChainRenderPass)
        {
            framebuffer.Initialize(*pVulkan,
                renderPass,
                passData.RenderTarget.m_ColorAttachments,
                &passData.RenderTarget.m_DepthAttachment,
                sRenderPassNames[whichPass]);
        }

        passData.RenderContext.push_back({ std::move(renderPass), {}/*pipeline*/, std::move(framebuffer), sRenderPassNames[whichPass] });
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitGui(uintptr_t windowHandle)
//-----------------------------------------------------------------------------
{
    const auto& hudRenderTarget = m_RenderPassData[RP_HUD].RenderTarget;
    m_Gui = std::make_unique<GuiImguiGfx>(*GetVulkan(), m_RenderPassData[RP_HUD].RenderContext[0].GetRenderPass().Copy());
    if (!m_Gui->Initialize(windowHandle, TextureFormat::R8G8B8A8_UNORM, hudRenderTarget.m_Width, hudRenderTarget.m_Height))
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

    const auto* pBlitQuadShader = m_ShaderManager->GetShader("Blit");
    if (!pBlitQuadShader)
    {
        return false;
    }

    LOGI("*********************");
    LOGI("Creating Quad mesh...");
    LOGI("*********************");

    Mesh blitQuadMesh;
    if (!MeshHelper::CreateMesh<Vulkan>(
        pVulkan->GetMemoryManager(),
        MeshObjectIntermediate::CreateScreenSpaceMesh(),
        0,
        pBlitQuadShader->m_shaderDescription->m_vertexFormats,
        &blitQuadMesh))
    {
        return false;
    }

    // Blit Material
    auto blitQuadShaderMaterial = m_MaterialManager->CreateMaterial(*pBlitQuadShader, 2,
        [this](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
    {
        if (texName == "Overlay")
        {
            return { &m_RenderPassData[RP_HUD].RenderTarget.m_ColorAttachments[0] };
        }
        return {};
    },
        [this](const std::string& bufferName) -> PerFrameBuffer
    {
        return {};
    }
    );

    m_BlitQuadDrawable = std::make_unique<Drawable>(*pVulkan, std::move(blitQuadShaderMaterial));
    if (!m_BlitQuadDrawable->Init(m_RenderPassData[RP_BLIT].RenderContext[0], std::move(blitQuadMesh)))
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

    m_RenderPassData[RP_HUD].PassCmdBuffer.resize(NUM_VULKAN_BUFFERS);
    m_RenderPassData[RP_HUD].ObjectsCmdBuffer.resize(NUM_VULKAN_BUFFERS);
    m_RenderPassData[RP_BLIT].PassCmdBuffer.resize(pVulkan->m_SwapchainImageCount);
    m_RenderPassData[RP_BLIT].ObjectsCmdBuffer.resize(pVulkan->m_SwapchainImageCount);

    char szName[256];
    const auto CmdBuffLevel = CommandListBase::Type::Secondary;
    for (uint32_t whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        for (uint32_t whichBuffer = 0; whichBuffer < m_RenderPassData[whichPass].PassCmdBuffer.size(); whichBuffer++)
        {
            // The Pass Command Buffer => Primary
            sprintf(szName, "Primary (%s; Buffer %d of %d)", GetPassName(whichPass), whichBuffer + 1, NUM_VULKAN_BUFFERS);
            if (!m_RenderPassData[whichPass].PassCmdBuffer[whichBuffer].Initialize(pVulkan, szName, CommandListBase::Type::Primary))
            {
                return false;
            }

            // Model => Secondary
            sprintf(szName, "Model (%s; Buffer %d of %d)", GetPassName(whichPass), whichBuffer + 1, NUM_VULKAN_BUFFERS);
            if (!m_RenderPassData[whichPass].ObjectsCmdBuffer[whichBuffer].Initialize(pVulkan, szName, CmdBuffLevel))
            {
                return false;
            }
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
        VkResult retVal = vkCreateSemaphore(GetVulkan()->m_VulkanDevice, &SemaphoreInfo, NULL, &m_RenderPassData[whichPass].PassCompleteSemaphore);
        if (!CheckVkError("vkCreateSemaphore()", retVal))
        {
            return false;
        }
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

    // Begin recording
    for (uint32_t whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        auto& renderPassData         = m_RenderPassData[whichPass];
        bool  bisSwapChainRenderPass = whichPass == RP_BLIT;

        for (uint32_t whichBuffer = 0; whichBuffer < renderPassData.ObjectsCmdBuffer.size(); whichBuffer++)
        {
            auto& cmdBufer = renderPassData.ObjectsCmdBuffer[whichBuffer];

            uint32_t targetWidth  = bisSwapChainRenderPass ? pVulkan->m_SurfaceWidth : renderPassData.RenderTarget.GetWidth();
            uint32_t targetHeight = bisSwapChainRenderPass ? pVulkan->m_SurfaceHeight : renderPassData.RenderTarget.GetHeight();

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
            VkFramebuffer whichFramebuffer = bisSwapChainRenderPass ? pVulkan->m_SwapchainBuffers[whichBuffer].framebuffer : renderPassData.RenderContext[0].GetFramebuffer()->m_FrameBuffer;
            VkRenderPass  whichRenderPass  = renderPassData.RenderContext[0].GetRenderPass().mRenderPass;

            // Objects (can render into any pass except Blit)
            if (!cmdBufer.Begin(whichFramebuffer, whichRenderPass, bisSwapChainRenderPass))
            {
                return false;
            }
            vkCmdSetViewport(cmdBufer.m_VkCommandBuffer, 0, 1, &viewport);
            vkCmdSetScissor(cmdBufer.m_VkCommandBuffer, 0, 1, &scissor);
        }
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

        if (ImGui::Begin("Fully Fused MLP", (bool*)nullptr, ImGuiWindowFlags_NoTitleBar))
        {
            ImGui::Text("FPS: %.1f", m_CurrentFPS);

            if (m_fused_mlp_runner)
            {
                m_fused_mlp_runner->RenderUI();
            }
        }
        ImGui::End();

        return;
    }
}

//-----------------------------------------------------------------------------
void Application::Render(float fltDiffTime)
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();

    if (m_fused_mlp_runner)
    {
        pVulkan->WaitUntilIdle();
        m_fused_mlp_runner->TriggerPendingTests();
    }

    // Obtain the next swap chain image for the next frame.
    auto currentVulkanBuffer = pVulkan->SetNextBackBuffer();
    uint32_t whichBuffer     = currentVulkanBuffer.idx;

    // ********************************
    // Application Draw() - Begin
    // ********************************

    UpdateGui();

    // Update camera
    m_Camera.UpdateController(fltDiffTime * 10.0f, *m_CameraController);
    m_Camera.UpdateMatrices();

    // First time through, wait for the back buffer to be ready
    std::span<const VkSemaphore> pWaitSemaphores = { &currentVulkanBuffer.semaphore, 1 };

    const VkPipelineStageFlags DefaultGfxWaitDstStageMasks[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

    // RP_HUD
    VkCommandBuffer guiCommandBuffer = VK_NULL_HANDLE;
    if (m_Gui)
    {
        // Render gui (has its own command buffer, optionally returns vk_null_handle if not rendering anything)
        guiCommandBuffer = GetGui()->Render(whichBuffer, m_RenderPassData[RP_HUD].RenderContext[0].GetFramebuffer()->m_FrameBuffer);
        if (guiCommandBuffer != VK_NULL_HANDLE)
        {
            BeginRenderPass(whichBuffer, RP_HUD, currentVulkanBuffer.swapchainPresentIdx);
            vkCmdExecuteCommands(m_RenderPassData[RP_HUD].PassCmdBuffer[whichBuffer].m_VkCommandBuffer, 1, &guiCommandBuffer);
            EndRenderPass(whichBuffer, RP_HUD);

            // Submit the commands to the queue.
            SubmitRenderPass(whichBuffer, RP_HUD, pWaitSemaphores, DefaultGfxWaitDstStageMasks, { &m_RenderPassData[RP_HUD].PassCompleteSemaphore,1 });
            pWaitSemaphores = { &m_RenderPassData[RP_HUD].PassCompleteSemaphore,1 };
        }
    }

    // Blit Results to the screen
    {
        BeginRenderPass(whichBuffer, RP_BLIT, currentVulkanBuffer.swapchainPresentIdx);
        AddPassCommandBuffer(whichBuffer, RP_BLIT);
        EndRenderPass(whichBuffer, RP_BLIT);

        // Submit the commands to the queue.
        SubmitRenderPass(whichBuffer, RP_BLIT, pWaitSemaphores, DefaultGfxWaitDstStageMasks, { &m_RenderPassData[RP_BLIT].PassCompleteSemaphore,1 }, currentVulkanBuffer.fence);
        pWaitSemaphores = { &m_RenderPassData[RP_BLIT].PassCompleteSemaphore,1 };
    }

    // Queue is loaded up, tell the driver to start processing
    pVulkan->PresentQueue(pWaitSemaphores, currentVulkanBuffer.swapchainPresentIdx);

    // ********************************
    // Application Draw() - End
    // ********************************
}

//-----------------------------------------------------------------------------
void Application::BeginRenderPass(uint32_t whichBuffer, RENDER_PASS whichPass, uint32_t WhichSwapchainImage)
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();
    auto& renderPassData         = m_RenderPassData[whichPass];
    bool  bisSwapChainRenderPass = whichPass == RP_BLIT;

    if (!m_RenderPassData[whichPass].PassCmdBuffer[whichBuffer].Reset())
    {
        LOGE("Pass (%d) command buffer Reset() failed !", whichPass);
    }

    if (!m_RenderPassData[whichPass].PassCmdBuffer[whichBuffer].Begin())
    {
        LOGE("Pass (%d) command buffer Begin() failed !", whichPass);
    }

    VkFramebuffer framebuffer = nullptr;
    switch (whichPass)
    {
    case RP_HUD:
        framebuffer = m_RenderPassData[whichPass].RenderContext[0].GetFramebuffer()->m_FrameBuffer;
        break;
    case RP_BLIT:
        framebuffer = pVulkan->m_SwapchainBuffers[WhichSwapchainImage].framebuffer;
        break;
    default:
        framebuffer = nullptr;
        break;
    }

    assert(framebuffer != nullptr);

    VkRect2D passArea = {};
    passArea.offset.x = 0;
    passArea.offset.y = 0;
    passArea.extent.width  = bisSwapChainRenderPass ? pVulkan->m_SurfaceWidth  : renderPassData.RenderTarget.m_Width;
    passArea.extent.height = bisSwapChainRenderPass ? pVulkan->m_SurfaceHeight : renderPassData.RenderTarget.m_Height;

    TextureFormat                  swapChainColorFormat = pVulkan->m_SurfaceFormat;
    auto                           swapChainColorFormats = std::span<const TextureFormat>({ &swapChainColorFormat, 1 });
    TextureFormat                  swapChainDepthFormat = pVulkan->m_SwapchainDepth.format;
    std::span<const TextureFormat> colorFormats         = bisSwapChainRenderPass ? swapChainColorFormats : m_RenderPassData[whichPass].RenderTarget.m_pLayerFormats;
    TextureFormat                  depthFormat          = bisSwapChainRenderPass ? swapChainDepthFormat : m_RenderPassData[whichPass].RenderTarget.m_DepthFormat;

    VkClearColorValue clearColor = { renderPassData.PassSetup.ClearColor[0], renderPassData.PassSetup.ClearColor[1], renderPassData.PassSetup.ClearColor[2], renderPassData.PassSetup.ClearColor[3] };

    m_RenderPassData[whichPass].PassCmdBuffer[whichBuffer].BeginRenderPass(
        passArea,
        0.0f,
        1.0f,
        { &clearColor , 1 },
        (uint32_t)colorFormats.size(),
        depthFormat != TextureFormat::UNDEFINED,
        m_RenderPassData[whichPass].RenderContext[0].GetRenderPass().mRenderPass,
        bisSwapChainRenderPass,
        framebuffer,
        VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
}


//-----------------------------------------------------------------------------
void Application::AddPassCommandBuffer(uint32_t whichBuffer, RENDER_PASS whichPass)
//-----------------------------------------------------------------------------
{
    if (m_RenderPassData[whichPass].ObjectsCmdBuffer[whichBuffer].m_NumDrawCalls)
    {
        vkCmdExecuteCommands(m_RenderPassData[whichPass].PassCmdBuffer[whichBuffer].m_VkCommandBuffer, 1, &m_RenderPassData[whichPass].ObjectsCmdBuffer[whichBuffer].m_VkCommandBuffer);
    }
}

//-----------------------------------------------------------------------------
void Application::EndRenderPass(uint32_t whichBuffer, RENDER_PASS whichPass)
//-----------------------------------------------------------------------------
{
    m_RenderPassData[whichPass].PassCmdBuffer[whichBuffer].EndRenderPass();
}

//-----------------------------------------------------------------------------
void Application::SubmitRenderPass(uint32_t whichBuffer, RENDER_PASS whichPass, const std::span<const VkSemaphore> WaitSemaphores, const std::span<const VkPipelineStageFlags> WaitDstStageMasks, std::span<VkSemaphore> SignalSemaphores, VkFence CompletionFence)
//-----------------------------------------------------------------------------
{
    m_RenderPassData[whichPass].PassCmdBuffer[whichBuffer].End();
    m_RenderPassData[whichPass].PassCmdBuffer[whichBuffer].QueueSubmit(WaitSemaphores, WaitDstStageMasks, SignalSemaphores, CompletionFence);
}
