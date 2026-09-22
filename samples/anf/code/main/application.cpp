//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "application.hpp"
#include "anf_gpu_profiler.hpp"
#include "anf_dispatch_timing.hpp"
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
#include "vulkan/extensionLib.hpp"
#include "vulkan/renderContext.hpp"

#include <random>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cassert>

VAR( char*,     gSceneAssetModel,        "SteamPunkSauna.gltf", kVariableNonpersistent );
VAR( float,     gSceneScale,             1.0f,  kVariableNonpersistent );
VAR( glm::vec3, gCameraStartPos,         glm::vec3(0.0f, 5.0f, -15.0f), kVariableNonpersistent );
VAR( glm::vec3, gCameraStartRot,         glm::vec3(0.0f, 180.0f, 0.0f), kVariableNonpersistent );
VAR( float,     gNearPlane,              0.3f,   kVariableNonpersistent );
VAR( float,     gFarPlane,               1800.0f, kVariableNonpersistent );
VAR( float,     gFov,                    45.0f, kVariableNonpersistent );
VAR( int,       gUpscaleMode,            1, kVariableNonpersistent );

VAR( bool,      gUpscalingEnabled,       false, kVariableNonpersistent );
VAR( bool,      gFrameGenerationEnabled, false, kVariableNonpersistent );
VAR( bool,      gAnfIgnoreAdbRuntimeCommands, false, kVariableNonpersistent );
VAR( bool,      gAnfUseInverseDepth,    true,  kVariableNonpersistent );
VAR( int,       gAnfSrQualityMode,      0,     kVariableNonpersistent);
VAR( bool,      gAnfDispatchImmediate,  false, kVariableNonpersistent);
VAR( int,       gAnfMaxFramesInFlight,  NUM_VULKAN_BUFFERS, kVariableNonpersistent);

VAR( bool,      gAnimationPaused,        false, kVariableNonpersistent );
VAR( bool,      gLoadSauna,              true,  kVariableNonpersistent );

namespace
{
    uint32_t ClampAnfMaxFramesInFlight(int configuredFramesInFlight)
    {
        return static_cast<uint32_t>(std::clamp(configuredFramesInFlight, 1, int(NUM_VULKAN_BUFFERS)));
    }

    uint32_t AnfFrameIndex(uint32_t appFrameIndex, uint32_t anfFramesInFlight)
    {
        const uint32_t frameCount = std::max(1u, anfFramesInFlight);
        return appFrameIndex % frameCount;
    }

    const char* AnfCompatibilityUsageName(VkImageUsageFlags usage)
    {
        switch (usage)
        {
        case VK_IMAGE_USAGE_TRANSFER_SRC_BIT: return "TRANSFER_SRC";
        case VK_IMAGE_USAGE_TRANSFER_DST_BIT: return "TRANSFER_DST";
        case VK_IMAGE_USAGE_SAMPLED_BIT: return "SAMPLED";
        case VK_IMAGE_USAGE_STORAGE_BIT: return "STORAGE";
        case VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT: return "COLOR_ATTACHMENT";
        case VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT: return "DEPTH_STENCIL_ATTACHMENT";
        case VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT: return "TRANSIENT_ATTACHMENT";
        case VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT: return "INPUT_ATTACHMENT";
        default: return "UNKNOWN";
        }
    }

    bool AnfCompatibilityAssert(bool condition, const char* label, const char* message)
    {
        if (!condition)
        {
            LOGE("ANF resource compatibility failed for %s: %s", label, message);
            assert(condition);
            return false;
        }
        return true;
    }

    bool ValidateAnfTextureCompatibility(
        const char* label,
        const TextureVulkan& texture,
        TextureFormat expectedFormat,
        VkImageUsageFlags requiredUsage,
        VkImageAspectFlags expectedAspectMask,
        VkSampleCountFlagBits expectedSamples = VK_SAMPLE_COUNT_1_BIT)
    {
        if (!AnfCompatibilityAssert(static_cast<bool>(texture), label, "texture was not created"))
            return false;

        bool ok = true;
        if (expectedFormat != TextureFormat::UNDEFINED)
        {
            ok &= AnfCompatibilityAssert(
                texture.Format == expectedFormat,
                label,
                "format does not match ANF requirement");
        }

        const auto& props = texture.GetProperties();
        if (requiredUsage != 0)
        {
            ok &= AnfCompatibilityAssert(
                (props.Usage & requiredUsage) == requiredUsage,
                label,
                "created VkImageUsageFlags do not include all ANF-required usage bits");
        }

        if (expectedAspectMask != 0)
        {
            ok &= AnfCompatibilityAssert(
                (props.AspectMask & expectedAspectMask) == expectedAspectMask,
                label,
                "image view aspect mask does not match expected resource aspect");
        }

        ok &= AnfCompatibilityAssert(
            props.Samples == expectedSamples,
            label,
            "sample count is not compatible with ANF single-sample resources");

        ok &= AnfCompatibilityAssert(
            props.Tiling == VK_IMAGE_TILING_OPTIMAL,
            label,
            "image tiling is not optimal");

        if (!ok)
        {
            LOGE("ANF resource %s details: format=%d expectedFormat=%d usage=0x%08x requiredUsage=0x%08x aspect=0x%08x expectedAspect=0x%08x samples=0x%08x expectedSamples=0x%08x tiling=%d",
                 label,
                 static_cast<int>(texture.Format),
                 static_cast<int>(expectedFormat),
                 props.Usage,
                 requiredUsage,
                 props.AspectMask,
                 expectedAspectMask,
                 props.Samples,
                 expectedSamples,
                 props.Tiling);
            for (VkImageUsageFlags bit = 1; bit != 0; bit <<= 1)
            {
                if ((requiredUsage & bit) != 0 && (props.Usage & bit) == 0)
                    LOGE("ANF resource %s missing usage bit: %s (0x%08x)", label, AnfCompatibilityUsageName(bit), bit);
            }
        }
        return ok;
    }

    TextureFormat AnfTextureFormatOrUndefined(AnfFormat format)
    {
        return Anf::AnfInterface::ToTextureFormat(format).value_or(TextureFormat::UNDEFINED);
    }

    const char* AnfDispatchModeName(bool dispatchImmediate)
    {
        return dispatchImmediate ? "immediate" : "indirect";
    }

    template <size_t N>
    bool CreateSemaphoreArray(Vulkan* vulkan, std::array<VkSemaphore, N>& semaphores)
    {
        const VkSemaphoreCreateInfo semaphoreInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        for (auto& semaphore : semaphores)
        {
            const VkResult retVal = vkCreateSemaphore(vulkan->m_VulkanDevice, &semaphoreInfo, nullptr, &semaphore);
            if (!CheckVkError("vkCreateSemaphore()", retVal)) return false;
        }
        return true;
    }

    template <size_t N>
    void DestroySemaphoreArray(VkDevice device, std::array<VkSemaphore, N>& semaphores)
    {
        for (auto& semaphore : semaphores)
        {
            if (semaphore != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(device, semaphore, nullptr);
                semaphore = VK_NULL_HANDLE;
            }
        }
    }

    static constexpr std::array<const char*const, NUM_RENDER_PASSES> sRenderPassNames = { "RP_SCENE", "RP_HUD", "RP_BLIT" };

    float   gNormalAmount = 0.3f;
    float   gNormalMirrorReflectAmount = 0.05f;

    glm::mat4 rotateEuler(const glm::mat4& input, const glm::vec3& angles)
    {
        glm::mat4x4 rotate;
        rotate = glm::rotate(input, glm::radians(angles.x), glm::vec3(1.0f, 0.0f, 0.0f));
        rotate = glm::rotate(rotate, glm::radians(angles.y), glm::vec3(0.0f, 1.0f, 0.0f));
        rotate = glm::rotate(rotate, glm::radians(angles.z), glm::vec3(0.0f, 0.0f, 1.0f));
        return rotate;
    }

    glm::vec3 lerp(const glm::vec3& a, const glm::vec3& b, const float& t)
    {
        return glm::vec3(
            std::lerp(a.x, b.x, t),
            std::lerp(a.y, b.y, t),
            std::lerp(a.z, b.z, t)
        );
    }

    float clamp(float x, float lowerlimit = 0.0f, float upperlimit = 1.0f) {
        if (x < lowerlimit) return lowerlimit;
        if (x > upperlimit) return upperlimit;
        return x;
    }

    float smoothstep_float(float edge0, float edge1, float x) {
        x = clamp((x - edge0) / (edge1 - edge0));
        return x * x * (3.0f - 2.0f * x);
    }

    glm::vec3 smoothstep(const glm::vec3& a, const glm::vec3& b, const float& t)
    {
        return glm::vec3(
            smoothstep_float(a.x, b.x, t),
            smoothstep_float(a.y, b.y, t),
            smoothstep_float(a.z, b.z, t)
        );
    }

    MeshObjectIntermediate CreateBasicCubeIntermediate()
    {
        MeshObjectIntermediate cubeMeshIntermediate;
        cubeMeshIntermediate.m_VertexBuffer.resize(24);

		std::array<glm::vec3, 24> positions = {{
            { -0.5f, -0.5f, -0.5f }, { 0.5f, -0.5f, -0.5f }, { -0.5f, 0.5f, -0.5f }, { 0.5f, 0.5f, -0.5f },
            { -0.5f, 0.5f, -0.5f }, { 0.5f, 0.5f, -0.5f }, { -0.5f, 0.5f, 0.5f }, { 0.5f, 0.5f, 0.5f },
            { 0.5f, -0.5f, -0.5f }, { 0.5f, -0.5f, 0.5f }, { 0.5f, 0.5f, -0.5f }, { 0.5f, 0.5f, 0.5f },
            { 0.5f, -0.5f, 0.5f }, { -0.5f, -0.5f, 0.5f }, { 0.5f, 0.5f, 0.5f }, { -0.5f, 0.5f, 0.5f },
            { -0.5f, -0.5f, 0.5f }, { -0.5f, -0.5f, -0.5f }, { -0.5f, 0.5f, 0.5f }, { -0.5f, 0.5f, -0.5f },
            { -0.5f, -0.5f, 0.5f }, { 0.5f, -0.5f, 0.5f }, { -0.5f, -0.5f, -0.5f }, { 0.5f, -0.5f, -0.5f },
		}};

        std::array<glm::vec3, 6> normals = {{
            { 0.0f,  0.0f, -1.0f },
            { 0.0f,  1.0f,  0.0f },
            { 1.0f,  0.0f,  0.0f },
            { 0.0f,  0.0f,  1.0f },
            { -1.0f, 0.0f,  0.0f },
            { 0.0f, -1.0f,  0.0f },
        }};

        for (size_t i = 0; i < cubeMeshIntermediate.m_VertexBuffer.size(); i++)
        {
            auto& vert = cubeMeshIntermediate.m_VertexBuffer[i];
            vert.color[0] = vert.color[1] = vert.color[2] = vert.color[3] = 1.0f;
            vert.position[0] = positions[i].x;
            vert.position[1] = positions[i].y;
            vert.position[2] = positions[i].z;
            auto& normal = normals[i / 4];
            vert.normal[0] = normal.x; vert.normal[1] = normal.y; vert.normal[2] = normal.z;
            auto& tangent = normals[((i / 4) + 1) % 6];
            vert.tangent[0] = tangent.x; vert.tangent[1] = tangent.y; vert.tangent[2] = tangent.z;
            auto bitangent = glm::cross(normal, tangent);
            vert.bitangent[0] = bitangent.x; vert.bitangent[1] = bitangent.y; vert.bitangent[2] = bitangent.z;
            uint32_t uvx = ((i % 4) & 1) >> 0;
            uint32_t uvy = ((i % 4) & 2) >> 1;
            vert.uv0[0] = uvx ? 1.0f : 0.0f;
            vert.uv0[1] = uvy ? 0.0f : 1.0f;
        }

        const std::array<uint32_t, 36> indices = {
            0,2,1, 2,3,1, 4,6,5, 6,7,5, 8,10,9, 10,11,9,
            12,14,13, 14,15,13, 16,18,17, 18,19,17, 20,22,21, 22,23,21,
        };
        cubeMeshIntermediate.m_IndexBuffer.emplace<std::vector<uint16_t>>(std::begin(indices), std::end(indices));

        return cubeMeshIntermediate;
    }
}

FrameworkApplicationBase* Application_ConstructApplication()
{
    return new Application();
}

Application::Application()
    : ApplicationHelperBase()
{
    AnfSampleUiBindings bindings{};
    bindings.DispatchImmediate       = &gAnfDispatchImmediate;
    bindings.MaxFramesInFlight       = &gAnfMaxFramesInFlight;
    bindings.UpscalingEnabled        = &gUpscalingEnabled;
    bindings.FrameGenerationEnabled  = &gFrameGenerationEnabled;
    bindings.IgnoreAdbRuntimeCommands = &gAnfIgnoreAdbRuntimeCommands;
    bindings.UseInverseDepth         = &gAnfUseInverseDepth;
    bindings.AnfSrQualityMode       = &gAnfSrQualityMode;
    bindings.AnimationPaused         = &gAnimationPaused;
    bindings.DebugForceZeroMv        = &m_Debug_ForceZeroMV;
    bindings.DebugForceDisableJitter = &m_Debug_ForceDisableJitter;
    bindings.DebugForceWaitForIdle   = &m_Debug_ForceWaitForIdle;

    m_RuntimeControls = std::make_unique<AnfRuntimeControls>(bindings);
}

Application::~Application()
{
}

//-----------------------------------------------------------------------------
void Application::PreInitializeSetVulkanConfiguration(Vulkan::AppConfiguration& appConfig)
//-----------------------------------------------------------------------------
{
    ApplicationHelperBase::PreInitializeSetVulkanConfiguration(appConfig);
    appConfig.RequiredExtension<ExtensionLib::Ext_VK_EXT_host_query_reset>();
    appConfig.RequiredExtension<ExtensionLib::Ext_VK_KHR_synchronization2>();
    appConfig.RequiredExtension<ExtensionLib::Ext_VK_KHR_create_renderpass2>();
    appConfig.RequiredExtension<ExtensionLib::Ext_VK_KHR_get_physical_device_properties2>();

    // External memory - ANF requirement
    appConfig.RequiredExtension<ExtensionLib::Ext_VK_KHR_external_memory_capabilities>();
    appConfig.RequiredExtension<ExtensionLib::Ext_VK_KHR_external_memory>();

    // External semaphore - ANF requirement
    appConfig.RequiredExtension<ExtensionLib::Ext_VK_KHR_external_semaphore_capabilities>();
    appConfig.RequiredExtension<ExtensionLib::Ext_VK_KHR_external_semaphore>();

    // Data graph - ANF requirement
    appConfig.OptionalExtension<ExtensionLib::Ext_VK_ARM_tensors>();
    appConfig.OptionalExtension<ExtensionLib::Ext_VK_ARM_data_graph>();
    appConfig.OptionalExtension<ExtensionLib::Ext_VK_QCOM_data_graph_model>();

    appConfig.OptionalExtension<ExtensionLib::Ext_VK_KHR_external_memory_fd>();
    appConfig.OptionalExtension<ExtensionLib::Ext_VK_KHR_external_semaphore_fd>();
#if defined(VK_USE_PLATFORM_WIN32_KHR)
    appConfig.OptionalExtension<ExtensionLib::Ext_VK_KHR_external_memory_win32>();
    appConfig.OptionalExtension<ExtensionLib::Ext_VK_KHR_external_semaphore_win32>();
#endif

    appConfig.SwapchainDepthFormat = TextureFormat::UNDEFINED;
}

//-----------------------------------------------------------------------------
bool Application::Initialize(uintptr_t windowHandle, uintptr_t hInstance)
//-----------------------------------------------------------------------------
{
    // Use a fixed 1080p output size for this sample.
    gRenderWidth  = 1920;
    gRenderHeight = 1080;
    m_WindowHandle = windowHandle;

    if (!ApplicationHelperBase::Initialize(windowHandle, hInstance)) return false;
    if (!InitializeCamera())    return false;
    if (!InitializeLights())    return false;
    if (!LoadShaders())         return false;
    if (!InitUniforms())        return false;

    m_AnfDispatchTiming = std::make_unique<AnfDispatchTimingController>();

    if (!InitSemaphores())      return false;

    m_GpuProfiler = std::make_unique<AnfGpuProfiler>(*GetVulkan());
    if (!m_GpuProfiler->Initialize())
    {
        LOGW("GPU timer profiling disabled (missing timer query support).");
    }

    if (!InitCommandBuffers())  return false;

    LOGI("**************************");
    LOGI("Initializing ANF...");
    LOGI("**************************");

    InitializeAnf();

    if (!CreateRenderTargets()) return false;

    if (!InitAllRenderPasses()) return false;
    if (!InitGui(windowHandle)) return false;
    if (!LoadMeshObjects())     return false;
    if (!LoadAnimatedObjects()) return false;
    if (!BuildCmdBuffers())     return false;

    return true;
}

//-----------------------------------------------------------------------------
bool Application::ReInitialize(uintptr_t windowHandle, uintptr_t hInstance)
//-----------------------------------------------------------------------------
{
    // ANF techniques retain the dimensions and Vulkan images supplied when they
    // are created. Retire them before the framework replaces the Android surface
    // and swapchain, then recreate dependent descriptors and render passes.
    m_WindowHandle = windowHandle;
    LOGI("Rebuilding ANF resources for a recreated surface");
    if (!RebuildResolutionDependentResources())
        return false;
    if (!ApplicationHelperBase::ReInitialize(windowHandle, hInstance))
        return false;
    if (!InitializeResolutionDependentResources(windowHandle))
        return false;

    LOGI("ANF surface-resource rebuild complete");
    return true;
}

//-----------------------------------------------------------------------------
VkExtent2D Application::GetAnfOutputExtent() const
//-----------------------------------------------------------------------------
{
    return {
        std::max(1u, static_cast<uint32_t>(gRenderWidth)),
        std::max(1u, static_cast<uint32_t>(gRenderHeight))
    };
}

//-----------------------------------------------------------------------------
VkExtent2D Application::GetAnfSceneExtent() const
//-----------------------------------------------------------------------------
{
    const VkExtent2D outputExtent = GetAnfOutputExtent();
    return {
        std::max(1u, outputExtent.width / 2),
        std::max(1u, outputExtent.height / 2)
    };
}

//-----------------------------------------------------------------------------
void Application::ReleaseResolutionDependentResources()
//-----------------------------------------------------------------------------
{
    auto* const pVulkan = GetVulkan();

    m_Gui.reset();
    m_BlitQuadDrawable.reset();
    m_BlitQuadDrawableSr.reset();
    m_BlitQuadDrawableFg.reset();
    m_BlitQuadDrawableFgInput.reset();
    m_SceneDrawables.clear();
    m_AnimatedDrawables.clear();
    for (auto& [id, prop] : m_AnimationProperties)
        ReleaseUniformBuffer(pVulkan, prop.objectVertUniform);
    m_AnimationProperties.clear();

    if (m_Anf)
    {
        m_Anf->Release();
        m_Anf.reset();
    }

    for (auto& tex : m_FgInputTextures)
    {
        if (tex) tex.Release(pVulkan);
    }
    for (auto& tex : m_FgDepthInputTextures)
    {
        if (tex) tex.Release(pVulkan);
    }
    for (auto& tex : m_FgInverseDepthInputTextures)
    {
        if (tex) tex.Release(pVulkan);
    }
    for (auto& tex : m_FgMotionInputTextures)
    {
        if (tex) tex.Release(pVulkan);
    }

    for (auto& cmdList : m_FgInputBlitCmdList)
        cmdList.Release();
    if (m_AnfDispatchTiming)
        m_AnfDispatchTiming->Release();

    for (auto& passData : m_RenderPassData)
    {
        for (auto& cmdList : passData.PassCommandList)
            cmdList.Release();
        for (auto& cmdList : passData.ObjectsCmdBuffer)
            cmdList.Release();
        passData.RenderContext.clear();
        passData.RenderTarget.Release();
    }

    m_FgFramePending = false;
    ClearPendingFgRealFramePresent();
    m_AnfReset = true;
    m_AnfFgReset = true;
}

//-----------------------------------------------------------------------------
bool Application::InitializeResolutionDependentResources(uintptr_t windowHandle)
//-----------------------------------------------------------------------------
{
    return InitializeAnf() &&
           CreateRenderTargets() &&
           InitAllRenderPasses() &&
           InitGui(windowHandle) &&
           LoadMeshObjects() &&
           LoadAnimatedObjects() &&
           InitCommandBuffers() &&
           BuildCmdBuffers();
}

//-----------------------------------------------------------------------------
bool Application::RebuildResolutionDependentResources()
//-----------------------------------------------------------------------------
{
    auto* const pVulkan = GetVulkan();
    vkDeviceWaitIdle(pVulkan->m_VulkanDevice);
    ReleaseResolutionDependentResources();
    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitializeAnf()
//-----------------------------------------------------------------------------
{
    if (m_RuntimeControls)
        m_RuntimeControls->ApplyStartupProperties();

    const VkExtent2D output_extent = GetAnfOutputExtent();

    const AnfSRQualityMode sr_mode               = static_cast<AnfSRQualityMode>(gAnfSrQualityMode);
    const bool              requestedDispatchImm  = gAnfDispatchImmediate;
    const uint32_t          anfFramesInFlight    = ClampAnfMaxFramesInFlight(gAnfMaxFramesInFlight);
    const bool              mifWasClamped         = static_cast<int>(anfFramesInFlight) != gAnfMaxFramesInFlight;
    const bool              fgOnSrMode            = m_RuntimeControls->ShouldUseFgOnSrMode();
    m_AnfFramesInFlight = anfFramesInFlight;
    auto releaseAnfAttemptState = [this]()
    {
        auto* const vulkan = GetVulkan();

        if (m_Anf)
        {
            m_Anf->Release();
            m_Anf.reset();
        }

        m_BlitQuadDrawableSr.reset();
        m_BlitQuadDrawableFg.reset();
        m_BlitQuadDrawableFgInput.reset();

        for (auto& tex : m_FgInputTextures)
        {
            if (tex) tex.Release(vulkan);
        }
        for (auto& tex : m_FgDepthInputTextures)
        {
            if (tex) tex.Release(vulkan);
        }
        for (auto& tex : m_FgInverseDepthInputTextures)
        {
            if (tex) tex.Release(vulkan);
        }
        for (auto& tex : m_FgMotionInputTextures)
        {
            if (tex) tex.Release(vulkan);
        }

        m_FgFramePending = false;
        ClearPendingFgRealFramePresent();
    };

    auto cacheAnfSettings = [this, fgOnSrMode](bool dispatchImmediate)
    {
        m_LastAnfSrQualityMode     = gAnfSrQualityMode;
        m_LastAnfDispatchImmediate = dispatchImmediate;
        m_LastAnfMaxFramesInFlight = gAnfMaxFramesInFlight;
        m_LastAnfFgOnSrMode        = fgOnSrMode;
    };

    const VkExtent2D scene_extent = GetAnfSceneExtent();
    m_LastAnfOutputExtent = output_extent;
    m_LastAnfSceneExtent  = scene_extent;

    auto tryInitializeForMode = [&](bool dispatchImmediate) -> bool
    {
        releaseAnfAttemptState();
        m_Anf = std::make_unique<Anf::AnfInterface>(*GetVulkan());
        m_AnfFgOnSrMode = fgOnSrMode;

        if (m_AnfDispatchTiming)
            m_AnfDispatchTiming->SetDispatchImmediate(dispatchImmediate);

        LOGI("Initializing ANF instance (dispatch: %s, maxInFlight: req=%d active=%u%s)",
            AnfDispatchModeName(dispatchImmediate),
            gAnfMaxFramesInFlight,
            m_AnfFramesInFlight,
            mifWasClamped ? " [clamped]" : "");
        const auto inst_result = m_Anf->InitializeInstance(m_AnfFramesInFlight, output_extent, dispatchImmediate);
        if (inst_result != Anf::StatusCode::SUCCESS)
        {
            LOGW("ANF instance creation failed in %s mode", AnfDispatchModeName(dispatchImmediate));
            releaseAnfAttemptState();
            return false;
        }

        // Step 2: Create the SR technique (independent of FG).
        LOGI("Initializing ANF SR technique");
        const auto sr_result = m_Anf->InitializeSR(sr_mode);
        if (sr_result == Anf::StatusCode::ERROR_NOT_IMPLEMENTED)
        {
            LOGI("ANF SR not supported by the SDK - SR will be unavailable");
        }
        else if (sr_result != Anf::StatusCode::SUCCESS)
        {
            LOGI("ANF SR initialization failed - SR will be unavailable");
        }

        const VkExtent2D fg_extent = fgOnSrMode
            ? output_extent
            : scene_extent;

        // Step 3: Create the FG technique (independent of SR).
        // FG inputs must all use one extent:
        //  - FG-only: scene resolution (existing behavior)
        //  - SR+FG:   output resolution (full FG-on-SR path)
        LOGI("Initializing ANF FG technique (%s)",
            fgOnSrMode ? "full FG-on-SR (output resolution)" : "scene resolution");
        const auto fg_result = m_Anf->InitializeFG(fg_extent);
        if (fg_result == Anf::StatusCode::ERROR_NOT_IMPLEMENTED)
        {
            LOGI("ANF FG not yet supported by the SDK - FG will be unavailable");
        }
        else if (fg_result != Anf::StatusCode::SUCCESS)
        {
            LOGI("ANF FG initialization failed - FG will be unavailable");
        }

        if (!m_Anf->IsSrValid() && !m_Anf->IsFgValid())
        {
            LOGW("Neither ANF SR nor FG is available in %s mode", AnfDispatchModeName(dispatchImmediate));
            releaseAnfAttemptState();
            return false;
        }

        const auto fg_formats = m_Anf->GetFgFormats();
        const TextureFormat fg_input_color_format = Anf::AnfInterface::ToTextureFormat(
            fg_formats.input_color_format).value_or(TextureFormat::R8G8B8A8_UNORM);
        const TextureFormat fg_input_motion_format = Anf::AnfInterface::ToTextureFormat(
            fg_formats.motion_format).value_or(TextureFormat::R16G16_SFLOAT);
        const TextureFormat fg_input_depth_format = Anf::AnfInterface::ToTextureFormat(
            fg_formats.depth_format).value_or(TextureFormat::D32_SFLOAT);
        const TextureFormat fg_input_inverse_depth_format = TextureFormat::R32_SFLOAT;

        LOGI("Allocating FG input intermediate textures (%ux%u color=%d motion=%d depth=%d inverseDepth=%d)",
             fg_extent.width,
             fg_extent.height,
             static_cast<int>(fg_input_color_format),
             static_cast<int>(fg_input_motion_format),
             static_cast<int>(fg_input_depth_format),
             static_cast<int>(fg_input_inverse_depth_format));
        for (uint32_t i = 0; i < NUM_VULKAN_BUFFERS; ++i)
        {
            CreateTexObjectInfo info{};
            info.uiWidth  = fg_extent.width;
            info.uiHeight = fg_extent.height;
            info.Format   = fg_input_color_format;
            info.TexType  = TT_COMPUTE_TARGET;
            info.pName    = "FG Input Intermediate";
            m_FgInputTextures[i] = std::move(CreateTextureObject<Vulkan>(*GetVulkan(), info));

            if (fgOnSrMode)
            {
                CreateTexObjectInfo motionInfo{};
                motionInfo.uiWidth  = fg_extent.width;
                motionInfo.uiHeight = fg_extent.height;
                motionInfo.Format   = fg_input_motion_format;
                motionInfo.TexType  = TT_COMPUTE_TARGET;
                motionInfo.pName    = "FG Motion Intermediate";
                m_FgMotionInputTextures[i] = std::move(CreateTextureObject<Vulkan>(*GetVulkan(), motionInfo));

                CreateTexObjectInfo depthInfo{};
                depthInfo.uiWidth   = fg_extent.width;
                depthInfo.uiHeight  = fg_extent.height;
                depthInfo.Format    = fg_input_depth_format;
                depthInfo.TexType   = TT_DEPTH_TARGET;
                depthInfo.pName     = "FG Depth Intermediate";
                m_FgDepthInputTextures[i] = std::move(CreateTextureObject<Vulkan>(*GetVulkan(), depthInfo));

                CreateTexObjectInfo inverseDepthInfo{};
                inverseDepthInfo.uiWidth  = fg_extent.width;
                inverseDepthInfo.uiHeight = fg_extent.height;
                inverseDepthInfo.Format   = fg_input_inverse_depth_format;
                inverseDepthInfo.TexType  = TT_COMPUTE_TARGET;
                inverseDepthInfo.pName    = "FG Inverse Depth Intermediate";
                m_FgInverseDepthInputTextures[i] = std::move(CreateTextureObject<Vulkan>(*GetVulkan(), inverseDepthInfo));
            }
        }

        if (m_Anf->IsFgValid())
        {
            const TextureFormat expectedFgColorFormat  = AnfTextureFormatOrUndefined(fg_formats.input_color_format);
            const TextureFormat expectedFgMotionFormat = AnfTextureFormatOrUndefined(fg_formats.motion_format);
            const TextureFormat expectedFgDepthFormat  = AnfTextureFormatOrUndefined(fg_formats.depth_format);
            for (uint32_t i = 0; i < NUM_VULKAN_BUFFERS; ++i)
            {
                if (!ValidateAnfTextureCompatibility("FG input color", m_FgInputTextures[i], expectedFgColorFormat, fg_formats.input_color_usage, VK_IMAGE_ASPECT_COLOR_BIT))
                {
                    releaseAnfAttemptState();
                    return false;
                }
                if (fgOnSrMode)
                {
                    if (!ValidateAnfTextureCompatibility("FG input motion", m_FgMotionInputTextures[i], expectedFgMotionFormat, fg_formats.motion_usage, VK_IMAGE_ASPECT_COLOR_BIT))
                    {
                        releaseAnfAttemptState();
                        return false;
                    }
                    if (!ValidateAnfTextureCompatibility("FG input depth", m_FgDepthInputTextures[i], expectedFgDepthFormat, fg_formats.depth_usage, VK_IMAGE_ASPECT_DEPTH_BIT))
                    {
                        releaseAnfAttemptState();
                        return false;
                    }
                    if (!ValidateAnfTextureCompatibility("FG input inverse depth", m_FgInverseDepthInputTextures[i], TextureFormat::R32_SFLOAT, fg_formats.depth_usage, VK_IMAGE_ASPECT_COLOR_BIT))
                    {
                        releaseAnfAttemptState();
                        return false;
                    }
                }
            }
        }

        return true;
    };

    bool actualDispatchImmediate = requestedDispatchImm;
    if (!tryInitializeForMode(requestedDispatchImm))
    {
        const bool fallbackDispatchImmediate = !requestedDispatchImm;
        LOGW("ANF init failed in %s mode; retrying once in %s mode",
             AnfDispatchModeName(requestedDispatchImm),
             AnfDispatchModeName(fallbackDispatchImmediate));

        if (!tryInitializeForMode(fallbackDispatchImmediate))
        {
            LOGW("ANF init failed in both %s and %s modes - continuing without ANF",
                 AnfDispatchModeName(requestedDispatchImm),
                 AnfDispatchModeName(fallbackDispatchImmediate));
            gAnfDispatchImmediate = requestedDispatchImm;
            cacheAnfSettings(gAnfDispatchImmediate);
            m_AnfReset   = true;
            m_AnfFgReset = true;
            return true;
        }

        actualDispatchImmediate = fallbackDispatchImmediate;
        LOGW("ANF dispatch mode reverted from %s to %s after recreation failure",
             AnfDispatchModeName(requestedDispatchImm),
             AnfDispatchModeName(actualDispatchImmediate));
    }

    gAnfDispatchImmediate = actualDispatchImmediate;
    cacheAnfSettings(actualDispatchImmediate);
    if (m_AnfDispatchTiming)
        m_AnfDispatchTiming->SetDispatchImmediate(actualDispatchImmediate);
    m_AnfReset   = true;
    m_AnfFgReset = true;
    m_FgFramePending = false;
    ClearPendingFgRealFramePresent();

    return true;
}

//-----------------------------------------------------------------------------
void Application::CheckAndReinitializeAnf()
//-----------------------------------------------------------------------------
{
    const bool desiredFgOnSrMode = m_RuntimeControls->ShouldUseFgOnSrMode();
    const VkExtent2D outputExtent = GetAnfOutputExtent();
    const VkExtent2D sceneExtent = GetAnfSceneExtent();
    const bool extentChanged =
        m_LastAnfOutputExtent.width  != outputExtent.width ||
        m_LastAnfOutputExtent.height != outputExtent.height ||
        m_LastAnfSceneExtent.width   != sceneExtent.width ||
        m_LastAnfSceneExtent.height  != sceneExtent.height;

    if (extentChanged)
    {
        LOGI("ANF extent changed from output=%ux%u scene=%ux%u to output=%ux%u scene=%ux%u; rebuilding dependent resources",
             m_LastAnfOutputExtent.width,
             m_LastAnfOutputExtent.height,
             m_LastAnfSceneExtent.width,
             m_LastAnfSceneExtent.height,
             outputExtent.width,
             outputExtent.height,
             sceneExtent.width,
             sceneExtent.height);
        if (!RebuildResolutionDependentResources() || !InitializeResolutionDependentResources(m_WindowHandle))
            LOGE("Unable to rebuild ANF resources after a resolution change");
        return;
    }

    // Recreating techniques mid-session (e.g. quality mode or dispatch mode change) may
    // not handle an in-flight technique gracefully. Only trigger a full restart for
    // settings that strictly require it; fgOnSrMode changes just need an FG history reset.
    const bool settingsRequiringRestart =
        m_LastAnfDispatchImmediate != gAnfDispatchImmediate ||
        m_LastAnfSrQualityMode     != gAnfSrQualityMode ||
        m_LastAnfMaxFramesInFlight != gAnfMaxFramesInFlight;

    // Track fgOnSrMode change for resetting FG history without full restart.
    const bool fgOnSrModeChanged = (m_LastAnfFgOnSrMode != desiredFgOnSrMode);
    if (fgOnSrModeChanged)
    {
        m_AnfFgReset = true;
        m_LastAnfFgOnSrMode = desiredFgOnSrMode;
    }

    // Nothing to do if the settings that require a full restart haven't changed.
    if (!settingsRequiringRestart)
        return;

    auto* const pVulkan = GetVulkan();

    // Wait for all in-flight GPU work before tearing down ANF resources.
    vkDeviceWaitIdle(pVulkan->m_VulkanDevice);

    if (m_Anf)
    {
        m_Anf->Release();
        m_Anf.reset();
    }

    // The blit drawables hold descriptor sets that reference the old ANF output textures,
    // so they must be recreated together with ANF.
    m_BlitQuadDrawableSr.reset();
    m_BlitQuadDrawableFg.reset();
    m_BlitQuadDrawableFgInput.reset();
    // Release the old intermediate textures (new ones are allocated in InitializeAnf).
    for (auto& tex : m_FgInputTextures)
    {
        if (tex) tex.Release(pVulkan);
    }
    for (auto& tex : m_FgDepthInputTextures)
    {
        if (tex) tex.Release(pVulkan);
    }
    for (auto& tex : m_FgInverseDepthInputTextures)
    {
        if (tex) tex.Release(pVulkan);
    }
    for (auto& tex : m_FgMotionInputTextures)
    {
        if (tex) tex.Release(pVulkan);
    }
    m_FgFramePending = false;

    InitializeAnf();

    // Recreate the SR and FG blit drawables so they point to the new output textures.
    const auto* pBlitQuadShader = m_ShaderManager->GetShader("Blit");
    if (!pBlitQuadShader || !m_Anf)
        return;

    // SR blit drawable
    Mesh blitQuadMeshSr;
    if (MeshHelper::CreateMesh<Vulkan>(pVulkan->GetMemoryManager(),
            MeshObjectIntermediate::CreateScreenSpaceMesh(), 0,
            pBlitQuadShader->m_shaderDescription->m_vertexFormats, &blitQuadMeshSr))
    {
        auto blitMaterialSr = m_MaterialManager->CreateMaterial(*pBlitQuadShader, NUM_VULKAN_BUFFERS,
            [this](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
            {
                if (texName == "Diffuse")
                {
                    if (m_Anf && m_Anf->IsSrValid())
                    {
                        static std::array<TextureVulkan*, NUM_VULKAN_BUFFERS> s_sr = {};
                        for (uint32_t i = 0; i < NUM_VULKAN_BUFFERS; ++i)
                        {
                            const auto& [tex, _] = m_Anf->GetSrOutputTexture(AnfFrameIndex(i, m_AnfFramesInFlight));
                            s_sr[i] = tex;
                        }
                        return { s_sr.begin(), s_sr.end() };
                    }
                    return { &m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[0] };
                }
                if (texName == "Overlay") return { &m_RenderPassData[RP_HUD].RenderTarget.m_ColorAttachments[0] };
                return {};
            },
            [](const std::string&) -> PerFrameBuffer { return {}; });

        m_BlitQuadDrawableSr = std::make_unique<Drawable>(*pVulkan, std::move(blitMaterialSr));
        m_BlitQuadDrawableSr->Init(m_RenderPassData[RP_BLIT].RenderContext[0], std::move(blitQuadMeshSr));
    }

    // FG blit drawable (only if FG is actually supported by the SDK)
    if (m_Anf->IsFgValid())
    {
        Mesh blitQuadMeshFg;
        if (MeshHelper::CreateMesh<Vulkan>(pVulkan->GetMemoryManager(),
                MeshObjectIntermediate::CreateScreenSpaceMesh(), 0,
                pBlitQuadShader->m_shaderDescription->m_vertexFormats, &blitQuadMeshFg))
        {
            auto blitMaterialFg = m_MaterialManager->CreateMaterial(*pBlitQuadShader, NUM_VULKAN_BUFFERS,
                [this](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
                {
                    if (texName == "Diffuse")
                    {
                        if (m_Anf && m_Anf->IsFgValid())
                        {
                            static std::array<TextureVulkan*, NUM_VULKAN_BUFFERS> s_fg = {};
                            for (uint32_t i = 0; i < NUM_VULKAN_BUFFERS; ++i)
                            {
                                const auto& [tex, _] = m_Anf->GetFgOutputTexture(AnfFrameIndex(i, m_AnfFramesInFlight));
                                s_fg[i] = tex;
                            }
                            return { s_fg.begin(), s_fg.end() };
                        }
                        return { &m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[0] };
                    }
                    if (texName == "Overlay") return { &m_RenderPassData[RP_HUD].RenderTarget.m_ColorAttachments[0] };
                    return {};
                },
                [](const std::string&) -> PerFrameBuffer { return {}; });

            m_BlitQuadDrawableFg = std::make_unique<Drawable>(*pVulkan, std::move(blitMaterialFg));
            m_BlitQuadDrawableFg->Init(m_RenderPassData[RP_BLIT].RenderContext[0], std::move(blitQuadMeshFg));
        }
    }
}

//-----------------------------------------------------------------------------
VkSemaphore Application::RenderFgFrame(uint32_t whichBuffer, std::span<const VkSemaphore> waitSems)
//-----------------------------------------------------------------------------
{
    auto* const pVulkan = GetVulkan();
    const uint32_t qfi   = static_cast<uint32_t>(pVulkan->m_VulkanQueues[Vulkan::eGraphicsQueue].QueueFamilyIndex);
    const VkQueue  queue = pVulkan->m_VulkanQueues[Vulkan::eGraphicsQueue].Queue;
    const uint32_t anfBuffer = AnfFrameIndex(whichBuffer, m_AnfFramesInFlight);
    TextureVulkan* sceneAnfDepthImage = gAnfUseInverseDepth
        ? &m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[2]
        : &m_RenderPassData[RP_SCENE].RenderTarget.m_DepthAttachment;
    VkImageLayout sceneAnfDepthLayout = gAnfUseInverseDepth
        ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    TextureVulkan* fgPreparedDepthImage = gAnfUseInverseDepth
        ? &m_FgInverseDepthInputTextures[m_FgFrameBuffer]
        : &m_FgDepthInputTextures[m_FgFrameBuffer];
    const VkImageLayout fgPreparedDepthLayout = gAnfUseInverseDepth
        ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    const bool useFgOnSrInputs =
        m_AnfFgOnSrMode &&
        static_cast<bool>(*fgPreparedDepthImage) &&
        static_cast<bool>(m_FgMotionInputTextures[m_FgFrameBuffer]);

    Anf::FrameGenInitData fg_props{};
    fg_props.ColorImage         = &m_FgInputTextures[m_FgFrameBuffer];
    fg_props.ColorLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    fg_props.DepthImage         = useFgOnSrInputs
        ? fgPreparedDepthImage
        : sceneAnfDepthImage;
    fg_props.DepthLayout        = useFgOnSrInputs
        ? fgPreparedDepthLayout
        : sceneAnfDepthLayout;
    fg_props.MotionVectorImage  = useFgOnSrInputs
        ? &m_FgMotionInputTextures[m_FgFrameBuffer]
        : &m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[1];
    fg_props.MotionVectorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    const std::array<VkSemaphore, 1> fgSignal = { m_AnfFgSemaphores[whichBuffer] };
    const std::span<const VkSemaphore> fallbackAnfWaits = m_Anf->IsDispatchImmediate()
        ? waitSems
        : std::span<const VkSemaphore>{};
    const std::span<const VkSemaphore> fallbackAnfSignals = m_Anf->IsDispatchImmediate()
        ? std::span<const VkSemaphore>{ fgSignal.data(), fgSignal.size() }
        : std::span<const VkSemaphore>{};

    AnfDispatchTimingController::DispatchToken dispatchToken{};
    if (m_AnfDispatchTiming)
    {
        dispatchToken = m_AnfDispatchTiming->BeginDispatch(
            AnfDispatchTimingController::Technique::Fg,
            whichBuffer,
            waitSems,
            m_GpuProfiler.get(),
            AnfGpuProfiler::Region::FgDispatch);
    }

    const auto dispatchWaitSems = m_AnfDispatchTiming
        ? m_AnfDispatchTiming->GetDispatchWaitSemaphores(dispatchToken, fallbackAnfWaits)
        : fallbackAnfWaits;
    const auto dispatchSignalSems = m_AnfDispatchTiming
        ? m_AnfDispatchTiming->GetDispatchSignalSemaphores(dispatchToken, fallbackAnfSignals)
        : fallbackAnfSignals;
    const VkCommandBuffer dispatchCmd = m_AnfDispatchTiming
        ? m_AnfDispatchTiming->GetDispatchCommandBuffer(dispatchToken)
        : VK_NULL_HANDLE;

    const auto result = m_Anf->RenderFG(anfBuffer, dispatchCmd, fg_props,
        dispatchWaitSems, dispatchSignalSems, qfi, queue, m_AnfFgReset);
    if (result != Anf::StatusCode::SUCCESS)
    {
        if (m_AnfDispatchTiming)
            m_AnfDispatchTiming->CancelDispatch(AnfDispatchTimingController::Technique::Fg, whichBuffer, dispatchToken);
        return VK_NULL_HANDLE;
    }

    std::vector<VkPipelineStageFlags> submitWaitStages(waitSems.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    if (!submitWaitStages.empty())
        submitWaitStages[0] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    if (m_AnfDispatchTiming)
    {
        m_AnfDispatchTiming->EndDispatch(
            AnfDispatchTimingController::Technique::Fg,
            whichBuffer,
            waitSems,
            submitWaitStages,
            m_AnfFgSemaphores[whichBuffer],
            dispatchToken);
    }
    m_AnfFgReset = false;

    return m_AnfFgSemaphores[whichBuffer];
}

//-----------------------------------------------------------------------------
void Application::ClearPendingFgRealFramePresent()
//-----------------------------------------------------------------------------
{
    m_FgRealFramePresentPending = false;
    m_FgRealFramePresentSemaphore = VK_NULL_HANDLE;
    m_FgRealFramePresentIdx = 0;
}

//-----------------------------------------------------------------------------
void Application::Destroy()
//-----------------------------------------------------------------------------
{
    auto* const pVulkan = GetVulkan();

    // ANF techniques retain references to their scene, intermediate, and output
    // images. Release the techniques before any of those images are destroyed,
    // especially when Android tears down and recreates the native window.
    vkDeviceWaitIdle(pVulkan->m_VulkanDevice);
    LOGI("Releasing ANF techniques before resolution-dependent images");
    m_BlitQuadDrawableSr.reset();
    m_BlitQuadDrawableFg.reset();
    m_BlitQuadDrawableFgInput.reset();
    if (m_Anf)
    {
        // During final Android teardown, leave SDK resources for process cleanup.
        LOGI("Deferring ANF SDK destruction to final process cleanup");
        m_Anf->AbandonSdkForAndroidTeardown();
        m_Anf.release();
    }
    for (auto& tex : m_FgInputTextures)
    {
        if (tex) tex.Release(pVulkan);
    }
    for (auto& tex : m_FgDepthInputTextures)
    {
        if (tex) tex.Release(pVulkan);
    }
    for (auto& tex : m_FgInverseDepthInputTextures)
    {
        if (tex) tex.Release(pVulkan);
    }
    for (auto& tex : m_FgMotionInputTextures)
    {
        if (tex) tex.Release(pVulkan);
    }

    ReleaseUniformBuffer(pVulkan, m_ObjectVertUniform);
    ReleaseUniformBuffer(pVulkan, m_LightUniform);

    for (auto& [hash, objectUniform] : m_ObjectFragUniforms)
        ReleaseUniformBuffer(pVulkan, &objectUniform.objectFragUniform);

    for (auto& [id, prop] : m_AnimationProperties)
        ReleaseUniformBuffer(pVulkan, prop.objectVertUniform);

    for (int whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        for (auto& cmdBuffer : m_RenderPassData[whichPass].ObjectsCmdBuffer)
            cmdBuffer.Release();
        m_RenderPassData[whichPass].RenderTarget.Release();
    }

    for (int whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
        for (auto& commandList : m_RenderPassData[whichPass].PassCommandList)
            commandList.Release();

    for (int whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
        DestroySemaphoreArray(pVulkan->m_VulkanDevice, m_RenderPassData[whichPass].PassCompleteSemaphores);

    DestroySemaphoreArray(pVulkan->m_VulkanDevice, m_AnfSemaphores);
    DestroySemaphoreArray(pVulkan->m_VulkanDevice, m_AnfFgSemaphores);
    if (m_AnfDispatchTiming)
        m_AnfDispatchTiming->DestroySemaphores(pVulkan->m_VulkanDevice);

    DestroySemaphoreArray(pVulkan->m_VulkanDevice, m_FgReadySemaphores);
    DestroySemaphoreArray(pVulkan->m_VulkanDevice, m_FgInputReadySemaphores);

    for (auto& cmdList : m_FgInputBlitCmdList)
        cmdList.Release();
    if (m_AnfDispatchTiming)
        m_AnfDispatchTiming->Release();

    for (auto& tex : m_FgInputTextures)
    {
        if (tex) tex.Release(pVulkan);
    }
    for (auto& tex : m_FgDepthInputTextures)
    {
        if (tex) tex.Release(pVulkan);
    }
    for (auto& tex : m_FgInverseDepthInputTextures)
    {
        if (tex) tex.Release(pVulkan);
    }
    for (auto& tex : m_FgMotionInputTextures)
    {
        if (tex) tex.Release(pVulkan);
    }

    m_SceneDrawables.clear();
    m_AnimatedDrawables.clear();
    m_BlitQuadDrawable.reset();
    m_BlitQuadDrawableSr.reset();
    m_BlitQuadDrawableFg.reset();
    m_BlitQuadDrawableFgInput.reset();

    m_ShaderManager.reset();
    m_MaterialManager.reset();
    m_CameraController.reset();
    m_AssetManager.reset();

    if (m_Anf)
    {
        m_Anf->Release();
        m_Anf.reset();
    }

    if (m_GpuProfiler)
    {
        m_GpuProfiler->Destroy();
        m_GpuProfiler.reset();
    }

    m_AnfDispatchTiming.reset();

    ApplicationHelperBase::Destroy();
}

//-----------------------------------------------------------------------------
bool Application::InitializeLights()
//-----------------------------------------------------------------------------
{
    m_LightUniformData.AmbientColor = glm::vec4{1.0f,1.0f,1.0f,1.0f};

    m_LightUniformData.SpotLights_pos[0] = glm::vec4(-6.900000f, 32.299999f, -1.900000f, 1.0f);
    m_LightUniformData.SpotLights_pos[1] = glm::vec4(3.300000f, 26.900000f, 7.600000f, 1.0f);
    m_LightUniformData.SpotLights_pos[2] = glm::vec4(12.100000f, 41.400002f, -2.800000f, 1.0f);
    m_LightUniformData.SpotLights_pos[3] = glm::vec4(-5.400000f, 18.500000f, 28.500000f, 1.0f);

    m_LightUniformData.SpotLights_dir[0] = glm::vec4(-0.534696f, -0.834525f, 0.132924f, 0.0f);
    m_LightUniformData.SpotLights_dir[1] = glm::vec4(0.000692f, -0.197335f, 0.980336f, 0.0f);
    m_LightUniformData.SpotLights_dir[2] = glm::vec4(0.985090f, -0.172016f, 0.003000f, 0.0f);
    m_LightUniformData.SpotLights_dir[3] = glm::vec4(0.674125f, -0.295055f, -0.677125f, 0.0f);

    m_LightUniformData.SpotLights_color[0] = glm::vec4(1.0f, 1.0f, 1.0f, 3.0f);
    m_LightUniformData.SpotLights_color[1] = glm::vec4(1.0f, 1.0f, 1.0f, 3.5f);
    m_LightUniformData.SpotLights_color[2] = glm::vec4(1.0f, 1.0f, 1.0f, 2.0f);
    m_LightUniformData.SpotLights_color[3] = glm::vec4(1.0f, 1.0f, 1.0f, 2.8f);

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
    m_Camera.SetFov(gFov * TO_RADIANS);
    m_Camera.SetClipPlanes(gNearPlane, gFarPlane);

#if defined(OS_ANDROID)
    typedef CameraControllerTouch tCameraController;
#else
    typedef CameraController      tCameraController;
#endif

    auto cameraController = std::make_unique<tCameraController>();
    if (!cameraController->Initialize(gSurfaceWidth, gSurfaceHeight))
        return false;

    m_CameraController = std::move(cameraController);
    m_CameraController->SetMoveSpeed(10.f);

    m_Camera.UpdateMatrices();

    m_CurrViewProjMatrix         = m_Camera.ProjectionMatrix() * m_Camera.ViewMatrix();
    m_CurrViewProjMatrixNoJitter = m_Camera.ProjectionMatrixNoJitter() * m_Camera.ViewMatrix();
    m_PrevViewProjMatrix         = m_CurrViewProjMatrix;
    m_PrevViewProjMatrixNoJitter = m_CurrViewProjMatrixNoJitter;
    m_CurrJitter = m_Camera.Jitter();
    m_PrevJitter = m_CurrJitter;

    return true;
}

//-----------------------------------------------------------------------------
bool Application::LoadShaders()
//-----------------------------------------------------------------------------
{
    m_ShaderManager  = std::make_unique<ShaderManager>(*GetVulkan());
    m_ShaderManager->RegisterRenderPassNames(sRenderPassNames);
    m_MaterialManager = std::make_unique<MaterialManager>(*GetVulkan());

    LOGI("******************************");
    LOGI("Loading Shaders...");
    LOGI("******************************");

    typedef std::pair<std::string, std::string> tIdAndFilename;
    for (const tIdAndFilename& i :
            {
                tIdAndFilename { "Blit",             "Blit.json" },
                tIdAndFilename { "SceneOpaque",      "SceneOpaque.json" },
                tIdAndFilename { "SceneTransparent", "SceneTransparent.json" },
                tIdAndFilename { "Animated",         "AnimatedOpaque.json" },
            })
    {
        if (!m_ShaderManager->AddShader(*m_AssetManager, i.first, i.second, SHADER_DESTINATION_PATH))
        {
            LOGE("Error Loading shader %s from %s", i.first.c_str(), i.second.c_str());
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
    const Anf::AnfSceneFormats scene_formats = m_Anf ? m_Anf->GetSceneFormats() : Anf::AnfSceneFormats{};

    const auto pick_format = [](AnfFormat anf_format, TextureFormat fallback)
    {
        return Anf::AnfInterface::ToTextureFormat(anf_format).value_or(fallback);
    };

    LOGI("**************************");
    LOGI("Creating Render Targets...");
    LOGI("**************************");

    const TextureFormat desiredSceneColorFormat  = pick_format(scene_formats.color_format, TextureFormat::B10G11R11_UFLOAT_PACK32);
    const TextureFormat desiredSceneMotionFormat = pick_format(scene_formats.motion_format, TextureFormat::R16G16_SFLOAT);
    const TextureFormat desiredDepthFormat       = pick_format(scene_formats.depth_format, TextureFormat::D32_SFLOAT);

    const TextureFormat SceneColorTypes[] =
    {
        desiredSceneColorFormat,
        desiredSceneMotionFormat,
        TextureFormat::R32_SFLOAT
    };

    const TEXTURE_TYPE SceneTextureTypes[] =
    {
        TEXTURE_TYPE::TT_RENDER_TARGET_WITH_STORAGE_TRANSFERSRC,
        TEXTURE_TYPE::TT_RENDER_TARGET_WITH_STORAGE_TRANSFERSRC,
        TEXTURE_TYPE::TT_RENDER_TARGET_WITH_STORAGE_TRANSFERSRC
    };
    const TextureFormat HudColorType[] = { TextureFormat::R8G8B8A8_SRGB };

    RenderTargetInitializeInfo sceneRenderTargetInfo{};
    sceneRenderTargetInfo.Width        = gRenderWidth / 2.0f;
    sceneRenderTargetInfo.Height       = gRenderHeight / 2.0f;
    sceneRenderTargetInfo.LayerFormats = SceneColorTypes;
    sceneRenderTargetInfo.DepthFormat  = desiredDepthFormat;
    sceneRenderTargetInfo.TextureTypes = SceneTextureTypes;
    sceneRenderTargetInfo.DepthTextureType = TEXTURE_TYPE::TT_DEPTH_TARGET;

    LOGI("Scene RT formats: color=%d motion=%d depth=%d",
         static_cast<int>(desiredSceneColorFormat),
         static_cast<int>(desiredSceneMotionFormat),
         static_cast<int>(desiredDepthFormat));

    if (!m_RenderPassData[RP_SCENE].RenderTarget.Initialize(pVulkan, sceneRenderTargetInfo, "Scene RT"))
    {
        LOGE("Unable to create scene render target");
        return false;
    }

    // Sampled descriptors must match the scene pass's StoreReadOnly final layout.
    // Transfer-capable textures otherwise default their descriptor layout to TRANSFER_SRC.
    for (auto& attachment : m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments)
        attachment.ImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    if (m_Anf)
    {
        auto& sceneTarget = m_RenderPassData[RP_SCENE].RenderTarget;
        if (!ValidateAnfTextureCompatibility("Scene color", sceneTarget.m_ColorAttachments[0], desiredSceneColorFormat, scene_formats.color_usage, VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!ValidateAnfTextureCompatibility("Scene motion", sceneTarget.m_ColorAttachments[1], desiredSceneMotionFormat, scene_formats.motion_usage, VK_IMAGE_ASPECT_COLOR_BIT))
            return false;
        if (!ValidateAnfTextureCompatibility("Scene depth", sceneTarget.m_DepthAttachment, desiredDepthFormat, scene_formats.depth_usage, VK_IMAGE_ASPECT_DEPTH_BIT))
            return false;
        if (!ValidateAnfTextureCompatibility("Scene inverse depth", sceneTarget.m_ColorAttachments[2], TextureFormat::R32_SFLOAT, scene_formats.depth_usage, VK_IMAGE_ASPECT_COLOR_BIT))
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
    LOGI("******************************");
    LOGI("Initializing Uniforms...");
    LOGI("******************************");

    auto* const pVulkan = GetVulkan();

    if (!CreateUniformBuffer(pVulkan, m_ObjectVertUniform)) return false;
    if (!CreateUniformBuffer(pVulkan, m_LightUniform))      return false;

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitSemaphores()
//-----------------------------------------------------------------------------
{
    LOGI("********************************");
    LOGI("Initializing Local Semaphores...");
    LOGI("********************************");

    auto* const pVulkan = GetVulkan();

    for (int whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        if (!CreateSemaphoreArray(pVulkan, m_RenderPassData[whichPass].PassCompleteSemaphores))
            return false;
    }

    if (!CreateSemaphoreArray(pVulkan, m_AnfSemaphores))   return false;
    if (!CreateSemaphoreArray(pVulkan, m_AnfFgSemaphores)) return false;

    if (m_AnfDispatchTiming &&
        !m_AnfDispatchTiming->CreateSemaphores(pVulkan->m_VulkanDevice))
    {
        LOGE("Unable to create ANF dispatch timing semaphores");
        return false;
    }

    if (!CreateSemaphoreArray(pVulkan, m_FgReadySemaphores))      return false;
    if (!CreateSemaphoreArray(pVulkan, m_FgInputReadySemaphores)) return false;

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitAllRenderPasses()
//-----------------------------------------------------------------------------
{
    auto& vulkan = *GetVulkan();

    m_RenderPassData[RP_SCENE].RenderPassSetup = { RenderPassInputUsage::Clear,    true,  RenderPassOutputUsage::StoreReadOnly, RenderPassOutputUsage::StoreReadOnly, {}};
    m_RenderPassData[RP_HUD].RenderPassSetup   = { RenderPassInputUsage::Clear,    false, RenderPassOutputUsage::StoreReadOnly, RenderPassOutputUsage::Discard,       {}};
    m_RenderPassData[RP_BLIT].RenderPassSetup  = { RenderPassInputUsage::DontCare, true,  RenderPassOutputUsage::Present,       RenderPassOutputUsage::Discard,       {}};

    TextureFormat surfaceFormat        = vulkan.m_SurfaceFormat;
    auto          swapChainColorFormat = std::span<const TextureFormat>({ &surfaceFormat, 1 });
    auto          swapChainDepthFormat = vulkan.m_SwapchainDepth.format;

    LOGI("******************************");
    LOGI("Initializing Render Passes... ");
    LOGI("******************************");

    for (uint32_t whichPass = 0; whichPass < RP_BLIT; whichPass++)
    {
        std::span<const TextureFormat> colorFormats = m_RenderPassData[whichPass].RenderTarget.m_pLayerFormats;
        TextureFormat                  depthFormat  = m_RenderPassData[whichPass].RenderTarget.m_DepthFormat;

        const auto& passSetup = m_RenderPassData[whichPass].RenderPassSetup;
        auto&       passData  = m_RenderPassData[whichPass];

        RenderPass renderPass;
        if (!vulkan.CreateRenderPass(
            { colorFormats }, depthFormat, Msaa::Samples1,
            passSetup.ColorInputUsage, passSetup.ColorOutputUsage,
            passSetup.ClearDepthRenderPass, passSetup.DepthOutputUsage,
            renderPass))
        {
            return false;
        }
        Framebuffer<Vulkan> framebuffer;
        framebuffer.Initialize(vulkan, renderPass, passData.RenderTarget.m_ColorAttachments, &passData.RenderTarget.m_DepthAttachment, sRenderPassNames[whichPass]);
        passData.RenderContext.push_back({std::move(renderPass), {}, std::move(framebuffer), sRenderPassNames[whichPass]});
    }
    for (auto whichBuffer = 0; whichBuffer < vulkan.GetSwapchainBufferCount(); ++whichBuffer)
    {
        m_RenderPassData[RP_BLIT].RenderContext.push_back({vulkan.m_SwapchainRenderPass.Copy(), {}, vulkan.GetSwapchainFramebuffer(whichBuffer), "RP_BLIT"});
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

    const auto* pSceneOpaqueShader      = m_ShaderManager->GetShader("SceneOpaque");
    const auto* pSceneTransparentShader = m_ShaderManager->GetShader("SceneTransparent");
    const auto* pBlitQuadShader         = m_ShaderManager->GetShader("Blit");
    if (!pSceneOpaqueShader || !pSceneTransparentShader || !pBlitQuadShader)
        return false;

    {
        float mipBias = std::max(std::log2(float(gRenderWidth) / float(gSurfaceWidth)), std::log2(float(gRenderHeight) / float(gSurfaceHeight)));
        ReleaseSampler(vulkan, &m_SamplerRepeat);
        m_SamplerRepeat = CreateSampler(vulkan, SamplerAddressMode::Repeat, SamplerFilter::Linear, SamplerBorderColor::TransparentBlackFloat, mipBias);
        if (m_SamplerRepeat.IsEmpty()) return false;
    }

    LOGI("***********************************");
    LOGI("Loading and preparing the museum...");
    LOGI("***********************************");

    m_TextureManager->SetDefaultFilenameManipulators(PathManipulator_PrefixDirectory(TEXTURE_DESTINATION_PATH));

    const PathManipulator_PrefixDirectory prefixTextureDir{ TEXTURE_DESTINATION_PATH };
    auto* whiteTexture         = m_TextureManager->GetOrLoadTexture("white_d.ktx",         m_SamplerRepeat, prefixTextureDir);
    auto* blackTexture         = m_TextureManager->GetOrLoadTexture("black_d.ktx",         m_SamplerRepeat, prefixTextureDir);
    auto* normalDefaultTexture = m_TextureManager->GetOrLoadTexture("normal_default.ktx",  m_SamplerRepeat, prefixTextureDir);

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
            if (!CreateUniformBuffer(&vulkan, iter.first->second.objectFragUniform))
                LOGE("Failed to create object uniform buffer");
        }
        return iter.first->second;
    };

    m_MaterialLoader = [this, pSceneOpaqueShader, pSceneTransparentShader, whiteTexture, normalDefaultTexture, blackTexture, UniformBufferLoader](const MeshObjectIntermediate::MaterialDef& materialDef)->std::optional<Material>
    {
        const PathManipulator_PrefixDirectory prefixTextureDir{ TEXTURE_DESTINATION_PATH };
        const PathManipulator_ChangeExtension changeTextureExt{".ktx"};

        auto* diffuseTexture           = m_TextureManager->GetOrLoadTexture(materialDef.diffuseFilename,  m_SamplerRepeat, prefixTextureDir, changeTextureExt);
        auto* normalTexture            = m_TextureManager->GetOrLoadTexture(materialDef.bumpFilename,     m_SamplerRepeat, prefixTextureDir, changeTextureExt);
        auto* emissiveTexture          = m_TextureManager->GetOrLoadTexture(materialDef.emissiveFilename, m_SamplerRepeat, prefixTextureDir, changeTextureExt);
        auto* metallicRoughnessTexture = m_TextureManager->GetOrLoadTexture(materialDef.specMapFilename,  m_SamplerRepeat, prefixTextureDir, changeTextureExt);
        bool  transparent              = materialDef.transparent;

        const auto* targetShader = transparent ? pSceneTransparentShader : pSceneOpaqueShader;

        ObjectMaterialParameters objectMaterial;
        objectMaterial.objectFragUniformData.Color.r = static_cast<float>(materialDef.baseColorFactor[0]);
        objectMaterial.objectFragUniformData.Color.g = static_cast<float>(materialDef.baseColorFactor[1]);
        objectMaterial.objectFragUniformData.Color.b = static_cast<float>(materialDef.baseColorFactor[2]);
        objectMaterial.objectFragUniformData.Color.a = static_cast<float>(materialDef.baseColorFactor[3]);
        objectMaterial.objectFragUniformData.ORM.b   = static_cast<float>(materialDef.metallicFactor);
        objectMaterial.objectFragUniformData.ORM.g   = static_cast<float>(materialDef.roughnessFactor);

        bool isAnimated = materialDef.materialName.substr(0, 4).compare("anim") == 0;

        if (!diffuseTexture || !normalTexture) return std::nullopt;

        auto shaderMaterial = m_MaterialManager->CreateMaterial(*targetShader, NUM_VULKAN_BUFFERS,
            [&](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
            {
                if (texName == "Diffuse")          return { diffuseTexture ? diffuseTexture : whiteTexture };
                if (texName == "Normal")           return { normalTexture  ? normalTexture  : normalDefaultTexture };
                if (texName == "Emissive")         return { emissiveTexture ? emissiveTexture : blackTexture };
                if (texName == "MetallicRoughness") return { metallicRoughnessTexture ? metallicRoughnessTexture : blackTexture };
                return {};
            },
            [&](const std::string& bufferName) -> PerFrameBufferVulkan
            {
                if (bufferName == "Vert")
                {
                    if (isAnimated)
                        return { m_AnimationProperties[materialDef.materialName].objectVertUniform.bufferHandles.begin(),
                        m_AnimationProperties[materialDef.materialName].objectVertUniform.bufferHandles.end() };
                    else
                        return { m_ObjectVertUniform.bufferHandles.begin(), m_ObjectVertUniform.bufferHandles.end() };
                }
                else if (bufferName == "Frag")  return { UniformBufferLoader(objectMaterial).objectFragUniform.buf.GetVkBuffer() };
                else if (bufferName == "Light") return { m_LightUniform.bufferHandles.begin(), m_LightUniform.bufferHandles.end() };
                return {};
            });

        return shaderMaterial;
    };

    const uint32_t loaderFlags      = 0;
    const bool     ignoreTransforms = (loaderFlags & DrawableLoader::LoaderFlags::IgnoreHierarchy) != 0;

    const auto sceneAssetPath = std::filesystem::path(MESH_DESTINATION_PATH).append(gSceneAssetModel).string();
    MeshLoaderModelSceneSanityCheck meshSanityCheckProcessor(sceneAssetPath);
    MeshObjectIntermediateGltfProcessor meshObjectProcessor(sceneAssetPath, ignoreTransforms, glm::vec3(1.0f,1.0f,1.0f));
    CameraGltfProcessor meshCameraProcessor{};

    if (gLoadSauna && !MeshLoader::LoadGltf(*m_AssetManager, sceneAssetPath, meshSanityCheckProcessor, meshObjectProcessor, meshCameraProcessor) ||
        !DrawableLoader::CreateDrawables(vulkan, std::move(meshObjectProcessor.m_meshObjects),
            m_RenderPassData[RP_SCENE].RenderContext, m_MaterialLoader, m_SceneDrawables, loaderFlags))
    {
        LOGE("Error Loading the scene gltf file");
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
    if (!MeshHelper::CreateMesh<Vulkan>(vulkan.GetMemoryManager(), MeshObjectIntermediate::CreateScreenSpaceMesh(), 0, pBlitQuadShader->m_shaderDescription->m_vertexFormats, &blitQuadMesh))
        return false;

    // Blit: scene RT -> swapchain (used when ANF is disabled)
    auto blitMaterial = m_MaterialManager->CreateMaterial(*pBlitQuadShader, NUM_VULKAN_BUFFERS,
        [this](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
        {
            if (texName == "Diffuse") return { &m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[0] };
            if (texName == "Overlay") return { &m_RenderPassData[RP_HUD].RenderTarget.m_ColorAttachments[0] };
            return {};
        },
        [](const std::string&) -> PerFrameBuffer { return {}; });

    m_BlitQuadDrawable = std::make_unique<Drawable>(vulkan, std::move(blitMaterial));
    if (!m_BlitQuadDrawable->Init(m_RenderPassData[RP_BLIT].RenderContext[0], std::move(blitQuadMesh)))
        return false;

    // Blit: SR output -> swapchain
    Mesh blitQuadMeshSr;
    if (!MeshHelper::CreateMesh<Vulkan>(vulkan.GetMemoryManager(), MeshObjectIntermediate::CreateScreenSpaceMesh(), 0, pBlitQuadShader->m_shaderDescription->m_vertexFormats, &blitQuadMeshSr))
        return false;

    auto blitMaterialSr = m_MaterialManager->CreateMaterial(*pBlitQuadShader, NUM_VULKAN_BUFFERS,
        [this](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
        {
            if (texName == "Diffuse")
            {
                if (m_Anf && m_Anf->IsSrValid())
                {
                    static std::array<TextureVulkan*, NUM_VULKAN_BUFFERS> s_sr_outputs = {};
                    for (uint32_t i = 0; i < NUM_VULKAN_BUFFERS; ++i)
                    {
                        const auto& [tex, _] = m_Anf->GetSrOutputTexture(AnfFrameIndex(i, m_AnfFramesInFlight));
                        s_sr_outputs[i] = tex;
                    }
                    return { s_sr_outputs.begin(), s_sr_outputs.end() };
                }
                return { &m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[0] };
            }
            if (texName == "Overlay") return { &m_RenderPassData[RP_HUD].RenderTarget.m_ColorAttachments[0] };
            return {};
        },
        [](const std::string&) -> PerFrameBuffer { return {}; });

    m_BlitQuadDrawableSr = std::make_unique<Drawable>(vulkan, std::move(blitMaterialSr));
    if (!m_BlitQuadDrawableSr->Init(m_RenderPassData[RP_BLIT].RenderContext[0], std::move(blitQuadMeshSr)))
        return false;

    // Blit: FG output -> swapchain (only created when FG is actually supported by the SDK)
    if (m_Anf && m_Anf->IsFgValid())
    {
        Mesh blitQuadMeshFg;
        if (!MeshHelper::CreateMesh<Vulkan>(vulkan.GetMemoryManager(), MeshObjectIntermediate::CreateScreenSpaceMesh(), 0, pBlitQuadShader->m_shaderDescription->m_vertexFormats, &blitQuadMeshFg))
            return false;

        auto blitMaterialFg = m_MaterialManager->CreateMaterial(*pBlitQuadShader, NUM_VULKAN_BUFFERS,
            [this](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
            {
                if (texName == "Diffuse")
                {
                    if (m_Anf && m_Anf->IsFgValid())
                    {
                        static std::array<TextureVulkan*, NUM_VULKAN_BUFFERS> s_fg_outputs = {};
                        for (uint32_t i = 0; i < NUM_VULKAN_BUFFERS; ++i)
                        {
                            const auto& [tex, _] = m_Anf->GetFgOutputTexture(AnfFrameIndex(i, m_AnfFramesInFlight));
                            s_fg_outputs[i] = tex;
                        }
                        return { s_fg_outputs.begin(), s_fg_outputs.end() };
                    }
                    return { &m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[0] };
                }
                if (texName == "Overlay") return { &m_RenderPassData[RP_HUD].RenderTarget.m_ColorAttachments[0] };
                return {};
            },
            [](const std::string&) -> PerFrameBuffer { return {}; });

        m_BlitQuadDrawableFg = std::make_unique<Drawable>(vulkan, std::move(blitMaterialFg));
        if (!m_BlitQuadDrawableFg->Init(m_RenderPassData[RP_BLIT].RenderContext[0], std::move(blitQuadMeshFg)))
            return false;
    }

    // Debug blit: intermediate FG-input texture -> swapchain.
    // The real frame presents scene color or SR output; the generated frame
    // presents FG output.  Keeping this drawable available makes the FG color
    // input easy to inspect while debugging descriptor/resource issues.
    if (m_Anf)
    {
        Mesh blitQuadMeshFgInput;
        if (!MeshHelper::CreateMesh<Vulkan>(vulkan.GetMemoryManager(), MeshObjectIntermediate::CreateScreenSpaceMesh(), 0, pBlitQuadShader->m_shaderDescription->m_vertexFormats, &blitQuadMeshFgInput))
            return false;

        auto blitMaterialFgInput = m_MaterialManager->CreateMaterial(*pBlitQuadShader, NUM_VULKAN_BUFFERS,
            [this](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
            {
                if (texName == "Diffuse")
                {
                    static std::array<TextureVulkan*, NUM_VULKAN_BUFFERS> s_fg_inputs = {};
                    for (uint32_t i = 0; i < NUM_VULKAN_BUFFERS; ++i)
                        s_fg_inputs[i] = &m_FgInputTextures[i];
                    return { s_fg_inputs.begin(), s_fg_inputs.end() };
                }
                if (texName == "Overlay") return { &m_RenderPassData[RP_HUD].RenderTarget.m_ColorAttachments[0] };
                return {};
            },
            [](const std::string&) -> PerFrameBuffer { return {}; });

        m_BlitQuadDrawableFgInput = std::make_unique<Drawable>(vulkan, std::move(blitMaterialFgInput));
        if (!m_BlitQuadDrawableFgInput->Init(m_RenderPassData[RP_BLIT].RenderContext[0], std::move(blitQuadMeshFgInput)))
            return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::LoadAnimatedObjects()
//-----------------------------------------------------------------------------
{
    auto& vulkan = *GetVulkan();

    LOGI("*****************************");
    LOGI("Loading Animated Objects...  ");
    LOGI("*****************************");

    using AB = ObjectAnimationProperties::AnimationBehavior;
    using IB = ObjectAnimationProperties::InterpolationBehavior;

    struct AnimSetup
    {
        const char*                                         name;
        ObjectAnimationProperties::Keyframe                 start;
        ObjectAnimationProperties::Keyframe                 end;
        float                                               durationSeconds;
        ObjectAnimationProperties::AnimationBehavior        behavior;
        ObjectAnimationProperties::InterpolationBehavior    interp;
    };

    // GLTF-based animated objects (fan and gear from the scene mesh)
    const std::array<AnimSetup, 2> animSetups = {{
        {
            "anim_fan",
            { glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 0.0f),  glm::vec3(1.0f) },
            { glm::vec3(0.0f), glm::vec3(0.0f, 360.0f, 0.0f), glm::vec3(1.0f) },
            3.0f, AB::Repeat, IB::Lerp
        },
        {
            "anim_gear",
            { glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 0.0f),   glm::vec3(1.0f) },
            { glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 360.0f),  glm::vec3(1.0f) },
            5.0f, AB::Repeat, IB::Lerp
        },
    }};

    for (const auto& setup : animSetups)
    {
        auto& prop = m_AnimationProperties[setup.name];
        prop.Start              = setup.start;
        prop.End                = setup.end;
        prop.DurationSeconds    = setup.durationSeconds;
        prop.AnimationSpeed     = 1.0f;
        prop.State              = 0.0f;
        prop.PrevState          = 0.0f;
        prop.Behavior           = setup.behavior;
        prop.Interpolation      = setup.interp;
        prop.CurrentTranslation = setup.start.Translation;
        prop.CurrentRotation    = setup.start.Rotation;
        prop.CurrentScale       = setup.start.Scale;
        prop.PrevTranslation    = setup.start.Translation;
        prop.PrevRotation       = setup.start.Rotation;
        prop.PrevScale          = setup.start.Scale;

        if (!CreateUniformBuffer(&vulkan, prop.objectVertUniform))
        {
            LOGE("Failed to create animated object uniform buffer for '%s'", setup.name);
            return false;
        }
    }

    // Animated cube objects
    struct CubeSetup
    {
        std::string                                         cubeId;
        std::string                                         diffuseFilename;
        std::string                                         bumpFilename;
        std::string                                         specMapFilename;
        ObjectAnimationProperties::Keyframe                 start;
        ObjectAnimationProperties::Keyframe                 end;
        float                                               durationSeconds;
        ObjectAnimationProperties::AnimationBehavior        behavior;
        ObjectAnimationProperties::InterpolationBehavior    interp;
    };

    const std::array<CubeSetup, 6> cubeSetups = {{
        { "anim_roaming_cube",
          "SPS_tileWallA_albedo.ktx", "SPS_tileWallA_NM.ktx", "SPS_tileWallA_ORM_Linear.ktx",
          { glm::vec3(0,3,14),   glm::vec3(0,0,0),       glm::vec3(1,1,1) },
          { glm::vec3(0,3,-21),  glm::vec3(0,720,540),   glm::vec3(1,1,1) },
          10.0f, AB::Mirror, IB::Lerp },
        { "anim_jumping_cube",
          "plasterA_albedo.ktx", "plasterB_NM.ktx", "plasterA_ORM_Linear.ktx",
          { glm::vec3(-1.5f,0.7f,14), glm::vec3(0,0,0),   glm::vec3(1,1,1) },
          { glm::vec3(-1.5f,10,14),   glm::vec3(0,360,0), glm::vec3(1,1,1) },
          3.0f, AB::Mirror, IB::Lerp },
        { "anim_spinning_cube",
          "plasterB_albedo.ktx", "plasterB_NM.ktx", "plasterB_ORM_Linear.ktx",
          { glm::vec3(0,6,-4), glm::vec3(0,0,0),         glm::vec3(1,1,1) },
          { glm::vec3(0,6,-4), glm::vec3(240,1080,540),  glm::vec3(1,1,1) },
          10.0f, AB::Mirror, IB::Lerp },
        { "anim_sprinting_cube",
          "sideWalk_albedo.ktx", "sideWalk_NM.ktx", "sideWalk_ORM_Linear.ktx",
          { glm::vec3(1.5f,3,14),  glm::vec3(0,0,0), glm::vec3(1,1,1) },
          { glm::vec3(1.5f,3,-21), glm::vec3(0,0,0), glm::vec3(1,1,1) },
          4.0f, AB::Mirror, IB::Lerp },
        { "anim_stationary_cube",
          "SPS_stoneFloor_albedo.ktx", "SPS_stoneFloor_NM.ktx", "SPS_stoneFloor_ORM_Linear.ktx",
          { glm::vec3(-1.5f,2,0), glm::vec3(43,287,112), glm::vec3(1,1,1) },
          { glm::vec3(-1.5f,2,0), glm::vec3(43,287,112), glm::vec3(1,1,1) },
          1.0f, AB::Mirror, IB::Lerp },
        { "anim_growing_cube",
          "SPS_tileWallB_albedo.ktx", "SPS_tileWallB_NM.ktx", "SPS_tileWallB_ORM_Linear.ktx",
          { glm::vec3(-4,2.75f,2.8f), glm::vec3(45,30,0), glm::vec3(1,1,1) },
          { glm::vec3(-4,2.75f,2.8f), glm::vec3(45,30,0), glm::vec3(3,3,3) },
          6.0f, AB::Mirror, IB::Lerp },
    }};

    std::vector<MeshObjectIntermediate> cubeMeshVector;
    for (const auto& cube : cubeSetups)
    {
        // Register animation properties and create uniform buffer BEFORE
        // DrawableLoader::CreateDrawables calls the material loader, which
        // looks up m_AnimationProperties[materialName].objectVertUniform.
        auto& prop = m_AnimationProperties[cube.cubeId];
        prop.Start              = cube.start;
        prop.End                = cube.end;
        prop.DurationSeconds    = cube.durationSeconds;
        prop.AnimationSpeed     = 1.0f;
        prop.State              = 0.0f;
        prop.PrevState          = 0.0f;
        prop.Behavior           = cube.behavior;
        prop.Interpolation      = cube.interp;
        prop.CurrentTranslation = cube.start.Translation;
        prop.CurrentRotation    = cube.start.Rotation;
        prop.CurrentScale       = cube.start.Scale;
        prop.PrevTranslation    = cube.start.Translation;
        prop.PrevRotation       = cube.start.Rotation;
        prop.PrevScale          = cube.start.Scale;

        if (!CreateUniformBuffer(&vulkan, prop.objectVertUniform))
        {
            LOGE("Failed to create animated cube uniform buffer for '%s'", cube.cubeId.c_str());
            return false;
        }

        MeshObjectIntermediate cubeMeshIntermediate = CreateBasicCubeIntermediate();
        MeshObjectIntermediate::MaterialDef matDef;
        matDef.materialName    = cube.cubeId;
        matDef.materialId      = 0;
        matDef.diffuseFilename = cube.diffuseFilename;
        matDef.baseColorFactor = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
        matDef.bumpFilename    = cube.bumpFilename;
        matDef.specMapFilename = cube.specMapFilename;
        cubeMeshIntermediate.m_Materials.push_back(std::move(matDef));
        cubeMeshIntermediate.m_Transform = glm::identity<glm::mat4>();
        cubeMeshIntermediate.m_MeshName  = "Basic Cube Mesh";
        cubeMeshVector.emplace_back(std::move(cubeMeshIntermediate));
    }

    const uint32_t cubeLoaderFlags = 0;
    if (!DrawableLoader::CreateDrawables(vulkan, std::move(cubeMeshVector),
            m_RenderPassData[RP_SCENE].RenderContext,
            m_MaterialLoader, m_AnimatedDrawables, cubeLoaderFlags))
    {
        LOGE("Failed to create animated cube drawables");
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitCommandBuffers()
//-----------------------------------------------------------------------------
{
    auto* const pVulkan = GetVulkan();
    auto* const gpuTimerPool = m_GpuProfiler ? m_GpuProfiler->GetTimerPoolBase() : nullptr;

    LOGI("*****************************");
    LOGI("Initializing Command Buffers...");
    LOGI("*****************************");

    char szName[256];

    for (int whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        for (uint32_t whichBuffer = 0; whichBuffer < NUM_VULKAN_BUFFERS; ++whichBuffer)
        {
            sprintf(szName, "Primary (%s; Buffer %d)", sRenderPassNames[whichPass], whichBuffer);
            if (!m_RenderPassData[whichPass].PassCommandList[whichBuffer].Initialize(
                    pVulkan, szName, CommandListBase::Type::Primary, Vulkan::eGraphicsQueue, gpuTimerPool))
                return false;
        }
    }

    if (!m_AnfDispatchTiming ||
        !m_AnfDispatchTiming->Initialize(pVulkan, gpuTimerPool, gAnfDispatchImmediate))
    {
        LOGE("Unable to initialize ANF dispatch timing controller command buffers");
        return false;
    }

    for (uint32_t i = 0; i < NUM_VULKAN_BUFFERS; ++i)
    {
        sprintf(szName, "FG Input Blit CMD Buffer %d", i);
        if (!m_FgInputBlitCmdList[i].Initialize(pVulkan, szName, CommandListBase::Type::Primary, Vulkan::eGraphicsQueue, gpuTimerPool))
            return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::BuildCmdBuffers()
//-----------------------------------------------------------------------------
{
    return true;
}

//-----------------------------------------------------------------------------
void Application::UpdateGui()
//-----------------------------------------------------------------------------
{
    if (!m_Gui) return;

    m_Gui->Update();

    AnfRuntimeControlsUiState uiState{};
    uiState.SrSupported              = (m_Anf && m_Anf->IsSrValid());
    uiState.FgSupported              = (m_Anf && m_Anf->IsFgValid());
    uiState.ProfilingReady           = (m_GpuProfiler && m_GpuProfiler->IsInitialized());
    uiState.ActiveAnfFramesInFlight = m_AnfFramesInFlight;
    uiState.MaxFramesInFlightLimit   = NUM_VULKAN_BUFFERS;
    uiState.HasCalculatedFps         = m_HasCalculatedFps;
    uiState.CalculatedRealFps        = m_CalculatedRealFps;
    uiState.CalculatedPresentedFps   = m_CalculatedPresentedFps;
    uiState.Profiler                 = m_GpuProfiler.get();

    const auto uiActions = m_RuntimeControls->UpdateUi(uiState);
    if (uiActions.RequestSrReset)
        m_AnfReset = true;
    if (uiActions.RequestFgReset)
        m_AnfFgReset = true;
}

//-----------------------------------------------------------------------------
bool Application::UpdateUniforms(uint32_t whichBuffer, float delta)
//-----------------------------------------------------------------------------
{
    auto* const pVulkan = GetVulkan();

    // Vert data
    {
        glm::mat4 LocalModel = glm::mat4(1.0f);
        LocalModel           = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
        LocalModel           = glm::scale(LocalModel, glm::vec3(gSceneScale));

        m_ObjectVertUniformData.CurrViewProj         = m_CurrViewProjMatrix;
        m_ObjectVertUniformData.PrevViewProj         = m_PrevViewProjMatrix;
        m_ObjectVertUniformData.CurrViewProjNoJitter = m_CurrViewProjMatrixNoJitter;
        m_ObjectVertUniformData.PrevViewProjNoJitter = m_PrevViewProjMatrixNoJitter;
        m_ObjectVertUniformData.ModelMatrix          = LocalModel;
        m_ObjectVertUniformData.PrevModelMatrix      = LocalModel;
        m_ObjectVertUniformData.CurrentJitter        = m_CurrJitter;
        m_ObjectVertUniformData.PrevJitter           = m_PrevJitter;
        UpdateUniformBuffer(pVulkan, m_ObjectVertUniform, m_ObjectVertUniformData, whichBuffer);
    }

    // Animation data
    {
        for (auto& [ id, prop ] : m_AnimationProperties)
        {
            float diff = delta * prop.AnimationSpeed;

            if (gAnimationPaused)
                diff = 0.0f;

            prop.PrevState = prop.State;

            switch (prop.Behavior)
            {
            case ObjectAnimationProperties::AnimationBehavior::Mirror:
                prop.State += diff;
                if (prop.State > prop.DurationSeconds)
                    prop.State -= 2.0f * prop.DurationSeconds;
                prop.State = std::clamp(prop.State, -prop.DurationSeconds, prop.DurationSeconds);
                break;
            case ObjectAnimationProperties::AnimationBehavior::Repeat:
                prop.State += diff;
                if (prop.State > prop.DurationSeconds)
                    prop.State -= prop.DurationSeconds;
                prop.State = std::clamp(prop.State, 0.0f, prop.DurationSeconds);
                break;
            default:
                break;
            }

            std::function<glm::vec3(const glm::vec3&, const glm::vec3&, const float&)> interpolationFunction;
            switch (prop.Interpolation)
            {
            case ObjectAnimationProperties::InterpolationBehavior::Lerp:
                interpolationFunction = lerp;
                break;
            case ObjectAnimationProperties::InterpolationBehavior::Smoothstep:
                interpolationFunction = smoothstep;
                break;
            default:
                interpolationFunction = lerp;
                break;
            }

            prop.CurrentTranslation = interpolationFunction(prop.Start.Translation, prop.End.Translation, std::fabsf(prop.State) / prop.DurationSeconds);
            prop.CurrentRotation    = interpolationFunction(prop.Start.Rotation,    prop.End.Rotation,    std::fabsf(prop.State) / prop.DurationSeconds);
            prop.CurrentScale       = interpolationFunction(prop.Start.Scale,       prop.End.Scale,       std::fabsf(prop.State) / prop.DurationSeconds);

            prop.PrevTranslation    = interpolationFunction(prop.Start.Translation, prop.End.Translation, std::fabsf(prop.PrevState) / prop.DurationSeconds);
            prop.PrevRotation       = interpolationFunction(prop.Start.Rotation,    prop.End.Rotation,    std::fabsf(prop.PrevState) / prop.DurationSeconds);
            prop.PrevScale          = interpolationFunction(prop.Start.Scale,       prop.End.Scale,       std::fabsf(prop.PrevState) / prop.DurationSeconds);

            glm::mat4 LocalModel = glm::mat4(1.0f);
            LocalModel = glm::translate(LocalModel, prop.CurrentTranslation);
            LocalModel = rotateEuler(LocalModel, prop.CurrentRotation);
            LocalModel = glm::scale(LocalModel, glm::vec3(gSceneScale) * prop.CurrentScale);

            glm::mat4 PrevLocalModel = glm::mat4(1.0f);
            PrevLocalModel = glm::translate(PrevLocalModel, prop.PrevTranslation);
            PrevLocalModel = rotateEuler(PrevLocalModel, prop.PrevRotation);
            PrevLocalModel = glm::scale(PrevLocalModel, glm::vec3(gSceneScale) * prop.PrevScale);

            prop.objectVertUniformData.CurrViewProj         = m_CurrViewProjMatrix;
            prop.objectVertUniformData.PrevViewProj         = m_PrevViewProjMatrix;
            prop.objectVertUniformData.CurrViewProjNoJitter = m_CurrViewProjMatrixNoJitter;
            prop.objectVertUniformData.PrevViewProjNoJitter = m_PrevViewProjMatrixNoJitter;
            prop.objectVertUniformData.ModelMatrix          = LocalModel;
            prop.objectVertUniformData.PrevModelMatrix      = PrevLocalModel;
            prop.objectVertUniformData.CurrentJitter        = m_CurrJitter;
            prop.objectVertUniformData.PrevJitter           = m_PrevJitter;
            UpdateUniformBuffer(pVulkan, prop.objectVertUniform, prop.objectVertUniformData, whichBuffer);
        }
    }

    // Frag data
    for (auto& [hash, objectUniform] : m_ObjectFragUniforms)
        UpdateUniformBuffer(pVulkan, objectUniform.objectFragUniform, objectUniform.objectFragUniformData);

    // Light data
    {
        glm::mat4 CameraViewInv       = glm::inverse(m_Camera.ViewMatrix());
        glm::mat4 CameraProjection    = m_Camera.ProjectionMatrix();
        glm::mat4 CameraProjectionInv = glm::inverse(CameraProjection);

        m_LightUniformData.ProjectionInv     = CameraProjectionInv;
        m_LightUniformData.ViewInv           = CameraViewInv;
        m_LightUniformData.ViewProjectionInv = glm::inverse(CameraProjection * m_Camera.ViewMatrix());
        m_LightUniformData.ProjectionInvW    = glm::vec4(CameraProjectionInv[0].w, CameraProjectionInv[1].w, CameraProjectionInv[2].w, CameraProjectionInv[3].w);
        m_LightUniformData.CameraPos         = glm::vec4(m_Camera.Position(), 0.0f);
        m_LightUniformData.Width             = gRenderWidth / 2.0f;
        m_LightUniformData.Height            = gRenderHeight / 2.0f;
        m_LightUniformData.Debug_MVEnabled   = !m_Debug_ForceZeroMV;
        UpdateUniformBuffer(pVulkan, m_LightUniform, m_LightUniformData, whichBuffer);
    }

    return true;
}

//-----------------------------------------------------------------------------
void Application::Render(float fltDiffTime)
//-----------------------------------------------------------------------------
{
    auto* const pVulkan = GetVulkan();

    const auto runtimeActions = m_RuntimeControls->UpdatePerFrame(fltDiffTime);
    if (runtimeActions.RequestSrReset)
        m_AnfReset = true;
    if (runtimeActions.RequestFgReset)
        m_AnfFgReset = true;

    const bool desiredFgOnSrMode = m_RuntimeControls->ShouldUseFgOnSrMode();
    const bool anfSettingsChanged =
        m_LastAnfDispatchImmediate != gAnfDispatchImmediate ||
        m_LastAnfSrQualityMode     != gAnfSrQualityMode ||
        m_LastAnfMaxFramesInFlight != gAnfMaxFramesInFlight ||
        m_LastAnfFgOnSrMode        != desiredFgOnSrMode;

    if (m_FgRealFramePresentPending &&
        (!gFrameGenerationEnabled || anfSettingsChanged || !m_Anf || !m_Anf->IsFgValid()))
    {
        PresentQueue({ &m_FgRealFramePresentSemaphore, 1 }, m_FgRealFramePresentIdx);
        ClearPendingFgRealFramePresent();
        m_FgFramePending = false;
    }

    // -----------------------------------------------------------------------
    // Check for settings changes that require ANF re-initialization.
    // -----------------------------------------------------------------------
    CheckAndReinitializeAnf();

    // -----------------------------------------------------------------------
    // Determine what is active this frame.
    // is_fg_frame: this Render() call is the FG fake frame inserted between two
    // real frames.  When true, scene rendering, camera updates, SR, and the
    // intermediate blit are all skipped; only HUD, FG dispatch, and blit run.
    // -----------------------------------------------------------------------
    const bool is_sr_active = gUpscalingEnabled       && m_Anf && m_Anf->IsSrValid();
    const bool is_fg_active = gFrameGenerationEnabled && m_Anf && m_Anf->IsFgValid();
    const bool is_fg_frame  = is_fg_active && m_FgFramePending;

    if (!is_fg_frame)
        ++m_FpsAccumRealFrames;

    if (m_AnfFramesInFlight == 1)
    {
        if ((is_sr_active || is_fg_active) && m_Anf)
        {
            // When the SDK is configured to run with a single in-flight frame,
            // stall before any new ANF dispatch so the lone SDK slot is not
            // reused while still busy.
            vkDeviceWaitIdle(pVulkan->m_VulkanDevice);
        }
    }

    // Acquire the next swapchain image (needed for both real and fake frames).
    auto currentVulkanBuffer   = pVulkan->SetNextBackBuffer();
    const uint32_t whichBuffer = currentVulkanBuffer.idx;
    VkSemaphore swapchainAcquireSemaphore = currentVulkanBuffer.semaphore;

    if (m_GpuProfiler)
        m_GpuProfiler->UpdateCompleted(whichBuffer);

    const uint32_t qfi   = static_cast<uint32_t>(pVulkan->m_VulkanQueues[Vulkan::eGraphicsQueue].QueueFamilyIndex);
    const VkQueue  queue = pVulkan->m_VulkanQueues[Vulkan::eGraphicsQueue].Queue;

    // contentSem: semaphore signaling the content image is ready for the blit pass.
    // Set by the intermediate blit (real frame) or FG dispatch (fake frame).
    VkSemaphore contentSem = VK_NULL_HANDLE;

    UpdateGui();

    // -----------------------------------------------------------------------
    // Real frame: camera update, scene render, SR dispatch, intermediate blit.
    // Skipped entirely on the FG fake frame.
    // -----------------------------------------------------------------------
    if (!is_fg_frame)
    {
        const uint32_t frame_index = m_FrameCounter++;

        // Camera jitter (Halton sequence) - only when SR is enabled.
        glm::vec2 currentJitterPixel = glm::vec2(0.0f, 0.0f);
        if (is_sr_active && !m_Debug_ForceDisableJitter)
        {
            constexpr float upscale = 2.0f;
            const float render_w = float(m_RenderPassData[RP_SCENE].RenderTarget.m_Width);
            const float render_h = float(m_RenderPassData[RP_SCENE].RenderTarget.m_Height);

            static constexpr uint32_t kJitterSampleCount = 16;
            glm::vec2 jitter_out_px = Anf::HaltonJitter::GetJitter(frame_index, kJitterSampleCount);

            // Invert jitter (due to how we apply to our camera)
            jitter_out_px.x = -jitter_out_px.x;
            jitter_out_px.y = -jitter_out_px.y;
            currentJitterPixel = jitter_out_px;

            glm::vec2 jitter_render_px = jitter_out_px / upscale;
            glm::vec2 jitter_ndc = {
                (2.0f * jitter_render_px.x) / render_w,
                (2.0f * jitter_render_px.y) / render_h
            };
            m_Camera.SetJitter(jitter_ndc);
        }
        else
        {
            m_Camera.SetJitter(glm::vec2(0.0f, 0.0f));
        }

        m_CurrJitter = m_Camera.Jitter();

        if (m_CameraController)
            m_Camera.UpdateController(fltDiffTime, *m_CameraController);
        m_Camera.UpdateMatrices();

        m_CurrViewProjMatrix         = m_Camera.ProjectionMatrix() * m_Camera.ViewMatrix();
        m_CurrViewProjMatrixNoJitter = m_Camera.ProjectionMatrixNoJitter() * m_Camera.ViewMatrix();

        UpdateUniforms(whichBuffer, fltDiffTime);

        // Update prev matrices AFTER UpdateUniforms
        m_PrevViewProjMatrixNoJitter = m_CurrViewProjMatrixNoJitter;
        m_PrevViewProjMatrix         = m_CurrViewProjMatrix;
        m_PrevJitter                 = m_CurrJitter;

        if (m_Debug_ForceWaitForIdle)
            vkDeviceWaitIdle(pVulkan->m_VulkanDevice);

        // Scene render pass
        {
            auto& passData = m_RenderPassData[RP_SCENE];
            auto& cmdList  = passData.PassCommandList[whichBuffer];

            cmdList.Reset();
            cmdList.Begin();
            const int sceneTimerId = m_GpuProfiler ? m_GpuProfiler->BeginRegion(cmdList, AnfGpuProfiler::Region::SceneReal) : -1;
            cmdList.BeginRenderPass(passData.RenderContext[0], VK_SUBPASS_CONTENTS_INLINE);
            {
                const uint32_t w = passData.RenderTarget.m_Width;
                const uint32_t h = passData.RenderTarget.m_Height;
                VkViewport vp{};
                vp.x        = 0.f;
                vp.y        = float(h);
                vp.width    = float(w);
                vp.height   = -float(h);
                vp.minDepth = 0.f;
                vp.maxDepth = 1.f;
                VkRect2D sc{};
                sc.offset = {0, 0};
                sc.extent = {w, h};
                vkCmdSetViewport(cmdList.m_VkCommandBuffer, 0, 1, &vp);
                vkCmdSetScissor (cmdList.m_VkCommandBuffer, 0, 1, &sc);
            }

            for (const auto& drawable : m_SceneDrawables)
                AddDrawableToCmdBuffer(drawable, cmdList, whichBuffer);
            for (const auto& drawable : m_AnimatedDrawables)
                AddDrawableToCmdBuffer(drawable, cmdList, whichBuffer);

            cmdList.EndRenderPass();
            AnfGpuProfiler::EndRegion(cmdList, sceneTimerId);
            cmdList.End();

            cmdList.QueueSubmit(
                std::span<const VkSemaphore>{},
                std::span<const VkPipelineStageFlags>{},
                { &passData.PassCompleteSemaphores[whichBuffer], 1 });
        }

        // ANF SR dispatch
        std::span<const VkSemaphore> pWaitSemaphores = { &m_RenderPassData[RP_SCENE].PassCompleteSemaphores[whichBuffer], 1 };
        VkSemaphore realContentSem = m_RenderPassData[RP_SCENE].PassCompleteSemaphores[whichBuffer];
        bool srDispatchSucceeded = false;
        TextureVulkan* sceneAnfDepthImage = gAnfUseInverseDepth
            ? &m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[2]
            : &m_RenderPassData[RP_SCENE].RenderTarget.m_DepthAttachment;
        VkImageLayout sceneAnfDepthLayout = gAnfUseInverseDepth
            ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        VkImageAspectFlags sceneAnfDepthAspect = gAnfUseInverseDepth
            ? VK_IMAGE_ASPECT_COLOR_BIT
            : VK_IMAGE_ASPECT_DEPTH_BIT;

        if (is_sr_active)
        {
            const uint32_t anfBuffer = AnfFrameIndex(whichBuffer, m_AnfFramesInFlight);

            Anf::UpscalingInitData sr_props{};
            sr_props.ColorImage         = &m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[0];
            sr_props.ColorLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            sr_props.DepthImage         = sceneAnfDepthImage;
            sr_props.DepthLayout        = sceneAnfDepthLayout;
            sr_props.MotionVectorImage  = &m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[1];
            sr_props.MotionVectorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            // currentJitterPixel is the inverted Halton value (-Halton).
            // Invert again to recover the raw Halton value for ANF.
            sr_props.JitterOffset   = currentJitterPixel;
            sr_props.JitterOffset.x = -sr_props.JitterOffset.x;
            sr_props.JitterOffset.y = -sr_props.JitterOffset.y;

            const std::span<const VkSemaphore> srSignal = { &m_AnfSemaphores[whichBuffer], 1 };

            const std::span<const VkSemaphore> fallbackAnfWaits = m_Anf->IsDispatchImmediate()
                ? pWaitSemaphores
                : std::span<const VkSemaphore>{};
            const std::span<const VkSemaphore> fallbackAnfSignals = m_Anf->IsDispatchImmediate()
                ? srSignal
                : std::span<const VkSemaphore>{};

            AnfDispatchTimingController::DispatchToken dispatchToken{};
            if (m_AnfDispatchTiming)
            {
                dispatchToken = m_AnfDispatchTiming->BeginDispatch(
                    AnfDispatchTimingController::Technique::Sr,
                    whichBuffer,
                    pWaitSemaphores,
                    m_GpuProfiler.get(),
                    AnfGpuProfiler::Region::SrDispatch);
            }

            const auto dispatchWaitSems = m_AnfDispatchTiming
                ? m_AnfDispatchTiming->GetDispatchWaitSemaphores(dispatchToken, fallbackAnfWaits)
                : fallbackAnfWaits;
            const auto dispatchSignalSems = m_AnfDispatchTiming
                ? m_AnfDispatchTiming->GetDispatchSignalSemaphores(dispatchToken, fallbackAnfSignals)
                : fallbackAnfSignals;
            const VkCommandBuffer dispatchCmd = m_AnfDispatchTiming
                ? m_AnfDispatchTiming->GetDispatchCommandBuffer(dispatchToken)
                : VK_NULL_HANDLE;

            const auto srStatus = m_Anf->RenderSR(anfBuffer, dispatchCmd, sr_props,
                dispatchWaitSems, dispatchSignalSems, qfi, queue, m_AnfReset);
            if (srStatus != Anf::StatusCode::SUCCESS)
            {
                if (m_AnfDispatchTiming)
                    m_AnfDispatchTiming->CancelDispatch(AnfDispatchTimingController::Technique::Sr, whichBuffer, dispatchToken);
                LOGW("ANF SR dispatch failed for buffer %u", whichBuffer);
                m_AnfReset = true;
            }
            else
            {
                if (m_AnfDispatchTiming)
                {
                    std::vector<VkPipelineStageFlags> submitWaitStages(pWaitSemaphores.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
                    m_AnfDispatchTiming->EndDispatch(
                        AnfDispatchTimingController::Technique::Sr,
                        whichBuffer,
                        pWaitSemaphores,
                        submitWaitStages,
                        m_AnfSemaphores[whichBuffer],
                        dispatchToken);
                }

                m_AnfReset     = false;
                pWaitSemaphores = { &m_AnfSemaphores[whichBuffer], 1 };
                realContentSem  = m_AnfSemaphores[whichBuffer];
                srDispatchSucceeded = true;

                m_RuntimeControls->ConfirmSrRanOnce();
            }
        }

        // Intermediate FG input prep.
        // FG-only path: scene-res color copy (existing behavior).
        // SR+FG path:   full-res SR color + upscaled depth/motion to full-res.
        if (is_fg_active && m_FgInputTextures[whichBuffer])
        {
            const auto& sceneColorTex = m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[0];
            const auto& sceneMvTex    = m_RenderPassData[RP_SCENE].RenderTarget.m_ColorAttachments[1];
            const auto& sceneDepthTex = *sceneAnfDepthImage;

            bool useFgOnSrInputs =
                m_AnfFgOnSrMode &&
                is_sr_active &&
                srDispatchSucceeded &&
                m_Anf &&
                m_Anf->IsSrValid() &&
                static_cast<bool>(gAnfUseInverseDepth ? m_FgInverseDepthInputTextures[whichBuffer] : m_FgDepthInputTextures[whichBuffer]) &&
                static_cast<bool>(m_FgMotionInputTextures[whichBuffer]);

            TextureVulkan& fgPreparedDepthTex = gAnfUseInverseDepth
                ? m_FgInverseDepthInputTextures[whichBuffer]
                : m_FgDepthInputTextures[whichBuffer];
            VkImageAspectFlags fgPreparedDepthAspect = gAnfUseInverseDepth
                ? VK_IMAGE_ASPECT_COLOR_BIT
                : VK_IMAGE_ASPECT_DEPTH_BIT;
            VkImageLayout fgPreparedDepthLayout = gAnfUseInverseDepth
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            VkAccessFlags sceneDepthReadAccess = gAnfUseInverseDepth
                ? VK_ACCESS_SHADER_READ_BIT
                : (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            const TextureVulkan* fgColorSourceTex = &sceneColorTex;
            VkImageLayout fgColorSourceLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            if (useFgOnSrInputs)
            {
                const auto& [srOutputTex, srOutputLayout] = m_Anf->GetSrOutputTexture(AnfFrameIndex(whichBuffer, m_AnfFramesInFlight));
                if (srOutputTex)
                {
                    fgColorSourceTex = srOutputTex;
                    fgColorSourceLayout = srOutputLayout;
                }
                else
                    useFgOnSrInputs = false;
            }

            const VkImage    colorSrcImage  = fgColorSourceTex->GetVkImage();
            const VkExtent2D colorSrcExtent = { fgColorSourceTex->Width, fgColorSourceTex->Height };

            const VkImage    colorDstImage  = m_FgInputTextures[whichBuffer].GetVkImage();
            const VkExtent2D colorDstExtent = { m_FgInputTextures[whichBuffer].Width,
                                                m_FgInputTextures[whichBuffer].Height };

            auto& cmd = m_FgInputBlitCmdList[whichBuffer];
            cmd.Reset();
            cmd.Begin();
            const int fgPrepTimerId = m_GpuProfiler ? m_GpuProfiler->BeginRegion(cmd, AnfGpuProfiler::Region::FgPrepareReal) : -1;

            auto transitionImage = [&](VkImage image,
                                       VkImageAspectFlags aspectMask,
                                       VkImageLayout oldLayout,
                                       VkImageLayout newLayout,
                                       VkAccessFlags srcAccess,
                                       VkAccessFlags dstAccess)
            {
                VkImageMemoryBarrier barrier{};
                barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.oldLayout           = oldLayout;
                barrier.newLayout           = newLayout;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image               = image;
                barrier.subresourceRange    = { aspectMask, 0, 1, 0, 1 };
                barrier.srcAccessMask       = srcAccess;
                barrier.dstAccessMask       = dstAccess;
                vkCmdPipelineBarrier(cmd.m_VkCommandBuffer,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &barrier);
            };

            auto blitImage = [&](VkImage srcImage,
                                 VkExtent2D srcExtent,
                                 VkImageAspectFlags srcAspectMask,
                                 VkImage dstImage,
                                 VkExtent2D dstExtent,
                                 VkImageAspectFlags dstAspectMask,
                                 VkFilter filter)
            {
                VkImageBlit region{};
                region.srcSubresource = { srcAspectMask, 0, 0, 1 };
                region.srcOffsets[0]  = { 0, 0, 0 };
                region.srcOffsets[1]  = { static_cast<int32_t>(srcExtent.width), static_cast<int32_t>(srcExtent.height), 1 };
                region.dstSubresource = { dstAspectMask, 0, 0, 1 };
                region.dstOffsets[0]  = { 0, 0, 0 };
                region.dstOffsets[1]  = { static_cast<int32_t>(dstExtent.width), static_cast<int32_t>(dstExtent.height), 1 };
                vkCmdBlitImage(cmd.m_VkCommandBuffer,
                    srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &region, filter);
            };

            transitionImage(
                colorSrcImage,
                VK_IMAGE_ASPECT_COLOR_BIT,
                fgColorSourceLayout,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_TRANSFER_READ_BIT);
            transitionImage(
                colorDstImage,
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0,
                VK_ACCESS_TRANSFER_WRITE_BIT);
            blitImage(
                colorSrcImage, colorSrcExtent, VK_IMAGE_ASPECT_COLOR_BIT,
                colorDstImage, colorDstExtent, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_FILTER_LINEAR);
            transitionImage(
                colorSrcImage,
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                fgColorSourceLayout,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_ACCESS_SHADER_READ_BIT);
            transitionImage(
                colorDstImage,
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT);

            if (useFgOnSrInputs)
            {
                const VkImage    mvSrcImage  = sceneMvTex.GetVkImage();
                const VkExtent2D mvSrcExtent = { sceneMvTex.Width, sceneMvTex.Height };
                const VkImage    mvDstImage  = m_FgMotionInputTextures[whichBuffer].GetVkImage();
                const VkExtent2D mvDstExtent = { m_FgMotionInputTextures[whichBuffer].Width, m_FgMotionInputTextures[whichBuffer].Height };

                transitionImage(
                    mvSrcImage,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_ACCESS_SHADER_READ_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT);
                transitionImage(
                    mvDstImage,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    0,
                    VK_ACCESS_TRANSFER_WRITE_BIT);
                blitImage(
                    mvSrcImage, mvSrcExtent, VK_IMAGE_ASPECT_COLOR_BIT,
                    mvDstImage, mvDstExtent, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_FILTER_LINEAR);
                transitionImage(
                    mvSrcImage,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_ACCESS_SHADER_READ_BIT);
                transitionImage(
                    mvDstImage,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT);

                const VkImage    depthSrcImage  = sceneDepthTex.GetVkImage();
                const VkExtent2D depthSrcExtent = { sceneDepthTex.Width, sceneDepthTex.Height };
                const VkImage    depthDstImage  = fgPreparedDepthTex.GetVkImage();
                const VkExtent2D depthDstExtent = { fgPreparedDepthTex.Width, fgPreparedDepthTex.Height };

                transitionImage(
                    depthSrcImage,
                    sceneAnfDepthAspect,
                    sceneAnfDepthLayout,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    sceneDepthReadAccess,
                    VK_ACCESS_TRANSFER_READ_BIT);
                transitionImage(
                    depthDstImage,
                    fgPreparedDepthAspect,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    0,
                    VK_ACCESS_TRANSFER_WRITE_BIT);
                blitImage(
                    depthSrcImage, depthSrcExtent, sceneAnfDepthAspect,
                    depthDstImage, depthDstExtent, fgPreparedDepthAspect,
                    VK_FILTER_NEAREST);
                transitionImage(
                    depthSrcImage,
                    sceneAnfDepthAspect,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    sceneAnfDepthLayout,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_ACCESS_SHADER_READ_BIT);
                transitionImage(
                    depthDstImage,
                    fgPreparedDepthAspect,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    fgPreparedDepthLayout,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT);
            }

            AnfGpuProfiler::EndRegion(cmd, fgPrepTimerId);
            cmd.End();

            const std::array<VkSemaphore, 2> blitSignalBoth = {
                m_FgInputReadySemaphores[whichBuffer],
                m_FgReadySemaphores[whichBuffer]
            };
            std::vector<VkPipelineStageFlags> blitWaitStages(pWaitSemaphores.size(), VK_PIPELINE_STAGE_TRANSFER_BIT);
            cmd.QueueSubmit(pWaitSemaphores, { blitWaitStages.data(), blitWaitStages.size() },
                { blitSignalBoth.data(), blitSignalBoth.size() });

            realContentSem = m_FgInputReadySemaphores[whichBuffer];
        }

        contentSem = realContentSem;
    }

    // -----------------------------------------------------------------------
    // HUD render pass (both real and fake frames)
    // -----------------------------------------------------------------------
    VkSemaphore hudSemaphore = VK_NULL_HANDLE;
    {
        auto& passData = m_RenderPassData[RP_HUD];
        auto& cmdList  = passData.PassCommandList[whichBuffer];

        VkCommandBuffer guiCmdBuffer = m_Gui
            ? static_cast<GuiImguiGfx*>(m_Gui.get())->Render(whichBuffer, passData.RenderContext[0].GetFramebuffer()->m_FrameBuffer)
            : VK_NULL_HANDLE;

        if (guiCmdBuffer != VK_NULL_HANDLE)
        {
            cmdList.Reset();
            cmdList.Begin();
            const AnfGpuProfiler::Region hudRegion = is_fg_frame ? AnfGpuProfiler::Region::HudFake : AnfGpuProfiler::Region::HudReal;
            const int hudTimerId = m_GpuProfiler ? m_GpuProfiler->BeginRegion(cmdList, hudRegion) : -1;
            cmdList.BeginRenderPass(passData.RenderContext[0], VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
            vkCmdExecuteCommands(cmdList.m_VkCommandBuffer, 1, &guiCmdBuffer);
            cmdList.EndRenderPass();
            AnfGpuProfiler::EndRegion(cmdList, hudTimerId);
            cmdList.End();

            cmdList.QueueSubmit(
                std::span<const VkSemaphore>{},
                std::span<const VkPipelineStageFlags>{},
                { &passData.PassCompleteSemaphores[whichBuffer], 1 });
            hudSemaphore = passData.PassCompleteSemaphores[whichBuffer];
        }
    }

    // -----------------------------------------------------------------------
    // FG dispatch (fake frame only)
    // Waits on swapchain acquire + m_FgReadySemaphores from the real frame.
    // The swapchain semaphore is consumed here; the blit pass does NOT wait on it.
    // -----------------------------------------------------------------------
    if (is_fg_frame)
    {
        const std::array<VkSemaphore, 2> fgWaitSems = {
            swapchainAcquireSemaphore,
            m_FgReadySemaphores[m_FgFrameBuffer]
        };
        contentSem = RenderFgFrame(whichBuffer, fgWaitSems);

        if (contentSem != VK_NULL_HANDLE)
            m_RuntimeControls->ConfirmFgRanOnce();

        if (contentSem != VK_NULL_HANDLE)
        {
            // FG dispatch consumes the swapchain acquire semaphore for fake frames.
            // Avoid waiting on the same binary semaphore again in the blit submit.
            swapchainAcquireSemaphore = VK_NULL_HANDLE;
        }

    }

    // -----------------------------------------------------------------------
    // Blit pass (both real and fake frames)
    // Composites the content image with the HUD overlay onto the swapchain.
    //
    // Real frame:  drawable = SR output when SR is active, else scene color
    //              waits on contentSem + hudSem + swapchain acquire
    // Fake frame:  drawable = m_BlitQuadDrawableFg (or scene fallback)
    //              FG dispatch consumes swapchain acquire; blit waits on FG output
    //              plus HUD. If FG dispatch fails, blit consumes acquire instead.
    // -----------------------------------------------------------------------
    {
        auto& passData = m_RenderPassData[RP_BLIT];

        Drawable* pBlitDrawable;
        if (is_fg_frame)
            pBlitDrawable = m_BlitQuadDrawableFg ? m_BlitQuadDrawableFg.get() : m_BlitQuadDrawable.get();
        else
        {
            pBlitDrawable = m_BlitQuadDrawable.get();
            if (is_sr_active && m_BlitQuadDrawableSr) pBlitDrawable = m_BlitQuadDrawableSr.get();
        }

        auto& cmdList = passData.PassCommandList[whichBuffer];
        cmdList.Reset();
        cmdList.Begin();
        const AnfGpuProfiler::Region blitRegion = is_fg_frame ? AnfGpuProfiler::Region::BlitFake : AnfGpuProfiler::Region::BlitReal;
        const int blitTimerId = m_GpuProfiler ? m_GpuProfiler->BeginRegion(cmdList, blitRegion) : -1;

        cmdList.BeginRenderPass(passData.RenderContext[currentVulkanBuffer.swapchainPresentIdx], VK_SUBPASS_CONTENTS_INLINE);
        {
            const auto& cd = passData.RenderContext[currentVulkanBuffer.swapchainPresentIdx].GetRenderPassClearData();
            vkCmdSetViewport(cmdList.m_VkCommandBuffer, 0, 1, &cd.viewport);
            vkCmdSetScissor (cmdList.m_VkCommandBuffer, 0, 1, &cd.scissor);
        }

        if (pBlitDrawable)
            AddDrawableToCmdBuffer(*pBlitDrawable, cmdList, whichBuffer);

        cmdList.EndRenderPass();
        AnfGpuProfiler::EndRegion(cmdList, blitTimerId);
        if (m_GpuProfiler)
            m_GpuProfiler->RecordReadback(cmdList, whichBuffer);
        cmdList.End();

        // Build wait list: content + HUD + swapchain
        std::array<VkSemaphore, 3>          waitSems{};
        std::array<VkPipelineStageFlags, 3> waitStages{};
        uint32_t waitCount = 0;

        if (contentSem != VK_NULL_HANDLE)
        {
            waitSems[waitCount]   = contentSem;
            waitStages[waitCount] = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            ++waitCount;
        }
        if (hudSemaphore != VK_NULL_HANDLE)
        {
            waitSems[waitCount]   = hudSemaphore;
            waitStages[waitCount] = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            ++waitCount;
        }
        if (swapchainAcquireSemaphore != VK_NULL_HANDLE)
        {
            waitSems[waitCount]   = swapchainAcquireSemaphore;
            waitStages[waitCount] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            ++waitCount;
        }

        cmdList.QueueSubmit(
            std::span<const VkSemaphore>{ waitSems.data(), waitCount },
            std::span<const VkPipelineStageFlags>{ waitStages.data(), waitCount },
            { &passData.PassCompleteSemaphores[whichBuffer], 1 }, currentVulkanBuffer.fence);
    }

    // -----------------------------------------------------------------------
    // Present
    // -----------------------------------------------------------------------
    uint32_t presentsThisRender = 0;
    if (is_fg_frame)
    {
        PresentQueue(
            { &m_RenderPassData[RP_BLIT].PassCompleteSemaphores[whichBuffer], 1 },
            currentVulkanBuffer.swapchainPresentIdx);
        ++presentsThisRender;

        if (m_FgRealFramePresentPending)
        {
            PresentQueue({ &m_FgRealFramePresentSemaphore, 1 }, m_FgRealFramePresentIdx);
            ClearPendingFgRealFramePresent();
            ++presentsThisRender;
        }
    }
    else if (is_fg_active)
    {
        m_FgRealFramePresentPending = true;
        m_FgRealFramePresentSemaphore = m_RenderPassData[RP_BLIT].PassCompleteSemaphores[whichBuffer];
        m_FgRealFramePresentIdx = currentVulkanBuffer.swapchainPresentIdx;
    }
    else
    {
        PresentQueue(
            { &m_RenderPassData[RP_BLIT].PassCompleteSemaphores[whichBuffer], 1 },
            currentVulkanBuffer.swapchainPresentIdx);
        ++presentsThisRender;
    }

    // -----------------------------------------------------------------------
    // Update FG pending state
    // -----------------------------------------------------------------------
    if (is_fg_frame)
        m_FgFramePending = false;
    else if (is_fg_active)
    {
        m_FgFramePending = true;
        m_FgFrameBuffer  = whichBuffer;
    }

    m_FpsAccumPresentedFrames += presentsThisRender;
    m_FpsAccumSeconds += std::max(0.0f, fltDiffTime);
    constexpr float kFpsUpdateWindowSeconds = 0.5f;
    if (m_FpsAccumSeconds >= kFpsUpdateWindowSeconds)
    {
        const double seconds = static_cast<double>(m_FpsAccumSeconds);
        if (seconds > 0.0)
        {
            m_CalculatedRealFps = static_cast<double>(m_FpsAccumRealFrames) / seconds;
            m_CalculatedPresentedFps = static_cast<double>(m_FpsAccumPresentedFrames) / seconds;
            m_HasCalculatedFps = true;
        }

        m_FpsAccumSeconds = 0.0f;
        m_FpsAccumRealFrames = 0;
        m_FpsAccumPresentedFrames = 0;
    }
}
