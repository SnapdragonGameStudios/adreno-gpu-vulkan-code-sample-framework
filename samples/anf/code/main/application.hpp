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
#include "mesh/meshHelper.hpp"
#include "vulkan/commandBuffer.hpp"
#include "vulkan/framebuffer.hpp"
#include "vulkan/renderPass.hpp"
#include "anf_interface.hpp"
#include "anf_runtime_controls.hpp"
#include "halton_jitter.hpp"
#include <unordered_map>

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
    glm::mat4   CurrViewProj;
    glm::mat4   PrevViewProj;
    glm::mat4   CurrViewProjNoJitter;
    glm::mat4   PrevViewProjNoJitter;
    glm::mat4   ModelMatrix;
    glm::mat4   PrevModelMatrix;
    glm::mat4   ShadowMatrix;
    glm::vec2   CurrentJitter;
    glm::vec2   PrevJitter;
};

struct ObjectFragUB
{
    glm::vec4   Color;
    glm::vec4   ORM;
};

struct UpscalerUB
{
    glm::vec4   Color;
    glm::vec4   ORM;
};

struct LightUB
{
    glm::mat4 ProjectionInv;
    glm::mat4 ViewInv;
    glm::mat4 ViewProjectionInv;
    glm::vec4 ProjectionInvW;
    glm::vec4 CameraPos;

    int Width;
    int Height;
    int Debug_MVEnabled;
    int Debug_MVInvertX;
    int Debug_MVInvertY;
    float Debug_MVCompensate;
    int Debug_MVAxisInvertX;
    int Debug_MVAxisInvertY;

    glm::vec4 LightDirection = glm::vec4(-0.564000f, 0.826000f, 0.000000f, 0.0f);
    glm::vec4 LightColor = glm::vec4(1.000000f, 1.000000f, 1.000000f, 1.000000);

    glm::vec4 SpotLights_pos[NUM_SPOT_LIGHTS];
    glm::vec4 SpotLights_dir[NUM_SPOT_LIGHTS];
    glm::vec4 SpotLights_color[NUM_SPOT_LIGHTS];

    glm::vec4 AmbientColor = glm::vec4(0.340000f, 0.340000f, 0.340000f, 0.0f);

    float AmbientOcclusionScale = 1.0f;
};

// **********************
// Render Pass
// **********************
struct RenderPassSetupInfo
{
    RenderPassInputUsage    ColorInputUsage;
    bool                    ClearDepthRenderPass;
    RenderPassOutputUsage   ColorOutputUsage;
    RenderPassOutputUsage   DepthOutputUsage;
    glm::vec4               ClearColor;
};

struct RenderPassData
{
    RenderPassSetupInfo                 RenderPassSetup;
    std::vector<RenderContext<Vulkan>>  RenderContext;
    std::vector<CommandListVulkan>      ObjectsCmdBuffer;
    RenderTarget<Vulkan>                RenderTarget;
    std::array<VkSemaphore, NUM_VULKAN_BUFFERS> PassCompleteSemaphores{};
    std::array<CommandListVulkan, NUM_VULKAN_BUFFERS> PassCommandList;
};

class AnfGpuProfiler;
class AnfDispatchTimingController;

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

    struct ObjectAnimationProperties
    {
        enum class AnimationBehavior { Repeat, Mirror };
        enum class InterpolationBehavior { Lerp, Smoothstep };

        struct Keyframe
        {
            glm::vec3 Translation;
            glm::vec3 Rotation;  // Euler angles in degrees (Roll, Yaw, Pitch)
            glm::vec3 Scale;
        };

        UniformArrayT<ObjectVertUB, NUM_VULKAN_BUFFERS> objectVertUniform;
        ObjectVertUB                                    objectVertUniformData;
        Keyframe                                        Start;
        Keyframe                                        End;
        float                                           DurationSeconds;
        float                                           AnimationSpeed;
        float                                           State;
        float                                           PrevState;
        AnimationBehavior                               Behavior;
        InterpolationBehavior                           Interpolation;

        glm::vec3 CurrentTranslation;
        glm::vec3 CurrentRotation;
        glm::vec3 CurrentScale;

        glm::vec3 PrevTranslation;
        glm::vec3 PrevRotation;
        glm::vec3 PrevScale;
    };

public:
    Application();
    ~Application() override;

    virtual void PreInitializeSetVulkanConfiguration(Vulkan::AppConfiguration&) override;
    virtual bool Initialize(uintptr_t windowHandle, uintptr_t hInstance) override;
    virtual bool ReInitialize(uintptr_t windowHandle, uintptr_t hInstance) override;
    virtual void Destroy() override;
    virtual void Render(float fltDiffTime) override;

private:
    bool InitializeLights();
    bool InitializeCamera();
    bool LoadShaders();
    bool CreateRenderTargets();
    bool InitUniforms();
    bool InitSemaphores();
    bool InitAllRenderPasses();
    bool InitGui(uintptr_t windowHandle);
    bool LoadMeshObjects();
    bool LoadAnimatedObjects();
    bool InitCommandBuffers();
    bool BuildCmdBuffers();

    // Initializes (or re-initializes) ANF SR and FG using the current settings.
    // Caller must call vkDeviceWaitIdle before invoking this if ANF is already running.
    bool InitializeAnf();

    // Checks whether ANF settings (dispatch mode, SR quality) have changed since the last
    // frame and, if so, tears down and rebuilds ANF plus the dependent blit drawables.
    // Called at the top of Render() before any GPU work is submitted.
    void CheckAndReinitializeAnf();

    VkExtent2D GetAnfOutputExtent() const;
    VkExtent2D GetAnfSceneExtent() const;
    void ReleaseResolutionDependentResources();
    bool InitializeResolutionDependentResources(uintptr_t windowHandle);
    bool RebuildResolutionDependentResources();

    // Dispatches the FG interpolation for the fake frame.
    // No geometry is drawn; FG is dispatched using m_FgInputTextures[m_FgFrameBuffer]
    // (the scene-resolution intermediate texture written at the end of the previous real frame).
    // waitSems: semaphores to wait on before FG dispatch (swapchain acquire + FG ready).
    // Returns: the semaphore signaled when FG dispatch completes (m_AnfFgSemaphores[whichBuffer]).
    // whichBuffer is the app frame slot; the SDK frame slot is derived in application.cpp
    // from the configured ANF frame count.
    VkSemaphore RenderFgFrame(uint32_t whichBuffer, std::span<const VkSemaphore> waitSems);

    void ClearPendingFgRealFramePresent();

private:
    void UpdateGui();
    bool UpdateUniforms(uint32_t WhichBuffer, float delta);

private:
    uintptr_t                                          m_WindowHandle = 0;
    uint32_t m_FrameCounter = 0;

    RenderPass                                                  m_ObjectRenderPass;
    std::array<RenderPassData, NUM_RENDER_PASSES>               m_RenderPassData;

    UniformArrayT<ObjectVertUB, NUM_VULKAN_BUFFERS>             m_ObjectVertUniform;
    ObjectVertUB                                                m_ObjectVertUniformData;
    UniformArrayT<LightUB, NUM_VULKAN_BUFFERS>                  m_LightUniform;
    LightUB                                                     m_LightUniformData;
    std::unordered_map<std::size_t, ObjectMaterialParameters>   m_ObjectFragUniforms;
    std::unordered_map<std::string, ObjectAnimationProperties>  m_AnimationProperties;

    std::vector<Drawable>                                       m_SceneDrawables;
    std::vector<Drawable>                                       m_AnimatedDrawables;
    std::unique_ptr<Drawable>                                   m_BlitQuadDrawable;
    std::unique_ptr<Drawable>                                   m_BlitQuadDrawableSr;
    std::unique_ptr<Drawable>                                   m_BlitQuadDrawableFg;
    // Blit drawable that reads from m_FgInputTextures (the intermediate scene-res texture).
    // Kept for debugging the FG color input path.
    std::unique_ptr<Drawable>                                   m_BlitQuadDrawableFgInput;

    glm::mat4 m_PrevViewProjMatrix          = glm::mat4(1.0f);
    glm::mat4 m_CurrViewProjMatrix          = glm::mat4(1.0f);
    glm::mat4 m_PrevViewProjMatrixNoJitter  = glm::mat4(1.0f);
    glm::mat4 m_CurrViewProjMatrixNoJitter  = glm::mat4(1.0f);

    glm::vec2 m_PrevJitter = glm::vec2(0.0f);
    glm::vec2 m_CurrJitter = glm::vec2(0.0f);

    // ANF
    std::unique_ptr<Anf::AnfInterface>              m_Anf;
    std::unique_ptr<AnfGpuProfiler>                  m_GpuProfiler;
    std::unique_ptr<AnfDispatchTimingController>     m_AnfDispatchTiming;
    std::unique_ptr<AnfRuntimeControls>              m_RuntimeControls;
    int                                               m_LastAnfMaxFramesInFlight = 1;
    uint32_t                                          m_AnfFramesInFlight         = 1;
    bool                                              m_LastAnfFgOnSrMode       = false;
    bool                                              m_AnfFgOnSrMode           = false;
    int                                               m_LastAnfSrQualityMode    = 0;
    bool                                              m_LastAnfDispatchImmediate = true;
    VkExtent2D                                        m_LastAnfOutputExtent{};
    VkExtent2D                                        m_LastAnfSceneExtent{};
    std::array<VkSemaphore, NUM_VULKAN_BUFFERS>       m_AnfSemaphores{};
    // Signaled when the FG dispatch itself completes.
    std::array<VkSemaphore, NUM_VULKAN_BUFFERS>       m_AnfFgSemaphores{};
    // Signaled at the end of the real frame's main work (after SR if active, after scene if not).
    // Consumed by the fake frame's FG dispatch to ensure the FG input is ready.
    std::array<VkSemaphore, NUM_VULKAN_BUFFERS>       m_FgReadySemaphores{};
    // Signaled when the intermediate FG-input blit is complete.
    // Consumed by the real frame's blit pass so it reads from the fully-written
    // intermediate texture rather than directly from the SR output or scene color.
    std::array<VkSemaphore, NUM_VULKAN_BUFFERS>       m_FgInputReadySemaphores{};
    float                                             m_AnfUpscaleFactor        = 2.0f;
    // Per-frame command lists used to blit the SR output (or upscaled scene color)
    // into m_FgInputTextures.
    std::array<CommandListVulkan, NUM_VULKAN_BUFFERS> m_FgInputBlitCmdList;
    bool                                              m_AnfReset   = true;
    bool                                              m_AnfFgReset = true;
    // Frame Generation fake-frame state.
    // After each real frame, if FG is active, m_FgFramePending is set to true and
    // m_FgFrameBuffer records which buffer index was used.  The next Render() call
    // will dispatch the fake frame and clear m_FgFramePending.
    bool                                              m_FgFramePending  = false;
    uint32_t                                          m_FgFrameBuffer   = 0;

    // Per-frame scene-resolution intermediate textures (R8G8B8A8_UNORM).
    // FG requires color, depth, motion vectors, and output to use one consistent
    // extent.  The sample renders depth/motion at scene resolution, so the FG color
    // input is also copied from scene color at scene resolution.  SR still presents
    // its own full-resolution output on real frames.
    //
    // When m_AnfFgOnSrMode is true (SR+FG full mode), these textures are output
    // resolution and the depth/motion intermediates below are also populated.
    std::array<Texture<Vulkan>, NUM_VULKAN_BUFFERS>   m_FgInputTextures;
    std::array<Texture<Vulkan>, NUM_VULKAN_BUFFERS>   m_FgDepthInputTextures;
    std::array<Texture<Vulkan>, NUM_VULKAN_BUFFERS>   m_FgInverseDepthInputTextures;
    std::array<Texture<Vulkan>, NUM_VULKAN_BUFFERS>   m_FgMotionInputTextures;
    bool                                              m_FgRealFramePresentPending = false;
    VkSemaphore                                       m_FgRealFramePresentSemaphore = VK_NULL_HANDLE;
    uint32_t                                          m_FgRealFramePresentIdx = 0;
    float                                             m_FpsAccumSeconds = 0.0f;
    uint32_t                                          m_FpsAccumRealFrames = 0;
    uint32_t                                          m_FpsAccumPresentedFrames = 0;
    bool                                              m_HasCalculatedFps = false;
    double                                            m_CalculatedRealFps = 0.0;
    double                                            m_CalculatedPresentedFps = 0.0;

    std::function<std::optional<Material>(const MeshObjectIntermediate::MaterialDef&)> m_MaterialLoader;

    // Debug knobs
    bool m_Debug_ForceZeroMV        = false;
    bool m_Debug_ForceDisableJitter = false;
    bool m_Debug_ForceWaitForIdle   = false;
    bool m_Debug_UseFullDepth       = false;
};
