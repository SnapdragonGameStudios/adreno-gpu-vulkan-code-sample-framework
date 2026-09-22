// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2020 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

///
/// Sample app for VK_QCOM_rotated_copy_commands extension
///

#include "application.hpp"
#include "camera/cameraController.hpp"
#include "camera/cameraControllerTouch.hpp"
#include "gui/imguiVulkan.hpp"
#include "material/vulkan/descriptorSetLayout.hpp"
#include "material/vulkan/drawable.hpp"
#include "material/vulkan/material.hpp"
#include "material/vulkan/materialPass.hpp"
#include "material/vulkan/materialManager.hpp"
#include "material/vulkan/pipeline.hpp"
#include "material/vulkan/shaderModule.hpp"
#include "material/shaderManagerT.hpp"
#include "mesh/meshHelper.hpp"
#include "system/math_common.hpp"
#include "texture/vulkan/textureManager.hpp"
#include "imgui/imgui.h"
#include <bitset>
#include <string>

// Global Variables From Config File
VAR(char*,      gSceneAssetModel,   "SteamPunkSauna.gltf", kVariableNonpersistent);
VAR(glm::vec3,  gCameraStartPos,    glm::vec3(0.0f, 3.5f, 0.0f), kVariableNonpersistent);
VAR(glm::vec3,  gCameraStartRot,    glm::vec3(0.0f, 0.0f, 0.0f), kVariableNonpersistent);
VAR(float,      gFOV,               PI_DIV_4, kVariableNonpersistent);
VAR(float,      gNearPlane,         0.1f, kVariableNonpersistent);
VAR(float,      gFarPlane,          50.0f, kVariableNonpersistent);

VAR(bool,       gRotatedFinalBlit,  true, kVariableNonpersistent);


///
/// @brief Implementation of the Application entrypoint (called by the framework)
/// @return Pointer to Application (derived from @ApplicationHelperBase ... derived from @FrameworkApplicationBase).
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
int Application::PreInitializeSelectSurfaceFormat(std::span<const SurfaceFormat> formats)
//-----------------------------------------------------------------------------
{
    // On Snapdragon if the surfaceflinger has to do the rotation to the display native orientation then it will do it at 8bit colordepth.
    // To avoid this we need to enable the 'pre-rotation' of the display (and the use of VK_QCOM_render_pass_transform so we dont have to rotate our buffers/passes manually).
    GetVulkan()->m_UseRenderPassTransform = false;
    GetVulkan()->m_UsePreTransform = gRotatedFinalBlit;

    // We want to select a SRGB output format (if one exists)
    int index = 0;
    for (const auto& format : formats)
    {
        if (format.format == TextureFormat::B8G8R8A8_SRGB)
            return index;
        ++index;
    }
    return -1;
}

//-----------------------------------------------------------------------------
void Application::PreInitializeSetVulkanConfiguration( Vulkan::AppConfiguration& config )
//-----------------------------------------------------------------------------
{
    ApplicationHelperBase::PreInitializeSetVulkanConfiguration( config );
    config.RequiredExtension( "VK_KHR_copy_commands2" );
    config.OptionalExtension( "VK_QCOM_rotated_copy_commands" );
}

//-----------------------------------------------------------------------------
bool Application::Initialize( uintptr_t windowHandle, uintptr_t hInstance )
//-----------------------------------------------------------------------------
{
    if( !ApplicationHelperBase::Initialize( windowHandle, hInstance ) )
    {
        return false;
    }

    auto* const pVulkan = GetVulkan();

    // Get the pointer to the rotated blit function.
    if (vkCmdBlitImage2KHR == nullptr)
    {
        LOGE("Unable to get function pointer from instance: vkCmdBlitImage2KHR");
        // Disable rotated blit as we can't get the function pointer.
        gRotatedFinalBlit = false;
    }

    m_RequestedRotatedFinalBlit = gRotatedFinalBlit;

    LoadShaders();

    CreateUniformBuffer( pVulkan, m_ObjectVertUniform );
    CreateUniformBuffer( pVulkan, m_ObjectFragUniform );

    InitGui(windowHandle);

    InitCommandBuffers();

    InitFramebuffersRenderPassesAndDrawables();

    InitCamera();

    return true;
}


//-----------------------------------------------------------------------------
bool Application::LoadShaders()
//-----------------------------------------------------------------------------
{
    auto* const pVulkan = GetVulkan();

    //m_ShaderManager->RegisterRenderPassNames( {sRenderPassNames} );

    LOGI("Loading Shaders...");

    typedef std::pair<std::string, std::string> tIdAndFilename;
    for (const tIdAndFilename& i :
          { tIdAndFilename { "Object",                        "Object.json" },
            tIdAndFilename { "Tonemap",                       "Tonemap.json"}
          })
    {
        if (!m_ShaderManager->AddShader(*m_AssetManager, i.first, i.second, SHADER_DESTINATION_PATH))
        {
            LOGE("Error Loading shader %s from %s", i.first.c_str(), i.second.c_str());
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitFramebuffersRenderPassesAndDrawables()
//-----------------------------------------------------------------------------
{
    auto* const pVulkan = GetVulkan();

    //
    // Clean up old render targets and render passes
    m_SceneRT.Release();
    m_TonemapRT.Release();


    //
    // Create two seperate passes - Scene and Tonemap
    //

    const TextureFormat ScenePassColorFormats[] = { TextureFormat::A2B10G10R10_UNORM_PACK32 };
    const TEXTURE_TYPE ScenePassTextureTypes[] = { TT_RENDER_TARGET };
    const Msaa ScenePassMsaa[] = { Msaa::Samples1 };
    const TextureFormat ScenePassDepthFormat = pVulkan->GetBestSurfaceDepthFormat();

    const TextureFormat TonemapPassColorFormats[] = { TextureFormat::A2B10G10R10_UNORM_PACK32 };
    const Msaa TonemapPassMsaa[] = { Msaa::Samples1 };
    const TEXTURE_TYPE TonemapPassTextureTypes[] = { TT_RENDER_TARGET_TRANSFERSRC };

    //
    // Scene render pass
    RenderPass SceneRenderPass;
    if( !pVulkan->CreateRenderPass( ScenePassColorFormats,
                                    ScenePassDepthFormat,
                                    Msaa::Samples1,
                                    RenderPassInputUsage::Clear/*color*/,
                                    RenderPassOutputUsage::StoreReadOnly,/*color*/
                                    true,/*clear depth*/
                                    RenderPassOutputUsage::Discard,/*depth*/
                                    SceneRenderPass ) )
    {
        return false;
    }
    pVulkan->SetDebugObjectName( SceneRenderPass, "SceneRenderPass" );

    //
    // Tonemap render pass
    RenderPass TonemapRenderPass;
    if (!pVulkan->CreateRenderPass(TonemapPassColorFormats,
        TextureFormat::UNDEFINED,
        Msaa::Samples1,
        RenderPassInputUsage::DontCare/*color*/,
        RenderPassOutputUsage::StoreTransferSrc,/*color*/
        false,/*dont clear depth*/
        RenderPassOutputUsage::Discard,/*depth*/
        TonemapRenderPass))
    {
        return false;
    }
    pVulkan->SetDebugObjectName(TonemapRenderPass, "TonemapRenderPass");

    // Create render target for 3d scene.
    if (!m_SceneRT.Initialize( pVulkan, gRenderWidth, gRenderHeight, ScenePassColorFormats, ScenePassDepthFormat, "Scene RT", ScenePassTextureTypes, ScenePassMsaa)
        || !m_SceneRT.InitializeFrameBuffer(pVulkan, SceneRenderPass))
    {
        LOGE("Error initializing LinearColorRT");
        return false;
    }

    // Create the render target for the Tonemap/composite
    if( !m_TonemapRT.Initialize( pVulkan, gRenderWidth, gRenderHeight, TonemapPassColorFormats, TextureFormat::UNDEFINED, "Tonemap RT", TonemapPassTextureTypes, TonemapPassMsaa)
        || !m_TonemapRT.InitializeFrameBuffer(pVulkan, TonemapRenderPass))
    {
        LOGE( "Error initializing LinearColorRT" );
        return false;
    }

    m_SceneRenderContext   = RenderContext(std::move(SceneRenderPass), m_SceneRT.GetFrameBuffer(), "RP_OPAQUE");
    m_TonemapRenderContext = RenderContext(std::move(TonemapRenderPass), m_TonemapRT.GetFrameBuffer(), "Fullscreen");

    // Load Scene (object) drawable
    LoadSceneDrawables(m_SceneRenderContext, ScenePassMsaa[0]);

    // Tonemap drawable
    std::map<const std::string, const TextureBase*> tonemapTex = { {"Diffuse", &m_SceneRT.m_ColorAttachments.front() },
                                                                   {"Overlay", &m_GuiRT.m_ColorAttachments.front() } };
    m_TonemapDrawable = InitFullscreenDrawable( "Tonemap", tonemapTex, {}, m_TonemapRenderContext, 0, TonemapPassMsaa[0]);

    return true;
}

//-----------------------------------------------------------------------------
bool Application::LoadSceneDrawables( const RenderContext& renderContext, Msaa renderPassMultisamples )
//-----------------------------------------------------------------------------
{
    const auto sceneAssetPath = std::filesystem::path(MESH_DESTINATION_PATH).append(gSceneAssetModel).string();

    LOGI( "Loading Scene Drawable %s", sceneAssetPath.c_str());

    auto* const pVulkan = GetVulkan();

    const auto* pObjectShader = m_ShaderManager->GetShader( "Object" );
    if( !pObjectShader )
    {
        // Error (missing shader)
        return false;
    }

    m_TextureManager->SetDefaultFilenameManipulators(PathManipulator_PrefixDirectory(TEXTURE_DESTINATION_PATH));

    const PathManipulator_PrefixDirectory prefixTextureDir{ TEXTURE_DESTINATION_PATH };
    auto* whiteTexture = m_TextureManager->GetOrLoadTexture("white_d.ktx", m_SamplerRepeat, prefixTextureDir);
    if (!whiteTexture)
    {
        return false;
    }

    // Lambda to load a texture for the given material (slot)
    const auto& textureLoader = [&]( const MeshObjectIntermediate::MaterialDef& materialDef, const std::string& textureSlotName ) -> const MaterialManagerBase::tPerFrameTexInfo
    {
        const PathManipulator_PrefixDirectory prefixTextureDir{ TEXTURE_DESTINATION_PATH };
        const PathManipulator_ChangeExtension changeTextureExt{ ".ktx" };

        auto* texture = m_TextureManager->GetOrLoadTexture(materialDef.diffuseFilename, m_SamplerRepeat, prefixTextureDir, changeTextureExt);

        return { texture ? texture : whiteTexture };
    };

    // Lambda to associate (uniform) buffers with their shader binding/slot names.
    const auto& bufferLoader = [&]( const std::string& bufferSlotName ) -> PerFrameBuffer
    {
        if( bufferSlotName == "Vert" )
        {
            return { m_ObjectVertUniform.buf.GetVkBuffer() };
        }
        else if( bufferSlotName == "Frag" )
        {
            return { m_ObjectFragUniform.buf.GetVkBuffer() };
        }
        else
        {
            assert( 0 );
            return {};
        }
    };

    const auto& materialLoader = [&]( const MeshObjectIntermediate::MaterialDef& materialDef ) -> std::optional<Material>
    {
        using namespace std::placeholders;
        return m_MaterialManager->CreateMaterial( *pObjectShader, NUM_VULKAN_BUFFERS, std::bind( textureLoader, std::cref( materialDef ), _1 ), bufferLoader );
    };


    m_SceneObject.clear();

    if (!DrawableLoader::LoadDrawables( *pVulkan, *m_AssetManager, renderContext, sceneAssetPath, materialLoader, m_SceneObject, 0 ))
    {
        LOGE( "Error loading Object mesh: %s", sceneAssetPath.c_str());
        return false;
    }

    return true;
}


//-----------------------------------------------------------------------------
std::unique_ptr<Application::Drawable> Application::InitFullscreenDrawable( const char* pShaderName, const std::map<const std::string, const TextureBase*>& ColorAttachmentsLookup, const std::map<const std::string, const ImageInfo>& ImageAttachmentsLookup, const RenderContext& renderContext, uint32_t subpassIdx, Msaa renderPassMultisamples )
//-----------------------------------------------------------------------------
{
    auto* const pVulkan = GetVulkan();

    LOGI("Creating %s mesh...", pShaderName);

    const auto* pShader = m_ShaderManager->GetShader(pShaderName);
    assert(pShader);
    if( !pShader )
    {
        LOGE("Error, %s shader is unknown (not loaded?)", pShaderName);
        return nullptr;
    }

    Mesh mesh;
    if (!MeshHelper::CreateMesh(GetVulkan()->GetMemoryManager(), MeshObjectIntermediate::CreateScreenSpaceMesh(), 0, pShader->m_shaderDescription->m_vertexFormats, &mesh))
    {
        LOGE("Error creating Fullscreen Mesh (for %s)", pShaderName);
        return nullptr;
    }

    auto shaderMaterial = m_MaterialManager->CreateMaterial(*pShader, NUM_VULKAN_BUFFERS,
        [this, &ColorAttachmentsLookup](const std::string& texName) -> const MaterialManagerBase::tPerFrameTexInfo {
            return { ColorAttachmentsLookup.find(texName)->second };
            assert(0);
            return {};
        },
        []( const std::string& bufferName ) -> const PerFrameBuffer {
            assert(0);
            return {};
        },
        [this, &ImageAttachmentsLookup]( const std::string& imageName ) -> const ImageInfo {
            return { ImageAttachmentsLookup.find(imageName)->second };
        }
    );

    auto drawable = std::make_unique<Drawable>( *pVulkan, std::move( shaderMaterial ) );
    if( !drawable->Init(renderContext, std::move(mesh), std::nullopt, std::nullopt ) )
    {
        LOGE("Error creating Blit Drawable");
        return nullptr;
    }

    return std::move(drawable);
}


//-----------------------------------------------------------------------------
bool Application::InitCommandBuffers()
//-----------------------------------------------------------------------------
{
    auto* const pVulkan = GetVulkan();

    char szName[256];
    for( uint32_t WhichBuffer = 0; WhichBuffer < NUM_VULKAN_BUFFERS; WhichBuffer++ )
    {
        sprintf( szName, "CommandBuffer (%d of %d)", WhichBuffer + 1, NUM_VULKAN_BUFFERS );
        if( !m_CommandBuffer[WhichBuffer].Initialize( pVulkan, szName, CommandList::Type::Primary ) )
        {
            return false;
        }
    }
    return true;
}


//-----------------------------------------------------------------------------
bool Application::InitCamera()
//-----------------------------------------------------------------------------
{
    if (!ApplicationHelperBase::InitCamera())
    {
        return false;
    }
    m_CameraController->SetMoveSpeed(0.001f);

    // Camera
    m_Camera.SetPosition(gCameraStartPos, glm::quat(gCameraStartRot * TO_RADIANS));
    m_Camera.SetFov(gFOV);
    m_Camera.SetClipPlanes(gNearPlane, gFarPlane);

    return true;
}


//-----------------------------------------------------------------------------
void Application::Render(float fltDiffTime)
//-----------------------------------------------------------------------------
{
    UpdateGui();

    UpdateCamera( fltDiffTime );

    auto* const pVulkan = GetVulkan();

    // See if we want to change swapchain/rotate mode
    if( gRotatedFinalBlit != m_RequestedRotatedFinalBlit)
    {
        ChangeSwapchainMode();
    }

    auto CurrentVulkanBuffer = pVulkan->SetNextBackBuffer();

    UpdateUniforms( CurrentVulkanBuffer.idx );

    UpdateCommandBuffer( CurrentVulkanBuffer.idx );

    // ... submit the command buffer to the device queue
    m_CommandBuffer[CurrentVulkanBuffer.idx].QueueSubmit( CurrentVulkanBuffer, pVulkan->m_RenderCompleteSemaphore );
    // and Present
    PresentQueue( pVulkan->m_RenderCompleteSemaphore, CurrentVulkanBuffer.swapchainPresentIdx );
}


//-----------------------------------------------------------------------------
void Application::UpdateCamera(float elapsedTime)
//-----------------------------------------------------------------------------
{
    if (m_CameraController)
        m_Camera.UpdateController(elapsedTime, *m_CameraController);
    m_Camera.UpdateMatrices();
}


//-----------------------------------------------------------------------------
bool Application::UpdateUniforms( uint32_t bufferIdx )
//-----------------------------------------------------------------------------
{
    ObjectVertUniform data{ };
    data.MVPMatrix = m_Camera.ProjectionMatrix() * m_Camera.ViewMatrix();
    data.ModelMatrix = glm::identity<glm::mat4>();
    UpdateUniformBuffer( GetVulkan(), m_ObjectVertUniform, data );

    return true;
}


//-----------------------------------------------------------------------------
bool Application::UpdateCommandBuffer(uint32_t bufferIdx)
//-----------------------------------------------------------------------------
{
    auto* const pVulkan = GetVulkan();

    auto& commandBuffer = m_CommandBuffer[bufferIdx];

    if( !commandBuffer.Begin() )
    {
        return false;
    }

    VkImage swapchainImage = pVulkan->m_SwapchainBuffers[bufferIdx].image;

    {
        // Need to transition the swapchain before it can be blitted to (not needed if written by a render pass)
        VkImageMemoryBarrier presentPreBlitTransitionBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        presentPreBlitTransitionBarrier.srcAccessMask = 0;
        presentPreBlitTransitionBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        presentPreBlitTransitionBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        presentPreBlitTransitionBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        presentPreBlitTransitionBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        presentPreBlitTransitionBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        presentPreBlitTransitionBarrier.image = swapchainImage;
        presentPreBlitTransitionBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        presentPreBlitTransitionBarrier.subresourceRange.baseMipLevel = 0;
        presentPreBlitTransitionBarrier.subresourceRange.levelCount = 1;
        presentPreBlitTransitionBarrier.subresourceRange.baseArrayLayer = 0;
        presentPreBlitTransitionBarrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(commandBuffer.m_VkCommandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &presentPreBlitTransitionBarrier);
    }

    VkRect2D Scissor = {};
    Scissor.offset.x = 0;
    Scissor.offset.y = 0;
    Scissor.extent.width = m_SceneRT.m_Width;
    Scissor.extent.height = m_SceneRT.m_Height;

    VkClearColorValue ClearColor[1] {0.1f, 0.0f, 0.0f, 0.0f};

    if( !commandBuffer.BeginRenderPass( Scissor, 0.0f, 1.0f, ClearColor, m_SceneRT.GetNumColorLayers(), true, m_SceneRenderContext.GetRenderPass(), false, *m_SceneRenderContext.GetFramebuffer(), VK_SUBPASS_CONTENTS_INLINE))
    {
        return false;
    }

    for(const auto& sceneObjectDrawable : m_SceneObject)
        AddDrawableToCmdBuffers(sceneObjectDrawable, &commandBuffer, 1, 1);

    commandBuffer.EndRenderPass();

    //
    // Do the Gui
    //
    if (m_Gui)
    {
        // Render gui (has its own command buffer, optionally returns vk_null_handle if not rendering anything)
        if( !commandBuffer.BeginRenderPass( Scissor, 0.0f, 1.0f, ClearColor, 1, true, m_GuiContext.GetRenderPass(), false, *m_GuiContext.GetFramebuffer(), VK_SUBPASS_CONTENTS_INLINE))
        {
            return false;
        }
        GetGui()->Render(commandBuffer.m_VkCommandBuffer);

        commandBuffer.EndRenderPass();
    }

    //
    // Now add the tonemapping (and add UI overlay)...
    //
    if (!commandBuffer.BeginRenderPass(Scissor, 0.0f, 1.0f, ClearColor, 1, true, m_TonemapRenderContext.GetRenderPass(), false, *m_TonemapRenderContext.GetFramebuffer(), VK_SUBPASS_CONTENTS_INLINE))
    {
        return false;
    }

    AddDrawableToCmdBuffers(*m_TonemapDrawable, &commandBuffer, 1, 1);

    commandBuffer.EndRenderPass();

    if (gRotatedFinalBlit)
    {
        // Rotated blit
        AddRotatedSwapchainBlitToCmdBuffer(commandBuffer, m_TonemapRT, swapchainImage);
    }
    else
    {
        // Non rotated blit
        AddSwapchainBlitToCmdBuffer(commandBuffer, m_TonemapRT, swapchainImage);
    }

    {
        // Need to transition the swapchain before it can be presented
        VkImageMemoryBarrier presentPostBlitTransitionBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        presentPostBlitTransitionBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        presentPostBlitTransitionBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        presentPostBlitTransitionBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        presentPostBlitTransitionBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        presentPostBlitTransitionBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        presentPostBlitTransitionBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        presentPostBlitTransitionBarrier.image = swapchainImage;
        presentPostBlitTransitionBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        presentPostBlitTransitionBarrier.subresourceRange.baseMipLevel = 0;
        presentPostBlitTransitionBarrier.subresourceRange.levelCount = 1;
        presentPostBlitTransitionBarrier.subresourceRange.baseArrayLayer = 0;
        presentPostBlitTransitionBarrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(commandBuffer.m_VkCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &presentPostBlitTransitionBarrier);
    }

    commandBuffer.End();

    return true;
}


//-----------------------------------------------------------------------------
void Application::AddRotatedSwapchainBlitToCmdBuffer(CommandListVulkan& commandBuffer, const RenderTarget& srcRT, VkImage swapchainImage)
//-----------------------------------------------------------------------------
{
    //
    // Now add the final Blit to swapchain backbuffer...
    //

    VkImageBlit2KHR blitImageRegion{ VK_STRUCTURE_TYPE_IMAGE_BLIT_2_KHR };
    blitImageRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitImageRegion.srcSubresource.layerCount = 1;
    blitImageRegion.srcOffsets[0] = { 0,0,0 };
    blitImageRegion.srcOffsets[1] = { (int)srcRT.m_Width, (int)srcRT.m_Height, 1 };
    blitImageRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitImageRegion.dstSubresource.layerCount = 1;
    blitImageRegion.dstOffsets[0] = { 0,0,0 };
    blitImageRegion.dstOffsets[1] = { (int)GetVulkan()->m_SurfaceWidth, (int)GetVulkan()->m_SurfaceHeight, 1 };

    VkBlitImageInfo2KHR blitImageInfo{ VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2_KHR };
    blitImageInfo.srcImage = srcRT.m_ColorAttachments.back().GetVkImage();
    blitImageInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    blitImageInfo.dstImage = swapchainImage;
    blitImageInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    blitImageInfo.regionCount = 1;
    blitImageInfo.pRegions = &blitImageRegion;
    blitImageInfo.filter = VK_FILTER_NEAREST;

    // If the render surface needs a pre-transofrm (rotation) then add the VkCopyCommandTransformInfoQCOM
    // structure on to the end of the VkImageBlit2KHR region to indicate we want to rotate the blit.
    VkCopyCommandTransformInfoQCOM blitRotate{ VK_STRUCTURE_TYPE_COPY_COMMAND_TRANSFORM_INFO_QCOM };

    blitRotate.transform = GetVulkan()->GetPreTransform();
    if (blitRotate.transform != VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
    {
        blitImageRegion.pNext = &blitRotate;
        if (blitRotate.transform == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR || blitRotate.transform == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR)
        {
            // The output surface is rotated - so we need to compensate for that.
            std::swap(blitImageRegion.dstOffsets[1].x, blitImageRegion.dstOffsets[1].y);
        }
    }

    // Add the BlitImage2 command to the command buffer.
    if (vkCmdBlitImage2KHR)
        vkCmdBlitImage2KHR(commandBuffer.m_VkCommandBuffer, &blitImageInfo);
}


//-----------------------------------------------------------------------------
void Application::AddSwapchainBlitToCmdBuffer(CommandListVulkan& commandBuffer, const RenderTarget& srcRT,  VkImage swapchainImage)
//-----------------------------------------------------------------------------
{
    VkImageBlit blitImageRegion{};
    blitImageRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitImageRegion.srcSubresource.layerCount = 1;
    blitImageRegion.srcOffsets[0] = { 0,0,0 };
    blitImageRegion.srcOffsets[1] = { (int)srcRT.m_Width, (int)srcRT.m_Height, 1 };
    blitImageRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitImageRegion.dstSubresource.layerCount = 1;
    blitImageRegion.dstOffsets[0] = { 0,0,0 };
    blitImageRegion.dstOffsets[1] = { (int)GetVulkan()->m_SurfaceWidth, (int)GetVulkan()->m_SurfaceHeight, 1 };
    vkCmdBlitImage(commandBuffer.m_VkCommandBuffer, srcRT.m_ColorAttachments.back().GetVkImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitImageRegion, VK_FILTER_NEAREST);
}


//-----------------------------------------------------------------------------
void Application::ChangeSwapchainMode()
//-----------------------------------------------------------------------------
{
    gRotatedFinalBlit = m_RequestedRotatedFinalBlit;

    GetVulkan()->m_UseRenderPassTransform = false;
    GetVulkan()->m_UsePreTransform = gRotatedFinalBlit;

    GetVulkan()->RecreateSwapChain();

    InitFramebuffersRenderPassesAndDrawables();
}


//-----------------------------------------------------------------------------
bool Application::InitGui(uintptr_t windowHandle)
//-----------------------------------------------------------------------------
{
    auto* const pVulkan = GetVulkan();

    const TextureFormat GuiColorFormat[]{ TextureFormat::R8G8B8A8_UNORM };
    const TEXTURE_TYPE GuiTextureTypes[]{ TT_RENDER_TARGET };

    //
    // Gui render pass
    RenderPass GuiRenderPass;
    if (!pVulkan->CreateRenderPass(GuiColorFormat,
        TextureFormat::UNDEFINED,
        Msaa::Samples1,
        RenderPassInputUsage::Clear,
        RenderPassOutputUsage::Store,
        false,
        RenderPassOutputUsage::Discard,
        GuiRenderPass))
    {
        return false;
    }
    pVulkan->SetDebugObjectName(GuiRenderPass, "GuiRenderPass");

    if (!m_GuiRT.Initialize(pVulkan, gRenderWidth, gRenderHeight, GuiColorFormat, TextureFormat::UNDEFINED, "Gui RT", GuiTextureTypes)
        || !m_GuiRT.InitializeFrameBuffer(pVulkan, GuiRenderPass))
    {
        LOGE("Error initializing GuiRT");
        return false;
    }

    m_GuiContext = RenderContext(std::move(GuiRenderPass), m_GuiRT.GetFrameBuffer(), "Gui RC");

    m_Gui = std::make_unique<GuiImguiGfx>(*pVulkan, m_GuiContext.GetRenderPass().Copy());
    if (!m_Gui->Initialize(windowHandle, m_GuiRT.m_pLayerFormats[0], m_GuiRT.m_Width, m_GuiRT.m_Height))
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
        // Update Gui
        m_Gui->Update();

        // Begin our window.
        static bool settingsOpen = true;
        ImGui::SetNextWindowSize(ImVec2((gRenderWidth * 3.0f) / 4.0f, 500.f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2((float)gRenderWidth / 8.0f, gRenderHeight / 2.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Settings", &settingsOpen, (ImGuiWindowFlags)0))
        {
            // Add our widgets
            if (vkCmdBlitImage2KHR)
            {
                ImGui::Checkbox("Rotated backbuffer Blit", &m_RequestedRotatedFinalBlit);
            }
            else
                ImGui::Text("Rotated blit not available");
        }
        ImGui::End(); 

        if (ImGui::Begin("FPS", (bool*)nullptr, ImGuiWindowFlags_NoTitleBar))
        {
            ImGui::Text("FPS: %.1f", m_CurrentFPS);
        }
        ImGui::End();

        return;
    }
}

//-----------------------------------------------------------------------------
void Application::Destroy()
//-----------------------------------------------------------------------------
{
    auto* const pVulkan = GetVulkan();

    pVulkan->WaitUntilIdle();

    m_SceneObject.clear();
    m_TonemapDrawable.reset();

    for (auto& texture : m_LoadedTextures)
    {
        ReleaseTexture(*pVulkan, &texture.second);
    }

    ReleaseUniformBuffer( pVulkan, &m_ObjectVertUniform );
    ReleaseUniformBuffer( pVulkan, &m_ObjectFragUniform );

    m_GuiRT.Release();
    m_TonemapRT.Release();
    m_SceneRT.Release();

    m_MaterialManager.reset();
    m_ShaderManager.reset();

    m_Gui.reset();

    // Finally call into base class destroy
    ApplicationHelperBase::Destroy();
}
