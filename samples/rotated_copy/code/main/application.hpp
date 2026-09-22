// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2020 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

///
/// @file rotatedCopy.hpp
/// @brief Sample app for VK_QCOM_rotated_copy_commands extension.
/// 
/// Very simple application that uses VK_QCOM_rotated_copy_commands to blit the final image to the present framebuffer with pre-rotation.
/// 

#include "main/applicationHelperBase.hpp"
#include "vulkan/renderTarget.hpp"
#include "memory/vulkan/uniform.hpp"
#include "vulkan/commandBuffer.hpp"
#include "VK_QCOM_rotated_copy_commands.h"
#include <map>
#include <string>

// Forward declarations
class ShaderManagerBase;
class MaterialManagerBase;


class Application : public ApplicationHelperBase
{
public:
    Application();
    ~Application() override;

    //
    // Override ApplicationHelperBase
    //
    /// Override Vulkan application configuration
    void PreInitializeSetVulkanConfiguration(Vulkan::AppConfiguration&) override;
    /// Override surface format selection.
    int PreInitializeSelectSurfaceFormat(std::span<const SurfaceFormat> formats) override;
    /// Override Application entry point!
    bool Initialize(uintptr_t windowHandle, uintptr_t hInstance) override;

    bool LoadShaders();

    bool InitFramebuffersRenderPassesAndDrawables();
    bool LoadSceneDrawables( const RenderContext& renderContext, Msaa renderPassMultisamples );
    std::unique_ptr<Drawable> InitFullscreenDrawable( const char* pShaderName, const std::map<const std::string, const TextureBase*>& inputLookup, const std::map<const std::string, const ImageInfo>& ImageAttachmentsLookup, const RenderContext& renderContext, uint32_t subpassIdx, Msaa renderPassMultisamples );
    bool InitCommandBuffers();
    bool InitCamera() override;

    /// @brief Ticked every frame (by the Framework)
    /// @param fltDiffTime time (in seconds) since the last call to Render.
    void Render( float fltDiffTime ) override;

    void UpdateCamera( float elapsedTime );
    bool UpdateUniforms( uint32_t bufferIdx );
    bool UpdateCommandBuffer( uint32_t bufferIdx );

    //
    // Blit helpers
    //

    /// Adds a rotated blit command to the commandBuffer.
    /// Blits from the m_TonemapRT to the given swapchain image.
    void AddRotatedSwapchainBlitToCmdBuffer(CommandListVulkan& commandBuffer, const RenderTarget& srcRT, VkImage swapchainImage);

    /// Adds a (non rotated) blit command to the commandBuffer.
    /// Blits from the m_TonemapRT to the given swapchain image.
    void AddSwapchainBlitToCmdBuffer(CommandListVulkan& commandBuffer, const RenderTarget& srcRT, VkImage swapchainImage);


    void ChangeSwapchainMode();

    bool InitGui( uintptr_t windowHandle );
    void UpdateGui();

    /// Shutdown function
    void Destroy() override;

private:
    // Textures
    std::map<std::string, TextureVulkan>    m_LoadedTextures;

    // Materials
    
    // Drawable(s) for rendering the main scene object
    std::vector<Drawable>                   m_SceneObject;

    // Drawable for tonemapping
    std::unique_ptr<Drawable>               m_TonemapDrawable;

    // Uniform buffers
    struct ObjectVertUniform
    {
        glm::mat4 MVPMatrix;
        glm::mat4 ModelMatrix;
    };
    UniformT<ObjectVertUniform>             m_ObjectVertUniform;
    struct ObjectfragUniform
    {
        float unused;
    };
    UniformT<ObjectfragUniform>             m_ObjectFragUniform;

    /// Render target for 3d Objects.
    RenderTarget                            m_SceneRT;
    RenderTarget                            m_TonemapRT;
    RenderTarget                            m_GuiRT;
    RenderContext                           m_SceneRenderContext;
    RenderContext                           m_TonemapRenderContext;
    RenderContext                           m_GuiContext;

    /// Single primary command buffer
    CommandListVulkan                       m_CommandBuffer[NUM_VULKAN_BUFFERS];

    /// Modified by UI to request enable/disable of the final rotated blit
    bool                                    m_RequestedRotatedFinalBlit = false;
};
