// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2021 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include "dx12/dx12.hpp"
#include "frameworkApplicationBase.hpp"
#include "camera/camera.hpp"
#include "texture/dx12/sampler.hpp"
#include "texture/dx12/texture.hpp"

class CameraController;
class CameraControllerTouch;
class ComputableBase;
class MaterialManagerBase;
class ShaderManagerBase;
class TextureManagerBase;
class VertexFormat;
template<typename T_GFXAPI> class CommandList;
template<typename T_GFXAPI> class Computable;
template<typename T_GFXAPI> class Drawable;
template<typename T_GFXAPI> class DrawableLoader;
template<typename T_GFXAPI> class DrawIndirectBuffer;
template<typename T_GFXAPI> class GuiImguiGfx;
template<typename T_GFXAPI> class Material;
template<typename T_GFXAPI> class MaterialManager;
template<typename T_GFXAPI> class Mesh;
template<typename T_GFXAPI> struct PerFrameBuffer;
template<typename T_GFXAPI> class RenderPass;
template<typename T_GFXAPI> class RenderTarget;
template<typename T_GFXAPI> class ShaderManager;
template<typename T_GFXAPI> class TextureManager;
template<typename T_GFXAPI> struct Uniform;
template<typename T_GFXAPI, typename T> struct UniformT;
template<typename T_GFXAPI, size_t T_NUM_BUFFERS> struct UniformArray;
template<typename T_GFXAPI, typename T, size_t T_NUM_BUFFERS> struct UniformArrayT;

using TimerPoolBase = void; ///TODO: implement Dx12 timers

/// Helper class that applications can be derived from.
/// Provides Camera and Drawable functionality to reduce code duplication of more 'boilderplate' application features.
class ApplicationHelperBase : public FrameworkApplicationBase
{
protected:
    ApplicationHelperBase();
    ~ApplicationHelperBase() override;

    // This class is for Dx12 applications
    using tGfxApi = Dx12;
    using CommandList = CommandList<tGfxApi>;
    using CommandBuffer = CommandList;
    using Computable = Computable<tGfxApi>;
    using Drawable = Drawable<tGfxApi>;
    using DrawableLoader = DrawableLoader<tGfxApi>;
    using DrawIndirectBuffer = DrawIndirectBuffer<tGfxApi>;
    using Material = Material<tGfxApi>;
    using MaterialManager = MaterialManager<tGfxApi>;
    using Mesh = Mesh<tGfxApi>;
    using PerFrameBuffer = PerFrameBuffer<tGfxApi>;
    using RenderPass = RenderPass<tGfxApi>;
    using RenderTarget = RenderTarget<tGfxApi>;
    using ShaderManager = ShaderManager<tGfxApi>;
    using Uniform = Uniform<tGfxApi>;
    template<size_t T_NUM_BUFFERS> using UniformArray = UniformArray<tGfxApi, T_NUM_BUFFERS>;
    template<typename T> using UniformT = UniformT<tGfxApi, T>;
    template<typename T, size_t T_NUM_BUFFERS> using UniformArrayT = UniformArrayT<tGfxApi, T, T_NUM_BUFFERS>;
    using GuiImguiGfx = GuiImguiGfx<tGfxApi>;

    struct AppConfiguration
    {
        tGfxApi::AppConfiguration gfx;
    };

    virtual void PreInitializeSetVulkanConfiguration( AppConfiguration& );

    bool    Initialize(uintptr_t hWnd, uintptr_t hInstance) override;           ///< Override FrameworkApplicationBase::Initialize
    void    Destroy() override;                                                 ///< Override FrameworkApplicationBase::Destroy

    void    KeyDownEvent(uint32_t key) override;                                ///< Override FrameworkApplicationBase::KeyDownEvent
    void    KeyRepeatEvent(uint32_t key) override;                              ///< Override FrameworkApplicationBase::KeyRepeatEvent
    void    KeyUpEvent(uint32_t key) override;                                  ///< Override FrameworkApplicationBase::KeyUpEvent
    void    TouchDownEvent(int iPointerID, float xPos, float yPos) override;    ///< Override FrameworkApplicationBase::TouchDownEvent
    void    TouchMoveEvent(int iPointerID, float xPos, float yPos) override;    ///< Override FrameworkApplicationBase::TouchMoveEvent
    void    TouchUpEvent(int iPointerID, float xPos, float yPos) override;      ///< Override FrameworkApplicationBase::TouchUpEvent

    /// @brief Get pointer to the framework Dx12 class
    Dx12* GetDx12() const { return static_cast<tGfxApi*>(m_gfxBase.get()); }
    /// @brief Get pointer to the framework graphicsApi Dx12 class (will get the Vulkan class when called from the Vulkan ApplicationHelperBase).  Can use FrameworkApplicationBase::GetGraphicsApiBase if you want the base class only!
    Dx12* GetGfxApi() const { return static_cast<tGfxApi*>(m_gfxBase.get()); }
    /// @brief Get pointer to the framework gui (Dx12) class
    GuiImguiGfx* GetGui() const { return (GuiImguiGfx*)(m_Gui.get()); }

    // These MUST match the order of the attrib locations and sFormat must reflect the layout of this struct too!
    struct vertex_layout
    {
        float pos[3];           // SHADER_ATTRIB_LOC_POSITION
        float normal[3];        // SHADER_ATTRIB_LOC_NORMAL
        float uv[2];            // SHADER_ATTRIB_LOC_TEXCOORD0
        float color[4];         // SHADER_ATTRIB_LOC_COLOR
        float tangent[3];       // SHADER_ATTRIB_LOC_TANGENT
        // float binormal[3];      // SHADER_ATTRIB_LOC_BITANGENT
        static VertexFormat sFormat;
    };

    /// Mesh loader helper to load the first shape in a .gltf file (no materials).
    /// Returned MeshObject does not have an index buffer (3 verts per triangle) and data is in the ApplicationHelperBase::vertex_layout format.
    /// @returns true on success
    bool    LoadGLTF(const std::string& filename, uint32_t binding, Mesh* meshObject);

    /// @brief Add the commands to draw this drawable to the given commandbuffers.
    /// @param drawable the Drawable object we want to add commands for (may contain multiple DrawablePass)
    /// @param cmdBuffers pointer to array, assumed to be sized [numVulkanBuffers][numRenderPasses]
    /// @param numRenderPasses number of render passes in the array (the DrawablePass has the pass index that is written to) 
    /// @param numVulkanBuffers number of buffers in the array (number of frames to build the command buffers for)
    void    AddDrawableToCmdBuffers(const Drawable& drawable, CommandList* cmdBuffers, uint32_t numRenderPasses, uint32_t numFrameBuffers, uint32_t startDescriptorSetIdx = 0) const;
    void    AddDrawableToCmdBuffer(const Drawable& drawable, CommandList& cmdBuffer, uint32_t startDescriptorSetIdx = 0) const
    {
        AddDrawableToCmdBuffers(drawable, &cmdBuffer, 1, 1, startDescriptorSetIdx);
    }
    /// @brief Add the commands to dispatch this computable to a command buffer.
    /// Potentially adds multiple vkCmdDispatch (will make one per 'pass' in the Computable) and inserts appropiate memory barriers between passes.
    /// @param computable
    /// @param cmdBuffers pointer to array of commandbuffers we want to fill, assumed to be sized [numRenderPasses]
    /// @param numRenderPasses number of cmdBuffers to fill
    /// @param startDescriptorSetIdx index of the first descriptor set to add
    void    AddComputableToCmdBuffer(const Computable& computable, CommandList* cmdBuffers, uint32_t numRenderPasses, uint32_t startDescriptorSetIdx, TimerPoolBase* timerPool = nullptr ) const;
    void    AddComputableToCmdBuffer( const Computable& computable, CommandList& cmdBuffer, TimerPoolBase* timerPool = nullptr ) const
    {
        AddComputableToCmdBuffer( computable, &cmdBuffer, 1, 0, timerPool );
    }


protected:
    // Scene Camera
    Camera                                  m_Camera;

    // Camera Controller
#if defined(OS_ANDROID) ///TODO: make this an option
    std::unique_ptr<CameraControllerTouch>  m_CameraController;
#else
    std::unique_ptr<CameraController>       m_CameraController;
#endif

    // Shaders
    std::unique_ptr<ShaderManager>         m_ShaderManager;

    // Materials
    std::unique_ptr<MaterialManager>       m_MaterialManager;

    // Texture manager
    std::unique_ptr<TextureManager<tGfxApi>> m_TextureManager;

    // Ready to use samplers
    // Default samplers
    Sampler<tGfxApi>                       m_SamplerRepeat;
    Sampler<tGfxApi>                       m_SamplerEdgeClamp;
    Sampler<tGfxApi>                       m_SamplerMirroredRepeat;
};
