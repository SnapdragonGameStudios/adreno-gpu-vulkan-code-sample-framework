// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2021 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "applicationHelperBaseDx12.hpp"
#include "camera/cameraController.hpp"
#include "camera/cameraControllerTouch.hpp"
#include "material/vertexFormat.hpp"
#include "material/dx12/descriptorSetLayout.hpp"
#include "material/dx12/drawableDx12.hpp"
#include "material/dx12/material.hpp"
#include "material/dx12/materialManager.hpp"
#include "material/dx12/shader.hpp"
#include "material/dx12/shaderModule.hpp"
#include "material/shaderManagerT.hpp"
#include "mesh/meshHelper.hpp"
#include "memory/dx12/indexBufferObject.hpp"
#include "memory/dx12/vertexBufferObject.hpp"
#include "texture/dx12/textureManager.hpp"
#include "dx12/dx12.hpp"
#include "dx12/commandList.hpp"


///////////////////////////////////////////////////////////////////////////////

// Format of vertex_layout
VertexFormat ApplicationHelperBase::vertex_layout::sFormat{
    sizeof(ApplicationHelperBase::vertex_layout),
    VertexFormat::eInputRate::Vertex,
    {
        { offsetof(ApplicationHelperBase::vertex_layout, pos),     VertexElementType::Vec3 },  //float pos[3];         // SHADER_ATTRIB_LOC_POSITION
        { offsetof(ApplicationHelperBase::vertex_layout, normal),  VertexElementType::Vec3 },  //float normal[3];      // SHADER_ATTRIB_LOC_NORMAL
        { offsetof(ApplicationHelperBase::vertex_layout, uv),      VertexElementType::Vec2 },  //float uv[2];          // SHADER_ATTRIB_LOC_TEXCOORD0
        { offsetof(ApplicationHelperBase::vertex_layout, color),   VertexElementType::Vec4 },  //float color[4];       // SHADER_ATTRIB_LOC_COLOR
        { offsetof(ApplicationHelperBase::vertex_layout, tangent), VertexElementType::Vec3 },  //float tangent[3];     // SHADER_ATTRIB_LOC_TANGENT
        //{ offsetof(ApplicationHelperBase::vertex_layout, bitangent), VertexElementType::Vec3 }, // float binormal[3];  // SHADER_ATTRIB_LOC_BITANGENT
     },
     {"Position", "Normal", "UV", "Color", "Tangent" /*,"Bitangent"*/ }
};

//-----------------------------------------------------------------------------
ApplicationHelperBase::ApplicationHelperBase()
//-----------------------------------------------------------------------------
    : FrameworkApplicationBase()
{
    m_gfxBase = std::make_unique<Dx12>();
}

//-----------------------------------------------------------------------------
ApplicationHelperBase::~ApplicationHelperBase()
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
void ApplicationHelperBase::PreInitializeSetVulkanConfiguration( ApplicationHelperBase::AppConfiguration& )
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
bool ApplicationHelperBase::Initialize(uintptr_t hWnd, uintptr_t hInstance)
//-----------------------------------------------------------------------------
{
    Dx12* pDx12 = GetDx12();
    if (!pDx12->Init(hWnd, hInstance))
    {
        LOGE("Unable to initialize DirectX12!!");
        return false;
    }

    auto textureManager = std::make_unique<TextureManagerDx12>(*pDx12, *m_AssetManager);
    if (!textureManager->Initialize())
        return false;
    m_TextureManager = std::move(textureManager);

    m_ShaderManager = std::make_unique<ShaderManager>(*pDx12);

    m_MaterialManager = std::make_unique<MaterialManager>(*pDx12);

    return true;
}

//-----------------------------------------------------------------------------
void ApplicationHelperBase::Destroy()
//-----------------------------------------------------------------------------
{
     m_MaterialManager.reset();
     
     m_ShaderManager.reset();
     
     m_TextureManager.reset();

     FrameworkApplicationBase::Destroy();
}

//-----------------------------------------------------------------------------
void ApplicationHelperBase::KeyDownEvent(uint32_t key)
//-----------------------------------------------------------------------------
{
    if (m_CameraController)
        m_CameraController->KeyDownEvent(key);
}

//-----------------------------------------------------------------------------
void ApplicationHelperBase::KeyRepeatEvent(uint32_t key)
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
void ApplicationHelperBase::KeyUpEvent(uint32_t key)
//-----------------------------------------------------------------------------
{
    if (m_CameraController)
        m_CameraController->KeyUpEvent(key);
}

//-----------------------------------------------------------------------------
void ApplicationHelperBase::TouchDownEvent(int iPointerID, float xPos, float yPos)
//-----------------------------------------------------------------------------
{
    if (m_CameraController)
        m_CameraController->TouchDownEvent(iPointerID, xPos, yPos);
}

//-----------------------------------------------------------------------------
void ApplicationHelperBase::TouchMoveEvent(int iPointerID, float xPos, float yPos)
//-----------------------------------------------------------------------------
{
    if (m_CameraController)
        m_CameraController->TouchMoveEvent(iPointerID, xPos, yPos);
}

//-----------------------------------------------------------------------------
void ApplicationHelperBase::TouchUpEvent(int iPointerID, float xPos, float yPos)
//-----------------------------------------------------------------------------
{
    if (m_CameraController)
        m_CameraController->TouchUpEvent(iPointerID, xPos, yPos);
}

//-----------------------------------------------------------------------------
// Loads a .gltf file and builds a single Vertex Buffer containing relevant vertex.
// DOES not use (or populate) the index buffer although it does respect the gltf index buffer when loading the mesh
// positions, normals and threadColors. Vertices are held in TRIANGLE_LIST format.
//
// Vertex layout corresponds to ApplicationHelperBase::vertex_layout structure.
bool ApplicationHelperBase::LoadGLTF(const std::string& filename, uint32_t binding, Mesh* meshObject)
{
    const auto meshObjects = MeshObjectIntermediate::LoadGLTF(*m_AssetManager, filename);
    if (meshObjects.empty())
        return false;

    return MeshHelper::CreateMesh(GetDx12()->GetMemoryManager(), meshObjects[0].CopyFlattened()/*remove indices*/, binding, { &MeshHelper::vertex_layout::sFormat, 1 }, meshObject);
}

//-----------------------------------------------------------------------------
void ApplicationHelperBase::AddDrawableToCmdBuffers(const Drawable& drawable, CommandList* cmdLists, uint32_t numRenderPasses, uint32_t numFrameBuffers, uint32_t startDescriptorSetIdx) const
//-----------------------------------------------------------------------------
{
    const auto& drawablePasses = drawable.GetDrawablePasses();
    for (const auto& drawablePass : drawablePasses)
    {
        const auto passIdx = drawablePass.mPassIdx;
        assert(passIdx < numRenderPasses);

        for (uint32_t bufferIdx = 0; bufferIdx < numFrameBuffers; ++bufferIdx)
        {
            auto& cmdList = cmdLists[bufferIdx * numRenderPasses + passIdx];

            // Add commands to bind the pipeline, buffers etc and issue the draw.
            drawable.DrawPass(cmdList, drawablePass, /*drawablePass.mDescriptorSet.empty() ? 0 : (startDescriptorSetIdx + bufferIdx) % drawablePass.mDescriptorSet.size()*/ startDescriptorSetIdx + bufferIdx );

            ++cmdList.m_NumDrawCalls;
            cmdList.m_NumTriangles += drawablePass.mNumVertices / 3;
        }
    }
}

