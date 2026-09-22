// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include "mesh.hpp"
#include "meshIntermediate.hpp"
#include "material/vertexFormat.hpp"
#include "memory/memory.hpp"
#include "system/os_common.h"
#include <span>

// Forward declarations
class MeshObjectIntermediate;
template<typename T_GFXAPI> class MemoryManager;


struct MeshHelper
{
    // vertex_layout (can be used in cases where we dont want to define our own layout)
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

    /// @brief Templated function to create a renderable Mesh from an intermediate (cpu) representation
    /// @tparam T_GFXAPI Graphics API class.
    /// @param memoryManager 
    /// @param meshObject 
    /// @param bindingIndex 
    /// @param pVertexFormat 
    /// @param meshObjectOut 
    /// @return true on success
    template<typename T_GFXAPI>
    static bool CreateMesh(MemoryManager<T_GFXAPI>& memoryManager, const MeshObjectIntermediate& meshObject, uint32_t bindingIndex, const std::span<const VertexFormat> pVertexFormat, Mesh<T_GFXAPI>* meshObjectOut);

    /// Helper to create a IndexBuffer object, IF the mesh object has index buffer data.
    /// @returns true on success (including no index buffer data existing in IndexBuffer object), false on error.
    template<typename T_GFXAPI>
    static bool CreateIndexBuffer(MemoryManager<T_GFXAPI>& memoryManager, const MeshObjectIntermediate& meshObject, std::optional<class IndexBuffer<T_GFXAPI>>& indexBufferOut, BufferUsageFlags usage = BufferUsageFlags::Index);

    /// @brief Helper to create a screen-space 2d mesh that fills the screen (assuming -1.0 to 1.0 screen coordinates, ie OpenGl style).  Will make a mesh with 2 triangles!
    /// @return true on success
    template<typename T_GFXAPI>
    static bool CreateScreenSpaceMesh(MemoryManager<T_GFXAPI>& memoryManager, uint32_t binding, Mesh<T_GFXAPI>* meshObjectOut);

    /// @brief Helper to create a screen-space 2d mesh with the given size.  Will make a mesh with 2 triangles!
    /// @return true on success
    template<typename T_GFXAPI>
    static bool CreateScreenSpaceMesh(MemoryManager<T_GFXAPI>& memoryManager, glm::vec4 PosLLRadius, glm::vec4 UVLLRadius, uint32_t binding, Mesh<T_GFXAPI>* meshObjectOut);

    /// @brief Templated function to update a renderable Mesh 
    /// @tparam T_GFXAPI Graphics API class.
    /// @param memoryManager 
    /// @param meshObject
    /// @param whichBuffer 
    /// @param dataSize 
    /// @param newData 
    /// @return true on success
    template<typename T_GFXAPI>
    static bool UpdateMesh(MemoryManager<T_GFXAPI>& memoryManager, Mesh<T_GFXAPI>& meshObject, uint32_t whichBuffer, size_t dataSize, void *newData);

    /// @brief Templated function to update a renderable Mesh 
    /// @tparam T_GFXAPI Graphics API class.
    /// @param memoryManager 
    /// @param meshObject
    /// @param whichBuffer 
    /// @param dataSize 
    /// @param outputData 
    /// @return true on success
    template<typename T_GFXAPI>
    static bool GetMeshData(MemoryManager<T_GFXAPI>& memoryManager, const Mesh<T_GFXAPI>& meshObject, uint32_t whichBuffer, size_t dataSize, void *outputData);

};

///////////////////////////////////////////////////////////////////////////////

template<typename T_GFXAPI>
bool MeshHelper::CreateMesh(MemoryManager<T_GFXAPI>& memoryManager, const MeshObjectIntermediate& meshObject, uint32_t bindingIndex, const std::span<const VertexFormat> pVertexFormat, Mesh<T_GFXAPI>* meshObjectOut)
{
    assert(meshObjectOut);
    meshObjectOut->Destroy();

    const size_t numVertices = meshObject.m_VertexBuffer.size();
    // It is valid to have no vertex buffers (empty pVertexFormat) but still render vertices... vertex shader could generate verts procedurally.
    meshObjectOut->m_NumVertices = (uint32_t)numVertices;

    //
    // We can have more than one buffer in a 'mesh'.  Use this to do things like splitting the position away from uv/color/normal etc so the shadow passes use less vertex bandwidth.
    // We also want to skip any Instance bindings (not handled by the Mesh).
    // Beware of putting instance (rate) bindings in the 'middle' of vertex rate buffers as the output MeshObject::m_VertexBuffers cannot be all bound in one chunk in that case.
    //
    for (uint32_t vertexBufferIdx = 0; vertexBufferIdx < pVertexFormat.size(); ++vertexBufferIdx)
    {
        // Convert from the 'fat' vertex data to the required destination format for this vertex buffer.
        const auto& vertexFormat = pVertexFormat[vertexBufferIdx];
        if (vertexFormat.inputRate == VertexFormat::eInputRate::Vertex)
        {
            const std::vector<uint32_t> formattedVertexData = MeshObjectIntermediate::CopyFatVertexToFormattedBuffer(meshObject.m_VertexBuffer, meshObject.m_WeightBuffer, pVertexFormat[vertexBufferIdx]);

            if (!meshObjectOut->m_VertexBuffers.emplace_back().Initialize(&memoryManager, vertexFormat.span, numVertices, formattedVertexData.data()))
            {
                LOGE("Cannot Initialize vertex buffer %d", vertexBufferIdx);
                return false;
            }
            meshObjectOut->m_VertexBuffers.rbegin()->AddBindingAndAtributes(bindingIndex + vertexBufferIdx, vertexFormat);
        }
    }

    // Create the index buffer (with an appropriate VkIndexType index type, IF the meshObject has index data).
    if (!CreateIndexBuffer<T_GFXAPI>(memoryManager, meshObject, meshObjectOut->m_IndexBuffer))
    {
        return false;
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////

template<typename T_GFXAPI>
bool MeshHelper::UpdateMesh(MemoryManager<T_GFXAPI>& memoryManager, Mesh<T_GFXAPI>& meshObject, uint32_t whichBuffer, size_t dataSize, void* newData)
{
    // Make sure the vertex buffer is in range
    if (whichBuffer >= meshObject.m_VertexBuffers.size())
    {
        LOGE("MeshHelper::UpdateMesh: Error! Buffer index (%d) out of range (%d vertex buffers)", whichBuffer, (int)meshObject.m_VertexBuffers.size());
        return false;
    }

    meshObject.m_VertexBuffers[0].Update(&memoryManager, dataSize, newData);

    return true;
}

///////////////////////////////////////////////////////////////////////////////

template<typename T_GFXAPI>
bool MeshHelper::GetMeshData(MemoryManager<T_GFXAPI>& memoryManager, const Mesh<T_GFXAPI>& meshObject, uint32_t whichBuffer, size_t dataSize, void* outputData)
{
    // Make sure the vertex buffer is in range
    if (whichBuffer >= meshObject.m_VertexBuffers.size())
    {
        LOGE("MeshHelper::GetMeshData: Error! Buffer index (%d) out of range (%d vertex buffers)", whichBuffer, (int)meshObject.m_VertexBuffers.size());
        return false;
    }

    meshObject.m_VertexBuffers[0].GetMeshData(&memoryManager, dataSize, outputData);

    return true;
}

///////////////////////////////////////////////////////////////////////////////

template<typename T_GFXAPI>
bool MeshHelper::CreateIndexBuffer(MemoryManager<T_GFXAPI>& memoryManager, const MeshObjectIntermediate& meshObject, std::optional<class IndexBuffer<T_GFXAPI>>& indexBufferOut, BufferUsageFlags usage)
{
    //
    // If the source has an index buffer then copy the data in to a Vulkan buffer
    //
    if (!std::visit(
        [&meshObject, &indexBufferOut, &memoryManager, usage](auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
                auto& indexBuffer = indexBufferOut.emplace(IndexType::IndexU32);
                const auto& vec32 = std::get<std::vector<uint32_t>>(meshObject.m_IndexBuffer);
                if (!indexBuffer.Initialize(&memoryManager, std::span(vec32), false, usage))
                {
                    return false;
                }
            }
            else if constexpr (std::is_same_v<T, std::vector<uint16_t>>) {
                auto& indexBuffer = indexBufferOut.emplace(IndexType::IndexU16);
                const auto& vec16 = std::get<std::vector<uint16_t>>(meshObject.m_IndexBuffer);
                if (!indexBuffer.Initialize(&memoryManager, std::span(vec16), false, usage))
                {
                    return false;
                }
            }
            return true;
        },
        meshObject.m_IndexBuffer))
    {
        LOGE("Cannot Initialize index buffer");
        indexBufferOut.reset();
        return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////

template<typename T_GFXAPI>
bool MeshHelper::CreateScreenSpaceMesh( MemoryManager<T_GFXAPI>& memoryManager, uint32_t binding, Mesh<T_GFXAPI>* meshObjectOut )
{
    return MeshHelper::CreateMesh( memoryManager, MeshObjectIntermediate::CreateScreenSpaceMesh(), binding, { &vertex_layout::sFormat,1 }, meshObjectOut );
}

///////////////////////////////////////////////////////////////////////////////

template<typename T_GFXAPI>
bool MeshHelper::CreateScreenSpaceMesh( MemoryManager<T_GFXAPI>& memoryManager, glm::vec4 PosLLRadius, glm::vec4 UVLLRadius, uint32_t binding, Mesh<T_GFXAPI>* meshObjectOut )
{
    return MeshHelper::CreateMesh( memoryManager, MeshObjectIntermediate::CreateScreenSpaceMesh(PosLLRadius, UVLLRadius), binding, { &vertex_layout::sFormat,1 }, meshObjectOut );
}
