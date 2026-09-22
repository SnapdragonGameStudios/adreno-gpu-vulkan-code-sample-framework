// SPDX-License-Identifier: BSD-3-Clause

//============================================================================================================
//
//                  Copyright (c) 2020 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//============================================================================================================
#pragma once

#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>
#include "system/glm_common.hpp"
//#include "memory/indexBufferObject.hpp"
//#include "memory/memoryManager.hpp"

// Forward declarations
class VertexFormat;
class AssetManager;
class VertexBufferObject;
class MeshObjectIntermediate;

/// Defines a simple object for creating and holding DX12 state corresponding to a single mesh.
template<typedef T_GFXAPI>
class MeshObject
{
    MeshObject(const MeshObject&) = delete;
    MeshObject& operator=(const MeshObject&) = delete;
public:
    MeshObject() = default;
    MeshObject(MeshObject&&) noexcept = default;
    MeshObject& operator=(MeshObject&&) noexcept = default;
    ~MeshObject();

    /// Builds a screen space mesh as a single Vertex Buffer containing relevant vertex positions, normals and colors. Vertices are held in TRIANGLE_LIST format.
    /// Vertex layout corresponds to MeshObject::vertex_layout structure.
    /// @returns true on success
    static bool CreateScreenSpaceMesh(MemoryManager& memoryManager, glm::vec4 PosLLRadius, glm::vec4 UVLLRadius, uint32_t binding, MeshObject* meshObject);
    /// Helper for CreateScreenSpaceMesh that has a x,y position from -1,-1 to 1,1 and UV from 0,0 to 1,1.
    /// @returns true on success
    static bool CreateScreenSpaceMesh(MemoryManager& memoryManager, uint32_t binding, MeshObject* meshObject)
    {
        return CreateScreenSpaceMesh(memoryManager, glm::vec4(-1.0f, -1.0f, 2.0f, 2.0f), glm::vec4(0.0f, 0.0f, 1.0f, 1.0f), binding, meshObject);
    }

    /// Create a MeshObject from a 'fat' MeshObjectIntermediate object, rearranging the vertex data to match the supplied vertex format(s).
    /// Can have multiple VertexFormats, which will create multiple vertex buffers (eg if we want to split vertex position data away from other vertex attributes)
    /// @param pVertexFormat format of the vertex data being output
    /// @returns true on success
    static bool CreateMesh(MemoryManager& memoryManager, const MeshObjectIntermediate& meshObject, uint32_t binding, const std::span<const VertexFormat> pVertexFormat, MeshObject* meshObjectOut);

    virtual bool Destroy();

    /// Helper to create a IndexBufferObject, IF the mesh object has index buffer data.
    /// @returns true for success (including no index buffer data existing in IndexBufferObject), false on error.
    static bool CreateIndexBuffer(MemoryManager& memoryManager, const MeshObjectIntermediate& meshObject, std::optional<IndexBufferObject>& indexBufferOut, VkBufferUsageFlags usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

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

public:
    uint32_t                        m_NumVertices;
    std::vector<VertexBufferObject> m_VertexBuffers;
    std::optional<IndexBufferObject>m_IndexBuffer;
};


namespace MeshObjectHelper
{
    /// Create a MeshObject for the first shape in a .gltf file (no materials).
    /// Returned MeshObject does not have an index buffer (3 verts per triangle) and data is in the MeshObject::vertex_layout format.
    /// @returns true on success
    static bool LoadGLTF(MemoryManager& memoryManager, AssetManager&, const std::string& filename, uint32_t binding, MeshObject<Dx12>* meshObject);

} // namespace MeshObjectHelper;


template<typedef T_GFXAPI>
MeshObject<T_GFXAPI>::~MeshObject() {}
