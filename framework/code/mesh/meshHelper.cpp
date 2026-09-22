// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "meshHelper.hpp"
#include "material/vertexFormat.hpp"

// Format of vertex_layout (can be used in cases where we dont want to define our own layout)
VertexFormat MeshHelper::vertex_layout::sFormat{
    sizeof(MeshHelper::vertex_layout),
    VertexFormat::eInputRate::Vertex,
    {
        { offsetof(MeshHelper::vertex_layout, pos),     VertexElementType::Vec3 },  //float pos[3];
        { offsetof(MeshHelper::vertex_layout, normal),  VertexElementType::Vec3 },  //float normal[3];
        { offsetof(MeshHelper::vertex_layout, uv),      VertexElementType::Vec2 },  //float uv[2];
        { offsetof(MeshHelper::vertex_layout, color),   VertexElementType::Vec4 },  //float color[4];
        { offsetof(MeshHelper::vertex_layout, tangent), VertexElementType::Vec3 },  //float tangent[3];
        //{ offsetof(MeshHelper::vertex_layout, bitangent), VertexElementType::Vec3 }, // float binormal[3];
     },
     {"POSITION", "NORMAL", "TEXCOORD", "COLOR", "TANGENT" /*,"BINORMAL"*/ }
};

