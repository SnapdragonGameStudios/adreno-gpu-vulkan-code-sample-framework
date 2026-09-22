// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include <optional>
#include <vector>

// Forward declarations
template<typename T_GFXAPI> class VertexBuffer;
template<typename T_GFXAPI> class IndexBuffer;


/// @brief Simple object for creating and holding graphics (renderable) state corresponding to a single mesh.
/// @tparam T_GFXAPI Graphics API class.
template<typename T_GFXAPI>
class Mesh final
{
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
public:
    Mesh() = default;
    Mesh(Mesh&&) noexcept = default;
    Mesh& operator=(Mesh&&) noexcept = default;
    ~Mesh();

    void Destroy();

public:
    size_t                                  m_NumVertices = 0;
    std::vector<VertexBuffer<T_GFXAPI>>     m_VertexBuffers;
    std::optional<IndexBuffer<T_GFXAPI>>    m_IndexBuffer;
};

template<typename T_GFXAPI>
Mesh<T_GFXAPI>::~Mesh() {}

template<typename T_GFXAPI>
void Mesh<T_GFXAPI>::Destroy()
{
    m_NumVertices = 0;
    m_VertexBuffers.clear();
    m_IndexBuffer.reset();
}
