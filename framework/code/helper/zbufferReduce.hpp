// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2024 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#pragma once

#include <memory>
#include <vector>

// Forward declarations
class Vulkan;
template<typename T_GFXAPI> class CommandList;
template<typename T_GFXAPI> class Computable;
template<typename T_GFXAPI> class MaterialManager;
template<typename T_GFXAPI> class MemoryPool;
template<typename T_GFXAPI> class Shader;
template<typename T_GFXAPI> class Texture;

/// @brief Class to implement a fast zbuffer 'reduction' (pyramid/mipmap of depth images, each half size of previous level)
class ZBufferReduce
{
    ZBufferReduce( const ZBufferReduce& ) = delete;
    ZBufferReduce& operator=( const ZBufferReduce& ) = delete;
public:
    ZBufferReduce() = default;
    bool Init( Vulkan* pVulkan, const Texture<Vulkan>& srcDepth, const MaterialManager<Vulkan>& materialManager, const Shader<Vulkan>* const pReduce4x4Shader, MemoryPool<Vulkan>* pPool = nullptr);
    void Release( Vulkan* pVulkan );
    void UpdateCommandBuffer( CommandList<Vulkan>& );
    auto& GetHierarchicalZTexture() const { return *m_ReducedZBuffer; }

protected:
    std::unique_ptr<Texture<Vulkan>>        m_ReducedZBuffer;           ///< half sized 'depth' buffer (half in each dimension of original ZBuffer size) with mips.  Populated by the reduce.
    std::vector<Texture<Vulkan>>            m_ReducedZBufferMipTexInfos;///< Texture<Vulkan> pointing to each of the mip levels in m_ReducedZBuffer.  All points to Image owned by m_ReducedZBuffer
    std::unique_ptr<Computable<Vulkan>>     m_ReduceComputable;         ///< Computable that does the reduce (some passes may not be needed depending on mip count needed)
    int                                     m_NumMips = 0;
};
