// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include <map>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <optional>
#include "../drawable.hpp"
#include "dx12/dx12.hpp"
#include "mesh/mesh.hpp"
#include "material.hpp"
#include "pipeline.hpp"
#include "dx12/renderPass.hpp"
#include "material/shaderDescription.hpp"
#include "memory/dx12/indexBufferObject.hpp"
#include "memory/dx12/vertexBufferObject.hpp"
#include "memory/drawIndirectBuffer.hpp"
#include "mesh/instanceGenerator.hpp"
#include "mesh/meshHelper.hpp"
#include "mesh/meshIntermediate.hpp"
#include "system/os_common.h"
//#include "pipelineVertexInputState.hpp"

// Forward Declarations
class Dx12;
class MaterialBase;
struct MeshInstance;
enum class Msaa;
template<typename tGfxApi> class Material;
template<typename tGfxApi> class Shader;

template<>
struct DrawablePassVertexBuffers<Dx12> : public DrawablePassVertexBuffersBase
{
    std::vector<D3D12_VERTEX_BUFFER_VIEW> views;
};


/// Encapsulates a drawable pass.
/// References vertex/index buffers etc from the parent Drawable.
/// Users are expected to use Drawable (which contains a vector of DrawablePasses and is more 'user friendly').
/// @ingroup Material
template<>
class DrawablePass<Dx12> : public DrawablePassBase
{
    DrawablePass(const DrawablePass&) = delete;
    DrawablePass& operator=(const DrawablePass&) = delete;
    using tGfxApi = Dx12;
public:
    DrawablePass(DrawablePass&&) noexcept = default;
    DrawablePass() = delete;
    ~DrawablePass();
    DrawablePass(const MaterialPass<tGfxApi>&      MaterialPass,
                Pipeline<tGfxApi>                   Pipeline,
                //VkPipelineLayout                  PipelineLayout,
                 std::vector<RootItem>              RootDescriptorTableItems,
                 std::vector<DescriptorTableHandle> DescriptorTables,
                //const PipelineVertexInputState<tGfxApi>& PipelineVertexInputState,
                DrawablePassVertexBuffers<tGfxApi>  VertexBuffers,
                D3D12_INDEX_BUFFER_VIEW             IndexBuffer,
                //VkIndexType                       IndexBufferType,
                //VkBuffer                          DrawIndirectBuffer,
                //VkBuffer                          DrawIndirectCountBuffer,
                uint32_t                            NumVertices,
                uint32_t                            NumIndices,
//                uint32_t                          NumDrawIndirect,
//                uint32_t                          DrawIndirectOffset,
                uint32_t                            PassIdx
                 ) noexcept
        : mMaterialPass( MaterialPass )
        , mPipeline( std::move(Pipeline) )
        //, mPipelineLayout( PipelineLayout )*/
        , mRootItems( std::move(RootDescriptorTableItems) )
        , mDescriptorTables( std::move(DescriptorTables) )
        /*, mPipelineVertexInputState( PipelineVertexInputState )*/
        , mVertexBuffers( std::move(VertexBuffers) )
        , mIndexBuffer( IndexBuffer )
        /*, mIndexBufferType( IndexBufferType ), mDrawIndirectBuffer( DrawIndirectBuffer ), mDrawIndirectCountBuffer( DrawIndirectCountBuffer )*/
        , mNumVertices( NumVertices )
        , mNumIndices( NumIndices)
        /*mNumDrawIndirect( NumDrawIndirect), mDrawIndirectOffset( DrawIndirectOffset),*/
        , mPassIdx( PassIdx)
    {
    }

    const MaterialPass<tGfxApi>&        mMaterialPass;
    Pipeline<tGfxApi>                    mPipeline;
    //VkPipeline                      mPipeline = VK_NULL_HANDLE; // Owned by us
    //VkPipelineLayout                mPipelineLayout;            // Owned by shader
    std::vector<RootItem>             mRootItems;
    std::vector<DescriptorTableHandle> mDescriptorTables;         // 
    //const PipelineVertexInputState<tGfxApi>& mPipelineVertexInputState;  // contains vertex binding and attribute descriptions
    //DrawablePassVertexBuffers       mVertexBuffers;             // contains vkbuffer/offsets; one per mVertexBuffersLookup entry.  May be vertex rate or instance rate.
    DrawablePassVertexBuffers<tGfxApi> mVertexBuffers;         // one per mVertexBuffersLookup entry.  May be vertex rate or instance rate
    D3D12_INDEX_BUFFER_VIEW         mIndexBuffer = {0,0,DXGI_FORMAT_UNKNOWN};
    //VkIndexType                     mIndexBufferType = VK_INDEX_TYPE_MAX_ENUM;
    //VkBuffer                        mDrawIndirectBuffer = VK_NULL_HANDLE;
    //VkBuffer                        mDrawIndirectCountBuffer = VK_NULL_HANDLE;
    uint32_t                        mNumVertices;
    uint32_t                        mNumIndices;
    //uint32_t                        mNumDrawIndirect;
    //uint32_t                        mDrawIndirectOffset;        // if non zero offset mDrawIndirectBuffer by this
    uint32_t                        mPassIdx;                   // index of the bit in Drawable::m_passMask
};
