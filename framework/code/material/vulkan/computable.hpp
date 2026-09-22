// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include "material.hpp"
#include "pipeline.hpp"
#include "../computable.hpp"


template<>
class ImageMemoryBarrier<Vulkan> final : public ImageMemoryBarrierBase
{
public:
    typedef VkImageMemoryBarrier tBarrier;
    ImageMemoryBarrier( VkImageMemoryBarrier&& barrier ) noexcept
        : ImageMemoryBarrierBase()
        , vkBarrier( std::move( barrier ) )
    {}
    VkImageMemoryBarrier vkBarrier;
};
static_assert(sizeof( ImageMemoryBarrier<Vulkan> ) == sizeof( VkImageMemoryBarrier ));

template<>
class BufferMemoryBarrier<Vulkan> final : public BufferMemoryBarrierBase
{
public:
    typedef VkBufferMemoryBarrier tBarrier;
    BufferMemoryBarrier( VkBufferMemoryBarrier&& barrier ) noexcept
        : BufferMemoryBarrierBase()
        , vkBarrier( std::move( barrier ) )
    {}
    tBarrier vkBarrier;
};
static_assert(sizeof( BufferMemoryBarrier<Vulkan> ) == sizeof( VkBufferMemoryBarrier ));


template<>
class ComputablePass<Vulkan> final : public ComputablePassBase
{
    ComputablePass( const ComputablePass<Vulkan>& ) = delete;
    ComputablePass& operator=( const ComputablePass<Vulkan>& ) = delete;
public:
    ComputablePass( const MaterialPass<Vulkan>& materialPass, Pipeline<Vulkan> pipeline, VkPipelineLayout pipelineLayout, std::vector<VkImageMemoryBarrier> imageMemoryBarriers, std::vector<VkBufferMemoryBarrier> bufferMemoryBarriers, bool needsExecutionBarrier )
        : ComputablePassBase( materialPass )
        , mPipeline( std::move( pipeline ) )
        , mPipelineLayout( pipelineLayout )
        , mImageMemoryBarriers( std::move( imageMemoryBarriers ) )
        , mBufferMemoryBarriers( std::move( bufferMemoryBarriers ) )
        , mNeedsExecutionBarrier( needsExecutionBarrier )
    {}
    ComputablePass( ComputablePass<Vulkan>&& ) noexcept;
    ~ComputablePass();

    const auto& GetMaterialPass() const { return apiCast<Vulkan>( mMaterialPass ); }

    const auto& GetVkImageMemoryBarriers() const { return mImageMemoryBarriers; }
    const auto& GetVkBufferMemoryBarriers() const { return mBufferMemoryBarriers; }
    const bool NeedsBarrier() const { return mNeedsExecutionBarrier || (!mImageMemoryBarriers.empty()) || (!mBufferMemoryBarriers.empty()); };   ///< @return true if there needs to be a barrier before executing this compute pass

    /// number of workgroup dispatches to execute (value after the local workgroup sizes are accounted for)
    void SetDispatchGroupCount( std::array<uint32_t, 3> count ) { mDispatchGroupCount = count; }
    const auto& GetDispatchGroupCount() const { return mDispatchGroupCount; }
    /// number of global workgroup threads to run (value before the local workgroup sizes are accounted for).  Requires "WorkGroup" : { "LocalSize": {x,y,z} } in the shader definition json.
    void SetDispatchThreadCount( std::array<uint32_t, 3> count );

    Pipeline<Vulkan>                    mPipeline;                  // Owned by us
    VkPipelineLayout                    mPipelineLayout;            // Owned by ShaderPass or MaterialPass

protected:
    std::vector<VkImageMemoryBarrier>   mImageMemoryBarriers;     ///< Image barriers for ENTRY in to this pass.  Could be stored in material if non compute shaders wanted this barrier information, non compute should use the pass depenancies tho
    std::vector<VkBufferMemoryBarrier>  mBufferMemoryBarriers;   ///< Buffer barriers for ENTRY in to this pass.  Could be stored in material if non compute shaders wanted this barrier information, non compute should use the pass depenancies tho
    bool                                mNeedsExecutionBarrier = false;///< Denotes if we need an execution barrier for ENTRY to this pass (even if there are no image or buffer barriers).
};


