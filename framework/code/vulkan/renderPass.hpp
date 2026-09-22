// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include <span>
#include <volk/volk.h>
#include "graphicsApi/graphicsApiBase.hpp"
#include "vulkan/refHandle.hpp"
#include "graphicsApi/renderPass.hpp"
#include <cassert>

// Forward declarations
class Vulkan;
enum class Msaa;
enum class TextureFormat;


/// Simple wrapper around VkRenderPass.
/// Simplifies creation (and checks for leaks on destruction - is up to the owner to call Destroy)
/// Specialization of RenderPass<T_GFXAPI>
/// @ingroup Vulkan
template<>
class RenderPass<Vulkan>
{
    RenderPass& operator=(const RenderPass<Vulkan>&) = delete;
public:
    RenderPass() noexcept;
    RenderPass( RenderPass<Vulkan>&& other ) noexcept = default;
    RenderPass& operator=( RenderPass&& other ) noexcept = default;

    RenderPass(VkDevice device, VkRenderPass renderPass) noexcept;
    //RenderPass& operator=( VkRenderPass renderPass ) noexcept {
    //    assert( mRenderPass == VK_NULL_HANDLE );
    //    mRenderPass = renderPass;
    //    return *this;
    //}
    ~RenderPass();
    operator bool() const { return mRenderPass != VK_NULL_HANDLE;  }

    RenderPass Copy() const { return RenderPass{*this}; }

    RefHandle<VkRenderPass> mRenderPass;

private:
    RenderPass( const RenderPass<Vulkan>& src ) noexcept {
        mRenderPass = src.mRenderPass;
    }
};


RenderPass<Vulkan> CreateRenderPass( Vulkan& vulkan,
                                    std::span < const TextureFormat > ColorFormats,
                                    TextureFormat DepthFormat,
                                    Msaa Msaa,
                                    RenderPassInputUsage ColorInputUsage,
                                    RenderPassOutputUsage ColorOutputUsage,
                                    bool ShouldClearDepth,
                                    RenderPassOutputUsage DepthOutputUsage,
                                    std::span < const TextureFormat > ResolveFormats);

//RenderPass<Vulkan> CreateRenderPass( Vulkan& vulkan,
//                                     const VkRenderPassCreateInfo& );
