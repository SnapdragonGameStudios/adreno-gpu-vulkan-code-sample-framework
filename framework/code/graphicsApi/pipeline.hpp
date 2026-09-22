// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

// Forward declarations


/// Simple wrapper around VkPipeline or DirectX render target data used by BeginRenderPass and CreateGraphicsPipelineState.
/// Simplifies creation (and checks for leaks on destruction - is up to the owner to call Destroy)
/// This template class expected to be specialized (if this template throws compiler errors then the code is not using the specialization classes which is an issue!)
/// @ingroup Material
template<typename T_GFXAPI>
class Pipeline
{
    Pipeline& operator=(const Pipeline<T_GFXAPI>&) = delete;
    Pipeline(const Pipeline<T_GFXAPI>&) = delete;
public:
    Pipeline() noexcept = delete;
    Pipeline( Pipeline<T_GFXAPI>&&) noexcept = delete;
    ~Pipeline() = delete;

    static_assert(sizeof( Pipeline<T_GFXAPI>) >= 1);   // Ensure this class template is specialized (and not used as-is)
};

