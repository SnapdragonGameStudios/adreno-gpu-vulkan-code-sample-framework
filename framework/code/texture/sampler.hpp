// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include <cassert>
#include <cfloat>
#include <memory>
#include "json/include/nlohmann/json_fwd.hpp"

// Forward declarations
class GraphicsApiBase;
template<typename T_GFXAPI> class Texture;
template<typename T_GFXAPI> class Image;
template<typename T_GFXAPI> class ImageView;
template<typename T_GFXAPI> class Sampler;
using Json = nlohmann::json;

/// @brief Base class for a sampler object.
/// Is subclassed for each graphics API.
class SamplerBase
{
    SamplerBase(const SamplerBase&) = delete;
    SamplerBase& operator=(const SamplerBase&) = delete;
public:
    template<typename T_GFXAPI> using tApiDerived = Sampler<T_GFXAPI>; // make apiCast work!
protected:
    SamplerBase() noexcept {}
};


/// @brief Template for sampler to be specialized by Vulkan graphics api.
template<typename T_GFXAPI>
class Sampler final : public SamplerBase
{
    Sampler(const Sampler<T_GFXAPI>&) = delete;
    Sampler& operator=(const Sampler<T_GFXAPI>&) = delete;
public:
    Sampler() noexcept = delete;                                         // template class expected to be specialized
    Sampler(Sampler<T_GFXAPI>&&) noexcept = delete;                     // template class expected to be specialized
    Sampler<T_GFXAPI>& operator=(Sampler<T_GFXAPI>&&) noexcept = delete;// template class expected to be specialized

    static_assert(sizeof(Sampler<T_GFXAPI>) != sizeof(SamplerBase));   // Ensure this class template is specialized (and not used as-is)
};


enum class SamplerAddressMode {
    Undefined = 0,
    Repeat = 1,
    MirroredRepeat = 2,
    ClampEdge = 3,
    ClampBorder = 4,
    MirroredClampEdge = 5,
    End
};


enum class SamplerFilter {
    Undefined = 0,
    Nearest = 1,
    Linear = 2,
    Anisotropic = 3 //dx12
};


enum class SamplerBorderColor {
    TransparentBlackFloat = 0,
    TransparentBlackInt = 1,
    OpaqueBlackFloat = 2,
    OpaqueBlackInt = 3,
    OpaqueWhiteFloat = 4,
    OpaqueWhiteInt = 5
};

/// Parameters for CreateTextureObject
struct CreateSamplerObjectInfo
{
    SamplerAddressMode  Mode = SamplerAddressMode::Repeat;
    SamplerFilter       Filter = SamplerFilter::Linear;
    SamplerFilter       MipFilter = SamplerFilter::Linear;
    SamplerBorderColor  BorderColor = SamplerBorderColor::TransparentBlackFloat;
    bool                UnnormalizedCoordinates = false;
    float               MipBias = 0.0f;
    float               MinLod = 0.0f;
    float               MaxLod = FLT_MAX;
    float               Anisotropy = 4.0f;
};

/// Create a texture sampler (with some commonly used parameters)
template<typename T_GFXAPI>
Sampler<T_GFXAPI> CreateSampler( T_GFXAPI& gfxApi, SamplerAddressMode SamplerMode, SamplerFilter FilterMode, SamplerBorderColor BorderColor, float MipBias )
{
    CreateSamplerObjectInfo createInfo{};
    createInfo.Mode = SamplerMode;
    createInfo.Filter = FilterMode;
    createInfo.MipFilter = FilterMode,
    createInfo.BorderColor = BorderColor;
    createInfo.MipBias = MipBias;
    return CreateSampler( gfxApi, createInfo );
}

/// Create a texture sampler (must be specialized for the graphics api)
/// Must be specialized for the graphics api - this fallback will assert if called by application code.
template<typename T_GFXAPI>
Sampler<T_GFXAPI> CreateSampler( T_GFXAPI& gfxApi, const CreateSamplerObjectInfo&)
{
    static_assert(sizeof( T_GFXAPI ) != sizeof( T_GFXAPI ), "Must use the specialized version of this function.  Your are likely missing #include \"texture/<GFXAPI>/sampler.hpp\"");
    assert( 0 && "Expecting CreateSampler (per graphics api) to be used" );
    return {};
}

/// Release a texture sampler.
template<typename T_GFXAPI>
void ReleaseSampler( T_GFXAPI& gfxApi, Sampler<T_GFXAPI>* pSampler )
{
    if (!pSampler)
        return;
    *pSampler = Sampler<T_GFXAPI>{};    // destroy and clear
}

void from_json( const Json& j, CreateSamplerObjectInfo& info );
