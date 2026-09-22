// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "sampler.hpp"


constexpr D3D12_FILTER EnumToDx12( SamplerFilter s ) {
    assert(s != SamplerFilter::Undefined && s <= SamplerFilter::Anisotropic);
    if (s <= SamplerFilter::Linear)
        return D3D12_FILTER( int( s ) - 1 );
    return D3D12_FILTER_ANISOTROPIC;
}
static_assert(D3D12_FILTER_MIN_MAG_MIP_POINT == int(SamplerFilter::Nearest) - 1);
static_assert(D3D12_FILTER_MIN_MAG_POINT_MIP_LINEAR == int(SamplerFilter::Linear) - 1);

constexpr D3D12_TEXTURE_ADDRESS_MODE EnumToDx12(SamplerAddressMode s) { return D3D12_TEXTURE_ADDRESS_MODE(int(s) + 1); }
static_assert(D3D12_TEXTURE_ADDRESS_MODE_WRAP == int(SamplerAddressMode::Repeat));
static_assert(D3D12_TEXTURE_ADDRESS_MODE_MIRROR == int(SamplerAddressMode::MirroredRepeat));
static_assert(D3D12_TEXTURE_ADDRESS_MODE_CLAMP == int(SamplerAddressMode::ClampEdge));
static_assert(D3D12_TEXTURE_ADDRESS_MODE_BORDER == int(SamplerAddressMode::ClampBorder));
static_assert(D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE == int(SamplerAddressMode::MirroredClampEdge));


//-----------------------------------------------------------------------------
// Implementation of template function specialization
template<>
Sampler<Dx12> CreateSampler<Dx12>(Dx12& dx12, const CreateSamplerObjectInfo& createInfo)
//-----------------------------------------------------------------------------
{
    return Sampler<Dx12>{ StructToDx12( createInfo ) };
}

//-----------------------------------------------------------------------------
D3D12_SAMPLER_DESC StructToDx12( const CreateSamplerObjectInfo& createInfo )
//-----------------------------------------------------------------------------
{
    const D3D12_TEXTURE_ADDRESS_MODE SamplerModeDx12 = EnumToDx12( createInfo.Mode );
    D3D12_SAMPLER_DESC samplerDesc{
        .Filter = EnumToDx12( createInfo.Filter ),
        .AddressU = SamplerModeDx12,
        .AddressV = SamplerModeDx12,
        .AddressW = SamplerModeDx12,
        .MipLODBias = createInfo.MipBias,
        .MaxAnisotropy = std::max( (UINT)1, (UINT)createInfo.Anisotropy ),
        .ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS,
        .BorderColor = {},
        .MinLOD = createInfo.MinLod,
        .MaxLOD = createInfo.MaxLod
    };
    switch (createInfo.BorderColor) {
    case SamplerBorderColor::TransparentBlackFloat:
    case SamplerBorderColor::TransparentBlackInt:
        break;
    case SamplerBorderColor::OpaqueBlackFloat:
    case SamplerBorderColor::OpaqueBlackInt:
        samplerDesc.BorderColor[3] = 1.0f;
        break;
    case SamplerBorderColor::OpaqueWhiteFloat:
    case SamplerBorderColor::OpaqueWhiteInt:
        std::fill( std::begin( samplerDesc.BorderColor ), std::end( samplerDesc.BorderColor ), 1.0f );
        break;
    }
    return samplerDesc;
}
