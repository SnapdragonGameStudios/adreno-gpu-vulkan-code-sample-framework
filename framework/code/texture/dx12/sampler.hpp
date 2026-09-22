// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include "../sampler.hpp"
#define NOMINMAX
#include <d3d12.h>

// Forward declarations
class Dx12;
struct D3D12_SAMPLER_DESC;

using SamplerDx12 = Sampler<Dx12>;

D3D12_SAMPLER_DESC StructToDx12( const CreateSamplerObjectInfo& createInfo );


/// @brief Template specialization of sampler container for Dx12 graphics api.
template<>
class Sampler<Dx12> final : public SamplerBase
{
public:
    Sampler() noexcept {};
    Sampler(const Sampler<Dx12>& src) noexcept : m_SamplerDesc(src.m_SamplerDesc) {};
    Sampler(const D3D12_SAMPLER_DESC& desc) noexcept : m_SamplerDesc(desc) {};
    Sampler<Dx12>& operator=(const Sampler<Dx12>& src) noexcept {
        if (this != &src)
            m_SamplerDesc = src.m_SamplerDesc;
        return *this;
    }
    const auto& GetDesc() const { return m_SamplerDesc; }
private:
    D3D12_SAMPLER_DESC m_SamplerDesc{};
};


/// Template specialization for Dx12 CreateSampler
template<>
Sampler<Dx12> CreateSampler( Dx12&, const CreateSamplerObjectInfo& );
