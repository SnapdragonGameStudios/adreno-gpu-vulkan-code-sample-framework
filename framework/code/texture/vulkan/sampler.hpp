// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include "../sampler.hpp"
#include "vulkan/refHandle.hpp"
#include <volk/volk.h>
#include <vector>

class Vulkan;

/// @brief Template specialization of sampler container for Vulkan graphics api.
template<>
class Sampler<Vulkan> final : public SamplerBase
{
public:
    Sampler() noexcept;
    ~Sampler() noexcept;
    Sampler(VkDevice, VkSampler) noexcept;
    Sampler(Sampler<Vulkan>&& src) noexcept;
    Sampler& operator=(Sampler<Vulkan>&& src) noexcept;
    Sampler Copy() const { return Sampler{*this}; }

    VkSampler GetVkSampler() const { return m_Sampler; }
    bool IsEmpty() const { return m_Sampler == VK_NULL_HANDLE; }

private:
    Sampler( const Sampler<Vulkan>& src ) noexcept {
        m_Sampler = src.m_Sampler;
    }

    RefHandle<VkSampler> m_Sampler;
};
