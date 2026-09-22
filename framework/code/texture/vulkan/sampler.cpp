// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "sampler.hpp"
#include "vulkan/vulkan.hpp"

VkFilter EnumToVk(SamplerFilter s) { assert(s != SamplerFilter::Undefined && s <= SamplerFilter::Linear ); return VkFilter(int( s ) - 1); }
static_assert(VK_FILTER_NEAREST == int(SamplerFilter::Nearest) - 1);
static_assert(VK_FILTER_LINEAR == int(SamplerFilter::Linear) - 1);

VkSamplerMipmapMode EnumToVkSamplerMipmapMode(SamplerFilter s) { assert(s != SamplerFilter::Undefined); return VkSamplerMipmapMode(int(s) - 1); }
static_assert(VK_SAMPLER_MIPMAP_MODE_NEAREST == int(SamplerFilter::Nearest) - 1);
static_assert(VK_SAMPLER_MIPMAP_MODE_LINEAR == int(SamplerFilter::Linear) - 1);

constexpr VkSamplerAddressMode EnumToVk(SamplerAddressMode s) { assert(s != SamplerAddressMode::Undefined); return VkSamplerAddressMode(int(s) - 1); }
static_assert(VK_SAMPLER_ADDRESS_MODE_REPEAT == int(SamplerAddressMode::Repeat) -1);
static_assert(VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT == int(SamplerAddressMode::MirroredRepeat) - 1);
static_assert(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE == int(SamplerAddressMode::ClampEdge) - 1);
static_assert(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER == int(SamplerAddressMode::ClampBorder) - 1);
static_assert(VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE == int(SamplerAddressMode::MirroredClampEdge) - 1);

constexpr VkBorderColor EnumToVk(SamplerBorderColor s) { return VkBorderColor(int(s)); }
static_assert(VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK == int(SamplerBorderColor::TransparentBlackFloat));
static_assert(VK_BORDER_COLOR_INT_TRANSPARENT_BLACK == int(SamplerBorderColor::TransparentBlackInt));
static_assert(VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK == int(SamplerBorderColor::OpaqueBlackFloat));
static_assert(VK_BORDER_COLOR_INT_OPAQUE_BLACK == int(SamplerBorderColor::OpaqueBlackInt));
static_assert(VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE == int(SamplerBorderColor::OpaqueWhiteFloat));
static_assert(VK_BORDER_COLOR_INT_OPAQUE_WHITE == int(SamplerBorderColor::OpaqueWhiteInt));


static_assert(int(SamplerAddressMode::Repeat) == VK_SAMPLER_ADDRESS_MODE_REPEAT + 1);
static_assert(int(SamplerAddressMode::MirroredRepeat) == VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT + 1);
static_assert(int(SamplerAddressMode::ClampEdge) == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE + 1);
static_assert(int(SamplerAddressMode::ClampBorder) == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER + 1);
static_assert(int(SamplerAddressMode::MirroredClampEdge) == VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE + 1);

static_assert(int(SamplerFilter::Nearest) == VK_FILTER_NEAREST + 1);
static_assert(int(SamplerFilter::Linear) == VK_FILTER_LINEAR + 1);

static_assert(int(SamplerBorderColor::TransparentBlackFloat) == VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
static_assert(int(SamplerBorderColor::TransparentBlackInt) == VK_BORDER_COLOR_INT_TRANSPARENT_BLACK);
static_assert(int(SamplerBorderColor::OpaqueBlackFloat) == VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK);
static_assert(int(SamplerBorderColor::OpaqueBlackInt) == VK_BORDER_COLOR_INT_OPAQUE_BLACK);
static_assert(int(SamplerBorderColor::OpaqueWhiteFloat) == VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE);
static_assert(int(SamplerBorderColor::OpaqueWhiteInt) == VK_BORDER_COLOR_INT_OPAQUE_WHITE);

//
// Constructors/move-operators for Sampler<Vulkan>.
//
Sampler<Vulkan>::Sampler() noexcept
    : m_Sampler( VK_NULL_HANDLE, VK_NULL_HANDLE )
{}
Sampler<Vulkan>::~Sampler() noexcept
{
}
Sampler<Vulkan>::Sampler( VkDevice device, VkSampler sampler ) noexcept
    : m_Sampler( device, sampler )
{}
Sampler<Vulkan>::Sampler(Sampler<Vulkan>&& src) noexcept
    : m_Sampler( std::move( src.m_Sampler ) ) 
{
}
Sampler<Vulkan>& Sampler<Vulkan>::operator=(Sampler<Vulkan>&& src ) noexcept
{
    if (this != &src)
    {
        m_Sampler = src.m_Sampler;
        src.m_Sampler = {};
    }
    return *this;
}


//-----------------------------------------------------------------------------
template<>
Sampler<Vulkan> CreateSampler<Vulkan>(Vulkan& vulkan, const CreateSamplerObjectInfo& createInfo)
//-----------------------------------------------------------------------------
{
    const bool anisotropyEnable = createInfo.Anisotropy > 1.0f && createInfo.Filter == SamplerFilter::Linear;
    const VkSamplerCreateInfo SamplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .flags = 0,
        .magFilter = EnumToVk(createInfo.Filter),
        .minFilter = EnumToVk(createInfo.Filter),
        .mipmapMode = EnumToVkSamplerMipmapMode(createInfo.MipFilter),
        .addressModeU = EnumToVk(createInfo.Mode),
        .addressModeV = EnumToVk(createInfo.Mode),
        .addressModeW = EnumToVk(createInfo.Mode),
        .mipLodBias = createInfo.MipBias,
        .anisotropyEnable = anisotropyEnable ? VK_TRUE : VK_FALSE,
        .maxAnisotropy = anisotropyEnable == VK_TRUE ? createInfo.Anisotropy : 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = createInfo.MinLod,
        .maxLod = createInfo.MaxLod,
        .borderColor = EnumToVk(createInfo.BorderColor),
        .unnormalizedCoordinates = createInfo.UnnormalizedCoordinates ? VK_TRUE : VK_FALSE };

    VkSampler vkSampler;
    VkResult retVal = vkCreateSampler(vulkan.m_VulkanDevice, &SamplerInfo, NULL, &vkSampler);
    if (!CheckVkError("vkCreateSampler()", retVal))
        return {};
    return {vulkan.m_VulkanDevice, vkSampler};
}

