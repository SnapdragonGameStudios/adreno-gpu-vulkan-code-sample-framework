// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2023 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "textureFormat.hpp"


//-----------------------------------------------------------------------------
bool FormatHasStencil( TextureFormat format )
//-----------------------------------------------------------------------------
{
    switch( format ) {
    case TextureFormat::D32_SFLOAT_S8_UINT:
    case TextureFormat::D24_UNORM_S8_UINT:
    case TextureFormat::D16_UNORM_S8_UINT:
        return true;
    default:
        return false;
    }
}

//-----------------------------------------------------------------------------
bool FormatHasDepth( TextureFormat format )
//-----------------------------------------------------------------------------
{
    switch (format) {
    case TextureFormat::D32_SFLOAT_S8_UINT:
    case TextureFormat::D24_UNORM_S8_UINT:
    case TextureFormat::D16_UNORM_S8_UINT:
    case TextureFormat::D16_UNORM:
    case TextureFormat::D32_SFLOAT:
        return true;
    default:
        return false;
    }
}

//-----------------------------------------------------------------------------
bool FormatIsCompressed( TextureFormat format )
//-----------------------------------------------------------------------------
{
    switch( format ) {
    case TextureFormat::BC1_RGB_UNORM_BLOCK:
    case TextureFormat::BC1_RGB_SRGB_BLOCK:
    case TextureFormat::BC1_RGBA_UNORM_BLOCK:
    case TextureFormat::BC1_RGBA_SRGB_BLOCK:
    case TextureFormat::BC2_UNORM_BLOCK:
    case TextureFormat::BC2_SRGB_BLOCK:
    case TextureFormat::BC3_UNORM_BLOCK:
    case TextureFormat::BC3_SRGB_BLOCK:
    case TextureFormat::BC4_UNORM_BLOCK:
    case TextureFormat::BC4_SNORM_BLOCK:
    case TextureFormat::BC5_UNORM_BLOCK:
    case TextureFormat::BC5_SNORM_BLOCK:
    case TextureFormat::BC6H_UFLOAT_BLOCK:
    case TextureFormat::BC6H_SFLOAT_BLOCK:
    case TextureFormat::BC7_UNORM_BLOCK:
    case TextureFormat::BC7_SRGB_BLOCK:
    case TextureFormat::ETC2_R8G8B8_UNORM_BLOCK:
    case TextureFormat::ETC2_R8G8B8_SRGB_BLOCK:
    case TextureFormat::ETC2_R8G8B8A1_UNORM_BLOCK:
    case TextureFormat::ETC2_R8G8B8A1_SRGB_BLOCK:
    case TextureFormat::ETC2_R8G8B8A8_UNORM_BLOCK:
    case TextureFormat::ETC2_R8G8B8A8_SRGB_BLOCK:
    case TextureFormat::EAC_R11_UNORM_BLOCK:
    case TextureFormat::EAC_R11_SNORM_BLOCK:
    case TextureFormat::EAC_R11G11_UNORM_BLOCK:
    case TextureFormat::EAC_R11G11_SNORM_BLOCK:
    case TextureFormat::ASTC_4x4_UNORM_BLOCK:
    case TextureFormat::ASTC_4x4_SRGB_BLOCK:
    case TextureFormat::ASTC_5x4_UNORM_BLOCK:
    case TextureFormat::ASTC_5x4_SRGB_BLOCK:
    case TextureFormat::ASTC_5x5_UNORM_BLOCK:
    case TextureFormat::ASTC_5x5_SRGB_BLOCK:
    case TextureFormat::ASTC_6x5_UNORM_BLOCK:
    case TextureFormat::ASTC_6x5_SRGB_BLOCK:
    case TextureFormat::ASTC_6x6_UNORM_BLOCK:
    case TextureFormat::ASTC_6x6_SRGB_BLOCK:
    case TextureFormat::ASTC_8x5_UNORM_BLOCK:
    case TextureFormat::ASTC_8x5_SRGB_BLOCK:
    case TextureFormat::ASTC_8x6_UNORM_BLOCK:
    case TextureFormat::ASTC_8x6_SRGB_BLOCK:
    case TextureFormat::ASTC_8x8_UNORM_BLOCK:
    case TextureFormat::ASTC_8x8_SRGB_BLOCK:
    case TextureFormat::ASTC_10x5_UNORM_BLOCK:
    case TextureFormat::ASTC_10x5_SRGB_BLOCK:
    case TextureFormat::ASTC_10x6_UNORM_BLOCK:
    case TextureFormat::ASTC_10x6_SRGB_BLOCK:
    case TextureFormat::ASTC_10x8_UNORM_BLOCK:
    case TextureFormat::ASTC_10x8_SRGB_BLOCK:
    case TextureFormat::ASTC_10x10_UNORM_BLOCK:
    case TextureFormat::ASTC_10x10_SRGB_BLOCK:
    case TextureFormat::ASTC_12x10_UNORM_BLOCK:
    case TextureFormat::ASTC_12x10_SRGB_BLOCK:
    case TextureFormat::ASTC_12x12_UNORM_BLOCK:
    case TextureFormat::ASTC_12x12_SRGB_BLOCK:
    case TextureFormat::PVRTC1_2BPP_UNORM_BLOCK_IMG:
    case TextureFormat::PVRTC1_4BPP_UNORM_BLOCK_IMG:
    case TextureFormat::PVRTC2_2BPP_UNORM_BLOCK_IMG:
    case TextureFormat::PVRTC2_4BPP_UNORM_BLOCK_IMG:
    case TextureFormat::PVRTC1_2BPP_SRGB_BLOCK_IMG:
    case TextureFormat::PVRTC1_4BPP_SRGB_BLOCK_IMG:
    case TextureFormat::PVRTC2_2BPP_SRGB_BLOCK_IMG:
    case TextureFormat::PVRTC2_4BPP_SRGB_BLOCK_IMG:
        return true;
    default:
        return false;
    }
}

//-----------------------------------------------------------------------------
bool FormatIsSrgb( TextureFormat format )
//-----------------------------------------------------------------------------
{
    switch (format) {
    case TextureFormat::R8_SRGB:
    case TextureFormat::R8G8_SRGB:
    case TextureFormat::R8G8B8_SRGB:
    case TextureFormat::B8G8R8_SRGB:
    case TextureFormat::R8G8B8A8_SRGB:
    case TextureFormat::B8G8R8A8_SRGB:
    case TextureFormat::A8B8G8R8_SRGB_PACK32:
    case TextureFormat::BC1_RGB_SRGB_BLOCK:
    case TextureFormat::BC1_RGBA_SRGB_BLOCK:
    case TextureFormat::BC2_SRGB_BLOCK:
    case TextureFormat::BC3_SRGB_BLOCK:
    case TextureFormat::BC7_SRGB_BLOCK:
    case TextureFormat::ETC2_R8G8B8_UNORM_BLOCK:
    case TextureFormat::ETC2_R8G8B8_SRGB_BLOCK:
    case TextureFormat::ETC2_R8G8B8A1_UNORM_BLOCK:
    case TextureFormat::ETC2_R8G8B8A1_SRGB_BLOCK:
    case TextureFormat::ETC2_R8G8B8A8_UNORM_BLOCK:
    case TextureFormat::ETC2_R8G8B8A8_SRGB_BLOCK:
    case TextureFormat::EAC_R11_UNORM_BLOCK:
    case TextureFormat::EAC_R11_SNORM_BLOCK:
    case TextureFormat::EAC_R11G11_UNORM_BLOCK:
    case TextureFormat::EAC_R11G11_SNORM_BLOCK:
    case TextureFormat::ASTC_4x4_UNORM_BLOCK:
    case TextureFormat::ASTC_4x4_SRGB_BLOCK:
    case TextureFormat::ASTC_5x4_UNORM_BLOCK:
    case TextureFormat::ASTC_5x4_SRGB_BLOCK:
    case TextureFormat::ASTC_5x5_UNORM_BLOCK:
    case TextureFormat::ASTC_5x5_SRGB_BLOCK:
    case TextureFormat::ASTC_6x5_UNORM_BLOCK:
    case TextureFormat::ASTC_6x5_SRGB_BLOCK:
    case TextureFormat::ASTC_6x6_UNORM_BLOCK:
    case TextureFormat::ASTC_6x6_SRGB_BLOCK:
    case TextureFormat::ASTC_8x5_UNORM_BLOCK:
    case TextureFormat::ASTC_8x5_SRGB_BLOCK:
    case TextureFormat::ASTC_8x6_UNORM_BLOCK:
    case TextureFormat::ASTC_8x6_SRGB_BLOCK:
    case TextureFormat::ASTC_8x8_UNORM_BLOCK:
    case TextureFormat::ASTC_8x8_SRGB_BLOCK:
    case TextureFormat::ASTC_10x5_UNORM_BLOCK:
    case TextureFormat::ASTC_10x5_SRGB_BLOCK:
    case TextureFormat::ASTC_10x6_UNORM_BLOCK:
    case TextureFormat::ASTC_10x6_SRGB_BLOCK:
    case TextureFormat::ASTC_10x8_UNORM_BLOCK:
    case TextureFormat::ASTC_10x8_SRGB_BLOCK:
    case TextureFormat::ASTC_10x10_UNORM_BLOCK:
    case TextureFormat::ASTC_10x10_SRGB_BLOCK:
    case TextureFormat::ASTC_12x10_UNORM_BLOCK:
    case TextureFormat::ASTC_12x10_SRGB_BLOCK:
    case TextureFormat::ASTC_12x12_UNORM_BLOCK:
    case TextureFormat::ASTC_12x12_SRGB_BLOCK:
        return true;
    default:
        return false;
    }
}

//-----------------------------------------------------------------------------
size_t FormatBytesPerPixel( TextureFormat format )
//-----------------------------------------------------------------------------
{
    switch (format)
    {
    case TextureFormat::R4G4_UNORM_PACK8:
    case TextureFormat::R8_UNORM:
    case TextureFormat::R8_SNORM:
    case TextureFormat::R8_USCALED:
    case TextureFormat::R8_SSCALED:
    case TextureFormat::R8_UINT:
    case TextureFormat::R8_SINT:
    case TextureFormat::R8_SRGB:
    case TextureFormat::S8_UINT:
        return 1;
    case TextureFormat::R8G8_UNORM:
    case TextureFormat::R8G8_SNORM:
    case TextureFormat::R8G8_USCALED:
    case TextureFormat::R8G8_SSCALED:
    case TextureFormat::R8G8_UINT:
    case TextureFormat::R8G8_SINT:
    case TextureFormat::R8G8_SRGB:
    case TextureFormat::R4G4B4A4_UNORM_PACK16:
    case TextureFormat::B4G4R4A4_UNORM_PACK16:
    case TextureFormat::R5G6B5_UNORM_PACK16:
    case TextureFormat::B5G6R5_UNORM_PACK16:
    case TextureFormat::R5G5B5A1_UNORM_PACK16:
    case TextureFormat::B5G5R5A1_UNORM_PACK16:
    case TextureFormat::A1R5G5B5_UNORM_PACK16:
    case TextureFormat::R16_UNORM:
    case TextureFormat::R16_SNORM:
    case TextureFormat::R16_USCALED:
    case TextureFormat::R16_SSCALED:
    case TextureFormat::R16_UINT:
    case TextureFormat::R16_SINT:
    case TextureFormat::R16_SFLOAT:
    case TextureFormat::D16_UNORM:
    case TextureFormat::R12X4_UNORM_PACK16:
    case TextureFormat::A4R4G4B4_UNORM_PACK16:
    case TextureFormat::A4B4G4R4_UNORM_PACK16:
        return 2;
    case TextureFormat::R8G8B8_UNORM:
    case TextureFormat::R8G8B8_SNORM:
    case TextureFormat::R8G8B8_USCALED:
    case TextureFormat::R8G8B8_SSCALED:
    case TextureFormat::R8G8B8_UINT:
    case TextureFormat::R8G8B8_SINT:
    case TextureFormat::R8G8B8_SRGB:
    case TextureFormat::B8G8R8_UNORM:
    case TextureFormat::B8G8R8_SNORM:
    case TextureFormat::B8G8R8_USCALED:
    case TextureFormat::B8G8R8_SSCALED:
    case TextureFormat::B8G8R8_UINT:
    case TextureFormat::B8G8R8_SINT:
    case TextureFormat::B8G8R8_SRGB:
        return 3;
    case TextureFormat::R8G8B8A8_UNORM:
    case TextureFormat::R8G8B8A8_SNORM:
    case TextureFormat::R8G8B8A8_USCALED:
    case TextureFormat::R8G8B8A8_SSCALED:
    case TextureFormat::R8G8B8A8_UINT:
    case TextureFormat::R8G8B8A8_SINT:
    case TextureFormat::R8G8B8A8_SRGB:
    case TextureFormat::B8G8R8A8_UNORM:
    case TextureFormat::B8G8R8A8_SNORM:
    case TextureFormat::B8G8R8A8_USCALED:
    case TextureFormat::B8G8R8A8_SSCALED:
    case TextureFormat::B8G8R8A8_UINT:
    case TextureFormat::B8G8R8A8_SINT:
    case TextureFormat::B8G8R8A8_SRGB:
    case TextureFormat::A8B8G8R8_UNORM_PACK32:
    case TextureFormat::A8B8G8R8_SNORM_PACK32:
    case TextureFormat::A8B8G8R8_USCALED_PACK32:
    case TextureFormat::A8B8G8R8_SSCALED_PACK32:
    case TextureFormat::A8B8G8R8_UINT_PACK32:
    case TextureFormat::A8B8G8R8_SINT_PACK32:
    case TextureFormat::A8B8G8R8_SRGB_PACK32:
    case TextureFormat::A2R10G10B10_UNORM_PACK32:
    case TextureFormat::A2R10G10B10_SNORM_PACK32:
    case TextureFormat::A2R10G10B10_USCALED_PACK32:
    case TextureFormat::A2R10G10B10_SSCALED_PACK32:
    case TextureFormat::A2R10G10B10_UINT_PACK32:
    case TextureFormat::A2R10G10B10_SINT_PACK32:
    case TextureFormat::A2B10G10R10_UNORM_PACK32:
    case TextureFormat::A2B10G10R10_SNORM_PACK32:
    case TextureFormat::A2B10G10R10_USCALED_PACK32:
    case TextureFormat::A2B10G10R10_SSCALED_PACK32:
    case TextureFormat::A2B10G10R10_UINT_PACK32:
    case TextureFormat::A2B10G10R10_SINT_PACK32:
    case TextureFormat::R16G16_UNORM:
    case TextureFormat::R16G16_SNORM:
    case TextureFormat::R16G16_USCALED:
    case TextureFormat::R16G16_SSCALED:
    case TextureFormat::R16G16_UINT:
    case TextureFormat::R16G16_SINT:
    case TextureFormat::R16G16_SFLOAT:
    case TextureFormat::R32_UINT:
    case TextureFormat::R32_SINT:
    case TextureFormat::R32_SFLOAT:
    case TextureFormat::B10G11R11_UFLOAT_PACK32:
    case TextureFormat::E5B9G9R9_UFLOAT_PACK32:
    case TextureFormat::X8_D24_UNORM_PACK32:
    case TextureFormat::D32_SFLOAT:
        return 4;
    case TextureFormat::R16G16B16_UNORM:
    case TextureFormat::R16G16B16_SNORM:
    case TextureFormat::R16G16B16_USCALED:
    case TextureFormat::R16G16B16_SSCALED:
    case TextureFormat::R16G16B16_UINT:
    case TextureFormat::R16G16B16_SINT:
    case TextureFormat::R16G16B16_SFLOAT:
        return 6;
    case TextureFormat::R32G32B32_UINT:
    case TextureFormat::R32G32B32_SINT:
    case TextureFormat::R32G32B32_SFLOAT:
        return 12;
    case TextureFormat::R32G32B32A32_UINT:
    case TextureFormat::R32G32B32A32_SINT:
    case TextureFormat::R32G32B32A32_SFLOAT:
        return 16;
    case TextureFormat::R16G16B16A16_UNORM:
    case TextureFormat::R16G16B16A16_SNORM:
    case TextureFormat::R16G16B16A16_USCALED:
    case TextureFormat::R16G16B16A16_SSCALED:
    case TextureFormat::R16G16B16A16_UINT:
    case TextureFormat::R16G16B16A16_SINT:
    case TextureFormat::R16G16B16A16_SFLOAT:
    case TextureFormat::R64_UINT:
    case TextureFormat::R64_SINT:
    case TextureFormat::R64_SFLOAT:
    case TextureFormat::R32G32_UINT:
    case TextureFormat::R32G32_SINT:
    case TextureFormat::R32G32_SFLOAT:
    case TextureFormat::R64G64_UINT:
    case TextureFormat::R64G64_SINT:
    case TextureFormat::R64G64_SFLOAT:
        return 8;
        return 32;
    case TextureFormat::R64G64B64_UINT:
    case TextureFormat::R64G64B64_SINT:
    case TextureFormat::R64G64B64_SFLOAT:
        return 48;
    case TextureFormat::R64G64B64A64_UINT:
    case TextureFormat::R64G64B64A64_SINT:
    case TextureFormat::R64G64B64A64_SFLOAT:
        return 64;
    case TextureFormat::UNDEFINED:
    default:
        return 0;
    }
}
