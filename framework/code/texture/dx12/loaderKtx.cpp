// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "loaderKtx.hpp"
#include "sampler.hpp"
#include "texture.hpp"
#include "dx12/dx12.hpp"
#include <ktx.h>  // KTX-Software
#include <algorithm>
#include <KTX-Software/lib/gl_format.h>	// for GL_* defines

#undef min
#undef max

// Forward declarations
static uint32_t WidthToPitch(DXGI_FORMAT dxgiFormat, uint32_t width);
static uint32_t AreaToBytes(DXGI_FORMAT dxgiFormat, uint32_t width, uint32_t height);


TextureKtx<Dx12>::TextureKtx(Dx12& dx12) noexcept : TextureKtxBase(), m_Dx12(dx12)
{
}

TextureKtx<Dx12>::~TextureKtx()
{}

bool TextureKtx<Dx12>::Initialize()
{
    if (!TextureKtxBase::Initialize())
        return false;
    return true;
}

void TextureKtx<Dx12>::Release()
{
    TextureKtxBase::Release();
}


static constexpr std::array<VkFormat, 247> sTextureFormatToVk{
	VK_FORMAT_UNDEFINED,
	VK_FORMAT_R4G4_UNORM_PACK8,
	VK_FORMAT_R4G4B4A4_UNORM_PACK16,
	VK_FORMAT_B4G4R4A4_UNORM_PACK16,
	VK_FORMAT_R5G6B5_UNORM_PACK16,
	VK_FORMAT_B5G6R5_UNORM_PACK16,
	VK_FORMAT_R5G5B5A1_UNORM_PACK16,
	VK_FORMAT_B5G5R5A1_UNORM_PACK16,
	VK_FORMAT_A1R5G5B5_UNORM_PACK16,
	VK_FORMAT_R8_UNORM,
	VK_FORMAT_R8_SNORM,
	VK_FORMAT_R8_USCALED,
	VK_FORMAT_R8_SSCALED,
	VK_FORMAT_R8_UINT,
	VK_FORMAT_R8_SINT,
	VK_FORMAT_R8_SRGB,
	VK_FORMAT_R8G8_UNORM,
	VK_FORMAT_R8G8_SNORM,
	VK_FORMAT_R8G8_USCALED,
	VK_FORMAT_R8G8_SSCALED,
	VK_FORMAT_R8G8_UINT,
	VK_FORMAT_R8G8_SINT,
	VK_FORMAT_R8G8_SRGB,
	VK_FORMAT_R8G8B8_UNORM,
	VK_FORMAT_R8G8B8_SNORM,
	VK_FORMAT_R8G8B8_USCALED,
	VK_FORMAT_R8G8B8_SSCALED,
	VK_FORMAT_R8G8B8_UINT,
	VK_FORMAT_R8G8B8_SINT,
	VK_FORMAT_R8G8B8_SRGB,
	VK_FORMAT_B8G8R8_UNORM,
	VK_FORMAT_B8G8R8_SNORM,
	VK_FORMAT_B8G8R8_USCALED,
	VK_FORMAT_B8G8R8_SSCALED,
	VK_FORMAT_B8G8R8_UINT,
	VK_FORMAT_B8G8R8_SINT,
	VK_FORMAT_B8G8R8_SRGB,
	VK_FORMAT_R8G8B8A8_UNORM,
	VK_FORMAT_R8G8B8A8_SNORM,
	VK_FORMAT_R8G8B8A8_USCALED,
	VK_FORMAT_R8G8B8A8_SSCALED,
	VK_FORMAT_R8G8B8A8_UINT,
	VK_FORMAT_R8G8B8A8_SINT,
	VK_FORMAT_R8G8B8A8_SRGB,
	VK_FORMAT_B8G8R8A8_UNORM,
	VK_FORMAT_B8G8R8A8_SNORM,
	VK_FORMAT_B8G8R8A8_USCALED,
	VK_FORMAT_B8G8R8A8_SSCALED,
	VK_FORMAT_B8G8R8A8_UINT,
	VK_FORMAT_B8G8R8A8_SINT,
	VK_FORMAT_B8G8R8A8_SRGB,
	VK_FORMAT_A8B8G8R8_UNORM_PACK32,
	VK_FORMAT_A8B8G8R8_SNORM_PACK32,
	VK_FORMAT_A8B8G8R8_USCALED_PACK32,
	VK_FORMAT_A8B8G8R8_SSCALED_PACK32,
	VK_FORMAT_A8B8G8R8_UINT_PACK32,
	VK_FORMAT_A8B8G8R8_SINT_PACK32,
	VK_FORMAT_A8B8G8R8_SRGB_PACK32,
	VK_FORMAT_A2R10G10B10_UNORM_PACK32,
	VK_FORMAT_A2R10G10B10_SNORM_PACK32,
	VK_FORMAT_A2R10G10B10_USCALED_PACK32,
	VK_FORMAT_A2R10G10B10_SSCALED_PACK32,
	VK_FORMAT_A2R10G10B10_UINT_PACK32,
	VK_FORMAT_A2R10G10B10_SINT_PACK32,
	VK_FORMAT_A2B10G10R10_UNORM_PACK32,
	VK_FORMAT_A2B10G10R10_SNORM_PACK32,
	VK_FORMAT_A2B10G10R10_USCALED_PACK32,
	VK_FORMAT_A2B10G10R10_SSCALED_PACK32,
	VK_FORMAT_A2B10G10R10_UINT_PACK32,
	VK_FORMAT_A2B10G10R10_SINT_PACK32,
	VK_FORMAT_R16_UNORM,
	VK_FORMAT_R16_SNORM,
	VK_FORMAT_R16_USCALED,
	VK_FORMAT_R16_SSCALED,
	VK_FORMAT_R16_UINT,
	VK_FORMAT_R16_SINT,
	VK_FORMAT_R16_SFLOAT,
	VK_FORMAT_R16G16_UNORM,
	VK_FORMAT_R16G16_SNORM,
	VK_FORMAT_R16G16_USCALED,
	VK_FORMAT_R16G16_SSCALED,
	VK_FORMAT_R16G16_UINT,
	VK_FORMAT_R16G16_SINT,
	VK_FORMAT_R16G16_SFLOAT,
	VK_FORMAT_R16G16B16_UNORM,
	VK_FORMAT_R16G16B16_SNORM,
	VK_FORMAT_R16G16B16_USCALED,
	VK_FORMAT_R16G16B16_SSCALED,
	VK_FORMAT_R16G16B16_UINT,
	VK_FORMAT_R16G16B16_SINT,
	VK_FORMAT_R16G16B16_SFLOAT,
	VK_FORMAT_R16G16B16A16_UNORM,
	VK_FORMAT_R16G16B16A16_SNORM,
	VK_FORMAT_R16G16B16A16_USCALED,
	VK_FORMAT_R16G16B16A16_SSCALED,
	VK_FORMAT_R16G16B16A16_UINT,
	VK_FORMAT_R16G16B16A16_SINT,
	VK_FORMAT_R16G16B16A16_SFLOAT,
	VK_FORMAT_R32_UINT,
	VK_FORMAT_R32_SINT,
	VK_FORMAT_R32_SFLOAT,
	VK_FORMAT_R32G32_UINT,
	VK_FORMAT_R32G32_SINT,
	VK_FORMAT_R32G32_SFLOAT,
	VK_FORMAT_R32G32B32_UINT,
	VK_FORMAT_R32G32B32_SINT,
	VK_FORMAT_R32G32B32_SFLOAT,
	VK_FORMAT_R32G32B32A32_UINT,
	VK_FORMAT_R32G32B32A32_SINT,
	VK_FORMAT_R32G32B32A32_SFLOAT,
	VK_FORMAT_R64_UINT,
	VK_FORMAT_R64_SINT,
	VK_FORMAT_R64_SFLOAT,
	VK_FORMAT_R64G64_UINT,
	VK_FORMAT_R64G64_SINT,
	VK_FORMAT_R64G64_SFLOAT,
	VK_FORMAT_R64G64B64_UINT,
	VK_FORMAT_R64G64B64_SINT,
	VK_FORMAT_R64G64B64_SFLOAT,
	VK_FORMAT_R64G64B64A64_UINT,
	VK_FORMAT_R64G64B64A64_SINT,
	VK_FORMAT_R64G64B64A64_SFLOAT,
	VK_FORMAT_B10G11R11_UFLOAT_PACK32,
	VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
	VK_FORMAT_D16_UNORM,
	VK_FORMAT_X8_D24_UNORM_PACK32,
	VK_FORMAT_D32_SFLOAT,
	VK_FORMAT_S8_UINT,
	VK_FORMAT_D16_UNORM_S8_UINT,
	VK_FORMAT_D24_UNORM_S8_UINT,
	VK_FORMAT_D32_SFLOAT_S8_UINT,
	VK_FORMAT_BC1_RGB_UNORM_BLOCK,
	VK_FORMAT_BC1_RGB_SRGB_BLOCK,
	VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
	VK_FORMAT_BC1_RGBA_SRGB_BLOCK,
	VK_FORMAT_BC2_UNORM_BLOCK,
	VK_FORMAT_BC2_SRGB_BLOCK,
	VK_FORMAT_BC3_UNORM_BLOCK,
	VK_FORMAT_BC3_SRGB_BLOCK,
	VK_FORMAT_BC4_UNORM_BLOCK,
	VK_FORMAT_BC4_SNORM_BLOCK,
	VK_FORMAT_BC5_UNORM_BLOCK,
	VK_FORMAT_BC5_SNORM_BLOCK,
	VK_FORMAT_BC6H_UFLOAT_BLOCK,
	VK_FORMAT_BC6H_SFLOAT_BLOCK,
	VK_FORMAT_BC7_UNORM_BLOCK,
	VK_FORMAT_BC7_SRGB_BLOCK,
	VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
	VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK,
	VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK,
	VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK,
	VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
	VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,
	VK_FORMAT_EAC_R11_UNORM_BLOCK,
	VK_FORMAT_EAC_R11_SNORM_BLOCK,
	VK_FORMAT_EAC_R11G11_UNORM_BLOCK,
	VK_FORMAT_EAC_R11G11_SNORM_BLOCK,
	VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
	VK_FORMAT_ASTC_4x4_SRGB_BLOCK,
	VK_FORMAT_ASTC_5x4_UNORM_BLOCK,
	VK_FORMAT_ASTC_5x4_SRGB_BLOCK,
	VK_FORMAT_ASTC_5x5_UNORM_BLOCK,
	VK_FORMAT_ASTC_5x5_SRGB_BLOCK,
	VK_FORMAT_ASTC_6x5_UNORM_BLOCK,
	VK_FORMAT_ASTC_6x5_SRGB_BLOCK,
	VK_FORMAT_ASTC_6x6_UNORM_BLOCK,
	VK_FORMAT_ASTC_6x6_SRGB_BLOCK,
	VK_FORMAT_ASTC_8x5_UNORM_BLOCK,
	VK_FORMAT_ASTC_8x5_SRGB_BLOCK,
	VK_FORMAT_ASTC_8x6_UNORM_BLOCK,
	VK_FORMAT_ASTC_8x6_SRGB_BLOCK,
	VK_FORMAT_ASTC_8x8_UNORM_BLOCK,
	VK_FORMAT_ASTC_8x8_SRGB_BLOCK,
	VK_FORMAT_ASTC_10x5_UNORM_BLOCK,
	VK_FORMAT_ASTC_10x5_SRGB_BLOCK,
	VK_FORMAT_ASTC_10x6_UNORM_BLOCK,
	VK_FORMAT_ASTC_10x6_SRGB_BLOCK,
	VK_FORMAT_ASTC_10x8_UNORM_BLOCK,
	VK_FORMAT_ASTC_10x8_SRGB_BLOCK,
	VK_FORMAT_ASTC_10x10_UNORM_BLOCK,
	VK_FORMAT_ASTC_10x10_SRGB_BLOCK,
	VK_FORMAT_ASTC_12x10_UNORM_BLOCK,
	VK_FORMAT_ASTC_12x10_SRGB_BLOCK,
	VK_FORMAT_ASTC_12x12_UNORM_BLOCK,
	VK_FORMAT_ASTC_12x12_SRGB_BLOCK,    //184

	VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG, //1000054000
	VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG,
	VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG,
	VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG,
	VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG,
	VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG,
	VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG,
	VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG,

	VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT, //1000066000
	VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK_EXT,
	VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK_EXT,
	VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK_EXT,
	VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK_EXT,
	VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK_EXT,
	VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK_EXT,
	VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK_EXT,
	VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK_EXT,
	VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK_EXT,
	VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK_EXT,
	VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK_EXT,
	VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK_EXT,
	VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT,

	VK_FORMAT_G8B8G8R8_422_UNORM,   //1000156000
	VK_FORMAT_B8G8R8G8_422_UNORM,
	VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
	VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
	VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM,
	VK_FORMAT_G8_B8R8_2PLANE_422_UNORM,
	VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM,
	VK_FORMAT_R10X6_UNORM_PACK16,
	VK_FORMAT_R10X6G10X6_UNORM_2PACK16,
	VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16,
	VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16,
	VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16,
	VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16,
	VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
	VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16,
	VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16,
	VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16,
	VK_FORMAT_R12X4_UNORM_PACK16,
	VK_FORMAT_R12X4G12X4_UNORM_2PACK16,
	VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16,
	VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16,
	VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16,
	VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16,
	VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16,
	VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16,
	VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16,
	VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16,
	VK_FORMAT_G16B16G16R16_422_UNORM,
	VK_FORMAT_B16G16R16G16_422_UNORM,
	VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM,
	VK_FORMAT_G16_B16R16_2PLANE_420_UNORM,
	VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM,
	VK_FORMAT_G16_B16R16_2PLANE_422_UNORM,
	VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM,
	(VkFormat)1000330000, //VK_FORMAT_G8_B8R8_2PLANE_444_UNORM,
	(VkFormat)1000330001, //VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16,
	(VkFormat)1000330002, //VK_FORMAT_G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16,
	(VkFormat)1000330003, //VK_FORMAT_G16_B16R16_2PLANE_444_UNORM,
	VK_FORMAT_A4R4G4B4_UNORM_PACK16_EXT,
	VK_FORMAT_A4B4G4R4_UNORM_PACK16_EXT,
};
#include <algorithm>
// Ensure the sTextureFormatToVk array is in strict numerical order (so we can quickly std::find
consteval bool CheckFormatOrdering() {
	return std::is_sorted( std::begin( sTextureFormatToVk ), std::end( sTextureFormatToVk ) );
}
static_assert(CheckFormatOrdering());


// do some rough checks to ensure our formats are in the order we expect (or else the above lookup doesnt work!)
static_assert(int( TextureFormat::B8G8R8A8_SRGB ) == 50);
static_assert(sTextureFormatToVk[int( TextureFormat::B8G8R8A8_SRGB )] == int( VK_FORMAT_B8G8R8A8_SRGB ));
static_assert(int( TextureFormat::R32_SFLOAT ) == 100);
static_assert(int( TextureFormat::ETC2_R8G8B8A1_SRGB_BLOCK ) == 150);
static_assert(sTextureFormatToVk[int( TextureFormat::ETC2_R8G8B8A1_SRGB_BLOCK )] == int( VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK ));
static_assert(int( TextureFormat::ASTC_12x12_SRGB_BLOCK ) == 184);
static_assert(int( TextureFormat::PVRTC1_2BPP_UNORM_BLOCK_IMG ) == 185);
static_assert(sTextureFormatToVk[int( TextureFormat::PVRTC1_2BPP_UNORM_BLOCK_IMG )] == int( VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG ));
static_assert(int( TextureFormat::ASTC_4x4_SFLOAT_BLOCK ) == 193);
static_assert(sTextureFormatToVk[int( TextureFormat::ASTC_4x4_SFLOAT_BLOCK )] == int( VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT ));
static_assert(int( TextureFormat::G8B8G8R8_422_UNORM ) == 207);
static_assert(sTextureFormatToVk[int( TextureFormat::G8B8G8R8_422_UNORM )] == int( VK_FORMAT_G8B8G8R8_422_UNORM ));
static_assert(int( TextureFormat::EndCount ) == 247);

static inline VkFormat vkGetFormatFromOpenGLFormat( const GLenum format, const GLenum type )
{
	switch (type)
	{
		//
		// 8 bits per component
		//
	case GL_UNSIGNED_BYTE:
	{
		switch (format)
		{
		case GL_RED:					return VK_FORMAT_R8_UNORM;
		case GL_RG:						return VK_FORMAT_R8G8_UNORM;
		case GL_RGB:					return VK_FORMAT_R8G8B8_UNORM;
		case GL_BGR:					return VK_FORMAT_B8G8R8_UNORM;
		case GL_RGBA:					return VK_FORMAT_R8G8B8A8_UNORM;
		case GL_BGRA:					return VK_FORMAT_B8G8R8A8_UNORM;
		case GL_RED_INTEGER:			return VK_FORMAT_R8_UINT;
		case GL_RG_INTEGER:				return VK_FORMAT_R8G8_UINT;
		case GL_RGB_INTEGER:			return VK_FORMAT_R8G8B8_UINT;
		case GL_BGR_INTEGER:			return VK_FORMAT_B8G8R8_UINT;
		case GL_RGBA_INTEGER:			return VK_FORMAT_R8G8B8A8_UINT;
		case GL_BGRA_INTEGER:			return VK_FORMAT_B8G8R8A8_UINT;
		case GL_STENCIL_INDEX:			return VK_FORMAT_S8_UINT;
		case GL_DEPTH_COMPONENT:		return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_STENCIL:			return VK_FORMAT_UNDEFINED;
		}
		break;
	}
	case GL_BYTE:
	{
		switch (format)
		{
		case GL_RED:					return VK_FORMAT_R8_SNORM;
		case GL_RG:						return VK_FORMAT_R8G8_SNORM;
		case GL_RGB:					return VK_FORMAT_R8G8B8_SNORM;
		case GL_BGR:					return VK_FORMAT_B8G8R8_SNORM;
		case GL_RGBA:					return VK_FORMAT_R8G8B8A8_SNORM;
		case GL_BGRA:					return VK_FORMAT_B8G8R8A8_SNORM;
		case GL_RED_INTEGER:			return VK_FORMAT_R8_SINT;
		case GL_RG_INTEGER:				return VK_FORMAT_R8G8_SINT;
		case GL_RGB_INTEGER:			return VK_FORMAT_R8G8B8_SINT;
		case GL_BGR_INTEGER:			return VK_FORMAT_B8G8R8_SINT;
		case GL_RGBA_INTEGER:			return VK_FORMAT_R8G8B8A8_SINT;
		case GL_BGRA_INTEGER:			return VK_FORMAT_B8G8R8A8_SINT;
		case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_COMPONENT:		return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_STENCIL:			return VK_FORMAT_UNDEFINED;
		}
		break;
	}

	//
	// 16 bits per component
	//
	case GL_UNSIGNED_SHORT:
	{
		switch (format)
		{
		case GL_RED:					return VK_FORMAT_R16_UNORM;
		case GL_RG:						return VK_FORMAT_R16G16_UNORM;
		case GL_RGB:					return VK_FORMAT_R16G16B16_UNORM;
		case GL_BGR:					return VK_FORMAT_UNDEFINED;
		case GL_RGBA:					return VK_FORMAT_R16G16B16A16_UNORM;
		case GL_BGRA:					return VK_FORMAT_UNDEFINED;
		case GL_RED_INTEGER:			return VK_FORMAT_R16_UINT;
		case GL_RG_INTEGER:				return VK_FORMAT_R16G16_UINT;
		case GL_RGB_INTEGER:			return VK_FORMAT_R16G16B16_UINT;
		case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_RGBA_INTEGER:			return VK_FORMAT_R16G16B16A16_UINT;
		case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_COMPONENT:		return VK_FORMAT_D16_UNORM;
		case GL_DEPTH_STENCIL:			return VK_FORMAT_D16_UNORM_S8_UINT;
		}
		break;
	}
	case GL_SHORT:
	{
		switch (format)
		{
		case GL_RED:					return VK_FORMAT_R16_SNORM;
		case GL_RG:						return VK_FORMAT_R16G16_SNORM;
		case GL_RGB:					return VK_FORMAT_R16G16B16_SNORM;
		case GL_BGR:					return VK_FORMAT_UNDEFINED;
		case GL_RGBA:					return VK_FORMAT_R16G16B16A16_SNORM;
		case GL_BGRA:					return VK_FORMAT_UNDEFINED;
		case GL_RED_INTEGER:			return VK_FORMAT_R16_SINT;
		case GL_RG_INTEGER:				return VK_FORMAT_R16G16_SINT;
		case GL_RGB_INTEGER:			return VK_FORMAT_R16G16B16_SINT;
		case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_RGBA_INTEGER:			return VK_FORMAT_R16G16B16A16_SINT;
		case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_COMPONENT:		return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_STENCIL:			return VK_FORMAT_UNDEFINED;
		}
		break;
	}
	case GL_HALF_FLOAT:
	case GL_HALF_FLOAT_OES:
	{
		switch (format)
		{
		case GL_RED:					return VK_FORMAT_R16_SFLOAT;
		case GL_RG:						return VK_FORMAT_R16G16_SFLOAT;
		case GL_RGB:					return VK_FORMAT_R16G16B16_SFLOAT;
		case GL_BGR:					return VK_FORMAT_UNDEFINED;
		case GL_RGBA:					return VK_FORMAT_R16G16B16A16_SFLOAT;
		case GL_BGRA:					return VK_FORMAT_UNDEFINED;
		case GL_RED_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_RG_INTEGER:				return VK_FORMAT_UNDEFINED;
		case GL_RGB_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_RGBA_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_COMPONENT:		return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_STENCIL:			return VK_FORMAT_UNDEFINED;
		}
		break;
	}

	//
	// 32 bits per component
	//
	case GL_UNSIGNED_INT:
	{
		switch (format)
		{
		case GL_RED:					return VK_FORMAT_R32_UINT;
		case GL_RG:						return VK_FORMAT_R32G32_UINT;
		case GL_RGB:					return VK_FORMAT_R32G32B32_UINT;
		case GL_BGR:					return VK_FORMAT_UNDEFINED;
		case GL_RGBA:					return VK_FORMAT_R32G32B32A32_UINT;
		case GL_BGRA:					return VK_FORMAT_UNDEFINED;
		case GL_RED_INTEGER:			return VK_FORMAT_R32_UINT;
		case GL_RG_INTEGER:				return VK_FORMAT_R32G32_UINT;
		case GL_RGB_INTEGER:			return VK_FORMAT_R32G32B32_UINT;
		case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_RGBA_INTEGER:			return VK_FORMAT_R32G32B32A32_UINT;
		case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_COMPONENT:		return VK_FORMAT_X8_D24_UNORM_PACK32;
		case GL_DEPTH_STENCIL:			return VK_FORMAT_D24_UNORM_S8_UINT;
		}
		break;
	}
	case GL_INT:
	{
		switch (format)
		{
		case GL_RED:					return VK_FORMAT_R32_SINT;
		case GL_RG:						return VK_FORMAT_R32G32_SINT;
		case GL_RGB:					return VK_FORMAT_R32G32B32_SINT;
		case GL_BGR:					return VK_FORMAT_UNDEFINED;
		case GL_RGBA:					return VK_FORMAT_R32G32B32A32_SINT;
		case GL_BGRA:					return VK_FORMAT_UNDEFINED;
		case GL_RED_INTEGER:			return VK_FORMAT_R32_SINT;
		case GL_RG_INTEGER:				return VK_FORMAT_R32G32_SINT;
		case GL_RGB_INTEGER:			return VK_FORMAT_R32G32B32_SINT;
		case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_RGBA_INTEGER:			return VK_FORMAT_R32G32B32A32_SINT;
		case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_COMPONENT:		return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_STENCIL:			return VK_FORMAT_UNDEFINED;
		}
		break;
	}
	case GL_FLOAT:
	{
		switch (format)
		{
		case GL_RED:					return VK_FORMAT_R32_SFLOAT;
		case GL_RG:						return VK_FORMAT_R32G32_SFLOAT;
		case GL_RGB:					return VK_FORMAT_R32G32B32_SFLOAT;
		case GL_BGR:					return VK_FORMAT_UNDEFINED;
		case GL_RGBA:					return VK_FORMAT_R32G32B32A32_SFLOAT;
		case GL_BGRA:					return VK_FORMAT_UNDEFINED;
		case GL_RED_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_RG_INTEGER:				return VK_FORMAT_UNDEFINED;
		case GL_RGB_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_RGBA_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_COMPONENT:		return VK_FORMAT_D32_SFLOAT;
		case GL_DEPTH_STENCIL:			return VK_FORMAT_D32_SFLOAT_S8_UINT;
		}
		break;
	}

	//
	// 64 bits per component
	//
	case GL_UNSIGNED_INT64:
	{
		switch (format)
		{
		case GL_RED:					return VK_FORMAT_R64_UINT;
		case GL_RG:						return VK_FORMAT_R64G64_UINT;
		case GL_RGB:					return VK_FORMAT_R64G64B64_UINT;
		case GL_BGR:					return VK_FORMAT_UNDEFINED;
		case GL_RGBA:					return VK_FORMAT_R64G64B64A64_UINT;
		case GL_BGRA:					return VK_FORMAT_UNDEFINED;
		case GL_RED_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_RG_INTEGER:				return VK_FORMAT_UNDEFINED;
		case GL_RGB_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_RGBA_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_COMPONENT:		return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_STENCIL:			return VK_FORMAT_UNDEFINED;
		}
		break;
	}
	case GL_INT64:
	{
		switch (format)
		{
		case GL_RED:					return VK_FORMAT_R64_SINT;
		case GL_RG:						return VK_FORMAT_R64G64_SINT;
		case GL_RGB:					return VK_FORMAT_R64G64B64_SINT;
		case GL_BGR:					return VK_FORMAT_UNDEFINED;
		case GL_RGBA:					return VK_FORMAT_R64G64B64A64_SINT;
		case GL_BGRA:					return VK_FORMAT_UNDEFINED;
		case GL_RED_INTEGER:			return VK_FORMAT_R64_SINT;
		case GL_RG_INTEGER:				return VK_FORMAT_R64G64_SINT;
		case GL_RGB_INTEGER:			return VK_FORMAT_R64G64B64_SINT;
		case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_RGBA_INTEGER:			return VK_FORMAT_R64G64B64A64_SINT;
		case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_COMPONENT:		return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_STENCIL:			return VK_FORMAT_UNDEFINED;
		}
		break;
	}
	case GL_DOUBLE:
	{
		switch (format)
		{
		case GL_RED:					return VK_FORMAT_R64_SFLOAT;
		case GL_RG:						return VK_FORMAT_R64G64_SFLOAT;
		case GL_RGB:					return VK_FORMAT_R64G64B64_SFLOAT;
		case GL_BGR:					return VK_FORMAT_UNDEFINED;
		case GL_RGBA:					return VK_FORMAT_R64G64B64A64_SFLOAT;
		case GL_BGRA:					return VK_FORMAT_UNDEFINED;
		case GL_RED_INTEGER:			return VK_FORMAT_R64_SFLOAT;
		case GL_RG_INTEGER:				return VK_FORMAT_R64G64_SFLOAT;
		case GL_RGB_INTEGER:			return VK_FORMAT_R64G64B64_SFLOAT;
		case GL_BGR_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_RGBA_INTEGER:			return VK_FORMAT_R64G64B64A64_SFLOAT;
		case GL_BGRA_INTEGER:			return VK_FORMAT_UNDEFINED;
		case GL_STENCIL_INDEX:			return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_COMPONENT:		return VK_FORMAT_UNDEFINED;
		case GL_DEPTH_STENCIL:			return VK_FORMAT_UNDEFINED;
		}
		break;
	}

	//
	// Packed
	//
	case GL_UNSIGNED_BYTE_3_3_2:
		assert( format == GL_RGB || format == GL_RGB_INTEGER );
		return VK_FORMAT_UNDEFINED;
	case GL_UNSIGNED_BYTE_2_3_3_REV:
		assert( format == GL_BGR || format == GL_BGR_INTEGER );
		return VK_FORMAT_UNDEFINED;
	case GL_UNSIGNED_SHORT_5_6_5:
		assert( format == GL_RGB || format == GL_RGB_INTEGER );
		return VK_FORMAT_R5G6B5_UNORM_PACK16;
	case GL_UNSIGNED_SHORT_5_6_5_REV:
		assert( format == GL_BGR || format == GL_BGR_INTEGER );
		return VK_FORMAT_B5G6R5_UNORM_PACK16;
	case GL_UNSIGNED_SHORT_4_4_4_4:
		assert( format == GL_RGB || format == GL_BGRA || format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER );
		return VK_FORMAT_R4G4B4A4_UNORM_PACK16;
	case GL_UNSIGNED_SHORT_4_4_4_4_REV:
		assert( format == GL_RGB || format == GL_BGRA || format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER );
		return VK_FORMAT_B4G4R4A4_UNORM_PACK16;
	case GL_UNSIGNED_SHORT_5_5_5_1:
		assert( format == GL_RGB || format == GL_BGRA || format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER );
		return VK_FORMAT_R5G5B5A1_UNORM_PACK16;
	case GL_UNSIGNED_SHORT_1_5_5_5_REV:
		assert( format == GL_RGB || format == GL_BGRA || format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER );
		return VK_FORMAT_A1R5G5B5_UNORM_PACK16;
	case GL_UNSIGNED_INT_8_8_8_8:
		assert( format == GL_RGB || format == GL_BGRA || format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER );
		return (format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER) ? VK_FORMAT_R8G8B8A8_UINT : VK_FORMAT_R8G8B8A8_UNORM;
	case GL_UNSIGNED_INT_8_8_8_8_REV:
		assert( format == GL_RGB || format == GL_BGRA || format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER );
		return (format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER) ? VK_FORMAT_A8B8G8R8_UINT_PACK32 : VK_FORMAT_A8B8G8R8_UNORM_PACK32;
	case GL_UNSIGNED_INT_10_10_10_2:
		assert( format == GL_RGB || format == GL_BGRA || format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER );
		return (format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER) ? VK_FORMAT_A2R10G10B10_UINT_PACK32 : VK_FORMAT_A2R10G10B10_UNORM_PACK32;
	case GL_UNSIGNED_INT_2_10_10_10_REV:
		assert( format == GL_RGB || format == GL_BGRA || format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER );
		return (format == GL_RGB_INTEGER || format == GL_BGRA_INTEGER) ? VK_FORMAT_A2B10G10R10_UINT_PACK32 : VK_FORMAT_A2B10G10R10_UNORM_PACK32;
	case GL_UNSIGNED_INT_10F_11F_11F_REV:
		assert( format == GL_RGB || format == GL_BGR );
		return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
	case GL_UNSIGNED_INT_5_9_9_9_REV:
		assert( format == GL_RGB || format == GL_BGR );
		return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
	case GL_UNSIGNED_INT_24_8:
		assert( format == GL_DEPTH_STENCIL );
		return VK_FORMAT_D24_UNORM_S8_UINT;
	case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
		assert( format == GL_DEPTH_STENCIL );
		return VK_FORMAT_D32_SFLOAT_S8_UINT;
	}

	return VK_FORMAT_UNDEFINED;
}

static inline VkFormat vkGetFormatFromOpenGLInternalFormat( const GLenum internalFormat )
{
	switch (internalFormat)
	{
		//
		// 8 bits per component
		//
	case GL_R8:												return VK_FORMAT_R8_UNORM;					// 1-component, 8-bit unsigned normalized
	case GL_RG8:											return VK_FORMAT_R8G8_UNORM;				// 2-component, 8-bit unsigned normalized
	case GL_RGB8:											return VK_FORMAT_R8G8B8_UNORM;				// 3-component, 8-bit unsigned normalized
	case GL_RGBA8:											return VK_FORMAT_R8G8B8A8_UNORM;			// 4-component, 8-bit unsigned normalized

	case GL_R8_SNORM:										return VK_FORMAT_R8_SNORM;					// 1-component, 8-bit signed normalized
	case GL_RG8_SNORM:										return VK_FORMAT_R8G8_SNORM;				// 2-component, 8-bit signed normalized
	case GL_RGB8_SNORM:										return VK_FORMAT_R8G8B8_SNORM;				// 3-component, 8-bit signed normalized
	case GL_RGBA8_SNORM:									return VK_FORMAT_R8G8B8A8_SNORM;			// 4-component, 8-bit signed normalized

	case GL_R8UI:											return VK_FORMAT_R8_UINT;					// 1-component, 8-bit unsigned integer
	case GL_RG8UI:											return VK_FORMAT_R8G8_UINT;					// 2-component, 8-bit unsigned integer
	case GL_RGB8UI:											return VK_FORMAT_R8G8B8_UINT;				// 3-component, 8-bit unsigned integer
	case GL_RGBA8UI:										return VK_FORMAT_R8G8B8A8_UINT;				// 4-component, 8-bit unsigned integer

	case GL_R8I:											return VK_FORMAT_R8_SINT;					// 1-component, 8-bit signed integer
	case GL_RG8I:											return VK_FORMAT_R8G8_SINT;					// 2-component, 8-bit signed integer
	case GL_RGB8I:											return VK_FORMAT_R8G8B8_SINT;				// 3-component, 8-bit signed integer
	case GL_RGBA8I:											return VK_FORMAT_R8G8B8A8_SINT;				// 4-component, 8-bit signed integer

	case GL_SR8:											return VK_FORMAT_R8_SRGB;					// 1-component, 8-bit sRGB
	case GL_SRG8:											return VK_FORMAT_R8G8_SRGB;					// 2-component, 8-bit sRGB
	case GL_SRGB8:											return VK_FORMAT_R8G8B8_SRGB;				// 3-component, 8-bit sRGB
	case GL_SRGB8_ALPHA8:									return VK_FORMAT_R8G8B8A8_SRGB;				// 4-component, 8-bit sRGB

		//
		// 16 bits per component
		//
	case GL_R16:											return VK_FORMAT_R16_UNORM;					// 1-component, 16-bit unsigned normalized
	case GL_RG16:											return VK_FORMAT_R16G16_UNORM;				// 2-component, 16-bit unsigned normalized
	case GL_RGB16:											return VK_FORMAT_R16G16B16_UNORM;			// 3-component, 16-bit unsigned normalized
	case GL_RGBA16:											return VK_FORMAT_R16G16B16A16_UNORM;		// 4-component, 16-bit unsigned normalized

	case GL_R16_SNORM:										return VK_FORMAT_R16_SNORM;					// 1-component, 16-bit signed normalized
	case GL_RG16_SNORM:										return VK_FORMAT_R16G16_SNORM;				// 2-component, 16-bit signed normalized
	case GL_RGB16_SNORM:									return VK_FORMAT_R16G16B16_SNORM;			// 3-component, 16-bit signed normalized
	case GL_RGBA16_SNORM:									return VK_FORMAT_R16G16B16A16_SNORM;		// 4-component, 16-bit signed normalized

	case GL_R16UI:											return VK_FORMAT_R16_UINT;					// 1-component, 16-bit unsigned integer
	case GL_RG16UI:											return VK_FORMAT_R16G16_UINT;				// 2-component, 16-bit unsigned integer
	case GL_RGB16UI:										return VK_FORMAT_R16G16B16_UINT;			// 3-component, 16-bit unsigned integer
	case GL_RGBA16UI:										return VK_FORMAT_R16G16B16A16_UINT;			// 4-component, 16-bit unsigned integer

	case GL_R16I:											return VK_FORMAT_R16_SINT;					// 1-component, 16-bit signed integer
	case GL_RG16I:											return VK_FORMAT_R16G16_SINT;				// 2-component, 16-bit signed integer
	case GL_RGB16I:											return VK_FORMAT_R16G16B16_SINT;			// 3-component, 16-bit signed integer
	case GL_RGBA16I:										return VK_FORMAT_R16G16B16A16_SINT;			// 4-component, 16-bit signed integer

	case GL_R16F:											return VK_FORMAT_R16_SFLOAT;				// 1-component, 16-bit floating-point
	case GL_RG16F:											return VK_FORMAT_R16G16_SFLOAT;				// 2-component, 16-bit floating-point
	case GL_RGB16F:											return VK_FORMAT_R16G16B16_SFLOAT;			// 3-component, 16-bit floating-point
	case GL_RGBA16F:										return VK_FORMAT_R16G16B16A16_SFLOAT;		// 4-component, 16-bit floating-point

		//
		// 32 bits per component
		//
	case GL_R32UI:											return VK_FORMAT_R32_UINT;					// 1-component, 32-bit unsigned integer
	case GL_RG32UI:											return VK_FORMAT_R32G32_UINT;				// 2-component, 32-bit unsigned integer
	case GL_RGB32UI:										return VK_FORMAT_R32G32B32_UINT;			// 3-component, 32-bit unsigned integer
	case GL_RGBA32UI:										return VK_FORMAT_R32G32B32A32_UINT;			// 4-component, 32-bit unsigned integer

	case GL_R32I:											return VK_FORMAT_R32_SINT;					// 1-component, 32-bit signed integer
	case GL_RG32I:											return VK_FORMAT_R32G32_SINT;				// 2-component, 32-bit signed integer
	case GL_RGB32I:											return VK_FORMAT_R32G32B32_SINT;			// 3-component, 32-bit signed integer
	case GL_RGBA32I:										return VK_FORMAT_R32G32B32A32_SINT;			// 4-component, 32-bit signed integer

	case GL_R32F:											return VK_FORMAT_R32_SFLOAT;				// 1-component, 32-bit floating-point
	case GL_RG32F:											return VK_FORMAT_R32G32_SFLOAT;				// 2-component, 32-bit floating-point
	case GL_RGB32F:											return VK_FORMAT_R32G32B32_SFLOAT;			// 3-component, 32-bit floating-point
	case GL_RGBA32F:										return VK_FORMAT_R32G32B32A32_SFLOAT;		// 4-component, 32-bit floating-point

		//
		// Packed
		//
	case GL_R3_G3_B2:										return VK_FORMAT_UNDEFINED;					// 3-component 3:3:2,       unsigned normalized
	case GL_RGB4:											return VK_FORMAT_UNDEFINED;					// 3-component 4:4:4,       unsigned normalized
	case GL_RGB5:											return VK_FORMAT_R5G5B5A1_UNORM_PACK16;		// 3-component 5:5:5,       unsigned normalized
	case GL_RGB565:											return VK_FORMAT_R5G6B5_UNORM_PACK16;		// 3-component 5:6:5,       unsigned normalized
	case GL_RGB10:											return VK_FORMAT_A2R10G10B10_UNORM_PACK32;	// 3-component 10:10:10,    unsigned normalized
	case GL_RGB12:											return VK_FORMAT_UNDEFINED;					// 3-component 12:12:12,    unsigned normalized
	case GL_RGBA2:											return VK_FORMAT_UNDEFINED;					// 4-component 2:2:2:2,     unsigned normalized
	case GL_RGBA4:											return VK_FORMAT_R4G4B4A4_UNORM_PACK16;		// 4-component 4:4:4:4,     unsigned normalized
	case GL_RGBA12:											return VK_FORMAT_UNDEFINED;					// 4-component 12:12:12:12, unsigned normalized
	case GL_RGB5_A1:										return VK_FORMAT_A1R5G5B5_UNORM_PACK16;		// 4-component 5:5:5:1,     unsigned normalized
	case GL_RGB10_A2:										return VK_FORMAT_A2R10G10B10_UNORM_PACK32;	// 4-component 10:10:10:2,  unsigned normalized
	case GL_RGB10_A2UI:										return VK_FORMAT_A2R10G10B10_UINT_PACK32;	// 4-component 10:10:10:2,  unsigned integer
	case GL_R11F_G11F_B10F:									return VK_FORMAT_B10G11R11_UFLOAT_PACK32;	// 3-component 11:11:10,    floating-point
	case GL_RGB9_E5:										return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;	// 3-component/exp 9:9:9/5, floating-point

		//
		// S3TC/DXT/BC
		//

	case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:					return VK_FORMAT_BC1_RGB_UNORM_BLOCK;		// line through 3D space, 4x4 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT:					return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;		// line through 3D space plus 1-bit alpha, 4x4 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT:					return VK_FORMAT_BC2_UNORM_BLOCK;			// line through 3D space plus line through 1D space, 4x4 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:					return VK_FORMAT_BC3_UNORM_BLOCK;			// line through 3D space plus 4-bit alpha, 4x4 blocks, unsigned normalized

	case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT:					return VK_FORMAT_BC1_RGB_SRGB_BLOCK;		// line through 3D space, 4x4 blocks, sRGB
	case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT:			return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;		// line through 3D space plus 1-bit alpha, 4x4 blocks, sRGB
	case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT:			return VK_FORMAT_BC2_SRGB_BLOCK;			// line through 3D space plus line through 1D space, 4x4 blocks, sRGB
	case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT:			return VK_FORMAT_BC3_SRGB_BLOCK;			// line through 3D space plus 4-bit alpha, 4x4 blocks, sRGB

	case GL_COMPRESSED_LUMINANCE_LATC1_EXT:					return VK_FORMAT_BC4_UNORM_BLOCK;			// line through 1D space, 4x4 blocks, unsigned normalized
	case GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT:			return VK_FORMAT_BC5_UNORM_BLOCK;			// two lines through 1D space, 4x4 blocks, unsigned normalized
	case GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT:			return VK_FORMAT_BC4_SNORM_BLOCK;			// line through 1D space, 4x4 blocks, signed normalized
	case GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT:	return VK_FORMAT_BC5_SNORM_BLOCK;			// two lines through 1D space, 4x4 blocks, signed normalized

	case GL_COMPRESSED_RED_RGTC1:							return VK_FORMAT_BC4_UNORM_BLOCK;			// line through 1D space, 4x4 blocks, unsigned normalized
	case GL_COMPRESSED_RG_RGTC2:							return VK_FORMAT_BC5_UNORM_BLOCK;			// two lines through 1D space, 4x4 blocks, unsigned normalized
	case GL_COMPRESSED_SIGNED_RED_RGTC1:					return VK_FORMAT_BC4_SNORM_BLOCK;			// line through 1D space, 4x4 blocks, signed normalized
	case GL_COMPRESSED_SIGNED_RG_RGTC2:						return VK_FORMAT_BC5_SNORM_BLOCK;			// two lines through 1D space, 4x4 blocks, signed normalized

	case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:				return VK_FORMAT_BC6H_UFLOAT_BLOCK;			// 3-component, 4x4 blocks, unsigned floating-point
	case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:				return VK_FORMAT_BC6H_SFLOAT_BLOCK;			// 3-component, 4x4 blocks, signed floating-point
	case GL_COMPRESSED_RGBA_BPTC_UNORM:						return VK_FORMAT_BC7_UNORM_BLOCK;			// 4-component, 4x4 blocks, unsigned normalized
	case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:				return VK_FORMAT_BC7_SRGB_BLOCK;			// 4-component, 4x4 blocks, sRGB

		//
		// ETC
		//
	case GL_ETC1_RGB8_OES:									return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;	// 3-component ETC1, 4x4 blocks, unsigned normalized

	case GL_COMPRESSED_RGB8_ETC2:							return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;	// 3-component ETC2, 4x4 blocks, unsigned normalized
	case GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2:		return VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK;	// 4-component ETC2 with 1-bit alpha, 4x4 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA8_ETC2_EAC:						return VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;	// 4-component ETC2, 4x4 blocks, unsigned normalized

	case GL_COMPRESSED_SRGB8_ETC2:							return VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK;	// 3-component ETC2, 4x4 blocks, sRGB
	case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:		return VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK;	// 4-component ETC2 with 1-bit alpha, 4x4 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:				return VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;	// 4-component ETC2, 4x4 blocks, sRGB

	case GL_COMPRESSED_R11_EAC:								return VK_FORMAT_EAC_R11_UNORM_BLOCK;		// 1-component ETC, 4x4 blocks, unsigned normalized
	case GL_COMPRESSED_RG11_EAC:							return VK_FORMAT_EAC_R11G11_UNORM_BLOCK;	// 2-component ETC, 4x4 blocks, unsigned normalized
	case GL_COMPRESSED_SIGNED_R11_EAC:						return VK_FORMAT_EAC_R11_SNORM_BLOCK;		// 1-component ETC, 4x4 blocks, signed normalized
	case GL_COMPRESSED_SIGNED_RG11_EAC:						return VK_FORMAT_EAC_R11G11_SNORM_BLOCK;	// 2-component ETC, 4x4 blocks, signed normalized

		//
		// PVRTC
		//
	case GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG:				return VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;	// 3-component PVRTC, 16x8 blocks, unsigned normalized
	case GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG:				return VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG;	// 3-component PVRTC,  8x8 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG:				return VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG;	// 4-component PVRTC, 16x8 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG:				return VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG;	// 4-component PVRTC,  8x8 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_PVRTC_2BPPV2_IMG:				return VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG;	// 4-component PVRTC,  8x4 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_PVRTC_4BPPV2_IMG:				return VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG;	// 4-component PVRTC,  4x4 blocks, unsigned normalized

	case GL_COMPRESSED_SRGB_PVRTC_2BPPV1_EXT:				return VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG;	// 3-component PVRTC, 16x8 blocks, sRGB
	case GL_COMPRESSED_SRGB_PVRTC_4BPPV1_EXT:				return VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG;	// 3-component PVRTC,  8x8 blocks, sRGB
	case GL_COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV1_EXT:			return VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG;	// 4-component PVRTC, 16x8 blocks, sRGB
	case GL_COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV1_EXT:			return VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG;	// 4-component PVRTC,  8x8 blocks, sRGB
	case GL_COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV2_IMG:			return VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG;	// 4-component PVRTC,  8x4 blocks, sRGB
	case GL_COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV2_IMG:			return VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG;	// 4-component PVRTC,  4x4 blocks, sRGB

		//
		// ASTC
		//
	case GL_COMPRESSED_RGBA_ASTC_4x4_KHR:					return VK_FORMAT_ASTC_4x4_UNORM_BLOCK;		// 4-component ASTC, 4x4 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_5x4_KHR:					return VK_FORMAT_ASTC_5x4_UNORM_BLOCK;		// 4-component ASTC, 5x4 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_5x5_KHR:					return VK_FORMAT_ASTC_5x5_UNORM_BLOCK;		// 4-component ASTC, 5x5 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_6x5_KHR:					return VK_FORMAT_ASTC_6x5_UNORM_BLOCK;		// 4-component ASTC, 6x5 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_6x6_KHR:					return VK_FORMAT_ASTC_6x6_UNORM_BLOCK;		// 4-component ASTC, 6x6 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_8x5_KHR:					return VK_FORMAT_ASTC_8x5_UNORM_BLOCK;		// 4-component ASTC, 8x5 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_8x6_KHR:					return VK_FORMAT_ASTC_8x6_UNORM_BLOCK;		// 4-component ASTC, 8x6 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_8x8_KHR:					return VK_FORMAT_ASTC_8x8_UNORM_BLOCK;		// 4-component ASTC, 8x8 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_10x5_KHR:					return VK_FORMAT_ASTC_10x5_UNORM_BLOCK;		// 4-component ASTC, 10x5 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_10x6_KHR:					return VK_FORMAT_ASTC_10x6_UNORM_BLOCK;		// 4-component ASTC, 10x6 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_10x8_KHR:					return VK_FORMAT_ASTC_10x8_UNORM_BLOCK;		// 4-component ASTC, 10x8 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_10x10_KHR:					return VK_FORMAT_ASTC_10x10_UNORM_BLOCK;	// 4-component ASTC, 10x10 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_12x10_KHR:					return VK_FORMAT_ASTC_12x10_UNORM_BLOCK;	// 4-component ASTC, 12x10 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_12x12_KHR:					return VK_FORMAT_ASTC_12x12_UNORM_BLOCK;	// 4-component ASTC, 12x12 blocks, unsigned normalized

	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:			return VK_FORMAT_ASTC_4x4_SRGB_BLOCK;		// 4-component ASTC, 4x4 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:			return VK_FORMAT_ASTC_5x4_SRGB_BLOCK;		// 4-component ASTC, 5x4 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:			return VK_FORMAT_ASTC_5x5_SRGB_BLOCK;		// 4-component ASTC, 5x5 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:			return VK_FORMAT_ASTC_6x5_SRGB_BLOCK;		// 4-component ASTC, 6x5 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:			return VK_FORMAT_ASTC_6x6_SRGB_BLOCK;		// 4-component ASTC, 6x6 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:			return VK_FORMAT_ASTC_8x5_SRGB_BLOCK;		// 4-component ASTC, 8x5 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:			return VK_FORMAT_ASTC_8x6_SRGB_BLOCK;		// 4-component ASTC, 8x6 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:			return VK_FORMAT_ASTC_8x8_SRGB_BLOCK;		// 4-component ASTC, 8x8 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:			return VK_FORMAT_ASTC_10x5_SRGB_BLOCK;		// 4-component ASTC, 10x5 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:			return VK_FORMAT_ASTC_10x6_SRGB_BLOCK;		// 4-component ASTC, 10x6 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:			return VK_FORMAT_ASTC_10x8_SRGB_BLOCK;		// 4-component ASTC, 10x8 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:			return VK_FORMAT_ASTC_10x10_SRGB_BLOCK;		// 4-component ASTC, 10x10 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:			return VK_FORMAT_ASTC_12x10_SRGB_BLOCK;		// 4-component ASTC, 12x10 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:			return VK_FORMAT_ASTC_12x12_SRGB_BLOCK;		// 4-component ASTC, 12x12 blocks, sRGB

	case GL_COMPRESSED_RGBA_ASTC_3x3x3_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 3x3x3 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_4x3x3_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 4x3x3 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_4x4x3_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 4x4x3 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_4x4x4_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 4x4x4 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_5x4x4_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 5x4x4 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_5x5x4_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 5x5x4 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_5x5x5_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 5x5x5 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_6x5x5_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 6x5x5 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_6x6x5_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 6x6x5 blocks, unsigned normalized
	case GL_COMPRESSED_RGBA_ASTC_6x6x6_OES:					return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 6x6x6 blocks, unsigned normalized

	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_3x3x3_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 3x3x3 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x3x3_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 4x3x3 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4x3_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 4x4x3 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4x4_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 4x4x4 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4x4_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 5x4x4 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5x4_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 5x5x4 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5x5_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 5x5x5 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5x5_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 6x5x5 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6x5_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 6x6x5 blocks, sRGB
	case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6x6_OES:			return VK_FORMAT_UNDEFINED;					// 4-component ASTC, 6x6x6 blocks, sRGB

		//
		// ATC
		//
	case GL_ATC_RGB_AMD:									return VK_FORMAT_UNDEFINED;					// 3-component, 4x4 blocks, unsigned normalized
	case GL_ATC_RGBA_EXPLICIT_ALPHA_AMD:					return VK_FORMAT_UNDEFINED;					// 4-component, 4x4 blocks, unsigned normalized
	case GL_ATC_RGBA_INTERPOLATED_ALPHA_AMD:				return VK_FORMAT_UNDEFINED;					// 4-component, 4x4 blocks, unsigned normalized

		//
		// Palletized
		//
	case GL_PALETTE4_RGB8_OES:								return VK_FORMAT_UNDEFINED;					// 3-component 8:8:8,   4-bit palette, unsigned normalized
	case GL_PALETTE4_RGBA8_OES:								return VK_FORMAT_UNDEFINED;					// 4-component 8:8:8:8, 4-bit palette, unsigned normalized
	case GL_PALETTE4_R5_G6_B5_OES:							return VK_FORMAT_UNDEFINED;					// 3-component 5:6:5,   4-bit palette, unsigned normalized
	case GL_PALETTE4_RGBA4_OES:								return VK_FORMAT_UNDEFINED;					// 4-component 4:4:4:4, 4-bit palette, unsigned normalized
	case GL_PALETTE4_RGB5_A1_OES:							return VK_FORMAT_UNDEFINED;					// 4-component 5:5:5:1, 4-bit palette, unsigned normalized
	case GL_PALETTE8_RGB8_OES:								return VK_FORMAT_UNDEFINED;					// 3-component 8:8:8,   8-bit palette, unsigned normalized
	case GL_PALETTE8_RGBA8_OES:								return VK_FORMAT_UNDEFINED;					// 4-component 8:8:8:8, 8-bit palette, unsigned normalized
	case GL_PALETTE8_R5_G6_B5_OES:							return VK_FORMAT_UNDEFINED;					// 3-component 5:6:5,   8-bit palette, unsigned normalized
	case GL_PALETTE8_RGBA4_OES:								return VK_FORMAT_UNDEFINED;					// 4-component 4:4:4:4, 8-bit palette, unsigned normalized
	case GL_PALETTE8_RGB5_A1_OES:							return VK_FORMAT_UNDEFINED;					// 4-component 5:5:5:1, 8-bit palette, unsigned normalized

		//
		// Depth/stencil
		//
	case GL_DEPTH_COMPONENT16:								return VK_FORMAT_D16_UNORM;
	case GL_DEPTH_COMPONENT24:								return VK_FORMAT_X8_D24_UNORM_PACK32;
	case GL_DEPTH_COMPONENT32:								return VK_FORMAT_UNDEFINED;
	case GL_DEPTH_COMPONENT32F:								return VK_FORMAT_D32_SFLOAT;
	case GL_DEPTH_COMPONENT32F_NV:							return VK_FORMAT_D32_SFLOAT;
	case GL_STENCIL_INDEX1:									return VK_FORMAT_UNDEFINED;
	case GL_STENCIL_INDEX4:									return VK_FORMAT_UNDEFINED;
	case GL_STENCIL_INDEX8:									return VK_FORMAT_S8_UINT;
	case GL_STENCIL_INDEX16:								return VK_FORMAT_UNDEFINED;
	case GL_DEPTH24_STENCIL8:								return VK_FORMAT_D24_UNORM_S8_UINT;
	case GL_DEPTH32F_STENCIL8:								return VK_FORMAT_D32_SFLOAT_S8_UINT;
	case GL_DEPTH32F_STENCIL8_NV:							return VK_FORMAT_D32_SFLOAT_S8_UINT;

	default:												return VK_FORMAT_UNDEFINED;
	}
}



VkFormat
ktxTexture1_GetVkFormat( ktxTexture1* This )
{
    VkFormat vkFormat;

    vkFormat = vkGetFormatFromOpenGLInternalFormat( This->glInternalformat );
    if (vkFormat == VK_FORMAT_UNDEFINED) {
        vkFormat = vkGetFormatFromOpenGLFormat( This->glFormat,
                                                This->glType );
    }
    return vkFormat;
}

//-----------------------------------------------------------------------------
TextureFormat VkToTextureFormat( VkFormat f )
//-----------------------------------------------------------------------------
{
	// Use the fact we have ensured the TextureFormats are sorted with respect to the values of VkFormat, so we can binary search.
	auto foundIt = std::lower_bound( std::begin( sTextureFormatToVk ), std::end( sTextureFormatToVk ), f );
	if (foundIt != std::end( sTextureFormatToVk ) && *foundIt == f)
		return TextureFormat( std::distance( std::begin( sTextureFormatToVk ), foundIt ) );
	return TextureFormat::UNDEFINED;
}

//-----------------------------------------------------------------------------
VkFormat ktxTexture2_GetVkFormat( ktxTexture2* This )
//-----------------------------------------------------------------------------
{
    return (VkFormat) This->vkFormat;
}

//-----------------------------------------------------------------------------
VkFormat ktxTexture_GetVkFormat( ktxTexture* This )
//-----------------------------------------------------------------------------
{
    if (This->classId == ktxTexture2_c)
        return (VkFormat) ktxTexture2_GetVkFormat( (ktxTexture2*)This );
    else
        return (VkFormat) ktxTexture1_GetVkFormat( (ktxTexture1*)This );
}

//-----------------------------------------------------------------------------
TextureFormat ktxTexture_GetTextureFormat( ktxTexture* This )
//-----------------------------------------------------------------------------
{
	return VkToTextureFormat( ktxTexture_GetVkFormat( This ) );
}

//-----------------------------------------------------------------------------
TextureDx12 TextureKtx<Dx12>::LoadKtx(Dx12& graphicsApi, const TextureKtxFileWrapper& fileData, Sampler<Dx12> sampler)
//-----------------------------------------------------------------------------
{
    auto* const pKtxData = GetKtxTexture(fileData);
    if (!pKtxData)
        return {};
    TextureFormat textureFormat = TextureFormat::UNDEFINED;
    if (ktxTexture_NeedsTranscoding(pKtxData) && pKtxData->classId == class_id::ktxTexture2_c)
    {
        auto pKtx2Data = (ktxTexture2* const)pKtxData;
        if (KTX_SUCCESS != ktxTexture2_TranscodeBasis(pKtx2Data, KTX_TTF_RGBA32, (ktx_transcode_flag_bits_e)0))
        {
            return {};
        }

        textureFormat = TextureFormat::R8G8B8A8_UNORM;
    }
    else
    {
        textureFormat = ktxTexture_GetTextureFormat( pKtxData );
        ktxTexture_LoadImageData(pKtxData, nullptr, 0);
    }

    auto& memoryManager = graphicsApi.GetMemoryManager();
    //bool Buffer<Dx12>::Initialize(MemoryManager * pManager, size_t size, BufferUsageFlags bufferUsageFlags, MemoryUsage memoryUsage)

    auto dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
    switch (pKtxData->numDimensions) {
    case 2:
        dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        break;
    case 3:
        dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        break;
    default: break;
    }
    const uint8_t* pSrcData = ktxTexture_GetData(pKtxData);

	bool rgbTorgba = false;
	switch (textureFormat) {
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
		static_assert(uint32_t( TextureFormat::R8G8B8A8_UNORM ) == uint32_t( TextureFormat::R8G8B8_UNORM ) + 14);
		static_assert(uint32_t( TextureFormat::R8G8B8A8_SNORM ) == uint32_t( TextureFormat::R8G8B8_SNORM ) + 14);
		static_assert(uint32_t( TextureFormat::R8G8B8A8_USCALED ) == uint32_t( TextureFormat::R8G8B8_USCALED ) + 14);
		static_assert(uint32_t( TextureFormat::R8G8B8A8_SSCALED ) == uint32_t( TextureFormat::R8G8B8_SSCALED ) + 14);
		static_assert(uint32_t( TextureFormat::R8G8B8A8_UINT ) == uint32_t( TextureFormat::R8G8B8_UINT ) + 14);
		static_assert(uint32_t( TextureFormat::R8G8B8A8_SINT ) == uint32_t( TextureFormat::R8G8B8_SINT ) + 14);
		static_assert(uint32_t( TextureFormat::R8G8B8A8_SRGB ) == uint32_t( TextureFormat::R8G8B8_SRGB ) + 14);
		static_assert(uint32_t( TextureFormat::B8G8R8A8_UNORM ) == uint32_t( TextureFormat::B8G8R8_UNORM ) + 14);
		static_assert(uint32_t( TextureFormat::B8G8R8A8_SNORM ) == uint32_t( TextureFormat::B8G8R8_SNORM ) + 14);
		static_assert(uint32_t( TextureFormat::B8G8R8A8_USCALED ) == uint32_t( TextureFormat::B8G8R8_USCALED ) + 14);
		static_assert(uint32_t( TextureFormat::B8G8R8A8_SSCALED ) == uint32_t( TextureFormat::B8G8R8_SSCALED ) + 14);
		static_assert(uint32_t( TextureFormat::B8G8R8A8_UINT ) == uint32_t( TextureFormat::B8G8R8_UINT ) + 14);
		static_assert(uint32_t( TextureFormat::B8G8R8A8_SINT ) == uint32_t( TextureFormat::B8G8R8_SINT ) + 14);
		static_assert(uint32_t( TextureFormat::B8G8R8A8_SRGB ) == uint32_t( TextureFormat::B8G8R8_SRGB ) + 14);
		textureFormat = TextureFormat( uint32_t( textureFormat ) + 14 );
		rgbTorgba = true;
		break;
	default:
		break;
	}

    const DXGI_FORMAT dxgiFormat = TextureFormatToDx( textureFormat );
    const D3D12_RESOURCE_DESC textureDesc = { .Dimension = dimension,
                                              .Alignment = 0,
                                              .Width = pKtxData->baseWidth,
                                              .Height = pKtxData->baseHeight,
                                              .DepthOrArraySize = (UINT16) pKtxData->baseDepth,
                                              .MipLevels = (uint16_t)pKtxData->numLevels,
                                              .Format = dxgiFormat,
                                              .SampleDesc = {.Count = 1, .Quality = 0 },
                                              .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
                                              .Flags = D3D12_RESOURCE_FLAG_NONE
    };

    D3D12_CLEAR_VALUE clearColor{};
    auto gpuBuffer = memoryManager.CreateImage(textureDesc, MemoryUsage::GpuExclusive, D3D12_RESOURCE_STATE_COPY_DEST, clearColor);

    {
        const size_t numSubresources = textureDesc.DepthOrArraySize * textureDesc.MipLevels;
        auto gpuBufferDesc = gpuBuffer.GetResource()->GetDesc();
        std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> resourceLayouts;
        resourceLayouts.resize(numSubresources);
        size_t resourceDataSize = 0;
        graphicsApi.GetDevice()->GetCopyableFootprints(&gpuBufferDesc, 0/*startSubresource*/, numSubresources, 0, resourceLayouts.data(), nullptr, nullptr, &resourceDataSize);

        //auto stagingBuffer = memoryManager.CreateImage(textureDesc, MemoryUsage::CpuToGpu, D3D12_RESOURCE_STATE_COPY_SOURCE);
        auto stagingBuffer = memoryManager.CreateBuffer(resourceDataSize, BufferUsageFlags::TransferSrc, MemoryUsage::CpuToGpu);
        {
            assert( pKtxData->dataSize <= resourceDataSize );
            auto stagingCpu = memoryManager.Map<uint8_t>( stagingBuffer );

            for (size_t mip = 0; mip < pKtxData->numLevels; ++mip)
            {
                ktx_size_t srcOffset = 0;
                ktxTexture_GetImageOffset(pKtxData, mip, 0/*layer*/, 0/*faceSlice*/, &srcOffset);
                auto* pSrc = pSrcData + srcOffset;

                auto pDestMip = stagingCpu.data() + resourceLayouts[mip].Offset;
                auto pitch = resourceLayouts[mip].Footprint.RowPitch;
                auto mipWidth = resourceLayouts[mip].Footprint.Width;
                auto mipHeight = resourceLayouts[mip].Footprint.Height;
                auto mipDepth = resourceLayouts[mip].Footprint.Depth;

                auto srcPitch = WidthToPitch(dxgiFormat, mipWidth);

                uint32_t slicePitch = pitch * mipHeight;
                for (size_t z = 0; z < mipDepth; ++z)
                {
					auto pDest = pDestMip + slicePitch * z;
					{
						for (auto y = 0; y < mipHeight; ++y)
						{
							if (rgbTorgba)
							{
								auto pS = pSrc;
								auto pD = pDest;
								for (auto x = 0; x < mipWidth; ++x)
								{
									*pD++ = *pS++;
									*pD++ = *pS++;
									*pD++ = *pS++;
									*pD++ = 255;	// fill alpha
								}
							}
							else
								memcpy(pDest, pSrc, pitch/*we may only want to copy the actual width (in bytes), for now copy the entire pitch*/);
							pDest += pitch;
							pSrc += srcPitch;
						}
					}
                }
            }
            memoryManager.Unmap(stagingBuffer, std::move(stagingCpu));
        }
        ID3D12GraphicsCommandList* cmdList = graphicsApi.StartSetupCommandBuffer();
        assert(cmdList);

        for(UINT subResourceIdx = 0; subResourceIdx < numSubresources; ++subResourceIdx)
        {
            const D3D12_TEXTURE_COPY_LOCATION gpuDest{ .pResource = gpuBuffer.GetResource(),
                                                       .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                                                       .SubresourceIndex = subResourceIdx
            };
            const D3D12_TEXTURE_COPY_LOCATION stagingSrc{ .pResource = stagingBuffer.GetResource(),
                                                          .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                                                          .PlacedFootprint = resourceLayouts[subResourceIdx]
            };
            cmdList->CopyTextureRegion(&gpuDest, 0, 0, 0, &stagingSrc, nullptr);
        }
        //cmdList->CopyResource(gpuBuffer.GetResource(), stagingBuffer.GetResource());
        graphicsApi.FinishSetupCommandBuffer(cmdList);

        memoryManager.Destroy(std::move(stagingBuffer));
    }

    // Return the fully formed texture object
    return TextureDx12{ (uint32_t)textureDesc.Width, textureDesc.Height, (uint32_t)textureDesc.DepthOrArraySize, (uint32_t)textureDesc.MipLevels, textureFormat, textureDesc.Layout, std::move(gpuBuffer)/*, sampler*/};
}

TextureDx12 TextureKtx<Dx12>::LoadKtx(Dx12& graphicsApi, AssetManager& assetManager, const char* const pFileName, Sampler<Dx12> sampler)
{
    auto ktxData = LoadFile(assetManager, pFileName);
    if (!ktxData)
        return {};
    return LoadKtx(graphicsApi, ktxData, sampler);
}

TextureKtxFileWrapper TextureKtx<Dx12>::Transcode(TextureKtxFileWrapper&& fileData)
{
    auto* const pKtxData = GetKtxTexture(fileData);
    if (pKtxData != nullptr && ktxTexture_NeedsTranscoding(pKtxData) && pKtxData->classId == class_id::ktxTexture2_c)
    {
        auto pKtx2Data = (ktxTexture2* const)pKtxData;
        if (KTX_SUCCESS != ktxTexture2_TranscodeBasis(pKtx2Data, KTX_TTF_RGBA32, (ktx_transcode_flag_bits_e)0))
        {
            return {};
        }
    }
    return std::move(fileData);
}

static uint32_t WidthToPitch( DXGI_FORMAT dxgiFormat, uint32_t width )
{
    return Dx12::FormatBytesPerPixel( dxgiFormat ) * width;
}

static uint32_t AreaToBytes( DXGI_FORMAT dxgiFormat, uint32_t width, uint32_t height )
{
    return WidthToPitch( dxgiFormat, width) * height;
}

