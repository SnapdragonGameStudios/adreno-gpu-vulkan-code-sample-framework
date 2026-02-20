//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "memory/vulkan/memoryMapped.hpp"
#include "texture.hpp"
#include "vulkan/vulkan.hpp"
#include <cstring>

Texture<Vulkan>::Texture() noexcept
{}

Texture<Vulkan>::Texture(uint32_t width, uint32_t height, uint32_t depth, uint32_t mipLevels, uint32_t firstMip, uint32_t faces, uint32_t firstFace, TextureFormat format, VkImageLayout imageLayout, VkClearValue clearValue, ::Image<Vulkan> image, ::Sampler<Vulkan> sampler, ::ImageView<Vulkan> imageView) noexcept
    : TextureBase()
    , Width(width)
    , Height(height)
    , Depth(depth)
    , MipLevels(mipLevels)
    , Faces(faces)
    , Format(format)
    , ImageLayout(imageLayout)
    , ClearValue(clearValue)
    , Image(std::move(image))
    , Sampler(std::move(sampler))
    , ImageView(std::move(imageView))
{
}

/// @brief Construct Texture from a pre-existing Vulkan image/memory handles.
/// @param image - ownership NOT passed in to this Texture, beware of lifetime issues.
/// @param sampler - ownership passed to this Texture.
/// @param imageView - ownership passed to this Texture.
Texture<Vulkan>::Texture(uint32_t width, uint32_t height, uint32_t depth, uint32_t mipLevels, uint32_t firstMip, uint32_t faces, uint32_t firstFace, TextureFormat format, VkImageLayout imageLayout, VkClearValue clearValue, VkImage image, VkDeviceMemory memory, ::Sampler<Vulkan> sampler, ::ImageView<Vulkan> imageView) noexcept
    : TextureBase()
    , Width(width)
    , Height(height)
    , Depth(depth)
    , MipLevels( mipLevels )
    , FirstMip( firstMip )
    , Faces( faces)
    , FirstFace( firstFace )
    , Format(format)
    , ImageLayout(imageLayout)
    , ClearValue(clearValue)
    , Sampler(std::move(sampler))
    , ImageView(std::move(imageView))
    , Image({image, memory})
{
}

//
// Constructors/move-operators for Texture<Vulkan>.
// Ensures we are not leaking owned members.
//
Texture<Vulkan>::Texture(Texture<Vulkan>&& other) noexcept
{
    *this = std::move(other);
}
Texture<Vulkan>& Texture<Vulkan>::operator=(Texture<Vulkan>&& other) noexcept
{
    if (this != &other)
    {
        // Some of the data can be copied and is ok to leave 'other' alone (move operator just has to ensure the 'other' is in a valid state and can be safely deleted)
        Width = other.Width;
        Height = other.Height;
        Depth = other.Depth;
        MipLevels = other.MipLevels;
        FirstMip = other.FirstMip;
        Faces = other.Faces;
        FirstFace = other.FirstFace;
        Format = other.Format;
        ImageLayout = other.ImageLayout;
        ClearValue = other.ClearValue;
        // Actually transfer ownership from 'other'
        Image = std::move(other.Image);
        Sampler = std::move(other.Sampler);
        ImageView = std::move(other.ImageView);
    }
    return *this;
}

Texture<Vulkan>::~Texture() noexcept
{
    // Asserts to ensure we called Release on this already.
    assert(Image.IsEmpty());
    assert(ImageView.IsEmpty());
    assert(Sampler.IsEmpty());
}

void Texture<Vulkan>::Release(GraphicsApiBase* pGraphicsApi)
{
    auto* pVulkan = static_cast<Vulkan*>(pGraphicsApi);

    ReleaseSampler( *pVulkan, &Sampler );

    ReleaseImageView(*pVulkan, &ImageView);

    ReleaseImage(*pVulkan, &Image);
}

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

constexpr VkImageViewType EnumToVk(ImageViewType t) { return VkImageViewType(int( t )); }
constexpr ImageViewType VkToEnum(VkImageViewType t) { return ImageViewType( int( t ) ); }
static_assert(int(ImageViewType::View1D) == VK_IMAGE_VIEW_TYPE_1D);
static_assert(int( ImageViewType::View2D ) == VK_IMAGE_VIEW_TYPE_2D);
static_assert(int( ImageViewType::View3D ) == VK_IMAGE_VIEW_TYPE_3D);
static_assert(int( ImageViewType::ViewCube ) == VK_IMAGE_VIEW_TYPE_CUBE);
static_assert(int( ImageViewType::View1DArray ) == VK_IMAGE_VIEW_TYPE_1D_ARRAY);
static_assert(int( ImageViewType::View2DArray ) == VK_IMAGE_VIEW_TYPE_2D_ARRAY);
static_assert(int( ImageViewType::ViewCubeArray ) == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY);

static constexpr std::array<VkFormat, 247> sTextureFormatToVk {
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

    VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK, //1000066000
    VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK,
    VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK,
    VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK,
    VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK,
    VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK,
    VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK,
    VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK,
    VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK,
    VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK,
    VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK,
    VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK,
    VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK,
    VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK,

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
    VK_FORMAT_G8_B8R8_2PLANE_444_UNORM,
    VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16,
    VK_FORMAT_G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16,
    VK_FORMAT_G16_B16R16_2PLANE_444_UNORM,
    VK_FORMAT_A4R4G4B4_UNORM_PACK16,
    VK_FORMAT_A4B4G4R4_UNORM_PACK16,
};
#include <algorithm>
// Ensure the sTextureFormatToVk array is in strict numerical order (so we can quickly std::find
consteval bool CheckFormatOrdering() {
    return std::is_sorted(std::begin(sTextureFormatToVk), std::end(sTextureFormatToVk));
}
static_assert(CheckFormatOrdering());


// do some rough checks to ensure our formats are in the order we expect (or else the above lookup doesnt work!)
static_assert(int(TextureFormat::B8G8R8A8_SRGB) == 50);
static_assert(sTextureFormatToVk[int(TextureFormat::B8G8R8A8_SRGB)] == int(VK_FORMAT_B8G8R8A8_SRGB));
static_assert(int(TextureFormat::R32_SFLOAT) == 100);
static_assert(int(TextureFormat::ETC2_R8G8B8A1_SRGB_BLOCK) == 150);
static_assert(sTextureFormatToVk[int(TextureFormat::ETC2_R8G8B8A1_SRGB_BLOCK)] == int(VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK));
static_assert(int(TextureFormat::ASTC_12x12_SRGB_BLOCK) == 184);
static_assert(int(TextureFormat::PVRTC1_2BPP_UNORM_BLOCK_IMG) == 185);
static_assert(sTextureFormatToVk[int(TextureFormat::PVRTC1_2BPP_UNORM_BLOCK_IMG)] == int(VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG));
static_assert(int(TextureFormat::ASTC_4x4_SFLOAT_BLOCK) == 193);
static_assert(sTextureFormatToVk[int(TextureFormat::ASTC_4x4_SFLOAT_BLOCK)] == int(VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK));
static_assert(int(TextureFormat::G8B8G8R8_422_UNORM) == 207);
static_assert(sTextureFormatToVk[int(TextureFormat::G8B8G8R8_422_UNORM)] == int(VK_FORMAT_G8B8G8R8_422_UNORM));
static_assert(int(TextureFormat::EndCount) == 247);

//-----------------------------------------------------------------------------
VkFormat TextureFormatToVk(TextureFormat f)
//-----------------------------------------------------------------------------
{
    return sTextureFormatToVk[uint32_t(f)];
}

//-----------------------------------------------------------------------------
TextureFormat VkToTextureFormat(VkFormat f)
//-----------------------------------------------------------------------------
{
    // Use the fact we have ensured the TextureFormats are sorted with respect to the values of VkFormat, so we can binary search.
    auto foundIt = std::lower_bound(std::begin(sTextureFormatToVk), std::end(sTextureFormatToVk), f);
    if (foundIt != std::end(sTextureFormatToVk) && *foundIt == f)
        return TextureFormat( std::distance( std::begin(sTextureFormatToVk), foundIt ) );
    return TextureFormat::UNDEFINED;
}

//-----------------------------------------------------------------------------
VkSampleCountFlagBits EnumToVk(Msaa msaa)
//-----------------------------------------------------------------------------
{
    return (VkSampleCountFlagBits) msaa;
}

//-----------------------------------------------------------------------------
template<>
Texture<Vulkan> CreateTextureObject<Vulkan>(Vulkan& vulkan, const CreateTexObjectInfo& texInfo, MemoryPool<Vulkan>* pPool)
//-----------------------------------------------------------------------------
{
    VkResult retVal;

    if (texInfo.pName == nullptr)
        LOGI("CreateTextureObject (%dx%d): <No Name>", texInfo.uiWidth, texInfo.uiHeight);
    else
        LOGI("CreateTextureObject (%dx%d): %s", texInfo.uiWidth, texInfo.uiHeight, texInfo.pName);

    VkImageView retImageView = VK_NULL_HANDLE;
    VkImageLayout retImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkClearValue retClearValue{};

    // How this texture object will be used.
    MemoryUsage memoryUsage = MemoryUsage::GpuExclusive;

    VkFormat vkFormat = TextureFormatToVk(texInfo.Format);

    // Create the image...
    VkImageCreateInfo ImageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ImageInfo.flags = 0;
    assert(texInfo.uiDepth != 0);
    ImageInfo.imageType = (texInfo.uiDepth == 1) ? VK_IMAGE_TYPE_2D : VK_IMAGE_TYPE_3D;
    ImageInfo.format = vkFormat;
    ImageInfo.extent.width = texInfo.uiWidth;
    ImageInfo.extent.height = texInfo.uiHeight;
    ImageInfo.extent.depth = texInfo.uiDepth;
    ImageInfo.mipLevels = texInfo.uiMips;
    ImageInfo.arrayLayers = texInfo.uiFaces;
    ImageInfo.samples = EnumToVk(texInfo.Msaa);
    ImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    ImageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    ImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ImageInfo.queueFamilyIndexCount = 0;
    ImageInfo.pQueueFamilyIndices = NULL;
    ImageInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;

    switch (texInfo.TexType)
    {
    case TT_SHADING_RATE_IMAGE:
        ImageInfo.usage = VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        break;
    case TT_CPU_UPDATE:
        ImageInfo.tiling = VK_IMAGE_TILING_LINEAR;
        ImageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        break;
    case TT_NORMAL:
        ImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;// VK_IMAGE_TILING_LINEAR;      // VK_IMAGE_TILING_OPTIMAL
        ImageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        break;
    case TT_SAMPLED_TRANSFERDST:
        ImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        ImageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        break;
    case TT_RENDER_TARGET:
        // If using VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT then tiling MUST be VK_IMAGE_TILING_OPTIMAL
        ImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        ImageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT; // DO NOT enable Storage_Bit (potential performance impact)
        ImageInfo.samples = EnumToVk(texInfo.Msaa);
        break;
    case TT_RENDER_TARGET_WITH_STORAGE:
        // If using VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT then tiling MUST be VK_IMAGE_TILING_OPTIMAL
        // Also enabling Storage may disable some hardware render buffer optimizations
        ImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        assert(texInfo.Msaa == Msaa::Samples1 && texInfo.Format != TextureFormat::R8G8B8A8_SRGB);  //TODO: use Vulkan to determine if this texture buffer can have storeage set, for now assert on formats known to be problematic (msaa or srgb)
        ImageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        ImageInfo.samples = (VkSampleCountFlagBits)texInfo.Msaa;
        break;
    case TT_RENDER_TARGET_TRANSFERSRC:
        ImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        ImageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ImageInfo.samples = (VkSampleCountFlagBits)texInfo.Msaa;
        break;
    case TT_RENDER_TARGET_SUBPASS:
    case TT_RENDER_TARGET_LOCAL_READ_TRANSIENT:
        // Subpass render targets dont need to be backed by memory!
        memoryUsage = MemoryUsage::GpuLazilyAllocated;
        [[fallthrough]];
    case TT_RENDER_TARGET_LOCAL_READ:
        ImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        ImageInfo.usage = VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
        ImageInfo.samples = (VkSampleCountFlagBits)texInfo.Msaa;
        break;
    case TT_COMPUTE_TARGET:
        ImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        ImageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;// | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        break;
    case TT_COMPUTE_STORAGE:
        ImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        ImageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
        break;
    case TT_DEPTH_TARGET:
        // If using VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT then tiling MUST be VK_IMAGE_TILING_OPTIMAL
        ImageInfo.mipLevels = 1;
        ImageInfo.arrayLayers = 1;
        ImageInfo.samples = (VkSampleCountFlagBits)texInfo.Msaa;
        ImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        ImageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT /*| VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT*/;
        if (texInfo.Msaa != Msaa::Samples1)
            ImageInfo.flags |= VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT;
        break;
    case TT_DEPTH_TARGET_LOCAL_READ:
        // If using VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT then tiling MUST be VK_IMAGE_TILING_OPTIMAL
        ImageInfo.mipLevels = 1;
        ImageInfo.arrayLayers = 1;
        ImageInfo.samples = (VkSampleCountFlagBits)texInfo.Msaa;
        ImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        ImageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT /* | VK_IMAGE_USAGE_STORAGE_BIT*/ | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
        if (texInfo.Msaa != Msaa::Samples1)
            ImageInfo.flags |= VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT;
        break;
    default:
        assert(0);
        break;
    }

    if ((texInfo.Flags & TEXTURE_FLAGS::MutableFormat) != 0)
        ImageInfo.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    if ((texInfo.Flags & TEXTURE_FLAGS::ForceLinearTiling) != 0)
        ImageInfo.tiling = VK_IMAGE_TILING_LINEAR;

    // Create the Image
    auto& memoryManager = vulkan.GetMemoryManager();
    auto createImage = [&]() {
        if (pPool && *pPool)
            return memoryManager.CreateImage( *pPool, ImageInfo );
        else
            return memoryManager.CreateImage( ImageInfo, memoryUsage );
    };
    auto vmaImage = createImage();

    if (!vmaImage && memoryUsage == MemoryUsage::GpuLazilyAllocated)
    {
        LOGI("Unable to initialize GpuLazilyAllocated image (probably not supported by GPU hardware).  Falling back to GpuExclusive");
        memoryUsage = MemoryUsage::GpuExclusive;
        ImageInfo.usage &= ~VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
        vmaImage = memoryManager.CreateImage( ImageInfo, memoryUsage );
    }
    if (!vmaImage)
    {
        LOGE("Unable to initialize image (Not from file)");
        return {};
    }

    VkCommandBuffer SetupCmdBuffer = vulkan.StartSetupCommandBuffer();
    switch (texInfo.TexType)
    {
    case TT_SHADING_RATE_IMAGE:
        vulkan.SetImageLayout(vmaImage.GetVkBuffer(), SetupCmdBuffer, VK_IMAGE_ASPECT_COLOR_BIT, ImageInfo.initialLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, (VkPipelineStageFlags)0/*unused param*/, (VkPipelineStageFlags)0/*unused param*/, 0, ImageInfo.mipLevels, 0, ImageInfo.arrayLayers);
        retImageLayout = VK_IMAGE_LAYOUT_GENERAL;
        break;
    case TT_CPU_UPDATE:
        vulkan.SetImageLayout(vmaImage.GetVkBuffer(), SetupCmdBuffer, VK_IMAGE_ASPECT_COLOR_BIT, ImageInfo.initialLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, (VkPipelineStageFlags)0/*unused param*/, (VkPipelineStageFlags)0/*unused param*/, 0, ImageInfo.mipLevels, 0, ImageInfo.arrayLayers);
        retImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        break;
    case TT_SAMPLED_TRANSFERDST:
    case TT_NORMAL:
        vulkan.SetImageLayout(vmaImage.GetVkBuffer(), SetupCmdBuffer, VK_IMAGE_ASPECT_COLOR_BIT, ImageInfo.initialLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, (VkPipelineStageFlags)0/*unused param*/, (VkPipelineStageFlags)0/*unused param*/, 0, ImageInfo.mipLevels, 0, ImageInfo.arrayLayers);
        retImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        break;
    case TT_RENDER_TARGET:
    case TT_RENDER_TARGET_WITH_STORAGE:
        vulkan.SetImageLayout(vmaImage.GetVkBuffer(), SetupCmdBuffer, VK_IMAGE_ASPECT_COLOR_BIT, ImageInfo.initialLayout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, (VkPipelineStageFlags)0/*unused param*/, (VkPipelineStageFlags)0/*unused param*/, 0, 1, 0, 1);
        retImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        break;
    case TT_RENDER_TARGET_TRANSFERSRC:
        vulkan.SetImageLayout(vmaImage.GetVkBuffer(), SetupCmdBuffer, VK_IMAGE_ASPECT_COLOR_BIT, ImageInfo.initialLayout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, (VkPipelineStageFlags)0/*unused param*/, (VkPipelineStageFlags)0/*unused param*/, 0, 1, 0, 1);
        retImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        break;
    case TT_RENDER_TARGET_LOCAL_READ:
    case TT_RENDER_TARGET_LOCAL_READ_TRANSIENT:
        retImageLayout = VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR;
        break;
    case TT_RENDER_TARGET_SUBPASS:
        retImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        break;
    case TT_COMPUTE_TARGET:
    case TT_COMPUTE_STORAGE:
        vulkan.SetImageLayout(vmaImage.GetVkBuffer(), SetupCmdBuffer, VK_IMAGE_ASPECT_COLOR_BIT, ImageInfo.initialLayout, VK_IMAGE_LAYOUT_GENERAL, (VkPipelineStageFlags)0/*unused param*/, (VkPipelineStageFlags)0/*unused param*/, 0, ImageInfo.mipLevels, 0, ImageInfo.arrayLayers);
        retImageLayout = VK_IMAGE_LAYOUT_GENERAL;
        break;
    case TT_DEPTH_TARGET_LOCAL_READ:
        retImageLayout = VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR;
        memoryUsage = MemoryUsage::GpuLazilyAllocated;
        [[fallthrough]];
    case TT_DEPTH_TARGET:
        if (FormatHasStencil( texInfo.Format))
        {
            // Can have depth and stencil flag
            vulkan.SetImageLayout(vmaImage.GetVkBuffer(), SetupCmdBuffer, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, ImageInfo.initialLayout, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, (VkPipelineStageFlags)0/*unused param*/, (VkPipelineStageFlags)0/*unused param*/, 0, 1, 0, 1);
            retClearValue.depthStencil.depth = 1.0f;
        }
        else if (FormatHasDepth(texInfo.Format))
        {
            // Only has the depth flag set
            vulkan.SetImageLayout(vmaImage.GetVkBuffer(), SetupCmdBuffer, VK_IMAGE_ASPECT_DEPTH_BIT, ImageInfo.initialLayout, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, (VkPipelineStageFlags)0/*unused param*/, (VkPipelineStageFlags)0/*unused param*/, 0, 1, 0, 1);
            retClearValue.depthStencil.depth = 1.0f;
        }
        else
        {
            LOGE("Unhandled depth format!!!");
        }
        if(retImageLayout != VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR)
            retImageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        break;
    default:
        assert(0);
        break;
    }

    vulkan.FinishSetupCommandBuffer(SetupCmdBuffer);

    // Create the Image wrapper (containing the vmaimage)
    Image<Vulkan> Image{ std::move( vmaImage ) };
    vulkan.SetDebugObjectName( Image.GetVkImage(), texInfo.pName );

    // ... and an ImageView
    const ImageViewType ViewType = (texInfo.uiDepth == 1) ? ImageViewType::View2D : ImageViewType::View3D; // <== No support for VK_IMAGE_VIEW_TYPE_CUBE
    VkImageViewCreateInfo ImageViewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    ImageViewInfo.flags = 0;
    ImageViewInfo.image = Image.GetVkImage();
    ImageViewInfo.viewType = EnumToVk( ViewType );
    ImageViewInfo.format = ImageInfo.format;
    ImageViewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY; // VK_COMPONENT_SWIZZLE_R;
    ImageViewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY; // VK_COMPONENT_SWIZZLE_G;
    ImageViewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY; // VK_COMPONENT_SWIZZLE_B;
    ImageViewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY; // VK_COMPONENT_SWIZZLE_A;
    ImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ImageViewInfo.subresourceRange.baseMipLevel = 0;
    ImageViewInfo.subresourceRange.levelCount = ImageInfo.mipLevels;
    ImageViewInfo.subresourceRange.baseArrayLayer = 0;
    ImageViewInfo.subresourceRange.layerCount = ImageInfo.arrayLayers;

    SamplerAddressMode SamplerMode = texInfo.SamplerMode;

    SamplerBorderColor BorderColor = SamplerBorderColor::TransparentBlackFloat;

    switch (texInfo.TexType)
    {
    case TT_SHADING_RATE_IMAGE:
    case TT_CPU_UPDATE:
    case TT_NORMAL:
    case TT_SAMPLED_TRANSFERDST:
    case TT_RENDER_TARGET:
    case TT_RENDER_TARGET_WITH_STORAGE:
    case TT_RENDER_TARGET_TRANSFERSRC:
    case TT_RENDER_TARGET_SUBPASS:
    case TT_COMPUTE_TARGET:
    case TT_COMPUTE_STORAGE:
        ImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        break;
    case TT_RENDER_TARGET_LOCAL_READ:
    case TT_RENDER_TARGET_LOCAL_READ_TRANSIENT:
        ImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT;
        break;
    case TT_DEPTH_TARGET:
    case TT_DEPTH_TARGET_LOCAL_READ:
        ImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        SamplerMode = (SamplerMode == SamplerAddressMode::Undefined) ? SamplerAddressMode::ClampBorder : SamplerMode;    // default to ClampBorder
        BorderColor = SamplerBorderColor::OpaqueWhiteFloat;
        break;
    default:
        assert(0);
        break;
    }
    SamplerMode = (SamplerMode == SamplerAddressMode::Undefined) ? SamplerAddressMode::ClampEdge : SamplerMode;         // default to ClampEdge

    retVal = vkCreateImageView(vulkan.m_VulkanDevice, &ImageViewInfo, NULL, &retImageView);
    if (!CheckVkError("vkCreateImageView()", retVal))
    {
        return {};
    }
    ImageView<Vulkan> ImageView{retImageView, ImageViewInfo.viewType};

    // LOGI("vkCreateImageView: %s -> %p", pName, retImageView);

    // Need a sampler...
    SamplerFilter FilterMode = texInfo.FilterMode;
    if (FilterMode == SamplerFilter::Undefined)
    {
        const auto& FormatProperties = vulkan.GetFormatProperties(vkFormat);
        auto TilingFeatures = (ImageInfo.tiling == VK_IMAGE_TILING_LINEAR) ? FormatProperties.linearTilingFeatures : FormatProperties.optimalTilingFeatures;
        if ((TilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0)
            FilterMode = SamplerFilter::Linear;
        else
            FilterMode = SamplerFilter::Nearest;
    }
    SamplerVulkan Sampler = CreateSampler(vulkan, SamplerMode, FilterMode, BorderColor, 0.0f);
    if (Sampler.IsEmpty())
    {
        return {};
    }

    return{ texInfo.uiWidth, texInfo.uiHeight, texInfo.uiDepth, texInfo.uiMips, 0/*first mip*/, texInfo.uiFaces, 0/*first face*/, texInfo.Format, retImageLayout, retClearValue, std::move(Image), std::move(Sampler), std::move(ImageView)};
}


//-----------------------------------------------------------------------------
template<>
Texture<Vulkan> CreateTextureObjectView( Vulkan& vulkan, const Texture<Vulkan>& original, TextureFormat viewFormat )
//-----------------------------------------------------------------------------
{
    auto imageView = CreateImageView( vulkan, original.Image, viewFormat, original.MipLevels, original.FirstMip, original.Faces, original.FirstFace, original.ImageView.GetImageViewType() );
    if (imageView.IsEmpty())
    {
        return {};
    }

    SamplerVulkan sampler = CreateSampler( vulkan, SamplerAddressMode::ClampEdge, SamplerFilter::Linear, SamplerBorderColor::TransparentBlackFloat, 0.0f );
    if (sampler.IsEmpty())
    {
        return {};
    }

    return { original.Width, original.Height, original.Depth, original.MipLevels, original.FirstMip, original.Faces, original.FirstFace, original.Format, original.GetVkImageLayout(), original.GetVkClearValue(), original.GetVkImage(), VK_NULL_HANDLE, std::move(sampler), std::move(imageView) };
}


//-----------------------------------------------------------------------------
Texture<Vulkan> CreateTextureObjectView( Vulkan& vulkan, const Texture<Vulkan>& original, uint32_t mipLevels, uint32_t firstMip, uint32_t faces, uint32_t firstFace )
//-----------------------------------------------------------------------------
{
    ImageViewVulkan imageView = CreateImageView( vulkan, original.Image, original.Format, mipLevels, firstMip, faces, firstFace, original.ImageView.GetImageViewType() );
    if (imageView.IsEmpty())
    {
        return {};
    }
    SamplerVulkan sampler = CreateSampler( vulkan, SamplerAddressMode::ClampEdge, SamplerFilter::Linear, SamplerBorderColor::TransparentBlackFloat, 0.0f );
    if (sampler.IsEmpty())
    {
        return {};
    }
    return {original.Width, original.Height, original.Depth, original.MipLevels, original.FirstMip, original.Faces, original.FirstFace, original.Format, original.GetVkImageLayout(), original.GetVkClearValue(), original.GetVkImage(), VK_NULL_HANDLE, std::move( sampler ), std::move( imageView )};
}


//-----------------------------------------------------------------------------
template<>
Texture<Vulkan> CreateTextureFromBuffer<Vulkan>( Vulkan& vulkan, const void* pData, size_t DataSize, uint32_t Width, uint32_t Height, uint32_t Depth, TextureFormat Format, SamplerAddressMode SamplerMode, SamplerFilter Filter, const char* pName )
//-----------------------------------------------------------------------------
{
    if (pName == nullptr)
        LOGI( "CreateTextureFromBuffer (%dx%d): <No Name>", Width, Height );
    else
        LOGI( "CreateTextureFromBuffer (%dx%d): %s", Width, Height, pName );

    auto& memoryManager = vulkan.GetMemoryManager();

    uint32_t Faces = 1;
    uint32_t MipLevels = 1;
    VkFormat vkFormat = TextureFormatToVk( Format );
    VkImageUsageFlags FinalUsage = ( VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT );
    VkImageLayout FinalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Image creation info.  Will change below based on need
    VkImageCreateInfo ImageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ImageInfo.flags = 0;
    ImageInfo.imageType = VK_IMAGE_TYPE_2D;
    ImageInfo.format = vkFormat;
    ImageInfo.extent.width = Width;
    ImageInfo.extent.height = Height;
    ImageInfo.extent.depth = Depth;
    ImageInfo.mipLevels = 1;	// Set to 1 since making each level.  Later will be set to MipLevels;
    ImageInfo.arrayLayers = 1;  // Set to 6 for cube maps
    ImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    ImageInfo.tiling = VK_IMAGE_TILING_LINEAR;
    ImageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ImageInfo.queueFamilyIndexCount = 0;
    ImageInfo.pQueueFamilyIndices = NULL;
    ImageInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;   // VK_IMAGE_LAYOUT_UNDEFINED says data not guaranteed to be preserved when changing state

    struct CubeFace
    {
        CubeFace( MemoryManager<Vulkan>& _memoryManager, uint32_t depthSlices, uint32_t mips ) : memoryManager(_memoryManager), mipsPerDepthSlice( mips ), images( depthSlices* mips )
        {}
        ~CubeFace() { for (auto& image : images) memoryManager.Destroy( std::move( image ) ); }
        CubeFace& operator=( CubeFace&& ) noexcept { assert( 0 ); return *this; }  //define but expect to never be called
        CubeFace(CubeFace&&other) noexcept : memoryManager(other.memoryManager), mipsPerDepthSlice(other.mipsPerDepthSlice), images(std::move(other.images)){}
        const auto& GetImage( uint32_t depth, uint32_t mip ) const { return images[depth * mipsPerDepthSlice]; }
        auto& GetImage( uint32_t depth, uint32_t mip ) { return images[depth * mipsPerDepthSlice + mip]; }
        auto& CreateImage( uint32_t depth, uint32_t mip, const VkImageCreateInfo& imageInfo, MemoryUsage usage ) {
            auto& image = GetImage( depth, mip );
            image.operator=( std::move( memoryManager.CreateImage( imageInfo, usage ) ) );
            return image;
        }
        MemoryManager<Vulkan>& memoryManager;
        const uint32_t mipsPerDepthSlice;
        std::vector<MemoryAllocatedBuffer<Vulkan, VkImage>> images;
    };

    // Allocate the depth slices (and contained mips) for each face
    std::vector<CubeFace> cubeFaces;
    cubeFaces.reserve( Faces );
    for (uint32_t WhichFace = 0; WhichFace < Faces; ++WhichFace)
    {
        cubeFaces.emplace_back( memoryManager, Depth, MipLevels );
    }

    // Create and copy mip levels

    // Need the setup command buffer for loading images
    VkCommandBuffer SetupCmdBuffer = vulkan.StartSetupCommandBuffer();

    uint32_t formatBytesPerPixel = 4;
    if (vkFormat == VK_FORMAT_R32G32B32A32_SFLOAT)
        formatBytesPerPixel = 16;
    else if (vkFormat >= VK_FORMAT_R8_UNORM && vkFormat <= VK_FORMAT_R8_SRGB)
        formatBytesPerPixel = 1;
    else if (vkFormat >= VK_FORMAT_R8G8_UNORM && vkFormat <= VK_FORMAT_R8G8_SRGB)
        formatBytesPerPixel = 2;
    else if (vkFormat >= VK_FORMAT_R16_UNORM && vkFormat <= VK_FORMAT_R16_SFLOAT)
        formatBytesPerPixel = 2;

    const uint8_t* pData8 = static_cast<const uint8_t*>( pData );

    // Load separate cube map faces into linear tiled textures (for copying only).
    // Split into faces and 2d depth slices because linear textures have extremely limited format/size restrictions.
    for (uint32_t WhichFace = 0; WhichFace < Faces; WhichFace++)
    {
        for (uint32_t WhichDepth = 0; WhichDepth < Depth; WhichDepth++)
        {
            for (uint32_t WhichMip = 0; WhichMip < MipLevels; WhichMip++)
            {
                ImageInfo.extent.width = Width;
                ImageInfo.extent.height = Height;
                ImageInfo.extent.depth = 1;

                MemoryAllocatedBuffer<Vulkan, VkImage>& faceImage = cubeFaces[WhichFace].CreateImage( WhichDepth, WhichMip, ImageInfo, MemoryUsage::CpuToGpu );
                if (!faceImage)
                {
                    LOGE( "CreateTextureFromBuffer: Unable to initialize mip %d of face %d", WhichMip + 1, WhichFace + 1 );
                    return {};
                }

                // ... copy texture data into the image
                VkImageSubresource SubresInfo = {};
                SubresInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                SubresInfo.mipLevel = 0;
                SubresInfo.arrayLayer = 0;

                VkSubresourceLayout SubResLayout = {};
                vkGetImageSubresourceLayout( vulkan.m_VulkanDevice, faceImage.GetVkBuffer(), &SubresInfo, &SubResLayout );

                {
                    auto mappedMemory = memoryManager.Map<uint8_t>( faceImage );
                    if (SubResLayout.rowPitch == Width * formatBytesPerPixel)
                    {
                        memcpy( mappedMemory.data(), pData8, SubResLayout.size );
                        pData8 += Width * Height * formatBytesPerPixel;
                    }
                    else
                    {
                        uint8_t* pDest = mappedMemory.data();
                        for (uint32_t h = 0; h < Height; ++h)
                        {
                            memcpy( pDest, pData8, Width * formatBytesPerPixel );
                            pDest += SubResLayout.rowPitch;
                            pData8 += Width * formatBytesPerPixel;
                        }
                    }
                    memoryManager.Unmap( faceImage, std::move( mappedMemory ) );
                }

                // Image barrier for linear image (base)
                // Linear image will be used as a source for the copy
                VkPipelineStageFlags srcMask = 0;
                VkPipelineStageFlags dstMask = 1;
                uint32_t baseMipLevel = 0;
                uint32_t mipLevelCount = 1;

                vulkan.SetImageLayout( faceImage.GetVkBuffer(),
                    SetupCmdBuffer,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_PREINITIALIZED,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    srcMask,
                    dstMask,
                    baseMipLevel,
                    mipLevelCount );
            }   // Each mipmap in each depth slice
        }   // Each depth slice in each face
    }   // Each Face

    // Transfer cube map faces to optimal tiling

    // Now that creating the whole thing we need the correct size and type
    ImageInfo.extent.width = Width;
    ImageInfo.extent.height = Height;
    ImageInfo.extent.depth = Depth;
    ImageInfo.imageType = Depth == 1 ? VK_IMAGE_TYPE_2D : VK_IMAGE_TYPE_3D;

    // Now that we are done creating single images, need to create all mip levels
    ImageInfo.mipLevels = MipLevels;

    // Setup texture as blit target with optimal tiling
    ImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    ImageInfo.usage = FinalUsage;

    if (Faces == 6)
        ImageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

    ImageInfo.arrayLayers = Faces;

    // Clear value (probably unused)
    VkClearValue ClearValue{};

    // Create the 'final' image
    auto FinalVmaImage = memoryManager.CreateImage( ImageInfo, MemoryUsage::GpuExclusive );
    if (!FinalVmaImage)
    {
        LOGE( "CreateTextureFromBuffer: Unable to initialize texture image" );
        return {};
    }

    // Image barrier for optimal image (target)
    // Optimal image will be used as destination for the copy
    VkPipelineStageFlags srcMask = 0;
    VkPipelineStageFlags dstMask = 1;
    uint32_t baseMipLevel = 0;
    uint32_t mipLevelCount = MipLevels;
    uint32_t baseLayer = 0;
    uint32_t layerCount = Faces;

    vulkan.SetImageLayout( FinalVmaImage.GetVkBuffer(),
        SetupCmdBuffer,
        VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_PREINITIALIZED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        srcMask,
        dstMask,
        baseMipLevel,
        mipLevelCount,
        baseLayer,
        layerCount );

    // Copy cube map faces one by one
    // Vulkan spec says the order is +X, -X, +Y, -Y, +Z, -Z
    for (uint32_t WhichFace = 0; WhichFace < Faces; WhichFace++)
    {
        // Copy depth slices one at a time
        for (uint32_t WhichDepth = 0; WhichDepth < Depth; WhichDepth++)
        {
            for (uint32_t WhichMip = 0; WhichMip < MipLevels; WhichMip++)
            {
                // Copy region for image blit
                VkImageCopy copyRegion = {};

                // Source is always base level because we have an image for each face and each mip
                copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                copyRegion.srcSubresource.baseArrayLayer = 0;
                copyRegion.srcSubresource.mipLevel = 0;
                copyRegion.srcSubresource.layerCount = 1;
                copyRegion.srcOffset.x = 0;
                copyRegion.srcOffset.y = 0;
                copyRegion.srcOffset.z = 0;

                // Source is the section of the main image
                copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                copyRegion.dstSubresource.baseArrayLayer = WhichFace;
                copyRegion.dstSubresource.mipLevel = WhichMip;
                copyRegion.dstSubresource.layerCount = 1;
                copyRegion.dstOffset.x = 0;
                copyRegion.dstOffset.y = 0;
                copyRegion.dstOffset.z = WhichDepth;

                // Size is the size of this mip
                copyRegion.extent.width = Width;
                copyRegion.extent.height = Height;
                copyRegion.extent.depth = 1;

                // Put image copy into command buffer
                vkCmdCopyImage(
                    SetupCmdBuffer,
                    cubeFaces[WhichFace].GetImage( WhichDepth, WhichMip ).GetVkBuffer(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    FinalVmaImage.GetVkBuffer(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &copyRegion );
            }   // Each mipmap in each depth slice
        }   // Each depth slice in each face
    }   // Each Face

    // Change texture image layout to the 'final' settings now we are done transferring
    srcMask = 0;
    dstMask = 1;
    baseMipLevel = 0;
    mipLevelCount = MipLevels;
    baseLayer = 0;
    layerCount = Faces;

    vulkan.SetImageLayout( FinalVmaImage.GetVkBuffer(), SetupCmdBuffer, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, FinalLayout,
        srcMask,
        dstMask,
        baseMipLevel,
        mipLevelCount,
        baseLayer,
        layerCount );

    // Submit the command buffer we have been working on
    vulkan.FinishSetupCommandBuffer( SetupCmdBuffer );

    // Create the Image wrapper (takes ownership of FinalVmaImage)
    Image<Vulkan> Image{ std::move(FinalVmaImage) };

    // Need a sampler...
    SamplerVulkan sampler = CreateSampler( vulkan, SamplerMode, Filter, SamplerBorderColor::TransparentBlackFloat, 0.0f );
    if (sampler.IsEmpty())
    {
        return {};
    }

    ImageViewType viewType;
    if (Depth > 1)
        viewType = ImageViewType::View3D;
    else if (Faces == 6)
        viewType = ImageViewType::ViewCube;
    else
        viewType = ImageViewType::View2D;
    ImageViewVulkan imageView = CreateImageView( vulkan, Image, Format, MipLevels, baseMipLevel, layerCount, baseLayer, viewType );
    if (imageView.IsEmpty())
    {
        ReleaseSampler(vulkan, &sampler);
        return {};
    }

    // Set the return values
    return { Width, Height, Depth, MipLevels, 0/*firstmip*/, Faces, 0/*firstface*/, Format, FinalLayout, ClearValue, std::move(Image), std::move(sampler), std::move(imageView)};
}

//-----------------------------------------------------------------------------
template<>
void ReleaseTexture<Vulkan>( Vulkan& vulkan, Texture<Vulkan>* pTexture )
//-----------------------------------------------------------------------------
{
    if (!pTexture)
        return;
    pTexture->Release( &vulkan );
    *pTexture = Texture<Vulkan>{};    // destroy and clear
}


//-----------------------------------------------------------------------------
Image<Vulkan>::Image( MemoryAllocatedBuffer<Vulkan, VkImage> vmaImage ) noexcept : m_VmaImage( std::move(vmaImage) )
//-----------------------------------------------------------------------------
{}

//-----------------------------------------------------------------------------
Image<Vulkan>::Image( VkImage image, VkDeviceMemory memory ) noexcept : m_VmaImage(), m_Image(image)/*no change of ownership*/, m_Memory(memory)/*no change of ownership*/
//-----------------------------------------------------------------------------
{}

//-----------------------------------------------------------------------------
Image<Vulkan>::~Image()
//-----------------------------------------------------------------------------
{
    assert( !m_VmaImage );    // expecting ReleaseImage to have been called already (and so this is empty)
    m_Image = VK_NULL_HANDLE;
    m_Memory = VK_NULL_HANDLE;
}

//-----------------------------------------------------------------------------
template<>
void ReleaseImage<Vulkan>( Vulkan& vulkan, Image<Vulkan>* pImage )
//-----------------------------------------------------------------------------
{
    if (pImage->m_VmaImage)
        vulkan.GetMemoryManager().Destroy( std::move( pImage->m_VmaImage ) );

    // We dont own Image or Memory, so just clear them!
    pImage->m_Image = VK_NULL_HANDLE;
    pImage->m_Memory = VK_NULL_HANDLE;

    *pImage = Image<Vulkan>{};    // destroy and clear
}


//-----------------------------------------------------------------------------
ImageView<Vulkan>::ImageView( VkImageView imageView, VkImageViewType viewType ) noexcept : m_ImageView( imageView ), m_ImageViewType( VkToEnum(viewType) )
//-----------------------------------------------------------------------------
{}

//-----------------------------------------------------------------------------
ImageView<Vulkan>::~ImageView()
//-----------------------------------------------------------------------------
{
    assert( m_ImageView == VK_NULL_HANDLE );
}

//-----------------------------------------------------------------------------
/// Create a vulkan image view object
template<>
ImageView<Vulkan> CreateImageView<Vulkan>(Vulkan& vulkan, const Image<Vulkan>& image, TextureFormat format, uint32_t numMipLevels, uint32_t baseMipLevel, uint32_t numFaces, uint32_t baseFace, ImageViewType viewType )
//-----------------------------------------------------------------------------
{
    const bool hasDepth = FormatHasDepth( format );
    const bool hasStencil = FormatHasStencil( format );
    VkImageViewCreateInfo ImageViewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    ImageViewInfo.flags = 0;
    ImageViewInfo.image = image.GetVkImage();
    ImageViewInfo.viewType = EnumToVk(viewType);

    ImageViewInfo.format = TextureFormatToVk(format);
    ImageViewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    ImageViewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    ImageViewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    ImageViewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    if (hasDepth)
        ImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    else if (hasStencil)//only depth OR stencil OR color should be set!
        ImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
    else
        ImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ImageViewInfo.subresourceRange.baseMipLevel = baseMipLevel;
    ImageViewInfo.subresourceRange.levelCount = numMipLevels;
    ImageViewInfo.subresourceRange.baseArrayLayer = baseFace;
    ImageViewInfo.subresourceRange.layerCount = numFaces;

    VkImageView imageView;
    auto retVal = vkCreateImageView(vulkan.m_VulkanDevice, &ImageViewInfo, NULL, &imageView);
    if (!CheckVkError("vkCreateImageView()", retVal))
    {
        return {};
    }
    return { imageView, ImageViewInfo.viewType };
}

//-----------------------------------------------------------------------------
template<>
void ReleaseImageView( Vulkan& vulkan, ImageView<Vulkan>* pImageView )
//-----------------------------------------------------------------------------
{
    if (!pImageView || pImageView->IsEmpty())
        return;
    vkDestroyImageView( vulkan.m_VulkanDevice, pImageView->m_ImageView, nullptr );
    pImageView->m_ImageView = VK_NULL_HANDLE;
    pImageView->m_ImageViewType = ImageViewType::View1D;
}

//-----------------------------------------------------------------------------
std::vector<Texture<Vulkan>> MakeTextureMipViews( Vulkan& vulkan, const Texture<Vulkan>& source, uint32_t maxMips )
//-----------------------------------------------------------------------------
{
    TextureVulkan               m_ReducedZBuffer;           ///< half sized 'depth' buffer (half in each dimension of original ZBuffer size) with mips.  Populated by the reduce.
    std::vector<TextureVulkan>  mips;

    const auto numMips = std::min( source.MipLevels, maxMips );
    mips.reserve( numMips );

    CreateSamplerObjectInfo samplerCreateInfo{
        .Mode = SamplerAddressMode::ClampEdge,
        .Filter = SamplerFilter::Nearest
    };
    for (int mipLevel = 0; mipLevel < numMips; ++mipLevel)
    {
        // Create an imageview and sampler to point to this mip.  Assigned/stored in a TextureVulkan per mip (all pointing to the same VkImage).
        samplerCreateInfo.MinLod = float( mipLevel );
        samplerCreateInfo.MaxLod = float( mipLevel );

        auto imageView = CreateImageView( vulkan, source.Image, source.Format, 1, mipLevel, source.Faces, 0, source.ImageView.GetImageViewType() );
        Sampler<Vulkan> sampler = CreateSampler( vulkan, samplerCreateInfo );
        mips.emplace_back( source.Width, source.Height, source.Depth, 1, mipLevel, 1, 0, source.Format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, source.GetVkClearValue(), source.GetVkImage(), (VkDeviceMemory)VK_NULL_HANDLE, std::move(sampler), std::move(imageView));
    }
    return mips;
}
