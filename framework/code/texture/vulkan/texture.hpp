// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include "../texture.hpp"
#include "sampler.hpp"
#include "memory/vulkan/memoryMapped.hpp"
#include "vulkan/refHandle.hpp"
#include <volk/volk.h>
#include <vector>

// Forward declarations
class Vulkan;
template<typename T_GFXAPI, typename T_BUFFER> class MemoryAllocatedBuffer;
template<typename T_GFXAPI> class MemoryPool;
using TextureVulkan = Texture<Vulkan>;
using ImageVulkan = Image<Vulkan>;
using ImageViewVulkan = ImageView<Vulkan>;
using SamplerVulkan = Sampler<Vulkan>;

struct TextureVulkanProperties
{
    VkImageUsageFlags    Usage       = 0;
    VkImageTiling        Tiling      = VK_IMAGE_TILING_OPTIMAL;
    VkSampleCountFlagBits Samples    = VK_SAMPLE_COUNT_1_BIT;
    VkImageCreateFlags   CreateFlags = 0;
    VkImageAspectFlags   AspectMask  = 0;
    TEXTURE_TYPE         TextureType = TEXTURE_TYPE::TT_NORMAL;
};

/// @brief Convert from our TextureFormat to Vulkan's VkFormat
extern VkFormat TextureFormatToVk(TextureFormat);
/// @brief Convert from Vulkan's VkFormat to our TextureFormat.
extern TextureFormat VkToTextureFormat(VkFormat);
/// @brief Convert from our MSaa enum to Vulkan's VkSampleCountFlagBits
extern VkSampleCountFlagBits EnumToVk(Msaa msaa);


/// @brief Template specialization of imageview container for Vulkan graphics api.
template<>
class ImageView<Vulkan> final : public ImageViewBase
{
public:
    ImageView() noexcept : m_ImageView( VK_NULL_HANDLE ), m_ImageViewType( ImageViewType::View1D ) {};
    ImageView( VkImageView imageView, VkImageViewType viewType ) noexcept;
    ImageView( ImageView<Vulkan>&& src ) noexcept {
        *this = std::move( src );
    }
    ImageView& operator=( ImageView<Vulkan>&& src ) noexcept {
        m_ImageView = src.m_ImageView;
        src.m_ImageView = VK_NULL_HANDLE;
        m_ImageViewType = src.m_ImageViewType;
        src.m_ImageViewType = ImageViewType::View1D;
        return *this;
    }
    ~ImageView();
    auto GetImageViewType() const { return m_ImageViewType; }
    auto GetVkImageView() const { return m_ImageView; }
    bool IsEmpty() const { return m_ImageView == VK_NULL_HANDLE; }

private:
    friend void ReleaseImageView<Vulkan>( Vulkan& vulkan, ImageView<Vulkan>* pImageView );
    VkImageView     m_ImageView;
    ImageViewType   m_ImageViewType;
};


/// @brief Template specialization of image container for Vulkan graphics api.
template<>
class Image<Vulkan> final : public ImageBase
{
public:
    Image() noexcept {};

    /// @brief Construct Image from a pre-existing vmaImage.
    /// @param vmaImage - ownership passed to this Image.
    Image( MemoryAllocatedBuffer<Vulkan, VkImage> vmaImage ) noexcept;

    /// @brief Construct Image from a pre-existing Vulkan image/memory handles.
    /// @param image - ownership NOT passed in to this Image, beware of lifetime issues.
    /// @param memory - ownership NOT passed to this Image, beware of lifetime issues.
    Image( VkImage image, VkDeviceMemory memory ) noexcept;

    Image( Image<Vulkan>&& src ) noexcept {
        *this = std::move( src );
    }
    Image& operator=( Image<Vulkan>&& src ) noexcept
    {
        m_VmaImage = std::move(src.m_VmaImage );
        m_Image = src.m_Image;
        src.m_Image = VK_NULL_HANDLE;
        m_Memory = src.m_Memory;
        src.m_Memory = VK_NULL_HANDLE;
        return *this;
    }
    ~Image();
    VkImage	GetVkImage() const { return m_VmaImage ? m_VmaImage.GetVkBuffer() : m_Image; }
    const auto& GetMemoryAllocation() const { return m_VmaImage.GetMemoryAllocation(); }
    bool IsEmpty() const { return !m_VmaImage && m_Image == VK_NULL_HANDLE; }

    //Image Copy();

private:
    friend void ReleaseImage<Vulkan>( Vulkan& vulkan, Image<Vulkan>* pImage );

    // Managed memory buffer allocation and VkImage
    MemoryAllocatedBuffer<Vulkan, VkImage>  m_VmaImage;

    // Needed for functions handling own memory (i.e. AndroidHardwareBuffers)
    VkImage                                 m_Image = VK_NULL_HANDLE;
    VkDeviceMemory                          m_Memory = VK_NULL_HANDLE;
};


/// @brief Template specialization of texture container for Vulkan graphics api.
template<>
class Texture<Vulkan> final : public TextureBase
{
public:
    Texture() noexcept;
    Texture(const Texture<Vulkan>&) = delete;
    Texture& operator=(const Texture<Vulkan>&) = delete;
    Texture(Texture<Vulkan>&&) noexcept;
    Texture& operator=(Texture<Vulkan>&&) noexcept;
    ~Texture() noexcept;

    operator bool() const { return !IsEmpty(); }

    /// @brief Construct Texture from a pre-existing vmaImage.
    /// @param vmaImage - ownership passed to this Texture.
    /// @param sampler - ownership passed to this Texture.
    /// @param imageView - ownership passed to this Texture.
    Texture(uint32_t width, uint32_t height, uint32_t depth, uint32_t mipLevels, uint32_t firstMip, uint32_t faces, uint32_t firstFace, TextureFormat, VkImageLayout imageLayout, VkClearValue clearValue, ::Image<Vulkan> image, ::Sampler<Vulkan> sampler, ::ImageView<Vulkan> imageView) noexcept;

    /// @brief Construct Texture from a pre-existing Vulkan image/memory handles.
    /// @param image - ownership NOT passed in to this Texture, beware of lifetime issues.
    /// @param sampler - ownership passed to this Texture.
    /// @param imageView - ownership passed to this Texture.
    Texture(uint32_t width, uint32_t height, uint32_t depth, uint32_t mipLevels, uint32_t firstMip, uint32_t faces, uint32_t firstFace, TextureFormat format, VkImageLayout imageLayout, VkClearValue clearValue, VkImage image, VkDeviceMemory memory, Sampler<Vulkan> sampler, ::ImageView<Vulkan> imageView) noexcept;

    void Release(GraphicsApiBase* pGraphicsApi) override;

    VkImage				  GetVkImage() const { return Image.GetVkImage(); }
    VkDescriptorImageInfo GetVkDescriptorImageInfo() const { return { Sampler.GetVkSampler(), ImageView.GetVkImageView(), ImageLayout }; }
    VkImageLayout		  GetVkImageLayout() const { return ImageLayout; }
    VkSampler			  GetVkSampler() const { return Sampler.GetVkSampler(); }
    VkImageView			  GetVkImageView() const { return ImageView.GetVkImageView(); }
    VkClearValue          GetVkClearValue() const { return ClearValue; }
    const TextureVulkanProperties& GetProperties() const { return Properties; }
    void                  SetProperties(const TextureVulkanProperties& properties) { Properties = properties; }
    bool				  IsEmpty() const { return Image.IsEmpty() || Sampler.IsEmpty(); }

    uint32_t  Width = 0;
    uint32_t  Height = 0;
    uint32_t  Depth = 0;
    uint32_t  MipLevels = 0;
    uint32_t  FirstMip = 0;
    uint32_t  Faces = 0;
    uint32_t  FirstFace = 0;
    TextureFormat  Format = TextureFormat::UNDEFINED;
    VkImageLayout ImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkClearValue  ClearValue {};
    TextureVulkanProperties Properties{};

public:
    Image<Vulkan>                          Image;
    Sampler<Vulkan>                        Sampler;
    ImageView<Vulkan>                      ImageView;
};


/// Template specialization for Vulkan CreateTextureObject
template<>
Texture<Vulkan> CreateTextureObject<Vulkan>( Vulkan&, const CreateTexObjectInfo& texInfo, MemoryPool<Vulkan>* pPool);

/// Template specialization for Vulkan CreateTextureFromBuffer
template<>
Texture<Vulkan> CreateTextureFromBuffer<Vulkan>( Vulkan&, const void* pData, size_t DataSize, uint32_t Width, uint32_t Height, uint32_t Depth, TextureFormat Format, SamplerAddressMode SamplerMode, SamplerFilter Filter, const char* pName );

/// Create a texture that views (aliases) another texture but using a different texture format (must be 'related' formats, which formats are related is dependant on graphics api)
template<>
Texture<Vulkan> CreateTextureObjectView( Vulkan&, const Texture<Vulkan>& original, TextureFormat viewFormat );

/// Template specialization for Vulkan ReleaseTexture
template<>
void ReleaseTexture<Vulkan>( Vulkan& vulkan, Texture<Vulkan>* );

/// Template specialization for Vulkan ReleaseImage
template<>
void ReleaseImage<Vulkan>( Vulkan& vulkan, Image<Vulkan>* );

/// Template specialization for Vulkan CreateImageView
template<>
ImageView<Vulkan> CreateImageView( Vulkan&, const Image<Vulkan>& image, TextureFormat format, uint32_t numMipLevels, uint32_t baseMipLevel, uint32_t numFaces, uint32_t firstFace, ImageViewType viewType );

/// Template specialization for Vulkan ReleaseImageView
template<>
void ReleaseImageView<Vulkan>( Vulkan& vulkan, ImageView<Vulkan>* );


/// Template specialization for Vulkan CreateSampler
template<>
Sampler<Vulkan> CreateSampler( Vulkan&, const CreateSamplerObjectInfo& );

/// Helper to take a sorce texture and make an array of textures where each one points to a single mip in the source
std::vector<Texture<Vulkan>> MakeTextureMipViews( Vulkan& vulkan, const Texture<Vulkan>& source, uint32_t maxMips );
