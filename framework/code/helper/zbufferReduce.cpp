// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2024 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "zbufferReduce.hpp"
#include "vulkan/vulkan.hpp"
#include "vulkan/commandBuffer.hpp"
#include "texture/vulkan/texture.hpp"
#include "material/vulkan/computable.hpp"
#include "material/vulkan/materialManager.hpp"
#include <memory>

//-----------------------------------------------------------------------------
bool ZBufferReduce::Init( Vulkan* pVulkan, const Texture<Vulkan>& srcDepth, const MaterialManager<Vulkan>& materialManager, const Shader<Vulkan>* const pReduce4x4Shader, MemoryPool<Vulkan>* pPool )
//-----------------------------------------------------------------------------
{
    using ImageInfo = ImageInfo<Vulkan>;

    int numMips = -1;
    {
        uint32_t x = std::max( srcDepth.Width, srcDepth.Height );
        while (x >>= 1)
        {
            ++numMips;
        }
    }
    assert( numMips > 1 );
    // Ensure an even number of mips, reduction shader generates 2 lod at a time
    if ((numMips & 1) != 0)
        --numMips;
    const CreateTexObjectInfo reducedZBufferInfo{
        srcDepth.Width / 2,
        srcDepth.Height / 2,
        1, // depth
        (uint32_t)numMips,
        1, // faces
        TextureFormat::R32_SFLOAT,
        TT_COMPUTE_TARGET,
        TEXTURE_FLAGS::None,
        "ReducedZ",
        Msaa::Samples1,
        SamplerFilter::Nearest,
        SamplerAddressMode::Undefined,	//default to picking from a default dependant on the texture type
        false   // UnNormalizedCoordinates
    };
    auto reducedZBuffer = CreateTextureObject( *pVulkan, reducedZBufferInfo, pPool );

    // override the starting image layout because we are binding the 'entire' texture as read input (and sub-image views as outputs and inputs).
    VkCommandBuffer setupCmdBuffer = pVulkan->StartSetupCommandBuffer();
    pVulkan->SetImageLayout( reducedZBuffer.GetVkImage(), setupCmdBuffer, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, (VkPipelineStageFlags)0/*unused param*/, (VkPipelineStageFlags)0/*unused param*/, reducedZBuffer.FirstMip, reducedZBuffer.MipLevels, 0, 1 );
    pVulkan->FinishSetupCommandBuffer( setupCmdBuffer );
    reducedZBuffer.ImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    std::map<const std::string, const TextureBase*> mipTextures = {
        { "InputDepth", &srcDepth }
    };
    constexpr int cMaxMips = 12;
    numMips = std::min( cMaxMips, numMips );
    std::map<const std::string, ImageInfo> mipImages = {
        { "InputDepth", ImageInfo( srcDepth ) },
        { "0", ImageInfo( reducedZBuffer ) },
        { "1", ImageInfo( reducedZBuffer ) },
        { "2", ImageInfo( reducedZBuffer ) },
        { "3", ImageInfo( reducedZBuffer ) },
        { "4", ImageInfo( reducedZBuffer ) },
        { "5", ImageInfo( reducedZBuffer ) },
        { "6", ImageInfo( reducedZBuffer ) },
        { "7", ImageInfo( reducedZBuffer ) },
        { "8", ImageInfo( reducedZBuffer ) },
        { "9", ImageInfo( reducedZBuffer ) },
        { "10", ImageInfo( reducedZBuffer ) },
        { "11", ImageInfo( reducedZBuffer ) }
    };
    // Create (and fill in) in the imageviews and 'textures' for each of the mip levels in m_ReducedZBuffer
    assert( m_ReducedZBufferMipTexInfos.empty() );
    m_ReducedZBufferMipTexInfos.reserve( cMaxMips );

    VkImageViewCreateInfo imageViewCreateInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    imageViewCreateInfo.image = reducedZBuffer.GetVkImage();
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCreateInfo.format = TextureFormatToVk( reducedZBuffer.Format );
    imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.layerCount = 1;

    VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

    int mipLevel = 0;
    for (; mipLevel < numMips; ++mipLevel)
    {
        VkImageView imageView = VK_NULL_HANDLE;
        VkSampler vkSampler = VK_NULL_HANDLE;

        // Create an imageview and sampler to point to this mip.  Assigned/stored in a Texture<Vulkan> per mip (all pointing to the same VkImage).
        imageViewCreateInfo.subresourceRange.baseMipLevel = mipLevel;
        samplerCreateInfo.minLod = float( mipLevel );
        samplerCreateInfo.maxLod = float( mipLevel );
        vkCreateImageView( pVulkan->m_VulkanDevice, &imageViewCreateInfo, nullptr, &imageView );
        vkCreateSampler( pVulkan->m_VulkanDevice, &samplerCreateInfo, nullptr, &vkSampler );
        SamplerVulkan sampler{pVulkan->m_VulkanDevice, vkSampler};

        ImageViewVulkan imageViewVulkan{imageView,imageViewCreateInfo.viewType};
        m_ReducedZBufferMipTexInfos.emplace_back( reducedZBufferInfo.uiWidth, reducedZBufferInfo.uiHeight, reducedZBufferInfo.uiDepth, 1, mipLevel, 1, 0, reducedZBufferInfo.Format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VkClearValue{}, reducedZBuffer.GetVkImage(), (VkDeviceMemory)VK_NULL_HANDLE, std::move(sampler), std::move(imageViewVulkan));

        const auto insertedTexture = mipTextures.insert( {std::to_string( mipLevel ), &m_ReducedZBufferMipTexInfos.back()} );    // map name to Texture* for this mip.
        const auto insertedImage = mipImages.insert_or_assign( insertedTexture.first->first, *insertedTexture.first->second );   // map name to an ImageInfo for this mip.
        insertedImage.first->second.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    for (; mipLevel < cMaxMips; ++mipLevel)
    {
        const auto insertedTexture = mipTextures.insert( {std::to_string( mipLevel ), &m_ReducedZBufferMipTexInfos.back()} );    // map name to Texture* for the last valid mip.
        const auto insertedImage = mipImages.insert_or_assign( insertedTexture.first->first, *insertedTexture.first->second );   // map name to an ImageInfo for this mip.
        insertedImage.first->second.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    assert( pReduce4x4Shader );
    auto material = materialManager.CreateMaterial( *pReduce4x4Shader, 1,
        [&mipTextures, this]( const std::string& textureName ) -> const MaterialManagerBase::tPerFrameTexInfo {
            return {mipTextures.find( textureName )->second};
        },
        nullptr,
            [&mipImages]( const std::string& imageName ) -> const ImageInfo {
                return mipImages.find( imageName )->second;
        }
    );

    auto pComputable = std::make_unique<Computable<Vulkan>>( *pVulkan, std::move( material ) );
    if (!pComputable->Init())
        return false;
    for (uint32_t passIdx = 0; passIdx < (uint32_t)pComputable->GetPasses().size(); ++passIdx)
        pComputable->SetDispatchThreadCount( passIdx, {srcDepth.Width, srcDepth.Height, 1} );
    m_ReduceComputable = std::move( pComputable );
    m_NumMips = numMips;
    m_ReducedZBuffer = std::make_unique<Texture<Vulkan>>( std::move( reducedZBuffer ) );

    return true;
}

//-----------------------------------------------------------------------------
void ZBufferReduce::Release( Vulkan* pVulkan )
//-----------------------------------------------------------------------------
{
    for (auto& texInfo : m_ReducedZBufferMipTexInfos)
    {
        texInfo.Release( pVulkan );
    }
    m_ReducedZBufferMipTexInfos.clear();
    if (m_ReducedZBuffer)
        m_ReducedZBuffer->Release( pVulkan );
    m_ReducedZBuffer.reset();
    m_ReduceComputable.reset();
}

//-----------------------------------------------------------------------------
void ZBufferReduce::UpdateCommandBuffer( CommandList<Vulkan>& vkCommandBuffer )
//-----------------------------------------------------------------------------
{
    std::vector< VkImageMemoryBarrier> reduceInputImageMemoryBarrier;
    reduceInputImageMemoryBarrier.push_back( {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                                nullptr,
                                                VK_ACCESS_SHADER_WRITE_BIT, //srcAccessMask
                                                VK_ACCESS_SHADER_READ_BIT,  //dstAccessMask
                                                VK_IMAGE_LAYOUT_UNDEFINED,  //oldLayout;
                                                VK_IMAGE_LAYOUT_GENERAL,    //newLayout;
                                                VK_QUEUE_FAMILY_IGNORED,    //srcQueueFamilyIndex;
                                                VK_QUEUE_FAMILY_IGNORED,    //dstQueueFamilyIndex;
                                                m_ReducedZBuffer->GetVkImage(),           //image;
                                                { VK_IMAGE_ASPECT_COLOR_BIT,//aspect;
                                                    m_ReducedZBuffer->FirstMip,  //baseMipLevel;
                                                    m_ReducedZBuffer->MipLevels, //mipLevelCount;
                                                    0,                      //baseLayer;
                                                    1,                      //layerCount;
                                                }//subresourceRange;
        } );

    // Barrier on input image, with correct layouts set.
    vkCmdPipelineBarrier( vkCommandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,   // srcMask,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,   // dstMask,
        0,
        0, nullptr,
        0, nullptr,
        (uint32_t)reduceInputImageMemoryBarrier.size(),
        reduceInputImageMemoryBarrier.empty() ? nullptr : reduceInputImageMemoryBarrier.data() );

    int mip = 0;
    for (const auto& pass : m_ReduceComputable->GetPasses())
    {
        m_ReduceComputable->DispatchPass( vkCommandBuffer, pass, 0 );
        mip += 2;
        if (mip >= m_NumMips)   // only using a subset of the computable passes
            break;
    }

    const auto& computableOutputBufferBarriers = m_ReduceComputable->GetBufferOutputMemoryBarriers();
    auto computableOutputImageBarriers = m_ReduceComputable->GetImageOutputMemoryBarriers();    // take a copy!
    // transition the last mip too!
    computableOutputImageBarriers.emplace_back( VkImageMemoryBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                                nullptr,
                                                VK_ACCESS_SHADER_WRITE_BIT, //srcAccessMask
                                                VK_ACCESS_SHADER_READ_BIT,  //dstAccessMask
                                                VK_IMAGE_LAYOUT_GENERAL,  //oldLayout;
                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,    //newLayout;
                                                VK_QUEUE_FAMILY_IGNORED,    //srcQueueFamilyIndex;
                                                VK_QUEUE_FAMILY_IGNORED,    //dstQueueFamilyIndex;
                                                m_ReducedZBuffer->GetVkImage(),           //image;
                                                { VK_IMAGE_ASPECT_COLOR_BIT,//aspect;
                                                  uint32_t( mip ) - 1,                  //baseMipLevel;
                                                  1,                      //mipLevelCount;
                                                  0,                      //baseLayer;
                                                  1,                      //layerCount;
                                                }//subresourceRange;
        } );

    // Barrier on output memory, with correct layouts set.
    using tBufferBarrier = std::remove_reference<decltype(computableOutputBufferBarriers)>::type::value_type::tBarrier;
    using tImageBarrier = std::remove_reference<decltype(computableOutputImageBarriers)>::type::value_type::tBarrier;
    vkCmdPipelineBarrier( vkCommandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,   // srcMask,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,   // dstMask,
        0,
        0, nullptr,
        (uint32_t)computableOutputBufferBarriers.size(),
        computableOutputBufferBarriers.empty() ? nullptr : (tBufferBarrier*)computableOutputBufferBarriers.data(),
        (uint32_t)computableOutputImageBarriers.size(),
        (tImageBarrier*)computableOutputImageBarriers.data() );
}

