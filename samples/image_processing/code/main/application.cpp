//============================================================================================================
//
//
//                  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "application.hpp"
#include "main/applicationEntrypoint.hpp"
#include "camera/cameraController.hpp"
#include "camera/cameraControllerTouch.hpp"
#include "camera/cameraData.hpp"
#include "camera/cameraGltfLoader.hpp"
#include "gui/imguiVulkan.hpp"
#include "material/drawable.hpp"
#include "material/vulkan/shaderModule.hpp"
#include "material/shaderManagerT.hpp"
#include "material/materialManager.hpp"
#include "material/vulkan/specializationConstantsLayout.hpp"
#include "mesh/meshHelper.hpp"
#include "mesh/meshLoader.hpp"
#include "system/math_common.hpp"
#include "texture/textureManager.hpp"
#include "imgui.h"

#if OS_ANDROID
VAR(bool, gUseExtension, true, kVariableNonpersistent);
#else  // OS_ANDROID
VAR(bool, gUseExtension, false, kVariableNonpersistent);
#endif // OS_ANDROID
VAR(unsigned, gBlurFilterSize, 7, kVariableNonpersistent);

#if OS_ANDROID
#define SHADERFILE(f) f
#define TEXTUREFILE(f) f
#else
#define SHADERFILE(f) f
#define TEXTUREFILE(f) f
#endif // OS_ANDROID

const char* BloomImageprocessing::ShaderSets[ShaderPair_Count][3] =
{
    {SHADERFILE("VertexShader.vert.spv"), SHADERFILE("Downsample.frag.spv"), SHADERFILE("Downsample-Ext.frag.spv")},// ShaderPair_Downsample,
    {SHADERFILE("VertexShader.vert.spv"), SHADERFILE("BlurBase-Horizontal.frag.spv"), SHADERFILE("BlurBase-Horizontal-Ext.frag.spv")},// ShaderPair_Blur_Horizontal,
    {SHADERFILE("VertexShader.vert.spv"), SHADERFILE("BlurBase-Vertical.frag.spv"), SHADERFILE("BlurBase-Vertical-Ext.frag.spv")},// ShaderPair_Blur_Vertical,
    {SHADERFILE("VertexShader.vert.spv"), SHADERFILE("Display.frag.spv"), SHADERFILE("Display.frag.spv")},// ShaderPair_Display,
};

///
/// @brief Implementation of the Application entrypoint (called by the framework)
/// @return Pointer to Application (derived from @FrameworkApplicationBase).
/// Creates the Application class.  Ownership is passed to the calling (framework) function.
/// 
FrameworkApplicationBase* Application_ConstructApplication()
{
    return new BloomImageprocessing();
}

BloomImageprocessing::BloomImageprocessing() : ApplicationHelperBase()
{
    //may be available on Android only (and only if the extension is available)
#if defined(OS_ANDROID)
    m_bUseExtension = gUseExtension;
#else
    m_bUseExtension = false;
#endif // OS_ANDROID

    gBlurFilterSize = mini(gBlurFilterSize, 25);

    memset(m_weightTextureViews, 0, sizeof(m_weightTextureViews));
}

BloomImageprocessing::~BloomImageprocessing()
{
}

void  BloomImageprocessing::PreInitializeSetVulkanConfiguration(Vulkan::AppConfiguration& config)
{
    ApplicationHelperBase::PreInitializeSetVulkanConfiguration(config);
    config.RequiredExtension(VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME);
    config.RequiredExtension("VK_KHR_format_feature_flags2");
    config.OptionalExtension<Ext_VK_QCOM_image_processing>();//optional means don't fail if extension is not supported (eg launched from RenderDoc or on a device without support)
}

//-----------------------------------------------------------------------------
bool BloomImageprocessing::Initialize(uintptr_t windowHandle, uintptr_t hInstance)
//-----------------------------------------------------------------------------
{
    if (!ApplicationHelperBase::Initialize(windowHandle, hInstance))
    {
        return false;
    }

    Vulkan* pVulkan = GetVulkan();
    {
        const auto* pExt = pVulkan->GetExtension<Ext_VK_QCOM_image_processing>();
        const bool extensionLoaded = (pExt != nullptr) &&
                                     (pExt->Status == VulkanExtensionStatus::eLoaded);
        if (m_bUseExtension && !extensionLoaded)
        {
            LOGI("VK_QCOM_image_processing not available; falling back to non-extension shaders");
            m_bUseExtension = false;
        }
    }

    memset(m_passes, 0, sizeof(m_passes));

    //--------------------------------------------------------------------------
    // Setup Render Targets — one complete set per swapchain image so that each in-flight frame has its own intermediate render targets
    {
        const TextureFormat bloomFormat[] = { k_RtFormat };

        for (uint32_t buf = 0; buf < pVulkan->m_SwapchainImageCount; ++buf)
        {
            for (uint32_t ii = 0; ii < kPassesThatNeedIntermediateRenderTargets_Count; ++ii)
            {
                uint32_t w = GlobalPassInfo[ii].renderWidth;
                uint32_t h = GlobalPassInfo[ii].renderHeight;

                if (w == 0)
                {
                    w = pVulkan->m_SurfaceWidth;
                }

                if (h == 0)
                {
                    h = pVulkan->m_SurfaceHeight;
                }

                char rtNames[256];
                snprintf(rtNames, 256, "Bloom RT buf%d pass%d %dx%d", buf, ii, w, h);

                RenderPass renderPass;
                if (!pVulkan->CreateRenderPass(bloomFormat, TextureFormat::UNDEFINED, Msaa::Samples1, RenderPassInputUsage::Clear, RenderPassOutputUsage::StoreReadOnly, true, RenderPassOutputUsage::StoreReadOnly, renderPass))
                {
                    LOGE("Unable to create render pass buf:%d id:%d", buf, ii);
                    return false;
                }

                if (!m_IntermediateRts[buf][ii].Initialize(pVulkan, w, h, bloomFormat, TextureFormat::UNDEFINED, Msaa::Samples1, rtNames)
                    || !m_IntermediateRts[buf][ii].InitializeFrameBuffer(pVulkan, renderPass))
                {
                    LOGE("Unable to create intermediate render target buf:%d pass:%d", buf, ii);
                    return false;
                }

                m_RenderContexts[buf][ii] = RenderContext(std::move(renderPass), m_IntermediateRts[buf][ii].GetFrameBuffer(), rtNames);
            }
        }
    }
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // Setup Shaders
    for (uint32_t ss = 0; ss < ShaderPair_Count; ++ss)
    {
        const auto& vertexShader = ShaderSets[ss][0];
        const auto& fragmentShader = m_bUseExtension ? ShaderSets[ss][2] : ShaderSets[ss][1];
        LOGI("vertexShader=%s, fragmentShader=%s", vertexShader, fragmentShader);

        LoadShader(
            pVulkan, 
            *m_AssetManager, 
            &m_shaders[ss], 
            std::filesystem::path(SHADER_DESTINATION_PATH).append(vertexShader).string(),
            std::filesystem::path(SHADER_DESTINATION_PATH).append(fragmentShader).string());
    }
    //--------------------------------------------------------------------------
    
    //--------------------------------------------------------------------------
    // Setup Weight Array
    float* pRawTexData = new float[gBlurFilterSize];
    float16* pHalfTexData = new float16[gBlurFilterSize];
    {
        BuildWeightArray(gBlurFilterSize, pRawTexData);
        for (uint32_t ii = 0; ii < gBlurFilterSize; ++ii)
        {
            pHalfTexData[ii] = float16(pRawTexData[ii]);
        }
    }
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // Setup Textures
    if (m_bUseExtension)
    {
        for (uint32_t ii = 0; ii < NumWeightImages; ++ii)
        {
            //VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLE_WEIGHT_BIT_QCOM;//no longer used
            const TextureFormat weightFormat = TextureFormat::R16_SFLOAT;

            VkImageViewSampleWeightCreateInfoQCOM weightViewInfo = {};
            weightViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_SAMPLE_WEIGHT_CREATE_INFO_QCOM;
            weightViewInfo.pNext = NULL;
            weightViewInfo.numPhases = 1;

            VkImageViewCreateInfo viewInfo = {};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.pNext = &weightViewInfo;
            viewInfo.flags = 0;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            viewInfo.components.r = VK_COMPONENT_SWIZZLE_R;
            viewInfo.components.g = VK_COMPONENT_SWIZZLE_G;
            viewInfo.components.b = VK_COMPONENT_SWIZZLE_B;
            viewInfo.components.a = VK_COMPONENT_SWIZZLE_A;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;

            if (ii == 1) // H
            {
                m_weightTextures[ii] = CreateTextureFromBuffer( *pVulkan,
                                                                         pHalfTexData, gBlurFilterSize * sizeof( float16 ),
                                                                         gBlurFilterSize, 1, 1, weightFormat,
                                                                         SamplerAddressMode::ClampEdge,
                                                                         SamplerFilter::Nearest,
                                                                         "weight texture");

                weightViewInfo.filterCenter.x = gBlurFilterSize / 2;
                weightViewInfo.filterCenter.y = 0;
                weightViewInfo.filterSize.width = gBlurFilterSize;
                weightViewInfo.filterSize.height = 1;
            }
            else // V
            {
                m_weightTextures[ii] = CreateTextureFromBuffer(
                      *pVulkan, pHalfTexData, gBlurFilterSize * sizeof(float16),
                      1, gBlurFilterSize, 1, weightFormat,
                      SamplerAddressMode::ClampEdge,
                      SamplerFilter::Nearest,
                  "weight texture" );

                weightViewInfo.filterCenter.x = 0;
                weightViewInfo.filterCenter.y = gBlurFilterSize / 2;
                weightViewInfo.filterSize.width = 1;
                weightViewInfo.filterSize.height = gBlurFilterSize;
            }

            viewInfo.image = m_weightTextures[ii].GetVkImage();
            viewInfo.format = TextureFormatToVk(m_weightTextures[ii].Format);

            if (!CheckVkError("vkCreateImageView()", vkCreateImageView(pVulkan->m_VulkanDevice, &viewInfo, NULL, &m_weightTextureViews[ii])))
            {
                return false;
            }
        }
    }

    const PathManipulator_PrefixDirectory prefixTextureDir{ TEXTURE_DESTINATION_PATH };

    if (/*pVulkan->IsTextureFormatSupported(TextureFormat::ASTC_4x4_UNORM_BLOCK)*/ false)
    {
        m_sourceTexture = apiCast<Vulkan>(m_TextureManager->GetOrLoadTexture(TEXTUREFILE("painting_astc.ktx"), SamplerAddressMode::ClampEdge, prefixTextureDir));
    }
    else
    {
        m_sourceTexture = apiCast<Vulkan>(m_TextureManager->GetOrLoadTexture(TEXTUREFILE("painting.ktx"), SamplerAddressMode::ClampEdge, prefixTextureDir));
    }
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // Setup UBOs
    for (uint32_t ii = 0; ii < Pass_Count; ++ii)
    {
        const ConstPassInfo* pPassInfo = &GlobalPassInfo[ii];

        size_t dataSize = 0;
        float* pUboData = NULL;

        switch (pPassInfo->shaderPair)
        {
        case ShaderPair_Downsample:
            break;
        case ShaderPair_Blur_Horizontal:
        case ShaderPair_Blur_Vertical:
        {
            dataSize = 32 * sizeof(float);
            pUboData = (float*)calloc(32, sizeof(float));

            pUboData[0] = (float)gBlurFilterSize;

            for (uint32_t index = 0; index < gBlurFilterSize; index++)
            {
                pUboData[index + 1] = pRawTexData[index];
            }
            break;
        }
        case ShaderPair_Display:
            break;
        case ShaderPair_Count:
            assert(false);
            break;
        default:
            assert(false);
            break;
        }

        if (pUboData != NULL)
        {
            CreateUniformBuffer(pVulkan, &m_uniforms[ii], dataSize, pUboData);
            free(pUboData);
        }
    }
    //--------------------------------------------------------------------------

    delete[] pRawTexData;
    delete[] pHalfTexData;

    //each swapchain image gets its full complement of intermediate passes to avoid race conditions
    for (uint32_t buf = 0; buf < pVulkan->m_SwapchainImageCount; ++buf)
    {
        for (uint32_t pp = 0; pp < Pass_Count; ++pp)
        {
            m_passes[buf][pp] = new PassInfo(pVulkan, &GlobalPassInfo[pp]);

            m_passes[buf][pp]->pRt = &m_IntermediateRts[buf][pp];
            m_passes[buf][pp]->pRc = &m_RenderContexts[buf][pp];
            m_passes[buf][pp]->pUniform = NULL;
            m_passes[buf][pp]->renderArea.width  = m_passes[buf][pp]->pPassInfo->renderWidth;
            m_passes[buf][pp]->renderArea.height = m_passes[buf][pp]->pPassInfo->renderHeight;
            m_passes[buf][pp]->weightViewIndex = -1;
            m_passes[buf][pp]->pShaderInfo = &m_shaders[m_passes[buf][pp]->pPassInfo->shaderPairId];

            switch (m_passes[buf][pp]->pPassInfo->shaderPair)
            {
            case ShaderPair_Downsample:
                m_passes[buf][pp]->vInputViews.push_back(m_sourceTexture->GetVkImageView());
                break;
            case ShaderPair_Blur_Horizontal:
            {
                m_passes[buf][pp]->pUniform = &m_uniforms[pp];
                m_passes[buf][pp]->vInputViews.push_back(m_IntermediateRts[buf][pp - 1].m_ColorAttachments[0].GetVkImageView());
                if (m_bUseExtension)
                {
                    m_passes[buf][pp]->vInputViews.push_back(m_weightTextureViews[0]);
                    m_passes[buf][pp]->weightViewIndex = m_passes[buf][pp]->vInputViews.size() - 1;
                }
                break;
            }
            case ShaderPair_Blur_Vertical:
                m_passes[buf][pp]->pUniform = &m_uniforms[pp];
                m_passes[buf][pp]->vInputViews.push_back(m_IntermediateRts[buf][pp - 1].m_ColorAttachments[0].GetVkImageView());
                if (m_bUseExtension)
                {
                    m_passes[buf][pp]->vInputViews.push_back(m_weightTextureViews[1]);
                    m_passes[buf][pp]->weightViewIndex = m_passes[buf][pp]->vInputViews.size() - 1;
                }
                break;
            case ShaderPair_Display:
                m_passes[buf][pp]->pRt = NULL;

                //Pass_Display has no intermediate render target, so m_RenderContexts[buf][pp] is default-constructed (empty).  Point pRc at the last
                //initialized render context, so FinalizePass() passes CreatePipeline() a valid render context and the pipeline is successfully created
                //TODO: this might be cleaned up, but such a cleanup might involve refactoring CreatePipeline()
                m_passes[buf][pp]->pRc = &m_RenderContexts[buf][pp - 1];

                m_passes[buf][pp]->renderArea.width = pVulkan->m_SurfaceWidth;
                m_passes[buf][pp]->renderArea.height = pVulkan->m_SurfaceHeight;
                m_passes[buf][pp]->vInputViews.push_back(m_sourceTexture->GetVkImageView());
                m_passes[buf][pp]->vInputViews.push_back(m_IntermediateRts[buf][pp - 1].m_ColorAttachments[0].GetVkImageView());
                break;
            case ShaderPair_Count:
                assert(false);
                break;
            default:
                assert(false);
                break;
            }

            FinalizePass(m_passes[buf][pp]);
        }
    }
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // Setup Command Buffers
    for (uint32_t ii = 0; ii < pVulkan->m_SwapchainImageCount; ++ii)
    {
        BuildCmdBuffer(ii);
    }
    //--------------------------------------------------------------------------

    return true;
}

//-----------------------------------------------------------------------------
void BloomImageprocessing::Destroy()
//-----------------------------------------------------------------------------
{
    GetGfxApi()->WaitUntilIdle();

    // Textures
    for (uint32_t ii = 0; ii < NumWeightImages; ++ii)
    {
        vkDestroyImageView(GetVulkan()->m_VulkanDevice, m_weightTextureViews[ii], NULL);
    }
    for (auto& t : m_weightTextures)
    {
        t.Release(GetVulkan());
    }
    m_sourceTexture = nullptr;

    // Shaders
    for (uint32_t ss = 0; ss < ShaderPair_Count; ++ss)
    {
        ReleaseShader(GetVulkan(), &m_shaders[ss]);
    }

    // Uniform Buffers
    for (uint32_t ii = 0; ii < Pass_Count; ++ii)
    {
        ReleaseUniformBuffer(GetVulkan(), &m_uniforms[ii]);
    }

    for (uint32_t buf = 0; buf < GetVulkan()->m_SwapchainImageCount; ++buf)
    {
        for (uint32_t pp = 0; pp < Pass_Count; ++pp)
        {
            delete m_passes[buf][pp];
            m_passes[buf][pp] = nullptr;
        }
        for (uint32_t ii = 0; ii < kPassesThatNeedIntermediateRenderTargets_Count; ++ii)
        {
            m_IntermediateRts[buf][ii].Release();
        }
    }

    ApplicationHelperBase::Destroy();
}

void BloomImageprocessing::Render(float fltDiffTime)
{
    // Grab the vulkan wrapper
    Vulkan* pVulkan = GetVulkan();

    // Obtain the next swap chain image for the next frame.
    auto currentBuffer = pVulkan->SetNextBackBuffer();

    m_commandBuffers[currentBuffer.idx].QueueSubmit(currentBuffer, pVulkan->m_RenderCompleteSemaphore);

    // Queue is loaded up, tell the driver to start processing
    pVulkan->PresentQueue(pVulkan->m_RenderCompleteSemaphore, currentBuffer.swapchainPresentIdx);

}

void BloomImageprocessing::FinalizePass(PassInfo* pPass)
{
    Vulkan* pVulkan = GetVulkan();

    if (pPass->pRt == NULL)
    {
        pPass->renderpass = pVulkan->m_SwapchainRenderPass.mRenderPass;
    }
    else
    {
        pPass->renderpass = pPass->pRc->GetRenderPass().mRenderPass;
        pPass->fbo = pPass->pRc->GetFramebuffer()->m_FrameBuffer;
    }

    // Descriptor Set Layout
    VkDescriptorSetLayoutCreateInfo dsLayoutInfo = {};
    dsLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsLayoutInfo.pNext = NULL;
    dsLayoutInfo.flags = 0x0;

    std::vector<VkDescriptorSetLayoutBinding> vBindings;
    vBindings.reserve(10);
    if (pPass->pUniform != NULL)
    {
        vBindings.push_back(VkDescriptorSetLayoutBinding());
        vBindings.back().binding = 0;
        vBindings.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        vBindings.back().descriptorCount = 1;
        vBindings.back().stageFlags = (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        vBindings.back().pImmutableSamplers = NULL;
    }

    for (uint32_t ii = 0; ii < pPass->vInputViews.size(); ++ii)
    {
        bool isWeightImage = (pPass->weightViewIndex >= 0) && (ii >= pPass->weightViewIndex);

        vBindings.push_back(VkDescriptorSetLayoutBinding());
        vBindings.back().binding = vBindings.size() - 1;
        vBindings.back().descriptorType = isWeightImage ? VK_DESCRIPTOR_TYPE_SAMPLE_WEIGHT_IMAGE_QCOM : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        vBindings.back().descriptorCount = 1;
        vBindings.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        vBindings.back().pImmutableSamplers = NULL;

        vBindings.push_back(VkDescriptorSetLayoutBinding());
        vBindings.back().binding = vBindings.size() - 1;
        vBindings.back().descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        vBindings.back().descriptorCount = 1;
        vBindings.back().stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        vBindings.back().pImmutableSamplers = NULL;
    }

    dsLayoutInfo.bindingCount = vBindings.size();
    dsLayoutInfo.pBindings = vBindings.data();
    if (!CheckVkError("vkCreateDescriptorSetLayout()", vkCreateDescriptorSetLayout(pVulkan->m_VulkanDevice, &dsLayoutInfo, NULL, &pPass->dsLayout)))
    {
        return;
    }

    // Pipeline Layout
    VkPipelineLayoutCreateInfo PipelineLayoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    PipelineLayoutInfo.pNext = NULL;
    PipelineLayoutInfo.setLayoutCount = 1;
    PipelineLayoutInfo.pSetLayouts = &pPass->dsLayout;

    if (!CheckVkError("vkCreatePipelineLayout()", vkCreatePipelineLayout(pVulkan->m_VulkanDevice, &PipelineLayoutInfo, NULL, &pPass->pLayout)))
    {
        return;
    }

    // Descriptor Pool
    std::vector< VkDescriptorPoolSize> vPoolSizes;
    vPoolSizes.reserve(4);
    for (auto dsIt = vBindings.begin(); dsIt != vBindings.end(); dsIt++)
    {
        vPoolSizes.push_back(VkDescriptorPoolSize());
        vPoolSizes.back().type = (*dsIt).descriptorType;
        vPoolSizes.back().descriptorCount = (*dsIt).descriptorCount;
    }
    VkDescriptorPoolCreateInfo PoolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    PoolInfo.pNext = NULL;
    PoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;     // VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT to allow them to be returned
    PoolInfo.maxSets = 1;  // Since descriptor sets come out of this pool we need more than one
    PoolInfo.poolSizeCount = vPoolSizes.size();
    PoolInfo.pPoolSizes = vPoolSizes.data();

    if (!CheckVkError("vkCreateDescriptorPool()", vkCreateDescriptorPool(pVulkan->m_VulkanDevice, &PoolInfo, NULL, &pPass->dsPool)))
    {
        return;
    }

    // Descriptor Set
    VkDescriptorSetAllocateInfo DescSetInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    DescSetInfo.pNext = NULL;
    DescSetInfo.descriptorPool = pPass->dsPool;
    DescSetInfo.descriptorSetCount = 1;
    DescSetInfo.pSetLayouts = &pPass->dsLayout;

    if (!CheckVkError("vkAllocateDescriptorSets()", vkAllocateDescriptorSets(pVulkan->m_VulkanDevice, &DescSetInfo, &pPass->ds)))
    {
        return;
    }

    VkDescriptorImageInfo vImageInfos[8] = {};
    VkDescriptorBufferInfo vBufferInfos[4] = {};
    VkWriteDescriptorSet vWriteInfos[12] = {};

    if (pPass->pUniform != NULL)
    {
        vBufferInfos[0] = pPass->pUniform->bufferInfo;
    }
    for (uint32_t tt = 0; tt < pPass->vInputViews.size(); ++tt)
    {
        VkImageView view = pPass->vInputViews[tt];

        bool isWeightImage = ((pPass->weightViewIndex >= 0) && (tt >= pPass->weightViewIndex));

        vImageInfos[tt * 2 + 0].imageView = view;
        vImageInfos[tt * 2 + 0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkSamplerReductionModeCreateInfo redInfo = {};
        redInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO;
        redInfo.pNext = NULL;
        redInfo.reductionMode = VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE;

        VkSamplerCreateInfo SamplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        SamplerInfo.flags = isWeightImage ? VK_SAMPLER_CREATE_IMAGE_PROCESSING_BIT_QCOM : 0;
        SamplerInfo.pNext = &redInfo;
        SamplerInfo.magFilter = isWeightImage ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        SamplerInfo.minFilter = isWeightImage ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        SamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        SamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        SamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        SamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        SamplerInfo.mipLodBias = 0.f;
        SamplerInfo.anisotropyEnable = VK_FALSE;
        SamplerInfo.maxAnisotropy = 1.0f;
        SamplerInfo.compareEnable = VK_FALSE;
        SamplerInfo.compareOp = VK_COMPARE_OP_NEVER;
        SamplerInfo.minLod = 0.0f;
        SamplerInfo.maxLod = 0.0f;
        SamplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;   // VkBorderColor
        SamplerInfo.unnormalizedCoordinates = VK_FALSE;

        VkSampler sampler;

        if (!CheckVkError("vkCreateSampler()", vkCreateSampler(pVulkan->m_VulkanDevice, &SamplerInfo, NULL, &sampler)))
        {
            return;
        }

        pPass->samplers.push_back(sampler);
        vImageInfos[tt * 2 + 1].sampler = sampler;
    }

    uint32_t writeIndex = 0;

    uint32_t baseBinding = 0;
    if (pPass->pUniform != NULL)
    {
        
        vWriteInfos[writeIndex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        vWriteInfos[writeIndex].pNext = NULL;
        vWriteInfos[writeIndex].dstSet = pPass->ds;
        vWriteInfos[writeIndex].dstBinding = 0;
        vWriteInfos[writeIndex].dstArrayElement = 0;
        vWriteInfos[writeIndex].descriptorCount = 1;
        vWriteInfos[writeIndex].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        vWriteInfos[writeIndex].pBufferInfo = &vBufferInfos[0];

        writeIndex++;
        baseBinding++;
    }

    for (uint32_t tt = 0; tt < pPass->vInputViews.size(); ++tt)
    {
        VkImageView view = pPass->vInputViews[tt];
        bool isWeightImage = (pPass->weightViewIndex >= 0) && (tt >= pPass->weightViewIndex);

        vWriteInfos[writeIndex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        vWriteInfos[writeIndex].pNext = NULL;
        vWriteInfos[writeIndex].dstSet = pPass->ds;
        vWriteInfos[writeIndex].dstBinding = baseBinding + tt * 2 + 0;
        vWriteInfos[writeIndex].dstArrayElement = 0;
        vWriteInfos[writeIndex].descriptorCount = 1;
        vWriteInfos[writeIndex].descriptorType = isWeightImage ? VK_DESCRIPTOR_TYPE_SAMPLE_WEIGHT_IMAGE_QCOM : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        vWriteInfos[writeIndex].pImageInfo = &vImageInfos[tt * 2 + 0];

        writeIndex++;

        vWriteInfos[writeIndex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        vWriteInfos[writeIndex].pNext = NULL;
        vWriteInfos[writeIndex].dstSet = pPass->ds;
        vWriteInfos[writeIndex].dstBinding = baseBinding + tt * 2 + 1;
        vWriteInfos[writeIndex].dstArrayElement = 0;
        vWriteInfos[writeIndex].descriptorCount = 1;
        vWriteInfos[writeIndex].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        vWriteInfos[writeIndex].pImageInfo = &vImageInfos[tt * 2 + 1];

        writeIndex++;
    }

    vkUpdateDescriptorSets(pVulkan->m_VulkanDevice, writeIndex, vWriteInfos, 0, NULL);

    // Pipeline
    VkPipelineVertexInputStateCreateInfo visci = {};
    visci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    // Common to all pipelines
    // State for rasterization, such as polygon fill mode is defined.
    VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.depthClampEnable = VK_FALSE;
    rs.rasterizerDiscardEnable = VK_FALSE;
    rs.depthBiasEnable = VK_FALSE;
    rs.lineWidth = 1.0f;

    // For the cube, we don't write or check depth
    VkPipelineDepthStencilStateCreateInfo ds = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    ds.depthBoundsTestEnable = VK_FALSE;
    ds.back.failOp = VK_STENCIL_OP_KEEP;
    ds.back.passOp = VK_STENCIL_OP_KEEP;
    ds.back.compareOp = VK_COMPARE_OP_ALWAYS;
    ds.stencilTestEnable = VK_FALSE;
    ds.front = ds.back;


    rs.cullMode = VK_CULL_MODE_NONE;
    rs.depthBiasEnable = VK_FALSE;
    ds.depthTestEnable = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;

    pVulkan->CreatePipeline(
        VK_NULL_HANDLE, //pipeline cache
        &visci,         // vertex input state
        pPass->pLayout,
        *pPass->pRc,
        &rs,            // rasterization
        &ds,            // depth stencil
        nullptr,        // blend
        nullptr,        // multisample
        nullptr,        // ia
        {},             // dynamic states
        nullptr,        // viewport
        nullptr,        // scissor
        VK_NULL_HANDLE, // task shader
        VK_NULL_HANDLE, // mesh shader
        pPass->pShaderInfo->VertShaderModule.GetVkShaderModule(),
        pPass->pShaderInfo->FragShaderModule.GetVkShaderModule(),
        nullptr,        // specialization
        false,          // allow derivation
        VK_NULL_HANDLE, // derive from
        &pPass->pipeline);

}

void BloomImageprocessing::DrawPass(CommandBuffer* cmd, const PassInfo* pPass, uint32_t idx)
{
    Vulkan* pVulkan = GetVulkan();

    // When starting the render pass, we can set clear values.
    VkClearColorValue clear_color = {};
    clear_color.float32[0] = 0.0f;
    clear_color.float32[1] = 0.0f;
    clear_color.float32[2] = 0.0f;
    clear_color.float32[3] = 1.0f;

    VkViewport Viewport = {};
    Viewport.width = (float)pPass->renderArea.width;
    Viewport.height = (float)pPass->renderArea.height;
    Viewport.minDepth = 0.0f;
    Viewport.maxDepth = 1.0f;
    VkRect2D Scissor = {};
    Scissor.offset.x = 0;
    Scissor.offset.y = 0;
    Scissor.extent.width = pPass->renderArea.width;
    Scissor.extent.height = pPass->renderArea.height;

    cmd->BeginRenderPass(Scissor, 0.0f, 1.0f, {&clear_color, 1}, 1,
                         pPass->pRt == NULL ? true : false, pPass->renderpass,
                         pPass->pRt == NULL ? true : false,
                         pPass->pRt == NULL
                             ? pVulkan->m_SwapchainBuffers[idx].framebuffer.m_FrameBuffer
                             : pPass->fbo,
                         VK_SUBPASS_CONTENTS_INLINE);

    vkCmdSetViewport(cmd->m_VkCommandBuffer, 0, 1, &Viewport);
    vkCmdSetScissor(cmd->m_VkCommandBuffer, 0, 1, &Scissor);

    // Bind the pipeline for this material
    vkCmdBindPipeline(cmd->m_VkCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pPass->pipeline);

    // Bind everything the shader needs
    vkCmdBindDescriptorSets(cmd->m_VkCommandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pPass->pLayout,
        0,
        1,
        &pPass->ds,
        0,
        NULL);

    vkCmdDraw(cmd->m_VkCommandBuffer, 6, 1, 0, 0);

    cmd->EndRenderPass();
}

void BloomImageprocessing::BuildCmdBuffer(uint32_t idx)
{
    CommandBuffer* pCmdBuf = &m_commandBuffers[idx];

    Vulkan* pVulkan = GetVulkan();

    if (!pCmdBuf->Initialize(pVulkan))
    {
        LOGE("Failed to initialize command buffer %d", idx);
        return;
    }

    pCmdBuf->Reset();
    pCmdBuf->Begin();

    for (uint32_t pp = 0; pp < Pass_Count; ++pp)
    {
        if (pp > 0)
        {
            VkMemoryBarrier memBarrier = {};
            memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            memBarrier.pNext = NULL;
            memBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;//src writes to color buffer
            memBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;//dst reads from color buffer

            vkCmdPipelineBarrier(
                pCmdBuf->m_VkCommandBuffer,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,  //src writes to color buffer
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,          //dst reads from color buffer
                0x0,
                1, &memBarrier,
                0, NULL,
                0, NULL);
        }

        DrawPass(pCmdBuf, m_passes[idx][pp], idx);
    }

    pCmdBuf->End();


}

void BloomImageprocessing::BuildWeightArray(uint32_t weightSize, float* pWeightArray)
{
    uint32_t tri = weightSize + 3;
    uint64_t fact_tri = Factorial(tri);

    double* pTriVals = new double[tri+1];
    double triValSum = 0.0;

    for (uint32_t ii = 0; ii < tri + 1; ++ii)
    {
        pTriVals[ii] = (double)fact_tri / ((double)Factorial(ii) * (double)Factorial(tri - ii));
        triValSum += pTriVals[ii];
    }

    for (uint32_t ii = 0; ii < weightSize; ++ii)
    {
        pWeightArray[ii] = (float)(pTriVals[ii + 2] / triValSum);
    }

    delete[] pTriVals;
}
