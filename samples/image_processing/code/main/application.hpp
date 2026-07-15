//============================================================================================================
//
//
//                  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

///
/// @file BloomImageprocessing.hpp
/// @brief BloomImageprocessing
/// 
/// Most basic application that compiles and runs with the Vulkan Framework.
/// DOES NOT initialize Vulkan.
/// 

#include "main/applicationHelperBase.hpp"
#include "memory/vulkan/uniform.hpp"
#include "vulkan/commandBuffer.hpp"
#include "vulkan/extensionHelpers.hpp"
#include "vulkan/vulkan_support.hpp"
#include <unordered_map>

class BloomImageprocessing : public ApplicationHelperBase
{
public:
    BloomImageprocessing();
    ~BloomImageprocessing() override;

    /// @brief Ticked every frame (by the Framework)
    /// @param fltDiffTime time (in seconds) since the last call to Render.
    virtual bool Initialize(uintptr_t windowHandle, uintptr_t hInstance) override;
    virtual void Destroy() override;
    virtual  void Render(float fltDiffTime) override;
    virtual void  PreInitializeSetVulkanConfiguration(Vulkan::AppConfiguration& config) override;
private:
    bool m_bUseExtension;

    static inline uint32_t maxi(uint32_t a, uint32_t b)
    {
        return ((a > b) ? a : b);
    }

    static inline uint32_t mini(uint32_t a, uint32_t b)
    {
        return ((a < b) ? a : b);
    }

    static const TextureFormat k_RtFormat = TextureFormat::B10G11R11_UFLOAT_PACK32;
    static const uint32_t      k_FullImageWidth = 1920;
    static const uint32_t      k_FullImageHeight = 1080;

    enum Passes
    {
        Pass_Downsample,
        Pass_Blur_Horizontal,
        Pass_Blur_Vertical,
        Pass_Display,
        Pass_Count,
    };
    static const uint32_t kPassesThatNeedIntermediateRenderTargets_Count = Pass_Display;//last pass renders to swapchain image, requiring no intermediate render target

    struct float16
    {
        float16() noexcept : v(0) {}
        float16(const float16& f16) noexcept : v(f16.v) {}
        float16(float f32) noexcept {
            const uint32_t u = *((uint32_t*)&f32) + 0x00001000;    // bitwise cast to int, round up mantissa
            const uint32_t e = (u & 0x7f800000) >> 23;              // exponent
            const uint32_t m = u & 0x007FFFFF;                      // mantissa
            v = uint16_t((u & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7c00) | m >> 13) | ((e < 113) & (e > 101)) * ((((0x007ff000 + m) >> (125 - e)) + 1) >> 1) | (e > 143) * 0x7fff);
        }
        uint16_t v;
    };

    enum ShaderPairs
    {
        ShaderPair_Downsample,
        ShaderPair_Blur_Horizontal,
        ShaderPair_Blur_Vertical,
        ShaderPair_Display,
        ShaderPair_Count,
    };

    struct ConstPassInfo
    {
        union
        {
            Passes    ePass;
            uint32_t nPassId;
        };

        union
        {
            ShaderPairs shaderPair;
            uint32_t    shaderPairId;
        };

        uint32_t renderWidth;
        uint32_t renderHeight;
    };

    const ConstPassInfo GlobalPassInfo[Pass_Count] =
    {
        { Pass_Downsample,      ShaderPair_Downsample,      k_FullImageWidth >> 1, k_FullImageHeight >> 1, },
        { Pass_Blur_Horizontal, ShaderPair_Blur_Horizontal, k_FullImageWidth >> 1, k_FullImageHeight >> 1, },
        { Pass_Blur_Vertical,   ShaderPair_Blur_Vertical,   k_FullImageWidth >> 1, k_FullImageHeight >> 1, },
        { Pass_Display,         ShaderPair_Display,         0,                     0, },
    };

    struct PassInfo
    {
        Vulkan*                      pVulkan;
        const Uniform*               pUniform;
        const RenderTarget*          pRt;
        const RenderContext*         pRc;
        std::vector<VkImageView>     vInputViews;
        const ShaderInfo*            pShaderInfo;
        VkExtent2D                   renderArea;
        int32_t                      weightViewIndex;

        VkPipeline                   pipeline;
        VkDescriptorSetLayout        dsLayout;
        VkPipelineLayout             pLayout;
        VkDescriptorPool             dsPool;
        VkDescriptorSet              ds;
        VkRenderPass                 renderpass;
        VkFramebuffer                fbo;
        std::vector<VkSampler>       samplers;

        const ConstPassInfo*         pPassInfo;

        PassInfo(Vulkan* pVulkanIn, const ConstPassInfo* pInfo) :
            pVulkan(pVulkanIn),
            pUniform(NULL), 
            pRt(NULL),
            pShaderInfo(NULL),
            weightViewIndex(-1),
            pipeline(VK_NULL_HANDLE),
            dsLayout(VK_NULL_HANDLE),
            pLayout(VK_NULL_HANDLE),
            dsPool(VK_NULL_HANDLE),
            ds(VK_NULL_HANDLE),
            renderpass(VK_NULL_HANDLE),
            fbo(VK_NULL_HANDLE),
            pPassInfo(pInfo)
        {
            vInputViews.reserve(8);
            samplers.reserve(8);
        }

        ~PassInfo()
        {
            vkDestroyPipeline(pVulkan->m_VulkanDevice, pipeline, NULL);
            vkDestroyDescriptorSetLayout(pVulkan->m_VulkanDevice, dsLayout, NULL);
            vkDestroyPipelineLayout(pVulkan->m_VulkanDevice, pLayout, NULL);
            vkFreeDescriptorSets(pVulkan->m_VulkanDevice, dsPool, 1, &ds);
            vkDestroyDescriptorPool(pVulkan->m_VulkanDevice, dsPool, NULL);
            for (auto sit = samplers.begin(); sit != samplers.end(); ++sit)
            {
                vkDestroySampler(pVulkan->m_VulkanDevice, (*sit), NULL);
            }
        }
    };

    static const char* ShaderSets[ShaderPair_Count][3];

    static const uint32_t NumWeightImages = 2;

    //intermediate render targets, render contexts, and passes per swapchain image; each in-flight frame needs its own set so concurrent GPU frames 
    //do not read/write the same intermediate render targets simultaneously.
    RenderTarget                      m_IntermediateRts[NUM_VULKAN_BUFFERS][Pass_Count];
    RenderContext                     m_RenderContexts[NUM_VULKAN_BUFFERS][Pass_Count];
    PassInfo*                         m_passes[NUM_VULKAN_BUFFERS][Pass_Count];
    
    //shared across all frames (initialized then read-only)
    Uniform                           m_uniforms[Pass_Count];
    ShaderInfo                        m_shaders[ShaderPair_Count];
    Texture<Vulkan>                   m_weightTextures[NumWeightImages];
    VkImageView                       m_weightTextureViews[NumWeightImages];
    const Texture<Vulkan>*            m_sourceTexture = nullptr;
    CommandBuffer                     m_commandBuffers[NUM_VULKAN_BUFFERS];


    void FinalizePass(PassInfo* pPass);
    void DrawPass(CommandBuffer* cmd, const PassInfo* pPass, uint32_t idx);
    void BuildCmdBuffer(uint32_t idx);

    static inline uint64_t Factorial(uint32_t n)
    {
        uint64_t ret = 1;
        for (uint64_t ii = 2; ii <= n; ++ii)
        {
            ret *= ii;
        }
        return ret;
    }
    static void BuildWeightArray(uint32_t weightSize, float* pWeightArray);

    struct Ext_VK_QCOM_image_processing : public VulkanDeviceFeaturePropertiesExtensionHelper<
        VkPhysicalDeviceImageProcessingFeaturesQCOM, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_FEATURES_QCOM,
        VkPhysicalDeviceImageProcessingPropertiesQCOM, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_PROPERTIES_QCOM>
    {
        static constexpr auto Name = VK_QCOM_IMAGE_PROCESSING_EXTENSION_NAME;
        Ext_VK_QCOM_image_processing(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired) : VulkanDeviceFeaturePropertiesExtensionHelper(Name, status)
        {}

        void PopulateRequestedFeatures() override
        {
            RequestedFeatures.textureSampleWeighted = VK_TRUE;
            RequestedFeatures.textureBoxFilter = VK_TRUE;
        }

        void PrintFeatures() const override
        {
            LOGI("VkPhysicalDeviceImageProcessingFeaturesQCOM (VK_QCOM_image_processing): ");
            LOGI("    TextureWeightedSample: %s", this->AvailableFeatures.textureSampleWeighted ? "True" : "False");
            LOGI("    TextureBoxFilter: %s", this->AvailableFeatures.textureBoxFilter ? "True" : "False");
            LOGI("    TextureBlockMatch: %s", this->AvailableFeatures.textureBlockMatch ? "True" : "False");
        }

        void PrintProperties() const override
        {
            LOGI("  VkPhysicalDeviceImageProcessingPropertiesQCOM (VK_QCOM_image_processing):");
            LOGI("        maxWeightFilterPhases: %u", Properties.maxWeightFilterPhases);
            LOGI("        maxWeightFilterDimension: %u x %u", Properties.maxWeightFilterDimension.width, Properties.maxWeightFilterDimension.height);
            LOGI("        maxBlockMatchRegion: %u x %u", Properties.maxBlockMatchRegion.width, Properties.maxBlockMatchRegion.height);
            LOGI("        maxBoxFilterBlockSize: %u x %u", Properties.maxBoxFilterBlockSize.width, Properties.maxBoxFilterBlockSize.height);
        }
    };
};
