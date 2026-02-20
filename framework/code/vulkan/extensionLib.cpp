//============================================================================================================
//
//
//                  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#include "vulkan.hpp"
#include "extensionLib.hpp"
#include <inttypes.h>

namespace ExtensionLib
{

#if VK_EXT_mesh_shader
    void Ext_VK_KHR_mesh_shader::LookupFunctionPointers( VkDevice vkDevice, PFN_vkGetDeviceProcAddr deviceProcAddr )
    {
        m_vkCmdDrawMeshTasksEXT = (PFN_vkCmdDrawMeshTasksEXT)deviceProcAddr( vkDevice, "vkCmdDrawMeshTasksEXT" );
        m_vkCmdDrawMeshTasksIndirectEXT = (PFN_vkCmdDrawMeshTasksIndirectEXT)deviceProcAddr( vkDevice, "vkCmdDrawMeshTasksIndirectEXT" );
        m_vkCmdDrawMeshTasksIndirectCountEXT = (PFN_vkCmdDrawMeshTasksIndirectCountEXT)deviceProcAddr( vkDevice, "vkCmdDrawMeshTasksIndirectCountEXT" );
    }
    void Ext_VK_KHR_mesh_shader::PrintFeatures() const
    {
        LOGI( "FeaturesMeshShader: " );
        LOGI( "    taskShader: %s", this->AvailableFeatures.taskShader ? "True" : "False" );
        LOGI( "    meshShader: %s", this->AvailableFeatures.meshShader ? "True" : "False" );
        LOGI( "    multiviewMeshShader: %s", this->AvailableFeatures.multiviewMeshShader ? "True" : "False" );
        LOGI( "    primitiveFragmentShadingRateMeshShader: %s", this->AvailableFeatures.primitiveFragmentShadingRateMeshShader ? "True" : "False" );
        LOGI( "    meshShaderQueries: %s", this->AvailableFeatures.meshShaderQueries ? "True" : "False" );
    }
    void Ext_VK_KHR_mesh_shader::PrintProperties() const
    {
        LOGI( "VK_KHR_mesh_shader (VkPhysicalDeviceMeshShaderPropertiesEXT): " );
        LOGI( "    maxTaskWorkGroupTotalCount: %u", Properties.maxTaskWorkGroupTotalCount );
        LOGI( "    maxTaskWorkGroupCount: %u %u %u", Properties.maxTaskWorkGroupCount[0], Properties.maxTaskWorkGroupCount[0], Properties.maxTaskWorkGroupCount[0] );
        LOGI( "    maxTaskWorkGroupTotalCount: %u", Properties.maxTaskWorkGroupTotalCount );
        LOGI( "    maxTaskWorkGroupCount: %u %u %u", Properties.maxTaskWorkGroupCount[0], Properties.maxTaskWorkGroupCount[0], Properties.maxTaskWorkGroupCount[0] );
        LOGI( "    maxTaskWorkGroupInvocations: %u", Properties.maxTaskWorkGroupInvocations );
        LOGI( "    maxTaskWorkGroupSize: %u %u %u", Properties.maxTaskWorkGroupSize[0], Properties.maxTaskWorkGroupSize[0], Properties.maxTaskWorkGroupSize[0] );
        LOGI( "    maxTaskPayloadSize: %u", Properties.maxTaskPayloadSize );
        LOGI( "    maxTaskSharedMemorySize: %u", Properties.maxTaskSharedMemorySize );
        LOGI( "    maxTaskPayloadAndSharedMemorySize: %u", Properties.maxTaskPayloadAndSharedMemorySize );
        LOGI( "    maxMeshWorkGroupTotalCount: %u", Properties.maxMeshWorkGroupTotalCount );
        LOGI( "    maxMeshWorkGroupCount: %u %u %u", Properties.maxMeshWorkGroupCount[0], Properties.maxMeshWorkGroupCount[0], Properties.maxMeshWorkGroupCount[0] );
        LOGI( "    maxMeshWorkGroupInvocations: %u", Properties.maxMeshWorkGroupInvocations );
        LOGI( "    maxMeshWorkGroupSize: %u %u %u", Properties.maxMeshWorkGroupSize[0], Properties.maxMeshWorkGroupSize[0], Properties.maxMeshWorkGroupSize[0] );
        LOGI( "    maxMeshSharedMemorySize: %u", Properties.maxMeshSharedMemorySize );
        LOGI( "    maxMeshPayloadAndSharedMemorySize: %u", Properties.maxMeshPayloadAndSharedMemorySize );
        LOGI( "    maxMeshOutputMemorySize: %u", Properties.maxMeshOutputMemorySize );
        LOGI( "    maxMeshPayloadAndOutputMemorySize: %u", Properties.maxMeshPayloadAndOutputMemorySize );
        LOGI( "    maxMeshOutputComponents: %u", Properties.maxMeshOutputComponents );
        LOGI( "    maxMeshOutputVertices: %u", Properties.maxMeshOutputVertices );
        LOGI( "    maxMeshOutputPrimitives: %u", Properties.maxMeshOutputPrimitives );
        LOGI( "    maxMeshOutputLayers: %u", Properties.maxMeshOutputLayers );
        LOGI( "    maxMeshMultiviewViewCount: %u", Properties.maxMeshMultiviewViewCount );
        LOGI( "    meshOutputPerVertexGranularity: %u", Properties.meshOutputPerVertexGranularity );
        LOGI( "    meshOutputPerPrimitiveGranularity: %u", Properties.meshOutputPerPrimitiveGranularity );
        LOGI( "    maxPreferredTaskWorkGroupInvocations: %u", Properties.maxPreferredTaskWorkGroupInvocations );
        LOGI( "    maxPreferredMeshWorkGroupInvocations: %u", Properties.maxPreferredMeshWorkGroupInvocations );
        LOGI( "    prefersLocalInvocationVertexOutput: %s", Properties.prefersLocalInvocationVertexOutput ? "True" : "False" );
        LOGI( "    prefersLocalInvocationVertexOutput: %s", Properties.prefersLocalInvocationPrimitiveOutput ? "True" : "False" );
        LOGI( "    prefersLocalInvocationVertexOutput: %s", Properties.prefersCompactVertexOutput ? "True" : "False" );
        LOGI( "    prefersLocalInvocationVertexOutput: %s", Properties.prefersCompactPrimitiveOutput ? "True" : "False" );
    }
#endif

#if VK_KHR_swapchain

    void Ext_VK_KHR_swapchain::LookupFunctionPointers( VkDevice vkDevice, PFN_vkGetDeviceProcAddr deviceProcAddr )
    {
        m_vkCreateSwapchainKHR = (PFN_vkCreateSwapchainKHR)deviceProcAddr( vkDevice, "vkCreateSwapchainKHR" );
        m_vkDestroySwapchainKHR = (PFN_vkDestroySwapchainKHR)deviceProcAddr( vkDevice, "vkDestroySwapchainKHR" );
        m_vkGetSwapchainImagesKHR = (PFN_vkGetSwapchainImagesKHR)deviceProcAddr( vkDevice, "vkGetSwapchainImagesKHR" );
        m_vkAcquireNextImageKHR = (PFN_vkAcquireNextImageKHR)deviceProcAddr( vkDevice, "vkAcquireNextImageKHR" );
        m_vkQueuePresentKHR = (PFN_vkQueuePresentKHR)deviceProcAddr( vkDevice, "vkQueuePresentKHR" );
        m_vkGetDeviceGroupPresentCapabilitiesKHR = (PFN_vkGetDeviceGroupPresentCapabilitiesKHR)deviceProcAddr( vkDevice, "vkGetDeviceGroupPresentCapabilitiesKHR" );
        m_vkGetDeviceGroupSurfacePresentModesKHR = (PFN_vkGetDeviceGroupSurfacePresentModesKHR)deviceProcAddr( vkDevice, "vkGetDeviceGroupSurfacePresentModesKHR" );
        m_vkAcquireNextImage2KHR = (PFN_vkAcquireNextImage2KHR)deviceProcAddr( vkDevice, "vkAcquireNextImage2KHR" );
    }
    void Ext_VK_KHR_swapchain::LookupFunctionPointers( VkInstance vkInstance )
    {
        m_vkGetPhysicalDevicePresentRectanglesKHR = (PFN_vkGetPhysicalDevicePresentRectanglesKHR)vkGetInstanceProcAddr( vkInstance, "vkGetPhysicalDevicePresentRectanglesKHR" );
    }

#endif // VK_KHR_swapchain


#if VK_KHR_shader_float16_int8
    void Ext_VK_KHR_shader_float16_int8::PrintFeatures() const
    {
        LOGI( "FeaturesShaderFloat16Int8: " );
        LOGI( "    shaderFloat16: %s", this->AvailableFeatures.shaderFloat16 ? "True" : "False" );
        LOGI( "    shaderInt8: %s", this->AvailableFeatures.shaderInt8 ? "True" : "False" );
    }
#endif // VK_KHR_shader_float16_int8

#if VK_EXT_shader_image_atomic_int64
    void Ext_VK_EXT_shader_image_atomic_int64::PrintFeatures() const
    {
        LOGI( "FeaturesShaderImageAtomicInt64: " );
        LOGI( "    shaderImageInt64Atomics: %s", this->AvailableFeatures.shaderImageInt64Atomics ? "True" : "False" );
        LOGI( "    sparseImageInt64Atomics: %s", this->AvailableFeatures.sparseImageInt64Atomics ? "True" : "False" );
    }
#endif // VK_EXT_shader_image_atomic_int64

#if VK_KHR_ray_tracing_position_fetch
    void Ext_VK_KHR_ray_tracing_position_fetch::PrintFeatures() const
    {
        LOGI( "FeaturesRayTracingPositionFetch: " );
        LOGI( "    rayTracingPositionFetch: %s", this->AvailableFeatures.rayTracingPositionFetch ? "True" : "False" );
    }
#endif // VK_EXT_shader_image_atomic_int64

#if VK_EXT_scalar_block_layout

    void Ext_VK_EXT_scalar_block_layout::PrintFeatures() const
    {
        LOGI( "FeaturesScalarBlockLayout: " );
        LOGI( "    scalarBlockLayout: %s", this->AvailableFeatures.scalarBlockLayout ? "True" : "False" );
    }

#endif // VK_EXT_scalar_block_layout


#if VK_EXT_index_type_uint8
    void Ext_VK_EXT_index_type_uint8::PrintFeatures() const
    {
        LOGI( "VK_EXT_index_type_uint8 (VkPhysicalDeviceIndexTypeUint8FeaturesEXT): " );
        LOGI( "    indexTypeUint8: %s", this->AvailableFeatures.indexTypeUint8 ? "True" : "False" );
    }
#endif // VK_EXT_index_type_uint8

#if VK_KHR_shader_subgroup_extended_types
    void Ext_VK_KHR_shader_subgroup_extended_types::PrintFeatures() const
    {
        LOGI( "FeaturesShaderSubgroupExtendedTypes: " );
        LOGI( "    shaderSubgroupExtendedTypes: %s", this->AvailableFeatures.shaderSubgroupExtendedTypes ? "True" : "False" );
    }
#endif // VK_KHR_shader_subgroup_extended_types

#if VK_EXT_descriptor_indexing
    void Ext_VK_EXT_descriptor_indexing::PrintFeatures() const
    {
        LOGI( "VK_EXT_descriptor_indexing (VkPhysicalDeviceDescriptorIndexingFeatures): " );
        LOGI( "    shaderInputAttachmentArrayDynamicIndexing: %s", this->AvailableFeatures.shaderInputAttachmentArrayDynamicIndexing ? "True" : "False" );
        LOGI( "    shaderUniformTexelBufferArrayDynamicIndexing: %s", this->AvailableFeatures.shaderUniformTexelBufferArrayDynamicIndexing ? "True" : "False" );
        LOGI( "    shaderStorageTexelBufferArrayDynamicIndexing: %s", this->AvailableFeatures.shaderStorageTexelBufferArrayDynamicIndexing ? "True" : "False" );
        LOGI( "    shaderUniformBufferArrayNonUniformIndexing: %s", this->AvailableFeatures.shaderUniformBufferArrayNonUniformIndexing ? "True" : "False" );
        LOGI( "    shaderSampledImageArrayNonUniformIndexing: %s", this->AvailableFeatures.shaderSampledImageArrayNonUniformIndexing ? "True" : "False" );
        LOGI( "    shaderStorageBufferArrayNonUniformIndexing: %s", this->AvailableFeatures.shaderStorageBufferArrayNonUniformIndexing ? "True" : "False" );
        LOGI( "    shaderStorageImageArrayNonUniformIndexing: %s", this->AvailableFeatures.shaderStorageImageArrayNonUniformIndexing ? "True" : "False" );
        LOGI( "    shaderInputAttachmentArrayNonUniformIndexing: %s", this->AvailableFeatures.shaderInputAttachmentArrayNonUniformIndexing ? "True" : "False" );
        LOGI( "    shaderUniformTexelBufferArrayNonUniformIndexing: %s", this->AvailableFeatures.shaderUniformTexelBufferArrayNonUniformIndexing ? "True" : "False" );
        LOGI( "    shaderStorageTexelBufferArrayNonUniformIndexing: %s", this->AvailableFeatures.shaderStorageTexelBufferArrayNonUniformIndexing ? "True" : "False" );
        LOGI( "    descriptorBindingUniformBufferUpdateAfterBind: %s", this->AvailableFeatures.descriptorBindingUniformBufferUpdateAfterBind ? "True" : "False" );
        LOGI( "    descriptorBindingSampledImageUpdateAfterBind: %s", this->AvailableFeatures.descriptorBindingSampledImageUpdateAfterBind ? "True" : "False" );
        LOGI( "    descriptorBindingStorageImageUpdateAfterBind: %s", this->AvailableFeatures.descriptorBindingStorageImageUpdateAfterBind ? "True" : "False" );
        LOGI( "    descriptorBindingStorageBufferUpdateAfterBind: %s", this->AvailableFeatures.descriptorBindingStorageBufferUpdateAfterBind ? "True" : "False" );
        LOGI( "    descriptorBindingUniformTexelBufferUpdateAfterBind: %s", this->AvailableFeatures.descriptorBindingUniformTexelBufferUpdateAfterBind ? "True" : "False" );
        LOGI( "    descriptorBindingStorageTexelBufferUpdateAfterBind: %s", this->AvailableFeatures.descriptorBindingStorageTexelBufferUpdateAfterBind ? "True" : "False" );
        LOGI( "    descriptorBindingUpdateUnusedWhilePending: %s", this->AvailableFeatures.descriptorBindingUpdateUnusedWhilePending ? "True" : "False" );
        LOGI( "    descriptorBindingPartiallyBound: %s", this->AvailableFeatures.descriptorBindingPartiallyBound ? "True" : "False" );
        LOGI( "    descriptorBindingVariableDescriptorCount: %s", this->AvailableFeatures.descriptorBindingVariableDescriptorCount ? "True" : "False" );
        LOGI( "    runtimeDescriptorArray: %s", this->AvailableFeatures.runtimeDescriptorArray ? "True" : "False" );
    }

    void Ext_VK_EXT_descriptor_indexing::PrintProperties() const
    {
        LOGI( "VK_EXT_descriptor_indexing (VkPhysicalDeviceDescriptorIndexingProperties): " );
        LOGI( "    maxUpdateAfterBindDescriptorsInAllPools: %u", Properties.maxUpdateAfterBindDescriptorsInAllPools );
        LOGI( "    shaderUniformBufferArrayNonUniformIndexingNative: %s", Properties.shaderUniformBufferArrayNonUniformIndexingNative ? "True" : "False" );
        LOGI( "    shaderSampledImageArrayNonUniformIndexingNative: %s", Properties.shaderSampledImageArrayNonUniformIndexingNative ? "True" : "False" );
        LOGI( "    shaderStorageBufferArrayNonUniformIndexingNative: %s", Properties.shaderStorageBufferArrayNonUniformIndexingNative ? "True" : "False" );
        LOGI( "    shaderStorageImageArrayNonUniformIndexingNative: %s", Properties.shaderStorageImageArrayNonUniformIndexingNative ? "True" : "False" );
        LOGI( "    shaderInputAttachmentArrayNonUniformIndexingNative: %s", Properties.shaderInputAttachmentArrayNonUniformIndexingNative ? "True" : "False" );
        LOGI( "    robustBufferAccessUpdateAfterBind: %s", Properties.robustBufferAccessUpdateAfterBind ? "True" : "False" );
        LOGI( "    quadDivergentImplicitLod: %s", Properties.quadDivergentImplicitLod ? "True" : "False" );
        LOGI( "    maxPerStageDescriptorUpdateAfterBindSamplers: %u", Properties.maxPerStageDescriptorUpdateAfterBindSamplers );
        LOGI( "    maxPerStageDescriptorUpdateAfterBindUniformBuffers: %u", Properties.maxPerStageDescriptorUpdateAfterBindUniformBuffers );
        LOGI( "    maxPerStageDescriptorUpdateAfterBindStorageBuffers: %u", Properties.maxPerStageDescriptorUpdateAfterBindStorageBuffers );
        LOGI( "    maxPerStageDescriptorUpdateAfterBindSampledImages: %u", Properties.maxPerStageDescriptorUpdateAfterBindSampledImages );
        LOGI( "    maxPerStageDescriptorUpdateAfterBindStorageImages: %u", Properties.maxPerStageDescriptorUpdateAfterBindStorageImages );
        LOGI( "    maxPerStageDescriptorUpdateAfterBindInputAttachments: %u", Properties.maxPerStageDescriptorUpdateAfterBindInputAttachments );
        LOGI( "    maxPerStageUpdateAfterBindResources: %u", Properties.maxPerStageUpdateAfterBindResources );
        LOGI( "    maxDescriptorSetUpdateAfterBindSamplers: %u", Properties.maxDescriptorSetUpdateAfterBindSamplers );
        LOGI( "    maxDescriptorSetUpdateAfterBindUniformBuffers: %u", Properties.maxDescriptorSetUpdateAfterBindUniformBuffers );
        LOGI( "    maxDescriptorSetUpdateAfterBindUniformBuffersDynamic: %u", Properties.maxDescriptorSetUpdateAfterBindUniformBuffersDynamic );
        LOGI( "    maxDescriptorSetUpdateAfterBindStorageBuffers: %u", Properties.maxDescriptorSetUpdateAfterBindStorageBuffers );
        LOGI( "    maxDescriptorSetUpdateAfterBindStorageBuffersDynamic: %u", Properties.maxDescriptorSetUpdateAfterBindStorageBuffersDynamic );
        LOGI( "    maxDescriptorSetUpdateAfterBindSampledImages: %u", Properties.maxDescriptorSetUpdateAfterBindSampledImages );
        LOGI( "    maxDescriptorSetUpdateAfterBindStorageImages: %u", Properties.maxDescriptorSetUpdateAfterBindStorageImages );
        LOGI( "    maxDescriptorSetUpdateAfterBindInputAttachments: %u", Properties.maxDescriptorSetUpdateAfterBindInputAttachments );
    }
#endif // VK_EXT_descriptor_indexing

#if VK_KHR_8bit_storage
    void Ext_VK_KHR_8bit_storage::PrintFeatures() const
    {
        LOGI( "VK_KHR_8bit_storage (VkPhysicalDevice8BitStorageFeaturesKHR): " );
        LOGI( "    storageBuffer8BitAccess: %s", this->AvailableFeatures.storageBuffer8BitAccess ? "True" : "False" );
        LOGI( "    uniformAndStorageBuffer8BitAccess: %s", this->AvailableFeatures.uniformAndStorageBuffer8BitAccess ? "True" : "False" );
        LOGI( "    storagePushConstant8: %s", this->AvailableFeatures.storagePushConstant8 ? "True" : "False" );
    }
#endif // VK_KHR_8bit_storage

#if VK_KHR_portability_subset
    void Ext_VK_KHR_portability_subset::PrintFeatures() const
    {
        LOGI( "VK_KHR_portability_subset (VkPhysicalDevicePortabilitySubsetFeaturesKHR): " );
        LOGI( "    constantAlphaColorBlendFactors: %s", this->AvailableFeatures.constantAlphaColorBlendFactors ? "True" : "False" );
        LOGI( "    events: %s", this->AvailableFeatures.events ? "True" : "False" );
        LOGI( "    imageViewFormatReinterpretation: %s", this->AvailableFeatures.imageViewFormatReinterpretation ? "True" : "False" );
        LOGI( "    imageViewFormatSwizzle: %s", this->AvailableFeatures.imageViewFormatSwizzle ? "True" : "False" );
        LOGI( "    imageView2DOn3DImage: %s", this->AvailableFeatures.imageView2DOn3DImage ? "True" : "False" );
        LOGI( "    multisampleArrayImage: %s", this->AvailableFeatures.multisampleArrayImage ? "True" : "False" );
        LOGI( "    mutableComparisonSamplers: %s", this->AvailableFeatures.mutableComparisonSamplers ? "True" : "False" );
        LOGI( "    pointPolygons: %s", this->AvailableFeatures.pointPolygons ? "True" : "False" );
        LOGI( "    samplerMipLodBias: %s", this->AvailableFeatures.samplerMipLodBias ? "True" : "False" );
        LOGI( "    separateStencilMaskRef: %s", this->AvailableFeatures.separateStencilMaskRef ? "True" : "False" );
        LOGI( "    shaderSampleRateInterpolationFunctions: %s", this->AvailableFeatures.shaderSampleRateInterpolationFunctions ? "True" : "False" );
        LOGI( "    tessellationIsolines: %s", this->AvailableFeatures.tessellationIsolines ? "True" : "False" );
        LOGI( "    tessellationPointMode: %s", this->AvailableFeatures.tessellationPointMode ? "True" : "False" );
        LOGI( "    triangleFans: %s", this->AvailableFeatures.triangleFans ? "True" : "False" );
        LOGI( "    vertexAttributeAccessBeyondStride: %s", this->AvailableFeatures.vertexAttributeAccessBeyondStride ? "True" : "False" );
    }
    void Ext_VK_KHR_portability_subset::PrintProperties() const
    {
        LOGI( "VK_KHR_portability_subset (VkPhysicalDevicePortabilitySubsetPropertiesKHR): " );
        LOGI( "    minVertexInputBindingStrideAlignment: %u", this->Properties.minVertexInputBindingStrideAlignment );
    }
#endif // VK_KHR_portability_subset

#if VK_KHR_fragment_shading_rate
    void Ext_VK_KHR_fragment_shading_rate::PrintFeatures() const
    {
        LOGI( "VK_KHR_fragment_shading_rate (VkPhysicalDeviceFragmentShadingRateFeaturesKHR): " );
        LOGI( "    pipelineFragmentShadingRate: %s", this->AvailableFeatures.pipelineFragmentShadingRate ? "True" : "False" );
        LOGI( "    primitiveFragmentShadingRate: %s", this->AvailableFeatures.primitiveFragmentShadingRate ? "True" : "False" );
        LOGI( "    attachmentFragmentShadingRate: %s", this->AvailableFeatures.attachmentFragmentShadingRate ? "True" : "False" );
    }
    void Ext_VK_KHR_fragment_shading_rate::PrintProperties() const
    {
        LOGI( "VK_KHR_fragment_shading_rate (VkPhysicalDeviceFragmentShadingRatePropertiesKHR): " );
        LOGI( "    minFragmentDensityTexelSize: %ux%u", this->Properties.minFragmentShadingRateAttachmentTexelSize.width, this->Properties.minFragmentShadingRateAttachmentTexelSize.height );
        LOGI( "    maxFragmentDensityTexelSize: %ux%u", this->Properties.maxFragmentShadingRateAttachmentTexelSize.width, this->Properties.maxFragmentShadingRateAttachmentTexelSize.height );
        LOGI( "    maxFragmentShadingRateAttachmentTexelSizeAspectRatio: %u", this->Properties.maxFragmentShadingRateAttachmentTexelSizeAspectRatio );
        LOGI( "    primitiveFragmentShadingRateWithMultipleViewports: %s", this->Properties.primitiveFragmentShadingRateWithMultipleViewports ? "True" : "False" );
        LOGI( "    layeredShadingRateAttachments: %s", this->Properties.layeredShadingRateAttachments ? "True" : "False" );
        LOGI( "    fragmentShadingRateNonTrivialCombinerOps: %s", this->Properties.fragmentShadingRateNonTrivialCombinerOps ? "True" : "False" );
        LOGI( "    maxFragmentSize: %ux%u", this->Properties.maxFragmentSize.width, this->Properties.maxFragmentSize.height );
        LOGI( "    maxFragmentSizeAspectRatio: %u", this->Properties.maxFragmentSizeAspectRatio );
        LOGI( "    maxFragmentShadingRateCoverageSamples: %u", this->Properties.maxFragmentShadingRateCoverageSamples );
        LOGI( "    maxFragmentShadingRateRasterizationSamples: 0x%02x", this->Properties.maxFragmentShadingRateRasterizationSamples );
        LOGI( "    fragmentShadingRateWithShaderDepthStencilWrites: %s", this->Properties.fragmentShadingRateWithShaderDepthStencilWrites ? "True" : "False" );
        LOGI( "    fragmentShadingRateWithSampleMask: %s", this->Properties.fragmentShadingRateWithSampleMask ? "True" : "False" );
        LOGI( "    fragmentShadingRateWithShaderSampleMask: %s", this->Properties.fragmentShadingRateWithShaderSampleMask ? "True" : "False" );
        LOGI( "    fragmentShadingRateWithConservativeRasterization: %s", this->Properties.fragmentShadingRateWithConservativeRasterization ? "True" : "False" );
        LOGI( "    fragmentShadingRateWithFragmentShaderInterlock: %s", this->Properties.fragmentShadingRateWithFragmentShaderInterlock ? "True" : "False" );
        LOGI( "    fragmentShadingRateWithCustomSampleLocations: %s", this->Properties.fragmentShadingRateWithCustomSampleLocations ? "True" : "False" );
        LOGI( "    fragmentShadingRateStrictMultiplyCombiner: %s", this->Properties.fragmentShadingRateStrictMultiplyCombiner ? "True" : "False" );
    }
#endif // VK_KHR_fragment_shading_rate

#if VK_KHR_cooperative_matrix
    void Ext_VK_KHR_cooperative_matrix::PrintFeatures() const
    {
        LOGI("VK_KHR_fragment_shading_rate (VkPhysicalDeviceCooperativeMatrixFeaturesKHR): ");
        LOGI("    pipelineFragmentShadingRate: %s", this->AvailableFeatures.cooperativeMatrix ? "True" : "False");
        LOGI("    primitiveFragmentShadingRate: %s", this->AvailableFeatures.cooperativeMatrixRobustBufferAccess ? "True" : "False");
    }
    void Ext_VK_KHR_cooperative_matrix::PrintProperties() const
    {
        LOGI("VK_KHR_fragment_shading_rate (VkPhysicalDeviceFragmentShadingRatePropertiesKHR): ");
        LOGI("    cooperativeMatrixSupportedStages: %d", this->Properties.cooperativeMatrixSupportedStages);
    }
#endif // VK_KHR_cooperative_matrix

#if VK_KHR_create_renderpass2
    void Ext_VK_KHR_create_renderpass2::LookupFunctionPointers( VkInstance vkInstance )
    {
        m_vkCreateRenderPass2KHR = (PFN_vkCreateRenderPass2KHR)vkGetInstanceProcAddr( vkInstance, "vkCreateRenderPass2KHR" );
        m_vkCmdBeginRenderPass2KHR = (PFN_vkCmdBeginRenderPass2KHR)vkGetInstanceProcAddr( vkInstance, "vkCmdBeginRenderPass2KHR" );
        m_vkCmdNextSubpass2KHR = (PFN_vkCmdNextSubpass2KHR)vkGetInstanceProcAddr( vkInstance, "vkCmdNextSubpass2KHR" );
        m_vkCmdEndRenderPass2KHR = (PFN_vkCmdEndRenderPass2KHR)vkGetInstanceProcAddr( vkInstance, "vkCmdEndRenderPass2KHR" );
    }
#endif // VK_KHR_create_renderpass2

#if VK_KHR_draw_indirect_count
    void Ext_VK_KHR_draw_indirect_count::LookupFunctionPointers( VkInstance vkInstance )
    {
        m_vkCmdDrawIndirectCountKHR = (PFN_vkCmdDrawIndirectCountKHR)vkGetInstanceProcAddr( vkInstance, "vkCmdDrawIndirectCountKHR" );
        m_vkCmdDrawIndexedIndirectCountKHR = (PFN_vkCmdDrawIndexedIndirectCountKHR)vkGetInstanceProcAddr( vkInstance, "vkCmdDrawIndexedIndirectCountKHR" );
    }
#endif // VK_KHR_draw_indirect_count

#if VK_KHR_dynamic_rendering
    void Ext_VK_KHR_dynamic_rendering::PrintFeatures() const
    {
        LOGI( "VK_KHR_dynamic_rendering (VkPhysicalDeviceDynamicRenderingFeaturesKHR): " );
        LOGI( "    dynamicRendering: %s", this->AvailableFeatures.dynamicRendering ? "True" : "False" );
    }
#endif // VK_KHR_dynamic_rendering

#if VK_KHR_dynamic_rendering_local_read
    void Ext_VK_KHR_dynamic_rendering_local_read::PrintFeatures() const
    {
        LOGI( "VK_KHR_dynamic_rendering_local_read (VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR): " );
        LOGI( "    dynamicRenderingLocalRead: %s", this->AvailableFeatures.dynamicRenderingLocalRead ? "True" : "False" );
    }
#endif // VK_KHR_dynamic_rendering_local_read

#if VK_EXT_debug_utils
    bool Ext_VK_EXT_debug_utils::SetDebugUtilsObjectName( VkDevice vkDevice, uint64_t object, VkObjectType objectType, const char* name ) const
    {
        if (m_vkSetDebugUtilsObjectNameEXT == nullptr)
            return false;
        VkDebugUtilsObjectNameInfoEXT nameInfo { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
        nameInfo.objectType = objectType;
        nameInfo.objectHandle = object;
        nameInfo.pObjectName = name;
        return m_vkSetDebugUtilsObjectNameEXT( vkDevice, &nameInfo ) == VK_SUCCESS;
    }
#endif // VK_EXT_debug_utils

#if VK_EXT_debug_marker
    bool Ext_VK_EXT_debug_marker::DebugMarkerSetObjectName( VkDevice vkDevice, uint64_t object, VkDebugReportObjectTypeEXT objectType, const char* name ) const
    {
        if (m_vkDebugMarkerSetObjectNameEXT == nullptr)
            return false;
        VkDebugMarkerObjectNameInfoEXT markerInfo { VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_NAME_INFO_EXT };
        markerInfo.objectType = objectType;
        markerInfo.object = object;
        markerInfo.pObjectName = name;
        return m_vkDebugMarkerSetObjectNameEXT( vkDevice, &markerInfo ) == VK_SUCCESS;
    }
#endif // VK_EXT_debug_marker

    static std::string GetSubgroupStagesString( VkShaderStageFlags Stages )
    {
        std::string StagesString;
        static const char* SubgroupStages[] = { "Vertex", "Tessellation Control", "Tessellation Evaluation", "Geometry", "Fragment", "Compute" };
        int i = 0;
        for (; i < sizeof( SubgroupStages ) / sizeof( *SubgroupStages ); ++i)
        {
            if ((Stages & (1 << i)) != 0)
            {
                StagesString.append( SubgroupStages[i] );
                StagesString.append( ", " );
            }
        }
        // remainder
        if ((Stages & ~((1 << i) - 1)) != 0)
        {
            StagesString.append( std::to_string( Stages & ~((1 << i) - 1) ) );
        }
        return StagesString;
    }


#if VK_EXT_subgroup_size_control
    void Ext_VK_EXT_subgroup_size_control::PrintFeatures() const
    {
        LOGI( "VK_EXT_subgroup_size_control (VkPhysicalDeviceSubgroupSizeControlFeaturesEXT): " );
        LOGI( "    subgroupSizeControl: %s", this->AvailableFeatures.subgroupSizeControl ? "True" : "False" );
        LOGI( "    computeFullSubgroups: %s", this->AvailableFeatures.computeFullSubgroups ? "True" : "False" );
    }

    void Ext_VK_EXT_subgroup_size_control::PrintProperties() const
    {
        LOGI( "VK_EXT_subgroup_size_control (VkPhysicalDeviceSubgroupSizeControlPropertiesEXT): " );
        LOGI( "    minSubgroupSize: %u", this->Properties.minSubgroupSize );
        LOGI( "    maxSubgroupSize: %u", this->Properties.maxSubgroupSize );
        LOGI( "    maxComputeWorkgroupSubgroups: %u", this->Properties.maxComputeWorkgroupSubgroups );
        auto SupportedStages = GetSubgroupStagesString( this->Properties.requiredSubgroupSizeStages );
        LOGI( "    requiredSubgroupSizeStages: %s", SupportedStages.c_str() );
    }
#endif // VK_EXT_subgroup_size_control

#if VK_EXT_host_query_reset
    void Ext_VK_EXT_host_query_reset::PrintFeatures() const
    {
        LOGI( "VK_EXT_host_query_reset (VkPhysicalDeviceHostQueryResetFeaturesEXT): " );
        LOGI( "    hostQueryReset: %s", this->AvailableFeatures.hostQueryReset ? "True" : "False" );
    }
#endif // VK_EXT_host_query_reset

#if VK_KHR_timeline_semaphore
    void Ext_VK_KHR_timeline_semaphore::PrintFeatures() const
    {
        LOGI("VK_KHR_timeline_semaphore (VkPhysicalDeviceTimelineSemaphoreFeaturesKHR): ");
        LOGI("    timelineSemaphore: %s", this->AvailableFeatures.timelineSemaphore ? "True" : "False");
    }
    void Ext_VK_KHR_timeline_semaphore::PrintProperties() const
    {
        LOGI("VK_KHR_timeline_semaphore (VkPhysicalDeviceTimelineSemaphorePropertiesKHR): ");
        LOGI("    maxTimelineSemaphoreValueDifference: %" PRIu64, this->Properties.maxTimelineSemaphoreValueDifference);
    }
#endif // VK_KHR_timeline_semaphore

#if VK_KHR_synchronization2
    void Ext_VK_KHR_synchronization2::PrintFeatures() const
    {
        LOGI("VK_KHR_synchronization2 (VkPhysicalDeviceSynchronization2FeaturesKHR): ");
        LOGI("    synchronization2: %s", this->AvailableFeatures.synchronization2 ? "True" : "False");
    }
    void Ext_VK_KHR_synchronization2::LookupFunctionPointers(VkInstance vkInstance)
    {
        m_vkQueueSubmit2KHR = (PFN_vkQueueSubmit2KHR)vkGetInstanceProcAddr(vkInstance, "vkQueueSubmit2KHR");
    }
#endif // VK_KHR_synchronization2

#if VK_ARM_tensors
    void Ext_VK_ARM_tensors::PrintFeatures() const
    {
        LOGI("VK_ARM_tensors (VkPhysicalDeviceTensorFeaturesARM):");
        LOGI("    tensorNonPacked: %s", this->AvailableFeatures.tensorNonPacked ? "True" : "False");
        LOGI("    shaderTensorAccess: %s", this->AvailableFeatures.shaderTensorAccess ? "True" : "False");
        LOGI("    shaderStorageTensorArrayDynamicIndexing: %s", this->AvailableFeatures.shaderStorageTensorArrayDynamicIndexing ? "True" : "False");
        LOGI("    shaderStorageTensorArrayNonUniformIndexing: %s", this->AvailableFeatures.shaderStorageTensorArrayNonUniformIndexing ? "True" : "False");
        LOGI("    descriptorBindingStorageTensorUpdateAfterBind: %s", this->AvailableFeatures.descriptorBindingStorageTensorUpdateAfterBind ? "True" : "False");
        LOGI("    tensors: %s", this->AvailableFeatures.tensors ? "True" : "False");
    }

    void Ext_VK_ARM_tensors::PrintProperties() const
    {
        LOGI("VK_ARM_tensors (VkPhysicalDeviceTensorPropertiesARM):");
        LOGI("    maxTensorDimensionCount: %u", this->Properties.maxTensorDimensionCount);
        LOGI("    maxTensorElements: %" PRIu64, this->Properties.maxTensorElements);
        LOGI("    maxPerDimensionTensorElements: %" PRIu64, this->Properties.maxPerDimensionTensorElements);
        LOGI("    maxTensorStride: %" PRId64, this->Properties.maxTensorStride);
        LOGI("    maxTensorSize: %" PRIu64, this->Properties.maxTensorSize);
        LOGI("    maxTensorShaderAccessArrayLength: %u", this->Properties.maxTensorShaderAccessArrayLength);
        LOGI("    maxTensorShaderAccessSize: %u", this->Properties.maxTensorShaderAccessSize);
        LOGI("    maxDescriptorSetStorageTensors: %u", this->Properties.maxDescriptorSetStorageTensors);
        LOGI("    maxPerStageDescriptorSetStorageTensors: %u", this->Properties.maxPerStageDescriptorSetStorageTensors);
        LOGI("    maxDescriptorSetUpdateAfterBindStorageTensors: %u", this->Properties.maxDescriptorSetUpdateAfterBindStorageTensors);
        LOGI("    maxPerStageDescriptorUpdateAfterBindStorageTensors: %u", this->Properties.maxPerStageDescriptorUpdateAfterBindStorageTensors);
        LOGI("    shaderStorageTensorArrayNonUniformIndexingNative: %s", this->Properties.shaderStorageTensorArrayNonUniformIndexingNative ? "True" : "False");
        LOGI("    shaderTensorSupportedStages: 0x%08X", this->Properties.shaderTensorSupportedStages);
    }
#endif // VK_ARM_tensors

#if VK_ARM_data_graph
    void Ext_VK_ARM_data_graph::PrintFeatures() const
    {
        LOGI("VK_ARM_data_graph (VkPhysicalDeviceDataGraphFeaturesARM):");
        LOGI("    dataGraph: %s", this->AvailableFeatures.dataGraph ? "True" : "False");
        LOGI("    dataGraphUpdateAfterBind: %s", this->AvailableFeatures.dataGraphUpdateAfterBind ? "True" : "False");
        LOGI("    dataGraphSpecializationConstants: %s", this->AvailableFeatures.dataGraphSpecializationConstants ? "True" : "False");
        LOGI("    dataGraphDescriptorBuffer: %s", this->AvailableFeatures.dataGraphDescriptorBuffer ? "True" : "False");
        LOGI("    dataGraphShaderModule: %s", this->AvailableFeatures.dataGraphShaderModule ? "True" : "False");
    }
#endif // VK_ARM_data_graph

#if VK_QCOM_data_graph_model
    void Ext_VK_QCOM_data_graph_model::PrintFeatures() const
    {
        LOGI("VK_QCOM_data_graph_model (VkPhysicalDeviceDataGraphModelFeaturesQCOM):");
        LOGI("    dataGraphModel: %s", this->AvailableFeatures.dataGraphModel ? "True" : "False");
    }
#endif // VK_QCOM_data_graph_model

#if VK_KHR_get_physical_device_properties2

    void Ext_VK_KHR_get_physical_device_properties2::LookupFunctionPointers(VkInstance vkInstance)
    {
        m_vkGetPhysicalDeviceFeatures2KHR = (PFN_vkGetPhysicalDeviceFeatures2KHR)vkGetInstanceProcAddr(vkInstance, "vkGetPhysicalDeviceFeatures2KHR");
        m_vkGetPhysicalDeviceFormatProperties2KHR = (PFN_vkGetPhysicalDeviceFormatProperties2KHR)vkGetInstanceProcAddr( vkInstance, "vkGetPhysicalDeviceFormatProperties2KHR" );
        m_vkGetPhysicalDeviceImageFormatProperties2KHR = (PFN_vkGetPhysicalDeviceImageFormatProperties2KHR)vkGetInstanceProcAddr( vkInstance, "vkGetPhysicalDeviceImageFormatProperties2KHR" );
        m_vkGetPhysicalDeviceMemoryProperties2KHR = (PFN_vkGetPhysicalDeviceMemoryProperties2KHR)vkGetInstanceProcAddr( vkInstance, "vkGetPhysicalDeviceMemoryProperties2KHR" );
        m_vkGetPhysicalDeviceProperties2KHR = (PFN_vkGetPhysicalDeviceProperties2KHR)vkGetInstanceProcAddr( vkInstance, "vkGetPhysicalDeviceProperties2KHR" );
    }

#endif // VK_KHR_get_physical_device_properties2

#if VK_KHR_surface

    void Ext_VK_KHR_surface::LookupFunctionPointers( VkInstance vkInstance )
    {
        m_vkDestroySurfaceKHR = (PFN_vkDestroySurfaceKHR)vkGetInstanceProcAddr( vkInstance, "vkDestroySurfaceKHR" );
        m_vkGetPhysicalDeviceSurfaceCapabilitiesKHR = (PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)vkGetInstanceProcAddr( vkInstance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR" );
        m_vkGetPhysicalDeviceSurfaceFormatsKHR = (PFN_vkGetPhysicalDeviceSurfaceFormatsKHR)vkGetInstanceProcAddr( vkInstance, "vkGetPhysicalDeviceSurfaceFormatsKHR" );
        m_vkGetPhysicalDeviceSurfacePresentModesKHR = (PFN_vkGetPhysicalDeviceSurfacePresentModesKHR)vkGetInstanceProcAddr( vkInstance, "vkGetPhysicalDeviceSurfacePresentModesKHR" );
        m_vkGetPhysicalDeviceSurfaceSupportKHR = (PFN_vkGetPhysicalDeviceSurfaceSupportKHR)vkGetInstanceProcAddr( vkInstance, "vkGetPhysicalDeviceSurfaceSupportKHR" );
    }

#endif // VK_KHR_surface

#if VK_KHR_get_surface_capabilities2

    void Ext_VK_KHR_get_surface_capabilities2::LookupFunctionPointers( VkInstance vkInstance )
    {
        m_vkGetPhysicalDeviceSurfaceCapabilities2KHR = (PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR)vkGetInstanceProcAddr(vkInstance, "vkGetPhysicalDeviceSurfaceCapabilities2KHR");
        m_vkGetPhysicalDeviceSurfaceFormats2KHR = (PFN_vkGetPhysicalDeviceSurfaceFormats2KHR)vkGetInstanceProcAddr(vkInstance, "vkGetPhysicalDeviceSurfaceFormats2KHR");
    }

#endif // VK_KHR_get_surface_capabilities2

#if VK_QCOM_tile_properties

    void Ext_VK_QCOM_tile_properties::LookupFunctionPointers( VkDevice vkDevice, PFN_vkGetDeviceProcAddr deviceProcAddr )
    {
        m_vkGetDynamicRenderingTilePropertiesQCOM = (PFN_vkGetDynamicRenderingTilePropertiesQCOM)deviceProcAddr( vkDevice, "vkGetDynamicRenderingTilePropertiesQCOM" );
        m_vkGetFramebufferTilePropertiesQCOM = (PFN_vkGetFramebufferTilePropertiesQCOM)deviceProcAddr( vkDevice, "vkGetFramebufferTilePropertiesQCOM" );
    }

    void Ext_VK_QCOM_tile_properties::PrintFeatures() const
    {
        LOGI( "VK_QCOM_tile_properties (VkPhysicalDeviceTilePropertiesFeaturesQCOM): " );
        LOGI( "    tileProperties: %s", this->AvailableFeatures.tileProperties ? "True" : "False" );
    }

#endif // VK_QCOM_tile_properties

#if VK_QCOM_tile_memory_heap
    void Ext_VK_QCOM_tile_memory_heap::PrintFeatures() const
    {
        LOGI("VK_QCOM_tile_memory_heap (VkPhysicalDeviceTileMemoryHeapFeaturesQCOM): ");
        LOGI("    tileMemoryHeap: %s", this->AvailableFeatures.tileMemoryHeap ? "True" : "False");
    }

    void Ext_VK_QCOM_tile_memory_heap::PrintProperties() const
    {
        LOGI("VK_QCOM_tile_memory_heap (VkPhysicalDeviceTileMemoryHeapPropertiesQCOM): ");
        LOGI("    queueSubmitBoundary: %s", this->Properties.queueSubmitBoundary ? "True" : "False");
        LOGI("    tileBufferTransfers: %s", this->Properties.tileBufferTransfers ? "True" : "False");
    }
#endif // VK_KHR_fragment_shading_rate

#if VK_QCOM_tile_shading

    void Ext_VK_QCOM_tile_shading::PopulateRequestedFeatures()
    {
        tBase::PopulateRequestedFeatures();
        RequestedFeatures.sType = AvailableFeatures.sType;
        RequestedFeatures.tileShading = AvailableFeatures.tileShading;
        RequestedFeatures.tileShadingPerTileDispatch = AvailableFeatures.tileShadingPerTileDispatch;
        RequestedFeatures.tileShadingColorAttachments = AvailableFeatures.tileShadingColorAttachments;
        RequestedFeatures.tileShadingAtomicOps = AvailableFeatures.tileShadingAtomicOps;
        RequestedFeatures.tileShadingFragmentStage = AvailableFeatures.tileShadingFragmentStage;
        RequestedFeatures.tileShadingPerTileDraw = AvailableFeatures.tileShadingPerTileDraw;
        RequestedFeatures.tileShadingApron = AvailableFeatures.tileShadingApron;
        RequestedFeatures.tileShadingDepthAttachments = AvailableFeatures.tileShadingDepthAttachments;
        RequestedFeatures.tileShadingStencilAttachments = AvailableFeatures.tileShadingStencilAttachments;
        RequestedFeatures.tileShadingInputAttachments = AvailableFeatures.tileShadingInputAttachments;
        RequestedFeatures.tileShadingSampledAttachments = AvailableFeatures.tileShadingSampledAttachments;
        RequestedFeatures.tileShadingAnisotropicApron = AvailableFeatures.tileShadingAnisotropicApron;
        RequestedFeatures.tileShadingDispatchTile = AvailableFeatures.tileShadingDispatchTile;
        RequestedFeatures.tileShadingImageProcessing = AvailableFeatures.tileShadingImageProcessing;
    }

    void Ext_VK_QCOM_tile_shading::LookupFunctionPointers( VkDevice vkDevice, PFN_vkGetDeviceProcAddr deviceProcAddr )
    {
        m_vkCmdDispatchTileQCOM = (PFN_vkCmdDispatchTileQCOM)deviceProcAddr( vkDevice, "vkCmdDispatchTileQCOM" );
        if (m_vkCmdDispatchTileQCOM == nullptr)
        {
            LOGE("Driver does not expose \"vkCmdDispatchTileQCOM\" function, trying \"vkCmdTileDispatchQCOM\"");
            m_vkCmdDispatchTileQCOM = (PFN_vkCmdDispatchTileQCOM)deviceProcAddr( vkDevice, "vkCmdTileDispatchQCOM" );
        }
        m_vkCmdBeginPerTileExecutionQCOM = (PFN_vkCmdBeginPerTileExecutionQCOM)deviceProcAddr( vkDevice, "vkCmdBeginPerTileExecutionQCOM" );
        m_vkCmdEndPerTileExecutionQCOM = (PFN_vkCmdEndPerTileExecutionQCOM)deviceProcAddr( vkDevice, "vkCmdEndPerTileExecutionQCOM" );

        LOGI( "Ext_VK_QCOM_tile_shading function pointers: %p %p %p", m_vkCmdDispatchTileQCOM, m_vkCmdBeginPerTileExecutionQCOM, m_vkCmdEndPerTileExecutionQCOM );
    }

    void Ext_VK_QCOM_tile_shading::PrintFeatures() const
    {
        LOGI( "VK_QCOM_tile_shading (VkPhysicalDeviceTileShadingFeaturesQCOM): " );


        LOGI( "    tileShading: %s", this->AvailableFeatures.tileShading ? "True" : "False" );
        LOGI( "    tileShadingFragmentStage: %s", this->AvailableFeatures.tileShadingFragmentStage ? "True" : "False" );
        LOGI( "    tileShadingColorAttachments: %s", this->AvailableFeatures.tileShadingColorAttachments ? "True" : "False" );
        LOGI( "    tileShadingDepthAttachments: %s", this->AvailableFeatures.tileShadingDepthAttachments ? "True" : "False" );
        LOGI( "    tileShadingStencilAttachments: %s", this->AvailableFeatures.tileShadingStencilAttachments ? "True" : "False" );
        LOGI( "    tileShadingInputAttachments: %s", this->AvailableFeatures.tileShadingInputAttachments ? "True" : "False" );
        LOGI( "    tileShadingSampledAttachments: %s", this->AvailableFeatures.tileShadingSampledAttachments ? "True" : "False" );
        LOGI( "    tileShadingPerTileDraw: %s", this->AvailableFeatures.tileShadingPerTileDraw ? "True" : "False" );
        LOGI( "    tileShadingPerTileDispatch: %s", this->AvailableFeatures.tileShadingPerTileDispatch ? "True" : "False" );
        LOGI( "    tileShadingApron: %s", this->AvailableFeatures.tileShadingApron ? "True" : "False" );
        LOGI( "    tileShadingAnisotropicApron: %s", this->AvailableFeatures.tileShadingAnisotropicApron ? "True" : "False" );
        LOGI( "    tileShadingAtomicOps: %s", this->AvailableFeatures.tileShadingAtomicOps ? "True" : "False" );
        LOGI( "    tileShadingDispatchTile: %s", this->AvailableFeatures.tileShadingDispatchTile ? "True" : "False" );
        LOGI( "    tileShadingImageProcessing: %s", this->AvailableFeatures.tileShadingImageProcessing ? "True" : "False" );
    }

    void Ext_VK_QCOM_tile_shading::PrintProperties() const
    {
        LOGI( "VK_QCOM_tile_shading (VkPhysicalDeviceTileShadingPropertiesQCOM): " );
        LOGI( "    maxApronSize: %d", this->Properties.maxApronSize );
        LOGI( "    preferNonCoherent: %s", this->Properties.preferNonCoherent ? "True" : "False" );
        LOGI( "    tileGranularity: [%d, %d]", this->Properties.tileGranularity.width, this->Properties.tileGranularity.height );
        LOGI( "    maxTileShadingRate: [%d, %d]", this->Properties.maxTileShadingRate.width, this->Properties.maxTileShadingRate.height );
    }

#endif // VK_QCOM_tile_shading

    void Vulkan_SubgroupPropertiesHook::PrintProperties() const
    {
        LOGI( "VkPhysicalDeviceSubgroupProperties: " );
        LOGI( "    subgroupSize: %d", Properties.subgroupSize );
        auto SupportedStages = GetSubgroupStagesString( Properties.supportedStages );
        LOGI( "        supportedStages: %s", SupportedStages.c_str() );

        std::string SupportedOperations;
        static const char* SubgroupOperations[] = { "Basic", "Vote", "Arithmetic", "Ballot", "Shuffle", "Shuffle Relative", "Clustered", "Quad" };
        for (int i = 0; i < sizeof( SubgroupOperations ) / sizeof( *SubgroupOperations ); ++i)
        {
            if ((Properties.supportedOperations & (1 << i)) != 0)
            {
                SupportedOperations.append( SubgroupOperations[i] );
                SupportedOperations.append( ", " );
            }
        }
        LOGI( "        supportedOperations: %s", SupportedOperations.c_str() );

        LOGI( "        quadOperationsInAllStages: %s", Properties.quadOperationsInAllStages ? "True" : "False" );
    }


    void Vulkan_StorageFeaturesHook::PrintFeatures() const
    {
        LOGI( "FeaturesStorage16Bit: " );
        LOGI( "    storageBuffer16BitAccess: %s", this->AvailableFeatures.storageBuffer16BitAccess ? "True" : "False" );
        LOGI( "    uniformAndStorageBuffer16BitAccess: %s", this->AvailableFeatures.uniformAndStorageBuffer16BitAccess ? "True" : "False" );
        LOGI( "    storagePushConstant16: %s", this->AvailableFeatures.storagePushConstant16 ? "True" : "False" );
        LOGI( "    storageInputOutput16: %s", this->AvailableFeatures.storageInputOutput16 ? "True" : "False" );
    }


}; //namespace
