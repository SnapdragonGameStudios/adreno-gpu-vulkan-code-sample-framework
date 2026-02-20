//============================================================================================================
//
//
//                  Copyright (c) 2022-2024, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once
#include "extensionHelpers.hpp"
#include <vulkan/vulkan_core.h>

class Vulkan;

///
/// Library of Vulkan extensions
/// 

namespace ExtensionLib
{
#if VK_KHR_buffer_device_address

    struct Ext_VK_KHR_buffer_device_address : public VulkanDeviceFeaturesExtensionHelper<VkPhysicalDeviceBufferDeviceAddressFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR>
    {
        static constexpr auto Name = VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME;
        Ext_VK_KHR_buffer_device_address(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired)
            : VulkanDeviceFeaturesExtensionHelper(Name, status)
        {}
        void PrintFeatures() const override
        {
            LOGI("FeaturesBufferDeviceAddress: ");
            LOGI("    bufferDeviceAddress: %s", this->AvailableFeatures.bufferDeviceAddress ? "True" : "False");
            LOGI("    bufferDeviceAddressCaptureReplay: %s", this->AvailableFeatures.bufferDeviceAddressCaptureReplay ? "True" : "False");
            LOGI("    bufferDeviceAddressMultiDevice: %s", this->AvailableFeatures.bufferDeviceAddressMultiDevice ? "True" : "False");
        }
        void PopulateRequestedFeatures() override
        {
            // Enable just the 'bufferDeviceAddress' feature
            RequestedFeatures.bufferDeviceAddress = AvailableFeatures.bufferDeviceAddress;
        }
    };

#endif // VK_KHR_buffer_device_address


#if VK_EXT_mesh_shader

    struct Ext_VK_KHR_mesh_shader : public VulkanFeaturesPropertiesAndFunctionPointerExtensionHelper<
        VkPhysicalDeviceMeshShaderFeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, 
        VkPhysicalDeviceMeshShaderPropertiesEXT,  VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT>
    {
        static constexpr auto Name = VK_EXT_MESH_SHADER_EXTENSION_NAME;
        explicit Ext_VK_KHR_mesh_shader( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : VulkanFeaturesPropertiesAndFunctionPointerExtensionHelper( Name, status ) 
        {}

        void PopulateRequestedFeatures() override
        {
            RequestedFeatures = AvailableFeatures;
            RequestedFeatures.multiviewMeshShader = VK_FALSE;                   //if we need multiview then device needs setting up for multiview also
            RequestedFeatures.primitiveFragmentShadingRateMeshShader = VK_FALSE;//if we need fragment shading rate then device needs setting up for it (and we need to revisit this flag!)
        }
        void LookupFunctionPointers( VkInstance ) override {}
        void LookupFunctionPointers( VkDevice, PFN_vkGetDeviceProcAddr ) override;
        void PrintFeatures() const override;
        void PrintProperties() const override;
        PFN_vkCmdDrawMeshTasksEXT                   m_vkCmdDrawMeshTasksEXT = nullptr;
        PFN_vkCmdDrawMeshTasksIndirectEXT           m_vkCmdDrawMeshTasksIndirectEXT = nullptr;
        PFN_vkCmdDrawMeshTasksIndirectCountEXT      m_vkCmdDrawMeshTasksIndirectCountEXT = nullptr;
    };

#endif // VK_EXT_mesh_shader

#if VK_KHR_swapchain

    struct Ext_VK_KHR_swapchain : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
        explicit Ext_VK_KHR_swapchain( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : VulkanFunctionPointerExtensionHelper( Name, status ) {}
        void LookupFunctionPointers( VkInstance ) override;
        void LookupFunctionPointers( VkDevice, PFN_vkGetDeviceProcAddr ) override;

        PFN_vkCreateSwapchainKHR                    m_vkCreateSwapchainKHR = nullptr;
        PFN_vkDestroySwapchainKHR                   m_vkDestroySwapchainKHR = nullptr;
        PFN_vkGetSwapchainImagesKHR                 m_vkGetSwapchainImagesKHR = nullptr;
        PFN_vkAcquireNextImageKHR                   m_vkAcquireNextImageKHR = nullptr;
        PFN_vkQueuePresentKHR                       m_vkQueuePresentKHR = nullptr;
        PFN_vkGetDeviceGroupPresentCapabilitiesKHR  m_vkGetDeviceGroupPresentCapabilitiesKHR = nullptr;
        PFN_vkGetDeviceGroupSurfacePresentModesKHR  m_vkGetDeviceGroupSurfacePresentModesKHR = nullptr;
        PFN_vkGetPhysicalDevicePresentRectanglesKHR m_vkGetPhysicalDevicePresentRectanglesKHR = nullptr;
        PFN_vkAcquireNextImage2KHR                  m_vkAcquireNextImage2KHR = nullptr;
    };

#endif // VK_KHR_swapchain


#if VK_KHR_shader_float16_int8

    struct Ext_VK_KHR_shader_float16_int8 : public VulkanDeviceFeaturesExtensionHelper<VkPhysicalDeviceShaderFloat16Int8FeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR>
    {
        static constexpr auto Name = VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME;
        explicit Ext_VK_KHR_shader_float16_int8( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired )
            : VulkanDeviceFeaturesExtensionHelper(Name, status)
        {}
        void PrintFeatures() const override;
    };
    
#endif // VK_KHR_shader_float16_int8

#if VK_EXT_shader_image_atomic_int64

    struct Ext_VK_EXT_shader_image_atomic_int64 : public VulkanDeviceFeaturesExtensionHelper<VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT>
    {
        static constexpr auto Name = VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME;
        explicit Ext_VK_EXT_shader_image_atomic_int64( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired )
            : VulkanDeviceFeaturesExtensionHelper( Name, status )
        {}
        void PrintFeatures() const override;
    };

#endif // VK_EXT_shader_image_atomic_int64

#if VK_EXT_index_type_uint8

    struct Ext_VK_EXT_index_type_uint8 : public VulkanDeviceFeaturesExtensionHelper<VkPhysicalDeviceIndexTypeUint8FeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT>
    {
        static constexpr auto Name = VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME;
        explicit Ext_VK_EXT_index_type_uint8( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired )
            : VulkanDeviceFeaturesExtensionHelper(Name, status)
        {}
        void PrintFeatures() const override;
    };

#endif // VK_EXT_index_type_uint8

#if VK_KHR_shader_subgroup_extended_types

    struct Ext_VK_KHR_shader_subgroup_extended_types : public VulkanDeviceFeaturesExtensionHelper<VkPhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR>
    {
        static constexpr auto Name = VK_KHR_SHADER_SUBGROUP_EXTENDED_TYPES_EXTENSION_NAME;
        explicit Ext_VK_KHR_shader_subgroup_extended_types( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired )
            : VulkanDeviceFeaturesExtensionHelper(Name, status)
        {}
        void PrintFeatures() const override;
    };

#endif // VK_KHR_shader_subgroup_extended_types

#if VK_EXT_descriptor_indexing

    struct Ext_VK_EXT_descriptor_indexing : public VulkanDeviceFeaturePropertiesExtensionHelper<
        VkPhysicalDeviceDescriptorIndexingFeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT,
        VkPhysicalDeviceDescriptorIndexingPropertiesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES_EXT>
    {
        static constexpr auto Name = VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME;
        explicit Ext_VK_EXT_descriptor_indexing( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : VulkanDeviceFeaturePropertiesExtensionHelper(Name, status)
        {}
        void PrintFeatures() const override;
        void PrintProperties() const override;
    };

#endif // VK_EXT_descriptor_indexing

#if VK_KHR_8bit_storage

    struct Ext_VK_KHR_8bit_storage : public VulkanDeviceFeaturesExtensionHelper<
        VkPhysicalDevice8BitStorageFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR>
    {
        static constexpr auto Name = VK_KHR_8BIT_STORAGE_EXTENSION_NAME;
        explicit Ext_VK_KHR_8bit_storage( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : VulkanDeviceFeaturesExtensionHelper(Name, status)
        {}
        void PrintFeatures() const override;
    };

#endif // VK_KHR_8bit_storage

#if VK_KHR_portability_subset

    struct Ext_VK_KHR_portability_subset : public VulkanDeviceFeaturePropertiesExtensionHelper<
        VkPhysicalDevicePortabilitySubsetFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR,
        VkPhysicalDevicePortabilitySubsetPropertiesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_PROPERTIES_KHR>
    {
        static constexpr auto Name = VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME;
        explicit Ext_VK_KHR_portability_subset( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : VulkanDeviceFeaturePropertiesExtensionHelper(Name, status)
        {}
        void PrintFeatures() const override;
        void PrintProperties() const override;
    };

#else

    // 'dummy' implementation of Ext_VK_KHR_portability_subset for when vulkan headers do not contain 'VK_KHR_portability_subset'
    struct Ext_VK_KHR_portability_subset : public VulkanExtension<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = "VK_KHR_portability_subset";
        Ext_VK_KHR_portability_subset( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : VulkanExtension( Name, status ) {}
    };

#endif // VK_KHR_portability_subset

#if VK_KHR_fragment_shading_rate

    struct Ext_VK_KHR_fragment_shading_rate : public VulkanFeaturesPropertiesAndFunctionPointerExtensionHelper<
        VkPhysicalDeviceFragmentShadingRateFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
        VkPhysicalDeviceFragmentShadingRatePropertiesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR>
    {
        static constexpr auto Name = VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME;
        explicit Ext_VK_KHR_fragment_shading_rate( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : VulkanFeaturesPropertiesAndFunctionPointerExtensionHelper( Name, status )
        {}

        void PopulateRequestedFeatures() override
        {
            RequestedFeatures.sType = AvailableFeatures.sType;
            RequestedFeatures.attachmentFragmentShadingRate = AvailableFeatures.attachmentFragmentShadingRate;
        }
        void LookupFunctionPointers( VkInstance vkInstance ) override
        {
            m_vkCmdSetFragmentShadingRateKHR = (PFN_vkCmdSetFragmentShadingRateKHR) vkGetInstanceProcAddr( vkInstance, "vkCmdSetFragmentShadingRateKHR" );
        }
        void LookupFunctionPointers( VkDevice, PFN_vkGetDeviceProcAddr ) override {}
        void PrintFeatures() const override;
        void PrintProperties() const override;
        PFN_vkCmdSetFragmentShadingRateKHR m_vkCmdSetFragmentShadingRateKHR = nullptr;
    };

#endif // VK_KHR_fragment_shading_rate

#if VK_KHR_create_renderpass2

    struct Ext_VK_KHR_create_renderpass2 : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME;
        explicit Ext_VK_KHR_create_renderpass2( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : VulkanFunctionPointerExtensionHelper( Name, status ) {}
        void LookupFunctionPointers(VkInstance vkInstance) override;
        void LookupFunctionPointers( VkDevice, PFN_vkGetDeviceProcAddr ) override {/*no device functions*/ }
        PFN_vkCreateRenderPass2KHR      m_vkCreateRenderPass2KHR = nullptr;
        PFN_vkCmdBeginRenderPass2KHR    m_vkCmdBeginRenderPass2KHR = nullptr;
        PFN_vkCmdNextSubpass2KHR        m_vkCmdNextSubpass2KHR = nullptr;
        PFN_vkCmdEndRenderPass2KHR      m_vkCmdEndRenderPass2KHR = nullptr;
    };

#endif // VK_KHR_create_renderpass2


#if VK_KHR_draw_indirect_count

    struct Ext_VK_KHR_draw_indirect_count : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME;
        explicit Ext_VK_KHR_draw_indirect_count( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : VulkanFunctionPointerExtensionHelper( Name, status ) {}
        void LookupFunctionPointers( VkInstance vkInstance ) override;
        void LookupFunctionPointers( VkDevice vkDevice, PFN_vkGetDeviceProcAddr ) override {/*no device functions*/}
        PFN_vkCmdDrawIndirectCountKHR        m_vkCmdDrawIndirectCountKHR = nullptr;
        PFN_vkCmdDrawIndexedIndirectCountKHR m_vkCmdDrawIndexedIndirectCountKHR = nullptr;
    };

#endif // VK_KHR_draw_indirect_count

#if VK_KHR_cooperative_matrix

    struct Ext_VK_KHR_cooperative_matrix : public VulkanFeaturesPropertiesAndFunctionPointerExtensionHelper<
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        VkPhysicalDeviceCooperativeMatrixPropertiesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR>
    {
        static constexpr auto Name = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
        explicit Ext_VK_KHR_cooperative_matrix(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired) : VulkanFeaturesPropertiesAndFunctionPointerExtensionHelper(Name, status)
        {
        }

        void PopulateRequestedFeatures() override
        {
            RequestedFeatures.sType = AvailableFeatures.sType;
            RequestedFeatures.cooperativeMatrix = AvailableFeatures.cooperativeMatrix;
            RequestedFeatures.cooperativeMatrixRobustBufferAccess = AvailableFeatures.cooperativeMatrixRobustBufferAccess;
        }
        void LookupFunctionPointers(VkInstance vkInstance) override
        {
            m_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(vkInstance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        }
        void LookupFunctionPointers(VkDevice, PFN_vkGetDeviceProcAddr) override {}
        void PrintFeatures() const override;
        void PrintProperties() const override;

        PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR m_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = nullptr;
    };

#endif // VK_KHR_cooperative_matrix

#if VK_KHR_depth_stencil_resolve

    struct Ext_VK_KHR_depth_stencil_resolve : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME;
        explicit Ext_VK_KHR_depth_stencil_resolve( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : VulkanFunctionPointerExtensionHelper( Name, status ) {}
        void LookupFunctionPointers( VkInstance vkInstance ) override
        {/*no instance functions*/}
        void LookupFunctionPointers( VkDevice vkDevice, PFN_vkGetDeviceProcAddr ) override {/*no device functions*/}
    };

#endif // VK_KHR_depth_stencil_resolve

#if VK_KHR_dynamic_rendering

    struct Ext_VK_KHR_dynamic_rendering : public VulkanFeaturesAndFunctionPointerExtensionHelper<
        VkPhysicalDeviceDynamicRenderingFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES >
    {
        using tBase = VulkanFeaturesAndFunctionPointerExtensionHelper;
        static constexpr auto Name = VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME;
        explicit Ext_VK_KHR_dynamic_rendering( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : tBase( Name, status ) {}
        void PopulateRequestedFeatures() override
        {
            RequestedFeatures.sType = AvailableFeatures.sType;
            RequestedFeatures.dynamicRendering = AvailableFeatures.dynamicRendering;
        }
        void LookupFunctionPointers( VkInstance vkInstance ) override
        {
            m_vkCmdBeginRenderingKHR = (PFN_vkCmdBeginRenderingKHR) vkGetInstanceProcAddr( vkInstance, "vkCmdBeginRenderingKHR" );
            m_vkCmdEndRenderingKHR = (PFN_vkCmdEndRenderingKHR) vkGetInstanceProcAddr( vkInstance, "vkCmdEndRenderingKHR" );
        }
        void LookupFunctionPointers( VkDevice vkDevice, PFN_vkGetDeviceProcAddr ) override {/*no device functions*/}
        void PrintFeatures() const override;
        PFN_vkCmdBeginRenderingKHR m_vkCmdBeginRenderingKHR = nullptr;
        PFN_vkCmdEndRenderingKHR   m_vkCmdEndRenderingKHR = nullptr;
    };

#endif // VK_KHR_dynamic_rendering

#if VK_KHR_dynamic_rendering_local_read

    struct Ext_VK_KHR_dynamic_rendering_local_read : public VulkanFeaturesAndFunctionPointerExtensionHelper<
        VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR >
    {
        using tBase = VulkanFeaturesAndFunctionPointerExtensionHelper;
        static constexpr auto Name = VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME;
        explicit Ext_VK_KHR_dynamic_rendering_local_read( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : tBase( Name, status ) {}
        void PopulateRequestedFeatures() override
        {
            RequestedFeatures.sType = AvailableFeatures.sType;
            RequestedFeatures.dynamicRenderingLocalRead = AvailableFeatures.dynamicRenderingLocalRead;
        }
        void LookupFunctionPointers( VkInstance vkInstance ) override
        {
            m_vkCmdSetRenderingAttachmentLocationsKHR = (PFN_vkCmdSetRenderingAttachmentLocationsKHR) vkGetInstanceProcAddr( vkInstance, "vkCmdSetRenderingAttachmentLocationsKHR" );
            m_vkCmdSetRenderingInputAttachmentIndicesKHR = (PFN_vkCmdSetRenderingInputAttachmentIndicesKHR) vkGetInstanceProcAddr( vkInstance, "vkCmdSetRenderingInputAttachmentIndicesKHR" );
        }
        void LookupFunctionPointers( VkDevice vkDevice, PFN_vkGetDeviceProcAddr ) override {/*no device functions*/}
        void PrintFeatures() const override;
        PFN_vkCmdSetRenderingAttachmentLocationsKHR m_vkCmdSetRenderingAttachmentLocationsKHR = nullptr;
        PFN_vkCmdSetRenderingInputAttachmentIndicesKHR m_vkCmdSetRenderingInputAttachmentIndicesKHR = nullptr;
    };

#endif // VK_KHR_dynamic_rendering_local_read

#if VK_EXT_hdr_metadata

    struct Ext_VK_EXT_hdr_metadata : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = VK_EXT_HDR_METADATA_EXTENSION_NAME;
        explicit Ext_VK_EXT_hdr_metadata(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired) : VulkanFunctionPointerExtensionHelper(Name, status) {}
        void LookupFunctionPointers( VkInstance vkInstance ) override {/*no instance functions*/}
        void LookupFunctionPointers( VkDevice vkDevice, PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr ) override
        {
            m_vkSetHdrMetadataEXT = (PFN_vkSetHdrMetadataEXT) fpGetDeviceProcAddr( vkDevice, "vkSetHdrMetadataEXT" );
        }
        PFN_vkSetHdrMetadataEXT             m_vkSetHdrMetadataEXT = nullptr;
    };

#endif // VK_EXT_hdr_metadata

#if VK_EXT_debug_utils

    struct Ext_VK_EXT_debug_utils : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        explicit Ext_VK_EXT_debug_utils( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : VulkanFunctionPointerExtensionHelper( Name, status ) {}
        void LookupFunctionPointers( VkInstance vkInstance ) override
        {
            m_vkSetDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT) vkGetInstanceProcAddr( vkInstance, "vkSetDebugUtilsObjectNameEXT" );
        }
        void LookupFunctionPointers( VkDevice, PFN_vkGetDeviceProcAddr ) override {/*no device functions*/ }
        bool SetDebugUtilsObjectName( VkDevice vkDevice, uint64_t object, VkObjectType objectType, const char* name ) const;

        PFN_vkSetDebugUtilsObjectNameEXT  m_vkSetDebugUtilsObjectNameEXT = nullptr;
    };

#endif // VK_EXT_debug_utils

#if VK_EXT_debug_marker

    struct Ext_VK_EXT_debug_marker : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = VK_EXT_DEBUG_MARKER_EXTENSION_NAME;
        explicit Ext_VK_EXT_debug_marker( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : VulkanFunctionPointerExtensionHelper( Name, status ) {}
        void LookupFunctionPointers( VkInstance ) override {/*no instance functions*/ }
        void LookupFunctionPointers( VkDevice vkDevice, PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr ) override
        {
            m_vkDebugMarkerSetObjectNameEXT = (PFN_vkDebugMarkerSetObjectNameEXT) fpGetDeviceProcAddr( vkDevice, "vkDebugMarkerSetObjectNameEXT" );
        }
        bool DebugMarkerSetObjectName( VkDevice vkDevice, uint64_t object, VkDebugReportObjectTypeEXT objectType, const char* name ) const;

        PFN_vkDebugMarkerSetObjectNameEXT m_vkDebugMarkerSetObjectNameEXT = nullptr;
    };

#endif // Ext_VK_EXT_debug_marker

#if VK_EXT_subgroup_size_control

    struct Ext_VK_EXT_subgroup_size_control : public VulkanDeviceFeaturePropertiesExtensionHelper<
        VkPhysicalDeviceSubgroupSizeControlFeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT,
        VkPhysicalDeviceSubgroupSizeControlPropertiesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT >
    {
        using tBase = VulkanDeviceFeaturePropertiesExtensionHelper;
        static constexpr auto Name = VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME;
        explicit Ext_VK_EXT_subgroup_size_control( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : VulkanDeviceFeaturePropertiesExtensionHelper( Name, status ), m_ShaderCreateHook(this)
        {}
        void PrintFeatures() const override;
        void PrintProperties() const override;

        VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT SubGroupSizeControl { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT };

    private:
        class ShaderCreateStageHook final : public ExtensionHook<VkPipelineShaderStageCreateInfo>
        {
            friend struct Ext_VK_EXT_subgroup_size_control;
            explicit ShaderCreateStageHook( Ext_VK_EXT_subgroup_size_control* _Parent ) : ExtensionHook<VkPipelineShaderStageCreateInfo>(), Parent( _Parent ) {}
            VkStructureType StructureType() const override { return (VkStructureType) 0; };
            VkBaseOutStructure* Obtain( tBase* pBase ) override {
                if ((pBase->stage & Parent->Properties.requiredSubgroupSizeStages)!=0
                    && Parent->RequestedFeatures.subgroupSizeControl
                    && Parent->SubGroupSizeControl.requiredSubgroupSize != 0)
                {
                    Parent->SubGroupSizeControl.pNext = nullptr;
                    pBase->flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
                    return (VkBaseOutStructure*) &Parent->SubGroupSizeControl;
                }
                return nullptr;
            }
            void Release( VkBaseOutStructure* pBase ) override {
                assert( pBase == (VkBaseOutStructure *) &Parent->SubGroupSizeControl );
            }
            Ext_VK_EXT_subgroup_size_control* Parent;
        } m_ShaderCreateHook;

        void Register( Vulkan& vulkan ) override
        {
            tBase::Register( vulkan );
            vulkan.AddExtensionHooks( &m_ShaderCreateHook );
        }
    };

#endif // VK_EXT_subgroup_size_control

#if VK_EXT_host_query_reset

    struct Ext_VK_EXT_host_query_reset : public VulkanFeaturesAndFunctionPointerExtensionHelper<
        VkPhysicalDeviceHostQueryResetFeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES_EXT >
    {
        using tBase = VulkanFeaturesAndFunctionPointerExtensionHelper;
        static constexpr auto Name = VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME;
        explicit Ext_VK_EXT_host_query_reset( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : tBase( Name, status )
        {}
        void PrintFeatures() const override;
        void LookupFunctionPointers( VkInstance vkInstance ) override {/*no instance functions*/ }
        void LookupFunctionPointers( VkDevice vkDevice, PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr ) override
        {
            if (RequestedFeatures.hostQueryReset == VK_TRUE)
                m_vkResetQueryPoolEXT = (PFN_vkResetQueryPoolEXT) fpGetDeviceProcAddr( vkDevice, "vkResetQueryPoolEXT" );
        }
        PFN_vkResetQueryPoolEXT m_vkResetQueryPoolEXT = nullptr;
    };

#endif // VK_EXT_host_query_reset

#if VK_KHR_timeline_semaphore

    struct Ext_VK_KHR_timeline_semaphore : public VulkanDeviceFeaturePropertiesExtensionHelper<
        VkPhysicalDeviceTimelineSemaphoreFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR,
        VkPhysicalDeviceTimelineSemaphorePropertiesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES_KHR >
    {
        using tBase = VulkanDeviceFeaturePropertiesExtensionHelper;
        static constexpr auto Name = VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME;
        explicit Ext_VK_KHR_timeline_semaphore( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : tBase( Name, status )
        {}
        void PrintFeatures() const override;
        void PrintProperties() const override;
    };

#endif // VK_KHR_timeline_semaphore

#if VK_KHR_synchronization2

    struct Ext_VK_KHR_synchronization2 : public VulkanFeaturesAndFunctionPointerExtensionHelper<
        VkPhysicalDeviceSynchronization2FeaturesKHR, (VkStructureType) VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR >
    {
        using tBase = VulkanFeaturesAndFunctionPointerExtensionHelper;
        static constexpr auto Name = VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME;
        explicit Ext_VK_KHR_synchronization2(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired) : tBase(Name, status)
        {}
        void PrintFeatures() const override;
        void LookupFunctionPointers(VkInstance vkInstance) override;
        void LookupFunctionPointers(VkDevice, PFN_vkGetDeviceProcAddr) override {/*no device functions*/ }
        PFN_vkQueueSubmit2KHR           m_vkQueueSubmit2KHR = nullptr;
    };

#endif // VK_KHR_synchronization2

#if VK_ARM_tensors

    struct Ext_VK_ARM_tensors : public VulkanFeaturesPropertiesAndFunctionPointerExtensionHelper<
        VkPhysicalDeviceTensorFeaturesARM, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM,
        VkPhysicalDeviceTensorPropertiesARM, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_PROPERTIES_ARM>
    {
        static constexpr auto Name = VK_ARM_TENSORS_EXTENSION_NAME;

        explicit Ext_VK_ARM_tensors(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired)
            : VulkanFeaturesPropertiesAndFunctionPointerExtensionHelper(Name, status)
        {
        }

        void PopulateRequestedFeatures() override
        {
            RequestedFeatures.sType = AvailableFeatures.sType;
            RequestedFeatures.tensorNonPacked = AvailableFeatures.tensorNonPacked;
            RequestedFeatures.shaderTensorAccess = AvailableFeatures.shaderTensorAccess;
            RequestedFeatures.shaderStorageTensorArrayDynamicIndexing = AvailableFeatures.shaderStorageTensorArrayDynamicIndexing;
            RequestedFeatures.shaderStorageTensorArrayNonUniformIndexing = AvailableFeatures.shaderStorageTensorArrayNonUniformIndexing;
            RequestedFeatures.descriptorBindingStorageTensorUpdateAfterBind = AvailableFeatures.descriptorBindingStorageTensorUpdateAfterBind;
            RequestedFeatures.tensors = AvailableFeatures.tensors;
        }

        void PrintFeatures() const override;
        void PrintProperties() const override;
        void LookupFunctionPointers(VkInstance vkInstance) override {}
        void LookupFunctionPointers(VkDevice, PFN_vkGetDeviceProcAddr) override {}
    };

#endif // VK_ARM_tensors

#if VK_ARM_data_graph

    struct Ext_VK_ARM_data_graph : public VulkanFeaturesAndFunctionPointerExtensionHelper<
        VkPhysicalDeviceDataGraphFeaturesARM, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM>
    {
        using tBase = VulkanFeaturesAndFunctionPointerExtensionHelper;
        static constexpr auto Name = VK_ARM_DATA_GRAPH_EXTENSION_NAME;

        explicit Ext_VK_ARM_data_graph(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired)
            : tBase(Name, status)
        {
        }

        void PopulateRequestedFeatures() override
        {
            RequestedFeatures.sType = AvailableFeatures.sType;
            RequestedFeatures.dataGraph = AvailableFeatures.dataGraph;
            RequestedFeatures.dataGraphUpdateAfterBind = AvailableFeatures.dataGraphUpdateAfterBind;
            RequestedFeatures.dataGraphSpecializationConstants = AvailableFeatures.dataGraphSpecializationConstants;
            RequestedFeatures.dataGraphDescriptorBuffer = AvailableFeatures.dataGraphDescriptorBuffer;
            RequestedFeatures.dataGraphShaderModule = AvailableFeatures.dataGraphShaderModule;
        }

        void PrintFeatures() const override;
        void LookupFunctionPointers(VkInstance vkInstance) override {}
        void LookupFunctionPointers(VkDevice, PFN_vkGetDeviceProcAddr) override {}
    };

#endif // VK_ARM_data_graph

#if VK_QCOM_data_graph_model

    struct Ext_VK_QCOM_data_graph_model : public VulkanFeaturesAndFunctionPointerExtensionHelper<
        VkPhysicalDeviceDataGraphModelFeaturesQCOM, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_MODEL_FEATURES_QCOM>
    {
        using tBase = VulkanFeaturesAndFunctionPointerExtensionHelper;
        static constexpr auto Name = VK_QCOM_DATA_GRAPH_MODEL_EXTENSION_NAME;

        explicit Ext_VK_QCOM_data_graph_model(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired)
            : tBase(Name, status)
        {
        }

        void PopulateRequestedFeatures() override
        {
            RequestedFeatures.sType = AvailableFeatures.sType;
            RequestedFeatures.dataGraphModel = AvailableFeatures.dataGraphModel;
        }

        void PrintFeatures() const override;
        void LookupFunctionPointers(VkInstance vkInstance) override {}
        void LookupFunctionPointers(VkDevice, PFN_vkGetDeviceProcAddr) override {}
    };

#endif // VK_QCOM_data_graph_model

#if VK_QCOM_tile_properties

    struct Ext_VK_QCOM_tile_properties : public VulkanFeaturesAndFunctionPointerExtensionHelper<
        VkPhysicalDeviceTilePropertiesFeaturesQCOM, (VkStructureType)VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_PROPERTIES_FEATURES_QCOM>
    {
            using tBase = VulkanFeaturesAndFunctionPointerExtensionHelper;
            static constexpr auto Name = VK_QCOM_TILE_PROPERTIES_EXTENSION_NAME;
            explicit Ext_VK_QCOM_tile_properties( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : tBase( Name, status )
            {}
            void LookupFunctionPointers( VkInstance vkInstance ) override  {/*no instance functions*/ };
            void LookupFunctionPointers( VkDevice, PFN_vkGetDeviceProcAddr ) override;
            void PrintFeatures() const override;
            PFN_vkGetDynamicRenderingTilePropertiesQCOM     m_vkGetDynamicRenderingTilePropertiesQCOM = nullptr;
            PFN_vkGetFramebufferTilePropertiesQCOM          m_vkGetFramebufferTilePropertiesQCOM = nullptr;
    };

#endif // VK_QCOM_tile_properties

#if VK_QCOM_tile_shading

    struct Ext_VK_QCOM_tile_shading : public VulkanFeaturesPropertiesAndFunctionPointerExtensionHelper<
        VkPhysicalDeviceTileShadingFeaturesQCOM, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_SHADING_FEATURES_QCOM,
        VkPhysicalDeviceTileShadingPropertiesQCOM, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_SHADING_PROPERTIES_QCOM>
    {
        using tBase = VulkanFeaturesPropertiesAndFunctionPointerExtensionHelper;
        static constexpr auto Name = VK_QCOM_TILE_SHADING_EXTENSION_NAME;
        explicit Ext_VK_QCOM_tile_shading( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : tBase( Name, status )
        {}
        void PopulateRequestedFeatures() override;
        void LookupFunctionPointers( VkInstance vkInstance ) override {/*no instance functions*/ }
        void LookupFunctionPointers( VkDevice, PFN_vkGetDeviceProcAddr ) override;
        void PrintFeatures() const override;
        void PrintProperties() const override;

        PFN_vkCmdDispatchTileQCOM          m_vkCmdDispatchTileQCOM = nullptr;
        PFN_vkCmdBeginPerTileExecutionQCOM m_vkCmdBeginPerTileExecutionQCOM = nullptr;
        PFN_vkCmdEndPerTileExecutionQCOM   m_vkCmdEndPerTileExecutionQCOM = nullptr;
    };
    inline namespace fvk {
        using VkPerTileBeginInfoQCOM = ::fvk::VkStructWrapper<VkPerTileBeginInfoQCOM, VK_STRUCTURE_TYPE_PER_TILE_BEGIN_INFO_QCOM>;
        using VkPerTileEndInfoQCOM = ::fvk::VkStructWrapper<VkPerTileEndInfoQCOM, VK_STRUCTURE_TYPE_PER_TILE_END_INFO_QCOM>;
        using VkDispatchTileInfoQCOM = ::fvk::VkStructWrapper<VkDispatchTileInfoQCOM, VK_STRUCTURE_TYPE_DISPATCH_TILE_INFO_QCOM>;
    };

#endif // VK_QCOM_tile_shading

#if VK_QCOM_tile_memory_heap

    struct Ext_VK_QCOM_tile_memory_heap : public VulkanFeaturesPropertiesAndFunctionPointerExtensionHelper<
        VkPhysicalDeviceTileMemoryHeapFeaturesQCOM, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_MEMORY_HEAP_FEATURES_QCOM,
        VkPhysicalDeviceTileMemoryHeapPropertiesQCOM, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TILE_MEMORY_HEAP_PROPERTIES_QCOM>
    {
        static constexpr auto Name = VK_QCOM_TILE_MEMORY_HEAP_EXTENSION_NAME;
        explicit Ext_VK_QCOM_tile_memory_heap(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired) : VulkanFeaturesPropertiesAndFunctionPointerExtensionHelper(Name, status)
        {}

        void PopulateRequestedFeatures() override
        {
            RequestedFeatures.sType = AvailableFeatures.sType;
            RequestedFeatures.tileMemoryHeap = true; /*AvailableFeatures.tileMemoryHeap;*/
        }
        void LookupFunctionPointers( VkInstance ) override {/*no instance functions*/ }
        void LookupFunctionPointers( VkDevice vkDevice, PFN_vkGetDeviceProcAddr ) override
        {
            m_vkCmdBindTileMemoryQCOM = (PFN_vkCmdBindTileMemoryQCOM)vkGetDeviceProcAddr(vkDevice, "vkCmdBindTileMemoryQCOM");
        }
        void PrintFeatures() const override;
        void PrintProperties() const override;
        PFN_vkCmdBindTileMemoryQCOM m_vkCmdBindTileMemoryQCOM = nullptr;
    };

#endif // VK_QCOM_tile_memory_heap

#if VK_KHR_get_memory_requirements2 

    struct Ext_VK_KHR_get_memory_requirements2  : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME;
        explicit Ext_VK_KHR_get_memory_requirements2(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired) : VulkanFunctionPointerExtensionHelper(Name, status) {}
        void LookupFunctionPointers( VkInstance ) override {/*no instance functions*/ }
        void LookupFunctionPointers( VkDevice vkDevice, PFN_vkGetDeviceProcAddr ) override
        {
            m_vkGetBufferMemoryRequirements2KHR = (PFN_vkGetBufferMemoryRequirements2KHR)vkGetDeviceProcAddr(vkDevice, "vkGetBufferMemoryRequirements2KHR");
            m_vkGetImageMemoryRequirements2KHR = (PFN_vkGetImageMemoryRequirements2KHR)vkGetDeviceProcAddr(vkDevice, "vkGetImageMemoryRequirements2KHR");
            m_vkGetImageSparseMemoryRequirements2KHR = (PFN_vkGetImageSparseMemoryRequirements2KHR)vkGetDeviceProcAddr(vkDevice, "vkGetImageSparseMemoryRequirements2KHR");
        }
        PFN_vkGetBufferMemoryRequirements2KHR m_vkGetBufferMemoryRequirements2KHR = nullptr;
        PFN_vkGetImageMemoryRequirements2KHR m_vkGetImageMemoryRequirements2KHR = nullptr;
        PFN_vkGetImageSparseMemoryRequirements2KHR m_vkGetImageSparseMemoryRequirements2KHR = nullptr;   
    };

#endif // VK_KHR_get_memory_requirements2 

#if VK_KHR_ray_tracing_position_fetch

    struct Ext_VK_KHR_ray_tracing_position_fetch : public VulkanDeviceFeaturesExtensionHelper<
        VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR>
    {
        static constexpr auto Name = VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME;
        explicit Ext_VK_KHR_ray_tracing_position_fetch( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired )
            : VulkanDeviceFeaturesExtensionHelper( Name, status )
        {}
        void PrintFeatures() const override;
    };

#endif // VK_KHR_ray_tracing_position_fetch

#if VK_EXT_scalar_block_layout

    struct Ext_VK_EXT_scalar_block_layout : public VulkanDeviceFeaturesExtensionHelper<
        VkPhysicalDeviceScalarBlockLayoutFeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES_EXT>
    {
        static constexpr auto Name = VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME;
        explicit Ext_VK_EXT_scalar_block_layout( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired )
            : VulkanDeviceFeaturesExtensionHelper( Name, status )
        {}
        void PrintFeatures() const override;
    };

#endif // VK_EXT_scalar_block_layout

#if VK_KHR_get_physical_device_properties2

    // Instance extension
    struct Ext_VK_KHR_get_physical_device_properties2 : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eInstance>
    {
        using tBase = VulkanFunctionPointerExtensionHelper;
        static constexpr auto Name = VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME;
        explicit Ext_VK_KHR_get_physical_device_properties2(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired) : tBase(Name, status)
        {}
        void LookupFunctionPointers(VkInstance vkInstance) override;
        void LookupFunctionPointers( VkDevice, PFN_vkGetDeviceProcAddr ) override {/*no device functions*/ }
        PFN_vkGetPhysicalDeviceFeatures2KHR             m_vkGetPhysicalDeviceFeatures2KHR = nullptr;
        PFN_vkGetPhysicalDeviceFormatProperties2KHR     m_vkGetPhysicalDeviceFormatProperties2KHR = nullptr;
        PFN_vkGetPhysicalDeviceImageFormatProperties2KHR m_vkGetPhysicalDeviceImageFormatProperties2KHR = nullptr;
        PFN_vkGetPhysicalDeviceMemoryProperties2KHR     m_vkGetPhysicalDeviceMemoryProperties2KHR = nullptr;
        PFN_vkGetPhysicalDeviceProperties2KHR           m_vkGetPhysicalDeviceProperties2KHR = nullptr;
    };

#endif // VK_KHR_get_physical_device_properties2

#if VK_KHR_external_memory_capabilities

    // Instance extension: exposes capability queries (buffer/image external memory).
    struct Ext_VK_KHR_external_memory_capabilities : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eInstance>
    {
        static constexpr auto Name = VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME;
        explicit Ext_VK_KHR_external_memory_capabilities(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired)
            : VulkanFunctionPointerExtensionHelper(Name, status)
        {
        }

        // Volk will resolve globally; we keep the overrides for consistency.
        void LookupFunctionPointers(VkInstance) override {}
        void LookupFunctionPointers(VkDevice, PFN_vkGetDeviceProcAddr) override {}
    };

#endif // VK_KHR_external_memory_capabilities

#if VK_KHR_external_memory

    // Device extension: enables external memory types/flags. No commands to load.
    struct Ext_VK_KHR_external_memory : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME;
        explicit Ext_VK_KHR_external_memory(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired)
            : VulkanFunctionPointerExtensionHelper(Name, status)
        {
        }

        void LookupFunctionPointers(VkInstance) override {}
        void LookupFunctionPointers(VkDevice, PFN_vkGetDeviceProcAddr) override {}
    };

#endif // VK_KHR_external_memory

#if VK_KHR_external_memory_fd

    // Device extension: POSIX file descriptor handles for external memory.
    struct Ext_VK_KHR_external_memory_fd : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
        explicit Ext_VK_KHR_external_memory_fd(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired)
            : VulkanFunctionPointerExtensionHelper(Name, status)
        {
        }

        void LookupFunctionPointers(VkInstance) override {}
        void LookupFunctionPointers(VkDevice, PFN_vkGetDeviceProcAddr) override {}
    };

#endif // VK_KHR_external_memory_fd

#if defined(VK_USE_PLATFORM_WIN32_KHR) && VK_KHR_external_memory_win32

    // Device extension: Win32 HANDLEs for external memory.
    struct Ext_VK_KHR_external_memory_win32 : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME;
        explicit Ext_VK_KHR_external_memory_win32(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired)
            : VulkanFunctionPointerExtensionHelper(Name, status)
        {
        }

        void LookupFunctionPointers(VkInstance) override {}
        void LookupFunctionPointers(VkDevice, PFN_vkGetDeviceProcAddr) override {}
    };

#endif // VK_USE_PLATFORM_WIN32_KHR && VK_KHR_external_memory_win32


#if VK_KHR_external_semaphore_capabilities

    // Instance extension: exposes semaphore external capability queries.
    struct Ext_VK_KHR_external_semaphore_capabilities : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eInstance>
    {
        static constexpr auto Name = VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME;
        explicit Ext_VK_KHR_external_semaphore_capabilities(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired)
            : VulkanFunctionPointerExtensionHelper(Name, status)
        {
        }

        void LookupFunctionPointers(VkInstance) override {}
        void LookupFunctionPointers(VkDevice, PFN_vkGetDeviceProcAddr) override {}
    };

#endif // VK_KHR_external_semaphore_capabilities

#if VK_KHR_external_semaphore

    // Device extension: enables external semaphore types/flags. No commands to load.
    struct Ext_VK_KHR_external_semaphore : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME;
        explicit Ext_VK_KHR_external_semaphore(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired)
            : VulkanFunctionPointerExtensionHelper(Name, status)
        {
        }

        void LookupFunctionPointers(VkInstance) override {}
        void LookupFunctionPointers(VkDevice, PFN_vkGetDeviceProcAddr) override {}
    };

#endif // VK_KHR_external_semaphore


#if VK_KHR_external_semaphore_fd

    // Device extension: POSIX file descriptor handles for external semaphores.
    struct Ext_VK_KHR_external_semaphore_fd : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
        explicit Ext_VK_KHR_external_semaphore_fd(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired)
            : VulkanFunctionPointerExtensionHelper(Name, status)
        {
        }

        void LookupFunctionPointers(VkInstance) override {}
        void LookupFunctionPointers(VkDevice, PFN_vkGetDeviceProcAddr) override {}
    };

#endif // VK_KHR_external_semaphore_fd


#if defined(VK_USE_PLATFORM_WIN32_KHR) && VK_KHR_external_semaphore_win32

    // Device extension: Win32 HANDLEs for external semaphores.
    struct Ext_VK_KHR_external_semaphore_win32 : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME;
        explicit Ext_VK_KHR_external_semaphore_win32(VulkanExtensionStatus status = VulkanExtensionStatus::eRequired)
            : VulkanFunctionPointerExtensionHelper(Name, status)
        {
        }

        void LookupFunctionPointers(VkInstance) override {}
        void LookupFunctionPointers(VkDevice, PFN_vkGetDeviceProcAddr) override {}
    };

#endif // VK_USE_PLATFORM_WIN32_KHR && VK_KHR_external_semaphore_win32

#if VK_KHR_surface

    // Instance extension
    struct Ext_VK_KHR_surface : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eInstance>
    {
        using tBase = VulkanFunctionPointerExtensionHelper;
        static constexpr auto Name = VK_KHR_SURFACE_EXTENSION_NAME;
        explicit Ext_VK_KHR_surface( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : tBase( Name, status )
        {}
        void LookupFunctionPointers( VkInstance vkInstance ) override;
        void LookupFunctionPointers( VkDevice, PFN_vkGetDeviceProcAddr ) override {/*no device functions*/ }
        PFN_vkDestroySurfaceKHR m_vkDestroySurfaceKHR = nullptr;
        PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR m_vkGetPhysicalDeviceSurfaceCapabilitiesKHR = nullptr;
        PFN_vkGetPhysicalDeviceSurfaceFormatsKHR m_vkGetPhysicalDeviceSurfaceFormatsKHR = nullptr;
        PFN_vkGetPhysicalDeviceSurfacePresentModesKHR m_vkGetPhysicalDeviceSurfacePresentModesKHR = nullptr;
        PFN_vkGetPhysicalDeviceSurfaceSupportKHR m_vkGetPhysicalDeviceSurfaceSupportKHR = nullptr;
    };

#endif // VK_KHR_surface

#if VK_KHR_get_surface_capabilities2

    // Instance extension
    struct Ext_VK_KHR_get_surface_capabilities2 : public VulkanFunctionPointerExtensionHelper<VulkanExtensionType::eInstance>
    {
        using tBase = VulkanFunctionPointerExtensionHelper;
        static constexpr auto Name = VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME;
        explicit Ext_VK_KHR_get_surface_capabilities2( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : tBase( Name, status )
        {}
        void LookupFunctionPointers( VkInstance vkInstance ) override;
        void LookupFunctionPointers( VkDevice, PFN_vkGetDeviceProcAddr ) override {/*no device functions*/ }
        PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR  m_vkGetPhysicalDeviceSurfaceCapabilities2KHR = nullptr;
        PFN_vkGetPhysicalDeviceSurfaceFormats2KHR       m_vkGetPhysicalDeviceSurfaceFormats2KHR = nullptr;
    };

#endif // VK_KHR_get_surface_capabilities2

    //
    // Vulkan 1.1 (VK_VERSION_1_1) provided features/properties.
    // Same interface as other extensions but do not need to be added to the list of extension names on vkCreateDevice.
    //

    struct Vulkan_SubgroupPropertiesHook : public VulkanExtension<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = "SubgroupProperties";
        using tBase = VulkanExtension<VulkanExtensionType::eDevice>;
        Vulkan_SubgroupPropertiesHook& operator=( const Vulkan_SubgroupPropertiesHook& ) = delete;
        Vulkan_SubgroupPropertiesHook( const Vulkan_SubgroupPropertiesHook& ) = delete;
        explicit Vulkan_SubgroupPropertiesHook( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : tBase( Name, status ), m_GetPhysicalDevicePropertiesHook( this ), m_VulkanDevicePropertiesPrintHook( this )
        {}
        VkPhysicalDeviceSubgroupProperties Properties { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };

    private:
        void Register( Vulkan& vulkan ) override
        {
            vulkan.AddExtensionHooks( &m_GetPhysicalDevicePropertiesHook, &m_VulkanDevicePropertiesPrintHook );
        }

        friend class GetPhysicalDevicePropertiesHook<Vulkan_SubgroupPropertiesHook, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES>;
        friend class VulkanDevicePropertiesPrintHook<Vulkan_SubgroupPropertiesHook, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES>;
        GetPhysicalDevicePropertiesHook<Vulkan_SubgroupPropertiesHook, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES> m_GetPhysicalDevicePropertiesHook;
        VulkanDevicePropertiesPrintHook<Vulkan_SubgroupPropertiesHook, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES> m_VulkanDevicePropertiesPrintHook;

        void PrintProperties() const;
    };

    struct Vulkan_StorageFeaturesHook : public VulkanExtension<VulkanExtensionType::eDevice>
    {
        static constexpr auto Name = "StorageFeatures";
        using tBase = VulkanExtension<VulkanExtensionType::eDevice>;
        Vulkan_StorageFeaturesHook& operator=( const Vulkan_StorageFeaturesHook& ) = delete;
        Vulkan_StorageFeaturesHook( const Vulkan_StorageFeaturesHook& ) = delete;
        explicit Vulkan_StorageFeaturesHook( VulkanExtensionStatus status = VulkanExtensionStatus::eRequired ) : tBase( std::string(), status ), m_DeviceCreateInfoHook(this), m_GetPhysicalDeviceFeaturesHook(this), m_VulkanDeviceFeaturePrintHook(this)
        {}
        VkPhysicalDevice16BitStorageFeatures AvailableFeatures { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR };

    private:
        friend class DeviceCreateInfoHook< Vulkan_StorageFeaturesHook, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR>;
        friend class GetPhysicalDeviceFeaturesHook< Vulkan_StorageFeaturesHook, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR>;
        friend class VulkanDeviceFeaturePrintHook< Vulkan_StorageFeaturesHook, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR>;
        DeviceCreateInfoHook<Vulkan_StorageFeaturesHook, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR>          m_DeviceCreateInfoHook;
        GetPhysicalDeviceFeaturesHook<Vulkan_StorageFeaturesHook, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR> m_GetPhysicalDeviceFeaturesHook;
        VulkanDeviceFeaturePrintHook<Vulkan_StorageFeaturesHook, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR>  m_VulkanDeviceFeaturePrintHook;

        void Register( Vulkan& vulkan ) override
        {
            vulkan.AddExtensionHooks( &m_DeviceCreateInfoHook, &m_GetPhysicalDeviceFeaturesHook, &m_VulkanDeviceFeaturePrintHook );
        }
        virtual void PrintFeatures() const;
        virtual void PopulateRequestedFeatures() { RequestedFeatures = AvailableFeatures; }
        VkPhysicalDevice16BitStorageFeatures RequestedFeatures { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR };
    };

}; // namespace
