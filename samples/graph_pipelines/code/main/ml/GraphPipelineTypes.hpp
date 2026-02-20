//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include "main/applicationHelperBase.hpp"

namespace Ml
{
    struct GraphPipelineTensor
    {
        std::vector<int64_t>         strides;
        std::vector<int64_t>         dimensions;
        uint32_t                     port_binding_index = 0;

        VkTensorDescriptionARM       tensor_description = {};
        VkTensorARM                  tensor = VK_NULL_HANDLE;
        VkTensorViewARM              tensor_view = VK_NULL_HANDLE;

        VkBuffer                     aliased_buffer = VK_NULL_HANDLE;
        VkDeviceMemory               tensor_memory = VK_NULL_HANDLE;
    };

    struct GraphPipelineResources
    {
        VkDescriptorPool             tensor_descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSetLayout        tensor_descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorSet              tensor_descriptor_set = VK_NULL_HANDLE;
    };

    struct DataGraphPipelineInstance
    {
        VkPipelineCache              pipeline_cache = VK_NULL_HANDLE;
        VkPipelineLayout             pipeline_layout = VK_NULL_HANDLE;

        VkPipeline                   graph_pipeline = VK_NULL_HANDLE;
        VkDataGraphPipelineSessionARM graph_session = VK_NULL_HANDLE;

        std::vector<VkDeviceMemory>  session_memory;
    };
} // namespace Ml