
//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include <cstdint>
#include <vector>

#include "main/applicationHelperBase.hpp"
#include "GraphPipelineTypes.hpp"

namespace Ml
{
    class DataGraphPipeline
    {
    public:
        DataGraphPipeline() = default;
        ~DataGraphPipeline() = default;

        DataGraphPipeline(const DataGraphPipeline&) = delete;
        DataGraphPipeline& operator=(const DataGraphPipeline&) = delete;

        static bool CreatePipelineCacheFromBlob(
            VkDevice                        device,
            const std::vector<unsigned char>& model_data,
            VkPipelineCache& out_cache);

        static bool CreatePipelineLayout(
            VkDevice                  device,
            VkDescriptorSetLayout     tensor_set_layout,
            VkPipelineLayout& out_pipeline_layout);

        static bool CreateGraphPipelineArmIdentifierPath(
            VkDevice                          device,
            VkPhysicalDeviceDataGraphProcessingEngineARM& data_graph_engine,
            VkPipelineLayout                  pipeline_layout,
            VkPipelineCache                   pipeline_cache,
            const VkDataGraphPipelineResourceInfoARM* resource_infos,
            uint32_t                          resource_info_count,
            const uint8_t* identifier_bytes,
            uint32_t                          identifier_size,
            VkPipeline& out_pipeline);

        static bool CreateSession(
            VkDevice                     device,
            VkPipeline                   pipeline,
            VkDataGraphPipelineSessionARM& out_session);

        static bool AllocateAndBindSessionMemory(
            VkDevice                          device,
            VkPhysicalDevice                  physical_device,
            VkDataGraphPipelineSessionARM     session,
            std::vector<VkDeviceMemory>& out_session_mem);

        inline bool IsValid() const { return true; }

    private:
        static bool FindMemoryType(
            VkPhysicalDevice         physical_device,
            uint32_t                 type_bits,
            VkMemoryPropertyFlags    properties,
            uint32_t& out_index);
    };
} // namespace Ml