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
    class TensorResources
    {
    public:
        TensorResources() = default;
        ~TensorResources() = default;

        TensorResources(const TensorResources&) = delete;
        TensorResources& operator=(const TensorResources&) = delete;

        bool Initialize(
            VkDevice                                      device,
            VkPhysicalDevice                              physical_device,
            VkPhysicalDeviceDataGraphProcessingEngineARM& data_graph_engine,
            GraphPipelineTensor&                          input_tensor,
            GraphPipelineTensor&                          output_tensor,
            uint32_t                                      max_port_index);

        void Destroy(VkDevice device);

        inline bool IsValid() const { return m_is_valid; }

        inline const GraphPipelineResources& GetResources() const { return m_resources; }

    private:

        static bool FindMemoryType(
            VkPhysicalDevice      physical_device,
            uint32_t              type_bits,
            VkMemoryPropertyFlags properties,
            uint32_t&             out_index);

        bool CreateTensorInternal(
            VkDevice             device,
            VkPhysicalDevice     physical_device,
            GraphPipelineTensor& target_tensor);

    private:
        bool                   m_is_valid  = false;
        GraphPipelineResources m_resources = {};
    };
} // namespace Ml
