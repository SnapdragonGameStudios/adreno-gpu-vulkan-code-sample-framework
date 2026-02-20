//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include "main/applicationHelperBase.hpp"

namespace Ml
{
    class GraphDispatch
    {
    public:
        GraphDispatch() = default;
        ~GraphDispatch() = default;

        static void RecordDispatch(
            VkCommandBuffer              cmd_buffer,
            VkPipeline                   pipeline,
            VkPipelineLayout             pipeline_layout,
            VkDescriptorSet              descriptor_set,
            VkDataGraphPipelineSessionARM session);

        static bool Submit(
            VkQueue              queue,
            VkCommandBuffer      cmd_buffer,
            VkSemaphore          wait_semaphore,
            VkPipelineStageFlags wait_stage_mask,
            VkSemaphore          signal_semaphore,
            VkFence              fence);

        inline bool IsValid() const { return true; }
    };
} // namespace Ml
