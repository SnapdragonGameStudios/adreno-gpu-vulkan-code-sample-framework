//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#include "GraphDispatch.hpp"

void Ml::GraphDispatch::RecordDispatch(
    VkCommandBuffer               cmd_buffer,
    VkPipeline                    pipeline,
    VkPipelineLayout              pipeline_layout,
    VkDescriptorSet               descriptor_set,
    VkDataGraphPipelineSessionARM session)
{
    vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, pipeline);

    vkCmdBindDescriptorSets(
        cmd_buffer,
        VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM,
        pipeline_layout,
        0,
        1,
        &descriptor_set,
        0,
        nullptr);

    VkDataGraphPipelineDispatchInfoARM dispatch_info = {};
    dispatch_info.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_DISPATCH_INFO_ARM;
    dispatch_info.pNext = nullptr;
    dispatch_info.flags = 0;

    vkCmdDispatchDataGraphARM(cmd_buffer, session, &dispatch_info);
}

bool Ml::GraphDispatch::Submit(
    VkQueue              queue,
    VkCommandBuffer      cmd_buffer,
    VkSemaphore          wait_semaphore,
    VkPipelineStageFlags wait_stage_mask,
    VkSemaphore          signal_semaphore,
    VkFence              fence)
{
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.pNext = nullptr;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buffer;

    VkPipelineStageFlags stage_mask = wait_stage_mask;

    if (wait_semaphore != VK_NULL_HANDLE)
    {
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &wait_semaphore;
        submit_info.pWaitDstStageMask = &stage_mask;
    }

    if (signal_semaphore != VK_NULL_HANDLE)
    {
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &signal_semaphore;
    }

    return vkQueueSubmit(queue, 1, &submit_info, fence) == VK_SUCCESS;
}