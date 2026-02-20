//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "DataGraphPipeline.hpp"

#include <cstring>

bool Ml::DataGraphPipeline::FindMemoryType(
    VkPhysicalDevice         physical_device,
    uint32_t                 type_bits,
    VkMemoryPropertyFlags    properties,
    uint32_t& out_index)
{
    VkPhysicalDeviceMemoryProperties mem_properties = {};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i)
    {
        const bool has_properties = (mem_properties.memoryTypes[i].propertyFlags & properties) == properties;
        const bool has_bit = (type_bits & (1u << i)) != 0;

        if (has_properties && has_bit)
        {
            out_index = i;
            return true;
        }
    }

    return false;
}

bool Ml::DataGraphPipeline::CreatePipelineCacheFromBlob(
    VkDevice                         device,
    const std::vector<unsigned char>& model_data,
    VkPipelineCache& out_cache)
{
    out_cache = VK_NULL_HANDLE;

    VkPipelineCacheCreateInfo cache_info = {};
    cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cache_info.pNext = nullptr;
    cache_info.flags = 0;
    cache_info.initialDataSize = model_data.size();
    cache_info.pInitialData = model_data.data();

    return vkCreatePipelineCache(device, &cache_info, nullptr, &out_cache) == VK_SUCCESS;
}

bool Ml::DataGraphPipeline::CreatePipelineLayout(
    VkDevice                  device,
    VkDescriptorSetLayout     tensor_set_layout,
    VkPipelineLayout& out_pipeline_layout)
{
    out_pipeline_layout = VK_NULL_HANDLE;

    VkPipelineLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.pNext = nullptr;
    layout_info.flags = 0;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &tensor_set_layout;
    layout_info.pushConstantRangeCount = 0;
    layout_info.pPushConstantRanges = nullptr;

    return vkCreatePipelineLayout(device, &layout_info, nullptr, &out_pipeline_layout) == VK_SUCCESS;
}

bool Ml::DataGraphPipeline::CreateGraphPipelineArmIdentifierPath(
    VkDevice                          device,
    VkPhysicalDeviceDataGraphProcessingEngineARM& data_graph_engine,
    VkPipelineLayout                  pipeline_layout,
    VkPipelineCache                   pipeline_cache,
    const VkDataGraphPipelineResourceInfoARM* resource_infos,
    uint32_t                          resource_info_count,
    const uint8_t* identifier_bytes,
    uint32_t                          identifier_size,
    VkPipeline& out_pipeline)
{
    out_pipeline = VK_NULL_HANDLE;

    VkDataGraphProcessingEngineCreateInfoARM engine_info = {};
    engine_info.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PROCESSING_ENGINE_CREATE_INFO_ARM;
    engine_info.pNext = nullptr;
    engine_info.processingEngineCount = 1;
    engine_info.pProcessingEngines = &data_graph_engine;

    VkDataGraphPipelineIdentifierCreateInfoARM identifier_info = {};
    identifier_info.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_IDENTIFIER_CREATE_INFO_ARM;
    identifier_info.pNext = &engine_info;
    identifier_info.identifierSize = identifier_size;
    identifier_info.pIdentifier = identifier_bytes;

    VkDataGraphPipelineShaderModuleCreateInfoARM module_info = {};
    module_info.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM;
    module_info.pNext = &identifier_info;
    module_info.module = VK_NULL_HANDLE;
    module_info.pName = "";

    VkDataGraphPipelineCreateInfoARM pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM;
    pipeline_info.pNext = &module_info;
    pipeline_info.flags = 0;
    pipeline_info.layout = pipeline_layout;
    pipeline_info.resourceInfoCount = resource_info_count;
    pipeline_info.pResourceInfos = resource_infos;

    return vkCreateDataGraphPipelinesARM(
        device,
        VK_NULL_HANDLE,
        pipeline_cache,
        1,
        &pipeline_info,
        nullptr,
        &out_pipeline) == VK_SUCCESS;
}

bool Ml::DataGraphPipeline::CreateSession(
    VkDevice                      device,
    VkPipeline                    pipeline,
    VkDataGraphPipelineSessionARM& out_session)
{
    out_session = VK_NULL_HANDLE;

    VkDataGraphPipelineSessionCreateInfoARM session_info = {};
    session_info.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM;
    session_info.pNext = nullptr;
    session_info.flags = 0;
    session_info.dataGraphPipeline = pipeline;

    return vkCreateDataGraphPipelineSessionARM(device, &session_info, nullptr, &out_session) == VK_SUCCESS;
}

bool Ml::DataGraphPipeline::AllocateAndBindSessionMemory(
    VkDevice                          device,
    VkPhysicalDevice                  physical_device,
    VkDataGraphPipelineSessionARM     session,
    std::vector<VkDeviceMemory>& out_session_mem)
{
    out_session_mem.clear();

    VkDataGraphPipelineSessionBindPointRequirementsInfoARM bind_reqs_info = {};
    bind_reqs_info.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM;
    bind_reqs_info.pNext = nullptr;
    bind_reqs_info.session = session;

    uint32_t bind_reqs_count = 0;
    if (vkGetDataGraphPipelineSessionBindPointRequirementsARM(device, &bind_reqs_info, &bind_reqs_count, nullptr) != VK_SUCCESS)
    {
        return false;
    }

    std::vector<VkDataGraphPipelineSessionBindPointRequirementARM> bind_reqs(bind_reqs_count);
    for (uint32_t i = 0; i < bind_reqs_count; ++i)
    {
        bind_reqs[i].sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM;
        bind_reqs[i].pNext = nullptr;
    }

    if (vkGetDataGraphPipelineSessionBindPointRequirementsARM(device, &bind_reqs_info, &bind_reqs_count, bind_reqs.data()) != VK_SUCCESS)
    {
        return false;
    }

    uint32_t mem_count = 0;
    for (uint32_t i = 0; i < bind_reqs_count; ++i)
    {
        if (bind_reqs[i].bindPointType != VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM)
        {
            return false;
        }

        const uint32_t old_size = static_cast<uint32_t>(out_session_mem.size());
        out_session_mem.resize(old_size + bind_reqs[i].numObjects);

        for (uint32_t j = 0; j < bind_reqs[i].numObjects; ++j)
        {
            VkDataGraphPipelineSessionMemoryRequirementsInfoARM mem_reqs_info = {};
            mem_reqs_info.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO_ARM;
            mem_reqs_info.pNext = nullptr;
            mem_reqs_info.session = session;
            mem_reqs_info.bindPoint = bind_reqs[i].bindPoint;
            mem_reqs_info.objectIndex = j;

            VkMemoryRequirements2 mem_reqs = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
            vkGetDataGraphPipelineSessionMemoryRequirementsARM(device, &mem_reqs_info, &mem_reqs);

            uint32_t memory_type_index = 0;
            if (!FindMemoryType(physical_device, mem_reqs.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory_type_index))
            {
                return false;
            }

            VkMemoryAllocateInfo alloc_info = {};
            alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc_info.pNext = nullptr;
            alloc_info.allocationSize = mem_reqs.memoryRequirements.size;
            alloc_info.memoryTypeIndex = memory_type_index;

            VkDeviceMemory memory = VK_NULL_HANDLE;
            if (vkAllocateMemory(device, &alloc_info, nullptr, &memory) != VK_SUCCESS)
            {
                return false;
            }

            VkBindDataGraphPipelineSessionMemoryInfoARM bind_info = {};
            bind_info.sType = VK_STRUCTURE_TYPE_BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM;
            bind_info.pNext = nullptr;
            bind_info.session = session;
            bind_info.bindPoint = bind_reqs[i].bindPoint;
            bind_info.objectIndex = j;
            bind_info.memory = memory;

            if (vkBindDataGraphPipelineSessionMemoryARM(device, 1, &bind_info) != VK_SUCCESS)
            {
                vkFreeMemory(device, memory, nullptr);
                return false;
            }

            out_session_mem[old_size + mem_count] = memory;
            ++mem_count;
        }
    }

    return true;
}