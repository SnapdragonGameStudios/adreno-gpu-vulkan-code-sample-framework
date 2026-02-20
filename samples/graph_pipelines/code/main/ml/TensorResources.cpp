//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#include <cstring>
#include "TensorResources.hpp"

bool Ml::TensorResources::FindMemoryType(
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

bool Ml::TensorResources::CreateTensorInternal(
    VkDevice            device,
    VkPhysicalDevice    physical_device,
    GraphPipelineTensor& target_tensor)
{
    // TENSOR DESCRIPTION
    target_tensor.tensor_description = VkTensorDescriptionARM
    {
        .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
        .pNext = nullptr,
        .tiling = VK_TENSOR_TILING_LINEAR_ARM,
        .format = VK_FORMAT_R8_UNORM,
        .dimensionCount = static_cast<uint32_t>(target_tensor.dimensions.size()),
        .pDimensions = target_tensor.dimensions.data(),
        .pStrides = nullptr,
        .usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM/* | VK_TENSOR_USAGE_SHADER_BIT_ARM*/
    };

    // TENSOR OBJECT
    VkExternalMemoryTensorCreateInfoARM external_info = {};
    external_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_TENSOR_CREATE_INFO_ARM;
    external_info.pNext = nullptr;
    external_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

    VkTensorCreateInfoARM tensor_info = {};
    tensor_info.sType = VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM;
    tensor_info.pNext = &external_info;
    tensor_info.flags = 0;
    tensor_info.pDescription = &target_tensor.tensor_description;
    tensor_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    tensor_info.queueFamilyIndexCount = 0;
    tensor_info.pQueueFamilyIndices = nullptr;

    if (vkCreateTensorARM(device, &tensor_info, nullptr, &target_tensor.tensor) != VK_SUCCESS)
    {
        return false;
    }

    // MEMORY REQUIREMENTS
    VkDeviceTensorMemoryRequirementsARM device_mem_req_info = {};
    device_mem_req_info.sType = VK_STRUCTURE_TYPE_DEVICE_TENSOR_MEMORY_REQUIREMENTS_ARM;
    device_mem_req_info.pNext = nullptr;
    device_mem_req_info.pCreateInfo = &tensor_info;

    VkMemoryRequirements2 tensor_mem_req = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
    vkGetDeviceTensorMemoryRequirementsARM(device, &device_mem_req_info, &tensor_mem_req);

    // ALIASED BUFFER
    const uint32_t buffer_size =
        static_cast<uint32_t>(target_tensor.dimensions[0] * target_tensor.dimensions[1] * target_tensor.dimensions[2]);

    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.pNext = nullptr;
    buffer_info.flags = 0;
    buffer_info.size = buffer_size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &buffer_info, nullptr, &target_tensor.aliased_buffer) != VK_SUCCESS)
    {
        return false;
    }

    VkMemoryRequirements buffer_mem_req = {};
    vkGetBufferMemoryRequirements(device, target_tensor.aliased_buffer, &buffer_mem_req);

    uint32_t memory_type_index = 0;
    if (!FindMemoryType(physical_device, buffer_mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory_type_index))
    {
        return false;
    }

    VkExportMemoryAllocateInfo export_alloc_info = {};
    export_alloc_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_alloc_info.pNext = nullptr;
    export_alloc_info.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = &export_alloc_info;
    alloc_info.allocationSize = buffer_mem_req.size;
    alloc_info.memoryTypeIndex = memory_type_index;

    if (vkAllocateMemory(device, &alloc_info, nullptr, &target_tensor.tensor_memory) != VK_SUCCESS)
    {
        return false;
    }

    VkBindTensorMemoryInfoARM bind_tensor_info = {};
    bind_tensor_info.sType = VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM;
    bind_tensor_info.pNext = nullptr;
    bind_tensor_info.tensor = target_tensor.tensor;
    bind_tensor_info.memory = target_tensor.tensor_memory;
    bind_tensor_info.memoryOffset = 0;

    if (vkBindTensorMemoryARM(device, 1, &bind_tensor_info) != VK_SUCCESS)
    {
        return false;
    }

    if (vkBindBufferMemory(device, target_tensor.aliased_buffer, target_tensor.tensor_memory, 0) != VK_SUCCESS)
    {
        return false;
    }

    // TENSOR VIEW
    VkTensorViewCreateInfoARM view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM;
    view_info.pNext = nullptr;
    view_info.flags = 0;
    view_info.tensor = target_tensor.tensor;
    view_info.format = target_tensor.tensor_description.format;

    if (vkCreateTensorViewARM(device, &view_info, nullptr, &target_tensor.tensor_view) != VK_SUCCESS)
    {
        return false;
    }

    return true;
}

bool Ml::TensorResources::Initialize(
    VkDevice                          device,
    VkPhysicalDevice                  physical_device,
    VkPhysicalDeviceDataGraphProcessingEngineARM& data_graph_engine,
    GraphPipelineTensor& input_tensor,
    GraphPipelineTensor& output_tensor,
    uint32_t                          max_port_index)
{
    m_is_valid = false;

    if (!CreateTensorInternal(device, physical_device, input_tensor))
    {
        return false;
    }

    if (!CreateTensorInternal(device, physical_device, output_tensor))
    {
        return false;
    }

    // DESCRIPTOR POOL
    VkDataGraphProcessingEngineCreateInfoARM engine_info = {};
    engine_info.sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PROCESSING_ENGINE_CREATE_INFO_ARM;
    engine_info.pNext = nullptr;
    engine_info.processingEngineCount = 1;
    engine_info.pProcessingEngines = &data_graph_engine;

    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
    pool_size.descriptorCount = max_port_index + 1;

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.pNext = &engine_info;
    pool_info.flags = 0;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &m_resources.tensor_descriptor_pool) != VK_SUCCESS)
    {
        return false;
    }

    // DESCRIPTOR SET LAYOUT
    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding = input_tensor.port_binding_index;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_ALL;

    bindings[1].binding = output_tensor.port_binding_index;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_ALL;

    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.pNext = nullptr;
    layout_info.flags = 0;
    layout_info.bindingCount = 2;
    layout_info.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &m_resources.tensor_descriptor_set_layout) != VK_SUCCESS)
    {
        return false;
    }

    // DESCRIPTOR SET ALLOC
    VkDescriptorSetAllocateInfo alloc_ds_info = {};
    alloc_ds_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_ds_info.descriptorPool = m_resources.tensor_descriptor_pool;
    alloc_ds_info.descriptorSetCount = 1;
    alloc_ds_info.pSetLayouts = &m_resources.tensor_descriptor_set_layout;

    if (vkAllocateDescriptorSets(device, &alloc_ds_info, &m_resources.tensor_descriptor_set) != VK_SUCCESS)
    {
        return false;
    }

    // DESCRIPTOR SET UPDATE (tensor view writes)
    VkWriteDescriptorSetTensorARM tensor_writes[2] = {};
    tensor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM;
    tensor_writes[0].pNext = nullptr;
    tensor_writes[0].tensorViewCount = 1;
    tensor_writes[0].pTensorViews = &input_tensor.tensor_view;

    tensor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM;
    tensor_writes[1].pNext = nullptr;
    tensor_writes[1].tensorViewCount = 1;
    tensor_writes[1].pTensorViews = &output_tensor.tensor_view;

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = &tensor_writes[0];
    writes[0].dstSet = m_resources.tensor_descriptor_set;
    writes[0].dstBinding = input_tensor.port_binding_index;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].pNext = &tensor_writes[1];
    writes[1].dstSet = m_resources.tensor_descriptor_set;
    writes[1].dstBinding = output_tensor.port_binding_index;
    writes[1].dstArrayElement = 0;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM;

    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

    m_is_valid = true;
    return true;
}

void Ml::TensorResources::Destroy(VkDevice device)
{
    if (m_resources.tensor_descriptor_pool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device, m_resources.tensor_descriptor_pool, nullptr);
        m_resources.tensor_descriptor_pool = VK_NULL_HANDLE;
    }

    if (m_resources.tensor_descriptor_set_layout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, m_resources.tensor_descriptor_set_layout, nullptr);
        m_resources.tensor_descriptor_set_layout = VK_NULL_HANDLE;
    }

    m_resources.tensor_descriptor_set = VK_NULL_HANDLE;
    m_is_valid = false;
}