//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#include "QcomDataGraphModel.hpp"
#include <cstring>
#include <vector>

bool Ml::QcomDataGraphModel::IsExtensionSupported(VkPhysicalDevice physical_device)
{
    uint32_t extension_count = 0;
    if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr) != VK_SUCCESS)
    {
        return false;
    }

    std::vector<VkExtensionProperties> props(extension_count);
    if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, props.data()) != VK_SUCCESS)
    {
        return false;
    }

    for (const auto& p : props)
    {
        if (std::strcmp(p.extensionName, VK_QCOM_DATA_GRAPH_MODEL_EXTENSION_NAME) == 0)
        {
            return true;
        }
    }

    return false;
}

bool Ml::QcomDataGraphModel::QueryFeatures(
    VkPhysicalDevice                          physical_device,
    VkPhysicalDeviceDataGraphModelFeaturesQCOM& out_features)
{
    out_features = {};
    out_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_MODEL_FEATURES_QCOM;
    out_features.pNext = nullptr;

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &out_features;

    vkGetPhysicalDeviceFeatures2(physical_device, &features2); // fills out_features
    return true;
}

bool Ml::QcomDataGraphModel::ValidateModelCacheBlob(
    const std::vector<unsigned char>& model_blob,
    uint32_t& out_cache_version)
{
    out_cache_version = 0;

    // Header is defined as VkPipelineCacheHeaderVersionDataGraphQCOM and intended size is 28 bytes.
    // It includes headerSize, headerVersion, cacheType, cacheVersion, toolchainVersion[]. 
    // https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCacheHeaderVersionDataGraphQCOM.html
    if (model_blob.size() < 28)
    {
        return false;
    }

    const unsigned char* p = model_blob.data();

    const uint32_t header_size = ReadU32LE(p + 0);
    const uint32_t header_version = ReadU32LE(p + 4);
    const uint32_t cache_type = ReadU32LE(p + 8);
    const uint32_t cache_version = ReadU32LE(p + 12);

    // Verify header identifies a QCOM data-graph model cache.
    // headerVersion must be VK_PIPELINE_CACHE_HEADER_VERSION_DATA_GRAPH_QCOM.
    // https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCacheHeaderVersion.html
    // https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCacheHeaderVersionDataGraphQCOM.html
    if (header_version != uint32_t(VK_PIPELINE_CACHE_HEADER_VERSION_DATA_GRAPH_QCOM))
    {
        return false;
    }

    // cacheType must match VkDataGraphModelCacheTypeQCOM. Generic binary is currently the defined value.
    // https://docs.vulkan.org/refpages/latest/refpages/source/VkDataGraphModelCacheTypeQCOM.html
    // https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCacheHeaderVersionDataGraphQCOM.html
    if (cache_type != uint32_t(VK_DATA_GRAPH_MODEL_CACHE_TYPE_GENERIC_BINARY_QCOM))
    {
        return false;
    }

    // headerSize should be 28 per intended layout, but we accept >= 28 to be tolerant.
    // https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCacheHeaderVersionDataGraphQCOM.html
    if (header_size < 28)
    {
        return false;
    }

    out_cache_version = cache_version;
    return true;
}

bool Ml::QcomDataGraphModel::SelectQcomEngineAndOperationForQueueFamily(
    VkPhysicalDevice              physical_device,
    uint32_t                      queue_family_index,
    QCOM_MODEL_OPERATION          preferred_operation,
    QcomSelectedDataGraphSupport& out_selection)
{
    out_selection = {};

    uint32_t count = 0;
    VkResult r = vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM(
        physical_device,
        queue_family_index,
        &count,
        nullptr);

    if (r != VK_SUCCESS || count == 0)
    {
        return false;
    }

    std::vector<VkQueueFamilyDataGraphPropertiesARM> props(count);
    for (auto& p : props)
    {
        p.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_DATA_GRAPH_PROPERTIES_ARM;
        p.pNext = nullptr;
    }

    r = vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM(
        physical_device,
        queue_family_index,
        &count,
        props.data());

    if (r != VK_SUCCESS || count == 0)
    {
        return false;
    }

    const VkPhysicalDeviceDataGraphOperationTypeARM wanted_op_type =
        (preferred_operation == QCOM_MODEL_OPERATION::NEURAL_MODEL)
        ? VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_NEURAL_MODEL_QCOM
        : VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_BUILTIN_MODEL_QCOM;

    // Find a QCOM engine type (NEURAL_QCOM preferred, COMPUTE_QCOM fallback) paired with the wanted operation.
    // QCOM engine types and operation types are defined by VK_QCOM_data_graph_model.
    // https://docs.vulkan.org/refpages/latest/refpages/source/VK_QCOM_data_graph_model.html
    // https://github.khronos.org/Vulkan-Site/refpages/latest/refpages/source/VkPhysicalDeviceDataGraphOperationTypeARM.html
    // https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPhysicalDeviceQueueFamilyDataGraphPropertiesARM.html
    auto Match = [&](VkPhysicalDeviceDataGraphProcessingEngineTypeARM engine_type) -> bool
    {
        for (const auto& p : props)
        {
            if (p.engine.type == engine_type && p.operation.operationType == wanted_op_type)
            {
                out_selection.is_valid = true;
                out_selection.engine = p.engine;
                out_selection.operation = p.operation;
                return true;
            }
        }
        return false;
    };

    if (Match(VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_NEURAL_QCOM))
    {
        return true;
    }

    if (Match(VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_COMPUTE_QCOM))
    {
        return true;
    }

    // If we couldn't match preferred op, try the other op type (some stacks publish only one).
    const VkPhysicalDeviceDataGraphOperationTypeARM alt_op_type =
        (wanted_op_type == VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_NEURAL_MODEL_QCOM)
        ? VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_BUILTIN_MODEL_QCOM
        : VK_PHYSICAL_DEVICE_DATA_GRAPH_OPERATION_TYPE_NEURAL_MODEL_QCOM;

    for (const auto& p : props)
    {
        if ((p.engine.type == VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_NEURAL_QCOM ||
            p.engine.type == VK_PHYSICAL_DEVICE_DATA_GRAPH_PROCESSING_ENGINE_TYPE_COMPUTE_QCOM) &&
            p.operation.operationType == alt_op_type)
        {
            out_selection.is_valid = true;
            out_selection.engine = p.engine;
            out_selection.operation = p.operation;
            return true;
        }
    }

    return false;
}
