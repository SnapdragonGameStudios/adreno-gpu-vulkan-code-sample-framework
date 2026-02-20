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

namespace Ml
{
    enum class QCOM_MODEL_OPERATION
    {
        NEURAL_MODEL,
        BUILTIN_MODEL
    };

    struct QcomSelectedDataGraphSupport
    {
        bool                                   is_valid = false;
        VkPhysicalDeviceDataGraphProcessingEngineARM engine = {};
        VkPhysicalDeviceDataGraphOperationSupportARM operation = {};
    };

    class QcomDataGraphModel
    {
    public:
        QcomDataGraphModel() = default;
        ~QcomDataGraphModel() = default;

        static bool IsExtensionSupported(VkPhysicalDevice physical_device);

        static bool QueryFeatures(
            VkPhysicalDevice                         physical_device,
            VkPhysicalDeviceDataGraphModelFeaturesQCOM& out_features);

        static bool ValidateModelCacheBlob(
            const std::vector<unsigned char>& model_blob,
            uint32_t& out_cache_version);

        static bool SelectQcomEngineAndOperationForQueueFamily(
            VkPhysicalDevice               physical_device,
            uint32_t                       queue_family_index,
            QCOM_MODEL_OPERATION           preferred_operation,
            QcomSelectedDataGraphSupport& out_selection);

    private:
        static inline uint32_t ReadU32LE(const unsigned char* ptr)
        {
            return (uint32_t(ptr[0])) |
                (uint32_t(ptr[1]) << 8) |
                (uint32_t(ptr[2]) << 16) |
                (uint32_t(ptr[3]) << 24);
        }
    };
} // namespace Ml
