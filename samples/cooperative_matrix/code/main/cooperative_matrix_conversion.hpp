//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once
#include "vulkan/extensionHelpers.hpp"

// ABI-compatible declaration for the vendored headers that predate this extension.
struct QcomCoopMatConversionFeatures
{
    VkStructureType sType;
    void*           pNext;
    VkBool32        cooperativeMatrixConversion;
};
// VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_CONVERSION_FEATURES_QCOM
static constexpr VkStructureType kSTypeQcomCoopMatConvFeatures =
    static_cast<VkStructureType>(1000172000);

class QcomCoopMatConversionExtension final
    : public VulkanDeviceFeaturesExtensionHelper<QcomCoopMatConversionFeatures, kSTypeQcomCoopMatConvFeatures>
{
public:
    static constexpr const char* Name = "VK_QCOM_cooperative_matrix_conversion";
    explicit QcomCoopMatConversionExtension(VulkanExtensionStatus status)
        : VulkanDeviceFeaturesExtensionHelper("VK_QCOM_cooperative_matrix_conversion", status) {}
    void PrintFeatures() const override
    {
        LOGI("VK_QCOM_cooperative_matrix_conversion: cooperativeMatrixConversion = %s",
             AvailableFeatures.cooperativeMatrixConversion ? "True" : "False");
    }
};
