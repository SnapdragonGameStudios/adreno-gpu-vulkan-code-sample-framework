// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "dx12/dx12.hpp"

#include "system/os_common.h"

// GLM Include Files
#define GLmFORCE_CXX03
#define GLmDEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

//=============================================================================
// Uniform Buffers
//=============================================================================
// ************************************
// Test
// ************************************
typedef struct _TestVertUB
{
    glm::mat4   MVPMatrix;
    glm::mat4   ModelMatrix;
} TestVertUB;

typedef struct _TestFragUB
{
    glm::vec4   Color;
    glm::vec4   EyePos;
    glm::vec4   LightDir;
    glm::vec4   LightColor;
} TestFragUB;



//=============================================================================
// Material Description
//=============================================================================

typedef struct _Material
{
    // Depth Test/Write
    bool                    DepthTestEnable;
    bool                    DepthWriteEnable;

    // Depth Bias
    bool                    DepthBiasEnable;
    float                   DepthBiasConstant;
    float                   DepthBiasClamp;
    float                   DepthBiasSlope;

    // Culling
    int/*VkCullModeFlagBits*/   CullMode;


    // Textures (Use references since we are not cleaning anything up)
    //VulkanTexInfo* pColorTexture;
    //VulkanTexInfo* pNormalTexture;
    //VulkanTexInfo* pShadowDepthTexture;
    //VulkanTexInfo* pShadowColorTexture;
    //VulkanTexInfo* pEnvironmentCubeMap;
    //VulkanTexInfo* pIrradianceCubeMap;
    //VulkanTexInfo* pReflectTexture;

    // Constant Buffers
    //Uniform*                pVertUniform;
    //Uniform*                pFragUniform;

    // Vulkan Objects
    VkDescriptorPool        DescPool;
    VkDescriptorSet         DescSet;

    VkDescriptorSetLayout   DescLayout;
    VkPipelineLayout        PipelineLayout;

    VkPipeline              Pipeline;
} Material;

