// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "vulkan/vulkan.hpp"

#include "system/os_common.h"

// GLM Include Files
#define GLmFORCE_CXX03
#define GLmDEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

class ShaderInfo;
template<typename T_GFXAPI> class TextureT;

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
// MaterialVk Description
//=============================================================================

typedef struct _MaterialVk
{
    // Depth Test/Write
    VkBool32                DepthTestEnable;
    VkBool32                DepthWriteEnable;

    // Depth Bias
    VkBool32                DepthBiasEnable;
    float                   DepthBiasConstant;
    float                   DepthBiasClamp;
    float                   DepthBiasSlope;

    // Culling
    VkCullModeFlagBits      CullMode;

    // The Shader (Use references since we are not cleaning anything up)
    ShaderInfo*             pShader;

    // Textures (Use references since we are not cleaning anything up)
    Texture<Vulkan>*        pColorTexture;
    Texture<Vulkan>*        pNormalTexture;
    Texture<Vulkan>*        pShadowDepthTexture;
    Texture<Vulkan>*        pShadowColorTexture;
    Texture<Vulkan>*        pEnvironmentCubeMap;
    Texture<Vulkan>*        pIrradianceCubeMap;
    Texture<Vulkan>*        pReflectTexture;

    // Constant Buffers
    uint32_t                VertUniformOffset;
    uint32_t                VertUniformLength;
    Uniform<Vulkan>*        pVertUniform;

    uint32_t                FragUniformOffset;
    uint32_t                FragUniformLength;
    Uniform<Vulkan>*        pFragUniform;

    // Vulkan Objects
    VkDescriptorPool        DescPool;
    VkDescriptorSet         DescSet;

    VkDescriptorSetLayout   DescLayout;
    VkPipelineLayout        PipelineLayout;

    VkPipeline              Pipeline;
} MaterialVk;
