// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#pragma once
#ifndef _MATERIAL_MATERIALPROPS_H_
#define _MATERIAL_MATERIALPROPS_H_

/// @file materialProps.hpp
/// @brief Simple material system for Vulkan samples.


#include "vulkan/vulkan.hpp"

// Under Vulkan we use uniform buffer objects in the shader. One for vert and one for frag
#define SHADER_VERT_UBO_LOCATION            0
#define SHADER_FRAG_UBO_LOCATION            1

// Texture Locations
#define SHADER_BASE_TEXTURE_LOC             2

// MRTs have arrays of blend states, color layers, etc.
#define MAX_GMEM_OUTPUT_LAYERS              8

// Number of "sampler2D", "samplerCube", etc. that can be bound in material
#define MAX_MATERIAL_SAMPLERS               16

// Forward declarations
struct ShaderInfo;
class TextureBase;
template <typename T_GFXAPI> class Mesh;
template <typename T_GFXAPI> struct Uniform;

//=============================================================================
// MaterialBase Description
//=============================================================================

typedef struct _MaterialProps
{
    // Used by the application, not the engine.  Nice to have it here
    uint32_t                RenderPassMask;

    // MaterialBase Flags
    // Need one per render target layer.  Usually this will be one, but not for MRTs
    uint32_t                            NumBlendStates;
    VkPipelineColorBlendAttachmentState BlendStates[MAX_GMEM_OUTPUT_LAYERS];

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

    // The Shader (responsibility of owner to clean up)
    ShaderInfo*             pShader;

    // Textures (responsibility of owner to clean up)
    TextureBase*                pTexture[MAX_MATERIAL_SAMPLERS];

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

    VkPipelineCache         PipelineCache;
    VkPipeline              Pipeline;


    // Helper Functions
    void                    InitOneLayout(Vulkan*);
    bool                    InitOnePipeline(Vulkan*, Mesh<Vulkan>* pMesh, uint32_t TargetWidth, uint32_t TargetHeight, const RenderContext<Vulkan>& renderingPassContext);
    bool                    InitDescriptorPool(Vulkan*);
    bool                    InitDescriptorSet(Vulkan*);

} MaterialProps;

#endif // _MATERIAL_MATERIALPROPS_H_
