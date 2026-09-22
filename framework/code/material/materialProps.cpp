// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2020 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "materialProps.h"
#include <cstring>
#include "system/os_common.h"
#include "memory/vulkan/indexBufferObject.hpp"
#include "memory/vulkan/uniform.hpp"
#include "memory/vulkan/vertexBufferObject.hpp"
#include "mesh/mesh.hpp"
#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_support.hpp"
#include "texture/vulkan/texture.hpp"


//-----------------------------------------------------------------------------
void MaterialProps::InitOneLayout(Vulkan* pVulkan)
//-----------------------------------------------------------------------------
{
    VkResult RetVal = VK_SUCCESS;

    // This layout needs to match one set for vkUpdateDescriptorSets

    // Used as scratch for setting them up below.  Just make large enough to play with
    uint32_t WhichLayout = 0;
    std::array<VkDescriptorSetLayoutBinding, 25> BindingLayouts {};

    // The only thing that changes on this structure is the number of valid bindings
    VkDescriptorSetLayoutCreateInfo LayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    LayoutInfo.flags = 0;
    LayoutInfo.bindingCount = 0;    // Changed by each instance below
    LayoutInfo.pBindings = BindingLayouts.data();

    // The only thing that changes on this structure is the pointer to the layout 
    VkPipelineLayoutCreateInfo PipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    PipelineLayoutInfo.setLayoutCount = 1;  // Should never change from 1
    PipelineLayoutInfo.pSetLayouts = NULL;  // Changed by each instance below

    // Always a Vert Buffer
    BindingLayouts[WhichLayout].binding = SHADER_VERT_UBO_LOCATION;
    BindingLayouts[WhichLayout].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    BindingLayouts[WhichLayout].descriptorCount = 1;
    BindingLayouts[WhichLayout].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    WhichLayout++;

    // Always a Frag Buffer
    BindingLayouts[WhichLayout].binding = SHADER_FRAG_UBO_LOCATION;
    BindingLayouts[WhichLayout].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    BindingLayouts[WhichLayout].descriptorCount = 1;
    BindingLayouts[WhichLayout].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    WhichLayout++;

    // Samplers
    for (uint32_t WhichTexture = 0; WhichTexture < MAX_MATERIAL_SAMPLERS; WhichTexture++)
    {
        if (pTexture[WhichTexture] != nullptr)
        {
            BindingLayouts[WhichLayout].binding = SHADER_BASE_TEXTURE_LOC + WhichTexture;
            BindingLayouts[WhichLayout].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            BindingLayouts[WhichLayout].descriptorCount = 1;
            BindingLayouts[WhichLayout].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            WhichLayout++;
        }
    }

    // How many layouts did we end up with?
    LayoutInfo.bindingCount = WhichLayout;

    // Update the layout info structure and create the layout
    RetVal = vkCreateDescriptorSetLayout(pVulkan->m_VulkanDevice, &LayoutInfo, NULL, &DescLayout);
    if (!CheckVkError("vkCreateDescriptorSetLayout()", RetVal))
    {
        return;
    }

    // Update the other layout structure and create the pipeline layout
    PipelineLayoutInfo.pSetLayouts = &DescLayout;
    RetVal = vkCreatePipelineLayout(pVulkan->m_VulkanDevice, &PipelineLayoutInfo, NULL, &PipelineLayout);
    if (!CheckVkError("vkCreatePipelineLayout()", RetVal))
    {
        return;
    }
}

//-----------------------------------------------------------------------------
bool MaterialProps::InitOnePipeline(Vulkan* pVulkan, Mesh<Vulkan>* pMesh, uint32_t TargetWidth, uint32_t TargetHeight, const RenderContext<Vulkan>& renderingPassContext)
//-----------------------------------------------------------------------------
{
    // Raster State
    VkPipelineRasterizationStateCreateInfo RasterState{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    RasterState.flags = 0;
    RasterState.depthClampEnable = VK_FALSE;
    RasterState.rasterizerDiscardEnable = VK_FALSE;
    RasterState.polygonMode = VK_POLYGON_MODE_FILL;
    RasterState.cullMode = CullMode;
    RasterState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    RasterState.depthBiasEnable = DepthBiasEnable;

    if (DepthBiasEnable == VK_TRUE)
    {
        RasterState.depthBiasConstantFactor = DepthBiasConstant;
        RasterState.depthBiasClamp = DepthBiasClamp;
        RasterState.depthBiasSlopeFactor = DepthBiasSlope;

        // Sample sets these to 0 and then lets dynamic state handle it
        RasterState.depthBiasConstantFactor = 0.0f;
        RasterState.depthBiasClamp = 0.0f;
        RasterState.depthBiasSlopeFactor = 0.0f;
    }
    else
    {
        RasterState.depthBiasConstantFactor = 0.0f;
        RasterState.depthBiasClamp = 0.0f;
        RasterState.depthBiasSlopeFactor = 0.0f;
    }

    RasterState.lineWidth = 1.0f;

    // Blend State (Needs to be an array)

    // TODO: Set up actual blending
    VkPipelineColorBlendAttachmentState BlendStates[MAX_GMEM_OUTPUT_LAYERS];
    memset(BlendStates, 0, sizeof(BlendStates));

    if (NumBlendStates > MAX_GMEM_OUTPUT_LAYERS)
    {
        LOGE("Material sets more than %d blend states!", NumBlendStates);
        return false;
    }
    for (uint32_t WhichBlendState = 0; WhichBlendState < NumBlendStates; WhichBlendState++)
    {
        // Simply assign the whole structure over
        BlendStates[WhichBlendState] = BlendStates[WhichBlendState];

        // Overwrite the color write mask to be sure
        BlendStates[WhichBlendState].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    }   // Each Blend State

    VkPipelineColorBlendStateCreateInfo BlendStateInfo{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    BlendStateInfo.flags = 0;
    BlendStateInfo.logicOpEnable = VK_FALSE;
    // BlendStateInfo.logicOp;
    BlendStateInfo.attachmentCount = NumBlendStates;
    BlendStateInfo.pAttachments = BlendStates;
    // BlendStateInfo.blendConstants[4];

    // Viewport and Scissor
    // TODO: Handle rendering into RT
    VkViewport Viewport{};
    Viewport.x = 0.0f;
    Viewport.y = 0.0f;
    Viewport.width = (float)TargetWidth;
    Viewport.height = (float)TargetHeight;
    Viewport.minDepth = 0.0f;
    Viewport.maxDepth = 1.0f;

    VkRect2D Scissor{};
    Scissor.offset.x = 0;
    Scissor.offset.y = 0;
    Scissor.extent.width = TargetWidth;
    Scissor.extent.height = TargetHeight;

    // Depth and Stencil
    VkPipelineDepthStencilStateCreateInfo DepthStencilInfo{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    DepthStencilInfo.flags = 0;
    DepthStencilInfo.depthTestEnable = DepthTestEnable;
    DepthStencilInfo.depthWriteEnable = DepthWriteEnable;
    DepthStencilInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    DepthStencilInfo.depthBoundsTestEnable = VK_FALSE;
    DepthStencilInfo.stencilTestEnable = VK_FALSE;

    DepthStencilInfo.front.failOp = VK_STENCIL_OP_KEEP;
    DepthStencilInfo.front.passOp = VK_STENCIL_OP_KEEP;
    DepthStencilInfo.front.depthFailOp = VK_STENCIL_OP_KEEP;
    DepthStencilInfo.front.compareOp = VK_COMPARE_OP_NEVER;
    DepthStencilInfo.front.compareMask = 0;
    DepthStencilInfo.front.writeMask = 0;
    DepthStencilInfo.front.reference = 0;

    DepthStencilInfo.back.failOp = VK_STENCIL_OP_KEEP;
    DepthStencilInfo.back.passOp = VK_STENCIL_OP_KEEP;
    DepthStencilInfo.back.depthFailOp = VK_STENCIL_OP_KEEP;
    DepthStencilInfo.back.compareOp = VK_COMPARE_OP_ALWAYS;
    DepthStencilInfo.back.compareMask = 0;
    DepthStencilInfo.back.writeMask = 0;
    DepthStencilInfo.back.reference = 0;

    DepthStencilInfo.minDepthBounds = 0.0f;
    DepthStencilInfo.maxDepthBounds = 0.0f;

    // Setup any dynamic states
    std::vector<VkDynamicState> dynamicStateEnables;

    if (DepthBiasEnable == VK_TRUE)
        dynamicStateEnables.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);

    // Grab the vertex input state for the vertex buffer we will be binding.
    VkPipelineVertexInputStateCreateInfo visci = pMesh->m_VertexBuffers[0].CreatePipelineState();

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
    };

    if( !pVulkan->CreatePipeline( PipelineCache,
                                  &visci,
                                  PipelineLayout,
                                  renderingPassContext,
                                  &RasterState,
                                  &DepthStencilInfo,
                                  &BlendStateInfo,
                                  nullptr,//default multisample state
                                  &inputAssemblyState,
                                  dynamicStateEnables,
                                  &Viewport,
                                  &Scissor,
                                  VK_NULL_HANDLE,//task shader
                                  VK_NULL_HANDLE,//mesh shader
                                  pShader->VertShaderModule.GetVkShaderModule(),
                                  pShader->FragShaderModule.GetVkShaderModule(),
                                  nullptr,//specializationInfo
                                  false,//bAllowDerivation
                                  VK_NULL_HANDLE,//deriveFromPipeline
                                  &Pipeline ) )
    {
        LOGE("CreatePipeline error");
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool MaterialProps::InitDescriptorPool(Vulkan* pVulkan)
//-----------------------------------------------------------------------------
{
    VkResult RetVal = VK_SUCCESS;

    // Used as scratch for setting them up below.  Just make large enough to play with
    uint32_t PoolCount = 0;
    VkDescriptorPoolSize PoolSize[10];
    memset(PoolSize, 0, sizeof(PoolSize));

    // Need uniform buffers for Vert and Frag shaders
    PoolSize[PoolCount].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    PoolSize[PoolCount].descriptorCount = NUM_VULKAN_BUFFERS * 2;    // Need for each buffer (Vert + Frag) * NUM_VULKAN_BUFFERS
    PoolCount++;

    // Need combined image samplers for each possible texture
    PoolSize[PoolCount].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    PoolSize[PoolCount].descriptorCount = NUM_VULKAN_BUFFERS * MAX_MATERIAL_SAMPLERS;    // Need for each buffer
    PoolCount++;

    VkDescriptorPoolCreateInfo PoolInfo;
    memset(&PoolInfo, 0, sizeof(VkDescriptorPoolCreateInfo));
    PoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    PoolInfo.pNext = NULL;
    PoolInfo.flags = 0;     // VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT to allow them to be returned
    PoolInfo.maxSets = NUM_VULKAN_BUFFERS;  // Since descriptor sets come out of this pool we need more than one
    PoolInfo.poolSizeCount = PoolCount;
    PoolInfo.pPoolSizes = PoolSize;

    RetVal = vkCreateDescriptorPool(pVulkan->m_VulkanDevice, &PoolInfo, NULL, &DescPool);
    if (!CheckVkError("vkCreateDescriptorPool()", RetVal))
    {
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool MaterialProps::InitDescriptorSet(Vulkan* pVulkan)
//-----------------------------------------------------------------------------
{
    VkResult RetVal = VK_SUCCESS;

    // Need Descriptor Buffers
    VkDescriptorBufferInfo VertDescBufferInfo;
    memset(&VertDescBufferInfo, 0, sizeof(VkDescriptorBufferInfo));
    VertDescBufferInfo.buffer = pVertUniform->bufferInfo.buffer;
    if (VertUniformLength == 0)
    {
        // This means it has not been set
        if (VertUniformOffset != 0)
        {
            LOGE("DescriptorSet defines vert uniform buffer offset but the length is 0!");
        }
        VertDescBufferInfo.offset = pVertUniform->bufferInfo.offset;
        VertDescBufferInfo.range = pVertUniform->bufferInfo.range;
    }
    else
    {
        VertDescBufferInfo.offset = VertUniformOffset;
        VertDescBufferInfo.range = VertUniformLength;
    }

    VkDescriptorBufferInfo FragDescBufferInfo;
    memset(&FragDescBufferInfo, 0, sizeof(VkDescriptorBufferInfo));
    FragDescBufferInfo.buffer = pFragUniform->bufferInfo.buffer;
    if (FragUniformLength == 0)
    {
        // This means it has not been set
        if (FragUniformOffset != 0)
        {
            LOGE("DescriptorSet defines frag uniform buffer offset but the length is 0!");
        }
        FragDescBufferInfo.offset = pFragUniform->bufferInfo.offset;
        FragDescBufferInfo.range = pFragUniform->bufferInfo.range;
    }
    else
    {
        FragDescBufferInfo.offset = FragUniformOffset;
        FragDescBufferInfo.range = FragUniformLength;
    }

    VkDescriptorSetAllocateInfo DescSetInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    DescSetInfo.descriptorPool = DescPool;
    DescSetInfo.descriptorSetCount = 1;
    DescSetInfo.pSetLayouts = &DescLayout;

    RetVal = vkAllocateDescriptorSets(pVulkan->m_VulkanDevice, &DescSetInfo, &DescSet);
    if (!CheckVkError("vkAllocateDescriptorSets()", RetVal))
    {
        return false;
    }

    // Used as scratch for setting them up below.  Just make large enough to play with
    uint32_t DescCount = 0;

    std::array<VkDescriptorImageInfo, 25> DescImageInfo{};

    std::array<VkWriteDescriptorSet, 25> WriteInfo{};

    // Always a Vert Buffer
    WriteInfo[DescCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    WriteInfo[DescCount].pNext = NULL;
    WriteInfo[DescCount].dstSet = DescSet;
    WriteInfo[DescCount].dstBinding = SHADER_VERT_UBO_LOCATION;
    WriteInfo[DescCount].dstArrayElement = 0;
    WriteInfo[DescCount].descriptorCount = 1;
    WriteInfo[DescCount].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;    // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    WriteInfo[DescCount].pImageInfo = NULL;
    WriteInfo[DescCount].pBufferInfo = &VertDescBufferInfo;
    WriteInfo[DescCount].pTexelBufferView = NULL;
    DescCount++;

    // Always a Frag Buffer
    WriteInfo[DescCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    WriteInfo[DescCount].pNext = NULL;
    WriteInfo[DescCount].dstSet = DescSet;
    WriteInfo[DescCount].dstBinding = SHADER_FRAG_UBO_LOCATION;
    WriteInfo[DescCount].dstArrayElement = 0;
    WriteInfo[DescCount].descriptorCount = 1;
    WriteInfo[DescCount].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;    // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    WriteInfo[DescCount].pImageInfo = NULL;
    WriteInfo[DescCount].pBufferInfo = &FragDescBufferInfo;
    WriteInfo[DescCount].pTexelBufferView = NULL;
    DescCount++;

    // Samplers
    for (uint32_t WhichTexture = 0; WhichTexture < MAX_MATERIAL_SAMPLERS; WhichTexture++)
    {
        if (pTexture[WhichTexture] != nullptr)
        {
            DescImageInfo[DescCount] = apiCast<Vulkan>(pTexture[WhichTexture])->GetVkDescriptorImageInfo();

            WriteInfo[DescCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            WriteInfo[DescCount].pNext = NULL;
            WriteInfo[DescCount].dstSet = DescSet;
            WriteInfo[DescCount].dstBinding = SHADER_BASE_TEXTURE_LOC + WhichTexture;
            WriteInfo[DescCount].dstArrayElement = 0;
            WriteInfo[DescCount].descriptorCount = 1;
            WriteInfo[DescCount].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            WriteInfo[DescCount].pImageInfo = &DescImageInfo[DescCount];
            WriteInfo[DescCount].pBufferInfo = NULL;
            WriteInfo[DescCount].pTexelBufferView = NULL;
            DescCount++;
        }
    }

    vkUpdateDescriptorSets(pVulkan->m_VulkanDevice, DescCount, WriteInfo.data(), 0, NULL);

    return true;
}
