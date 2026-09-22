// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2020 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "application.hpp"
#include "main/applicationEntrypoint.hpp"
#include "material/materialProps.h"
#include "system/math_common.hpp"
#include "system/os_common.h"
#include "texture/vulkan/loaderKtx.hpp"
#include "vulkan/vulkan_support.hpp"
#include <filesystem>

glm::vec4 gClearColor = glm::vec4(0.3f, 0.3f, 0.3f, 1.0f);

VAR( char*, gSceneAssetModel, "UVSphere_Separate.gltf", kVariableNonpersistent );

// The vertex buffer bind id, used as a constant in various places in the sample
#define VERTEX_BUFFER_BIND_ID 0

// These are local defines that need to match what is in the shader
#define SHADER_DIFFUSE_TEXTURE_LOC          (SHADER_BASE_TEXTURE_LOC + 0)
#define SHADER_NORMAL_TEXTURE_LOC           (SHADER_BASE_TEXTURE_LOC + 1)
#define SHADER_SHADOW_DEPTH_TEXTURE_LOC     (SHADER_BASE_TEXTURE_LOC + 2)
#define SHADER_SHADOW_COLOR_TEXTURE_LOC     (SHADER_BASE_TEXTURE_LOC + 3)
#define SHADER_ENVIRONMENT_TEXTURE_LOC      (SHADER_BASE_TEXTURE_LOC + 4)
#define SHADER_IRRADIANCE_TEXTURE_LOC       (SHADER_BASE_TEXTURE_LOC + 5)
#define SHADER_REFLECT_TEXTURE_LOC          (SHADER_BASE_TEXTURE_LOC + 6)
#define NUM_SHADER_TEXTURE_LOCATIONS        (7)

//
// Implementation of the Application entrypoint (called by the framework)
// Construct the Application class
//
//-----------------------------------------------------------------------------
FrameworkApplicationBase* Application_ConstructApplication()
//-----------------------------------------------------------------------------
{
    return new Application();
}

//-----------------------------------------------------------------------------
Application::Application() : ApplicationHelperBase(), m_TestTexture{}
//-----------------------------------------------------------------------------
{
    // Object rotation
    m_TotalRotation = 0.0f;
    m_RotationSpeed = 0.5f;     // Radians per second

    // Camera
    m_CurrentCameraPos = glm::vec3(0.0f, 2.0f, 5.5f);
    m_CurrentCameraLook = glm::vec3(0.0f, 1.0f, 0.0f);

    // The Test Object
    m_ObjectScale = 1.0f;
    m_ObjectWorldPos = glm::vec3(0.0f, 1.25f, 0.0f);
}

//-----------------------------------------------------------------------------
Application::~Application()
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
bool Application::Initialize( uintptr_t windowHandle, uintptr_t instanceHandle)
//-----------------------------------------------------------------------------
{
    if (!ApplicationHelperBase::Initialize(windowHandle, instanceHandle))
    {
        return false;
    }

    if (!LoadMeshObjects())
        return false;

    if (!LoadTextures())
        return false;

    if (!LoadShaders())
        return false;

    if (!InitUniforms())
        return false;

    if (!InitMaterials())
        return false;

    if (!InitRenderPass())
        return false;

    if (!InitPipelines())
        return false;

    if (!InitCommandBuffers())
        return false;

    if (!BuildCmdBuffers())
        return false;

    return true;
}

//-----------------------------------------------------------------------------
void Application::Destroy()
//-----------------------------------------------------------------------------
{
    Vulkan* pVulkan = GetVulkan();

    // Command Buffers
    for(auto& CommandBuffer: m_CommandBuffers)
        CommandBuffer.Release();

    // Meshes
    m_TestMesh.Destroy();

    // Textures
    ReleaseTexture(*pVulkan, &m_TestTexture);

    // Shaders
    ReleaseShader(pVulkan, &m_TestShader);

    // Uniform Buffers
    ReleaseUniformBuffer(pVulkan, &m_TestVertUniform);
    ReleaseUniformBuffer(pVulkan, &m_TestFragUniform);

    // Finally call into base class destroy
    ApplicationHelperBase::Destroy();
}

//-----------------------------------------------------------------------------
void Application::Render(float fltDiffTime)
//-----------------------------------------------------------------------------
{
    // LOGI("Render() Entered...");

    // Grab the vulkan wrapper
    Vulkan* pVulkan = GetVulkan();

    // Obtain the next swap chain image for the next frame.
    auto currentBuffer = pVulkan->SetNextBackBuffer();

    // ********************************
    // Application Draw() - Begin
    // ********************************

    // Update uniform buffers with latest data
    UpdateUniforms(fltDiffTime);

    m_CommandBuffers[currentBuffer.idx].QueueSubmit(currentBuffer.semaphore, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, pVulkan->m_RenderCompleteSemaphore, currentBuffer.fence);

    // ********************************
    // Application Draw() - End
    // ********************************

    pVulkan->PresentQueue( pVulkan->m_RenderCompleteSemaphore, currentBuffer.swapchainPresentIdx );
}

//-----------------------------------------------------------------------------
void Application::TouchDownEvent(int iPointerID, float xPos, float yPos)
//-----------------------------------------------------------------------------
{
    LOGI("TouchDownEvent(%0.2f, %0.2f) Entered...", xPos, yPos);
}

//-----------------------------------------------------------------------------
void Application::TouchMoveEvent(int iPointerID, float xPos, float yPos)
//-----------------------------------------------------------------------------
{
    // LOGI("TouchMoveEvent(%0.2f, %0.2f) Entered...", xPos, yPos);
}

//-----------------------------------------------------------------------------
void Application::TouchUpEvent(int iPointerID, float xPos, float yPos)
//-----------------------------------------------------------------------------
{
    LOGI("TouchUpEvent(%0.2f, %0.2f) Entered...", xPos, yPos);
}

//-----------------------------------------------------------------------------
bool Application::LoadMeshObjects()
//-----------------------------------------------------------------------------
{
    const auto sceneAssetPath = std::filesystem::path(MESH_DESTINATION_PATH).append(gSceneAssetModel).string();
    LOGI("Loading glTF mesh: %s...", sceneAssetPath.c_str());
    if (!LoadGLTF(sceneAssetPath, VERTEX_BUFFER_BIND_ID, &m_TestMesh))
    {
        LOGE("Error loading Object mesh: %s", sceneAssetPath.c_str());
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::LoadTextures()
//-----------------------------------------------------------------------------
{
    Vulkan* pVulkan = GetVulkan();

    TextureKtx textureLoader { *pVulkan };
    if (!textureLoader.Initialize())
        return false;

    const auto textureAssetPath = std::filesystem::path(TEXTURE_DESTINATION_PATH).append("surf_d.ktx").string();
    m_TestTexture = textureLoader.LoadKtx(*pVulkan, *m_AssetManager, textureAssetPath.c_str(), m_SamplerRepeat.Copy());
    return !m_TestTexture.IsEmpty();
}

//-----------------------------------------------------------------------------
bool Application::LoadShaders()
//-----------------------------------------------------------------------------
{
    Vulkan* pVulkan = GetVulkan();

    LOGI("Loading Test shader...");
    auto debugVertFilename = std::filesystem::path(SHADER_DESTINATION_PATH).append( "Debug.vert.spv").string();
    auto debugFragFilename = std::filesystem::path(SHADER_DESTINATION_PATH).append( "Debug.frag.spv").string();
    LoadShader(pVulkan, *m_AssetManager, &m_TestShader, debugVertFilename.c_str(), debugFragFilename.c_str());

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitUniforms()
//-----------------------------------------------------------------------------
{
    Vulkan* pVulkan = GetVulkan();

    // These are only created here, they are not set to initial values
    LOGI("Creating uniform buffers...");
    CreateUniformBuffer(pVulkan, &m_TestVertUniform, sizeof(m_TestVertUniformData), NULL);
    CreateUniformBuffer(pVulkan, &m_TestFragUniform, sizeof(m_TestFragUniformData), NULL);

    return true;
}

//-----------------------------------------------------------------------------
bool Application::UpdateUniforms(float fltDiffTime)
//-----------------------------------------------------------------------------
{
    Vulkan* pVulkan = GetVulkan();

    // Projection Matrix
    float aspect = (float)gRenderWidth / (float)gRenderHeight;

    // These are labeled as global because they should go to config file
    float gFOV = PI_DIV_4;
    float gNearPlane = 1.0f;
    float gFarPlane = 100.0f;

    m_ProjectionMatrix = glm::perspectiveRH(gFOV, aspect, gNearPlane, gFarPlane);

    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    m_ViewMatrix = glm::lookAtRH(m_CurrentCameraPos, m_CurrentCameraLook, up);

    // Handle object rotation
    m_TotalRotation += m_RotationSpeed * fltDiffTime;
    if (m_TotalRotation > PI_MUL_2)
        m_TotalRotation -= PI_MUL_2;


    glm::mat4 OneModel = glm::mat4(1.0f);

    // ********************************
    // Test Vert Uniform 
    // ********************************
    OneModel = glm::translate(glm::mat4(1.0f), m_ObjectWorldPos);
    OneModel = glm::scale(OneModel, glm::vec3(m_ObjectScale, m_ObjectScale, m_ObjectScale));
    OneModel = glm::rotate(OneModel, m_TotalRotation, glm::vec3(0.0f, 1.0f, 0.0f));

    m_TestVertUniformData.MVPMatrix = m_ProjectionMatrix * m_ViewMatrix * OneModel;
    m_TestVertUniformData.ModelMatrix = OneModel;
    UpdateUniformBuffer(pVulkan, &m_TestVertUniform, sizeof(m_TestVertUniformData), &m_TestVertUniformData);

    // ********************************
    // Test Frag Uniform 
    // ********************************
    // These are labeled as global because they should go to config file

    float gSpecularExponent = 256.0f;
    glm::vec4 gLightDirection = glm::vec4(-0.5f, -1.0f, -1.0f, 0.0);

    m_TestFragUniformData.Color = glm::vec4(0.9f, 0.9f, 0.9f, 1.0f);  // White by default
    m_TestFragUniformData.EyePos = glm::vec4(m_CurrentCameraPos.x, m_CurrentCameraPos.y, m_CurrentCameraPos.z, 1.0f);
    m_TestFragUniformData.LightDir = normalize(gLightDirection);
    m_TestFragUniformData.LightColor = glm::vec4(1.0f, 1.0f, 1.0f, gSpecularExponent);  // White by default;
    UpdateUniformBuffer(pVulkan, &m_TestFragUniform, sizeof(m_TestFragUniformData), &m_TestFragUniformData);

    return true;
}

//-----------------------------------------------------------------------------
bool Application::BuildCmdBuffers()
//-----------------------------------------------------------------------------
{
    Vulkan* pVulkan = GetVulkan();

    LOGI("Creating %d command buffers...", pVulkan->m_SwapchainImageCount);

    for (uint32_t WhichImage = 0; WhichImage < pVulkan->m_SwapchainImageCount; WhichImage++)
    {
        auto& CmdBuffer = m_CommandBuffers[WhichImage];

        // Begin should reset the command buffer, but Reset can be called
        // to make it more explicit.
        if (!CmdBuffer.Reset())
        {
            return false;
        }

        // By calling Begin, CmdBuffer is put into the recording state.
        if (!CmdBuffer.Begin())
        {
            return false;
        }

        // When starting the render pass, we can set clear values.
        VkClearColorValue ClearColor[1] = {};
        ClearColor[0].float32[0] = gClearColor.x;
        ClearColor[0].float32[1] = gClearColor.y;
        ClearColor[0].float32[2] = gClearColor.z;
        ClearColor[0].float32[3] = gClearColor.w;

        VkViewport Viewport = {};
        Viewport.width = (float)pVulkan->m_SurfaceWidth;
        Viewport.height = (float)pVulkan->m_SurfaceHeight;
        Viewport.minDepth = 0.0f;
        Viewport.maxDepth = 1.0f;
        VkRect2D Scissor = {};
        Scissor.offset.x = 0;
        Scissor.offset.y = 0;
        Scissor.extent.width = pVulkan->m_SurfaceWidth;
        Scissor.extent.height = pVulkan->m_SurfaceHeight;

        CmdBuffer.BeginRenderPass(Scissor, 0.0f, 1.0f, ClearColor, 1, true, pVulkan->m_SwapchainRenderPass, true/*swapchain*/, pVulkan->GetSwapchainFramebuffer(WhichImage).m_FrameBuffer, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdSetViewport(CmdBuffer, 0, 1, &Viewport);
        vkCmdSetScissor(CmdBuffer, 0, 1, &Scissor);

        // TODO: Add other mesh renders to the command buffer
        AddMeshRenderToCmdBuffer(CmdBuffer, &m_TestMesh, &m_TestMaterial);

        // Now our render pass has ended.
        CmdBuffer.EndRenderPass();

        // By ending the command buffer, it is put out of record mode.
        if (!CmdBuffer.End())
        {
            return false;
        }
    }   // WhichImage

    return true;
}

//-----------------------------------------------------------------------------
void Application::AddMeshRenderToCmdBuffer(CommandListVulkan& cmdBuffer, Mesh* pMesh, MaterialVk* pMaterial)
//-----------------------------------------------------------------------------
{
    Vulkan* pVulkan = GetVulkan();

    LOGI("AddMeshRenderToCmdBuffer() Entered...");

    // Bind the pipeline for this material
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pMaterial->Pipeline);

    // Bind everything the shader needs
    vkCmdBindDescriptorSets(cmdBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pMaterial->PipelineLayout,
        0,
        1,
        &pMaterial->DescSet,
        0,
        NULL);

    // Bind our mesh vertex buffer
    VkDeviceSize offsets[1] = { 0 };
    assert(pMesh->m_VertexBuffers.size() == 1);
    const VkBuffer vertexBuffer[1]{ pMesh->m_VertexBuffers[0].GetVkBuffer() };
    vkCmdBindVertexBuffers(cmdBuffer,
        VERTEX_BUFFER_BIND_ID,
        1,
        vertexBuffer,
        offsets);

    // Everything is set up, draw the mesh
    LOGI("    pMesh->GetNumVertices() = %d", (int)pMesh->m_NumVertices);
    vkCmdDraw(cmdBuffer, pMesh->m_NumVertices, 1, 0, 0);

}

//-----------------------------------------------------------------------------
bool Application::InitMaterials()
//-----------------------------------------------------------------------------
{
    Vulkan* pVulkan = GetVulkan();

    // ********************************
    // Test MaterialBase
    // ********************************
    LOGI("Setting up Test material...");
    memset(&m_TestMaterial, 0, sizeof(m_TestMaterial));

    // Depth Test/Write
    m_TestMaterial.DepthTestEnable = VK_TRUE;
    m_TestMaterial.DepthWriteEnable = VK_TRUE;

    // Depth Bias
    m_TestMaterial.DepthBiasEnable = VK_FALSE;

    // Culling
    m_TestMaterial.CullMode = VK_CULL_MODE_BACK_BIT;

    // The Shader (Use references since we are not cleaning anything up)
    m_TestMaterial.pShader = &m_TestShader;

    // Textures (Use references since we are not cleaning anything up)
    m_TestMaterial.pColorTexture = &m_TestTexture;
    m_TestMaterial.pNormalTexture = NULL;
    m_TestMaterial.pShadowDepthTexture = NULL;
    m_TestMaterial.pShadowColorTexture = NULL;
    m_TestMaterial.pEnvironmentCubeMap = NULL;
    m_TestMaterial.pIrradianceCubeMap = NULL;
    m_TestMaterial.pReflectTexture = NULL;

    // Constant Buffers
    m_TestMaterial.VertUniformOffset = 0;
    m_TestMaterial.VertUniformLength = sizeof(m_TestVertUniformData);
    m_TestMaterial.pVertUniform = &m_TestVertUniform;

    m_TestMaterial.FragUniformOffset = 0;
    m_TestMaterial.FragUniformLength = sizeof(m_TestFragUniformData);
    m_TestMaterial.pFragUniform = &m_TestFragUniform;

    // Initialze Vulkan objects for this material
    InitOneLayout(&m_TestMaterial);
    InitOneDescriptorSet(&m_TestMaterial);
    // Pipeline for this material will be created later (After renderpass is created)


    return true;
}

//-----------------------------------------------------------------------------
void Application::InitOneLayout(MaterialVk* pMaterial)
//-----------------------------------------------------------------------------
{
    VkResult RetVal = VK_SUCCESS;
    Vulkan* pVulkan = GetVulkan();

    // This layout needs to match one set for vkUpdateDescriptorSets

    // Used as scratch for setting them up below.  Just make large enough to play with
    uint32_t WhichLayout = 0;
    std::array<VkDescriptorSetLayoutBinding, 25> BindingLayouts = {};

    // The only thing that changes on this structure is the number of valid bindings
    VkDescriptorSetLayoutCreateInfo LayoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    LayoutInfo.pNext = NULL;
    LayoutInfo.flags = 0;
    LayoutInfo.bindingCount = 0;    // Changed by each instance below
    LayoutInfo.pBindings = BindingLayouts.data();

    // The only thing that changes on this structure is the pointer to the layout 
    VkPipelineLayoutCreateInfo PipelineLayoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    PipelineLayoutInfo.pNext = NULL;
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

    // Color Texture
    if (pMaterial->pColorTexture)
    {
        BindingLayouts[WhichLayout].binding = SHADER_DIFFUSE_TEXTURE_LOC;
        BindingLayouts[WhichLayout].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        BindingLayouts[WhichLayout].descriptorCount = 1;
        BindingLayouts[WhichLayout].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        WhichLayout++;
    }

    // Normal Texture
    if (pMaterial->pNormalTexture)
    {
        BindingLayouts[WhichLayout].binding = SHADER_NORMAL_TEXTURE_LOC;
        BindingLayouts[WhichLayout].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        BindingLayouts[WhichLayout].descriptorCount = 1;
        BindingLayouts[WhichLayout].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        WhichLayout++;
    }

    // Shadow Depth Texture
    if (pMaterial->pShadowDepthTexture)
    {
        BindingLayouts[WhichLayout].binding = SHADER_SHADOW_DEPTH_TEXTURE_LOC;
        BindingLayouts[WhichLayout].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        BindingLayouts[WhichLayout].descriptorCount = 1;
        BindingLayouts[WhichLayout].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        WhichLayout++;
    }

    // Shadow Color Texture
    if (pMaterial->pShadowColorTexture)
    {
        BindingLayouts[WhichLayout].binding = SHADER_SHADOW_COLOR_TEXTURE_LOC;
        BindingLayouts[WhichLayout].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        BindingLayouts[WhichLayout].descriptorCount = 1;
        BindingLayouts[WhichLayout].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        WhichLayout++;
    }

    // Environment Cubemap Texture
    if (pMaterial->pEnvironmentCubeMap)
    {
        BindingLayouts[WhichLayout].binding = SHADER_ENVIRONMENT_TEXTURE_LOC;
        BindingLayouts[WhichLayout].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        BindingLayouts[WhichLayout].descriptorCount = 1;
        BindingLayouts[WhichLayout].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        WhichLayout++;
    }

    // Irradiance Cubemap Texture
    if (pMaterial->pIrradianceCubeMap)
    {
        BindingLayouts[WhichLayout].binding = SHADER_IRRADIANCE_TEXTURE_LOC;
        BindingLayouts[WhichLayout].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        BindingLayouts[WhichLayout].descriptorCount = 1;
        BindingLayouts[WhichLayout].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        WhichLayout++;
    }

    // Reflection Texture
    if (pMaterial->pReflectTexture)
    {
        BindingLayouts[WhichLayout].binding = SHADER_REFLECT_TEXTURE_LOC;
        BindingLayouts[WhichLayout].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        BindingLayouts[WhichLayout].descriptorCount = 1;
        BindingLayouts[WhichLayout].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        WhichLayout++;
    }

    // How many layouts did we end up with?
    LayoutInfo.bindingCount = WhichLayout;

    // Update the layout info structure and create the layout
    RetVal = vkCreateDescriptorSetLayout(pVulkan->m_VulkanDevice, &LayoutInfo, NULL, &pMaterial->DescLayout);
    if (!CheckVkError("vkCreateDescriptorSetLayout()", RetVal))
    {
        return;
    }

    // Update the other layout structure and create the pipeline layout
    PipelineLayoutInfo.pSetLayouts = &pMaterial->DescLayout;
    RetVal = vkCreatePipelineLayout(pVulkan->m_VulkanDevice, &PipelineLayoutInfo, NULL, &pMaterial->PipelineLayout);
    if (!CheckVkError("vkCreatePipelineLayout()", RetVal))
    {
        return;
    }
}

//-----------------------------------------------------------------------------
void Application::InitOneDescriptorSet(MaterialVk* pMaterial)
//-----------------------------------------------------------------------------
{
    // This layout must match what is in vkCreateDescriptorSetLayout

    VkResult RetVal = VK_SUCCESS;
    Vulkan* pVulkan = GetVulkan();

    // Used as scratch for setting them up below.  Just make large enough to play with
    uint32_t PoolCount = 0;
    std::array<VkDescriptorPoolSize, 10> PoolSize = {};

    // Need uniform buffers for Vert and Frag shaders
    PoolSize[PoolCount].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    PoolSize[PoolCount].descriptorCount = NUM_VULKAN_BUFFERS * 2;    // Need for each buffer (Vert + Frag) * NUM_VULKAN_BUFFERS
    PoolCount++;

    // Need SSBO for vert and frag
    PoolSize[PoolCount].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    PoolSize[PoolCount].descriptorCount = NUM_VULKAN_BUFFERS * 2;    // Need for each buffer (Vert + Frag) * NUM_VULKAN_BUFFERS
    PoolCount++;

    // Need combined image samplers for each possible texture
    PoolSize[PoolCount].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    PoolSize[PoolCount].descriptorCount = NUM_VULKAN_BUFFERS * NUM_SHADER_TEXTURE_LOCATIONS;    // Need for each buffer
    PoolCount++;

    VkDescriptorPoolCreateInfo PoolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    PoolInfo.pNext = NULL;
    PoolInfo.flags = 0;     // VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT to allow them to be returned
    PoolInfo.maxSets = NUM_VULKAN_BUFFERS;  // Since descriptor sets come out of this pool we need more than one
    PoolInfo.poolSizeCount = PoolCount;
    PoolInfo.pPoolSizes = PoolSize.data();

    RetVal = vkCreateDescriptorPool(pVulkan->m_VulkanDevice, &PoolInfo, NULL, &pMaterial->DescPool);
    if (!CheckVkError("vkCreateDescriptorPool()", RetVal))
    {
        return;
    }

    // ****************************************************
    // Allocate the descriptor set
    // ****************************************************
    VkDescriptorSetAllocateInfo DescSetInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    DescSetInfo.pNext = NULL;
    DescSetInfo.descriptorPool = pMaterial->DescPool;
    DescSetInfo.descriptorSetCount = 1;
    DescSetInfo.pSetLayouts = &pMaterial->DescLayout;

    RetVal = vkAllocateDescriptorSets(pVulkan->m_VulkanDevice, &DescSetInfo, &pMaterial->DescSet);
    if (!CheckVkError("vkAllocateDescriptorSets()", RetVal))
    {
        return;
    }

    UpdateOneDescriptorSet(pMaterial);
}

//-----------------------------------------------------------------------------
void Application::UpdateOneDescriptorSet(MaterialVk* pMaterial)
//-----------------------------------------------------------------------------
{
    Vulkan* pVulkan = GetVulkan();

    uint32_t DescCount = 0;

    LOGI("Updating Descriptor Set:");

    std::array<VkDescriptorImageInfo, 25> DescImageInfo = {};

    std::array<VkWriteDescriptorSet, 25> WriteInfo = {};

    // Reset structures used to update descriptor sets
    DescCount = 0;

    // Always a Vert Buffer
    LOGI("    Contains Vertex Buffer");
    WriteInfo[DescCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    WriteInfo[DescCount].pNext = NULL;
    WriteInfo[DescCount].dstSet = pMaterial->DescSet;
    WriteInfo[DescCount].dstBinding = SHADER_VERT_UBO_LOCATION;
    WriteInfo[DescCount].dstArrayElement = 0;
    WriteInfo[DescCount].descriptorCount = 1;
    WriteInfo[DescCount].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;    // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    WriteInfo[DescCount].pImageInfo = NULL;
    WriteInfo[DescCount].pBufferInfo = &pMaterial->pVertUniform->bufferInfo;
    WriteInfo[DescCount].pTexelBufferView = NULL;
    DescCount++;

    // Always a Frag Buffer
    LOGI("    Contains Fragment Buffer");
    WriteInfo[DescCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    WriteInfo[DescCount].pNext = NULL;
    WriteInfo[DescCount].dstSet = pMaterial->DescSet;
    WriteInfo[DescCount].dstBinding = SHADER_FRAG_UBO_LOCATION;
    WriteInfo[DescCount].dstArrayElement = 0;
    WriteInfo[DescCount].descriptorCount = 1;
    WriteInfo[DescCount].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;    // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    WriteInfo[DescCount].pImageInfo = NULL;
    WriteInfo[DescCount].pBufferInfo = &pMaterial->pFragUniform->bufferInfo;
    WriteInfo[DescCount].pTexelBufferView = NULL;
    DescCount++;

    // Color Texture
    if (pMaterial->pColorTexture)
    {
        LOGI("    Contains Diffuse Texture");
        DescImageInfo[DescCount] = pMaterial->pColorTexture->GetVkDescriptorImageInfo();

        WriteInfo[DescCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        WriteInfo[DescCount].pNext = NULL;
        WriteInfo[DescCount].dstSet = pMaterial->DescSet;
        WriteInfo[DescCount].dstBinding = SHADER_DIFFUSE_TEXTURE_LOC;
        WriteInfo[DescCount].dstArrayElement = 0;
        WriteInfo[DescCount].descriptorCount = 1;
        WriteInfo[DescCount].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        WriteInfo[DescCount].pImageInfo = &DescImageInfo[DescCount];
        WriteInfo[DescCount].pBufferInfo = NULL;
        WriteInfo[DescCount].pTexelBufferView = NULL;
        DescCount++;
    }

    // Normal Texture
    if (pMaterial->pNormalTexture)
    {
        LOGI("    Contains Normal Texture");
        DescImageInfo[DescCount] = pMaterial->pNormalTexture->GetVkDescriptorImageInfo();

        WriteInfo[DescCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        WriteInfo[DescCount].pNext = NULL;
        WriteInfo[DescCount].dstSet = pMaterial->DescSet;
        WriteInfo[DescCount].dstBinding = SHADER_NORMAL_TEXTURE_LOC;
        WriteInfo[DescCount].dstArrayElement = 0;
        WriteInfo[DescCount].descriptorCount = 1;
        WriteInfo[DescCount].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        WriteInfo[DescCount].pImageInfo = &DescImageInfo[DescCount];
        WriteInfo[DescCount].pBufferInfo = NULL;
        WriteInfo[DescCount].pTexelBufferView = NULL;
        DescCount++;
    }

    // Shadow Depth Texture
    if (pMaterial->pShadowDepthTexture)
    {
        LOGI("    Contains Shadow Depth Texture");
        DescImageInfo[DescCount] = pMaterial->pShadowDepthTexture->GetVkDescriptorImageInfo();

        WriteInfo[DescCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        WriteInfo[DescCount].pNext = NULL;
        WriteInfo[DescCount].dstSet = pMaterial->DescSet;
        WriteInfo[DescCount].dstBinding = SHADER_SHADOW_DEPTH_TEXTURE_LOC;
        WriteInfo[DescCount].dstArrayElement = 0;
        WriteInfo[DescCount].descriptorCount = 1;
        WriteInfo[DescCount].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        WriteInfo[DescCount].pImageInfo = &DescImageInfo[DescCount];
        WriteInfo[DescCount].pBufferInfo = NULL;
        WriteInfo[DescCount].pTexelBufferView = NULL;
        DescCount++;
    }

    // Shadow Color Texture
    if (pMaterial->pShadowColorTexture)
    {
        LOGI("    Contains Shadow Color Texture");
        DescImageInfo[DescCount] = pMaterial->pShadowColorTexture->GetVkDescriptorImageInfo();

        WriteInfo[DescCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        WriteInfo[DescCount].pNext = NULL;
        WriteInfo[DescCount].dstSet = pMaterial->DescSet;
        WriteInfo[DescCount].dstBinding = SHADER_SHADOW_COLOR_TEXTURE_LOC;
        WriteInfo[DescCount].dstArrayElement = 0;
        WriteInfo[DescCount].descriptorCount = 1;
        WriteInfo[DescCount].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        WriteInfo[DescCount].pImageInfo = &DescImageInfo[DescCount];
        WriteInfo[DescCount].pBufferInfo = NULL;
        WriteInfo[DescCount].pTexelBufferView = NULL;
        DescCount++;
    }

    // Environment Cubemap Texture
    if (pMaterial->pEnvironmentCubeMap)
    {
        LOGI("    Contains Environment Texture");
        DescImageInfo[DescCount] = pMaterial->pEnvironmentCubeMap->GetVkDescriptorImageInfo();

        // DescImageInfo[DescCount].sampler = mTexEnvironment.GetSampler();
        // DescImageInfo[DescCount].imageView = mTexEnvironment.GetView();
        // DescImageInfo[DescCount].imageLayout = mTexEnvironment.GetLayout();

        WriteInfo[DescCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        WriteInfo[DescCount].pNext = NULL;
        WriteInfo[DescCount].dstSet = pMaterial->DescSet;
        WriteInfo[DescCount].dstBinding = SHADER_ENVIRONMENT_TEXTURE_LOC;
        WriteInfo[DescCount].dstArrayElement = 0;
        WriteInfo[DescCount].descriptorCount = 1;
        WriteInfo[DescCount].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        WriteInfo[DescCount].pImageInfo = &DescImageInfo[DescCount];
        WriteInfo[DescCount].pBufferInfo = NULL;
        WriteInfo[DescCount].pTexelBufferView = NULL;
        DescCount++;
    }

    // Irradiance Cubemap Texture
    if (pMaterial->pIrradianceCubeMap)
    {
        LOGI("    Contains Irradiance Texture");
        DescImageInfo[DescCount] = pMaterial->pIrradianceCubeMap->GetVkDescriptorImageInfo();

        WriteInfo[DescCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        WriteInfo[DescCount].pNext = NULL;
        WriteInfo[DescCount].dstSet = pMaterial->DescSet;
        WriteInfo[DescCount].dstBinding = SHADER_IRRADIANCE_TEXTURE_LOC;
        WriteInfo[DescCount].dstArrayElement = 0;
        WriteInfo[DescCount].descriptorCount = 1;
        WriteInfo[DescCount].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        WriteInfo[DescCount].pImageInfo = &DescImageInfo[DescCount];
        WriteInfo[DescCount].pBufferInfo = NULL;
        WriteInfo[DescCount].pTexelBufferView = NULL;
        DescCount++;
    }

    // Reflection Texture
    if (pMaterial->pReflectTexture)
    {
        LOGI("    Contains Reflect Texture");
        DescImageInfo[DescCount] = pMaterial->pReflectTexture->GetVkDescriptorImageInfo();

        WriteInfo[DescCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        WriteInfo[DescCount].pNext = NULL;
        WriteInfo[DescCount].dstSet = pMaterial->DescSet;
        WriteInfo[DescCount].dstBinding = SHADER_REFLECT_TEXTURE_LOC;
        WriteInfo[DescCount].dstArrayElement = 0;
        WriteInfo[DescCount].descriptorCount = 1;
        WriteInfo[DescCount].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        WriteInfo[DescCount].pImageInfo = &DescImageInfo[DescCount];
        WriteInfo[DescCount].pBufferInfo = NULL;
        WriteInfo[DescCount].pTexelBufferView = NULL;
        DescCount++;
    }

    LOGI("Updating Descriptor Set with %d objects", DescCount);
    vkUpdateDescriptorSets(pVulkan->m_VulkanDevice, DescCount, WriteInfo.data(), 0, NULL);

    LOGI("Descriptor Set Updated!");
}

//-----------------------------------------------------------------------------
bool Application::InitRenderPass()
//-----------------------------------------------------------------------------
{
    VkResult RetVal = VK_SUCCESS;
    Vulkan* pVulkan = GetVulkan();

    // The renderpass defines the attachments to the framebuffer object that gets
    // used in the pipelines. We have two attachments, the colour buffer, and the
    // depth buffer. The operations and layouts are set to defaults for this type
    // of attachment.
    VkAttachmentDescription attachmentDescriptions[2] = {};
    attachmentDescriptions[0].flags = 0;
    attachmentDescriptions[0].format = TextureFormatToVk( pVulkan->m_SurfaceFormat );
    attachmentDescriptions[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachmentDescriptions[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachmentDescriptions[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachmentDescriptions[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachmentDescriptions[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachmentDescriptions[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    attachmentDescriptions[1].flags = 0;
    attachmentDescriptions[1].format = TextureFormatToVk( pVulkan->m_SwapchainDepth.format );
    attachmentDescriptions[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachmentDescriptions[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachmentDescriptions[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachmentDescriptions[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescriptions[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachmentDescriptions[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachmentDescriptions[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // We have references to the attachment offsets, stating the layout type.
    VkAttachmentReference colorReference = {};
    colorReference.attachment = 0;
    colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthReference = {};
    depthReference.attachment = 1;
    depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // There can be multiple subpasses in a renderpass, but this example has only one.
    // We set the color and depth references at the grahics bind point in the pipeline.
    VkSubpassDescription subpassDescription = {};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.flags = 0;
    subpassDescription.inputAttachmentCount = 0;
    subpassDescription.pInputAttachments = nullptr;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorReference;
    subpassDescription.pResolveAttachments = nullptr;
    subpassDescription.pDepthStencilAttachment = &depthReference;
    subpassDescription.preserveAttachmentCount = 0;
    subpassDescription.pPreserveAttachments = nullptr;

    // Dependencies
    VkSubpassDependency dependencies[2];
    // dependencies[0] is an acquire dependency
    // dependencies[1] is a present dependency
    // We use subpass dependencies to define the color image layout transitions rather than
    // explicitly do them in the command buffer, as it is more efficient to do it this way.

    // Before we can use the back buffer from the swapchain, we must change the
    // image layout from the PRESENT mode to the COLOR_ATTACHMENT mode.
    dependencies[0] = {};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = 0;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // After writing to the back buffer of the swapchain, we need to change the
    // image layout from the COLOR_ATTACHMENT mode to the PRESENT mode which
    // is optimal for sending to the screen for users to see the completed rendering.
    dependencies[1] = {};
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = 0;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // The renderpass itself is created with the number of subpasses, and the
    // list of attachments which those subpasses can reference.
    VkRenderPassCreateInfo renderPassCreateInfo = {};
    renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCreateInfo.pNext = nullptr;
    renderPassCreateInfo.attachmentCount = 2;
    renderPassCreateInfo.pAttachments = attachmentDescriptions;
    renderPassCreateInfo.subpassCount = 1;
    renderPassCreateInfo.pSubpasses = &subpassDescription;
    renderPassCreateInfo.dependencyCount = 2;
    renderPassCreateInfo.pDependencies = dependencies;

    RetVal = vkCreateRenderPass(pVulkan->m_VulkanDevice, &renderPassCreateInfo, nullptr, &m_RenderPass);
    if (!CheckVkError("vkCreateRenderPass()", RetVal))
    {
        return false;
    }

    return true;

}

//-----------------------------------------------------------------------------
bool Application::InitPipelines()
//-----------------------------------------------------------------------------
{
    Vulkan* pVulkan = GetVulkan();

    VkPipelineVertexInputStateCreateInfo visci = {};

    // Common to all pipelines
    // State for rasterization, such as polygon fill mode is defined.
    VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.depthClampEnable = VK_FALSE;
    rs.rasterizerDiscardEnable = VK_FALSE;
    rs.depthBiasEnable = VK_FALSE;
    rs.lineWidth = 1.0f;

    // For the cube, we don't write or check depth
    VkPipelineDepthStencilStateCreateInfo ds = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    ds.depthBoundsTestEnable = VK_FALSE;
    ds.back.failOp = VK_STENCIL_OP_KEEP;
    ds.back.passOp = VK_STENCIL_OP_KEEP;
    ds.back.compareOp = VK_COMPARE_OP_ALWAYS;
    ds.stencilTestEnable = VK_FALSE;
    ds.front = ds.back;


    // ********************************
    // Test Object
    // ********************************
    rs.cullMode = m_TestMaterial.CullMode;
    rs.depthBiasEnable = m_TestMaterial.DepthBiasEnable;
    ds.depthTestEnable = m_TestMaterial.DepthTestEnable;
    ds.depthWriteEnable = m_TestMaterial.DepthWriteEnable;

    LOGI("Creating Test Object PipelineState and Pipeline...");
    visci = m_TestMesh.m_VertexBuffers[0].CreatePipelineState();
    return GetVulkan()->CreatePipeline(VK_NULL_HANDLE,
                                    &visci,
                                    m_TestMaterial.PipelineLayout,
                                    Vulkan::RenderContext(RenderPass(pVulkan->m_VulkanDevice, m_RenderPass), Framebuffer<Vulkan>(), ""),
                                    &rs,
                                    &ds,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    {},
                                    nullptr,
                                    nullptr,
                                    VK_NULL_HANDLE,//task shader
                                    VK_NULL_HANDLE,//mesh shader
                                    m_TestShader.VertShaderModule.GetVkShaderModule(),
                                    m_TestShader.FragShaderModule.GetVkShaderModule(),
                                    nullptr,
                                    false,
                                    VK_NULL_HANDLE,
                                    &m_TestMaterial.Pipeline);
}

//-----------------------------------------------------------------------------
bool Application::InitCommandBuffers()
//-----------------------------------------------------------------------------
{
    for (auto& CommandBuffer : m_CommandBuffers)
    {
        if (!CommandBuffer.Initialize(GetVulkan()))
            return false;
    }
    return true;
}
