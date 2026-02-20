//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "application.hpp"
#include "vulkan/extensionHelpers.hpp"
#include "main/applicationEntrypoint.hpp"
#include "camera/cameraController.hpp"
#include "camera/cameraControllerTouch.hpp"
#include "camera/cameraData.hpp"
#include "camera/cameraGltfLoader.hpp"
#include "gui/imguiVulkan.hpp"
#include "material/drawable.hpp"
#include "material/vulkan/shaderModule.hpp"
#include "material/shaderManagerT.hpp"
#include "material/materialManager.hpp"
#include "material/vulkan/specializationConstantsLayout.hpp"
#include "mesh/meshHelper.hpp"
#include "mesh/meshLoader.hpp"
#include "system/math_common.hpp"
#include "texture/textureManager.hpp"
#include "vulkan/extensionLib.hpp"
#include "material/vulkan/computable.hpp"
#include "material/vulkan/drawable.hpp"
#include "imgui.h"

#include <random>
#include <iostream>
#include <filesystem>
#include <fstream>

namespace
{
    static constexpr std::array<const char*, NUM_RENDER_PASSES> sRenderPassNames = { "RP_SCENE", "RP_HUD", "RP_BLIT" };

    glm::vec3 gCameraStartPos = glm::vec3(0.0f, 3.5f, 0.0f);
    glm::vec3 gCameraStartRot = glm::vec3(0.0f, 0.0f, 0.0f);

    float   gFOV = PI_DIV_4;
    float   gNearPlane = 1.0f;
    float   gFarPlane = 1800.0f;
    float   gNormalAmount = 0.3f;
    float   gNormalMirrorReflectAmount = 0.05f;

    const char* gSceneAssetGraphModel = "PipelineCache.bin"; // What we expect to load the model via VK_QCOM_data_graph_model
    const char* gSceneAssetModel      = "SteamPunkSauna.gltf";

    static uint32_t FindMemoryType(VkPhysicalDevice& physicalDevice, uint32_t type_bits, VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        {
            if ((memProperties.memoryTypes[i].propertyFlags & properties) == properties && type_bits & (1 << i))
                return i;
        }

        throw std::runtime_error("Failed to find suitable memory type!");
    };
}

FrameworkApplicationBase* Application_ConstructApplication()
{
    return new Application();
}

Application::Application() : ApplicationHelperBase()
{
}

Application::~Application()
{
}

//-----------------------------------------------------------------------------
void Application::PreInitializeSetVulkanConfiguration(Vulkan::AppConfiguration& config)
//-----------------------------------------------------------------------------
{
    ApplicationHelperBase::PreInitializeSetVulkanConfiguration(config);
    config.RequiredExtension<ExtensionLib::Ext_VK_KHR_synchronization2>();
    config.RequiredExtension<ExtensionLib::Ext_VK_KHR_create_renderpass2>();
    config.RequiredExtension<ExtensionLib::Ext_VK_KHR_get_physical_device_properties2>();

    config.OptionalExtension<ExtensionLib::Ext_VK_ARM_tensors>();
    config.OptionalExtension<ExtensionLib::Ext_VK_ARM_data_graph>();
    config.OptionalExtension<ExtensionLib::Ext_VK_QCOM_data_graph_model>();
}

//-----------------------------------------------------------------------------
bool Application::Initialize(uintptr_t windowHandle, uintptr_t hInstance)
//-----------------------------------------------------------------------------
{
    if (!ApplicationHelperBase::Initialize( windowHandle, hInstance ))
    {
        return false;
    }

    m_IsGraphPipelinesSupported &= GetVulkan()->HasLoadedVulkanDeviceExtension(VK_ARM_TENSORS_EXTENSION_NAME)
        && GetVulkan()->HasLoadedVulkanDeviceExtension(VK_ARM_DATA_GRAPH_EXTENSION_NAME);

    // If Ext_VK_ARM_data_graph->AvailableFeatures.dataGraph is supported, force graph pipeline support here in case the
    // driver has support but it isn't exposed
#if defined(OS_ANDROID)
    {
        auto* Ext_VK_ARM_tensors = static_cast<ExtensionLib::Ext_VK_ARM_tensors*>(GetVulkan()->m_DeviceExtensions.GetExtension(VK_ARM_TENSORS_EXTENSION_NAME));
        auto* Ext_VK_ARM_data_graph = static_cast<ExtensionLib::Ext_VK_ARM_data_graph*>(GetVulkan()->m_DeviceExtensions.GetExtension(VK_ARM_DATA_GRAPH_EXTENSION_NAME));
        if (Ext_VK_ARM_tensors && Ext_VK_ARM_data_graph && Ext_VK_ARM_data_graph->AvailableFeatures.dataGraph)
        {
            m_IsGraphPipelinesSupported = true;
        }
    }
#endif

    // If for some reason we are able to enable the extension, but no graph queue was detected, disable
    // graph pipelines here (this check probably isn't necessary)
    if (!GetVulkan()->IsDataGraphQueueSupported())
    {
        m_IsGraphPipelinesSupported = false;
    }

    // NOTE: You should configure these according to what the model expects.
    m_RenderResolution   = glm::ivec2(960, 540);
    m_UpscaledResolution = glm::ivec2(1920, 1080); // It just happens that this aligns with the default values of gSurfaceWidth and gSurfaceHeight.

    if (!InitializeCamera())
    {
        return false;
    }
    
    if (!InitializeLights())
    {
        return false;
    }

    if (!LoadShaders())
    {
        return false;
    }
     
    if (!CreateTensors())
    {
        return false;
    }
    
    if (!CreateGraphPipeline())
    {
        return false;
    }

    if (!InitUniforms())
    {
        return false;
    }

    if (!CreateRenderTargets())
    {
        return false;
    }

    if (!InitAllRenderPasses())
    {
        return false;
    }

    if (!InitGui(windowHandle))
    {
        return false;
    }

    if (!LoadMeshObjects())
    {
        return false;
    }

    if (!InitCommandBuffers())
    {
        return false;
    }

    if (!InitLocalSemaphores())
    {
        return false;
    }

    if (!BuildCmdBuffers())
    {
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
void Application::Destroy()
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();

    // Uniform Buffers
    ReleaseUniformBuffer(pVulkan, &m_ObjectVertUniform);
    ReleaseUniformBuffer(pVulkan, &m_LightUniform);
    ReleaseUniformBuffer(pVulkan, &m_BlitFragUniform);
     
    for (auto& [hash, objectUniform] : m_ObjectFragUniforms)
    {
        ReleaseUniformBuffer(pVulkan, &objectUniform.objectFragUniform);
    }

    // Cmd buffers
    for (int whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        for (auto& cmdBuffer : m_RenderPassData[whichPass].PassCmdBuffer)
        {
            cmdBuffer.Release();
        }

        for (auto& cmdBuffer : m_RenderPassData[whichPass].ObjectsCmdBuffer)
        {
            cmdBuffer.Release();
        }

        m_RenderPassData[whichPass].RenderTarget.Release();
    }

    // Render passes / Semaphores
    for (int whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        m_RenderPassData[whichPass].RenderContext.clear();
        vkDestroySemaphore(pVulkan->m_VulkanDevice, m_RenderPassData[whichPass].PassCompleteSemaphore, nullptr);
    }

    // Drawables
    m_SceneDrawables.clear();
    m_BlitQuadDrawable.reset();

    // Internal
    m_ShaderManager.reset();
    m_MaterialManager.reset();
    m_CameraController.reset();
    m_AssetManager.reset();

    ApplicationHelperBase::Destroy();
}

//-----------------------------------------------------------------------------
bool Application::InitializeLights()
//-----------------------------------------------------------------------------
{
    m_LightUniformData.SpotLights_pos[0] = glm::vec4(-6.900000f, 32.299999f, -1.900000f, 1.0f);
    m_LightUniformData.SpotLights_pos[1] = glm::vec4(3.300000f, 26.900000f, 7.600000f, 1.0f);
    m_LightUniformData.SpotLights_pos[2] = glm::vec4(12.100000f, 41.400002f, -2.800000f, 1.0f);
    m_LightUniformData.SpotLights_pos[3] = glm::vec4(-5.400000f, 18.500000f, 28.500000f, 1.0f);

    m_LightUniformData.SpotLights_dir[0] = glm::vec4(-0.534696f, -0.834525f, 0.132924f, 0.0f);
    m_LightUniformData.SpotLights_dir[1] = glm::vec4(0.000692f, -0.197335f, 0.980336f, 0.0f);
    m_LightUniformData.SpotLights_dir[2] = glm::vec4(0.985090f, -0.172016f, 0.003000f, 0.0f);
    m_LightUniformData.SpotLights_dir[3] = glm::vec4(0.674125f, -0.295055f, -0.677125f, 0.0f);

    m_LightUniformData.SpotLights_color[0] = glm::vec4(1.000000f, 1.000000f, 1.000000f, 3.000000f);
    m_LightUniformData.SpotLights_color[1] = glm::vec4(1.000000f, 1.000000f, 1.000000f, 3.500000f);
    m_LightUniformData.SpotLights_color[2] = glm::vec4(1.000000f, 1.000000f, 1.000000f, 2.000000f);
    m_LightUniformData.SpotLights_color[3] = glm::vec4(1.000000f, 1.000000f, 1.000000f, 2.800000f);

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitializeCamera()
//-----------------------------------------------------------------------------
{
    LOGI("******************************");
    LOGI("Initializing Camera...");
    LOGI("******************************");

    m_Camera.SetPosition(gCameraStartPos, glm::quat(gCameraStartRot * TO_RADIANS));
    m_Camera.SetAspect(float(gRenderWidth) / float(gRenderHeight));
    m_Camera.SetFov(gFOV);
    m_Camera.SetClipPlanes(gNearPlane, gFarPlane);

    // Camera Controller //

#if defined(OS_ANDROID)
    typedef CameraControllerTouch           tCameraController;
#else
    typedef CameraController                tCameraController;
#endif

    auto cameraController = std::make_unique<tCameraController>();
    if (!cameraController->Initialize(gRenderWidth, gRenderHeight))
    {
        return false;
    }

    m_CameraController = std::move(cameraController);

    return true;
}

//-----------------------------------------------------------------------------
bool Application::CreateTensors()
//-----------------------------------------------------------------------------
{
    if (!m_IsGraphPipelinesSupported)
    {
        LOGI("Not creating Tensors as base extension isn't supported");
        return true;
    }

    auto& vulkan = *GetVulkan();

    LOGI("Creating Tensors...");

    const int64_t componentsPerPixel = 3; // R8G8B8_UNORM and Model is RGB

    m_InputTensor.strides            = { componentsPerPixel * m_RenderResolution.x, componentsPerPixel, 1 };
    m_InputTensor.dimensions         = { m_RenderResolution.y, m_RenderResolution.x, componentsPerPixel };
    m_InputTensor.port_binding_index = m_QNNInputPortBinding;

    m_OutputTensor.strides            = { componentsPerPixel * m_UpscaledResolution.x, componentsPerPixel, 1 };
    m_OutputTensor.dimensions         = { m_UpscaledResolution.y, m_UpscaledResolution.x, componentsPerPixel };
    m_OutputTensor.port_binding_index = m_QNNOutputPortBinding;

    if (!m_tensor_resources.Initialize(
        vulkan.m_VulkanDevice,
        vulkan.m_VulkanGpu,
        vulkan.m_VulkanDataGraphProcessingEngine,
        m_InputTensor,
        m_OutputTensor,
        m_QNNMaxPortIndex))
    {
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::CreateGraphPipeline()
//-----------------------------------------------------------------------------
{
    if (!m_IsGraphPipelinesSupported)
    {
        LOGI("Not creating Graph Pipeline as base extension isn't supported");
        return true;
    }

    auto& vulkan = *GetVulkan();
    
    LOGI("Loading file model from disk...");

    std::vector<unsigned char> modelData;
    {
        const auto sceneAssetGraphModel = std::filesystem::path(MISC_DESTINATION_PATH).append(gSceneAssetGraphModel).string();
        if (!m_AssetManager->LoadFileIntoMemory(sceneAssetGraphModel, modelData))
        {
            LOGE("Failed to load Model file, disabling the Graph Pipelines extension");
            m_IsGraphPipelinesSupported = false;
            return true;
        }
    }

    LOGI("Validating model cache blob...");

    uint32_t cache_version = 0;
    if (!m_QCOM_data_graph_model.ValidateModelCacheBlob(modelData, cache_version))
    {
        return false;
    }

    LOGI("QCOM data-graph cache validated. CacheVersion=%u", cache_version);

    LOGI("Creating Pipeline Cache from Model...");

    if(!m_data_graph_pipeline.CreatePipelineCacheFromBlob(
        vulkan.m_VulkanDevice,
        modelData,
        m_GraphPipelineInstance.pipelineCache))
    { 
        return false;
    }

    LOGI("Creating Graph Pipeline Layout...");
    
    if (!m_data_graph_pipeline.CreatePipelineLayout(
        vulkan.m_VulkanDevice,
        m_tensor_resources.GetResources().tensor_descriptor_set_layout,
        m_GraphPipelineInstance.pipelineLayout))
    {
        return false;
    }

    LOGI("Creating Graph Pipeline...");

    std::array< VkDataGraphPipelineResourceInfoARM, 2> resourceInfos;

    resourceInfos[0].sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM;
    resourceInfos[0].binding = m_QNNInputPortBinding; // Same as the input tensor
    resourceInfos[0].pNext = &m_InputTensor.tensor_description;

    resourceInfos[1].sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM;
    resourceInfos[1].binding = m_QNNOutputPortBinding; // Same as the output tensor
    resourceInfos[1].pNext = &m_OutputTensor.tensor_description;

    ////////////////////
    // IMPORTANT NOTE // These values should be read from the file identifier!!!
    ////////////////////

    uint32_t graphId = 0;
    uint32_t qnnGraphIdSize = sizeof(uint32_t);
    uint8_t  qnnGraphId[32];
    std::memcpy(qnnGraphId, &graphId, qnnGraphIdSize);

    if (!m_data_graph_pipeline.CreateGraphPipelineArmIdentifierPath(
        vulkan.m_VulkanDevice,
        vulkan.m_VulkanDataGraphProcessingEngine,
        m_GraphPipelineInstance.pipelineLayout,
        m_GraphPipelineInstance.pipelineCache,
        resourceInfos.data(),
        static_cast<uint32_t>(resourceInfos.size()),
        qnnGraphId,
        qnnGraphIdSize,
        m_GraphPipelineInstance.graphPipeline))
    {
        return false;
    }

    LOGI("Creating Graph Pipeline Session...");

    if (!m_data_graph_pipeline.CreateSession(
        vulkan.m_VulkanDevice,
        m_GraphPipelineInstance.graphPipeline,
        m_GraphPipelineInstance.graphSession))
    {
        return false;
    }

    LOGI("Binding Graph Session Memory...");

    if (!m_data_graph_pipeline.AllocateAndBindSessionMemory(
        vulkan.m_VulkanDevice,
        vulkan.m_VulkanGpu,
        m_GraphPipelineInstance.graphSession, m_GraphPipelineInstance.sessionMemory))
    {
        return false;
    }

    LOGI("Graph Pipeline Created!");

    return true;
}

//-----------------------------------------------------------------------------
void Application::CopyImageToTensor(
    CommandListVulkan&         cmdList,
    const TextureVulkan&       srcImage,
    VkImageLayout              currentLayout, 
    const Ml::GraphPipelineTensor& tensorBinding)
//-----------------------------------------------------------------------------
{
    // Transition image -> TRANSFER_SRC
    VkImageMemoryBarrier2KHR toTransfer = {};
    toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR;
    toTransfer.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR;
    toTransfer.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT_KHR | VK_ACCESS_2_MEMORY_READ_BIT_KHR;
    toTransfer.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR;
    toTransfer.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT_KHR;
    toTransfer.oldLayout = currentLayout;
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.image = srcImage.GetVkImage();
    toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransfer.subresourceRange.baseMipLevel = 0;
    toTransfer.subresourceRange.levelCount = 1;
    toTransfer.subresourceRange.baseArrayLayer = 0;
    toTransfer.subresourceRange.layerCount = 1;

    VkDependencyInfoKHR dep = {};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &toTransfer;

    vkCmdPipelineBarrier2KHR(cmdList.m_VkCommandBuffer, &dep);

    // Copy image -> buffer
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { srcImage.Width, srcImage.Height, 1 };

    vkCmdCopyImageToBuffer(
        cmdList.m_VkCommandBuffer,
        srcImage.GetVkImage(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        tensorBinding.aliased_buffer,
        1,
        &region);

    // Transition image back to original layout for future reads (typically SHADER_READ_ONLY)
    VkImageMemoryBarrier2KHR fromTransfer = {};
    fromTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR;
    fromTransfer.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR;
    fromTransfer.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT_KHR;
    fromTransfer.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR;
    fromTransfer.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT_KHR | VK_ACCESS_2_MEMORY_WRITE_BIT_KHR;
    fromTransfer.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    fromTransfer.newLayout = currentLayout;
    fromTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    fromTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    fromTransfer.image = srcImage.GetVkImage();
    fromTransfer.subresourceRange = toTransfer.subresourceRange;

    dep.pImageMemoryBarriers = &fromTransfer;
    vkCmdPipelineBarrier2KHR(cmdList.m_VkCommandBuffer, &dep);
}

//-----------------------------------------------------------------------------
void Application::CopyTensorToImage(
    CommandListVulkan&         cmdList,
    const TextureVulkan&       dstImage,
    VkImageLayout              currentLayout, 
    const Ml::GraphPipelineTensor& tensorBinding)
//-----------------------------------------------------------------------------
{
    // Transition image -> TRANSFER_DST
    VkImageMemoryBarrier2KHR toTransfer = {};
    toTransfer.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR;
    toTransfer.srcStageMask        = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR;
    toTransfer.srcAccessMask       = VK_ACCESS_2_MEMORY_WRITE_BIT_KHR | VK_ACCESS_2_MEMORY_READ_BIT_KHR;
    toTransfer.dstStageMask        = VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR;
    toTransfer.dstAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR;
    toTransfer.oldLayout           = currentLayout;
    toTransfer.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.image               = dstImage.GetVkImage();
    toTransfer.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransfer.subresourceRange.baseMipLevel   = 0;
    toTransfer.subresourceRange.levelCount     = 1;
    toTransfer.subresourceRange.baseArrayLayer = 0;
    toTransfer.subresourceRange.layerCount     = 1;

    VkDependencyInfoKHR dep = {};
    dep.sType                  = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &toTransfer;

    vkCmdPipelineBarrier2KHR(cmdList.m_VkCommandBuffer, &dep);

    // Copy buffer -> image
    VkBufferImageCopy region = {};
    region.bufferOffset                    = 0;
    region.bufferRowLength                 = 0;
    region.bufferImageHeight               = 0;
    region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel       = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount     = 1;
    region.imageOffset                     = { 0, 0, 0 };
    region.imageExtent                     = { dstImage.Width, dstImage.Height, 1 };

    vkCmdCopyBufferToImage(
        cmdList.m_VkCommandBuffer,
        tensorBinding.aliased_buffer,
        dstImage.GetVkImage(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region);

    // Transition image back (shader read for blit sampling)
    VkImageMemoryBarrier2KHR fromTransfer = {};
    fromTransfer.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR;
    fromTransfer.srcStageMask        = VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR;
    fromTransfer.srcAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR;
    fromTransfer.dstStageMask        = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR;
    fromTransfer.dstAccessMask       = VK_ACCESS_2_MEMORY_READ_BIT_KHR | VK_ACCESS_2_MEMORY_WRITE_BIT_KHR;
    fromTransfer.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    fromTransfer.newLayout           = currentLayout;
    fromTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    fromTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    fromTransfer.image               = dstImage.GetVkImage();
    fromTransfer.subresourceRange    = toTransfer.subresourceRange;

    dep.pImageMemoryBarriers = &fromTransfer;
    vkCmdPipelineBarrier2KHR(cmdList.m_VkCommandBuffer, &dep);
}

//-----------------------------------------------------------------------------
void Application::CopyImageToImageBlit(
    CommandListVulkan&   cmdList,
    const TextureVulkan& srcImage,
    VkImageLayout        srcLayout,
    const TextureVulkan& dstImage,
    VkImageLayout        dstFinalLayout)
//-----------------------------------------------------------------------------
{
    const auto& synchronization2_extension = GetVulkan()->GetExtension<ExtensionLib::Ext_VK_KHR_synchronization2>();
    assert(synchronization2_extension != nullptr);

    VkImageMemoryBarrier2 dstBarrier = 
    {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = dstImage.GetVkImage(),
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };

    VkDependencyInfo depInfoDst = 
    {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &dstBarrier
    };

    vkCmdPipelineBarrier2KHR(cmdList.m_VkCommandBuffer, &depInfoDst);

    // Blit image
    VkImageBlit blitRegion = {};
    blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.srcSubresource.mipLevel = 0;
    blitRegion.srcSubresource.baseArrayLayer = 0;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcOffsets[0] = { 0, 0, 0 };
    blitRegion.srcOffsets[1] = { static_cast<int32_t>(srcImage.Width), static_cast<int32_t>(srcImage.Height), 1 };

    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.mipLevel = 0;
    blitRegion.dstSubresource.baseArrayLayer = 0;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[0] = { 0, 0, 0 };
    blitRegion.dstOffsets[1] = { static_cast<int32_t>(dstImage.Width), static_cast<int32_t>(dstImage.Height), 1 };

    vkCmdBlitImage(
        cmdList.m_VkCommandBuffer,
        srcImage.GetVkImage(),
        srcLayout,
        dstImage.GetVkImage(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &blitRegion,
        VK_FILTER_LINEAR
    );

    // Transition destination image to final layout
    dstBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    dstBarrier.newLayout = dstFinalLayout;
    dstBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    dstBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    dstBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    dstBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

    vkCmdPipelineBarrier2KHR(cmdList.m_VkCommandBuffer, &depInfoDst);
}


//-----------------------------------------------------------------------------
bool Application::LoadShaders()
//-----------------------------------------------------------------------------
{
    m_ShaderManager = std::make_unique<ShaderManager>(*GetVulkan());
    m_ShaderManager->RegisterRenderPassNames(sRenderPassNames);

    m_MaterialManager = std::make_unique<MaterialManager>(*GetVulkan());

    LOGI("******************************");
    LOGI("Loading Shaders...");
    LOGI("******************************");

    typedef std::pair<std::string, std::string> tIdAndFilename;
    for (const tIdAndFilename& i :
            { tIdAndFilename { "Blit",             "Blit.json" },
              tIdAndFilename { "SceneOpaque",      "SceneOpaque.json" },
              tIdAndFilename { "SceneTransparent", "SceneTransparent.json" }
            })
    {
        if (!m_ShaderManager->AddShader(*m_AssetManager, i.first, i.second, SHADER_DESTINATION_PATH))
        {
            LOGE("Error Loading shader %s from %s", i.first.c_str(), i.second.c_str());
            LOGI("Please verify if you have all required assets on the sample media folder");
            LOGI("If you are running on Android, don't forget to run the `02_CopyMediaToDevice.bat` script to copy all media files into the device memory");
            return false;
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::CreateRenderTargets()
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();

    LOGI("**************************");
    LOGI("Creating Render Targets...");
    LOGI("**************************");

    TextureFormat vkDesiredDepthFormat = pVulkan->GetBestSurfaceDepthFormat();
    TextureFormat desiredDepthFormat = vkDesiredDepthFormat;

    // Note: R8G8B8_UNORM is used here since that's what the upscaling model expects, if no upscaling will ever
    // be performed, just default to the usual R8G8B8A8_SRGB format
    // It's likely R8G8B8_UNORM isn't supported where graph pipelines aren't also supported
    const TextureFormat MainColorType[]       = { m_IsGraphPipelinesSupported ? TextureFormat::R8G8B8_UNORM : TextureFormat::R8G8B8A8_SRGB };
    const TEXTURE_TYPE  MainTextureType[]     = { TEXTURE_TYPE::TT_RENDER_TARGET_TRANSFERSRC }; // Needed for tensor copy from operation.
    const TEXTURE_TYPE  UpscaledTextureType   = TEXTURE_TYPE::TT_CPU_UPDATE;                    // Needed for tensor copy to operation.
    const TextureFormat HudColorType[]        = { TextureFormat::R8G8B8A8_SRGB };
    const Msaa          MSAA[]                = { Msaa::Samples1 };

    if (!m_RenderPassData[RP_SCENE].RenderTarget.Initialize(
        pVulkan, 
        m_RenderResolution.x, 
        m_RenderResolution.y, 
        MainColorType, 
        desiredDepthFormat, 
        "Scene RT", 
        MainTextureType, 
        MSAA))
    {
        LOGE("Unable to create scene render target");
        return false;
    }

    {
        CreateTexObjectInfo createInfo{};
        createInfo.uiWidth = m_UpscaledResolution.x;
        createInfo.uiHeight = m_UpscaledResolution.y;

        createInfo.Format = MainColorType[0];
        createInfo.TexType = UpscaledTextureType;
        createInfo.pName = "Upscaled RT";
        createInfo.Msaa = Msaa::Samples1;
        createInfo.FilterMode = SamplerFilter::Linear;

        m_UpscaledImageResult = CreateTextureObject(*GetVulkan(), createInfo);
    }

    // Notice no depth on the HUD RT
    if (!m_RenderPassData[RP_HUD].RenderTarget.Initialize(
        pVulkan, 
        gSurfaceWidth, 
        gSurfaceHeight, 
        HudColorType, 
        TextureFormat::UNDEFINED, 
        "HUD RT"))
    {
        LOGE("Unable to create hud render target");
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitUniforms()
//-----------------------------------------------------------------------------
{
    LOGI("******************************");
    LOGI("Initializing Uniforms...");
    LOGI("******************************");

    Vulkan* const pVulkan = GetVulkan();

    if (!CreateUniformBuffer(pVulkan, m_ObjectVertUniform))
    {
        return false;
    }

    if (!CreateUniformBuffer(pVulkan, m_LightUniform))
    {
        return false;
    }
    
    if (!CreateUniformBuffer(pVulkan, m_BlitFragUniform))
    {
        return false;
    }
    
    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitAllRenderPasses()
//-----------------------------------------------------------------------------
{
    Vulkan& vulkan = *GetVulkan();

    //                                             ColorInputUsage |               ClearDepthRenderPass | ColorOutputUsage |                     DepthOutputUsage |              ClearColor
    m_RenderPassData[RP_SCENE].RenderPassSetup = { RenderPassInputUsage::Clear,    true,                  RenderPassOutputUsage::StoreReadOnly,  RenderPassOutputUsage::Store,   {}};
    m_RenderPassData[RP_HUD].RenderPassSetup   = { RenderPassInputUsage::Clear,    false,                 RenderPassOutputUsage::StoreReadOnly,  RenderPassOutputUsage::Discard, {}};
    m_RenderPassData[RP_BLIT].RenderPassSetup  = { RenderPassInputUsage::DontCare, true,                  RenderPassOutputUsage::Present,        RenderPassOutputUsage::Discard, {}};

    TextureFormat surfaceFormat = vulkan.m_SurfaceFormat;
    auto swapChainColorFormat   = std::span<const TextureFormat>({ &surfaceFormat, 1 });
    auto swapChainDepthFormat   = vulkan.m_SwapchainDepth.format;

    LOGI("******************************");
    LOGI("Initializing Render Passes %d - %d... ", static_cast<int>(swapChainColorFormat[0]), static_cast<int>(vulkan.m_SurfaceColorSpace));
    LOGI("******************************");

    for (uint32_t whichPass = 0; whichPass < RP_BLIT; whichPass++)
    {
        std::span<const TextureFormat> colorFormats = m_RenderPassData[whichPass].RenderTarget.m_pLayerFormats;
        TextureFormat                  depthFormat  = m_RenderPassData[whichPass].RenderTarget.m_DepthFormat;

        const auto& passSetup = m_RenderPassData[whichPass].RenderPassSetup;
        auto& passData = m_RenderPassData[whichPass];
        
        RenderPass renderPass;
        if (!vulkan.CreateRenderPass(
            { colorFormats },
            depthFormat,
            Msaa::Samples1,
            passSetup.ColorInputUsage,
            passSetup.ColorOutputUsage,
            passSetup.ClearDepthRenderPass,
            passSetup.DepthOutputUsage,
            renderPass))
        {
            return false;
        }
        Framebuffer<Vulkan> framebuffer;
        framebuffer.Initialize( vulkan,
                                renderPass,
                                passData.RenderTarget.m_ColorAttachments,
                                &passData.RenderTarget.m_DepthAttachment,
                                sRenderPassNames[whichPass] );
        passData.RenderContext.push_back({std::move(renderPass), {}/*pipeline*/, std::move(framebuffer), sRenderPassNames[whichPass]});
    }
    for (auto whichBuffer = 0; whichBuffer < vulkan.GetSwapchainBufferCount(); ++whichBuffer)
    {
        m_RenderPassData[RP_BLIT].RenderContext.push_back( {vulkan.m_SwapchainRenderPass.Copy(), {}, vulkan.GetSwapchainFramebuffer( whichBuffer ), "RP_BLIT"} );
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitGui(uintptr_t windowHandle)
//-----------------------------------------------------------------------------
{
    const auto& hudRenderTarget = m_RenderPassData[RP_HUD].RenderTarget;
    m_Gui = std::make_unique<GuiImguiGfx>(*GetVulkan(), m_RenderPassData[RP_HUD].RenderContext[0].GetRenderPass().Copy());
    if (!m_Gui->Initialize(windowHandle, TextureFormat::R8G8B8A8_UNORM, hudRenderTarget.m_Width, hudRenderTarget.m_Height))
    {
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::LoadMeshObjects()
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();

    LOGI("***********************");
    LOGI("Initializing Meshes... ");
    LOGI("***********************");

    const auto* pSceneOpaqueShader      = m_ShaderManager->GetShader("SceneOpaque");
    const auto* pSceneTransparentShader = m_ShaderManager->GetShader("SceneTransparent");
    const auto* pBlitQuadShader         = m_ShaderManager->GetShader("Blit");
    if (!pSceneOpaqueShader || !pSceneTransparentShader || !pBlitQuadShader)
    {
        return false;
    }
    
    LOGI("***********************************");
    LOGI("Loading and preparing the museum...");
    LOGI("***********************************");

    m_TextureManager->SetDefaultFilenameManipulators(PathManipulator_PrefixDirectory(TEXTURE_DESTINATION_PATH));

    const PathManipulator_PrefixDirectory prefixTextureDir{ TEXTURE_DESTINATION_PATH };
    auto* whiteTexture         = m_TextureManager->GetOrLoadTexture("white_d.ktx", m_SamplerRepeat, prefixTextureDir);
    auto* blackTexture         = m_TextureManager->GetOrLoadTexture("black_d.ktx", m_SamplerRepeat, prefixTextureDir);
    auto* normalDefaultTexture = m_TextureManager->GetOrLoadTexture("normal_default.ktx", m_SamplerRepeat, prefixTextureDir);
    
    if (!whiteTexture || !blackTexture || !normalDefaultTexture)
    {
        LOGE("Failed to load supporting textures");
        return false;
    }

    auto UniformBufferLoader = [&](const ObjectMaterialParameters& objectMaterialParameters) -> ObjectMaterialParameters&
    {
        auto hash = objectMaterialParameters.GetHash();

        auto iter = m_ObjectFragUniforms.try_emplace(hash, ObjectMaterialParameters());
        if (iter.second)
        {
            iter.first->second.objectFragUniformData = objectMaterialParameters.objectFragUniformData;
            if (!CreateUniformBuffer(pVulkan, iter.first->second.objectFragUniform))
            {
                LOGE("Failed to create object uniform buffer");
            }
        }

        return iter.first->second;
    };

    auto MaterialLoader = [&](const MeshObjectIntermediate::MaterialDef& materialDef)->std::optional<Material>
    {
        const PathManipulator_PrefixDirectory prefixTextureDir{ TEXTURE_DESTINATION_PATH };
        const PathManipulator_ChangeExtension changeTextureExt{".ktx"};

        auto* diffuseTexture           = m_TextureManager->GetOrLoadTexture(materialDef.diffuseFilename, m_SamplerRepeat, prefixTextureDir, changeTextureExt);
        auto* normalTexture            = m_TextureManager->GetOrLoadTexture(materialDef.bumpFilename, m_SamplerRepeat, prefixTextureDir, changeTextureExt);
        auto* emissiveTexture          = m_TextureManager->GetOrLoadTexture(materialDef.emissiveFilename, m_SamplerRepeat, prefixTextureDir, changeTextureExt);
        auto* metallicRoughnessTexture = m_TextureManager->GetOrLoadTexture(materialDef.specMapFilename, m_SamplerRepeat, prefixTextureDir, changeTextureExt);
        bool alphaCutout               = materialDef.alphaCutout;
        bool transparent               = materialDef.transparent;

        const auto* targetShader = transparent ? pSceneTransparentShader : pSceneOpaqueShader;

        ObjectMaterialParameters objectMaterial;
        objectMaterial.objectFragUniformData.Color.r = static_cast<float>(materialDef.baseColorFactor[0]);
        objectMaterial.objectFragUniformData.Color.g = static_cast<float>(materialDef.baseColorFactor[1]);
        objectMaterial.objectFragUniformData.Color.b = static_cast<float>(materialDef.baseColorFactor[2]);
        objectMaterial.objectFragUniformData.Color.a = static_cast<float>(materialDef.baseColorFactor[3]);
        objectMaterial.objectFragUniformData.ORM.b   = static_cast<float>(materialDef.metallicFactor);
        objectMaterial.objectFragUniformData.ORM.g   = static_cast<float>(materialDef.roughnessFactor);

        if (diffuseTexture == nullptr || normalTexture == nullptr)
        {
            return std::nullopt;
        }

        auto shaderMaterial = m_MaterialManager->CreateMaterial(*targetShader, NUM_VULKAN_BUFFERS,
            [&](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
            {
                if (texName == "Diffuse")
                {
                    return { diffuseTexture ? diffuseTexture : whiteTexture };
                }
                if (texName == "Normal")
                {
                    return { normalTexture ? normalTexture : normalDefaultTexture };
                }
                if (texName == "Emissive")
                {
                    return { emissiveTexture ? emissiveTexture : blackTexture };
                }
                if (texName == "MetallicRoughness")
                {
                    return { metallicRoughnessTexture ? metallicRoughnessTexture : blackTexture };
                }

                assert(false);
                return {};
            },
            [&](const std::string& bufferName) -> PerFrameBufferVulkan
            {
                if (bufferName == "Vert")
                {
                    return { m_ObjectVertUniform.buf.GetVkBuffer() };
                }
                else if (bufferName == "Frag")
                {
                    return { UniformBufferLoader(objectMaterial).objectFragUniform.buf.GetVkBuffer() };
                }
                else if (bufferName == "Light")
                {
                    return { m_LightUniform.buf.GetVkBuffer() };
                }

                assert(false);
                return {};
            }
            );

        return shaderMaterial;
    };

    const auto loaderFlags = 0; // No instancing
    const bool ignoreTransforms = (loaderFlags & DrawableLoader::LoaderFlags::IgnoreHierarchy) != 0;

    const auto sceneAssetPath = std::filesystem::path(MESH_DESTINATION_PATH).append(gSceneAssetModel).string();
    MeshLoaderModelSceneSanityCheck meshSanityCheckProcessor(sceneAssetPath);
    MeshObjectIntermediateGltfProcessor meshObjectProcessor(sceneAssetPath, ignoreTransforms, glm::vec3(1.0f, 1.0f, 1.0f));
    CameraGltfProcessor meshCameraProcessor{};

    if (!MeshLoader::LoadGltf(*m_AssetManager, sceneAssetPath, meshSanityCheckProcessor, meshObjectProcessor, meshCameraProcessor) ||
        !DrawableLoader::CreateDrawables(
            *pVulkan,
            std::move(meshObjectProcessor.m_meshObjects),
            m_RenderPassData[RP_SCENE].RenderContext,
            MaterialLoader,
            m_SceneDrawables,
            loaderFlags))
    {
        LOGE("Error Loading the gltf file");
        LOGI("Please verify if you have all required assets on the media folder");
        return false;
    }

    if (!meshCameraProcessor.m_cameras.empty())
    {
        const auto& camera = meshCameraProcessor.m_cameras[0];
        m_Camera.SetPosition(camera.Position, camera.Orientation);
    }

    LOGI("*********************");
    LOGI("Creating Quad mesh...");
    LOGI("*********************");

    Mesh blitQuadMesh;
    MeshHelper::CreateScreenSpaceMesh(pVulkan->GetMemoryManager(), 0, &blitQuadMesh);

    // Blit Material
    auto blitQuadShaderMaterial = m_MaterialManager->CreateMaterial(*pBlitQuadShader, pVulkan->m_SwapchainImageCount,
        [this](const std::string& texName) -> const MaterialManager::tPerFrameTexInfo
        {
            if (texName == "Diffuse")
            {
                return { &m_UpscaledImageResult };
            }
            else if (texName == "Overlay")
            {
                return { &m_RenderPassData[RP_HUD].RenderTarget.GetColorAttachments()[0] };
            }
            return {};
        },
        [this](const std::string& bufferName) -> PerFrameBufferVulkan
        {
            if (bufferName == "Params")
            {
                return { m_BlitFragUniform.buf.GetVkBuffer() };
            }
            return {};
        }
        );

    m_BlitQuadDrawable = std::make_unique<Drawable>(*pVulkan, std::move(blitQuadShaderMaterial));
    if (!m_BlitQuadDrawable->Init(m_RenderPassData[RP_BLIT].RenderContext[0], std::move(blitQuadMesh)))
    {
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitCommandBuffers()
//-----------------------------------------------------------------------------
{
    LOGI("*******************************");
    LOGI("Initializing Command Buffers...");
    LOGI("*******************************");

    Vulkan* const pVulkan = GetVulkan();

    auto GetPassName = [](uint32_t whichPass)
    {
        if (whichPass >= sRenderPassNames.size())
        {
            LOGE("GetPassName() called with unknown pass (%d)!", whichPass);
            return "RP_UNKNOWN";
        }

        return sRenderPassNames[whichPass];
    };

    m_RenderPassData[RP_SCENE].PassCmdBuffer.resize(NUM_VULKAN_BUFFERS);
    m_RenderPassData[RP_SCENE].ObjectsCmdBuffer.resize(NUM_VULKAN_BUFFERS);
    m_RenderPassData[RP_HUD].PassCmdBuffer.resize(NUM_VULKAN_BUFFERS);
    m_RenderPassData[RP_HUD].ObjectsCmdBuffer.resize(NUM_VULKAN_BUFFERS);
    m_RenderPassData[RP_BLIT].PassCmdBuffer.resize(pVulkan->m_SwapchainImageCount);
    m_RenderPassData[RP_BLIT].ObjectsCmdBuffer.resize(pVulkan->m_SwapchainImageCount);

    char szName[256];
    const CommandListBase::Type CmdBuffLevel = CommandListBase::Type::Secondary;
    for (uint32_t whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        for (uint32_t whichBuffer = 0; whichBuffer < m_RenderPassData[whichPass].PassCmdBuffer.size(); whichBuffer++)
        {
            // The Pass Command Buffer => Primary
            sprintf(szName, "Primary (%s; Buffer %d of %d)", GetPassName(whichPass), whichBuffer + 1, NUM_VULKAN_BUFFERS);
            if (!m_RenderPassData[whichPass].PassCmdBuffer[whichBuffer].Initialize(pVulkan, szName, CommandListBase::Type::Primary))
            {
                return false;
            }

            // Model => Secondary
            sprintf(szName, "Model (%s; Buffer %d of %d)", GetPassName(whichPass), whichBuffer + 1, NUM_VULKAN_BUFFERS);
            if (!m_RenderPassData[whichPass].ObjectsCmdBuffer[whichBuffer].Initialize(pVulkan, szName, CmdBuffLevel))
            {
                return false;
            }
        }
    }

    LOGI("Creating Graph Pipeline Command Lists");
    if (m_IsGraphPipelinesSupported)
    {
        m_GraphPipelineCommandLists.resize(NUM_VULKAN_BUFFERS);
        for (auto& graphPipelineCommandList : m_GraphPipelineCommandLists)
        {
            if (!graphPipelineCommandList.Initialize(GetVulkan(), "Graph Pipeline CMD Buffer", CommandListBase::Type::Primary, Vulkan::eDataGraphQueue))
            {
                return false;
            }
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitLocalSemaphores()
//-----------------------------------------------------------------------------
{
    LOGI("********************************");
    LOGI("Initializing Local Semaphores...");
    LOGI("********************************");

    const VkSemaphoreCreateInfo SemaphoreInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

    for (uint32_t whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        VkResult retVal = vkCreateSemaphore(GetVulkan()->m_VulkanDevice, &SemaphoreInfo, NULL, &m_RenderPassData[whichPass].PassCompleteSemaphore);
        if (!CheckVkError("vkCreateSemaphore()", retVal))
        {
            return false;
        }
    }

    // Create the graph pipeline semaphore
    {
        VkResult retVal = vkCreateSemaphore(GetVulkan()->m_VulkanDevice, &SemaphoreInfo, NULL, &m_GraphPipelinePassCompleteSemaphore);
        if (!CheckVkError("vkCreateSemaphore()", retVal))
        {
            return false;
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::BuildCmdBuffers()
//-----------------------------------------------------------------------------
{
    LOGI("***************************");
    LOGI("Building Command Buffers...");
    LOGI("****************************");

    Vulkan* const pVulkan = GetVulkan();

    // Begin recording
    for (uint32_t whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        auto& renderPassData         = m_RenderPassData[whichPass];
        bool  bisSwapChainRenderPass = whichPass == RP_BLIT;

        for (uint32_t whichBuffer = 0; whichBuffer < renderPassData.ObjectsCmdBuffer.size(); whichBuffer++)
        {
            auto& cmdBufer = renderPassData.ObjectsCmdBuffer[whichBuffer];

            uint32_t targetWidth  = bisSwapChainRenderPass ? pVulkan->m_SurfaceWidth : renderPassData.RenderTarget.GetWidth();
            uint32_t targetHeight = bisSwapChainRenderPass ? pVulkan->m_SurfaceHeight : renderPassData.RenderTarget.GetHeight();

            VkViewport viewport = {};
            viewport.x          = 0.0f;
            viewport.y          = 0.0f;
            viewport.width      = (float)targetWidth;
            viewport.height     = (float)targetHeight;
            viewport.minDepth   = 0.0f;
            viewport.maxDepth   = 1.0f;

            VkRect2D scissor      = {};
            scissor.offset.x      = 0;
            scissor.offset.y      = 0;
            scissor.extent.width  = targetWidth;
            scissor.extent.height = targetHeight;

            // Set up some values that change based on render pass
            VkFramebuffer whichFramebuffer = bisSwapChainRenderPass ? pVulkan->m_SwapchainBuffers[whichBuffer].framebuffer : renderPassData.RenderContext[0].GetFramebuffer()->m_FrameBuffer;

            // Objects (can render into any pass except Blit)
            if (!cmdBufer.Begin(whichFramebuffer, renderPassData.RenderContext[0].GetRenderPass(), bisSwapChainRenderPass))
            {
                return false;
            }
            vkCmdSetViewport(cmdBufer.m_VkCommandBuffer, 0, 1, &viewport);
            vkCmdSetScissor(cmdBufer.m_VkCommandBuffer, 0, 1, &scissor);
        }
    }
    
    // Scene drawables
    for (const auto& sceneDrawable : m_SceneDrawables)
    {
        AddDrawableToCmdBuffers(sceneDrawable, m_RenderPassData[RP_SCENE].ObjectsCmdBuffer.data(), 1, static_cast<uint32_t>(m_RenderPassData[RP_SCENE].ObjectsCmdBuffer.size()));
    }

    // Blit quad drawable
    AddDrawableToCmdBuffers(*m_BlitQuadDrawable.get(), m_RenderPassData[RP_BLIT].ObjectsCmdBuffer.data(), 1, static_cast<uint32_t>(m_RenderPassData[RP_BLIT].ObjectsCmdBuffer.size()));

    // End recording
    for (uint32_t whichPass = 0; whichPass < NUM_RENDER_PASSES; whichPass++)
    {
        auto& renderPassData = m_RenderPassData[whichPass];

        for (uint32_t whichBuffer = 0; whichBuffer < renderPassData.ObjectsCmdBuffer.size(); whichBuffer++)
        {
            auto& cmdBufer = renderPassData.ObjectsCmdBuffer[whichBuffer];
            if (!cmdBufer.End())
            {
                return false;
            }
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
void Application::UpdateGui()
//-----------------------------------------------------------------------------
{
    if (m_Gui)
    {
        m_Gui->Update();
        ImGuiIO& io = ImGui::GetIO();

        if (ImGui::Begin("FPS", (bool*)nullptr, ImGuiWindowFlags_NoTitleBar))
        {
            ImGui::Text("FPS: %.1f", m_CurrentFPS);
            ImGui::Text("Camera [%f, %f, %f]", m_Camera.Position().x, m_Camera.Position().y, m_Camera.Position().z);

            ImGui::Separator();

            ImGui::BeginDisabled(!m_IsGraphPipelinesSupported);
            ImGui::Checkbox("Upscaling Enabled", &m_ShouldUpscale);
            ImGui::EndDisabled();

            if (ImGui::CollapsingHeader("Sun Light", ImGuiTreeNodeFlags_Framed))
            {
                ImGui::DragFloat3("Sun Dir", &m_LightUniformData.LightDirection.x, 0.01f, -1.0f, 1.0f);
                ImGui::DragFloat3("Sun Color", &m_LightUniformData.LightColor.x, 0.01f, 0.0f, 1.0f);
                ImGui::DragFloat("Sun Intensity", &m_LightUniformData.LightColor.w, 0.1f, 0.0f, 100.0f);
                ImGui::DragFloat3("Ambient Color", &m_LightUniformData.AmbientColor.x, 0.01f, 0.0f, 1.0f);
            }

            if (ImGui::CollapsingHeader("Spot Lights", ImGuiTreeNodeFlags_Framed))
            {
                for (int i = 0; i < NUM_SPOT_LIGHTS; i++)
                {
                    std::string childName = std::string("Spot Light ").append(std::to_string(i + 1));
                    ImGui::TextColored(ImVec4(1, 1, 0, 1), "%s", childName.c_str());

                    if (ImGui::CollapsingHeader(childName.c_str(), ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed))
                    {
                        ImGui::PushID(i);

                        ImGui::DragFloat3("Pos", &m_LightUniformData.SpotLights_pos[i].x, 0.1f);
                        ImGui::DragFloat3("Dir", &m_LightUniformData.SpotLights_dir[i].x, 0.01f, -1.0f, 1.0f);
                        ImGui::DragFloat3("Color", &m_LightUniformData.SpotLights_color[i].x, 0.01f, 0.0f, 1.0f);
                        ImGui::DragFloat("Intensity", &m_LightUniformData.SpotLights_color[i].w, 0.1f, 0.0f, 100.0f);

                        ImGui::PopID();
                    }

                    glm::vec3 LightDirNotNormalized = m_LightUniformData.SpotLights_dir[i];
                    LightDirNotNormalized = glm::normalize(LightDirNotNormalized);
                    m_LightUniformData.SpotLights_dir[i] = glm::vec4(LightDirNotNormalized, 0.0f);
                }
            }

            glm::vec3 LightDirNotNormalized   = m_LightUniformData.LightDirection;
            LightDirNotNormalized             = glm::normalize(LightDirNotNormalized);
            m_LightUniformData.LightDirection = glm::vec4(LightDirNotNormalized, 0.0f);
        }
        ImGui::End();

        return;
    }
}

//-----------------------------------------------------------------------------
bool Application::UpdateUniforms(uint32_t whichBuffer)
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();

    // Vert data
    {
        glm::mat4 LocalModel = glm::mat4(1.0f);
        LocalModel           = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f));
        LocalModel           = glm::scale(LocalModel, glm::vec3(1.0f));
        glm::mat4 LocalMVP   = m_Camera.ProjectionMatrix() * m_Camera.ViewMatrix() * LocalModel;

        m_ObjectVertUniformData.MVPMatrix   = LocalMVP;
        m_ObjectVertUniformData.ModelMatrix = LocalModel;
        UpdateUniformBuffer(pVulkan, m_ObjectVertUniform, m_ObjectVertUniformData);
    }

    // Frag data
    for (auto& [hash, objectUniform] : m_ObjectFragUniforms)
    {
        UpdateUniformBuffer(pVulkan, objectUniform.objectFragUniform, objectUniform.objectFragUniformData);
    }

    // Light data
    {
        glm::mat4 CameraViewInv       = glm::inverse(m_Camera.ViewMatrix());
        glm::mat4 CameraProjection    = m_Camera.ProjectionMatrix();
        glm::mat4 CameraProjectionInv = glm::inverse(CameraProjection);

        m_LightUniformData.ProjectionInv     = CameraProjectionInv;
        m_LightUniformData.ViewInv           = CameraViewInv;
        m_LightUniformData.ViewProjectionInv = glm::inverse(CameraProjection * m_Camera.ViewMatrix());
        m_LightUniformData.ProjectionInvW    = glm::vec4(CameraProjectionInv[0].w, CameraProjectionInv[1].w, CameraProjectionInv[2].w, CameraProjectionInv[3].w);
        m_LightUniformData.CameraPos         = glm::vec4(m_Camera.Position(), 0.0f);

        UpdateUniformBuffer(pVulkan, m_LightUniform, m_LightUniformData);
    }

    // Blit data
    {
        m_BlitFragUniformData.IsUpscalingActive = m_ShouldUpscale;
        UpdateUniformBuffer(pVulkan, m_BlitFragUniform, m_BlitFragUniformData);
    }

    return true;
}

//-----------------------------------------------------------------------------
void Application::Render(float fltDiffTime)
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();

    // Obtain the next swap chain image for the next frame.
    auto currentVulkanBuffer = pVulkan->SetNextBackBuffer();
    uint32_t whichBuffer     = currentVulkanBuffer.idx;

    // ********************************
    // Application Draw() - Begin
    // ********************************

    UpdateGui();

    // Update camera
    m_Camera.UpdateController(fltDiffTime * 0.1f, *m_CameraController);
    m_Camera.UpdateMatrices();
 
    // Update uniform buffers with latest data
    UpdateUniforms(whichBuffer);

    // First time through, wait for the back buffer to be ready
    std::span<const VkSemaphore> pWaitSemaphores = { &currentVulkanBuffer.semaphore, 1 };

    std::array<VkPipelineStageFlags, 1> waitDstStageMasks = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

    const bool isUpscalingActive = m_IsGraphPipelinesSupported && m_ShouldUpscale;

    // RP_SCENE
    {
        BeginRenderPass(whichBuffer, RP_SCENE, currentVulkanBuffer.swapchainPresentIdx);
        AddPassCommandBuffer(whichBuffer, RP_SCENE);
        EndRenderPass(whichBuffer, RP_SCENE);

        // Before finishing the scene cmd buffer, copy its color render target contents to the tensor buffer if upscaling 
        // is active, otherwise blit them directly to the upscaled image.
        if (isUpscalingActive)
        {
            CopyImageToTensor(
                m_RenderPassData[RP_SCENE].PassCmdBuffer[whichBuffer],
                m_RenderPassData[RP_SCENE].RenderTarget.GetColorAttachments()[0],
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                m_InputTensor);
        }
        else
        {
            CopyImageToImageBlit(
                m_RenderPassData[RP_SCENE].PassCmdBuffer[whichBuffer], 
                m_RenderPassData[RP_SCENE].RenderTarget.GetColorAttachments()[0], 
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                m_UpscaledImageResult,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }

        // Submit the commands to the queue.
        SubmitRenderPass(whichBuffer, RP_SCENE, pWaitSemaphores, waitDstStageMasks, { &m_RenderPassData[RP_SCENE].PassCompleteSemaphore,1 });
        pWaitSemaphores      = { &m_RenderPassData[RP_SCENE].PassCompleteSemaphore, 1 };
        waitDstStageMasks[0] = { VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
    }

    // Data Graph preparation + dispatch for Upscaling
    if (isUpscalingActive)
    {
        m_GraphPipelineCommandLists[whichBuffer].Begin();

        m_data_graph_dispatch.RecordDispatch(
            m_GraphPipelineCommandLists[whichBuffer].m_VkCommandBuffer,
            m_GraphPipelineInstance.graphPipeline,
            m_GraphPipelineInstance.pipelineLayout,
            m_tensor_resources.GetResources().tensor_descriptor_set,
            m_GraphPipelineInstance.graphSession);

        m_GraphPipelineCommandLists[whichBuffer].End();

        if (!m_data_graph_dispatch.Submit(
            pVulkan->m_VulkanQueues[m_GraphPipelineCommandLists[whichBuffer].m_QueueIndex].Queue,
            m_GraphPipelineCommandLists[whichBuffer].m_VkCommandBuffer,
            pWaitSemaphores[0],
            waitDstStageMasks[0],
            m_GraphPipelinePassCompleteSemaphore,
            VK_NULL_HANDLE))
        {
            LOGE("Data Graph dispatch failed");
        }

        pWaitSemaphores      = { &m_GraphPipelinePassCompleteSemaphore, 1 };
        waitDstStageMasks[0] = { VK_PIPELINE_STAGE_ALL_COMMANDS_BIT }; // Should be VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM, but need to update framework
                                                                       // to support VK_PIPELINE_STAGE_2.
    }

    // RP_HUD
    VkCommandBuffer guiCommandBuffer = VK_NULL_HANDLE;
    if (m_Gui)
    {
        // Render gui (has its own command buffer, optionally returns vk_null_handle if not rendering anything)
        guiCommandBuffer = GetGui()->Render(whichBuffer, m_RenderPassData[RP_HUD].RenderContext[0].GetFramebuffer()->m_FrameBuffer);
        if (guiCommandBuffer != VK_NULL_HANDLE)
        {
            BeginRenderPass(whichBuffer, RP_HUD, currentVulkanBuffer.swapchainPresentIdx);
            vkCmdExecuteCommands(m_RenderPassData[RP_HUD].PassCmdBuffer[whichBuffer].m_VkCommandBuffer, 1, &guiCommandBuffer);
            EndRenderPass(whichBuffer, RP_HUD);

            // Submit the commands to the queue.
            SubmitRenderPass(whichBuffer, RP_HUD, pWaitSemaphores, waitDstStageMasks, { &m_RenderPassData[RP_HUD].PassCompleteSemaphore,1 });
            pWaitSemaphores      = { &m_RenderPassData[RP_HUD].PassCompleteSemaphore,1 };
            waitDstStageMasks[0] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        }
    }

    // Blit Results to the screen
    {
        if (!m_RenderPassData[RP_BLIT].PassCmdBuffer[whichBuffer].Reset())
        {
            LOGE("Pass (%d) command buffer Reset() failed !", RP_BLIT);
        }

        if (!m_RenderPassData[RP_BLIT].PassCmdBuffer[whichBuffer].Begin())
        {
            LOGE("Pass (%d) command buffer Begin() failed !", RP_BLIT);
        }

        // Before begining the blit render pass, copy the output tensor contents to the upscaled image if upscaling
        // is active, otherwise nothing needs to be done as the scene pass should have already blit its contents directly
        // into the upscale image.
        if (isUpscalingActive)
        {
            CopyTensorToImage(
                m_RenderPassData[RP_BLIT].PassCmdBuffer[whichBuffer],
                m_UpscaledImageResult,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                m_OutputTensor);
        }

        BeginRenderPass(whichBuffer, RP_BLIT, currentVulkanBuffer.swapchainPresentIdx, false);
        AddPassCommandBuffer(whichBuffer, RP_BLIT);
        EndRenderPass(whichBuffer, RP_BLIT);

        // Submit the commands to the queue.
        SubmitRenderPass(whichBuffer, RP_BLIT, pWaitSemaphores, waitDstStageMasks, { &m_RenderPassData[RP_BLIT].PassCompleteSemaphore,1 }, currentVulkanBuffer.fence);
        pWaitSemaphores      = { &m_RenderPassData[RP_BLIT].PassCompleteSemaphore,1 };
        waitDstStageMasks[0] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    }

    // Queue is loaded up, tell the driver to start processing
    pVulkan->PresentQueue(pWaitSemaphores, currentVulkanBuffer.swapchainPresentIdx);

    // ********************************
    // Application Draw() - End
    // ********************************
}

//-----------------------------------------------------------------------------
void Application::BeginRenderPass(uint32_t whichBuffer, RENDER_PASS whichPass, uint32_t WhichSwapchainImage, bool beginCmdBuffer)
//-----------------------------------------------------------------------------
{
    Vulkan* const pVulkan = GetVulkan();
    auto& renderPassData         = m_RenderPassData[whichPass];
    bool  bisSwapChainRenderPass = whichPass == RP_BLIT;

    if (beginCmdBuffer)
    {
        if (!m_RenderPassData[whichPass].PassCmdBuffer[whichBuffer].Reset())
        {
            LOGE("Pass (%d) command buffer Reset() failed !", whichPass);
        }

        if (!m_RenderPassData[whichPass].PassCmdBuffer[whichBuffer].Begin())
        {
            LOGE("Pass (%d) command buffer Begin() failed !", whichPass);
        }
    }
    
    VkFramebuffer framebuffer = nullptr;
    switch (whichPass)
    {
    case RP_SCENE:
        framebuffer = m_RenderPassData[whichPass].RenderContext[0].GetFramebuffer()->m_FrameBuffer;
        break;
    case RP_HUD:
        framebuffer = m_RenderPassData[whichPass].RenderContext[0].GetFramebuffer()->m_FrameBuffer;
        break;
    case RP_BLIT:
        framebuffer = pVulkan->m_SwapchainBuffers[WhichSwapchainImage].framebuffer;
        break;
    default:
        framebuffer = nullptr;
        break;
    }

    assert(framebuffer != nullptr);

    VkRect2D passArea = {};
    passArea.offset.x = 0;
    passArea.offset.y = 0;
    passArea.extent.width  = bisSwapChainRenderPass ? pVulkan->m_SurfaceWidth  : renderPassData.RenderTarget.GetWidth();
    passArea.extent.height = bisSwapChainRenderPass ? pVulkan->m_SurfaceHeight : renderPassData.RenderTarget.GetHeight();

    TextureFormat                  swapChainColorFormat = pVulkan->m_SurfaceFormat;
    auto                           swapChainColorFormats = std::span<const TextureFormat>({ &swapChainColorFormat, 1 });
    TextureFormat                  swapChainDepthFormat = pVulkan->m_SwapchainDepth.format;
    std::span<const TextureFormat> colorFormats         = bisSwapChainRenderPass ? swapChainColorFormats : m_RenderPassData[whichPass].RenderTarget.GetLayerFormats();
    TextureFormat                  depthFormat          = bisSwapChainRenderPass ? swapChainDepthFormat : m_RenderPassData[whichPass].RenderTarget.GetDepthFormat();

    VkClearColorValue clearColor = { renderPassData.RenderPassSetup.ClearColor[0], renderPassData.RenderPassSetup.ClearColor[1], renderPassData.RenderPassSetup.ClearColor[2], renderPassData.RenderPassSetup.ClearColor[3] };

    m_RenderPassData[whichPass].PassCmdBuffer[whichBuffer].BeginRenderPass(
        passArea,
        0.0f,
        1.0f,
        { &clearColor , 1 },
        (uint32_t)colorFormats.size(),
        depthFormat != TextureFormat::UNDEFINED,
        m_RenderPassData[whichPass].RenderContext[0].GetRenderPass(),
        bisSwapChainRenderPass,
        framebuffer,
        VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
}


//-----------------------------------------------------------------------------
void Application::AddPassCommandBuffer(uint32_t whichBuffer, RENDER_PASS whichPass)
//-----------------------------------------------------------------------------
{
    if (m_RenderPassData[whichPass].ObjectsCmdBuffer[whichBuffer].m_NumDrawCalls)
    {
        vkCmdExecuteCommands(m_RenderPassData[whichPass].PassCmdBuffer[whichBuffer].m_VkCommandBuffer, 1, &m_RenderPassData[whichPass].ObjectsCmdBuffer[whichBuffer].m_VkCommandBuffer);
    }
}

//-----------------------------------------------------------------------------
void Application::EndRenderPass(uint32_t whichBuffer, RENDER_PASS whichPass)
//-----------------------------------------------------------------------------
{
    m_RenderPassData[whichPass].PassCmdBuffer[whichBuffer].EndRenderPass();
}

//-----------------------------------------------------------------------------
void Application::SubmitRenderPass(uint32_t whichBuffer, RENDER_PASS whichPass, const std::span<const VkSemaphore> WaitSemaphores, const std::span<const VkPipelineStageFlags> WaitDstStageMasks, std::span<VkSemaphore> SignalSemaphores, VkFence CompletionFence)
//-----------------------------------------------------------------------------
{
    m_RenderPassData[whichPass].PassCmdBuffer[whichBuffer].End();
    m_RenderPassData[whichPass].PassCmdBuffer[whichBuffer].QueueSubmit(WaitSemaphores, WaitDstStageMasks, SignalSemaphores, CompletionFence);
}
