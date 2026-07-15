//============================================================================================================
//
//
//                  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "imguiVulkan.hpp"
#include "vulkan/vulkan.hpp"
#include "texture/vulkan/texture.hpp"
#include "imgui/imgui.h"
#include "imgui/backends/imgui_impl_vulkan.h"
#include "imgui/backends/imgui_impl_win32.h"
#include "system/os_common.h"
#include <cassert>

static void check_vk_result(VkResult err)
{
    if (err == 0)
        return;
    LOGE("VkResult %d\n", err);
    if (err < 0)
        assert(0);
}

//-----------------------------------------------------------------------------

GuiImguiGfx<Vulkan>::GuiImguiGfx(Vulkan& vulkan, RenderPass<Vulkan> renderPass)
    : GuiImguiPlatform()
    , m_RenderPass(std::move(renderPass))
    , m_GfxApi(vulkan)
{}

//-----------------------------------------------------------------------------

GuiImguiGfx<Vulkan>::~GuiImguiGfx()
{
    //We dont own the render pass// vkDestroyRenderPass(m_GfxApi.m_VulkanDevice, m_RenderPass, nullptr);
    
    ImGui_ImplVulkan_Shutdown();

    // Imgui needs the descriptor pool to release the descriptor sets, we can destroy it after
    vkDestroyDescriptorPool(m_GfxApi.m_VulkanDevice, m_DescriptorPool, nullptr);
}

//-----------------------------------------------------------------------------

bool GuiImguiGfx<Vulkan>::Initialize(uintptr_t windowHandle, TextureFormat renderFormat, uint32_t renderWidth, uint32_t renderHeight)
{
    // Call the platform (windows/android) specific implementation initialize...
    if (!GuiImguiPlatform::Initialize(windowHandle, renderFormat, renderWidth, renderHeight, renderWidth, renderHeight ))
    {
        return false;
    }

    const auto outputFormat = TextureFormatToVk( renderFormat );

    // Give Imgui its own descriptor pool
    {
        // Sizes from imgui example main.cpp
        const VkDescriptorPoolSize pool_sizes[] =
        {
            { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
        };
        const VkDescriptorPoolCreateInfo pool_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 1000 * IM_ARRAYSIZE(pool_sizes),
            .poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes),
            .pPoolSizes = pool_sizes
        };
        VkResult err = vkCreateDescriptorPool(m_GfxApi.m_VulkanDevice, &pool_info, nullptr, &m_DescriptorPool);
        check_vk_result(err);
        if (err != VK_SUCCESS)
        {
            return false;
        }
    }

    // Give imgui its own commandBuffers
    {
        m_CommandBuffer.reserve(m_GfxApi.m_SwapchainImageCount);
        for (uint32_t imageIdx = 0; imageIdx < m_GfxApi.m_SwapchainImageCount; ++imageIdx)
        {
            if (!m_CommandBuffer.emplace_back().Initialize(&m_GfxApi, "ImGui", CommandListBase::Type::Secondary))
            {
                return false;
            }
        }
        if (!m_UploadCommandBuffer.Initialize(&m_GfxApi, "ImGui upload"))
        {
            return false;
        }
    }

    // Create the ImGui Vulkan implementation!
    {
        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = m_GfxApi.GetVulkanInstance();
        init_info.PhysicalDevice = m_GfxApi.m_VulkanGpu;
        init_info.Device = m_GfxApi.m_VulkanDevice;
        init_info.QueueFamily = m_GfxApi.m_VulkanQueues[Vulkan::eGraphicsQueue].QueueFamilyIndex;
        init_info.Queue = m_GfxApi.m_VulkanQueues[Vulkan::eGraphicsQueue].Queue;
        init_info.PipelineCache = nullptr;// g_PipelineCache;
        init_info.DescriptorPool = m_DescriptorPool;
        init_info.Allocator = nullptr;// g_Allocator;
        init_info.MinImageCount = m_GfxApi.m_SwapchainImageCount;   //unused?
        init_info.ImageCount = m_GfxApi.m_SwapchainImageCount;
        init_info.CheckVkResultFn = check_vk_result;
        init_info.UseDynamicRendering = !m_RenderPass;
        if (init_info.UseDynamicRendering)
        {
            init_info.PipelineRenderingCreateInfo = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
                .viewMask                = 0,
                .colorAttachmentCount    = 1,
                .pColorAttachmentFormats = &outputFormat,
            };
        }
        else
        {
            // Non dynamic rendering
            init_info.RenderPass = m_RenderPass.mRenderPass;
        }

        ImGui_ImplVulkan_LoadFunctions([](const char* function_name, void* vulkan_instance)
        {
            return vkGetInstanceProcAddr(*(reinterpret_cast<VkInstance*>(vulkan_instance)), function_name);
        }, &init_info.Instance);

        if (!ImGui_ImplVulkan_Init(&init_info))
        {
            vkDestroyDescriptorPool(m_GfxApi.m_VulkanDevice, m_DescriptorPool, nullptr);
            m_DescriptorPool = VK_NULL_HANDLE;
            return false;
        }
    }

    // While we are here, use the new command buffer to upload the font.
    {
        if (!m_UploadCommandBuffer.Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        {
            return false;
        }

        if (!ImGui_ImplVulkan_CreateFontsTexture())
        {
            return false;
        }
        if (!m_UploadCommandBuffer.End())
        {
            return false;
        }
        VkSubmitInfo end_info = {};
        end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        end_info.commandBufferCount = 1;
        end_info.pCommandBuffers = &m_UploadCommandBuffer.m_VkCommandBuffer;
        VkResult err = vkQueueSubmit(m_GfxApi.m_VulkanQueues[Vulkan::eGraphicsQueue].Queue, 1, &end_info, VK_NULL_HANDLE);
        check_vk_result(err);

        err = vkDeviceWaitIdle(m_GfxApi.m_VulkanDevice);
        check_vk_result(err);
        m_UploadCommandBuffer.Reset();
    }

    return true;
}

//-----------------------------------------------------------------------------

void GuiImguiGfx<Vulkan>::Update()
{
    ImGui_ImplVulkan_NewFrame();
    GuiImguiPlatform::Update();
}

//-----------------------------------------------------------------------------

VkCommandBuffer GuiImguiGfx<Vulkan>::Render(uint32_t frameIdx, VkFramebuffer frameBuffer)
{
    assert(m_RenderPass);   // cannot be called when doing dynamic rendering (use Render(VkCommandBuffer))
    if (!m_RenderPass)
        return VK_NULL_HANDLE;

    auto& commandBuffer = m_CommandBuffer[frameIdx];
    if (!commandBuffer.Begin(frameBuffer, m_RenderPass, false, 0, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
    {
        return VK_NULL_HANDLE;
    }

    if (commandBuffer.m_Type == CommandListBase::Type::Primary)
    {
        fvk::VkRenderPassBeginInfo info;
        info->renderPass = m_RenderPass.mRenderPass;
        info->framebuffer = frameBuffer;
        info->renderArea.extent.width = m_GfxApi.m_SurfaceWidth;
        info->renderArea.extent.height = m_GfxApi.m_SurfaceHeight;
        info->clearValueCount = 0;
        info->pClearValues = nullptr;
        commandBuffer.BeginRenderPass(*info, VK_SUBPASS_CONTENTS_INLINE);
    }

    Render(commandBuffer);

    if (commandBuffer.m_Type == CommandListBase::Type::Primary)
    {
        commandBuffer.EndRenderPass();
    }

    commandBuffer.End();
    return commandBuffer;
}

void GuiImguiGfx<Vulkan>::Render(VkCommandBuffer cmdBuffer)
{
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuffer);
}
