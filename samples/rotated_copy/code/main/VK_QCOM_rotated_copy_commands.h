// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#ifndef _VULKAN_VK_QCOM_ROTATED_COPY_H_
#define _VULKAN_VK_QCOM_ROTATED_COPY_H_

#include <vulkan/vulkan.h>

#if !defined(VK_QCOM_rotated_copy_commands)

// Provided by VK_QCOM_rotated_copy_commands
constexpr VkStructureType VK_STRUCTURE_TYPE_COPY_COMMAND_TRANSFORM_INFO_QCOM = (VkStructureType)1000333000;

#define VK_QCOM_rotated_copy_commands 1
#define VK_QCOM_rotated_copy_commands_SPEC_VERSION 0
#define VK_QCOM_rotated_copy_commands_EXTENSION_NAME "VK_QCOM_rotated_copy_commands"
typedef struct VkCopyCommandTransformInfoQCOM {
    VkStructureType                  sType;
    const void* pNext;
    VkSurfaceTransformFlagBitsKHR    transform;
} VkCopyCommandTransformInfoQCOM;

#endif // !defined(VK_QCOM_rotated_copy_commands)



#if !defined(VK_KHR_copy_commands2)

// Provided by VK_KHR_copy_commands2
constexpr VkStructureType VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2_KHR = (VkStructureType)1000337000;
// Provided by VK_KHR_copy_commands2
constexpr VkStructureType VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2_KHR = (VkStructureType)1000337001;
// Provided by VK_KHR_copy_commands2
constexpr VkStructureType VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2_KHR = (VkStructureType)1000337002;
// Provided by VK_KHR_copy_commands2
constexpr VkStructureType VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2_KHR = (VkStructureType)1000337003;
// Provided by VK_KHR_copy_commands2
constexpr VkStructureType VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2_KHR = (VkStructureType)1000337004;
// Provided by VK_KHR_copy_commands2
constexpr VkStructureType VK_STRUCTURE_TYPE_RESOLVE_IMAGE_INFO_2_KHR = (VkStructureType)1000337005;
// Provided by VK_KHR_copy_commands2
constexpr VkStructureType VK_STRUCTURE_TYPE_BUFFER_COPY_2_KHR = (VkStructureType)1000337006;
// Provided by VK_KHR_copy_commands2
constexpr VkStructureType VK_STRUCTURE_TYPE_IMAGE_COPY_2_KHR = (VkStructureType)1000337007;
// Provided by VK_KHR_copy_commands2
constexpr VkStructureType VK_STRUCTURE_TYPE_IMAGE_BLIT_2_KHR = (VkStructureType)1000337008;
// Provided by VK_KHR_copy_commands2
constexpr VkStructureType VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2_KHR = (VkStructureType)1000337009;
// Provided by VK_KHR_copy_commands2
constexpr VkStructureType VK_STRUCTURE_TYPE_IMAGE_RESOLVE_2_KHR = (VkStructureType)1000337010;

#define VK_KHR_copy_commands2 1
#define VK_KHR_COPY_COMMANDS_2_SPEC_VERSION 1
#define VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME "VK_KHR_copy_commands2"
typedef struct VkBufferCopy2KHR {
    VkStructureType    sType;
    const void* pNext;
    VkDeviceSize       srcOffset;
    VkDeviceSize       dstOffset;
    VkDeviceSize       size;
} VkBufferCopy2KHR;

typedef struct VkCopyBufferInfo2KHR {
    VkStructureType            sType;
    const void* pNext;
    VkBuffer                   srcBuffer;
    VkBuffer                   dstBuffer;
    uint32_t                   regionCount;
    const VkBufferCopy2KHR* pRegions;
} VkCopyBufferInfo2KHR;

typedef struct VkImageCopy2KHR {
    VkStructureType             sType;
    const void* pNext;
    VkImageSubresourceLayers    srcSubresource;
    VkOffset3D                  srcOffset;
    VkImageSubresourceLayers    dstSubresource;
    VkOffset3D                  dstOffset;
    VkExtent3D                  extent;
} VkImageCopy2KHR;

typedef struct VkCopyImageInfo2KHR {
    VkStructureType           sType;
    const void* pNext;
    VkImage                   srcImage;
    VkImageLayout             srcImageLayout;
    VkImage                   dstImage;
    VkImageLayout             dstImageLayout;
    uint32_t                  regionCount;
    const VkImageCopy2KHR* pRegions;
} VkCopyImageInfo2KHR;

typedef struct VkBufferImageCopy2KHR {
    VkStructureType             sType;
    const void* pNext;
    VkDeviceSize                bufferOffset;
    uint32_t                    bufferRowLength;
    uint32_t                    bufferImageHeight;
    VkImageSubresourceLayers    imageSubresource;
    VkOffset3D                  imageOffset;
    VkExtent3D                  imageExtent;
} VkBufferImageCopy2KHR;

typedef struct VkCopyBufferToImageInfo2KHR {
    VkStructureType                 sType;
    const void* pNext;
    VkBuffer                        srcBuffer;
    VkImage                         dstImage;
    VkImageLayout                   dstImageLayout;
    uint32_t                        regionCount;
    const VkBufferImageCopy2KHR* pRegions;
} VkCopyBufferToImageInfo2KHR;

typedef struct VkCopyImageToBufferInfo2KHR {
    VkStructureType                 sType;
    const void* pNext;
    VkImage                         srcImage;
    VkImageLayout                   srcImageLayout;
    VkBuffer                        dstBuffer;
    uint32_t                        regionCount;
    const VkBufferImageCopy2KHR* pRegions;
} VkCopyImageToBufferInfo2KHR;

typedef struct VkImageBlit2KHR {
    VkStructureType             sType;
    const void* pNext;
    VkImageSubresourceLayers    srcSubresource;
    VkOffset3D                  srcOffsets[2];
    VkImageSubresourceLayers    dstSubresource;
    VkOffset3D                  dstOffsets[2];
} VkImageBlit2KHR;

typedef struct VkBlitImageInfo2KHR {
    VkStructureType           sType;
    const void* pNext;
    VkImage                   srcImage;
    VkImageLayout             srcImageLayout;
    VkImage                   dstImage;
    VkImageLayout             dstImageLayout;
    uint32_t                  regionCount;
    const VkImageBlit2KHR* pRegions;
    VkFilter                  filter;
} VkBlitImageInfo2KHR;

typedef struct VkImageResolve2KHR {
    VkStructureType             sType;
    const void* pNext;
    VkImageSubresourceLayers    srcSubresource;
    VkOffset3D                  srcOffset;
    VkImageSubresourceLayers    dstSubresource;
    VkOffset3D                  dstOffset;
    VkExtent3D                  extent;
} VkImageResolve2KHR;

typedef struct VkResolveImageInfo2KHR {
    VkStructureType              sType;
    const void* pNext;
    VkImage                      srcImage;
    VkImageLayout                srcImageLayout;
    VkImage                      dstImage;
    VkImageLayout                dstImageLayout;
    uint32_t                     regionCount;
    const VkImageResolve2KHR* pRegions;
} VkResolveImageInfo2KHR;

typedef void (VKAPI_PTR* PFN_vkCmdCopyBuffer2KHR)(VkCommandBuffer commandBuffer, const VkCopyBufferInfo2KHR* pCopyBufferInfo);
typedef void (VKAPI_PTR* PFN_vkCmdCopyImage2KHR)(VkCommandBuffer commandBuffer, const VkCopyImageInfo2KHR* pCopyImageInfo);
typedef void (VKAPI_PTR* PFN_vkCmdCopyBufferToImage2KHR)(VkCommandBuffer commandBuffer, const VkCopyBufferToImageInfo2KHR* pCopyBufferToImageInfo);
typedef void (VKAPI_PTR* PFN_vkCmdCopyImageToBuffer2KHR)(VkCommandBuffer commandBuffer, const VkCopyImageToBufferInfo2KHR* pCopyImageToBufferInfo);
typedef void (VKAPI_PTR* PFN_vkCmdBlitImage2KHR)(VkCommandBuffer commandBuffer, const VkBlitImageInfo2KHR* pBlitImageInfo);
typedef void (VKAPI_PTR* PFN_vkCmdResolveImage2KHR)(VkCommandBuffer commandBuffer, const VkResolveImageInfo2KHR* pResolveImageInfo);

#endif // !defined(VK_KHR_copy_commands2)

#endif // _VULKAN_VK_QCOM_ROTATED_COPY_H_
