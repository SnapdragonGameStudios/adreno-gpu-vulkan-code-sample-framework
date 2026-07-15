//============================================================================================================
//
//
//                  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "memoryManager.hpp"
#include "vulkan/vulkan.hpp"
//#include "../../samples/tileMemory/code/extension/vk_qcom_tile_memory_heap.h"
#include <cassert>

//
// Include the VulkanMemoryAllocator AND compile the implementation in this cpp (is a header-only library)
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <volk/volk.h>
#include "VulkanMemoryAllocator/include/vk_mem_alloc.h"


// Ensure our enum matches the one in VMA.  This way we can pass out enum straight through to VMA (and not have to include the 700kb vma header everywhere)
static_assert(static_cast<VmaMemoryUsage>(MemoryUsage::GpuExclusive) == VMA_MEMORY_USAGE_GPU_ONLY, "enum mismatch");
static_assert(static_cast<VmaMemoryUsage>(MemoryUsage::CpuExclusive) == VMA_MEMORY_USAGE_CPU_ONLY, "enum mismatch");
static_assert(static_cast<VmaMemoryUsage>(MemoryUsage::CpuToGpu) == VMA_MEMORY_USAGE_CPU_TO_GPU, "enum mismatch");
static_assert(static_cast<VmaMemoryUsage>(MemoryUsage::GpuToCpu) == VMA_MEMORY_USAGE_GPU_TO_CPU, "enum mismatch");
static_assert(static_cast<VmaMemoryUsage>(MemoryUsage::CpuCopy) == VMA_MEMORY_USAGE_CPU_COPY, "enum mismatch");
static_assert(static_cast<VmaMemoryUsage>(MemoryUsage::GpuLazilyAllocated) == VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED, "enum mismatch");

MemoryManager<Vulkan>::MemoryManager()
{
}

///////////////////////////////////////////////////////////////////////////////

MemoryManager<Vulkan>::~MemoryManager()
{
    Destroy();
}

///////////////////////////////////////////////////////////////////////////////

bool MemoryManager<Vulkan>::Initialize(VkPhysicalDevice vkPhysicalDevice, VkDevice vkDevice, VkInstance vkInstance, bool EnableBufferDeviceAddress)
{
    assert(!mVmaAllocator);
    mGpuDevice = vkDevice;

    VmaVulkanFunctions vulkanFunctions{
    .vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)vkGetInstanceProcAddr,
    .vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr( vkInstance, "vkGetDeviceProcAddr" )
    };
    auto* fpGetDeviceProcAddr = vulkanFunctions.vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocatorInfo = {
        .flags = (VmaAllocatorCreateFlags) ( EnableBufferDeviceAddress ? VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT : 0 ),
        .physicalDevice = vkPhysicalDevice,
        .device = vkDevice,
        .pVulkanFunctions = &vulkanFunctions,
        .instance = vkInstance,
        .vulkanApiVersion = VK_MAKE_API_VERSION( 0,1,1,0 ),
    };

    VkResult result = vmaCreateAllocator(&allocatorInfo, &mVmaAllocator);
    if (result != VK_SUCCESS)
    {
        mVmaAllocator = nullptr;
        return false;
    }

    if ((allocatorInfo.flags & VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT) != 0)
    {
#if VK_KHR_buffer_device_address
    mFpGetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddressKHR)fpGetDeviceProcAddr(vkDevice, "vkGetBufferDeviceAddressKHR");
#elif VK_EXT_buffer_device_address
    mFpGetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddressEXT)fpGetDeviceProcAddr(vkDevice, "vkGetBufferDeviceAddressEXT");
#endif
    if (!mFpGetBufferDeviceAddress)
    {
        // When we go to Vulkan 1.2 switch to use vkGetBufferDeviceAddress
        assert(0 && "vkGetBufferDeviceAddressKHR/vkGetBufferDeviceAddressEXT not available, needed for VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT");
        return false;
    }
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////

void MemoryManager<Vulkan>::Destroy()
{
    vmaDestroyAllocator(mVmaAllocator);    // safe to pass nullptr
    mVmaAllocator = nullptr;
}

static VkBufferUsageFlags BufferUsageToVk(BufferUsageFlags usage)
{
    VkBufferUsageFlags vkUsage = 0;
    if ((usage & BufferUsageFlags::TransferSrc) != 0)
        vkUsage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if ((usage & BufferUsageFlags::TransferDst) != 0)
        vkUsage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if ((usage & BufferUsageFlags::Storage) != 0)
        vkUsage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if ((usage & BufferUsageFlags::Uniform) != 0)
        vkUsage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if ((usage & BufferUsageFlags::Index) != 0)
        vkUsage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if ((usage & BufferUsageFlags::Vertex) != 0)
        vkUsage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if ((usage & BufferUsageFlags::Indirect) != 0)
        vkUsage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if ((usage & BufferUsageFlags::AccelerationStructure) != 0)
        vkUsage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if ((usage & BufferUsageFlags::AccelerationStructureBuild) != 0)
        vkUsage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    if ((usage & BufferUsageFlags::ShaderBindingTable) != 0)
        vkUsage |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
    if ((usage & BufferUsageFlags::ShaderDeviceAddress) != 0)
        vkUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    static const BufferUsageFlags allKnown = BufferUsageFlags::TransferSrc | BufferUsageFlags::TransferDst | BufferUsageFlags::Storage | BufferUsageFlags::Uniform | BufferUsageFlags::Index | BufferUsageFlags::Vertex | BufferUsageFlags::Indirect | BufferUsageFlags::AccelerationStructure | BufferUsageFlags::AccelerationStructureBuild | BufferUsageFlags::ShaderBindingTable | BufferUsageFlags::ShaderDeviceAddress;
    if ((usage | allKnown) != allKnown)
    {
        assert(0 && "if check missing");
    }
    return vkUsage;
}

///////////////////////////////////////////////////////////////////////////////

MemoryAllocatedBuffer<Vulkan, VkBuffer> MemoryManager<Vulkan>::CreateBuffer(size_t size, BufferUsageFlags bufferUsage, MemoryUsage memoryUsage, VkDescriptorBufferInfo* pDescriptorBufferInfo)
{
    assert(memoryUsage != MemoryUsage::Unknown);
    VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = (VkDeviceSize)size;
    bufferInfo.usage = BufferUsageToVk(bufferUsage);

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = static_cast<VmaMemoryUsage>(memoryUsage);
    
    MemoryAllocatedBuffer<Vulkan, VkBuffer> vmaAllocatedBuffer;
    VmaAllocationInfo vmaAllocationInfo;
    VkResult result = vmaCreateBuffer(mVmaAllocator, &bufferInfo, &allocInfo, &vmaAllocatedBuffer.buffer, (VmaAllocation*)&vmaAllocatedBuffer.allocation.allocation, &vmaAllocationInfo);
    if (result != VK_SUCCESS)
    {
        return {};
    }
    if (pDescriptorBufferInfo)
    {
        *pDescriptorBufferInfo = { vmaAllocatedBuffer.GetVkBuffer(), 0, size };
    }
    return vmaAllocatedBuffer;
}

///////////////////////////////////////////////////////////////////////////////

MemoryAbhAllocatedBuffer MemoryManager<Vulkan>::CreateAndroidHardwareBuffer(size_t size, BufferUsageFlags bufferUsage, MemoryUsage memoryUsage)
{
    assert(0 && "unimplemented");
    return {};
}

///////////////////////////////////////////////////////////////////////////////

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateImageQCOM(
    VmaAllocator allocator,
    const VkImageCreateInfo* pImageCreateInfo,
    const VmaAllocationCreateInfo* pAllocationCreateInfo,
    VkImage* pImage,
    VmaAllocation* pAllocation,
    VmaAllocationInfo* pAllocationInfo )
{
    VMA_ASSERT( allocator && pImageCreateInfo && pAllocationCreateInfo && pImage && pAllocation );

    if (pImageCreateInfo->extent.width == 0 ||
         pImageCreateInfo->extent.height == 0 ||
         pImageCreateInfo->extent.depth == 0 ||
         pImageCreateInfo->mipLevels == 0 ||
         pImageCreateInfo->arrayLayers == 0)
    {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VMA_DEBUG_LOG( "vmaCreateImageQCOM" );

    VMA_DEBUG_GLOBAL_MUTEX_LOCK

        * pImage = VK_NULL_HANDLE;
    *pAllocation = VK_NULL_HANDLE;

    // 1. Create VkImage.
    VkResult res = (*allocator->GetVulkanFunctions().vkCreateImage)(
        allocator->m_hDevice,
        pImageCreateInfo,
        allocator->GetAllocationCallbacks(),
        pImage);
    if (res >= 0)
    {
        VmaSuballocationType suballocType = pImageCreateInfo->tiling == VK_IMAGE_TILING_OPTIMAL ?
            VMA_SUBALLOCATION_TYPE_IMAGE_OPTIMAL :
            VMA_SUBALLOCATION_TYPE_IMAGE_LINEAR;

        // 2. Allocate memory using allocator.
        VkMemoryRequirements vkMemReq = {};

        bool requiresDedicatedAllocation = false;
        bool prefersDedicatedAllocation = false;

        allocator->GetImageMemoryRequirements( *pImage, vkMemReq,
                                               requiresDedicatedAllocation, prefersDedicatedAllocation );

        res = allocator->AllocateMemory(
            vkMemReq,
            requiresDedicatedAllocation,
            prefersDedicatedAllocation,
            VK_NULL_HANDLE, // dedicatedBuffer
            *pImage, // dedicatedImage
            pImageCreateInfo->usage, // dedicatedBufferImageUsage
            *pAllocationCreateInfo,
            suballocType,
            1, // allocationCount
            pAllocation );

        if (res >= 0)
        {
            // 3. Bind image with memory.
            if ((pAllocationCreateInfo->flags & VMA_ALLOCATION_CREATE_DONT_BIND_BIT) == 0)
            {
                res = allocator->BindImageMemory( *pAllocation, 0, *pImage, VMA_NULL );
            }
            if (res >= 0)
            {
                // All steps succeeded.
#if VMA_STATS_STRING_ENABLED
                ( *pAllocation )->InitBufferImageUsage( pImageCreateInfo->usage );
#endif
                if (pAllocationInfo != VMA_NULL)
                {
                    allocator->GetAllocationInfo( *pAllocation, pAllocationInfo );
                }

                return VK_SUCCESS;
            }
            allocator->FreeMemory(
                1, // allocationCount
                pAllocation );
            *pAllocation = VK_NULL_HANDLE;
            (*allocator->GetVulkanFunctions().vkDestroyImage)(allocator->m_hDevice, *pImage, allocator->GetAllocationCallbacks());
            *pImage = VK_NULL_HANDLE;
            return res;
        }
        (*allocator->GetVulkanFunctions().vkDestroyImage)(allocator->m_hDevice, *pImage, allocator->GetAllocationCallbacks());
        *pImage = VK_NULL_HANDLE;
        return res;
    }
    return res;
}

///////////////////////////////////////////////////////////////////////////////

static PFN_vkGetImageMemoryRequirements2KHR s_oldGetImageMemReq = nullptr;
void vkGetImageMemoryRequirements2KHR_Qcom( VkDevice device, const VkImageMemoryRequirementsInfo2KHR* memReqInfo, VkMemoryRequirements2KHR* memReq2 )
{
    VkTileMemoryRequirementsQCOM memReqQCOM {
        .sType = VK_STRUCTURE_TYPE_TILE_MEMORY_REQUIREMENTS_QCOM,
        .pNext = memReq2->pNext
    };
    memReq2->pNext = &memReqQCOM;
    s_oldGetImageMemReq(device, memReqInfo, memReq2);

    if (true)
    {
        LOGI("vkGetImageMemoryRequirements2KHR_Qcom image size: %zu (vs %zu for non tiled);  alignment : %zu (vs %zu)  memory type bits: %d", memReqQCOM.size, memReq2->memoryRequirements.size, memReqQCOM.alignment, memReq2->memoryRequirements.alignment, memReq2->memoryRequirements.memoryTypeBits);
        //// Round alignment to next power of 2 (Qualcomm driver may not return a power of 2 alignment)
        uint32_t alignment = memReqQCOM.alignment;
        assert((memReqQCOM.size % alignment ) == 0);    // make sure the buffer's size is aligned, we are going to have to assume everything can be laid out in order!
        memReq2->memoryRequirements.alignment = 1;// alignment;
        memReq2->memoryRequirements.size = memReqQCOM.size;

        // memReq2->memoryRequirements.memoryTypeBits = memReqQCOM.memoryTypeBits;
    }
    memReq2->pNext = const_cast<void*>(memReqQCOM.pNext);
}

///////////////////////////////////////////////////////////////////////////////

MemoryAllocatedBuffer<Vulkan, VkImage> MemoryManager<Vulkan>::CreateImage(const VkImageCreateInfo& imageInfo, MemoryUsage memoryUsage)
{
    assert(memoryUsage != MemoryUsage::Unknown);
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = static_cast<VmaMemoryUsage>(memoryUsage);

    MemoryAllocatedBuffer<Vulkan, VkImage> vmaAllocatedImage;

    VkResult result = vmaCreateImage(mVmaAllocator, &imageInfo, &allocInfo, &vmaAllocatedImage.buffer, (VmaAllocation*)&vmaAllocatedImage.allocation.allocation, nullptr);

    if (result != VK_SUCCESS)
    {
        return {};
    }
    return vmaAllocatedImage;
}

///////////////////////////////////////////////////////////////////////////////

MemoryAllocatedBuffer<Vulkan, VkDeviceMemory> MemoryManager<Vulkan>::AllocateMemory(size_t size, uint32_t memoryTypeBits)
{
    VkMemoryRequirements memoryRequirements{ .size = size, .memoryTypeBits = memoryTypeBits };

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_UNKNOWN;
    allocInfo.memoryTypeBits = memoryTypeBits;
    allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    MemoryAllocatedBuffer<Vulkan, VkDeviceMemory> vmaAllocatedMemory;
    VkResult result = vmaAllocateMemory(mVmaAllocator, &memoryRequirements, &allocInfo, (VmaAllocation*)&vmaAllocatedMemory.allocation.allocation, nullptr);
    if (result != VK_SUCCESS)
    {
        return {};
    }
    vmaAllocatedMemory.buffer = static_cast<VmaAllocation>(vmaAllocatedMemory.allocation.allocation)->GetMemory();
    return vmaAllocatedMemory;
}

///////////////////////////////////////////////////////////////////////////////

MemoryAllocatedBuffer<Vulkan, VkImage> MemoryManager<Vulkan>::BindImageToMemory(VkImage image, MemoryAllocatedBuffer<Vulkan, VkDeviceMemory>&& memory) const
{
    VkResult result = vkBindImageMemory(mGpuDevice, image, static_cast<VmaAllocation>(memory.allocation.allocation)->GetMemory(), static_cast<VmaAllocation>(memory.allocation.allocation)->GetOffset());
    if (result != VK_SUCCESS)
    {
        return {};
    }
    MemoryAllocatedBuffer<Vulkan, VkImage> imageMemory;
    imageMemory.allocation = std::move(memory.allocation);
    imageMemory.buffer = image;
    return imageMemory;
}

///////////////////////////////////////////////////////////////////////////////

MemoryAllocatedBuffer<Vulkan, VkBuffer> MemoryManager<Vulkan>::BindBufferToMemory(VkBuffer buffer, MemoryAllocatedBuffer<Vulkan, VkDeviceMemory>&& memory) const
{
    VkResult result = vkBindBufferMemory(mGpuDevice, buffer, static_cast<VmaAllocation>(memory.allocation.allocation)->GetMemory(), static_cast<VmaAllocation>(memory.allocation.allocation)->GetOffset());
    if (result != VK_SUCCESS)
    {
        return {};
    }
    MemoryAllocatedBuffer<Vulkan, VkBuffer> bufferMemory;
    bufferMemory.allocation = std::move(memory.allocation);
    bufferMemory.buffer = buffer;
    return bufferMemory;
}

///////////////////////////////////////////////////////////////////////////////

bool MemoryManager<Vulkan>::CopyData( VkCommandBuffer vkCommandBuffer, const MemoryAllocatedBuffer<Vulkan, VkBuffer>& src, MemoryAllocatedBuffer<Vulkan, VkBuffer>& dst, size_t copySize, size_t srcOffset, size_t dstOffset)
{
    VkBufferCopy copyRegion {};
    copyRegion.srcOffset = srcOffset;
    copyRegion.dstOffset = dstOffset;
    copyRegion.size = copySize;
    mVmaAllocator->GetVulkanFunctions().vkCmdCopyBuffer( vkCommandBuffer, src.buffer, dst.buffer, 1, &copyRegion );
    return true;
}

///////////////////////////////////////////////////////////////////////////////

void MemoryManager<Vulkan>::Destroy(MemoryAllocatedBuffer<Vulkan, VkBuffer> vmaAllocatedBuffer)
{
    assert(mVmaAllocator);
    vmaDestroyBuffer(mVmaAllocator, vmaAllocatedBuffer.buffer, static_cast<VmaAllocation>(vmaAllocatedBuffer.allocation.allocation));
    // Set the allocated buffer to a clean (deletable) state.
    vmaAllocatedBuffer.allocation.clear();
    vmaAllocatedBuffer.buffer = VK_NULL_HANDLE;
    // no error code from vmaDestroyBuffer so assume it worked (is also safe to pass an empty vmaAllocatedBuffer)
}

///////////////////////////////////////////////////////////////////////////////

void MemoryManager<Vulkan>::Destroy(MemoryAllocatedBuffer<Vulkan, VkImage> vmaAllocatedImage)
{
    assert(mVmaAllocator);
    vmaDestroyImage(mVmaAllocator, vmaAllocatedImage.buffer, static_cast<VmaAllocation>(vmaAllocatedImage.allocation.allocation));
    // Set the allocated buffer to a clean (deletable) state.
    vmaAllocatedImage.allocation.clear();
    vmaAllocatedImage.buffer = VK_NULL_HANDLE;
    // no error code from vmaDestroyImage so assume it worked (is also safe to pass an empty vmaAllocatedImage)
}

///////////////////////////////////////////////////////////////////////////////

void MemoryManager<Vulkan>::Destroy(MemoryAllocatedBuffer<Vulkan, VkDeviceMemory> vmaAllocatedMemory)
{
    assert(mVmaAllocator);
    vmaFreeMemory(mVmaAllocator, (VmaAllocation_T*)vmaAllocatedMemory.allocation.allocation);
    // Set the allocated buffer to a clean (deletable) state.
    vmaAllocatedMemory.allocation.clear();
    vmaAllocatedMemory.buffer = VK_NULL_HANDLE;
}

///////////////////////////////////////////////////////////////////////////////

MemoryAllocatedBuffer<Vulkan, VkDeviceMemory> MemoryManager<Vulkan>::DestroyBufferOnly(MemoryAllocatedBuffer<Vulkan, VkBuffer> vmaAllocatedBuffer)
{
    assert(mVmaAllocator);
    vkDestroyBuffer(mGpuDevice, vmaAllocatedBuffer.buffer, nullptr);
    MemoryAllocatedBuffer<Vulkan, VkDeviceMemory> allocatedMemory;
    allocatedMemory.buffer = static_cast<VmaAllocation>(vmaAllocatedBuffer.allocation.allocation)->GetMemory();
    allocatedMemory.allocation = std::move(vmaAllocatedBuffer.allocation);
    return allocatedMemory;
}

///////////////////////////////////////////////////////////////////////////////

MemoryAllocatedBuffer<Vulkan, VkDeviceMemory> MemoryManager<Vulkan>::DestroyBufferOnly(MemoryAllocatedBuffer<Vulkan, VkImage> vmaAllocatedImage)
{
    assert(mVmaAllocator);
    vkDestroyImage(mGpuDevice, vmaAllocatedImage.buffer, nullptr);
    MemoryAllocatedBuffer<Vulkan, VkDeviceMemory> allocatedMemory;
    allocatedMemory.buffer = static_cast<VmaAllocation>(vmaAllocatedImage.allocation.allocation)->GetMemory();
    allocatedMemory.allocation = std::move(vmaAllocatedImage.allocation);
    return allocatedMemory;
}

///////////////////////////////////////////////////////////////////////////////

VkDeviceAddress MemoryManager<Vulkan>::GetBufferDeviceAddressInternal(VkBuffer buffer) const
{
    assert(mFpGetBufferDeviceAddress != nullptr);    // need EnableBufferDeviceAddress parameter to be set on Initialize

    VkBufferDeviceAddressInfoEXT bufferAddressInfo = {};
    bufferAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT;
    bufferAddressInfo.buffer = buffer;
    return mFpGetBufferDeviceAddress(mGpuDevice, &bufferAddressInfo);
}

///////////////////////////////////////////////////////////////////////////////

void MemoryManager<Vulkan>::MapInternal(void* vmaAllocation, void** outCpuLocation)
{
    assert(outCpuLocation != nullptr);
    assert(*outCpuLocation == nullptr);    // mapped twice?
    assert(vmaAllocation != nullptr);    // Likely caused by a double mapping of this data
    if (vmaMapMemory(mVmaAllocator, static_cast<VmaAllocation>(vmaAllocation), outCpuLocation) != VK_SUCCESS)
    {
        outCpuLocation = nullptr;
        assert(0);
    }
}

///////////////////////////////////////////////////////////////////////////////

void MemoryManager<Vulkan>::UnmapInternal(void* vmaAllocation, void* cpuLocation)
{
    assert(cpuLocation);
    vmaUnmapMemory(mVmaAllocator, static_cast<VmaAllocation>(vmaAllocation));
}

///////////////////////////////////////////////////////////////////////////////

MemoryPool<Vulkan> MemoryManager<Vulkan>::CreateCustomPool( uint32_t typeIndex, uint32_t poolAllocationSize, uint32_t maxPoolAllocations, uint32_t bufferUsageFlag, uint32_t imageUsageFlag ) const
{
    VmaPoolCreateInfo poolCreateInfo{
        .memoryTypeIndex = typeIndex,
        .blockSize = poolAllocationSize,
        //.flags,
        .maxBlockCount = maxPoolAllocations,
        .pMemoryAllocateNext = nullptr
    };

    VmaPool vmaPool{};
    if (vmaCreatePool( mVmaAllocator, &poolCreateInfo, &vmaPool ) != VK_SUCCESS)
        return {};
    return {vmaPool, bufferUsageFlag, imageUsageFlag};
}

///////////////////////////////////////////////////////////////////////////////

MemoryAllocatedBuffer<Vulkan, VkDeviceMemory> MemoryManager<Vulkan>::AllocateMemory( MemoryPool<Vulkan>& pool, size_t size, uint32_t memoryTypeBits )
{
    VkMemoryRequirements memoryRequirements{.size = size, .memoryTypeBits = memoryTypeBits};

    VmaAllocationCreateInfo allocInfo{
        .pool = (VmaPool)pool.pool
    };

    MemoryAllocatedBuffer<Vulkan, VkDeviceMemory> vmaAllocatedMemory;
    VmaAllocationInfo retInfo{};
    VkResult result = vmaAllocateMemory( mVmaAllocator, &memoryRequirements, &allocInfo, (VmaAllocation*)&vmaAllocatedMemory.allocation.allocation, &retInfo );
    if (result != VK_SUCCESS)
    {
        return {};
    }
    vmaAllocatedMemory.buffer = static_cast<VmaAllocation>(vmaAllocatedMemory.allocation.allocation)->GetMemory();

    LOGI("Allocated memory from pool.  size=%zu offset=%zu memoryType=0x%x", (size_t) retInfo.size, (size_t) retInfo.offset, (uint32_t) retInfo.memoryType);
    return vmaAllocatedMemory;
}

///////////////////////////////////////////////////////////////////////////////

MemoryAllocatedBuffer<Vulkan, VkBuffer> MemoryManager<Vulkan>::CreateBuffer( MemoryPool<Vulkan>& pool, size_t size, BufferUsageFlags bufferUsage, VkDescriptorBufferInfo* pDescriptorBufferInfo )
{
    assert( pool );
    static_assert(sizeof( VkBufferUsageFlags ) == sizeof( decltype(pool.bufferUsageFlag) ));
    const VkBufferCreateInfo bufferInfo {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = (VkDeviceSize)size,
        .usage = BufferUsageToVk( bufferUsage ) | pool.bufferUsageFlag
    };

    const VmaAllocationCreateInfo allocInfo = {
        .pool = (VmaPool)pool.pool
    };

    MemoryAllocatedBuffer<Vulkan, VkBuffer> vmaAllocatedBuffer;
    VmaAllocationInfo vmaAllocationInfo;
    VkResult result = vmaCreateBuffer( mVmaAllocator, &bufferInfo, &allocInfo, &vmaAllocatedBuffer.buffer, (VmaAllocation*)&vmaAllocatedBuffer.allocation.allocation, &vmaAllocationInfo );
    if (result != VK_SUCCESS)
    {
        return {};
    }
    if (pDescriptorBufferInfo)
    {
        *pDescriptorBufferInfo = {vmaAllocatedBuffer.GetVkBuffer(), 0, size};
    }
    LOGI( "Successfully called vmaCreateBuffer with pool.  size=%zu offset=%zu memoryType=0x%x", (size_t)vmaAllocationInfo.size, (size_t)vmaAllocationInfo.offset, (uint32_t)vmaAllocationInfo.memoryType );
    return vmaAllocatedBuffer;
}

///////////////////////////////////////////////////////////////////////////////

MemoryAllocatedBuffer<Vulkan, VkImage> MemoryManager<Vulkan>::CreateImage( MemoryPool<Vulkan>& pool, const VkImageCreateInfo& imageInfo )
{
    assert( pool );
    const VmaAllocationCreateInfo allocInfo {
        .pool = (VmaPool) pool.pool
    };
    // Add in any flags specified with this pool
    auto imageInfoCopy = imageInfo;
    static_assert(sizeof( VkImageUsageFlags ) == sizeof( decltype(pool.imageUsageFlag) ));
    imageInfoCopy.usage |= (VkImageUsageFlags) pool.imageUsageFlag;





    MemoryAllocatedBuffer<Vulkan, VkImage> vmaAllocatedImage;
    VmaAllocationInfo vmaAllocationInfo;

    // Swap in our stub of vkGetImageMemoryRequirements2KHR
    VmaVulkanFunctions& vmaVulkanFunctions = const_cast<VmaVulkanFunctions&>(mVmaAllocator->GetVulkanFunctions());
    s_oldGetImageMemReq = vmaVulkanFunctions.vkGetImageMemoryRequirements2KHR;
    vmaVulkanFunctions.vkGetImageMemoryRequirements2KHR = vkGetImageMemoryRequirements2KHR_Qcom;
    // Call vmaCreateImage (will use our stub to get memory requirements)
    VkResult result = vmaCreateImage( mVmaAllocator, &imageInfoCopy, &allocInfo, &vmaAllocatedImage.buffer, (VmaAllocation*)&vmaAllocatedImage.allocation.allocation, &vmaAllocationInfo );
    // Put vkGetImageMemoryRequirements2KHR back to how it was!
    vmaVulkanFunctions.vkGetImageMemoryRequirements2KHR = s_oldGetImageMemReq;

    if (result != VK_SUCCESS)
    {
        return {};
    }
    LOGI( "Successfully called vmaCreateImageQCOM with pool.  size=%zu offset=%zu memoryType=0x%x format=%s", (size_t)vmaAllocationInfo.size, (size_t)vmaAllocationInfo.offset, (uint32_t)vmaAllocationInfo.memoryType, Vulkan::VulkanFormatString(imageInfo.format) );
    return vmaAllocatedImage;
}

///////////////////////////////////////////////////////////////////////////////

VkDeviceMemory MemoryManager<Vulkan>::GetVkDeviceMemory( const MemoryAllocation<Vulkan>& alloc ) const
{
    return static_cast<VmaAllocation>(alloc.allocation)->GetMemory();
}

