//============================================================================================================
//
//
//                  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include "graphicsApi/graphicsApiBase.hpp"

#ifdef OS_WINDOWS
#define NOMINMAX
#include <Windows.h>
#elif OS_ANDROID
#endif // OS_WINDOWS | OS_WINDOWS

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <volk/volk.h>
#include "extension.hpp"
#include "memory/vulkan/memoryManager.hpp"
#include "texture/textureFormat.hpp"
#include "framebuffer.hpp"
#include "../material/pipeline.hpp"///TODO: move pipeline.[ch]pp
#include "renderPass.hpp"

// This should actually be defined in the makefile!
#define USES_VULKAN_DEBUG_LAYERS

#define NUM_VULKAN_BUFFERS      8   // Kept track of with mSwapchainCurrentIdx

// Comment this in if you need AHB support in your app (should be part of application configuration in the future).  May interfere with profiling!
//#define ANDROID_HARDWARE_BUFFER_SUPPORT

// Forward declarations
#if OS_ANDROID
struct ANativeWindow;
#elif defined(OS_LINUX)
struct GLFWwindow;
#endif // OS_ANDROID
template<typename T_GFXAPI> class IndexBuffer;
template<typename T_GFXAPI> class VertexBuffer;
template<typename T_GFXAPI> class RenderContext;
template<typename T_GFXAPI> class RenderPass;
class RenderPassClearData;
struct VulkanDeviceFeaturePrint;
struct VulkanDevicePropertiesPrint;
struct VulkanInstanceFunctionPointerLookup;
struct VulkanDeviceFunctionPointerLookup;
namespace ExtensionLib {
    struct Ext_VK_KHR_surface;
    struct Ext_VK_KHR_get_physical_device_properties2;
    struct Ext_VK_KHR_get_surface_capabilities2;
    struct Ext_VK_KHR_draw_indirect_count;
    struct Ext_VK_KHR_swapchain;
    struct Ext_VK_EXT_debug_utils;
    struct Ext_VK_EXT_debug_marker;
    struct Ext_VK_EXT_hdr_metadata;
    struct Ext_VK_KHR_fragment_shading_rate;
    struct Ext_VK_KHR_create_renderpass2;
    struct Ext_VK_KHR_buffer_device_address;
    struct Ext_VK_KHR_8bit_storage;
    struct Ext_VK_KHR_cooperative_matrix;
    struct Ext_VK_KHR_synchronization2;
    struct Ext_VK_QCOM_tile_properties;
    struct Ext_VK_QCOM_tile_shading;
    struct Ext_VK_QCOM_tile_memory_heap;
    struct Ext_VK_KHR_get_memory_requirements2;
    struct Ext_VK_ARM_tensors;
    struct Ext_VK_ARM_data_graph;
    struct Ext_VK_QCOM_data_graph_model;
    struct Vulkan_SubgroupPropertiesHook;
    struct Vulkan_StorageFeaturesHook;
    struct Ext_VK_KHR_mesh_shader;
    struct Ext_VK_KHR_dynamic_rendering;
    struct Ext_VK_KHR_external_memory_capabilities;
    struct Ext_VK_KHR_external_memory;
    struct Ext_VK_KHR_external_semaphore_capabilities;
    struct Ext_VK_KHR_external_semaphore;
    struct Ext_VK_KHR_external_memory_fd;
    struct Ext_VK_KHR_external_semaphore_fd;
    struct Ext_VK_KHR_external_memory_win32;
    struct Ext_VK_KHR_external_semaphore_win32;
};

namespace vk {};
class VulkanDebugCallback;
enum class TextureFormat;
enum class Msaa;

bool CheckVkError(const char* pPrefix, VkResult CheckVal);

//=============================================================================
// Structures
//=============================================================================

/// Single Swapchain image
struct SwapchainBuffers
{
    SwapchainBuffers() noexcept = default;
    SwapchainBuffers( SwapchainBuffers&& ) noexcept = default;
    Framebuffer<Vulkan>                     framebuffer;
    VkImage                                 image       = VK_NULL_HANDLE;
    VkImageView                             view        = VK_NULL_HANDLE;
    VkFence                                 fence       = VK_NULL_HANDLE;
    VkSemaphore                             semaphore   = VK_NULL_HANDLE;
};

/// DepthBuffer memory, image and view
typedef struct _DepthInfo
{
    TextureFormat                           format;
    VkImageView                             view;
    MemoryAllocatedBuffer<Vulkan, VkImage>  image;
} DepthInfo;

typedef struct _SurfaceFormat
{
    TextureFormat                           format;
    VkColorSpaceKHR                         colorSpace;
} SurfaceFormat;


//
// Wrapper template helpers for VkStructs
// in namespace fvk (framework vk).
//
namespace fvk
{
    template<typename, VkStructureType> struct VkStructWrapperNext;                             // forward declaration
    template<typename, typename, size_t>          struct VkStructWrapperMemberPtr;              // forward declaration
    template<typename, typename, size_t, typename, size_t> struct VkStructWrapperMemberArrayPtr;// forward declaration
    //#define offsetof(s,m) ((::size_t)&reinterpret_cast<char const volatile&>((((s*)0)->m)))
    //#define OwnedPointer(s,m) const_cast<decltype(s::m)>((((s*)0)->m))

    /// @brief Helper for creating Vulkan api structures (eg for create parameters) with chaining via pNext.
    /// @tparam VK_STRUCTURE Type of structure to create
    /// @tparam T_STRUCTURE_TYPE Enum matching the structure (from VkStructureType)

    template<typename VK_STRUCTURE, VkStructureType T_STRUCTURE_TYPE>
    struct VkStructWrapperBase {
        VkStructWrapperBase( const VkStructWrapperBase& ) = delete;
        VkStructWrapperBase& operator=( const VkStructWrapperBase& ) noexcept = delete;
        VkStructWrapperBase( VkStructWrapperBase&& other ) noexcept
        {
            *this = std::move( other );
        }
        VkStructWrapperBase& operator=( VkStructWrapperBase&& other ) noexcept
        {
            if (this != std::addressof(other))
            {
                // ugh!  We can rely on these being C (vulkan) structures so no constructors, destructors
                memcpy( this, &other, sizeof( *this ) );
                memset( &other, 0, sizeof( *this ) );
                other.s.sType = T_STRUCTURE_TYPE;
            }
            return *this;
        }

        VkStructWrapperBase( VK_STRUCTURE&& contents = {} ) noexcept : s( std::forward<VK_STRUCTURE>( contents ) )
        {
            static_assert(sizeof( *this ) == sizeof( VK_STRUCTURE ));
            assert( contents.pNext == nullptr );
            //allow us to pass contents where sType is not set, compiler will have checked it is correct structure type so assume the data is good.
            //this is nice because we can then use designated initializers to make things look nice
            //assert( contents.sType == T_STRUCTURE_TYPE );
            s.sType = T_STRUCTURE_TYPE;
        }
        VkStructWrapperBase<VK_STRUCTURE, T_STRUCTURE_TYPE> & operator=( VK_STRUCTURE&& other ) noexcept {
            if (&other != this) {
                assert( other.sType == T_STRUCTURE_TYPE );
                s = std::forward<VK_STRUCTURE>( other );
                // ugh!  We can rely on these being (vulkan) structures so no constructors, destructors
                memset( &other, 0, sizeof( *this ) );
            }
            return *this;
        };
        VK_STRUCTURE* operator&()               { return &(this->s); }    // This is 'unconventional'; if we start needing to use std::addressof then consider removing this and replacing with .get().  Saving grace may be that this is a non virtual class and only contains 's'
        const VK_STRUCTURE* operator&() const   { return &(this->s); }    // This is 'unconventional'; if we start needing to use std::addressof then consider removing this and replacing with .get().  Saving grace may be that this is a non virtual class and only contains 's'

        using tStruct = VK_STRUCTURE;
        static constexpr VkStructureType tStructType = T_STRUCTURE_TYPE;
        VK_STRUCTURE s{.sType = T_STRUCTURE_TYPE};
    };

    template<typename VK_STRUCTURE, VkStructureType T_STRUCTURE_TYPE, std::pair<size_t, size_t>... T_MEMBER_PTRS>
    struct VkStructWrapperWithMembers : public VkStructWrapperBase<VK_STRUCTURE, T_STRUCTURE_TYPE>
    {
        using tBase = VkStructWrapperBase<VK_STRUCTURE, T_STRUCTURE_TYPE>;
        VkStructWrapperWithMembers( const VkStructWrapperWithMembers& ) = delete;
        VkStructWrapperWithMembers& operator=( const VkStructWrapperWithMembers& ) = delete;
        VkStructWrapperWithMembers( VK_STRUCTURE&& contents ) noexcept : tBase( std::move( contents ) ) {}
        VkStructWrapperWithMembers( VkStructWrapperWithMembers&& other ) noexcept
        {
            *this = std::move(other);
        }
        VkStructWrapperWithMembers& operator=( VkStructWrapperWithMembers&& other ) noexcept
        {
            if (this != std::addressof(other))
            {
                tBase::operator=( std::move( static_cast<tBase&&>(other) ) );
                mMemberPointers = std::move( other.mMemberPointers );
                std::fill( std::begin( other.mMemberPointers ), std::end( other.mMemberPointers ), nullptr );
            }
            return *this;
        }
        ~VkStructWrapperWithMembers()
        {
            freemembers<T_MEMBER_PTRS...>();
        }
        template<typename T_MEMBER, size_t T_POINTEROFFSET>
        auto AddMember()
        {
            return AddMemberArray<T_MEMBER, T_POINTEROFFSET, size_t(0)>(1).data();
        }
        template<typename T_MEMBER, size_t T_POINTEROFFSET, size_t T_COUNTOFFSET>
        auto AddMemberArray( uint32_t arrayCount )
        {
            auto member = findmember<T_POINTEROFFSET, T_MEMBER_PTRS...>();
            void** pMember = reinterpret_cast<void**>(reinterpret_cast<uint8_t*>(&(this->s)) + member.offset);
            T_MEMBER* p = (T_MEMBER*)calloc(sizeof(T_MEMBER), arrayCount); //yes, calloc not new[] because we dont have T_MEMEBER when freeing (freemembers()) so cant do delete[].
            static_assert(std::is_trivially_default_constructible_v< T_MEMBER>);    // check that T_MEMBER is constructable with calloc (the Vulkan structs should all pass this check).  If you hit this at compile time then you are trying to add a member that is not compatible with AddMemberArray/AddMember)
            *pMember = p;
            mMemberPointers[member.pointerIndex] = p;
            if (member.countOffset > 0/*0 would be the offset of sType, so ok to use to indicate 'no count' */)
            {
                uint32_t* pCountMember = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&(this->s)) + member.countOffset);
                *pCountMember = arrayCount;
            }
            else
            {
                assert(arrayCount == 1);    // if there is no count member of the structure then we expect the pointer to just be to a single value.
            }
            return std::span( p, arrayCount );
        }
    protected:
        std::array<void*, sizeof...(T_MEMBER_PTRS)> mMemberPointers{};    // pointers to data created by AddMember (duplicate of that pointer).  Will be nullptr in the case where the member pointer was set but wasnt set via AddMember (and so we dont have ownership of the memory)

        struct Member {
            size_t offset;          //byte offset of pointer in VK_STRUCTURE
            size_t countOffset;     //byte offset of associated count(er) in VK_STRUCTURE
            size_t pointerIndex;    //index into mMemberPointers
        };
        template<size_t TT_POINTEROFFSET, std::pair<size_t, size_t> TT_MEMBER_PTR, std::pair<size_t, size_t>... TT_MEMBER_PTRS>
        static consteval auto findmember( size_t memberIdx = 0 )
        {
            if constexpr (TT_POINTEROFFSET == TT_MEMBER_PTR.first)
                return Member{TT_MEMBER_PTR.first, TT_MEMBER_PTR.second, memberIdx};
            else if constexpr (sizeof...(TT_MEMBER_PTRS) != 0)
                return findmember<TT_POINTEROFFSET, TT_MEMBER_PTRS...>( memberIdx + 1 );
            else
            {
                static_assert((sizeof...(TT_MEMBER_PTRS) != 0) && "Cannot find pointer member in structure");
            }
            return Member{};
        }

        template<std::pair<size_t, size_t> TT_MEMBER_PTR, std::pair<size_t, size_t>... TT_MEMBER_PTRS>
        void freemembers( size_t memberIdx = 0 )
        {
            if constexpr (sizeof...(TT_MEMBER_PTRS) != 0)
            {
                freemembers<TT_MEMBER_PTRS...>( memberIdx + 1 );
            }
            else
            {
                void** p = reinterpret_cast<void**>(reinterpret_cast<uint8_t*>(&(this->s)) + TT_MEMBER_PTR.first);
                if (*p && this->mMemberPointers[memberIdx])
                {
                    assert( *p == mMemberPointers[memberIdx] );
                    free(*p);   // was allocated using calloc! (and checked to be trivially constructable
                }
            }
        };
    };

    template<typename VK_STRUCTURE, VkStructureType T_STRUCTURE_TYPE, std::pair<size_t, size_t>... T_MEMBER_PTRS>
    struct VkStructWrapper : public std::conditional_t<sizeof...(T_MEMBER_PTRS) != 0, VkStructWrapperWithMembers<VK_STRUCTURE, T_STRUCTURE_TYPE, T_MEMBER_PTRS...>, VkStructWrapperBase<VK_STRUCTURE, T_STRUCTURE_TYPE>>
    {
        using tStruct = VK_STRUCTURE;
        using tBase = std::conditional_t<sizeof...(T_MEMBER_PTRS) != 0, VkStructWrapperWithMembers<VK_STRUCTURE, T_STRUCTURE_TYPE, T_MEMBER_PTRS...>, VkStructWrapperBase<VK_STRUCTURE, T_STRUCTURE_TYPE>>;
        VkStructWrapper( VkStructWrapper&& other ) noexcept : tBase( std::move( other ) )
        {}
        VkStructWrapper& operator=( VkStructWrapper&& other ) noexcept = default;

        VkStructWrapper( tStruct&& contents = {} ) noexcept : tBase( std::move( contents ) )
        {
        }
        VkStructWrapper& operator=( tStruct&& other ) noexcept {
            tBase::operator=( std::move( other ) );
            return *this;
        }
        tStruct& operator*() { return get(); }
        tStruct* operator->() { return &get(); }
        const tStruct& get() const { return this->s; }
        tStruct& get() { return this->s; }
        const tStruct& operator*() const { return get(); }
        const tStruct* operator->() const { return &get(); }
        tStruct* operator&() { return tBase::operator&(); }    // This is 'unconventional'; if we start needing to use std::addressof then consider removing this and replacing with .get().  Saving grace may be that this is a non virtual class and only contains 's'
        const tStruct* operator&() const { return tBase::operator&(); }    // This is 'unconventional'; if we start needing to use std::addressof then consider removing this and replacing with .get().  Saving grace may be that this is a non virtual class and only contains 's'
        template<typename VK_STRUCTURE_NEXT, VkStructureType T_STRUCTURE_TYPE_NEXT>
        auto& Add() {
            auto* p = new VkStructWrapperNext<VK_STRUCTURE_NEXT, T_STRUCTURE_TYPE_NEXT>();
            p->s.pNext = this->s.pNext;
            this->s.pNext = p;
            return *p;
        }
        template<typename VK_STRUCTURE_NEXT, VkStructureType T_STRUCTURE_TYPE_NEXT>
        auto& Add( VK_STRUCTURE_NEXT&& contents ) {
            auto* p = new VkStructWrapperNext<VK_STRUCTURE_NEXT, T_STRUCTURE_TYPE_NEXT>( std::forward< VK_STRUCTURE_NEXT>( contents ) );
            p->s.pNext = this->s.pNext;
            this->s.pNext = p;
            return *p;
        }
        template<typename VK_STRUCTURE_WRAPPER_NEXT>
        auto& Add( VK_STRUCTURE_WRAPPER_NEXT&& contents ) {
            auto* p = new std::remove_reference_t<VK_STRUCTURE_WRAPPER_NEXT>( std::move( contents.s ) );
            p->s.pNext = (void*)this->s.pNext;
            this->s.pNext = p;
            return *p;
        }

        ~VkStructWrapper() {
            VkBaseOutStructure* pNext = (VkBaseOutStructure*)this->s.pNext;
            while (pNext)
            {
                auto* pNextPrev = pNext;
                pNext = pNext->pNext;
                free( pNextPrev );
            }
            this->s.pNext = nullptr;
        }
        template<typename, VkStructureType> friend struct VkStructWrapperNext;
    };
    // Template for the structs chained to VkStructWrapper by calling Add

    /// @brief Helper class for vulkan structures chained (via pNext) onto a base VkStructWrapper
    template<typename VK_STRUCTURE, VkStructureType T_STRUCTURE_TYPE>
    struct VkStructWrapperNext final : public VkStructWrapper<VK_STRUCTURE, T_STRUCTURE_TYPE>
    {
        VkStructWrapperNext() noexcept : VkStructWrapper<VK_STRUCTURE, T_STRUCTURE_TYPE>() {}
        VkStructWrapperNext( VK_STRUCTURE&& contents ) noexcept : VkStructWrapper<VK_STRUCTURE, T_STRUCTURE_TYPE>( std::forward< VK_STRUCTURE>( contents ) )
        {}
    private:
        // Destructor is private because we shouldnt construct/destruct these classes outside of VkStructWrapper::Add (and then the ~VkStructWrapper will cleanup the VkStructWrapperChild classes)
        ~VkStructWrapperNext() noexcept { this->s.pNext = nullptr;/*detach from list so the base class destructor doesnt delete the pNext*/ }
    };

    //
    // Set of aliases for common VkStructs wrapped in our helper
    //
    using VkRenderPassCreateInfo = VkStructWrapper<VkRenderPassCreateInfo, VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO>;
    using VkRenderPassCreateInfo2 = VkStructWrapper<VkRenderPassCreateInfo2, VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2>;
    using VkRenderPassBeginInfo = VkStructWrapper < VkRenderPassBeginInfo, VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, std::pair{offsetof(VkRenderPassBeginInfo, pClearValues), offsetof(VkRenderPassBeginInfo, clearValueCount)} > ;
    using VkFramebufferCreateInfo = VkStructWrapper<VkFramebufferCreateInfo, VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO>;
    using VkRenderPassTransformBeginInfoQCOM = VkStructWrapper<VkRenderPassTransformBeginInfoQCOM, VK_STRUCTURE_TYPE_RENDER_PASS_TRANSFORM_BEGIN_INFO_QCOM>;
    using VkRenderingInfo = VkStructWrapper < VkRenderingInfo, VK_STRUCTURE_TYPE_RENDERING_INFO, std::pair{offsetof( VkRenderingInfo, pColorAttachments ), offsetof( VkRenderingInfo, colorAttachmentCount )}, std::pair{offsetof(VkRenderingInfo, pDepthAttachment), 0}, std::pair{offsetof(VkRenderingInfo, pStencilAttachment),0} >;
    using VkPipelineRenderingCreateInfo = VkStructWrapper < VkPipelineRenderingCreateInfo, VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, std::pair{offsetof(VkPipelineRenderingCreateInfo, pColorAttachmentFormats), offsetof(VkPipelineRenderingCreateInfo, colorAttachmentCount)} > ;

} // namespace fvk;


/// Vulkan API implementation
/// Contains Vulkan top level (driver etc) objects and provides a simple initialization interface.
class Vulkan : public ::GraphicsApiBase
{
    Vulkan(const Vulkan&) = delete;
    Vulkan& operator=(const Vulkan&) = delete;
public:
    // typedefs
    using ViewportClass = VkViewport;
    using Rect2DClass = VkRect2D;
    using MemoryManager = MemoryManager<Vulkan>;
    using RenderContext = RenderContext<Vulkan>;
    using RenderPass = RenderPass<Vulkan>;
    using BufferHandleType = VkBuffer;

public:
    Vulkan();
    virtual ~Vulkan();

    struct AppConfiguration
    {
        /// (optional) Vulkan api version this app would like.  If not set the framework will default to whatever version it choses.
        std::optional<uint32_t> ApiVerson;
        /// (optional) override of the priority used to initialize the async queue (assuming VK_EXT_global_priority is available and loaded)
        std::optional<VkQueueGlobalPriorityEXT> AsyncQueuePriority;
        /// (optional) override of the framebuffer depth format.  Setting to VK_FORMAT_UNDEFINED will disable the creation of a depth buffer during InitSwapChain
        std::optional<TextureFormat> SwapchainDepthFormat;
        /// (optional) presentation mode that this app would like.  If the requested mode is not available (or not requested) a 'fastest' default will be used.
        std::optional<VkPresentModeKHR> PresentMode;

        /// @brief Register the vulkan extension (templated) as required by this app
        /// @tparam T template class for the extension
        /// @return pointer to the registered extension (guaranteed to not go out of scope or move until Vulkan deleted)
        template<typename T, typename ...ARGS>
        const T* RequiredExtension(ARGS&& ...args) { return AddExtension( std::make_unique<T>( VulkanExtensionStatus::eRequired, std::forward<ARGS>(args)... ) ); }
        const VulkanExtension<VulkanExtensionType::eDevice>* RequiredExtension( const std::string& extensionName ) { return AddExtension( std::make_unique<VulkanExtension<VulkanExtensionType::eDevice>>( extensionName, VulkanExtensionStatus::eRequired ) ); }

        /// @brief Register the vulkan extension (templated) as desired by this app (but optional)
        /// @tparam T template class for the extension
        /// @return pointer to the registered extension (guaranteed to not go out of scope or move until Vulkan deleted)
        template<typename T, typename ...ARGS>
        const T* OptionalExtension(ARGS&& ...args) { return AddExtension( std::make_unique<T>( VulkanExtensionStatus::eOptional, std::forward<ARGS>(args)... ) ); }
        const VulkanExtension<VulkanExtensionType::eDevice>* OptionalExtension( const std::string& extensionName ) { return AddExtension( std::make_unique<VulkanExtension<VulkanExtensionType::eDevice>>( extensionName, VulkanExtensionStatus::eOptional ) ); }
 
        template<typename T>
        const T* AddExtension(std::unique_ptr<T> extension) {
            auto it = [&]() {
                if constexpr (T::Type == VulkanExtensionType::eInstance)
                    return AdditionalVulkanInstanceExtensions.try_emplace( extension->Name, std::move( extension ) );
                else if constexpr (T::Type == VulkanExtensionType::eDevice)
                    return AdditionalVulkanDeviceExtensions.try_emplace( extension->Name, std::move( extension ) );
                else if constexpr (T::Type == VulkanExtensionType::eLayer)
                    return AdditionalVulkanInstanceLayers.try_emplace( extension->Name, std::move( extension ) );
                else
                {
                    // constexpr static_assert in a template is 'problematic' in C++17 (and seemingly clang C++20).  At some point this can turn back into static_assert(0, "...");
                    [] <bool flag = false>() { static_assert(flag, "Unsupported VulkanExtensionType"); }();
                }
            }();
            if (!it.second)
            {
                assert(0 && "Double 'add' of vulkan extension not allowed");    // cannot add another extension with the same name but (potentially) a different implementation class
                return nullptr;
            }
            return static_cast<const T*>(&(*it.first->second.get()));
        }
    protected:
        friend class Vulkan;
        /// (optional) list of 'additional' instance extensions that this app requires or would like (optional).
        std::map<std::string, std::unique_ptr<VulkanExtension<VulkanExtensionType::eInstance>>> AdditionalVulkanInstanceExtensions;
        /// (optional) list of 'additional' device extensions that this app requires or would like (optional).
        std::map<std::string, std::unique_ptr<VulkanExtension<VulkanExtensionType::eDevice>>>   AdditionalVulkanDeviceExtensions;
        /// (optional) list of 'additional' device layers that this app requires or would like (optional).
        std::map<std::string, std::unique_ptr<VulkanExtension<VulkanExtensionType::eLayer>>>   AdditionalVulkanInstanceLayers;
    };

    typedef std::function<int(std::span<const SurfaceFormat>)> tSelectSurfaceFormatFn;
    typedef std::function<void(AppConfiguration&)> tConfigurationFn;
    bool Init(uintptr_t hWnd, uintptr_t hInst, const tSelectSurfaceFormatFn& SelectSurfaceFormatFn = nullptr, const tConfigurationFn& CustomConfigurationFn = nullptr);

    void Terminate();

    /// Attempt to re initialize (already initialized) vulkan instance
    /// @return true on success.
    bool ReInit(uintptr_t windowHandle);

    /// Set surface format to given format and re-create all the dependant items.
    /// Does nothing if newSurfaceFormat is same as existing otherwise calls RecreateSwapChain.
    /// @return true on success.
    bool ChangeSurfaceFormat(SurfaceFormat newSurfaceFormat);
    /// Rebuild the swapchain (and all dependant objects).
    /// Tears down and re-initializes swap chain, command buffers, render passes and frame buffers (owned by the Vulkan class)
    /// Waits for GPU to be idle before running.
    /// Will reset the current frame index.
    /// @return true on success.
    bool RecreateSwapChain();

    uint32_t GetSurfaceWidth() const    { return m_SurfaceWidth; }      ///< Swapchain width
    uint32_t GetSurfaceHeight() const   { return m_SurfaceHeight; }     ///< Swapchain height

    const Framebuffer<Vulkan>& GetSwapchainFramebuffer(uint32_t index) const { return m_SwapchainBuffers[index].framebuffer; }
    VkImage GetSwapchainImage(uint32_t index) const { return m_SwapchainBuffers[index].image; }
    TextureFormat GetSurfaceFormat() const { return m_SurfaceFormat; }

    TextureFormat GetSwapchainFormat() const { return m_SurfaceFormat; }
    size_t GetSwapchainBufferCount() const { return m_SwapchainBuffers.size(); }
    TextureFormat GetSwapchainDepthFormat() const { return m_SwapchainDepth.format; }

    /// Current buffer index (that can be filled) and the fence that should be signalled when the GPU completes this buffer and the semaphore to wait on before starting rendering.
    struct BufferIndexAndFence
    {
        /// Current frame index (internal - always in order)
        const uint32_t      idx;
        /// Current backbuffer index (from vkAcquireNextImageKHR - may be 'out of order')
        const uint32_t      swapchainPresentIdx;
        /// Fence set when GPU completes rendering this buffer
        const VkFence       fence;
        /// Backbuffer semaphore (start of rendering pipeline needs to wait for this)
        const VkSemaphore   semaphore;
    };

    /// Get the next image to render to and the fence to signal at the end, then queue a wait until the image is ready
	/// @return index of next backbuffer and the fence to signal upon comdbuffer completion.  May not be in sequential order!
    BufferIndexAndFence SetNextBackBuffer();

    /// Run vkSubmitQueue for the given command buffer(s)
    bool QueueSubmit( const std::span<const VkCommandBuffer> CommandBuffers, const std::span<const VkSemaphore> WaitSemaphores, const std::span<const VkPipelineStageFlags> WaitDstStageMasks, const std::span<const VkSemaphore> SignalSemaphores, uint32_t QueueIndex, VkFence CompletedFence );
    /// Run vkSubmitQueue for the given command buffer(s) (simplified helper)
    bool QueueSubmit( VkCommandBuffer CommandBuffer, VkSemaphore WaitSemaphore, VkPipelineStageFlags WaitDstStageMask, VkSemaphore SignalSemaphore, uint32_t QueueIndex, VkFence CompletedFence )
    {
        auto WaitSemaphores = (WaitSemaphore!=VK_NULL_HANDLE) ? std::span<VkSemaphore>( &WaitSemaphore, 1 ) : std::span<VkSemaphore>();
        auto SignalSemaphores = (SignalSemaphore!=VK_NULL_HANDLE) ? std::span<VkSemaphore>( &SignalSemaphore, 1 ) : std::span<VkSemaphore>();
        return QueueSubmit( { &CommandBuffer,1 }, WaitSemaphores, { &WaitDstStageMask, 1 }, SignalSemaphores, QueueIndex, CompletedFence );
    }
    bool QueueSubmit(const std::span<const VkSubmitInfo> SubmitInfo, uint32_t QueueIndex, VkFence CompletedFence);
    bool QueueSubmit(const std::span<const VkSubmitInfo2KHR> SubmitInfo, uint32_t QueueIndex, VkFence CompletedFence);

    /// Run the vkQueuePresentKHR
    bool PresentQueue(const std::span<const VkSemaphore> pWaitSemaphores, uint32_t SwapchainPresentIndx);
    /// Run the vkQueuePresentKHR (simplified helper)
    bool PresentQueue( VkSemaphore WaitSemaphore, uint32_t SwapchainPresentIndx )
    {
        auto WaitSemaphores = (WaitSemaphore!=VK_NULL_HANDLE) ? std::span<VkSemaphore>( &WaitSemaphore, 1 ) : std::span<VkSemaphore>();
        return PresentQueue( WaitSemaphores, SwapchainPresentIndx );
    }

    /// @brief Stall the cpu until the given queue is completed.
    /// Not recommended for use in a regular rendering pipeline as it will introduce an undesirable stall.
    bool QueueWaitIdle(uint32_t QueueIndex = eGraphicsQueue) const;

    /// @brief Stall the cpu until the Vulkan device is idle.
    /// Not recommended for use in a regular rendering pipeline as it will introduce an undesirable stall.
    /// Implements base class pure virtual.
    bool WaitUntilIdle() const override;

    typedef std::function<void( uint32_t width, uint32_t height, TextureFormat format, uint32_t span, const void* data)> tDumpSwapChainOutputFn;

    /// @brief Setup a command buffer ready to record 'setup' commands such as texture transfers etc
    /// @return command buffer ready to be recorded into
    VkCommandBuffer StartSetupCommandBuffer();
    /// @brief Close and execute the given command buffer (expected to have been created by StartSetupCommandBuffer)
    /// Will wait for command buffer execution to complete.
    void FinishSetupCommandBuffer(VkCommandBuffer setupCmdBuffer);

    VkCompositeAlphaFlagBitsKHR GetBestVulkanCompositeAlpha();
    /// @brief return the supported depth format with the highest precision depth/stencil supported with optimal tiling
    /// @param NeedStencil set if we have to have a format with stencil bits (defaulted to false).
    TextureFormat GetBestSurfaceDepthFormat(bool NeedStencil = false);

    /// Check if the Vulkan device supports the given texture format.
    /// Typically used to check block compression formats when loading images.
    /// @return true if format is supported
    bool IsTextureFormatSupported(TextureFormat) const;

    /// Get the image format properties for the given format.
    /// Will query Vulkan Physical Device if the format has not (yet) been queried, and store the results for future calls to this function.
    /// @param format format to be queried
    /// @return properties for the queried format
    const VkFormatProperties& GetFormatProperties(const VkFormat format) const;

    void SetImageLayout(VkImage image,
        VkCommandBuffer cmdBuffer,
        VkImageAspectFlags aspect,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        VkPipelineStageFlags srcMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VkPipelineStageFlags dstMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        uint32_t mipLevel = 0,
        uint32_t mipLevelCount = 1,
        uint32_t baseLayer = 0,
        uint32_t layerCount = 1);

    /// @brief Allocates a VkCommandBuffer with the given parameters.
    /// @param CmdBuffLevel whether this is a primary or secondary command buffer
    /// @param QueueIndex which queue to assign this command buffer to (index in to m_VulkanQueues, NOT the queue family index)
    /// @param pCmdBuffer output
    bool AllocateCommandBuffer(VkCommandBufferLevel CmdBuffLevel, uint32_t QueueIndex, VkCommandBuffer* pCmdBuffer/*out*/) const;

    /// @brief Frees a VkCommandBuffer.
    /// @param QueueIndex queue that this this command buffer was assigned to when allocated
    void FreeCommandBuffer(uint32_t QueueIndex, VkCommandBuffer CmdBuffer) const;


    enum class RenderPassTileShadingMode {
        Disabled = 0,
        Enabled,
        _unused,
        PerTileShading
    };

    /// @brief Parameters for creation of a render pass.
    struct RenderPassCreateData
    {
        std::span<const TextureFormat>          ColorFormats = {};
        TextureFormat                           DepthFormat = TextureFormat::UNDEFINED;
        Msaa                                    Msaa = Msaa::Samples1;
        std::span<const RenderPassInputUsage>   ColorInputUsage = {};
        std::span<const RenderPassOutputUsage>  ColorOutputUsage = {};
        RenderPassInputUsage                    DepthInputUsage = RenderPassInputUsage::Clear;
        RenderPassOutputUsage                   DepthOutputUsage = RenderPassOutputUsage::Discard;
        std::span<const TextureFormat>          ResolveFormats = {};
        RenderPassTileShadingMode               TileShading = RenderPassTileShadingMode::Disabled;
        std::array<uint8_t, 2>                  TileApron;
    };

    /// @brief Create a RenderPass (wrapper around VkRenderPass) for a single subpass with the given parameters.
    RenderPass CreateRenderPass( const RenderPassCreateData& createData );

    /// @brief Create a RenderPass (wrapper around VkRenderPass) for a single subpass with the given parameters. 
    /// When we have ResolveFormats we resolve each member of ColorFormats out to a matching ResolveFormats buffer (unless ResolveFormat is VK_FORMAT_UNDEFINED, render pass expects ColorPass + number of defined Resolve buffers - resolve buffers must be VK_SAMPLE_COUNT_1_BIT)
    /// @param ColorFormats Format and usage of all expected color buffers.
    /// @param DepthFormat Format and usage of the depth buffer.
    /// @param Msaa antialiasing for all the color/depth buffers
    /// @param ShouldClearDepth true if the renderpass should clear the depth buffer before use
    /// @param pRenderPass output
    /// @param ResolveFormats - the render pass will resolve to each of these, each entry corresponds to entry in ColorFormats - if VK_FORMAT_UNDEFINED then that color format slot is not resolved
    /// @return 
    bool CreateRenderPass(
        std::span<const TextureFormat> ColorFormats,
        TextureFormat DepthFormat,
        Msaa Msaa,
        RenderPassInputUsage ColorInputUsage,
        RenderPassOutputUsage ColorOutputUsage,
        bool ShouldClearDepth,
        RenderPassOutputUsage DepthOutputUsage,
        RenderPass& rRenderPass/*out*/,
        std::span<const TextureFormat> ResolveFormats = {} );

    bool CreateRenderPassVRS(
        std::span<const TextureFormat> ColorFormats,
        TextureFormat DepthFormat,
        Msaa Msaa,
        RenderPassInputUsage ColorInputUsage,
        RenderPassOutputUsage ColorOutputUsage,
        bool ShouldClearDepth,
        RenderPassOutputUsage DepthOutputUsage,
        RenderPass& rRenderPass/*out*/,
        std::span<const TextureFormat> ResolveFormats = {},
        bool hasDensityMap = false);

    /// @brief Create a VkRenderPass (two subpasses) with MSAA resolves (including shader resolves if supported).
    /// First subpass writes to the buffers described by 'InternalColorFormats'.  Those buffers are cleared before use and discarded at the end of the (entire) pass.
    /// Second subpass takes the buffers written by the first subpass as inputs and writes to buffers described by OutputColorFormats.  Those buffers are NOT cleared before use (assumed 2nd pass writes to all pixels).
    /// @param InternalColorFormats 
    /// @param OutputColorFormats 
    /// @param InternalDepthFormat format of depth buffer (is cleared on first subpass load and discarded after all subpasses)
    /// @param InternalPassMsaa antialisaing mode for internal color buffers and depth (all assumed to be the same) for each pass.  If PassMsaa[] != OutputMsaa AND UseShaderResolve the relevant render pass will attempt to use VK_SUBPASS_DESCRIPTION_SHADER_RESOLVE_BIT_QCOM.
    /// @param UseShaderResolve enable VK_SUBPASS_DESCRIPTION_SHADER_RESOLVE_BIT_QCOM for resolve pass, IF the internal and output msaa modes are different AND Vulkan has the VK_QCOM_render_pass_shader_resolve extension available.
    /// @param pRenderPass Output VkRenderPass
    /// @return true if successful
    bool Create2SubpassRenderPass( const std::span<const TextureFormat> InternalColorFormats,
        const std::span<const TextureFormat> OutputColorFormats,
        TextureFormat InternalDepthFormat,
        const std::span<const Msaa> InternalPassMsaa,
        Msaa OutputMsaa,
        RenderPass& rRenderPass/*out*/ );
    bool Create2SubpassRenderPass( const std::span<const TextureFormat> InternalColorFormats,
        const std::span<const TextureFormat> OutputColorFormats,
        TextureFormat InternalDepthFormat,
        Msaa InternalPassMsaa, /* same for both passes*/
        Msaa OutputMsaa,
        RenderPass& rRenderPass/*out*/ );

    /// @brief Create a VkRenderPass (two subpasses) with MSAA shader resolves (uses and requires shader resolve extension).
    /// First subpass writes to the buffers described by 'InternalColorFormats'.  Those buffers are cleared before use and discarded at the end of the (entire) pass.
    /// Second subpass takes the buffers written by the first subpass as inputs and writes to buffers described by OutputColorFormats.  Those buffers are NOT cleared before use (assumed 2nd pass writes to all pixels).
    /// @param InternalColorFormats 
    /// @param OutputColorFormats 
    /// @param InternalDepthFormat format of depth buffer (is cleared on first subpass load and discarded after all subpasses)
    /// @param InternalPassMsaa antialisaing mode for internal color buffers and depth (all assumed to be the same).
    /// @param OutputMsaa antialiasing mode for output of second 'shader resolve' pass.  Must be different (assumed lower) to @InternalPassMsaa
    /// @param pRenderPass Output VkRenderPass
    /// @return true if successful, false if shader resolve pass is not available, if inputs would not cause a shader resolve to be needed or if there is a Vulkan error creating the VkRenderPass
    bool CreateSubpassShaderResolveRenderPass(const std::span<const TextureFormat> InternalColorFormats,
        const std::span<const TextureFormat> OutputColorFormats,
        TextureFormat InternalDepthFormat,
        Msaa InternalMsaa,
        Msaa OutputMsaa,
        RenderPass& rRenderPass/*out*/ );

    /// @brief Create a render pass.
    bool CreateRenderPass( const VkRenderPassCreateInfo& createInfo, RenderPass& rRenderPass/*out*/ );
    bool CreateRenderPass( const VkRenderPassCreateInfo2KHR& createInfo, RenderPass& rRenderPass/*out*/ );

    /// @brief Create a render pass pipeline using a RenderContext that could contain renderpass/subpass
    /// @brief or dynamic render pass data.
    /// @param pipelineCache (optional) vulkan pipeline cache
    /// @param visci (required) vertex input state
    /// @param pipelineLayout (required) Vulkan pipeline layout
    /// @param renderContext (required) context (with render pass) to make this pipeline for
    /// @param subpass (required) subpass number (0 if first subpass or not using subpasses)
    /// @param providedRS (optional) rasterization state
    /// @param providedDSS (optional) depth stencil state
    /// @param specializationInfo (optional) specialization constants (shared between vert and frag shader)
    /// @return true on success
    bool CreatePipeline(
        VkPipelineCache                         pipelineCache,
        const VkPipelineVertexInputStateCreateInfo* visci,
        VkPipelineLayout                        pipelineLayout,
        const RenderContext&                    renderContext,
        const VkPipelineRasterizationStateCreateInfo* providedRS,
        const VkPipelineDepthStencilStateCreateInfo*  providedDSS,
        const VkPipelineColorBlendStateCreateInfo*    providedCBS,
        const VkPipelineMultisampleStateCreateInfo*   providedMS,
        const VkPipelineInputAssemblyStateCreateInfo* providedIA,
        std::span<const VkDynamicState>         dynamicStates,
        const VkViewport*                       viewport,
        const VkRect2D*                         scissor,
        VkShaderModule                          taskShaderModule,
        VkShaderModule                          meshShaderModule,
        VkShaderModule                          vertShaderModule,
        VkShaderModule                          fragShaderModule,
        const VkSpecializationInfo*             specializationInfo,
        bool                                    bAllowDerivation,
        VkPipeline                              deriveFromPipeline,
        VkPipeline*                             pipeline);

    /// @brief Create a compute shader pipeline
    /// @param pipelineCache (optional) vulkan pipeline cache
    /// @param pipelineLayout (required) vulkan pipeline layout
    /// @param computeModule (required) compute shader we are using in this pipeline
    /// @param specializationInfo (optional) specialization constants for this pipeline
    /// @param pipeline (output) created pipeline
    /// @return true on success
    bool CreateComputePipeline(
        VkPipelineCache             pipelineCache,
        VkPipelineLayout            pipelineLayout,
        VkShaderModule              computeModule,
        const VkSpecializationInfo* specializationInfo,
        VkPipeline*                 pipeline);

    /// @brief Return the Vulkan pipeline cache (for the default device)
    /// Pipeline cache may be VK_NULL_HANDLE (no cache)
    VkPipelineCache GetPipelineCache() const { return m_PipelineCache; }

    bool SetSwapchainHrdMetadata(const VkHdrMetadataEXT& RenderingHdrMetaData);

    // Accessors
    MemoryManager& GetMemoryManager() { return m_MemoryManager; }
    const MemoryManager& GetMemoryManager() const { return m_MemoryManager; }
    VkInstance GetVulkanInstance() const { return m_VulkanInstance; }
    const auto& GetGpuProperties() const { return m_VulkanGpuProperties; }
    const auto& GetGpuFeatures() const { return m_VulkanGpuFeatures; }

    bool GetExtGlobalPriorityAvailable() const { return m_ExtGlobalPriorityAvailable; }
    bool GetExtRenderPassShaderResolveAvailable() const { return m_ExtRenderPassShaderResolveAvailable; }
    bool GetExtRenderPassTransformAvailable() const { return m_ExtRenderPassTransformAvailable; }
    const std::vector<SurfaceFormat>& GetVulkanSurfaceFormats() const { return m_SurfaceFormats; }
    VkSurfaceTransformFlagBitsKHR GetPreTransform() const { return m_SwapchainPreTransform; }

    bool HasLoadedVulkanDeviceExtension(const std::string& extensionName) const; ///< Returns true if the named extension is loaded.

    bool FillRenderPassTransformBeginInfoQCOM(VkRenderPassTransformBeginInfoQCOM& ) const;
    bool FillCommandBufferInheritanceRenderPassTransformInfoQCOM(VkCommandBufferInheritanceRenderPassTransformInfoQCOM& ) const;

    bool SetDebugObjectName( uint64_t object, VkObjectType objectType, const char* name );
    bool SetDebugObjectName( VkCommandBuffer cmdBuffer, const char* name )
    {
        return SetDebugObjectName( (uint64_t)cmdBuffer, VK_OBJECT_TYPE_COMMAND_BUFFER, name );
    }
    bool SetDebugObjectName( VkImage image, const char* name )
    {
        return SetDebugObjectName( (uint64_t)image, VK_OBJECT_TYPE_IMAGE, name );
    }
    bool SetDebugObjectName( VkRenderPass renderPass, const char* name )
    {
        return SetDebugObjectName( (uint64_t)renderPass, VK_OBJECT_TYPE_RENDER_PASS, name );
    }
    bool SetDebugObjectName( VkPipeline pipeline, const char* name )
    {
        return SetDebugObjectName( (uint64_t)pipeline, VK_OBJECT_TYPE_PIPELINE, name );
    }
    bool SetDebugObjectName( VkFramebuffer framebuffer, const char* name )
    {
        return SetDebugObjectName( (uint64_t) framebuffer, VK_OBJECT_TYPE_FRAMEBUFFER, name );
    }
    bool SetDebugObjectName( VkDescriptorSet descriptorSet, const char* name )
    {
        return SetDebugObjectName( (uint64_t) descriptorSet, VK_OBJECT_TYPE_DESCRIPTOR_SET, name );
    }
    bool SetDebugObjectName( VkShaderModule shaderModule, const char* name )
    {
        return SetDebugObjectName( (uint64_t) shaderModule, VK_OBJECT_TYPE_SHADER_MODULE, name );
    }
    bool SetDebugObjectName( const RenderPass& renderPass, const char* name )
    {
        return SetDebugObjectName( renderPass.mRenderPass, name );
    }

    // Static helpers
    static const char* VulkanFormatString(VkFormat WhichFormat);
    static const char* VulkanColorSpaceString(VkColorSpaceKHR ColorSpace);

    float GetTimeStampPeriod() const
    {
        return m_VulkanGpuProperties.Base.properties.limits.timestampPeriod;
    }

    void ReportFramebufferProperties( VkFramebuffer vkFrameBuffer ) const;

    inline bool IsComputeQueueSupported() const
    {
        return m_VulkanGraphicsQueueSupportsCompute;
    }

    inline bool IsDataGraphQueueSupported() const
    {
        return m_VulkanGraphicsQueueSupportsDataGraph;
    }

private:
    // Vulkan takes a huge amount of code to initialize :)
    // Break up into multiple smaller functions.
    bool RegisterKnownExtensions();
    bool CreateInstance();
    bool InitInstanceExtensions();
    bool GetPhysicalDevices();
    bool GetDataGraphProcessingEngine();
    bool InitDeviceExtensions();
    bool InitQueue();
    bool InitInstanceFunctions();
    bool InitDebugCallback();
    bool InitSurface();
    bool InitCompute();
    bool InitDataGraph();
    bool InitDevice();
    bool InitSyncElements();
    bool InitCommandPools();
    bool InitMemoryManager();
    bool InitPipelineCache();
    bool QuerySurfaceCapabilities();
    bool QuerySurfaceCapabilities(VkSurfaceCapabilitiesKHR& out);
    bool InitSwapChain();
    bool InitSwapchainRenderPass();
    bool InitFrameBuffers();

    void DestroyDebugCallback();
    void DestroySwapChain();
    void DestroySwapchainRenderPass();
    void DestroyFrameBuffers();

    struct PhysicalDeviceFeatures;
    struct PhysicalDeviceProperties;
    void QueryPhysicalDeviceFeatures(VkPhysicalDevice, PhysicalDeviceFeatures& featuresOut);
    void QueryPhysicalDeviceProperties(VkPhysicalDevice, const PhysicalDeviceFeatures& features, PhysicalDeviceProperties& propertiesOut);

    /// Given an array of device properties return the index of the 'best' one (ie the one we would most liklely want as our primary rendering device).
    static size_t GetBestVulkanPhysicalDeviceId(const std::span<const PhysicalDeviceProperties> deviceProperties);

    void DumpDeviceInfo(const std::span<PhysicalDeviceFeatures> DeviceFeatures, const std::span<PhysicalDeviceProperties> DeviceProperties);

public:
    uint32_t                m_SurfaceWidth = 0;         ///< Swapchain width
    uint32_t                m_SurfaceHeight = 0;        ///< Swapchain height

    // Public Vulkan objects
    enum QueueIndex {
        eGraphicsQueue = 0,
        eComputeQueue = 1,
        eTransferQueue = 2,
        eDataGraphQueue = 3,
        eMaxNumQueues = 4
    };
    struct VulkanQueue {
        VkQueue Queue = VK_NULL_HANDLE;
        VkCommandPool CommandPool = VK_NULL_HANDLE;  // one command pool per queue, would need more if multithreading command buffer generation.
        int QueueFamilyIndex = -1;                  // -1 denotes invalid queue family (or unitialized)
    };

    VkDevice                m_VulkanDevice;
    std::array<VulkanQueue, eMaxNumQueues> m_VulkanQueues;

    VkPhysicalDevice        m_VulkanGpu;        ///< Current Vulkan GPU device being used (only one GPU is currently supported)

    VkPhysicalDeviceDataGraphProcessingEngineARM m_VulkanDataGraphProcessingEngine; ///< Current Vulkan NPU device type (only one NPU is currently supported)

    VkSemaphore             m_RenderCompleteSemaphore;

    uint32_t                m_SwapchainImageCount;
    VkSwapchainKHR          m_VulkanSwapchain;

    std::vector<SwapchainBuffers> m_SwapchainBuffers;
    DepthInfo               m_SwapchainDepth;       // ... but they all use the same depth
    std::array<VkSubpassDependency, 2> m_SwapchainRenderPassDependencies{};   // dependencies used when creating m_SwapchainRenderPass
    RenderPass              m_SwapchainRenderPass;

    TextureFormat           m_SurfaceFormat;        // Current surface format
    VkColorSpaceKHR         m_SurfaceColorSpace;    // Current surface colorspace
    bool                    m_UseRenderPassTransform;     // If set attempt to disable the surfaceflinger doing backbuffer rotation and to use the Qualcomm render pass transform extension, assumes the application will render in the device's default orientation
    bool                    m_UsePreTransform;      // If set attempt to disable the surfaceflinger doing backbuffer rotation without enabling the Qualcomm render pass transform extension.  Assumes the application will render to the rotated backbuffer.

    template<VulkanExtensionType T_TYPE>
    struct RegisteredExtensions
    {
        /// Template to add a VulkanExtension definition, parameters are the extension name and (optionally) status/version
        template<typename ... T>
        VulkanExtension<T_TYPE>* AddExtension(std::string name, T ...args) {
            auto pExtension = std::make_unique<VulkanExtension<T_TYPE>>(name, std::forward<T>(args)...);
            auto insertIt = m_Extensions.try_emplace( std::move(name), std::move(pExtension) );
            // since we are adding as a 'basic' extension (just a name, no extension class) it is ok if someone beat us to it and already registered.  We will return a pointer to their instance and the unique<VulkanExtension> we just created will be deleted upon return from this function!
            return static_cast<VulkanExtension<T_TYPE>*>(insertIt.first->second.get());
        }
        /// Template to add a more complex VulkanExtension definition (expects a class derived from VulkanExtension)
        template<typename T, typename ... TT>
        const T* AddExtension(TT ...args) {
            auto pExtension = std::make_unique<T>( std::forward<TT>(args)... );
            auto insertIt = m_Extensions.try_emplace( pExtension->Name, std::move( pExtension ) );
            assert( insertIt.second == true );//check we didnt already register this extension which may be problematic if registered once as a 'basic' VulkanExtension and once as a derived class.
            return static_cast<T*>(insertIt.first->second.get());
        }

        const VulkanExtension<T_TYPE>* GetExtension( const char*const extensionName ) const {
            auto it = m_Extensions.find( extensionName );
            if (it == m_Extensions.end())
                return nullptr;
            return it->second.get();
        }
        VulkanExtension<T_TYPE>* GetExtension( const char* const extensionName ) {
            auto it = m_Extensions.find( extensionName );
            if (it == m_Extensions.end())
                return nullptr;
            return it->second.get();
        }
        const VulkanExtension<T_TYPE>* GetExtension( const std::string& extensionName ) const {
            return GetExtension( extensionName.c_str() );
        }
        VulkanExtension<T_TYPE>* GetExtension( const std::string& extensionName ) {
            return GetExtension( extensionName.c_str() );
        }

        template<typename T>
        const T* GetExtension() const {
            static_assert(T_TYPE == T::Type, "");
            auto foundIt = m_Extensions.find( T::Name );
            if (foundIt == m_Extensions.end())
                return nullptr;
            return (const T*) foundIt->second.get();
        }

        // Call once, once all the known/wanted extensions have been added.
        void RegisterAll(Vulkan& vulkan)
        {
            for (auto& extension : m_Extensions)
            {
                LOGI("Registering extension: %s (%s)", extension.first.c_str(), VulkanExtensionBase::cStatusNames[static_cast<int>(extension.second->Status)]);
                extension.second->Register(vulkan);
            }
        }

        // Container of all the extensions we know about (app may add to this from its config).  Our GPU device driver may not support all these known extensions.
        std::map< std::string, std::unique_ptr<VulkanExtension<T_TYPE>>, std::less<> > m_Extensions;
    };

    // Containers for the registered (and potentially loaded) extensions.
    RegisteredExtensions<VulkanExtensionType::eInstance> m_InstanceExtensions;                ///< Instance extensions
    RegisteredExtensions<VulkanExtensionType::eDevice>   m_DeviceExtensions;                  ///< Device extensions
    RegisteredExtensions<VulkanExtensionType::eDevice>   m_Vulkan11ProvidedExtensions;        ///< 'extensions' included in Vulkan 1.1 (implicitly loaded).
    RegisteredExtensions<VulkanExtensionType::eLayer>    m_InstanceLayers;                    ///< Instance layers

    // Generic extension query (will fail to compile if type T does not define static Name).
    template<typename T>
    const T* GetExtension() const {
        if constexpr (T::Type == VulkanExtensionType::eInstance)
            return (const T*)m_InstanceExtensions.GetExtension(T::Name);
        else if constexpr (T::Type == VulkanExtensionType::eDevice)
            return (const T*)m_DeviceExtensions.GetExtension(T::Name);
        else
        {
            // constexpr static_assert in a template is 'problematic' in C++17 (and seemingly clang C++20).  At some point this can turn back into static_assert(0, "...");
            [] <bool flag = false>() { static_assert(flag, "Unsupported VulkanExtensionType"); }();
        }
        return nullptr;
    }
    // Template specializations for stored extension pointers (compile time lookup).
    template<>
    const ExtensionLib::Ext_VK_KHR_surface* GetExtension() const { return m_ExtKhrSurface; };
    template<>
    const ExtensionLib::Ext_VK_KHR_get_physical_device_properties2* GetExtension() const { return m_ExtKhrGetPhysicalDeviceProperties2; };
    template<>
    const ExtensionLib::Ext_VK_KHR_get_surface_capabilities2* GetExtension() const { return m_ExtSurfaceCapabilities2; };
    template<>
    const ExtensionLib::Ext_VK_KHR_draw_indirect_count* GetExtension() const { return m_ExtKhrDrawIndirectCount; };
    template<>
    const ExtensionLib::Ext_VK_EXT_debug_utils*  GetExtension() const { return m_ExtDebugUtils; };
    template<>
    const ExtensionLib::Ext_VK_EXT_debug_marker* GetExtension() const { return m_ExtDebugMarker; };
    template<>
    const ExtensionLib::Ext_VK_EXT_hdr_metadata* GetExtension() const { return m_ExtHdrMetadata; };
    template<>
    const ExtensionLib::Ext_VK_KHR_synchronization2* GetExtension() const { return m_ExtKhrSynchronization2; };
    template<>
    const ExtensionLib::Ext_VK_QCOM_tile_properties* GetExtension() const { return m_ExtQcomTileProperties; };
    template<>
    const ExtensionLib::Ext_VK_QCOM_tile_shading* GetExtension() const { return m_ExtQcomTileShading; };
    template<>
    const ExtensionLib::Ext_VK_QCOM_tile_memory_heap* GetExtension() const { return m_ExtQcomTileMemoryHeap; };
    template<>
    const ExtensionLib::Ext_VK_KHR_get_memory_requirements2* GetExtension() const { return m_ExtKhrGetMemoryRequirements; };
    template<>
    const ExtensionLib::Vulkan_SubgroupPropertiesHook* GetExtension() const { return m_SubgroupProperties; };
    template<>
    const ExtensionLib::Ext_VK_KHR_mesh_shader* GetExtension() const { return m_ExtMeshShader; };
    template<>
    const ExtensionLib::Ext_VK_KHR_dynamic_rendering* GetExtension() const { return m_ExtDynamicRendering; };

    template<typename T, typename ...TT>
    void AddExtensionHooks( ExtensionHook<T>* t, TT... tt ) {
        AddExtensionHook( t );
        if constexpr (sizeof...(TT) > 1)
            AddExtensionHooks( tt... );
        else if constexpr (sizeof...(TT) == 1)
            AddExtensionHook( tt... );
    }

    // The application may be using different queues (graphics/compute/transfer) and so
    // need access to information about the queues.  The application may also be doing timer
    // queries and the timer resolution changes based on the queue.
    // Therefore, these have been moved from "private" to "public"
    std::array<int, eMaxNumQueues>          m_VulkanQueueFamilyIndx;
    bool                                    m_VulkanGraphicsQueueSupportsCompute;
    bool                                    m_VulkanGraphicsQueueSupportsDataGraph;
    std::vector<VkQueueFamilyProperties>    m_pVulkanQueueProps;

private:
#if defined(OS_WINDOWS)
    HINSTANCE               m_hInstance;
    HWND                    m_hWnd;
#elif defined(OS_ANDROID)
    ANativeWindow*          m_pAndroidWindow = nullptr;
#elif defined(OS_LINUX)
    GLFWwindow*             m_pGlfwWindow = nullptr;
#endif // defined(OS_WINDOWS)

    /// App driven configuration overrides (setup by app before Vulkan is initialized in order to potentially override default Vulkan configuration settings)
    AppConfiguration        m_ConfigOverride;

    /// Current frame index (internal - always in order)
    uint32_t                m_SwapchainCurrentIndx;

    // Vulkan Objects
    VkInstance                          m_VulkanInstance;
    uint32_t                            m_VulkanApiVersion;
    uint32_t                            m_VulkanGpuCount;
    uint32_t                            m_VulkanGpuIdx;

    /// Features used in call to vkCreateDevice (ie the features available AND USED for the current device).
    struct PhysicalDeviceFeatures
    {
        VkPhysicalDeviceFeatures2       Base = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    } m_VulkanGpuFeatures;

    /// Properties used in call to vkCreateDevice
    struct PhysicalDeviceProperties
    {
        VkPhysicalDeviceProperties2     Base { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    } m_VulkanGpuProperties;

    VkPhysicalDeviceMemoryProperties2   m_PhysicalDeviceMemoryProperties = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2 };

    std::vector<const char*>            m_pDeviceLayerNames;            ///< Requested/Loaded device layers

    // Containers for handles in to VulkanExtensions that extend Vulkan structures (via pNext chain).
    // Allows for external applications to add suppoort for Vulkan extensions that are not supported by the framework by default.
    ExtensionChain<VkDeviceCreateInfo>                  m_DeviceCreateInfoExtensions;             ///< extensions that chain in to VkDeviceCreateInfo
    ExtensionChain<VkPhysicalDeviceFeatures2>           m_GetPhysicalDeviceFeaturesExtensions;    ///< extensions that chain in to VkPhysicalDeviceFeatures2
    ExtensionChain<VkPhysicalDeviceProperties2>         m_GetPhysicalDevicePropertiesExtensions;  ///< extensions that chain in to VkPhysicalDeviceProperties2
    ExtensionChain<VulkanDeviceFeaturePrint>            m_VulkanDeviceFeaturePrintExtensions;     ///< extensions that hook to VulkanDeviceFeaturePrint
    ExtensionChain<VulkanDevicePropertiesPrint>         m_VulkanDevicePropertiesPrintExtensions;  ///< extensions that hook to VulkanDevicePropertiesPrint
    ExtensionChain<VulkanInstanceFunctionPointerLookup> m_VulkanInstanceFunctionPointerLookupExtensions;    ///< extensions that hook to VulkanInstanceFunctionPointerLookup
    ExtensionChain<VulkanDeviceFunctionPointerLookup>   m_VulkanDeviceFunctionPointerLookupExtensions;      ///< extensions that hook to VulkanDeviceFunctionPointerLookup
    ExtensionChain<VkPipelineShaderStageCreateInfo>     m_VulkanPipelineShaderStageCreateInfoExtensions;    ///< extensions that chain to VkPipelineShaderStageCreateInfo

    void AddExtensionHook( ExtensionHook< VkDeviceCreateInfo >* p ) { m_DeviceCreateInfoExtensions.AddExtensionHook( p ); }
    void AddExtensionHook( ExtensionHook< VkPhysicalDeviceFeatures2 >* p ) { m_GetPhysicalDeviceFeaturesExtensions.AddExtensionHook( p ); }
    void AddExtensionHook( ExtensionHook< VkPhysicalDeviceProperties2 >* p ) { m_GetPhysicalDevicePropertiesExtensions.AddExtensionHook( p ); }
    void AddExtensionHook( ExtensionHook< VulkanDeviceFeaturePrint >* p ) { m_VulkanDeviceFeaturePrintExtensions.AddExtensionHook( p ); }
    void AddExtensionHook( ExtensionHook< VulkanDevicePropertiesPrint >* p ) { m_VulkanDevicePropertiesPrintExtensions.AddExtensionHook( p ); }
    void AddExtensionHook( ExtensionHook< VulkanInstanceFunctionPointerLookup >* p ) { m_VulkanInstanceFunctionPointerLookupExtensions.AddExtensionHook( p ); }
    void AddExtensionHook( ExtensionHook< VulkanDeviceFunctionPointerLookup >* p ) { m_VulkanDeviceFunctionPointerLookupExtensions.AddExtensionHook( p ); }
    void AddExtensionHook( ExtensionHook< VkPipelineShaderStageCreateInfo >* p ) { m_VulkanPipelineShaderStageCreateInfoExtensions.AddExtensionHook( p ); }

    bool                                m_LayerKhronosValidationAvailable;
    bool                                m_ExtGlobalPriorityAvailable;
    bool                                m_ExtRenderPassTransformAvailable;
    bool                                m_ExtRenderPassTransformEnabled;    ///< Set when the device is using the renderpasstransform extension (ie display pre-rotation is happening and m_ExtRenderPassTransformAvailable)
    bool                                m_ExtRenderPassTransformLegacy;     ///< Use the 'legacy' interface to VK_QCOM_render_pass_transform (older drivers)
    bool                                m_ExtRenderPassShaderResolveAvailable;
    bool                                m_ExtPortability;                   ///< Vulkan Portability extension present (and so must be enabled).  Limited subset of Vulkan functionality (for backwards compatibility with other graphics api)

    // Extensions loaded by this class
    const VulkanExtension<VulkanExtensionType::eInstance>*   m_ExtValidationFeatures = nullptr;
    const ExtensionLib::Ext_VK_KHR_surface*                  m_ExtKhrSurface = nullptr;
    const ExtensionLib::Ext_VK_KHR_get_physical_device_properties2*m_ExtKhrGetPhysicalDeviceProperties2 = nullptr;
    const ExtensionLib::Ext_VK_KHR_get_surface_capabilities2*m_ExtSurfaceCapabilities2 = nullptr;
    const ExtensionLib::Ext_VK_KHR_draw_indirect_count*      m_ExtKhrDrawIndirectCount = nullptr;
    const ExtensionLib::Ext_VK_KHR_swapchain*                m_ExtSwapchain = nullptr;
    const ExtensionLib::Ext_VK_EXT_debug_utils*              m_ExtDebugUtils = nullptr;
    const ExtensionLib::Ext_VK_EXT_debug_marker*             m_ExtDebugMarker = nullptr;
    const ExtensionLib::Ext_VK_EXT_hdr_metadata*             m_ExtHdrMetadata = nullptr;
    const ExtensionLib::Ext_VK_KHR_fragment_shading_rate*    m_ExtFragmentShadingRate = nullptr;
    const ExtensionLib::Ext_VK_KHR_create_renderpass2*       m_ExtRenderPass2 = nullptr;
    const ExtensionLib::Ext_VK_KHR_buffer_device_address*    m_ExtBufferDeviceAddress = nullptr;
    const ExtensionLib::Ext_VK_KHR_8bit_storage*             m_Ext8BitStorage = nullptr;
    const ExtensionLib::Ext_VK_KHR_cooperative_matrix*       m_ExtCooperativeMatrix = nullptr;
    const ExtensionLib::Ext_VK_KHR_synchronization2*         m_ExtKhrSynchronization2 = nullptr;
    const ExtensionLib::Ext_VK_QCOM_tile_properties*         m_ExtQcomTileProperties = nullptr;
    const ExtensionLib::Ext_VK_QCOM_tile_shading*            m_ExtQcomTileShading = nullptr;
    const ExtensionLib::Ext_VK_QCOM_tile_memory_heap*        m_ExtQcomTileMemoryHeap = nullptr;
    const ExtensionLib::Ext_VK_KHR_get_memory_requirements2* m_ExtKhrGetMemoryRequirements = nullptr;
    const ExtensionLib::Ext_VK_KHR_mesh_shader*              m_ExtMeshShader = nullptr;
    const ExtensionLib::Ext_VK_KHR_dynamic_rendering*        m_ExtDynamicRendering = nullptr;
    const ExtensionLib::Vulkan_SubgroupPropertiesHook*       m_SubgroupProperties = nullptr;
    const ExtensionLib::Vulkan_StorageFeaturesHook*          m_StorageFeatures = nullptr;

#if defined (OS_ANDROID)
    bool                                m_ExtExternMemoryCapsAvailable;
    bool                                m_ExtAndroidExternalMemoryAvailable;
#endif // defined (OS_ANDROID)

    std::unique_ptr<VulkanDebugCallback> m_DebugCallback;

    std::vector<SurfaceFormat>          m_SurfaceFormats;       ///< Available formats
    VkSurfaceKHR                        m_VulkanSurface;        ///< Current surface format format
    VkSurfaceCapabilitiesKHR            m_VulkanSurfaceCaps;    ///< Capabilities of current surface format
    VkSurfaceTransformFlagBitsKHR       m_SwapchainPreTransform;///< Current swapchain pre-transform
    mutable std::unordered_map<VkFormat, VkFormatProperties> m_FormatProperties;///< Known format properties - filled in as new formats are queried by @GetFormatProperties

    MemoryManager                       m_MemoryManager;

    VkCommandBuffer                     m_SetupCmdBuffer;

    VkPipelineCache                     m_PipelineCache;
};
