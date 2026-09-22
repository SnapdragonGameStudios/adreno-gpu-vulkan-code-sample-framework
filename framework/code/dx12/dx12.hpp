// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#define NOMINMAX
#include "graphicsApi/graphicsApiBase.hpp"
#include <Windows.h>
#include <wrl.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <span>
#include <string>
#include <optional>
#include <vector>
#include "descriptorHeapManager.hpp"
#include "memory/dx12/memoryManager.hpp"

using namespace Microsoft::WRL; // for ComPtr

// Forward declarations
template<typename T_GFXAPI> class IndexBuffer;
template<typename T_GFXAPI> class VertexBuffer;
template<typename T_GFXAPI> class MemoryManager;
enum class TextureFormat;
DXGI_FORMAT TextureFormatToDx(TextureFormat f);

#define NUM_SWAPCHAIN_BUFFERS (8)

struct SurfaceFormat
{
    TextureFormat   format;
};


/// DirectX 12 API implementation
/// Contains DirectX12 top level (driver etc) objects and provides a simple initialization interface.
class Dx12 : public GraphicsApiBase
{
    Dx12(const Dx12&) = delete;
    Dx12& operator=(const Dx12&) = delete;
public:
    // typedefs
    using ViewportClass = D3D12_VIEWPORT;
    using Rect2DClass = D3D12_RECT;
    using MemoryManager = MemoryManager<Dx12>;
    using BufferHandleType = ID3D12Resource*;

    struct AppConfiguration
    {
        /// (optional) override of the framebuffer depth format.  Setting to UNDEFINED will disable the creation of a depth buffer during InitSwapChain
        std::optional<TextureFormat> SwapchainDepthFormat;
    };

public:
    Dx12();
    virtual ~Dx12();

    typedef std::function<int( std::span<const SurfaceFormat> )> tSelectSurfaceFormatFn;
    typedef std::function<void( AppConfiguration& )> tConfigurationFn;

    /// @brief Main entry point to Dx12.
    /// Initialize the Dx12 device/driver and create objects needed to start recording and executing command lists
    /// @param hWnd windows window handle
    /// @param hInst windows application instance
    /// @param iDesiredMSAA 
    /// @return 
    bool Init(uintptr_t hWnd, uintptr_t hInst, const tSelectSurfaceFormatFn& SelectSurfaceFormatFn = nullptr, const tConfigurationFn& CustomConfigurationFn = nullptr );

    /// @brief Prepare for update of frame
    bool FrameInit(uint32_t WhichFrame);

    /// Stall CPU thread until GPU device is idle
    /// Not recommended for use in a regular rendering pipeline as it will introduce an undesirable stall.
    /// Implements base class pure virtual.
    bool WaitUntilIdle() const override;

    /// Stall CPU thread until GPU queue is idle
    bool QueueWaitIdle(uint32_t QueueIndex) const;

    // Accessors
    MemoryManager& GetMemoryManager() { return m_MemoryManager; }
    const MemoryManager& GetMemoryManager() const { return m_MemoryManager; }
    auto GetDevice() const { return m_Device.Get(); }

    uint32_t GetSurfaceWidth() const { return m_SurfaceWidth; }                 ///< Swapchain width
    uint32_t GetSurfaceHeight() const { return m_SurfaceHeight; }               ///< Swapchain height
    TextureFormat GetSwapchainFormat() const { return m_SwapchainFormat; }      ///< Swapchain format
    uint32_t GetSwapchainBufferCount() const { return m_SwapchainBufferCount; } ///< Swapchain number of images

    //
    // Helpers
    //

    /// @brief Check a windows hresult error code, and (on error) output error message string including pPrefix
    /// @return return true on 'ok', false on 'error'
    static bool CheckError(const char* const pPrefix, HRESULT result);

    /// @brief Create a command list object ready for recording commands.
    /// @param commandListType type of command list (type must be supported by CreateCommandAllocators)
    /// @param pCmdList output command list object
    /// @return true on success
    bool CreateCommandList(D3D12_COMMAND_LIST_TYPE commandListType, ID3D12PipelineState* const pPipelineState, ComPtr<ID3D12GraphicsCommandList>& pCmdList/*out*/);

    /// @brief Obtain a command allocator for use by ID3D12CommandList::Reset
    /// @param commandListType type of command list allocator (type must be supported by CreateCommandAllocators)
    /// @returns a valid command list allocator (or nullptr object) that can be used to record commands and MUST then be returned to Dx12 via CloseCommandAllocator
    ComPtr<ID3D12CommandAllocator> OpenCommandAllocator(D3D12_COMMAND_LIST_TYPE listType);

    /// @brief Returns a command allocator back to the pool of allocators than can be grabbed by OpenCommandAllocator
    /// Expected that the 'closed' command allocator is not used after this function is called
    /// @param commandListType type of command list allocator (must match the type passed to the matching OpenCommandAllocator)
    void CloseCommandAllocator(D3D12_COMMAND_LIST_TYPE listType, ComPtr<ID3D12CommandAllocator> && commandAllocator);

    /// @brief Submit/execute a command list on the GPU
    /// @return value that will be signalled when this command completes
    uint64_t CommandListExecute(ID3D12CommandList* const);

    /// @brief Submit/execute a compute command list on the GPU
    /// @return value that will be signalled when this command completes
    uint64_t ComputeCommandListExecute(ID3D12CommandList* const);

    /// @brief Setup a command buffer ready to record 'setup' commands such as texture transfers etc
    /// @return command buffer ready to be recorded into
    ID3D12GraphicsCommandList* const StartSetupCommandBuffer();
    /// @brief Close and execute the given command buffer (expected to have been created by StartSetupCommandBuffer)
    /// Will wait for command buffer execution to complete.
    void FinishSetupCommandBuffer(ID3D12GraphicsCommandList* const);

    void CreateRenderTargetView();

    /// @brief Add commands to transition and setup the backbuffer to be a render target (do prior to commands rendering to the backbuffer)
    void BackbufferRenderSetup(uint32_t whichFrame, ID3D12GraphicsCommandList* pCommandList);

    /// @brief Add commands to transition and setup the backbuffer to be presentable (do prior to calling PresentSwapchain)
    void BackbufferPresentSetup(uint32_t whichFrame, ID3D12GraphicsCommandList* pCommandList);

    /// @brief Present the swapchain to the display
    bool PresentSwapchain();

    /// Current buffer index (that can be filled) and the fence that should be signalled when the GPU completes this buffer and the semaphore to wait on before starting rendering.
    struct BufferIndexAndFence
    {
        /// Current frame index (internal - always in order)
        const uint32_t      idx;
        /// Current backbuffer index (from vkAcquireNextImageKHR - may be 'out of order')
        const uint32_t      swapchainPresentIdx;
        /// Fence set when GPU completes rendering this buffer
        //const VkFence       fence;
        /// Backbuffer semaphore (start of rendering pipeline needs to wait for this)
        //const VkSemaphore   semaphore;
    };

    /// Get the next image to render to and the fence to signal at the end, then queue a wait until the image is ready
    /// @return index of next backbuffer and the fence to signal upon comdbuffer completion.  May not be in sequential order!
    BufferIndexAndFence SetNextBackBuffer();

    DescriptorTableHandle AllocateShaderResourceViewDescriptors(uint32_t numHandles, const std::optional<uint32_t> frameNumTemporary = std::nullopt);
    DescriptorTableHandle AllocateSamplerDescriptors(uint32_t numHandles, const std::optional<uint32_t> frameNumTemporary = std::nullopt );
    void FreeShaderResourceViewDescriptors(DescriptorTableHandle &&);
    void FreeSamplerDescriptors(DescriptorTableHandle &&);
    void SetDescriptorHeaps(ID3D12GraphicsCommandList* pCommandList) const;
    const auto& GetShaderResourceViewDescriptorHeap() const { return m_ShaderResourceViewDescriptorHeap; }

    bool CreateSampler(const D3D12_SAMPLER_DESC&, D3D12_CPU_DESCRIPTOR_HANDLE destHandle);

    /// @brief return the supported depth format with the highest precision depth/stencil supported with optimal tiling
    /// @param NeedStencil set if we have to have a format with stencil bits (defaulted to false).
    TextureFormat GetBestSurfaceDepthFormat(bool NeedStencil = false) const;

    /// Check if the Dx12 device supports the given texture format.
    /// @return true if format is supported
    bool IsTextureFormatSupported(TextureFormat) const;

    /// Get the number of bytes per pixel of the given DXGI_FORMAT
    static uint32_t FormatBytesPerPixel(DXGI_FORMAT format);

    /// @brief Helper to set debug name (ascii) on any D3D12 object
    static void SetName(ID3D12DeviceChild* pObject, const std::string& Name);
    /// @brief Helper to set debug name (ascii) on any D3D12 object
    static void SetName(ID3D12DeviceChild* pObject, const std::string_view& Name);

    // Helper
    ComPtr<ID3D12DescriptorHeap> CreateDescriptorHeap( D3D12_DESCRIPTOR_HEAP_TYPE heapType, uint32_t numDescriptors, uint32_t* pOutDescriptorHandleSize );

protected:
    // Internal initialization functions (called by Init).
    bool InitMemoryManager();
    bool InitDevice();
    bool CreateQueues();
    bool CreateCommandAllocators();
    bool CreateSwapchain();
    bool CreateSamplerHeap();
    bool CreateShaderResourceViewHeap();

private:
    // Windows window/app handles
    HINSTANCE                   m_hInstance = 0;
    HWND                        m_hWnd = 0;

    // D3d objects
    ComPtr<IDXGIFactory4>       m_DxgiFactory;
    ComPtr<ID3D12Device2>       m_Device;
    ComPtr<IDXGIAdapter1>       m_Adaptor;

    ComPtr<IDXGISwapChain3>     m_Swapchain;
    ComPtr<ID3D12DescriptorHeap>m_SwapchainDescriptorHeap;
    uint32_t                    m_SwapchainDescriptorSize = 0;
    TextureFormat               m_SwapchainFormat;
    uint32_t                    m_SwapchainBufferCount = 0;

    ComPtr<ID3D12CommandQueue>  m_DirectCommandQueue;
    ComPtr<ID3D12Fence>         m_DirectCommandQueueFence;          // Fence that is signalled on every execute
    uint64_t                    m_DirectCommandQueueFenceValue = 0; // Value that will be signalled by m_DirectCommandQueueFence on the next execute

    ComPtr<ID3D12CommandQueue>  m_ComputeCommandQueue;
    ComPtr<ID3D12Fence>         m_ComputeCommandQueueFence;         // Fence that is signalled on every execute
    uint64_t                    m_ComputeCommandQueueFenceValue = 0;// Value that will be signalled by m_ComputeCommandQueueFence on the next execute

    ComPtr<ID3D12GraphicsCommandList> m_SetupCommandList;           // Command list created and returned by StartSetupCommandBuffer
    ComPtr<ID3D12CommandAllocator>    m_SetupCommandAllocator;      // Command allocator used by command list returned by StartSetupCommandBuffer

    static const size_t cNumCommandListTypes = 4;
    std::array< std::deque< ComPtr<ID3D12CommandAllocator> >, cNumCommandListTypes > m_commandAllocators{};

    // Memory manager
    MemoryManager               m_MemoryManager;

    std::array<ComPtr<ID3D12Resource>, NUM_SWAPCHAIN_BUFFERS> m_FrameBuffers;

    /// Current frame index (internal - always in order)
    uint32_t                    m_SwapchainCurrentIndx = 0;
    /// Current swapchain/backbuffer index (from directx)
    uint32_t                    m_SwapchainPresentIndx = 0;

    // TEMPORARY
    HANDLE                      m_FenceEvent = 0;
    ComPtr<ID3D12Fence>         m_Fence;
    UINT64                      m_FenceValue = 0;
    // END TEMPORARY

    uint32_t                    m_SurfaceWidth = 0;         ///< Swapchain width
    uint32_t                    m_SurfaceHeight = 0;        ///< Swapchain height

    DescriptorHeapManager       m_ShaderResourceViewDescriptorHeap;
    DescriptorHeapManager       m_SamplerDescriptorHeap;
};
