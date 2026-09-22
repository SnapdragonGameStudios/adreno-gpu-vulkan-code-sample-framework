// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "dx12.hpp"
#include "system/os_common.h"
#include <cassert>
#include <span>
#include "texture/texture.hpp"

static const uint32_t cnMaxSamplers = 64;
static const uint32_t cnMaxShaderResourceViewHandles = 16384;


//-----------------------------------------------------------------------------
Dx12::Dx12() : GraphicsApiBase()
    , m_SwapchainFormat(TextureFormat::UNDEFINED)
//-----------------------------------------------------------------------------
{}

//-----------------------------------------------------------------------------
Dx12::~Dx12()
//-----------------------------------------------------------------------------
{
//    DestroyFrameBuffers();
//    DestroySwapchainRenderPass();
//    DestroySwapChain();

    m_MemoryManager.Destroy();
}

//-----------------------------------------------------------------------------
bool Dx12::Init(uintptr_t hWnd, uintptr_t hInst, const Dx12::tSelectSurfaceFormatFn& SelectSurfaceFormatFn, const Dx12::tConfigurationFn& CustomConfigurationFn )
//-----------------------------------------------------------------------------
{
    m_hWnd = (HWND)hWnd;
    m_hInstance = (HINSTANCE)hInst;

    if (!InitDevice())
    {
        return false;
    }
    if (!InitMemoryManager())
    {
        return false;
    }
    if (!CreateQueues())
    {
        return false;
    }
    if (!CreateCommandAllocators())
    {
        return false;
    }    
    if (!CreateSwapchain())
    {
        return false;
    }
    if (!CreateSamplerHeap())
    {
        return false;
    }
    if (!CreateShaderResourceViewHeap())
    {
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Dx12::CheckError(const char*const pPrefix, HRESULT result)
//-----------------------------------------------------------------------------
{
    if (result == S_OK)
        return true;
    LOGE("DX12 error: (%ld) from %s", (long)result, pPrefix);
    return false;
}

//-----------------------------------------------------------------------------
bool Dx12::InitDevice()
//-----------------------------------------------------------------------------
{
#if defined(_DEBUG)
    // Enable debug layer before everything else.
    {
        ComPtr<ID3D12Debug> debugI;
        if (!CheckError("D3D12GetDebugInterface", D3D12GetDebugInterface(IID_PPV_ARGS(&debugI))))
            return false;
        debugI->EnableDebugLayer();
    }
#endif

    ComPtr<IDXGIFactory4> dxgiFactory;
    if (!CheckError( "CreateDXGIFactory2", CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG, IID_PPV_ARGS(&dxgiFactory)) ))
        return false;

    // Ignore warp drivers (could add as an option here if we waned)
    ComPtr<IDXGIAdapter1> dxgiAdapter;
    UINT adaptorIdx=0;
    bool foundValidAdaptor = false;
    // Grab the first non-software adaptor.
    ///TODO: pass in desired adaptor index
    while (dxgiFactory->EnumAdapters1(adaptorIdx, &dxgiAdapter) != DXGI_ERROR_NOT_FOUND)
    {
        DXGI_ADAPTER_DESC1 adaptorDescription;
        dxgiAdapter->GetDesc1(&adaptorDescription);
        if ((adaptorDescription.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0)
        {
            foundValidAdaptor = true;
            break;
        }
        ++adaptorIdx;
    }
    if (!foundValidAdaptor)
        return false;
    ComPtr<ID3D12Device2> device;
    if (!CheckError("D3D12CreateDevice", D3D12CreateDevice(dxgiAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))))
        return false;

    m_DxgiFactory = dxgiFactory;
    m_Device = device;
    m_Adaptor = dxgiAdapter;

    return true;
}

//-----------------------------------------------------------------------------
bool Dx12::InitMemoryManager()
//-----------------------------------------------------------------------------
{
    return m_MemoryManager.Initialize(m_Adaptor.Get(), m_Device.Get());
}

//-----------------------------------------------------------------------------
bool Dx12::CreateQueues()
//-----------------------------------------------------------------------------
{
    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.NodeMask = 0;
    ComPtr<ID3D12CommandQueue> graphicsQueue;
    if (!CheckError("CreateCommandQueue", m_Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&graphicsQueue))))
        return false;

    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.NodeMask = 0;
    ComPtr<ID3D12CommandQueue> computeQueue;
    if (!CheckError("CreateCommandQueue", m_Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&computeQueue))))
        return false;

    m_DirectCommandQueue = graphicsQueue;
    m_ComputeCommandQueue = computeQueue;

    CheckError("CreateQueues Fence", m_Device->CreateFence(m_DirectCommandQueueFenceValue++, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_DirectCommandQueueFence)));

    CheckError( "CreateQueues Fence", m_Device->CreateFence( m_ComputeCommandQueueFenceValue++, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS( &m_ComputeCommandQueueFence ) ) );


    return true;
}

//-----------------------------------------------------------------------------
bool Dx12::CreateCommandAllocators()
    //-----------------------------------------------------------------------------
{
    typedef std::pair<size_t, D3D12_COMMAND_LIST_TYPE> tAC;
    decltype(m_commandAllocators) commandAllocators{};

    for (auto& [allocatorCount, allocatorType] : { tAC{1 + NUM_SWAPCHAIN_BUFFERS, D3D12_COMMAND_LIST_TYPE_DIRECT}, tAC{64, D3D12_COMMAND_LIST_TYPE_BUNDLE}, tAC{NUM_SWAPCHAIN_BUFFERS, D3D12_COMMAND_LIST_TYPE_COMPUTE}, tAC{NUM_SWAPCHAIN_BUFFERS, D3D12_COMMAND_LIST_TYPE_COPY} })
    {
        for(auto i=0;i<allocatorCount;++i)
        if (!CheckError("CreateCommandAllocators", m_Device->CreateCommandAllocator(allocatorType, IID_PPV_ARGS(&commandAllocators[allocatorType].emplace_back()))))
            return false;
    }
    m_commandAllocators = std::move(commandAllocators);
    return true;
}

//-----------------------------------------------------------------------------
bool Dx12::CreateCommandList(D3D12_COMMAND_LIST_TYPE commandListType, ID3D12PipelineState*const pPipelineState, ComPtr<ID3D12GraphicsCommandList>& pCmdList/*out*/)
//-----------------------------------------------------------------------------
{
    assert(pCmdList.Get() == nullptr);  // check not already allocated
    bool error = false;

    // Get a command allocator for duretion of the create (no longer needed after we close it).  THREAD UNSAFE (for the duration of this function)
    if (m_commandAllocators[commandListType].empty())
        return false;
    auto commandAllocator = m_commandAllocators[commandListType].back();

    // Allocate the command buffer from the allocator
    auto RetVal = m_Device->CreateCommandList(0,//nodeMask
                                              commandListType,
                                              commandAllocator.Get(),
                                              pPipelineState,
                                              IID_PPV_ARGS(&pCmdList));
    if (!CheckError("CreateCommandList", RetVal))
        return false;
    return CheckError("CloseCommandList", pCmdList->Close());
}

//-----------------------------------------------------------------------------
ComPtr<ID3D12CommandAllocator> Dx12::OpenCommandAllocator(D3D12_COMMAND_LIST_TYPE listType)
//-----------------------------------------------------------------------------
{
    auto a = m_commandAllocators[listType].back();
    m_commandAllocators[listType].pop_back();
    return a;
}

//-----------------------------------------------------------------------------
void Dx12::CloseCommandAllocator(D3D12_COMMAND_LIST_TYPE listType, ComPtr<ID3D12CommandAllocator> && commandAllocator)
//-----------------------------------------------------------------------------
{
    m_commandAllocators[listType].push_back( std::move(commandAllocator) );
}

//-----------------------------------------------------------------------------
uint64_t Dx12::CommandListExecute( ID3D12CommandList* const pCommandList )
//-----------------------------------------------------------------------------
{
    std::span<ID3D12CommandList* const> commandLists{&pCommandList, 1};

    m_DirectCommandQueue->ExecuteCommandLists( commandLists.size(), commandLists.data() );
    const uint64_t valueToSignal = m_DirectCommandQueueFenceValue++;
    m_DirectCommandQueue->Signal( m_DirectCommandQueueFence.Get(), valueToSignal );
    return valueToSignal;
}

//-----------------------------------------------------------------------------
uint64_t Dx12::ComputeCommandListExecute(ID3D12CommandList* const pCommandList)
//-----------------------------------------------------------------------------
{
    std::span<ID3D12CommandList* const> commandLists{ &pCommandList, 1 };

    m_ComputeCommandQueue->ExecuteCommandLists(commandLists.size(), commandLists.data());
    const uint64_t valueToSignal = m_ComputeCommandQueueFenceValue++;
    m_ComputeCommandQueue->Signal(m_ComputeCommandQueueFence.Get(), valueToSignal);
    return valueToSignal;
}

//-----------------------------------------------------------------------------
ID3D12GraphicsCommandList* const Dx12::StartSetupCommandBuffer()
//-----------------------------------------------------------------------------
{
    assert(!m_SetupCommandList);

    ComPtr<ID3D12GraphicsCommandList> commandList;
    if (!CreateCommandList(D3D12_COMMAND_LIST_TYPE_DIRECT, nullptr, commandList))
        return nullptr;
    auto commandAllocator = OpenCommandAllocator( D3D12_COMMAND_LIST_TYPE_DIRECT );
    if (!commandAllocator)
        return nullptr;
    if (!Dx12::CheckError( "CommandListReset", commandList->Reset( commandAllocator.Get(), nullptr)))
         return nullptr;
    m_SetupCommandAllocator = commandAllocator;
    m_SetupCommandList = commandList;
    return m_SetupCommandList.Get();
}

//-----------------------------------------------------------------------------
void Dx12::FinishSetupCommandBuffer(ID3D12GraphicsCommandList* const cmdList)
//-----------------------------------------------------------------------------
{
    assert(cmdList);
    if (cmdList != m_SetupCommandList.Get())
    {
        LOGE("Setup CommandList does not match he one passed in to FinishSetupCommandBuffer!");
    }
    CheckError("CmdList Close", cmdList->Close());

    CommandListExecute(cmdList);
    QueueWaitIdle(0);

    CloseCommandAllocator( D3D12_COMMAND_LIST_TYPE_DIRECT, std::move(m_SetupCommandAllocator) );
    m_SetupCommandList = nullptr;
}

//-----------------------------------------------------------------------------
bool Dx12::CreateSwapchain()
//-----------------------------------------------------------------------------
{
    m_SwapchainFormat = TextureFormat::R8G8B8A8_UNORM;

    DXGI_SWAP_CHAIN_DESC1 swapchainDesc{};
    swapchainDesc.Width = 0; /* get from window */
    swapchainDesc.Height = 0; /* get from window */
    swapchainDesc.Format = TextureFormatToDx( m_SwapchainFormat );
    swapchainDesc.SampleDesc = { 1,0 };
    swapchainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapchainDesc.BufferCount = NUM_SWAPCHAIN_BUFFERS;
    swapchainDesc.Scaling = DXGI_SCALING_NONE;
    swapchainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapchainDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
    swapchainDesc.Flags = 0;///TODO: optional tearing

    ComPtr<IDXGISwapChain1> swapchain;
    if (!CheckError("CreateSwapChainForHwnd", m_DxgiFactory->CreateSwapChainForHwnd(m_DirectCommandQueue.Get(), m_hWnd, &swapchainDesc, nullptr, nullptr, &swapchain)))
        return false;

    // Make a descriptor heap for the swapchain render targets
    uint32_t rtvHandleSize = 0;
    ComPtr<ID3D12DescriptorHeap> descriptorHeap = CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, swapchainDesc.BufferCount, &rtvHandleSize);

    D3D12_CPU_DESCRIPTOR_HANDLE descHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();

    for (UINT i = 0; i < swapchainDesc.BufferCount; ++i)
    {
        if (!CheckError("Swapchain GetBuffer", swapchain->GetBuffer(i, IID_PPV_ARGS(&m_FrameBuffers[i]))))
            return false;
        m_Device->CreateRenderTargetView(m_FrameBuffers[i].Get(), nullptr, descHandle);
        descHandle.ptr += rtvHandleSize;
    }

    m_SwapchainDescriptorHeap = descriptorHeap;
    m_SwapchainDescriptorSize = rtvHandleSize;
    m_SwapchainBufferCount = swapchainDesc.BufferCount;

    swapchain.As(&m_Swapchain); // to IDXGISwapChain3
    m_SwapchainPresentIndx = m_Swapchain->GetCurrentBackBufferIndex();
    m_SwapchainCurrentIndx = 0;

    // Take the surface dimensions from the swapchain
    m_Swapchain->GetDesc1(&swapchainDesc);
    m_SurfaceWidth = swapchainDesc.Width;
    m_SurfaceHeight = swapchainDesc.Height;

    //DELETE THIS BLOCK
    // dumb sync object for waiting for framesync (slow).
    {
        HRESULT hr = m_Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_Fence));
        m_FenceValue = 0;

        // Create an event handle to use for frame synchronization.
        m_FenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (m_FenceEvent == nullptr)
        {
            hr = HRESULT_FROM_WIN32(GetLastError());
        }
    }

    return true;
}

//-----------------------------------------------------------------------------
ComPtr<ID3D12DescriptorHeap> Dx12::CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE heapType, uint32_t numDescriptors, uint32_t* pOutDescriptorHandleSize)
//-----------------------------------------------------------------------------
{
    ComPtr<ID3D12DescriptorHeap> heap;
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
    heapDesc.Type = heapType;
    heapDesc.NumDescriptors = numDescriptors;
    heapDesc.Flags = (heapType != D3D12_DESCRIPTOR_HEAP_TYPE_RTV && heapType != D3D12_DESCRIPTOR_HEAP_TYPE_DSV) ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    heapDesc.NodeMask = 0;
    if (!CheckError("CreateDescriptorHeap", m_Device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&heap))))
        return {};
    if (pOutDescriptorHandleSize)
    {
        *pOutDescriptorHandleSize = m_Device->GetDescriptorHandleIncrementSize(heapType);
    }

    return heap;
}

//-----------------------------------------------------------------------------
bool Dx12::FrameInit(uint32_t WhichFrame)
//-----------------------------------------------------------------------------
{
    m_ShaderResourceViewDescriptorHeap.FreeTemporaries(WhichFrame);
    m_SamplerDescriptorHeap.FreeTemporaries(WhichFrame);
    return true;
}

//-----------------------------------------------------------------------------
bool Dx12::QueueWaitIdle(uint32_t QueueIndex) const
//-----------------------------------------------------------------------------
{
    assert(QueueIndex == 0);

    // Wait until the previous frame is finished.
    if (m_DirectCommandQueueFence->GetCompletedValue() <= m_DirectCommandQueueFenceValue)
    {
        m_DirectCommandQueueFence->SetEventOnCompletion(m_DirectCommandQueueFenceValue-1, m_FenceEvent);
        WaitForSingleObject(m_FenceEvent, INFINITE);
    }

    assert(QueueIndex == 0);
    return true;
}

//-----------------------------------------------------------------------------
bool Dx12::WaitUntilIdle() const
//-----------------------------------------------------------------------------
{
    // blah!
    return true;
}

//-----------------------------------------------------------------------------
void Dx12::BackbufferRenderSetup(uint32_t whichFrame, ID3D12GraphicsCommandList* pCommandList)
//-----------------------------------------------------------------------------
{
    const D3D12_VIEWPORT viewport{
    .TopLeftX = 0.0f,
    .TopLeftY = 0.0f,
    .Width = (float)GetSurfaceWidth(),
    .Height = (float)GetSurfaceHeight(),
    .MinDepth = 0.0f,
    .MaxDepth = 1.0f
    };
    const D3D12_RECT scissor{
        .left = 0,
        .top = 0,
        .right = (LONG)GetSurfaceWidth(),
        .bottom = (LONG)GetSurfaceHeight()
    };

    pCommandList->RSSetViewports(1, &viewport);
    pCommandList->RSSetScissorRects(1, &scissor);

    D3D12_RESOURCE_BARRIER backbufferBarrier{
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE ,
        .Transition = {
            .pResource = m_FrameBuffers[whichFrame].Get(),
            .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            .StateBefore = D3D12_RESOURCE_STATE_PRESENT,
            .StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET
        }
    };
    pCommandList->ResourceBarrier(1, &backbufferBarrier);

    D3D12_CPU_DESCRIPTOR_HANDLE descriptorHandle{ .ptr = m_SwapchainDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + whichFrame * m_SwapchainDescriptorSize };
    pCommandList->OMSetRenderTargets(1, &descriptorHandle, FALSE, nullptr);

    const float clearColor[4] {};
    pCommandList->ClearRenderTargetView(descriptorHandle, clearColor, 0, nullptr);
}

//-----------------------------------------------------------------------------
void Dx12::BackbufferPresentSetup(uint32_t whichFrame, ID3D12GraphicsCommandList* pCommandList)
//-----------------------------------------------------------------------------
{
    // Backbuffer transition to presentable
    D3D12_RESOURCE_BARRIER backbufferBarrier{
        .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE ,
        .Transition = {
            .pResource = m_FrameBuffers[whichFrame].Get(),
            .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            .StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET,
            .StateAfter = D3D12_RESOURCE_STATE_PRESENT
        }
    };
    pCommandList->ResourceBarrier(1, &backbufferBarrier);
}

//-----------------------------------------------------------------------------
bool Dx12::PresentSwapchain()
//-----------------------------------------------------------------------------
{
    bool error = CheckError("PresentSwapchain", m_Swapchain->Present(1/*SyncInterval*/, 0/*Flags*/));

    // Signal and increment the fence value.
    const UINT64 fence = m_FenceValue;
    m_DirectCommandQueue->Signal(m_Fence.Get(), fence);
    m_FenceValue++;

    return error;
}

//-----------------------------------------------------------------------------
Dx12::BufferIndexAndFence Dx12::SetNextBackBuffer()
//-----------------------------------------------------------------------------
{
    QueueWaitIdle(0);

    // Get next frame
    const auto SwapchainPresentIndx = m_Swapchain->GetCurrentBackBufferIndex();

    // Grab the swapchain index and then increment.
    // This index is decoupled from the present index returned by GetCurrentBackBufferIndex (which may not run in sequence depending on present modes etc).
    const uint32_t CurrentIndex = m_SwapchainCurrentIndx++;
    if( m_SwapchainCurrentIndx == m_SwapchainBufferCount )
        m_SwapchainCurrentIndx = 0;
    return {CurrentIndex, SwapchainPresentIndx/*, Fence, BackBufferSemaphore*/};
}

//-----------------------------------------------------------------------------
bool Dx12::CreateSamplerHeap()
//-----------------------------------------------------------------------------
{
    uint32_t handleSize = 0;
    auto heap = CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, cnMaxSamplers, &handleSize);
    if (!heap)
        return false;
    return m_SamplerDescriptorHeap.Initialize(std::move(heap), cnMaxSamplers, handleSize);
}

//-----------------------------------------------------------------------------
bool Dx12::CreateShaderResourceViewHeap()
//-----------------------------------------------------------------------------
{
    uint32_t handleSize = 0;
    auto heap = CreateDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, cnMaxShaderResourceViewHandles, &handleSize);
    if (!heap)
        return false;
    return m_ShaderResourceViewDescriptorHeap.Initialize(std::move(heap), cnMaxShaderResourceViewHandles, handleSize);
}

//-----------------------------------------------------------------------------
DescriptorTableHandle Dx12::AllocateShaderResourceViewDescriptors(uint32_t numHandles, const std::optional<uint32_t> frameNumTemporary)
//-----------------------------------------------------------------------------
{
    if (frameNumTemporary.has_value())
        return m_ShaderResourceViewDescriptorHeap.AllocateTemporary(*frameNumTemporary, numHandles);
    else
        return m_ShaderResourceViewDescriptorHeap.Allocate(numHandles);
}

//-----------------------------------------------------------------------------
DescriptorTableHandle Dx12::AllocateSamplerDescriptors(uint32_t numHandles, const std::optional<uint32_t> frameNumTemporary)
//-----------------------------------------------------------------------------
{
    if (frameNumTemporary.has_value())
        return m_SamplerDescriptorHeap.AllocateTemporary( *frameNumTemporary, numHandles);
    else
        return m_SamplerDescriptorHeap.Allocate(numHandles);
}

//-----------------------------------------------------------------------------
void Dx12::FreeShaderResourceViewDescriptors(DescriptorTableHandle&& handle)
//-----------------------------------------------------------------------------
{
    return m_ShaderResourceViewDescriptorHeap.Free(std::move(handle));
}

//-----------------------------------------------------------------------------
void Dx12::FreeSamplerDescriptors(DescriptorTableHandle&& handle)
//-----------------------------------------------------------------------------
{
    return m_SamplerDescriptorHeap.Free(std::move(handle));
}

//-----------------------------------------------------------------------------
void Dx12::SetDescriptorHeaps(ID3D12GraphicsCommandList* pCommandList) const
//-----------------------------------------------------------------------------
{
    std::array<ID3D12DescriptorHeap*, 2> descriptorHeaps{ m_ShaderResourceViewDescriptorHeap.GetHeap(), m_SamplerDescriptorHeap.GetHeap() };
    pCommandList->SetDescriptorHeaps(descriptorHeaps.size(), descriptorHeaps.data());
}

//-----------------------------------------------------------------------------
bool Dx12::CreateSampler(const D3D12_SAMPLER_DESC& desc, D3D12_CPU_DESCRIPTOR_HANDLE destHandle)
//-----------------------------------------------------------------------------
{
    m_Device->CreateSampler(&desc, destHandle);
    return true;
}

//-----------------------------------------------------------------------------
TextureFormat Dx12::GetBestSurfaceDepthFormat(bool NeedStencil) const
//-----------------------------------------------------------------------------
{
    TextureFormat desiredDepthFormat = NeedStencil ? TextureFormat::D32_SFLOAT_S8_UINT : TextureFormat::D32_SFLOAT;

    D3D12_FEATURE_DATA_FORMAT_SUPPORT formatSupport = { .Format = TextureFormatToDx(desiredDepthFormat) };
    if (!CheckError("CheckFeatureSupport", m_Device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &formatSupport, sizeof(formatSupport))))
    {
        return TextureFormat::UNDEFINED;
    }
    if ((formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL) == 0)
    {
        assert(0);///TODO: try some other formats if the selected one isnt supported
        return TextureFormat::UNDEFINED;
    }

    return desiredDepthFormat;
}

//-----------------------------------------------------------------------------
bool Dx12::IsTextureFormatSupported(TextureFormat format) const
//-----------------------------------------------------------------------------
{
    D3D12_FEATURE_DATA_FORMAT_SUPPORT formatSupport = { .Format = TextureFormatToDx(format) };
    if (CheckError("CheckFeatureSupport", m_Device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &formatSupport, sizeof(formatSupport))))
    {
        return false;
    }
    ///TODO: what are we wanting to check here?  support for what feature?
    if ((formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D) == 0)
    {
        return false;
    }
    return true;
}

//-----------------------------------------------------------------------------
/*static*/ uint32_t Dx12::FormatBytesPerPixel(DXGI_FORMAT format)
//-----------------------------------------------------------------------------
{
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:
        return 16;
    case DXGI_FORMAT_R32G32B32_TYPELESS:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32_SINT:
        return 12;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UINT:
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R32G32_TYPELESS:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
    case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
        return 8;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_UINT:
    case DXGI_FORMAT_R16G16_SNORM:
    case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
    case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_TYPELESS:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        return 4;
    case DXGI_FORMAT_R8G8_TYPELESS:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8G8_SNORM:
    case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_D16_UNORM:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_SNORM:
    case DXGI_FORMAT_R16_SINT:
    case DXGI_FORMAT_B5G6R5_UNORM:
    case DXGI_FORMAT_B5G5R5A1_UNORM:
        return 2;
    case DXGI_FORMAT_R8_TYPELESS:
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SNORM:
    case DXGI_FORMAT_R8_SINT:
    case DXGI_FORMAT_A8_UNORM:
    case DXGI_FORMAT_R1_UNORM:
        return 1;
    default:
        assert(0);
        return 1;
    }
}

//-----------------------------------------------------------------------------
void Dx12::SetName(ID3D12DeviceChild* pObject, const std::string& Name)
//-----------------------------------------------------------------------------
{
    pObject->SetPrivateData(WKPDID_D3DDebugObjectName, Name.size(), Name.c_str());
}

//-----------------------------------------------------------------------------
void Dx12::SetName(ID3D12DeviceChild* pObject, const std::string_view& Name)
//-----------------------------------------------------------------------------
{
    pObject->SetPrivateData(WKPDID_D3DDebugObjectName, Name.size(), Name.data());
}
