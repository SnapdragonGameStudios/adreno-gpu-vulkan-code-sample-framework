// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include <string>
#include <wrl.h>
#include <d3d12.h>
#include "graphicsApi/commandList.hpp"

// Forward declares
using namespace Microsoft::WRL; // for ComPtr
class Dx12;


template<>
class CommandList<Dx12> final : public CommandListBase
{
    CommandList(const CommandList<Dx12>&) = delete;
    CommandList& operator=(const CommandList<Dx12>&) = delete;
public:
    CommandList() noexcept;
    ~CommandList();
    CommandList(CommandList<Dx12>&&) noexcept;
    CommandList<Dx12>& operator=(CommandList<Dx12>&&) noexcept;

    bool Initialize(Dx12*, const std::string& Name = {}, Type CommandListType = Type::Direct, uint32_t QueueIndex = 0, ID3D12PipelineState* pPipelineState = nullptr);

    // Begin command buffer
    bool Begin(ID3D12PipelineState* pPipelineState = nullptr);

    bool End();

    bool Reset();

    /// @brief Release the Dx12 resources used by this wrapper and cleanup.
    void Release() override;

    ID3D12GraphicsCommandList* Get() const { return m_CommandList.Get(); }
    ID3D12GraphicsCommandList* operator->() const { return m_CommandList.Get(); }
    ID3D12GraphicsCommandList* operator*() const { return m_CommandList.Get(); }

private:
    ComPtr<ID3D12GraphicsCommandList>   m_CommandList;
    ComPtr<ID3D12CommandAllocator>      m_CommandAllocator; // We take ownership of the allocator for the duration of recording (and return to Dx12 class in End)
    Dx12*                               m_pDx12 = nullptr;
};

