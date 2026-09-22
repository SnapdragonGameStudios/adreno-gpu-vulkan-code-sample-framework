// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "commandList.hpp"
#include "dx12.hpp"

//-----------------------------------------------------------------------------
CommandList<Dx12>::CommandList() noexcept
//-----------------------------------------------------------------------------
    : CommandListBase()
{
}

//-----------------------------------------------------------------------------
CommandList<Dx12>::~CommandList()
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
CommandList<Dx12>::CommandList(CommandList<Dx12>&& other) noexcept
//---------------------------------------------------------------------------
{
    assert(0);  // Currently move is not implemented for this class, but std::vector will not compile if a move (or copy) is not provided, we currently expect that the move will not get called (ie vector of CommandList<Dx12> will not resize)
}

//-----------------------------------------------------------------------------
CommandList<Dx12>& CommandList<Dx12>::operator=(CommandList<Dx12> && other) noexcept
//-----------------------------------------------------------------------------
{
    assert(0);  // Currently move is not implemented for this class, but std::vector will not compile if a move (or copy) is not provided, we currently expect that the move will not get called (ie vector of CommandList<Dx12> will not resize)
    return *this;
}

//-----------------------------------------------------------------------------
bool CommandList<Dx12>::Initialize(Dx12* gfxApi, const std::string& Name, CommandListBase::Type CommandListType, uint32_t QueueIndex, ID3D12PipelineState* pPipelineState)
//-----------------------------------------------------------------------------
{
    m_pDx12 = gfxApi;
    m_Name = Name;
    m_Type = CommandListType;
    // ensure framework enums match Dx12 enum values so they can be cast
    static_assert(uint32_t( Type::Direct ) == uint32_t( D3D12_COMMAND_LIST_TYPE_DIRECT ));
    static_assert(uint32_t( Type::Bundle ) == uint32_t( D3D12_COMMAND_LIST_TYPE_BUNDLE ));
    static_assert(uint32_t( Type::Compute ) == uint32_t( D3D12_COMMAND_LIST_TYPE_COMPUTE ));
    static_assert(uint32_t( Type::Copy ) == uint32_t( D3D12_COMMAND_LIST_TYPE_COPY ));
    static_assert(uint32_t( Type::VideoDecode ) == uint32_t( D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE ));
    static_assert(uint32_t( Type::VideoProcess ) == uint32_t( D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS ));
    static_assert(uint32_t( Type::VideoEncode ) == uint32_t( D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE ));

    // Reset draw counts
    m_NumDrawCalls = 0;
    m_NumTriangles = 0;

    if (!gfxApi->CreateCommandList( (D3D12_COMMAND_LIST_TYPE)m_Type, pPipelineState, m_CommandList))
        return false;
    Dx12::SetName(m_CommandList.Get(), Name);
    return true;
}

//-----------------------------------------------------------------------------
bool CommandList<Dx12>::Reset()
//-----------------------------------------------------------------------------
{
    m_NumDrawCalls = 0;
    m_NumTriangles = 0;

    //if (!Dx12::CheckError( "CmdList Reset", m_CommandList.Get()->Reset()))
    //     return false;

    return true;
}

//-----------------------------------------------------------------------------
bool CommandList<Dx12>::Begin(ID3D12PipelineState* pPipelineState)
//-----------------------------------------------------------------------------
{
    m_CommandAllocator = m_pDx12->OpenCommandAllocator( (D3D12_COMMAND_LIST_TYPE)m_Type );
    if (!m_CommandAllocator)
        return false;
    return Dx12::CheckError( "CommandListReset", m_CommandList->Reset(m_CommandAllocator.Get(), pPipelineState) );
}

//-----------------------------------------------------------------------------
bool CommandList<Dx12>::End()
//-----------------------------------------------------------------------------
{
    assert(m_CommandAllocator);
    bool success = Dx12::CheckError( "CommandListClose", m_CommandList->Close() );
    m_pDx12->CloseCommandAllocator( (D3D12_COMMAND_LIST_TYPE)m_Type, std::move(m_CommandAllocator) );
    return success;
}

//-----------------------------------------------------------------------------
void CommandList<Dx12>::Release()
//-----------------------------------------------------------------------------
{
    CommandListBase::Release();

    assert(!m_CommandAllocator);    // expected to be closed
    m_CommandList.Reset();
}
