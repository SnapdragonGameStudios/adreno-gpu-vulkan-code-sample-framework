// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2023 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

// Forward declarations
template<class T_GFXAPI> class CommandList;


// Base graphics Command list container class
class CommandListBase
{
    CommandListBase(const CommandListBase&) = delete;
    CommandListBase& operator=(const CommandListBase&) = delete;
public:
    CommandListBase() noexcept {}
    virtual ~CommandListBase() {}

    std::string         m_Name;
    uint32_t            m_NumDrawCalls = 0;
    uint32_t            m_NumTriangles = 0;

    enum class Type
    {
        // Match Dx12 types (Vulkan supports a subset)
        Primary = 0,
        Direct = Primary,
        Secondary = 1,
        Bundle = Secondary,
        Compute = 2,
        Copy,
        VideoDecode,
        VideoProcess,
        VideoEncode
    } m_Type = Type::Primary;

    virtual void Release() = 0;

    template<class T_GFXAPI> using tApiDerived = CommandList<T_GFXAPI>; // make apiCast work!
};

inline void CommandListBase::Release()
{
    m_Name.clear();
    m_NumDrawCalls = 0;
    m_NumTriangles = 0;
    m_Type = Type::Primary;
}


// Templated command list container class (templated against graphics api)
// Expected for this class to be specialized for each graphics api.
template<class T_GFXAPI>
class CommandList : public CommandListBase
{
    CommandList(const CommandList<T_GFXAPI>&) = delete;
    CommandList& operator=(const CommandList<T_GFXAPI>&) = delete;
public:
    CommandList() noexcept {}       // This class is expected to be specialized!
    ~CommandList() = delete;        // This class is expected to be specialized!

protected:
    static_assert(sizeof(CommandList<T_GFXAPI>) != sizeof(CommandListBase));   // Ensure this class template is specialized (and not used as-is)
};

