// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

/// Shared definitions used across all graphics APIs (Dx12, Vulkan)
/// @ingroup Memory

#include <type_traits>

// forward declarations
template<typename T_GFXAPI> class MemoryManager;


enum class MemoryUsage {
    Unknown = 0,
    GpuExclusive,
    CpuExclusive,
    CpuToGpu,
    GpuToCpu,
    CpuCopy,
    GpuLazilyAllocated
};
enum class BufferUsageFlags {
    Unknown = 0,
    TransferSrc = 1,
    TransferDst = 2,
    Uniform = 4,
    Storage = 8,
    Index = 16,
    Vertex = 32,
    Indirect = 64,
    AccelerationStructure = 128,
    AccelerationStructureBuild = 256,
    ShaderBindingTable = 512,
    ShaderDeviceAddress = 1024,
};
inline BufferUsageFlags operator|( BufferUsageFlags a, BufferUsageFlags b )
{
    return static_cast<BufferUsageFlags>( static_cast<std::underlying_type_t<BufferUsageFlags>>( a ) | static_cast<std::underlying_type_t<BufferUsageFlags>>( b ) );
}
inline auto operator&( BufferUsageFlags a, BufferUsageFlags b )
{
    return static_cast<std::underlying_type_t<BufferUsageFlags>>( a ) & static_cast<std::underlying_type_t<BufferUsageFlags>>( b );
}
enum class IndexType {
    Unknown,
    IndexU8 = 1,
    IndexU16 = 2,
    IndexU32 = 3
};

