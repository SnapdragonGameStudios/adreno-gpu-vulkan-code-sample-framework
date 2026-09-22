// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once
#include <memory>


//=============================================================================
// Enumerations
//=============================================================================
enum class RenderPassInputUsage {
    Clear,
    Load,
    DontCare
};

enum class RenderPassOutputUsage {
    Discard,
    Store,
    StoreReadOnly,
    StoreTransferSrc,
    Present,        // Pass writing to the backbuffer
    Clear           //needs exension
};


/// @brief Helpers to convert from base type to graphics api derived type
/// Use ONLY if you are sure the base object ACTUALLY matches give graphics api.
/// @tparam T_GFXAPI graphics api to cast to.
/// @param rBase base object (eg TextureBase)
template<class T_GFXAPI, class T>
const auto& apiCast(const T& rBase) {
    using tDerived = typename T::template tApiDerived<T_GFXAPI>;
    return static_cast<const tDerived&>(rBase);
}

template<class T_GFXAPI, typename T>
auto& apiCast(T& rBase) {
    using tDerived = typename T::template tApiDerived<T_GFXAPI>;
    return static_cast<tDerived&>(rBase);
}

template<class T_GFXAPI, typename T>
const auto* apiCast(const T* rBase) {
    using tDerived = typename T::template tApiDerived<T_GFXAPI>;
    return static_cast<const tDerived*>(rBase); }

template<class T_GFXAPI, typename T>
auto* apiCast(T* rBase) {
    using tDerived = typename T::template tApiDerived<T_GFXAPI>;
    return static_cast<tDerived*>(rBase);
}

template<class T_GFXAPI, typename T>
auto apiCast(std::unique_ptr<T>&& rBase) {
    using tDerived = typename T::template tApiDerived<T_GFXAPI>;
    return std::unique_ptr<tDerived>(static_cast<tDerived*>(rBase.get()));
}


/// Base class for graphics API implementation (eg Vulkan and Dx12 classes both derive from this class)
class GraphicsApiBase
{
public:
    virtual ~GraphicsApiBase();

    /// @brief Wait until graphics device is idle
    /// @return true on success
    virtual bool WaitUntilIdle() const = 0;

    //
    // Helpers
    //

};

inline GraphicsApiBase::~GraphicsApiBase() {}
