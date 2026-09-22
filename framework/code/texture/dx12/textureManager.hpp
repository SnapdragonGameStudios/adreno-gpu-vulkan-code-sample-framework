// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include "../textureManager.hpp"
#include <map>


// Forward declarations
class Dx12;
template<typename T_GFXAPI> class Texture;
template<typename T_GFXAPI> class TextureKtx;
template<typename T_GFXAPI> class TextureManager;
using TextureManagerDx12 = TextureManager<Dx12>;


/// @brief Templated implementation of TextureManagerBase that handles ktx file loading for Dx12 graphics api.
/// @tparam T_GFXAPI 
template<>
class TextureManager<Dx12> final : public TextureManagerBase
{
public:
    using tGfxApi = Dx12;
    using Texture = Texture<Dx12>;
    using Sampler = Sampler<Dx12>;

    TextureManager(tGfxApi& rGfxApi, AssetManager& rAssetManager) noexcept;

    bool Initialize();
    void Release();

    /// @brief Find a texture (by slot name) that may be already loaded
    /// @param textureSlotName name to look for
    /// @return pointer to already loaded texture, or null
    const TextureBase* GetTexture( const std::string& textureSlotName ) const override;

    /// Create texture (generally for render target usage).  Uses CreateTexObjectInfo structure to define texture creation parameters.
    /// Implements the base class virtual function.
    const TextureBase* CreateTextureObject( const CreateTexObjectInfo& texInfo ) override;

    /// Create texture from a block of texture data in memory (with correct format, span etc).
    const TextureBase* CreateTextureFromBuffer( const void* pData, size_t DataSize, uint32_t Width, uint32_t Height, uint32_t Depth, TextureFormat Format, SamplerAddressMode SamplerMode, SamplerFilter Filter, std::string name ) override;

    /// Create a texture that views (aliases) another texture but using a different texture format (must be 'related' formats, which formats are related is dependant on graphics api)
    const TextureBase* CreateTextureObjectView( const TextureBase& original, TextureFormat viewFormat, std::string name ) override;

    const SamplerBase* const GetSampler( SamplerAddressMode ) const override;

protected:
    const TextureBase* GetOrLoadTexture_( const std::string& textureSlotName, const std::string& filename, const SamplerBase& sampler ) override;
    void BatchLoad( const std::span<std::pair<std::string, std::string>>, const SamplerBase& defaultSampler ) override;

private:
    std::map<std::string, Texture> m_LoadedTextures;
    tGfxApi& m_GfxApi;
};
