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
class Vulkan;

template<typename T_GFXAPI> class LoaderKtxT;
template<typename T_GFXAPI> class Sampler;
template<typename T_GFXAPI> class Texture;
class TextureKtxBase;
template<typename T_GFXAPI> class TextureManager;
template<typename T_GFXAPI> class TexturePpm;
using TextureManagerVulkan = TextureManager<Vulkan>;


/// @brief Templated implementation of TextureManager that handles ktx file loading for Vulkan graphics api.
/// @tparam T_GFXAPI 
template<>
class TextureManager<Vulkan> final : public TextureManagerBase
{
public:
    using tGfxApi = Vulkan;
    using Texture = Texture<Vulkan>;
    using Sampler = Sampler<Vulkan>;

    TextureManager(tGfxApi& rGfxApi, AssetManager&) noexcept;
    virtual ~TextureManager() override;
    TextureManager& operator=(const TextureManager&) = delete;
    TextureManager(const TextureManager&) = delete;

    bool Initialize();
    void Release() override;

    /// @brief Find a texture (by slot name) that may be already loaded
    /// @param textureSlotName name to look for
    /// @return pointer to already loaded texture, or null
    const TextureBase* GetTexture(const std::string& textureSlotName) const override;

    /// Create texture (generally for render target usage).  Uses CreateTexObjectInfo structure to define texture creation parameters.
    /// Implements the base class virtual function.
    const TextureBase* CreateTextureObject( const CreateTexObjectInfo& texInfo) override;

    /// Create texture from a block of texture data in memory (with correct format, span etc).
    /// Implements the base class virtual function.
    const TextureBase* CreateTextureFromBuffer( const void* pData, size_t DataSize, uint32_t Width, uint32_t Height, uint32_t Depth, TextureFormat Format, SamplerAddressMode SamplerMode, SamplerFilter Filter, std::string name ) override;

    /// Create a texture that views (aliases) another texture but using a different texture format (must be 'related' formats, which formats are related is dependant on graphics api)
    /// Implements the base class virtual function.
    const TextureBase* CreateTextureObjectView( const TextureBase& original, TextureFormat viewFormat, std::string name ) override;

    /// Get a 'default' sampler for the given address mode (all other sampler settings assumed to be 'normal' ie linearly sampled etc)
    const SamplerBase* const GetSampler( SamplerAddressMode ) const override;

protected:
    const TextureBase* GetOrLoadTexture_(const std::string& textureSlotName, const std::string& filename, const SamplerBase& sampler) override;
    void BatchLoad(const std::span<std::pair<std::string, std::string>>, const SamplerBase& defaultSampler) override;

private:
    std::map<std::string, Texture>          m_LoadedTextures;
    std::vector<Sampler>                    m_DefaultSamplers;
    const bool                              m_MirrorClampToEdgeSupported = false;  // currently this is never set, if we add VK_KHR_sampler_mirror_clamp_to_edge extension support or samplerMirrorClampToEdge feature flag then we can change this to non const and set/reset
    tGfxApi&                                m_GfxApi;
};
