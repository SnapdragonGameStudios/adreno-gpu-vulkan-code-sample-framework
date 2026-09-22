// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

///
/// KTX image file loading for Dx12
/// 

#include "../loaderKtx.hpp"
#include "memory/dx12/memoryMapped.hpp"
#include <set>
#include <memory>
#include <d3d12.h>

// Forward declarations
class GraphicsApiBase;

struct ktxTexture;
class Dx12;
template<typename T_GFXAPI> class Texture;
template<typename T_GFXAPI> class Sampler;


/// @brief Class to handle loading KTX textures in to Vulkan memory
/// Generally applications will want a singleton of this class.
template<>
class TextureKtx<Dx12> : public TextureKtxBase
{
    TextureKtx(const TextureKtx<Dx12>&) = delete;
    TextureKtx& operator=(const TextureKtx<Dx12>&) = delete;
public:
    TextureKtx(Dx12&) noexcept;
    ~TextureKtx() noexcept override;

    /// @brief Initialize this loader class
    /// @return true on success
    bool Initialize() override;

    /// @brief Release (de-initialize) back to a clean (initializable) state
    void Release() override;

    /// @brief Do the necessary upload etc to go from a cpu texture representation to Vulkan format
    /// @param textureFile ktx file data we want to load as a vulkan texture
    /// @param sampler sampler that loaded texture will take OWNERSHIP of, may be VK_NULL_HANDLE (in which case LoadKtx creates an appropriate sampler)
    /// @returns a &TextureDx12, will be empty on failure
    Texture<Dx12> LoadKtx(Dx12& dx12, const TextureKtxFileWrapper& textureFile, Sampler<Dx12> sampler);

    /// @brief Load a ktx file and do the necessary upload etc to go from a cpu texture representation to Vulkan format
    /// @param filename of ktx (or ktx2) format file we want to load as a vulkan texture
    /// @param sampler sampler that loaded texture will take OWNERSHIP of, may be VK_NULL_HANDLE (in which case LoadKtx creates an appropriate sampler)
    /// @returns a &TextureDx12, will be empty on failure
    Texture<Dx12> LoadKtx(Dx12& dx12, AssetManager& assetManager, const char* const pFileName, Sampler<Dx12> sampler);

    /// @brief Run the Ktx2 transcoding step (if needed) 
    /// Will do nothing for textures that do not need transcoding.
    /// Performance will depend on ktx2 texture size and intermediate encoding format.
    /// @param fileData ktx file data loaded by TextureKtxBase::LoadData or similar.
    /// @return transcoded Ktx texture.
    TextureKtxFileWrapper Transcode(TextureKtxFileWrapper&& fileData);

protected:
    Dx12& m_Dx12;
};


///// @brief Function specialization
//template<>
//Texture<Dx12> TextureKtxBase::LoadKtx( Dx12& vulkan, const TextureKtxFileWrapper& textureFile, const Sampler<Dx12>& sampler )
//{
//    return apiCast<Dx12>( this )->LoadKtx( vulkan, textureFile, sampler );
//}
//
///// @brief Function specialization
//template<>
//Texture<Dx12> TextureKtxBase::LoadKtx( Dx12& vulkan, AssetManager& assetManager, const char* const pFileName, const Sampler<Dx12>& sampler )
//{
//    return apiCast<Dx12>( this )->LoadKtx( vulkan, assetManager, pFileName, sampler );
//}
