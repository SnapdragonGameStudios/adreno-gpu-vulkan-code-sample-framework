// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include "../texture.hpp"
#include "memory/dx12/memoryMapped.hpp"
#include "dx12/dx12.hpp"    ///TODO: can this be removed?

// Forward declarations
class Dx12;
enum DXGI_FORMAT;
enum D3D12_TEXTURE_LAYOUT;

using TextureDx12 = Texture<Dx12>;
using SamplerDx12 = Sampler<Dx12>;
template<typename T_GFXAPI, typename T_TYPE> class MemoryAllocatedBuffer;


/// @brief Template specialization of texture container for Dx12 graphics api.
template<>
class Texture<Dx12> final : public TextureBase
{
public:
    Texture() noexcept;
    Texture(const Texture<Dx12>&) = delete;
    Texture& operator=(const Texture<Dx12>&) = delete;
    Texture(Texture<Dx12>&&) noexcept;
    Texture& operator=(Texture<Dx12>&&) noexcept;
    ~Texture() noexcept;

    /// @brief Construct Texture from a pre-existing image resource.
    /// @param resourceBuffer - ownership passed to this Texture.
    /// @param sampler - ownership passed to this Texture.
    Texture(uint32_t width, uint32_t height, uint32_t depth, uint32_t mipLevels, TextureFormat format, D3D12_TEXTURE_LAYOUT textureLayout, MemoryAllocatedBuffer<Dx12, ID3D12Resource*> resourceBuffer /*, const Sampler<Dx12>& sampler*/) noexcept;

    void Release(GraphicsApiBase* pGfxApi) override;

    bool IsEmpty() const;

    auto* GetResource() const { return MemoryBuffer.GetResource(); }
    const auto& GetResourceViewDesc() const { return ResourceViewDesc; }

public:
    uint32_t                Width = 0;
    uint32_t                Height = 0;
    uint32_t                Depth = 0;
    uint32_t                MipLevels = 0;
    uint32_t                FirstMip = 0;
    TextureFormat           Format = TextureFormat::UNDEFINED;
    D3D12_TEXTURE_LAYOUT    TextureLayout{};
    D3D12_SHADER_RESOURCE_VIEW_DESC ResourceViewDesc{};

    MemoryAllocatedBuffer<Dx12, ID3D12Resource*> MemoryBuffer;
    //Sampler<Dx12>          Sampler;
};

template<>
Texture<Dx12> CreateTextureObject<Dx12>(Dx12&, const CreateTexObjectInfo&, MemoryPool<Dx12>*);

/// Template specialization for Dx12 CreateTextureFromBuffer
template<>
Texture<Dx12> CreateTextureFromBuffer<Dx12>(Dx12&, const void* pData, size_t DataSize, uint32_t Width, uint32_t Height, uint32_t Depth, TextureFormat Format, SamplerAddressMode SamplerMode, SamplerFilter Filter, const char* pName);

/// Create a texture that views (aliases) another texture but using a different texture format (must be 'related' formats, which formats are related is dependant on graphics api)
/// Template specialization for Dx12 CreateTextureFromBuffer
template<>
Texture<Dx12> CreateTextureObjectView(Dx12&, const Texture<Dx12>& original, TextureFormat viewFormat );

/// Template specialization for Dx12 CreateTextureFromBuffer
template<>
void ReleaseTexture<Dx12>(Dx12&, Texture<Dx12>* pTexture);


DXGI_FORMAT TextureFormatToDx(TextureFormat f);
