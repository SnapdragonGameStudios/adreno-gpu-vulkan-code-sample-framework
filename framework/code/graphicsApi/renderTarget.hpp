//=============================================================================
//
//                  Copyright (c) 2024 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "texture/textureFormat.hpp"
#include "texture/texture.hpp"
#include <optional>
#include <span>

class GraphicsApiBase;
template<typename T_GFXAPI> class RenderTarget;

class RenderTargetBase;

struct RenderTargetInitializeInfo
{
    uint32_t                                Width = 0;
    uint32_t                                Height = 0;
    std::span<const TextureFormat>          LayerFormats = {};
    TextureFormat                           DepthFormat = TextureFormat::UNDEFINED;
    RenderTargetBase*                       InheritedDepthAttachment = nullptr;
    std::span<const TEXTURE_TYPE>           TextureTypes = {};
    std::optional<TEXTURE_TYPE>             DepthTextureType = std::nullopt;
    std::span<const Msaa>                   Msaa = {};
    std::span<const TextureFormat>          ResolveTextureFormats = {};
    std::span<const SamplerFilter>          FilterModes = {};
};

class RenderTargetBase
{
    RenderTargetBase( const RenderTargetBase& ) = delete;
    RenderTargetBase& operator=( const RenderTargetBase& ) = delete;
public:
    RenderTargetBase() = default;
    virtual ~RenderTargetBase() = default;
    template<class T_GFXAPI> using tApiDerived = RenderTarget<T_GFXAPI>; // make apiCast work!
};


template<typename T_GFXAPI>
class RenderTarget : public RenderTargetBase
{
protected:
    static_assert(sizeof( RenderTarget<T_GFXAPI> ) != sizeof( RenderTargetBase ));   // Ensure this class template is specialized (and not used as-is)
};
