//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include <cassert>
#include <memory>
#include "textureFormat.hpp"
#include "sampler.hpp"


// Forward declarations
class GraphicsApiBase;
template<typename T_GFXAPI> class Texture;
template<typename T_GFXAPI> class Image;
template<typename T_GFXAPI> class ImageView;
template<typename T_GFXAPI> class MemoryPool;
template<typename T_GFXAPI> class Sampler;


/// @brief Base class for a texture object (image with a sampler).
/// Is subclassed for each graphics API.
class TextureBase
{
    TextureBase(const TextureBase&) = delete;
    TextureBase& operator=(const TextureBase&) = delete;
public:
    template<typename T_GFXAPI> using tApiDerived = Texture<T_GFXAPI>; // make apiCast work!

    TextureBase() noexcept {}
    virtual ~TextureBase() noexcept = 0;

    virtual void Release(GraphicsApiBase* pApi) = 0;
};
inline TextureBase::~TextureBase() noexcept {}


/// @brief Texture container for a (templated) graphics api.
/// Owns memory and sampler etc associated with a single texture.
/// Template is expected to be specialized for the graphics api (Vulkan, DirectX etc)
template<typename T_GFXAPI>
class Texture final : public TextureBase
{
    Texture(const Texture<T_GFXAPI>&) = delete;
    Texture& operator=(const Texture<T_GFXAPI>&) = delete;
public:
    Texture() noexcept = delete;                                         // template class expected to be specialized
    Texture(Texture<T_GFXAPI>&&) noexcept = delete;                     // template class expected to be specialized
    Texture<T_GFXAPI>& operator=(Texture<T_GFXAPI>&&) noexcept = delete;// template class expected to be specialized

    bool IsEmpty() const { return true; }

protected:
    static_assert(sizeof(Texture<T_GFXAPI>) != sizeof(TextureBase));   // Ensure this class template is specialized (and not used as-is)
};


/// @brief Base class for a image  object.
/// Is subclassed for each graphics API.
class ImageBase
{
    ImageBase( const ImageBase& ) = delete;
    ImageBase& operator=( const ImageBase& ) = delete;
public:
    template<typename T_GFXAPI> using tApiDerived = Image<T_GFXAPI>; // make apiCast work!
protected:
    ImageBase() noexcept {}
};


/// @brief Template for image to be specialized by graphics api.
template<typename T_GFXAPI>
class Image final : public ImageBase
{
    Image( const Image<T_GFXAPI>& ) = delete;
    Image& operator=( const Image<T_GFXAPI>& ) = delete;
public:
    Image() noexcept = delete;                                             // template class expected to be specialized
    Image( Image<T_GFXAPI>&& ) noexcept = delete;                         // template class expected to be specialized
    Image<T_GFXAPI>& operator=( Image<T_GFXAPI>&& ) noexcept = delete;    // template class expected to be specialized

    static_assert(sizeof( Image<T_GFXAPI> ) != sizeof( ImageBase ));           // Ensure this class template is specialized (and not used as-is)
};


/// @brief Base class for a image view object.
/// Is subclassed for each graphics API.
class ImageViewBase
{
    ImageViewBase(const ImageViewBase&) = delete;
    ImageViewBase& operator=(const ImageViewBase&) = delete;
public:
    template<typename T_GFXAPI> using tApiDerived = ImageView<T_GFXAPI>; // make apiCast work!
protected:
    ImageViewBase() noexcept {}
};


/// @brief Template for image view to be specialized by graphics api.
template<typename T_GFXAPI>
class ImageView final : public ImageViewBase
{
    ImageView(const ImageView<T_GFXAPI>&) = delete;
    ImageView& operator=(const ImageView<T_GFXAPI>&) = delete;
public:
    ImageView() noexcept = delete;                                             // template class expected to be specialized
    ImageView( ImageView<T_GFXAPI>&&) noexcept = delete;                      // template class expected to be specialized
    ImageView<T_GFXAPI>& operator=( ImageView<T_GFXAPI>&&) noexcept = delete; // template class expected to be specialized

    static_assert(sizeof(ImageView<T_GFXAPI>) != sizeof(ImageViewBase));           // Ensure this class template is specialized (and not used as-is)
};


/// Texture types (intended use) for CreateTextureObject
enum TEXTURE_TYPE
{
    TT_NORMAL = 0,
    TT_SAMPLED_TRANSFERDST,
    TT_RENDER_TARGET,
    TT_RENDER_TARGET_WITH_STORAGE,
    TT_RENDER_TARGET_TRANSFERSRC,
    TT_RENDER_TARGET_SUBPASS,
    TT_RENDER_TARGET_LOCAL_READ,
    TT_RENDER_TARGET_LOCAL_READ_TRANSIENT,
    TT_DEPTH_TARGET,
    TT_DEPTH_TARGET_LOCAL_READ, 
    TT_COMPUTE_TARGET,
    TT_COMPUTE_STORAGE,
    TT_CPU_UPDATE,
    TT_SHADING_RATE_IMAGE,
    NUM_TEXTURE_TYPES
};


/// Texture flags (feature enable/disable) for CreateTextureObject
enum TEXTURE_FLAGS
{
    None = 0,
    MutableFormat = 1,
    ForceLinearTiling = 2
};


enum class ImageViewType {
    View1D = 0,
    View2D = 1,
    View3D = 2,
    ViewCube = 3,
    View1DArray = 4,
    View2DArray = 5,
    ViewCubeArray = 6,
};;



/// Parameters for CreateTextureObject
struct CreateTexObjectInfo
{
    uint32_t uiWidth = 0;
    uint32_t uiHeight = 0;
    uint32_t uiDepth = 1;	                                                // default to 2d texture.
    uint32_t uiMips = 1;                                                    // default to no mips
    uint32_t uiFaces = 1;	                                                // default to 1 face (ie not a texture array or cubemap)
    TextureFormat Format = TextureFormat::UNDEFINED;
    TEXTURE_TYPE TexType = TEXTURE_TYPE::TT_NORMAL;
    TEXTURE_FLAGS Flags = TEXTURE_FLAGS::None;
    const char* pName = nullptr;
    Msaa Msaa = Msaa::Samples1;                                             ///< number of msaa samples per pixel (ie 1, 2, 4, etc).  Default to 1 (no msaa)
    SamplerFilter FilterMode = SamplerFilter::Undefined;					//default to picking from a default dependant on the texture format
    SamplerAddressMode SamplerMode = SamplerAddressMode::Undefined;	        //default to picking from a default dependant on the texture type
    bool UnNormalizedCoordinates = false;
};


/// Create texture (generally for render target usage)
template<typename T_GFXAPI>
Texture<T_GFXAPI> CreateTextureObject( T_GFXAPI&, const CreateTexObjectInfo&, MemoryPool<T_GFXAPI>* pPool = nullptr)
{
    static_assert(sizeof( T_GFXAPI ) != sizeof( T_GFXAPI ), "Must use the specialized version of this function.  Your are likely missing #include \"texture/<GFXAPI>/texture.hpp\"");
    assert( 0 && "Expecting CreateTextureObject (per graphics api) to be used" );
    return {};
}

/// Create texture (generally for render target usage)
template<typename T_GFXAPI>
Texture<T_GFXAPI> CreateTextureObject(T_GFXAPI& gfxApi, uint32_t uiWidth, uint32_t uiHeight, TextureFormat Format, TEXTURE_TYPE TexType, const char* pName, Msaa Msaa = Msaa::Samples1, TEXTURE_FLAGS Flags = TEXTURE_FLAGS::None)
{
    CreateTexObjectInfo createInfo{};
    createInfo.uiWidth = uiWidth;
    createInfo.uiHeight = uiHeight;
    createInfo.Format = Format;
    createInfo.TexType = TexType;
    createInfo.Flags = Flags;
    createInfo.pName = pName;
    createInfo.Msaa = Msaa;
    return CreateTextureObject(gfxApi, createInfo);
}

/// Create texture (unique_ptr) (generally for render target usage).  Uses CreateTexObjectInfo structure to define texture creation parameters.
template<typename T_GFXAPI>
std::unique_ptr<TextureBase> CreateTextureObject(GraphicsApiBase& gfxApi, const CreateTexObjectInfo& texInfo)
{
    auto pTexture = std::make_unique<Texture<T_GFXAPI>>();
    *pTexture = std::move(CreateTextureObject(static_cast<T_GFXAPI&>(gfxApi), texInfo));
    return pTexture;
}

/// Create texture from a memory buffer.
template<typename T_GFXAPI>
Texture<T_GFXAPI> CreateTextureFromBuffer( T_GFXAPI& gfxApi, const void* pData, size_t DataSize, uint32_t Width, uint32_t Height, uint32_t Depth, TextureFormat Format, SamplerAddressMode SamplerMode, SamplerFilter Filter, const char* pName = nullptr )
{
    static_assert(sizeof( T_GFXAPI ) != sizeof( T_GFXAPI ), "Must use the specialized version of this function.  Your are likely missing #include \"texture/<GFXAPI>/texture.hpp\"");
    assert( 0 && "Expecting CreateTextureFromBuffer (per graphics api) to be used" );
    return {};
}

/// Create texture (unique_ptr) (generally for render target usage).  Uses CreateTexObjectInfo structure to define texture creation parameters.
template<typename T_GFXAPI>
std::unique_ptr<TextureBase> CreateTextureFromBuffer( GraphicsApiBase& gfxApi, const void* pData, size_t DataSize, uint32_t Width, uint32_t Height, uint32_t Depth, TextureFormat Format, SamplerAddressMode SamplerMode, SamplerFilter Filter, const char* pName = nullptr )
{
    auto pTexture = std::make_unique<Texture<T_GFXAPI>>();
    *pTexture = std::move( CreateTextureFromBuffer( static_cast<T_GFXAPI&>( gfxApi ), pData, DataSize, Width, Height, Depth, Format, SamplerMode, Filter, pName ) );
    return pTexture;
}

/// Create a texture that views (aliases) another texture but using a different texture format (must be 'related' formats, which formats are related is dependant on graphics api)
template<typename T_GFXAPI>
Texture<T_GFXAPI> CreateTextureObjectView( T_GFXAPI&, const Texture<T_GFXAPI>& original, TextureFormat viewFormat )
{
    static_assert(sizeof( T_GFXAPI ) != sizeof( T_GFXAPI ), "Must use the specialized version of this function.  Your are likely missing #include \"texture/<GFXAPI>/texture.hpp\"");
    assert( 0 && "Expecting CreateTextureObjectView (per graphics api) to be used" );
    return {};
}


/// Release a texture.
/// Must be specialized for the graphics api - will give linker error if called by application code.
template<typename T_GFXAPI>
void ReleaseTexture(T_GFXAPI& gfxApi, Texture<T_GFXAPI>*)
{
    static_assert(sizeof( T_GFXAPI ) != sizeof( T_GFXAPI ), "Must use the specialized version of this function.  Your are likely missing #include \"texture/<GFXAPI>/texture.hpp\"");
    assert( 0 && "Expecting ReleaseTexture (per graphics api) to be used" );
}


/// Release an image.
/// Must be specialized for the graphics api - will give linker error if called by application code.
template<typename T_GFXAPI>
void ReleaseImage( T_GFXAPI& gfxApi, Image<T_GFXAPI>* )
{
    static_assert(sizeof( T_GFXAPI ) != sizeof( T_GFXAPI ), "Must use the specialized version of this function.  Your are likely missing #include \"texture/<GFXAPI>/texture.hpp\"");
    assert( 0 && "Expecting ReleaseImage (per graphics api) to be used" );
}


/// Create a texture image view
/// Must be specialized for the graphics api - will give linker error if called by application code.
template<typename T_GFXAPI>
ImageView<T_GFXAPI> CreateImageView( T_GFXAPI& gfxApi, const Image< T_GFXAPI>& image, TextureFormat format, uint32_t numMipLevels, uint32_t baseMipLevel, uint32_t numFaces, uint32_t firstFace, ImageViewType viewType )
{
    static_assert(sizeof( T_GFXAPI ) != sizeof( T_GFXAPI ), "Must use the specialized version of this function.  Your are likely missing #include \"texture/<GFXAPI>/texture.hpp\"");
    assert( 0 && "Expecting CreateImageView (per graphics api) to be used" );
    return {};
}

/// Release a texture image view.
/// Must be specialized for the graphics api - will give linker error if called by application code.
template<typename T_GFXAPI>
void ReleaseImageView( T_GFXAPI& gfxApi, ImageView<T_GFXAPI>* )
{
    static_assert(sizeof( T_GFXAPI ) != sizeof( T_GFXAPI ), "Must use the specialized version of this function.  Your are likely missing #include \"texture/<GFXAPI>/texture.hpp\"");
    assert( 0 && "Expecting ReleaseImageView (per graphics api) to be used" );
}
