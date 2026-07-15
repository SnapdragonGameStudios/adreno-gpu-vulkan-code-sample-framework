//============================================================================================================
//
//
//                  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include <optional>
#include <span>
#include "texture/textureFormat.hpp" //for Msaa

// Forward Declarations
class MaterialBase;
template<typename T_GFXAPI> class CommandList;
template<typename T_GFXAPI> class DrawablePass;
template<typename T_GFXAPI> class DrawIndirectBuffer;
template<typename T_GFXAPI> class Material;
template<typename T_GFXAPI> class Mesh;
template<typename T_GFXAPI> class Pipeline;
template<typename T_GFXAPI> class RenderContext;
template<typename T_GFXAPI> class RenderPass;
template<typename T_GFXAPI> class VertexBuffer;


class DrawablePassVertexBuffersBase
{};

template<typename T_GFXAPI>
class DrawablePassVertexBuffers : public DrawablePassVertexBuffersBase
{
    static_assert(sizeof( DrawablePassVertexBuffers ) != sizeof( DrawablePassVertexBuffersBase ));  // static assert if not specialized
};


/// Encapsulates a drawable object, owns the material (multiple passes, descriptor sets, etc) and the mesh (vertex list).
/// @ingroup Material
template<typename T_GFXAPI>
class Drawable
{
    Drawable( const Drawable& ) = delete;
    Drawable& operator=( const Drawable& ) = delete;
    using CommandList = CommandList<T_GFXAPI>;
    using DrawablePass = DrawablePass<T_GFXAPI>;
    using DrawIndirectBuffer = DrawIndirectBuffer<T_GFXAPI>;
    using DrawablePassVertexBuffers = DrawablePassVertexBuffers<T_GFXAPI>;
    using Mesh = Mesh<T_GFXAPI>;
    using Material = Material<T_GFXAPI>;
    using Pipeline = Pipeline<T_GFXAPI>;
    using RenderContext = RenderContext<T_GFXAPI>;
    using RenderPass = RenderPass<T_GFXAPI>;
    using VertexBuffer = VertexBuffer<T_GFXAPI>;
public:
    Drawable( T_GFXAPI&, Material&& material ) noexcept;
    Drawable( T_GFXAPI&, std::unique_ptr<MaterialBase> material ) noexcept;
    Drawable( Drawable&& ) noexcept;
    ~Drawable();

    /// Initialize this Drawable with the given mesh and single render pass.
    bool Init( const RenderPass& renderPass, const Pipeline&, const std::string& passName, Mesh meshObject, std::optional<VertexBuffer> vertexInstanceBuffer = std::nullopt, std::optional<DrawIndirectBuffer> indirectDrawBuffer = std::nullopt, int nodeId = -1 );


    /// Initialize this Drawable with the given mesh and single render pass.
    bool Init( const RenderContext& renderPass, Mesh meshObject, std::optional<VertexBuffer> vertexInstanceBuffer = std::nullopt, std::optional<DrawIndirectBuffer> indirectDrawBuffer = std::nullopt, int nodeId = -1 );
    /// Initialize this Drawable with the given mesh and render passes.
    bool Init( std::span<const RenderContext> renderPasses, uint32_t passMask, Mesh meshObject, std::optional<VertexBuffer> vertexInstanceBuffer = std::nullopt, std::optional<DrawIndirectBuffer> indirectDrawBuffer = std::nullopt, int nodeId = -1 );
    /// Initialize this Drawable without a mesh
    bool Init( std::span<const RenderContext> renderPasses, uint32_t passMask, std::optional<DrawIndirectBuffer> indirectDrawBuffer = std::nullopt, int nodeId = -1 );
    /// Initialize the mesh shader variant of the pipeline.
    bool InitMeshShader( std::span<const RenderContext> renderPasses, uint32_t passMask, std::optional<DrawIndirectBuffer> indirectDrawBuffer = std::nullopt, int nodeId = -1);

    /// Re-initialize the drawable.  Used internally by Init but can be used by the user when the render pass has been modified.
    bool ReInit( const RenderContext& renderPass );
    /// Re-initialize the drawable.  Used internally by Init but can be used by the user when the render passes have been modified.
    bool ReInit( std::span<const RenderContext> renderPasses, uint32_t passMask );

    bool ReInitMeshShader(std::span<const RenderContext> renderPasses, uint32_t passMask);

    /// Issues the commands needed to draw this DrawablePass.
    /// Binds the pipeline, descriptor sets, vertex buffers, index buffers, and issues the appropriate vkCmdDraw*
    /// @param vertexBindingsOverride allows user to replace @DrawablePass::mVertexBuffers with their own.  Span is for each 'bufferIdx' (can be size()==1 if all buffers bind the same).  DrawablePassVertexBuffers contains multiple buffers so all  mesh and instance streams are overridden.
    void DrawPass(CommandList& cmdList, const DrawablePass& drawablePass, uint32_t bufferIdx, const std::span<DrawablePassVertexBuffers> vertexBuffersOverride = {} ) const;

    const DrawablePass* GetDrawablePass( const std::string& passName ) const;
    const auto& GetDrawablePasses() const       { return mPasses; };
    uint32_t GetPassMask() const                { return mPassMask; }
    const auto& GetMaterial() const             { return mMaterial; }
    const auto& GetMeshObject() const { return mMeshObject; }
    auto& GetMeshObject()                       { return mMeshObject; }
    const auto& GetInstances() const            { return mVertexInstanceBuffer; }
    auto& GetInstances()                        { return mVertexInstanceBuffer; }
    const auto& GetDrawIndirectBuffer() const   { return mDrawIndirectBuffer; }
    const int GetNodeId() const                 { return mNodeId; }

    /// number of workgroup dispatches to execute (value after the local workgroup sizes are accounted for)
    void SetDispatchGroupCount(std::array<uint32_t, 3> count) { mDispatchGroupCount = count; }
    const auto& GetDispatchGroupCount() const   { return mDispatchGroupCount; }

public:
    Material                                    mMaterial;
    Mesh                                        mMeshObject;

protected:
    std::vector<DrawablePass>                   mPasses;
    std::map<std::string, uint32_t>             mPassNameToIndex;   // Index in to mpasses  ///TODO: allow for generation of a list of these - so each pass can iterate through the passes easily
    uint32_t                                    mPassMask = 0;
    int                                         mNodeId = -1;       // Identifier used by application to determine what this drawable is attached to, eg for attaching to animations.  Not used by Drawable.
    std::array<uint32_t, 3>                     mDispatchGroupCount{1u,1u,1u};

    std::optional<VertexBuffer>                 mVertexInstanceBuffer;
    std::optional<DrawIndirectBuffer>           mDrawIndirectBuffer;

    T_GFXAPI&                                   mGfxApi;
};


class DrawablePassBase
{
    DrawablePassBase( const DrawablePassBase& ) = delete;
    DrawablePassBase& operator=( const DrawablePassBase& ) = delete;
protected:
    DrawablePassBase() = default;
    DrawablePassBase& operator=( DrawablePassBase&& ) noexcept = default;
    DrawablePassBase( DrawablePassBase&& ) noexcept = default;
};

/// Encapsulates a drawable pass.
/// We require that this template is specialized by the graphics api (and this class not used as-is, in it's non specialized form)
/// @ingroup Material
template<typename T_GFXAPI>
class DrawablePass : public DrawablePassBase
{
    DrawablePass( const DrawablePass& ) = delete;
    DrawablePass& operator=( const DrawablePass& ) = delete;
public:
    DrawablePass() = delete;
    static_assert(sizeof( DrawablePass<T_GFXAPI> ) != sizeof( DrawablePassBase ));  // static assert if not specialized
};



template<typename T_GFXAPI>
Drawable<T_GFXAPI>::Drawable( T_GFXAPI& gfxapi, Material&& material ) noexcept
    : mMaterial( std::move( material ) )
    , mGfxApi( gfxapi )
{
}

template<typename T_GFXAPI>
Drawable<T_GFXAPI>::Drawable( T_GFXAPI& gfxapi, std::unique_ptr<MaterialBase> material ) noexcept
    : mMaterial( std::move(*apiCast<T_GFXAPI>(material.get()) ) )
    , mGfxApi( gfxapi )
{
}

template<typename T_GFXAPI>
Drawable<T_GFXAPI>::Drawable( Drawable<T_GFXAPI>&& other ) noexcept
    : mMaterial( std::move( other.mMaterial ) )
    , mGfxApi( other.mGfxApi )
    , mMeshObject( std::move( other.mMeshObject ) )
    , mPasses( std::move( other.mPasses ) )
    , mPassNameToIndex( std::move( other.mPassNameToIndex ) )
    , mPassMask( other.mPassMask )
    , mVertexInstanceBuffer( std::move( other.mVertexInstanceBuffer ) )
    , mDrawIndirectBuffer( std::move( other.mDrawIndirectBuffer ) )
{
    other.mPassMask = 0;
}

template<typename T_GFXAPI>
bool Drawable<T_GFXAPI>::Init( const RenderPass& renderPass, const Pipeline& pipeline, const std::string& passName, Mesh meshObject, std::optional<VertexBuffer> vertexInstanceBuffer, std::optional<DrawIndirectBuffer> indirectDrawBuffer, int nodeId )
{
    return Init( RenderContext{renderPass.Copy(), pipeline.Copy(), {}, passName}, std::move( meshObject ), std::move( vertexInstanceBuffer ), std::move( indirectDrawBuffer ), nodeId );
}

template<typename T_GFXAPI>
bool Drawable<T_GFXAPI>::Init( const RenderContext& renderPass, Mesh meshObject, std::optional<VertexBuffer> vertexInstanceBuffer, std::optional<DrawIndirectBuffer> drawIndirectBuffer, int nodeId )
{
    mMeshObject = std::move( meshObject );
    mVertexInstanceBuffer = std::move( vertexInstanceBuffer );
    mDrawIndirectBuffer = std::move( drawIndirectBuffer );
    mNodeId = nodeId;
    return ReInit( renderPass );
}

template<typename T_GFXAPI>
bool Drawable<T_GFXAPI>::Init( std::span<const RenderContext> renderPasses, uint32_t passMask, Mesh meshObject, std::optional<VertexBuffer> vertexInstanceBuffer, std::optional<DrawIndirectBuffer> drawIndirectBuffer, int nodeId )
{
    mMeshObject = std::move( meshObject );
    mVertexInstanceBuffer = std::move( vertexInstanceBuffer );
    mDrawIndirectBuffer = std::move( drawIndirectBuffer );
    mNodeId = nodeId;
    return ReInit( renderPasses, passMask );
}

template<typename T_GFXAPI>
bool Drawable<T_GFXAPI>::Init( std::span<const RenderContext> renderPasses, uint32_t passMask, std::optional<DrawIndirectBuffer> drawIndirectBuffer, int nodeId )
{
    mMeshObject = {};
    mVertexInstanceBuffer = {};
    mDrawIndirectBuffer = std::move( drawIndirectBuffer );
    mNodeId = nodeId;

    return ReInit( renderPasses, passMask );
}

template<typename T_GFXAPI>
bool Drawable<T_GFXAPI>::InitMeshShader( std::span<const RenderContext> renderPasses, uint32_t passMask, std::optional<DrawIndirectBuffer> drawIndirectBuffer, int nodeId )
{
    mMeshObject = {};
    mVertexInstanceBuffer = {};
    mDrawIndirectBuffer = std::move( drawIndirectBuffer );
    mNodeId = nodeId;

    return ReInitMeshShader( renderPasses, passMask );
}

template<typename T_GFXAPI>
bool Drawable<T_GFXAPI>::ReInit( const RenderContext& renderPass )
{
    return ReInit( {&renderPass,1}, 1);
}

template<typename T_GFXAPI>
const DrawablePass<T_GFXAPI>* Drawable<T_GFXAPI>::GetDrawablePass( const std::string& passName ) const
{
    auto it = mPassNameToIndex.find( passName );
    if (it != mPassNameToIndex.end())
    {
        return &mPasses[it->second];
    }
    return nullptr;
}
