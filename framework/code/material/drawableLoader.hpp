//============================================================================================================
//
//
//                  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include <functional>
#include <optional>
#include <span>
#include <string>
#include "mesh/instanceGenerator.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshHelper.hpp"
#include "mesh/meshIntermediate.hpp"
#include "pipeline.hpp"
#include "system/glm_common.hpp"
#include "system/os_common.h"
#include "texture/textureFormat.hpp"

// Forward Declarations
class AssetManager;
class MaterialBase;
struct MeshInstance;
class MeshObjectIntermediate;
template<typename T_GFXAPI> class Material;
template<typename T_GFXAPI> class Pipeline;
template<typename T_GFXAPI> class RenderContext;
template<typename T_GFXAPI> class RenderPass;
template<typename T_GFXAPI> class ShaderPass;

//class Dx12;
//class MaterialBase;
//enum class Msaa;
//template<typename T_GFXAPI> class Shader;

template<typename T_GFXAPI> class Drawable;
template<typename T_GFXAPI> class DrawableLoader;


/// Base (platform agnostic) wrapper class for loading drawables
class DrawableLoaderBase
{
    DrawableLoaderBase( const DrawableLoaderBase& ) = delete;
    DrawableLoaderBase& operator=( const DrawableLoaderBase& ) = delete;
    template<typename T_GFXAPI> using tApiDerived = DrawableLoader<T_GFXAPI>; // make apiCast work!
public:
    /// @brief Flags controlling the behaviour of LoadDrawables
    enum LoaderFlags : uint32_t {
        None = 0,
        FindInstances = 0x1,    // useInstancing pass true if drawable loader should try to find duplicated instances of meshes(same MaterialDef, same vertex uv sets, vertex positions onlly differing by rotation and translation). Can take a little time to process.
        BakeTransforms = 0x2,   // bake world transform in to mesh data (and clear the m_Transform for all baked drawables)
        IgnoreHierarchy = 0x4   // Ignore the gltf node hierarchy when loading model
    };

    /// @brief Print some combined statistics about the given meshObjects.
    /// @param meshObjects span of the objects we want to gather the statistics for.
    static void PrintStatistics( const std::span<MeshObjectIntermediate> meshObjects );

    struct MeshStatistics {
        size_t totalVerts;
        glm::vec3 boundingBoxMin;
        glm::vec3 boundingBoxMax;
    };

    /// @brief Collect some combined statistics about the given meshObjects.
    /// @param meshObjects span of the objects we want to gather the statistics for.
    /// @returns @DrawableLoaderMeshStatistics with statistics for the given objects (combined).
    static MeshStatistics GatherStatistics( const std::span<MeshObjectIntermediate> meshObjects );
};


/// Wrapper class for LoadDrawables (user entry point for loading a drawable mesh object).
/// @ingroup Material
template<typename T_GFXAPI>
class DrawableLoader : public DrawableLoaderBase
{
    DrawableLoader( const DrawableLoader<T_GFXAPI>& ) = delete;
    DrawableLoader<T_GFXAPI>& operator=(const DrawableLoader<T_GFXAPI>&) = delete;
    using Drawable = Drawable<T_GFXAPI>;
    using Material = Material<T_GFXAPI>;
    using RenderContext = RenderContext<T_GFXAPI>;
    using RenderPass = RenderPass<T_GFXAPI>;
    using PipelineRasterizationState = PipelineRasterizationState<T_GFXAPI>;
public:
    /// @brief Load a mesh object and create the @Drawable(s) for rendering it.
    /// This is the recommended way of loading meshes in to the Framework MaterialBase system.
    /// @param renderPasses span of render pass contexts that we may want to create DrawablePasses for (can be duplicated if we have subpasses)
    /// @param meshFilename name of the mesh filename to load (via assetManager)
    /// @param materialLoader user supplied function that returns a MaterialBase for each mesh material (@ref MeshObjectIntermediate::MaterialDef) that the loaded mesh returns.  Meshes can have multiple materials (possibly hundreds).
    /// @param drawables output vector of @Drawable objects
    /// @param renderPassMultisample optional multisample flags (if zero size assume no multisampling)
    /// @param loaderFlags loader feature enables
    /// @param globalScale global scale applied to every loaded Drawable object
    /// @return true on success
    static bool LoadDrawables(T_GFXAPI&, AssetManager& assetManager, std::span<const RenderContext> renderPasses, const std::string& meshFilename, const std::function<std::optional<Material>(const MeshObjectIntermediate::MaterialDef&)>& materialLoader, std::vector<Drawable>& drawables, /*LoaderFlags*/uint32_t loaderFlags, const glm::vec3 globalScale = glm::vec3(1.0f,1.0f,1.0f));
    static bool LoadDrawables(T_GFXAPI&, AssetManager& assetManager, std::span<const RenderContext> renderPasses, const std::string& meshFilename, const std::function<std::unique_ptr<MaterialBase>(const MeshObjectIntermediate::MaterialDef&)>& materialLoader, std::vector<Drawable>& drawables, /*LoaderFlags*/uint32_t loaderFlags, const glm::vec3 globalScale = glm::vec3(1.0f,1.0f,1.0f));

    /// @brief Load a mesh object and create the @Drawable(s) for rendering it.
    /// Identical to LoadDrawables but for a single pass only (helper to save end-user from creating spans with a single entry)
    static bool LoadDrawables(T_GFXAPI&, AssetManager& assetManager, const RenderContext& renderPass, const std::string& meshFilename, const std::function<std::optional<Material>( const MeshObjectIntermediate::MaterialDef& )>& materialLoader, std::vector<Drawable>& drawables, /*LoaderFlags*/uint32_t loaderFlags, const glm::vec3 globalScale = glm::vec3( 1.0f, 1.0f, 1.0f ) );
    static bool LoadDrawables(T_GFXAPI&, AssetManager& assetManager, const RenderContext& renderPass, const std::string& meshFilename, const std::function<std::unique_ptr<MaterialBase>( const MeshObjectIntermediate::MaterialDef& )>& materialLoader, std::vector<Drawable>& drawables, /*LoaderFlags*/uint32_t loaderFlags, const glm::vec3 globalScale = glm::vec3( 1.0f, 1.0f, 1.0f ) );

    /// @brief Create @Drawable(s) for rendering a given vector of @MeshObjectIntermediate objects.
    /// This is the recommended way of creating meshes in the Framework MaterialBase system and is used by the LoadDrawables function.
    /// @param intermediateMeshObjects vector of the (intermediate format) mesh objects we are going to make drawables from.  CreateDrawables takes ownership of this data.
    /// @param renderPasses span of render passes that we may want to create DrawablePasses for (can be duplicated if we have subpasses)
    /// @param renderPassNames names of the render passes, expected to match the names inside the MaterialBase 
    /// @param meshFilename name of the mesh filename to load (via assetManager)
    /// @param materialLoader user supplied function that returns a MaterialBase for each mesh material (@ref MeshObjectIntermediate::MaterialDef) that the loaded mesh returns.  Meshes can have multiple materials (possibly hundreds).
    /// @param drawables output vector of @Drawable objects
    /// @param renderPassMultisample optional multisample flags (if zero size assume no multisampling)
    /// @param loaderFlags loader feature enables
    /// @param RenderPassSubpasses subpass indices for each render pass (0 for first subpass of if there are no subpasses).  If empty treat everything as using subpass 0
    /// @return true on success
    static bool CreateDrawables(T_GFXAPI&, std::vector<MeshObjectIntermediate>&& intermediateMeshObjects, std::span<const RenderContext> renderPasses, const std::function<std::optional<Material>(const MeshObjectIntermediate::MaterialDef&)>& materialLoader, std::vector<Drawable>& drawables, /*DrawableLoader::LoaderFlags*/uint32_t loaderFlags);

    /// @brief Create @Drawables() for rendering the given @MeshInstance objects.
    /// Identical to CreateDrawables but does not generate the MeshInstance data (is required to be already generated).
    static bool CreateDrawables(T_GFXAPI&, std::vector<MeshInstance>&& intermediateMeshInstances, std::span<const RenderContext> renderPasses, const std::function<std::optional<Material>(const MeshObjectIntermediate::MaterialDef&)>& materialLoader, std::vector<Drawable>& drawables, /*DrawableLoader::LoaderFlags*/uint32_t loaderFlags);

    /// @brief Create @Drawables() for rendering the given @MeshInstance objects.
    /// Identical to CreateDrawables but for a single pass only (helper to save end-user from creating spans with a single entry)
    static bool CreateDrawables(T_GFXAPI&, std::vector<MeshObjectIntermediate>&& intermediateMeshObjects, const RenderContext& renderPass, const std::function<std::optional<Material>( const MeshObjectIntermediate::MaterialDef& )>& materialLoader, std::vector<Drawable>& drawables, /*DrawableLoader::LoaderFlags*/uint32_t loaderFlags);
};


template<typename T_GFXAPI>
bool DrawableLoader<T_GFXAPI>::LoadDrawables( T_GFXAPI& gfxapi, AssetManager& assetManager, std::span<const RenderContext> renderPasses, const std::string& meshFilename, const std::function<std::optional<Material>( const MeshObjectIntermediate::MaterialDef& )>& materialLoader, std::vector<Drawable>& drawables, /*DrawableLoader::LoaderFlags*/uint32_t loaderFlags, const glm::vec3 globalScale )
{
    LOGI( "Loading Object mesh: %s...", meshFilename.c_str() );

    std::vector<MeshObjectIntermediate> fatObjects;
    if (meshFilename.size() > 4 && meshFilename.substr( meshFilename.size() - 4 ) == std::string( ".obj" ))
    {
        // Load .obj file
        fatObjects = MeshObjectIntermediate::LoadObj( assetManager, meshFilename );
    }
    else
    {
        // Load .gltf file
        fatObjects = MeshObjectIntermediate::LoadGLTF( assetManager, meshFilename, (loaderFlags & DrawableLoader::LoaderFlags::IgnoreHierarchy) != 0, globalScale );
    }
    if (fatObjects.size() == 0)
    {
        LOGE( "Error loading Object mesh: %s", meshFilename.c_str() );
        return false;
    }
    // Print some debug
    DrawableLoader::PrintStatistics( fatObjects );

    // Turn the intermediate mesh objects into Drawables (and load the materials)
    if (!CreateDrawables( gfxapi, std::move( fatObjects ), renderPasses, materialLoader, drawables, loaderFlags ))
    {
        LOGE( "Error initializing Drawable: %s", meshFilename.c_str() );
        return false;
    }
    return true;    // success
}

template<typename T_GFXAPI>
bool DrawableLoader<T_GFXAPI>::LoadDrawables( T_GFXAPI& gfxapi, AssetManager& assetManager, std::span<const RenderContext> renderPasses, const std::string& meshFilename, const std::function<std::unique_ptr<MaterialBase>(const MeshObjectIntermediate::MaterialDef&)>& materialLoader, std::vector<Drawable>& drawables, /*DrawableLoader::LoaderFlags*/uint32_t loaderFlags, const glm::vec3 globalScale )
{
    auto materialLoader2 = [&materialLoader]( const MeshObjectIntermediate::MaterialDef& materialDef ) -> std::optional<Material>
    {
        auto pMaterial = materialLoader( materialDef );
        if (pMaterial)
        {
            return {std::move( *static_cast<Material*>(pMaterial.get()) )};
        }
        else
        {
            return std::nullopt;
        }
    };
    return LoadDrawables(gfxapi, assetManager, renderPasses, meshFilename, materialLoader2, drawables, loaderFlags, globalScale );
}

template<typename T_GFXAPI>
bool DrawableLoader<T_GFXAPI>::LoadDrawables( T_GFXAPI& gfxApi, AssetManager& assetManager, const RenderContext& renderPass, const std::string& meshFilename, const std::function<std::optional<Material>( const MeshObjectIntermediate::MaterialDef& )>& materialLoader, std::vector<Drawable>& drawables, /*LoaderFlags*/uint32_t loaderFlags, const glm::vec3 globalScale )
{
    return DrawableLoader<T_GFXAPI>::LoadDrawables( gfxApi, assetManager, {&renderPass, 1}, meshFilename, materialLoader, drawables, loaderFlags, globalScale );
}

template<typename T_GFXAPI>
bool DrawableLoader<T_GFXAPI>::LoadDrawables( T_GFXAPI& gfxApi, AssetManager& assetManager, const RenderContext& renderPass, const std::string& meshFilename, const std::function<std::unique_ptr<MaterialBase>( const MeshObjectIntermediate::MaterialDef& )>& materialLoader, std::vector<Drawable>& drawables, /*LoaderFlags*/uint32_t loaderFlags, const glm::vec3 globalScale )
{
    return DrawableLoader<T_GFXAPI>::LoadDrawables( gfxApi, assetManager, {&renderPass, 1}, meshFilename, materialLoader, drawables, loaderFlags, globalScale );
}

template<typename T_GFXAPI>
bool DrawableLoader<T_GFXAPI>::CreateDrawables( T_GFXAPI& gfxapi, std::vector<MeshObjectIntermediate>&& intermediateMeshObjects, std::span<const RenderContext> renderPasses, const std::function<std::optional<Material>( const MeshObjectIntermediate::MaterialDef& )>& materialLoader, std::vector<Drawable>& drawables, /*DrawableLoader::LoaderFlags*/uint32_t loaderFlags )
{
    // See if we can find instances, we assume there is no instance information in the gltf!
    auto instancedFatObjects = (loaderFlags & LoaderFlags::FindInstances) ? MeshInstanceGenerator::FindInstances( std::move( intermediateMeshObjects ) ) : MeshInstanceGenerator::NullFindInstances( std::move( intermediateMeshObjects ) );
    intermediateMeshObjects.clear();

    return CreateDrawables( gfxapi, std::move( instancedFatObjects ), renderPasses, materialLoader, drawables, loaderFlags );
}

template<typename T_GFXAPI>
bool DrawableLoader<T_GFXAPI>::CreateDrawables( T_GFXAPI& gfxapi, std::vector<MeshInstance>&& intermediateMeshInstances, std::span<const RenderContext> renderPasses, const std::function<std::optional<Material>( const MeshObjectIntermediate::MaterialDef& )>& materialLoader, std::vector<Drawable>& drawables, /*DrawableLoader::LoaderFlags*/uint32_t loaderFlags )
{
    // Create a pipeline container for each render pass, dont create the underlying pipeline until we know it is needed
    std::vector<Pipeline<T_GFXAPI>> pipelines;
    pipelines.resize( renderPasses.size() );

    drawables.reserve( intermediateMeshInstances.size() );
    for (auto& [fatObject, instances] : intermediateMeshInstances)
    {
        // Get the material for this mesh
        std::optional<Material> material;
        if (fatObject.m_Materials.size() == 0)
        {
            // default material (materialId = 0)
            auto loadedMaterial = materialLoader( MeshObjectIntermediate::MaterialDef{"", 0, "textures/white_d.ktx"} );
            if (loadedMaterial.has_value())
                material.emplace( std::move( loadedMaterial.value() ) );
        }
        else
        {
            if (fatObject.m_Materials.size() > 1)
            {
                LOGE( "  Drawable loader does not support per-face materials, using first material" );
            }
            auto loadedMaterial = materialLoader( fatObject.m_Materials[0] );
            if (loadedMaterial.has_value())
                material.emplace( std::move( loadedMaterial.value() ) );
        }
        if (!material)
        {
            // It is valid for the material loader to return a null material - denotes we dont want to render this object!
            continue;
        }
        const auto& shader = material->GetShader();

        // Determine which passes are being used
        uint32_t passMask = 0;  // max 32 passes!
        const ShaderPass<T_GFXAPI>* pFirstPass = nullptr;
        for (uint32_t pass = 0; pass < renderPasses.size(); ++pass)
        {
            const char* pRenderPassName = renderPasses[pass].name.c_str();
            const auto* pShaderPass = shader.GetShaderPass( pRenderPassName ); ///TODO: std::string generated here!
            if (pShaderPass)
            {
                passMask |= 1 << pass;
                // Make sure we have pipeline for this pass
                if (!pipelines[pass])
                {
                    const auto* materialPass = material->GetMaterialPass( pRenderPassName );
                    assert( materialPass );

                    PipelineRasterizationState rasterizationState{pShaderPass->m_shaderPassDescription};
                    pipelines[pass] = ::CreatePipeline( gfxapi,
                                                      pShaderPass->m_shaderPassDescription,
                                                      pShaderPass->GetPipelineLayout(),
                                                      pShaderPass->GetPipelineVertexInputState(),
                                                      rasterizationState,
                                                      materialPass->GetSpecializationConstants(),
                                                      pShaderPass->m_shaders,
                                                      renderPasses[pass],
                                                      renderPasses[pass].msaa );
                }
            }
            if (!pFirstPass)
                pFirstPass = pShaderPass;
        }

        // If this drawable was in one or more passes (hopefully it was used for something) then initialize them
        if (pFirstPass)
        {
            ///TODO: implement having different bindings and packing for different passes

            // Do the (optional) transform baking before creating the device mesh.
            if (true && (loaderFlags & LoaderFlags::BakeTransforms) != 0)
            {
                if ((instances.size() > 1) || ((loaderFlags & LoaderFlags::FindInstances) != 0))
                {
                    // When we are using instancing dont bake the transform into the mesh - apply the transform to each of the instance transforms.
                    // It is quite likely the fatObject.m_Transform matrix will be 'identity' as it should have already been applied while instances were being found.
                    glm::mat4x3 objectTransform4x3 = fatObject.m_Transform; // convert 4x4 to 4x3 (keeps rotation and translation, lost column is unimportant if the transform is a simple TRS (translation/rotation/scale) matrix
                    for (MeshObjectIntermediate::FatInstance& instance : instances)
                    {
                        instance.transform = instance.transform * objectTransform4x3;
                    }
                    fatObject.m_Transform = glm::identity<glm::mat4>();
                    fatObject.m_NodeId = -1;    // object (may) point to multiple instances so m_NodeId is likely not valid at the mesh level
                }
                else
                {
                    // Bake the object transform down into the mesh (instances[0] may well be identity but apply it to the transform incase it is not).
                    glm::mat4x3 combinedTransform = glm::transpose( instances[0].transform * glm::mat4x3( fatObject.m_Transform ) );
                    fatObject.m_Transform = combinedTransform;
                    instances[0].transform = glm::identity<MeshObjectIntermediate::FatInstance::tInstanceTransform>();
                    fatObject.BakeTransform();
                }
            }

            const auto nodeId = fatObject.m_NodeId; // grab the nodeId before it goes away!

            Mesh<T_GFXAPI> meshObject;
            const auto& vertexFormats = shader.m_shaderDescription->m_vertexFormats;
            MeshHelper::CreateMesh( gfxapi.GetMemoryManager(), fatObject, (uint32_t)pFirstPass->m_shaderPassDescription.m_vertexFormatBindings[0], vertexFormats, &meshObject );

            // We are done with the FatObject here, Release it to save some memory earlier.
            fatObject.Release();

            // Potentially the vertex format has some Instance bindings in here (will have been ignored by the CreateMesh).
            std::optional<VertexBuffer<T_GFXAPI>> vertexInstanceBuffer;
            const auto instanceFormatIt = std::find_if( vertexFormats.cbegin(), vertexFormats.cend(),
                []( const VertexFormat& f ) { return f.inputRate == VertexFormat::eInputRate::Instance; } );
            if (instanceFormatIt != vertexFormats.cend())
            {
                if (instanceFormatIt != vertexFormats.cbegin() && instanceFormatIt != vertexFormats.end() - 1)
                {
                    LOGE( "  Drawable loader (currently) only suports shaders with instance rate 'vertex' buffers at the beginning or end of their vertex layout" );
                    return false;
                }
                // Create the instance data
                std::span instancesSpan = instances;
                if (instancesSpan.empty())
                {
                    // Even if we are not instancing there should be one instance per mesh
                    LOGE( "  Drawable loader expected mesh to have (at least) one instance matrix" );
                    return false;
                }

                const std::vector<uint32_t> formattedVertexData = MeshObjectIntermediate::CopyFatInstanceToFormattedBuffer( instancesSpan, *instanceFormatIt );

                if (!vertexInstanceBuffer.emplace().Initialize( &gfxapi.GetMemoryManager(), instanceFormatIt->span, instancesSpan.size(), formattedVertexData.data() ))
                {
                    return false;
                }
            }
            else
            {
                if (instances.size() > 1)
                {
                    LOGE( "  Drawable loader found instances - expects shaders vertex layout to have instance data support" );
                    return false;
                }
            }

            // Create the drawable
            if (!drawables.emplace_back( gfxapi, std::move( material.value() ) ).Init( renderPasses, passMask, std::move( meshObject ), std::move( vertexInstanceBuffer ), std::nullopt, nodeId ))
            {
                return false;
            }
        }
    }
    return true;
}

template<typename T_GFXAPI>
bool DrawableLoader<T_GFXAPI>::CreateDrawables( T_GFXAPI& gfxApi, std::vector<MeshObjectIntermediate>&& intermediateMeshObjects, const RenderContext& renderPass, const std::function<std::optional<Material>( const MeshObjectIntermediate::MaterialDef& )>& materialLoader, std::vector<Drawable>& drawables, /*DrawableLoader::LoaderFlags*/uint32_t loaderFlags )
{
    return CreateDrawables( gfxApi, std::move( intermediateMeshObjects ), {&renderPass, 1}, std::move( materialLoader ), drawables, loaderFlags );
}
