// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#include "textureManager.hpp"
#include "texture.hpp"
#include "loaderKtx.hpp"
#include "../loaderPpm.hpp"
#include "vulkan/vulkan.hpp"
#include "memory/memory.hpp"


//-----------------------------------------------------------------------------
TextureManager<Vulkan>::TextureManager( Vulkan& rGfxApi, AssetManager& rAssetManager ) noexcept
    : TextureManagerBase(rAssetManager)
    , m_GfxApi( rGfxApi )
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
TextureManager<Vulkan>::~TextureManager()
//-----------------------------------------------------------------------------
{
    Release();
}

//-----------------------------------------------------------------------------
bool TextureManager<Vulkan>::Initialize()
//-----------------------------------------------------------------------------
{
    m_Loader = std::make_unique<TextureKtx<Vulkan>>( static_cast<Vulkan&>(m_GfxApi) );

    m_DefaultSamplers.reserve( size_t(SamplerAddressMode::End) );
    for (size_t i = (size_t)SamplerAddressMode::Repeat; i < (size_t)SamplerAddressMode::End; ++i)
    {
        SamplerAddressMode samplerMode = (SamplerAddressMode)i;
        if (samplerMode != SamplerAddressMode::MirroredClampEdge || m_MirrorClampToEdgeSupported)
            m_DefaultSamplers.push_back( CreateSampler( m_GfxApi, samplerMode, SamplerFilter::Linear, SamplerBorderColor::TransparentBlackFloat, 0.0f ) );
        else
            m_DefaultSamplers.push_back( {} );
    }

    return TextureManagerBase::Initialize(0/*as many threads as cores*/);
}

//-----------------------------------------------------------------------------
void TextureManager<Vulkan>::Release()
//-----------------------------------------------------------------------------
{
    for (auto& [key, texture] : m_LoadedTextures)
    {
        texture.Release(&m_GfxApi);
    }
    m_LoadedTextures.clear();
    m_Loader->Release();
    TextureManagerBase::Release();
}

//-----------------------------------------------------------------------------
const TextureBase* TextureManager<Vulkan>::GetTexture(const std::string& textureSlotName) const
//-----------------------------------------------------------------------------
{
    auto iter = m_LoadedTextures.find(textureSlotName);
    if (iter == m_LoadedTextures.end())
        return nullptr;
    return &iter->second;
}

//-----------------------------------------------------------------------------
const SamplerBase* const TextureManager<Vulkan>::GetSampler( SamplerAddressMode sam ) const
//-----------------------------------------------------------------------------
{
    return &m_DefaultSamplers[size_t(sam)];
}

//-----------------------------------------------------------------------------
const TextureBase* TextureManager<Vulkan>::GetOrLoadTexture_(const std::string& textureSlotName, const std::string& filename, const SamplerBase& sampler)
//-----------------------------------------------------------------------------
{
    const TextureBase* pTexture = GetTexture(textureSlotName);
    if (!pTexture)
    {
        const SamplerVulkan& samplerVulkan = static_cast<const SamplerVulkan&>(sampler);
        auto loadedTexture = GetLoader()->LoadKtx(m_GfxApi, m_AssetManager, filename.c_str(), std::move(samplerVulkan.Copy()));
        if (!loadedTexture.IsEmpty())
        {
            auto insertedIt = m_LoadedTextures.insert({ textureSlotName, std::move(loadedTexture) });
            pTexture = &(insertedIt.first->second);
        }
    }

    return pTexture;
}

typedef std::queue<std::pair<size_t, TextureKtxFileWrapper>> tLoadedFileQueue;
struct BatchLoadThreadParams {
    AssetManager& assetManager;
    std::mutex& loadedFileQueueMutex;
    tLoadedFileQueue& loadedFileQueue;
    Semaphore& dataReadySema;
    const std::string& filename;
    size_t slotIndex;;
};

//-----------------------------------------------------------------------------
void TextureManager<Vulkan>::BatchLoad(const std::span<std::pair<std::string/*textureSlotName*/, std::string/*filename*/>> slotAndFileNames, const SamplerBase& defaultSampler)
//-----------------------------------------------------------------------------
{
    std::mutex loadedFileQueueMutex;
    tLoadedFileQueue loadedFileQueue;
    Semaphore dataReadySema{ 0 };
    Semaphore finishedSema{ 0 };

    // Setup the output textures
    std::vector<Texture> vulkanTextures;
    vulkanTextures.resize(slotAndFileNames.size());

    // We have one worker job just grabbing loaded textures and transfering them to vulkan (gpu memory).
    struct TransferWorkerParams {
        std::mutex& loadedFileQueueMutex;
        tLoadedFileQueue& loadedFileQueue;
        Semaphore& dataReadySema;
        Semaphore& finishedSema;
        const SamplerVulkan& defaultSampler;
        std::vector<Texture>& vulkanTextures;
    } transferWorkerParams{ loadedFileQueueMutex, loadedFileQueue, dataReadySema, finishedSema, apiCast<Vulkan>(defaultSampler), vulkanTextures };

    m_LoadingThreadWorker.DoWork2([](TextureManagerVulkan* pThis, TransferWorkerParams params)
    {
        size_t texturesRemaining = params.vulkanTextures.size();
        while (texturesRemaining > 0)
        {
            TextureKtxFileWrapper ktxData;
            size_t slotIndex;
            params.dataReadySema.Wait();
            {
                std::lock_guard<std::mutex> lock(params.loadedFileQueueMutex);
                auto&& loadedData = params.loadedFileQueue.front();
                ktxData = std::move(loadedData.second);
                slotIndex = loadedData.first;
                params.loadedFileQueue.pop();
            }
            if (ktxData)
                params.vulkanTextures[slotIndex] = pThis->GetLoader()->LoadKtx(pThis->m_GfxApi, ktxData, std::move(params.defaultSampler.Copy()) );
            --texturesRemaining;
        }
        params.finishedSema.Post();

    }, this, transferWorkerParams);

    size_t currentSlotIndex = 0;
    for (const auto& [textureSlotName, filename] : slotAndFileNames)
    {
        auto iter = m_LoadedTextures.find(textureSlotName);
        if (iter == m_LoadedTextures.end())
        {
            BatchLoadThreadParams params{ m_AssetManager, loadedFileQueueMutex, loadedFileQueue, dataReadySema, filename, currentSlotIndex };

            m_LoadingThreadWorker.DoWork2([](TextureManagerVulkan* pThis, BatchLoadThreadParams params)
            {
                auto ktxData = pThis->m_Loader->LoadFile(params.assetManager, params.filename.c_str());
                auto* pKtxLoader = static_cast<TextureKtx<Vulkan>*>(pThis->GetLoader());
                ktxData = pKtxLoader->Transcode(std::move(ktxData));
                {
                    std::lock_guard<std::mutex> lock(params.loadedFileQueueMutex);
                    params.loadedFileQueue.emplace(std::pair{ params.slotIndex, std::move(ktxData) });
                }
                params.dataReadySema.Post();
            }, this, params);

            ++currentSlotIndex;
        }
        else
        {
            // Signal the transfer thread, but dont give it any data (it finishes when it has recieved a signal for every texture being BatchLoaded)
            dataReadySema.Post();
        }
    }

    finishedSema.Wait();

    // Transfer all the loaded textures to m_LoadedTextures 
    for (size_t i = 0; i < vulkanTextures.size(); ++i)
    {
        if (!vulkanTextures[i].IsEmpty())
        {
            m_LoadedTextures.emplace(std::pair<std::string, Texture>{ slotAndFileNames[i].first/*slot*/, std::move(vulkanTextures[i])});
        }
    }
}

//-----------------------------------------------------------------------------
const TextureBase* TextureManager<Vulkan>::CreateTextureObject( const CreateTexObjectInfo& texInfo) /*override*/
//-----------------------------------------------------------------------------
{
    auto texture = ::CreateTextureObject( m_GfxApi, texInfo);

    assert( texInfo.pName!=nullptr && texInfo.pName[0] != '\0' ); // must have a valid name
    auto it = m_LoadedTextures.try_emplace( texInfo.pName, std::move( texture ) );
    if (!it.second)
    {
        assert( 0 && "CreateTextureObjectView duplicate texture name, must be unique (or use ::CreateTextureObject)" );
        return nullptr;
    }
    else
        return &it.first->second;
}

//-----------------------------------------------------------------------------
const TextureBase* TextureManager<Vulkan>::CreateTextureFromBuffer( const void* pData, size_t DataSize, uint32_t Width, uint32_t Height, uint32_t Depth, TextureFormat Format, SamplerAddressMode SamplerMode, SamplerFilter Filter, std::string name )
//-----------------------------------------------------------------------------
{
    auto texture = ::CreateTextureFromBuffer( m_GfxApi, pData, DataSize, Width, Height, Depth, Format, SamplerMode, Filter, name.c_str() );

    assert( !name.empty() ); // must have a valid name
    auto it = m_LoadedTextures.try_emplace( name, std::move(texture) );
    if (!it.second)
    {
        assert( 0 && "CreateTextureFromBuffer duplicate texture name, must be unique (or use ::CreateTextureFromBuffer)" );
        return nullptr;
    }
    else
        return &it.first->second;
}

//-----------------------------------------------------------------------------
const TextureBase* TextureManager<Vulkan>::CreateTextureObjectView( const TextureBase& original, TextureFormat viewFormat, std::string name )
//-----------------------------------------------------------------------------
{
    auto texture = ::CreateTextureObjectView( m_GfxApi, apiCast<Vulkan>(original), viewFormat );

    assert( !name.empty() ); // must have a valid name
    auto it = m_LoadedTextures.try_emplace( name, std::move( texture ) );
    if (!it.second)
    {
        assert( 0 && "CreateTextureObjectView duplicate texture name, must be unique (or use ::CreateTextureObjectView)" );
        return nullptr;
    }
    else
        return &it.first->second;
}
