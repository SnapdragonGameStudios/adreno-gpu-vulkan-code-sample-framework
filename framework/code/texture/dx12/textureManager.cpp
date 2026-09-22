// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#include "textureManager.hpp"
#include "sampler.hpp"
#include "texture.hpp"
#include "loaderKtx.hpp"
#include "../loaderPpm.hpp"
#include "dx12/dx12.hpp"


TextureManager<Dx12>::TextureManager(tGfxApi& rGfxApi, AssetManager& rAssetManager) noexcept : TextureManagerBase(rAssetManager), m_GfxApi(rGfxApi)
{
}

bool TextureManager<Dx12>::Initialize()
{
    m_Loader = std::make_unique<TextureKtx<tGfxApi>>( static_cast<tGfxApi&>(m_GfxApi) );
    return TextureManagerBase::Initialize();
}

void TextureManager<Dx12>::Release()
{
    for (auto& [key, texture] : m_LoadedTextures)
    {
        texture.Release(&m_GfxApi);
    }
    m_LoadedTextures.clear();
    m_Loader->Release();
    TextureManagerBase::Release();
}

const TextureBase* TextureManager<Dx12>::GetTexture(const std::string& textureSlotName) const
{
    auto iter = m_LoadedTextures.find(textureSlotName);
    if (iter == m_LoadedTextures.end())
        return nullptr;
    return &iter->second;
}

const TextureBase* TextureManager<Dx12>::GetOrLoadTexture_(const std::string& textureSlotName, const std::string& filename, const SamplerBase& sampler)
{
    const TextureBase* pTexture = GetTexture(textureSlotName);
    if (!pTexture)
    {
        const SamplerDx12& samplerDx12 = static_cast<const SamplerDx12&>(sampler);
        auto loadedTexture = GetLoader()->LoadKtx(m_GfxApi, m_AssetManager, filename.c_str(), samplerDx12);
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
    size_t slotIndex;
};

void TextureManager<Dx12>::BatchLoad(const std::span<std::pair<std::string/*textureSlotName*/, std::string/*filename*/>> slotAndFileNames, const SamplerBase& defaultSampler)
{
    std::mutex loadedFileQueueMutex;
    tLoadedFileQueue loadedFileQueue;
    Semaphore dataReadySema{ 0 };
    Semaphore finishedSema{ 0 };

    // Setup the output textures
    std::vector<Texture> textures;
    textures.resize(slotAndFileNames.size());

    // We have one worker job just grabbing loaded textures and transfering them to vulkan (gpu memory).
    struct TransferWorkerParams {
        std::mutex& loadedFileQueueMutex;
        tLoadedFileQueue& loadedFileQueue;
        Semaphore& dataReadySema;
        Semaphore& finishedSema;
        const SamplerDx12& defaultSampler;
        std::vector<Texture>& textures;
    } transferWorkerParams{ loadedFileQueueMutex, loadedFileQueue, dataReadySema, finishedSema, apiCast<Dx12>(defaultSampler), textures };

    m_LoadingThreadWorker.DoWork2([](TextureManagerDx12* pThis, TransferWorkerParams params)
    {
        size_t texturesRemaining = params.textures.size();
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
                params.textures[slotIndex] = pThis->GetLoader()->LoadKtx(pThis->m_GfxApi, ktxData, params.defaultSampler);
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

            m_LoadingThreadWorker.DoWork2([](TextureManagerDx12* pThis, BatchLoadThreadParams params)
            {
                auto ktxData = pThis->m_Loader->LoadFile(params.assetManager, params.filename.c_str());
                auto* pKtxLoader = apiCast<Dx12>( pThis->GetLoader() );
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
    for (size_t i = 0; i < textures.size(); ++i)
    {
        if (!textures[i].IsEmpty())
        {
            m_LoadedTextures.emplace(std::pair<std::string, Texture>{ slotAndFileNames[i].first/*slot*/, std::move(textures[i])});
        }
    }
}

const TextureBase* TextureManager<Dx12>::CreateTextureObject( const CreateTexObjectInfo& texInfo ) /*override*/
{
    auto texture = ::CreateTextureObject( m_GfxApi, texInfo );

    assert( texInfo.pName != nullptr && texInfo.pName[0] != '\0' ); // must have a valid name
    auto it = m_LoadedTextures.try_emplace( texInfo.pName, std::move( texture ) );
    if (!it.second)
    {
        assert( 0 && "CreateTextureObjectView duplicate texture name, must be unique (or use ::CreateTextureObject)" );
        return nullptr;
    }
    else
        return &it.first->second;
}

const TextureBase* TextureManager<Dx12>::CreateTextureFromBuffer( const void* pData, size_t DataSize, uint32_t Width, uint32_t Height, uint32_t Depth, TextureFormat Format, SamplerAddressMode SamplerMode, SamplerFilter Filter, std::string name ) /*override*/
{
    auto texture = ::CreateTextureFromBuffer( m_GfxApi, pData, DataSize, Width, Height, Depth, Format, SamplerMode, Filter, name.c_str() );

    assert( name.empty() ); // must have a valid name
    auto it = m_LoadedTextures.try_emplace( name, std::move( texture ) );
    if (!it.second)
    {
        assert( 0 && "CreateTextureObjectView duplicate texture name, must be unique (or use ::CreateTextureObjectView)" );
        return nullptr;
    }
    else
        return &it.first->second;
}

const TextureBase* TextureManager<Dx12>::CreateTextureObjectView( const TextureBase& original, TextureFormat viewFormat, std::string name )
{
    assert( 0 && "TextureManager<Dx12>::CreateTextureObjectView needs implementing!" );
    return {};
}

const SamplerBase* const TextureManager<Dx12>::GetSampler( SamplerAddressMode ) const
{
    assert( 0 && "needs to be implemented" );
    return nullptr;
}
