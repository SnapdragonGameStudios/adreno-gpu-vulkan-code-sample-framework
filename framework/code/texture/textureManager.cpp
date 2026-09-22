// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "system/assetManager.hpp"
#include "textureManager.hpp"
#include "loaderKtx.hpp"
#include "loaderPpm.hpp"
#include <queue>

TextureManagerBase::TextureManagerBase( AssetManager& rAssetManager ) noexcept
    : m_AssetManager( rAssetManager )
{
}

TextureManagerBase::~TextureManagerBase()
{
    Release();
}

bool TextureManagerBase::Initialize(uint32_t numWorkerThreads)
{
    if (m_LoadingThreadWorker.Initialize("TextureThreadWorker", numWorkerThreads) <= 0)
        return false;
    return m_Loader->Initialize();
}

void TextureManagerBase::Release()
{
    m_LoadingThreadWorker.Terminate();
}

const TextureBase* TextureManagerBase::CreateTextureObject( uint32_t Width, uint32_t Height, TextureFormat Format, TEXTURE_TYPE TexType, const char* pName, Msaa Msaa, TEXTURE_FLAGS Flags )
{
    CreateTexObjectInfo createInfo{};
    createInfo.uiWidth = Width;
    createInfo.uiHeight = Height;
    createInfo.Format = Format;
    createInfo.TexType = TexType;
    createInfo.Flags = Flags;
    createInfo.pName = pName;
    createInfo.Msaa = Msaa;
    return CreateTextureObject( createInfo );
}
