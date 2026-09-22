// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2023 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#pragma once
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>
#include "graphicsApi/renderTarget.hpp"
#include "texture/dx12/texture.hpp"


class Dx12;
template<class T_GFXAPI> class Texture;
template<class T_GFXAPI> class CommandList;

union ClearColor
{
    float r32g32b32a32_float[4];
};

template<>
class RenderTarget<Dx12>  final : public RenderTargetBase
{
    RenderTarget(const RenderTarget<Dx12>&) = delete;
    RenderTarget& operator=(const RenderTarget<Dx12>&) = delete;
public:
    RenderTarget();
    ~RenderTarget();
    RenderTarget( RenderTarget<Dx12>&& ) noexcept;
    RenderTarget& operator=( RenderTarget<Dx12>&& ) noexcept;
    uint32_t GetNumColorLayers() const { return (uint32_t)m_pLayerFormats.size(); }

    bool Initialize(Dx12* pGfxApi, const RenderTargetInitializeInfo& info, const char* pName);

    bool Initialize(Dx12* pGfxApi, uint32_t uiWidth, uint32_t uiHeight, const std::span<const TextureFormat> pLayerFormats, TextureFormat DepthFormat = TextureFormat::D24_UNORM_S8_UINT, std::span<const Msaa> Msaa = {}, const char* pName = NULL);
    bool Initialize(Dx12* pGfxApi, uint32_t uiWidth, uint32_t uiHeight, const std::span<const TextureFormat> pLayerFormats, TextureFormat DepthFormat = TextureFormat::D24_UNORM_S8_UINT, Msaa Msaa = Msaa::Samples1, const char* pName = NULL)
    {
        RenderTargetInitializeInfo info{
            .Width = uiWidth,
            .Height = uiHeight,
            .LayerFormats = pLayerFormats,
            .DepthFormat = DepthFormat,
            .Msaa = {&Msaa, 1},
            //.TextureTypes
            //.FilterModes(filtermode)
        };
        //return Initialize(pGfxApi, uiWidth, uiHeight, pLayerFormats, DepthFormat, { &Msaa,1 }, pName);
        return Initialize( pGfxApi, info, pName );
    }

    void SetRenderTarget( CommandList<Dx12>& );
private:
    template<uint32_t T_NUM_BUFFERS> friend class RenderTargetArray;
    bool InitializeDepth();
    bool InitializeColor(const std::span<const TEXTURE_TYPE> TextureTypes);
    bool InitializeResolve(const std::span<const TextureFormat> ResolveTextureFormats);

    void SetClearColors(const std::span<const ClearColor> clearColors);

    void Release(bool bReleaseFramebuffers /*set true if we are the owner of the framebuffers (and so want to clean them up)*/);

    // Attributes
public:
    std::string                     m_Name;

    uint32_t                        m_Width = 0;
    uint32_t                        m_Height = 0;

    std::vector<TextureFormat>      m_pLayerFormats;
    std::vector<Msaa>               m_Msaa;
    std::vector<SamplerFilter>      m_FilterMode;
    TextureFormat                   m_DepthFormat = TextureFormat::UNDEFINED;

    // The Color Attachments

    std::vector<TextureDx12>        m_ColorAttachments;
    std::vector<ClearColor>         m_ClearColors;

    // The Resolve Attachments
    std::vector<TextureDx12>        m_ResolveAttachments;

    // The Depth Attachment
    TextureDx12		                m_DepthAttachment;

private:
    Dx12* m_pGfxApi = nullptr;

    ComPtr<ID3D12DescriptorHeap>    m_descriptorHeap;
    ComPtr<ID3D12DescriptorHeap>    m_depthDescriptorHeap;
};
