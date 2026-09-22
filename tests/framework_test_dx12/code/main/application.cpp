// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

#include "application.hpp"
#include "main/applicationEntrypoint.hpp"
#include "memory/dx12/indexBufferObject.hpp"//temporary
#include "memory/dx12/vertexBufferObject.hpp"//temporary
#include "system/math_common.hpp"
#include "texture/dx12/loaderKtx.hpp"

// The vertex buffer bind id, used as a constant in various places in the sample
#define VERTEX_BUFFER_BIND_ID 0

//
// Implementation of the Application entrypoint (called by the framework)
// Construct the Application class
//
//-----------------------------------------------------------------------------
FrameworkApplicationBase* Application_ConstructApplication()
//-----------------------------------------------------------------------------
{
    return new Application();
}

//-----------------------------------------------------------------------------
Application::Application() : ApplicationHelperBase()
//-----------------------------------------------------------------------------
{
    // Object rotation
    m_TotalRotation = 0.0f;
    m_RotationSpeed = 0.5f;     // Radians per second

    // Camera
    m_CurrentCameraPos = glm::vec3(0.0f, 2.0f, 5.5f);
    m_CurrentCameraLook = glm::vec3(0.0f, 1.0f, 0.0f);

    // The Test Object
    m_ObjectScale = 1.0f;
    m_ObjectWorldPos = glm::vec3(0.0f, 1.25f, 0.0f);
}

//-----------------------------------------------------------------------------
Application::~Application()
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
bool Application::Initialize( uintptr_t windowHandle, uintptr_t instanceHandle)
//-----------------------------------------------------------------------------
{
    if (!ApplicationHelperBase::Initialize(windowHandle, instanceHandle))
    {
        return false;
    }

    //auto* const pDx12 = GetDx12();

    if (!LoadMeshObjects())
        return false;

    if (!LoadTextures())
        return false;

    if (!LoadShaders())
        return false;

    if (!InitUniforms())
        return false;

    if (!InitMaterials())
        return false;

    if (!InitCommandBuffers())
        return false;

    return true;
}

//-----------------------------------------------------------------------------
bool Application::SetWindowSize(uint32_t width, uint32_t height)
//-----------------------------------------------------------------------------
{
    LOGI("SetSize(%dx%d) Entered...", width, height);
    return true;
}

//-----------------------------------------------------------------------------
void Application::Destroy()
//-----------------------------------------------------------------------------
{
    //m_TestMesh.Destroy();

    // Finally call into base class destroy
    ApplicationHelperBase::Destroy();
}

//-----------------------------------------------------------------------------
bool Application::LoadMeshObjects()
//-----------------------------------------------------------------------------
{
    const char* pGLTFMeshFile = "./Media/Objects/UVSphere_Separate.gltf";
    LOGI("Loading glTF mesh: %s...", pGLTFMeshFile);
    if (!LoadGLTF(pGLTFMeshFile, VERTEX_BUFFER_BIND_ID, &m_TestMesh))
    {
        LOGE("Error loading Object mesh: %s", pGLTFMeshFile);
        return false;
    }

    Buffer<Dx12> mybuffer;

    int data[4]{};
    if (!mybuffer.Initialize(&GetDx12()->GetMemoryManager(), 16, BufferUsageFlags::Storage, &data))
    {
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool Application::LoadTextures()
//-----------------------------------------------------------------------------
{
    auto* const pDx12 = GetDx12();

    m_SamplerRepeat = CreateSampler(*pDx12, SamplerAddressMode::Repeat, SamplerFilter::Linear, SamplerBorderColor::TransparentBlackFloat, 0.0f);

    TextureKtx textureLoader{ *pDx12 };
    if (!textureLoader.Initialize())
        return false;

    m_Texture = textureLoader.LoadKtx(*pDx12, *m_AssetManager, "./Media/Textures/surf_d.ktx", m_SamplerRepeat);
    return !m_Texture.IsEmpty();
}

//-----------------------------------------------------------------------------
bool Application::LoadShaders()
//-----------------------------------------------------------------------------
{
    auto*const pDx12 = GetDx12();

    const char* pDebugVertFile = "./Media/Shaders/Debug.vert.bin";
    const char* pDebugFragFile = "./Media/Shaders/Debug.frag.bin";

    //LOGI("Loading Test shader...");
    m_ShaderVert.Load(*pDx12, *m_AssetManager, pDebugVertFile);
    m_ShaderFrag.Load(*pDx12, *m_AssetManager, pDebugFragFile);

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitUniforms()
//-----------------------------------------------------------------------------
{
    auto*const pGfxApi = GetDx12();

    // These are only created here, they are not set to initial values
    LOGI("Creating uniform buffers...");
    CreateUniformBuffer(pGfxApi, &m_VertUniform, sizeof(TestVertUB), NULL);
    CreateUniformBuffer(pGfxApi, &m_FragUniform, sizeof(TestFragUB), NULL);

    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitMaterials()
//-----------------------------------------------------------------------------
{
    auto* const pGfxApi = GetDx12();
    auto* const dx12Device = pGfxApi->GetDevice();

    // Initialze Dx12 objects for this material

    // Vertex Input layout
    //D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
    //    { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(ApplicationHelperBase::vertex_layout, pos), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    //    { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, offsetof(ApplicationHelperBase::vertex_layout, uv), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
    //};
    const auto& inputElementDescs = m_TestMesh.m_VertexBuffers[0].GetElementDescs();

    // Root signature
#if 1

    D3D12_DESCRIPTOR_RANGE descriptorTable[]{ {
        .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        .NumDescriptors = 1,
        .BaseShaderRegister = 0,
        .RegisterSpace = 0,
        .OffsetInDescriptorsFromTableStart = 0,
        } };

    D3D12_ROOT_PARAMETER rootParams[]{{
                                        // VertCB
                                        .ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV,
                                        .Descriptor = {
                                            .ShaderRegister = 0/*register index*/,
                                            .RegisterSpace = 0 },
                                        .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL
                                      },
                                      {
                                          // FragCB
                                          .ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV,
                                          .Descriptor = {
                                              .ShaderRegister = 1/*register index*/,
                                              .RegisterSpace = 0 },
                                          .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL
                                      },
                                      {
                                          // Descriptor table (to Textures)
                                          .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                                          .DescriptorTable = {
                                              .NumDescriptorRanges = _countof(descriptorTable),
                                              .pDescriptorRanges = descriptorTable
                                          }
                                      }
                                    };

    D3D12_STATIC_SAMPLER_DESC staticSamplers[]{ {
            .Filter = D3D12_FILTER_ANISOTROPIC,
            .AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            .AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            .AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            .ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS,
            .BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
            .MaxLOD = FLT_MAX,
            .ShaderRegister = 0,
            .RegisterSpace = 0,
            .ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL
    } };

    D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc{
        .NumParameters = _countof(rootParams),
        .pParameters = rootParams,
        .NumStaticSamplers = _countof(staticSamplers),
        .pStaticSamplers = staticSamplers,
        .Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT };

    ComPtr<ID3DBlob> signatureData;
    ComPtr<ID3DBlob> error;
    if (!Dx12::CheckError("D3D12SerializeRootSignature", D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signatureData, &error)))
    {
        return false;
    }

    if (S_OK != dx12Device->CreateRootSignature(0, signatureData->GetBufferPointer(), signatureData->GetBufferSize(), IID_PPV_ARGS(&m_RootSignature)))
        return false;
#else
    if (S_OK != dx12Device->CreateRootSignature(0, m_ShaderVert.GetShaderData().data(), m_ShaderVert.GetShaderData().size(), IID_PPV_ARGS(&m_RootSignature)))
       return false;
#endif

    // Pipeline state

    D3D12_RASTERIZER_DESC rasterizerState{
        .FillMode = D3D12_FILL_MODE_SOLID,
        .CullMode = D3D12_CULL_MODE_BACK,
        .FrontCounterClockwise = FALSE,
        .DepthBias = D3D12_DEFAULT_DEPTH_BIAS,
        .DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
        .SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS,
        .DepthClipEnable = TRUE,
        .MultisampleEnable = FALSE,
        .AntialiasedLineEnable = FALSE,
        .ForcedSampleCount = 0,
        .ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF
    };
    D3D12_BLEND_DESC blendState{
        .AlphaToCoverageEnable = FALSE,
        .IndependentBlendEnable = FALSE,
    };
    for (auto& rt : blendState.RenderTarget)
    {
        rt = D3D12_RENDER_TARGET_BLEND_DESC{
            .BlendEnable = FALSE,
            .LogicOpEnable = FALSE,
            .SrcBlend = D3D12_BLEND_ONE,
            .DestBlend = D3D12_BLEND_ZERO,
            .BlendOp = D3D12_BLEND_OP_ADD,
            .SrcBlendAlpha = D3D12_BLEND_ONE,
            .DestBlendAlpha = D3D12_BLEND_ZERO,
            .BlendOpAlpha = D3D12_BLEND_OP_ADD,
            .LogicOp = D3D12_LOGIC_OP_NOOP,
            .RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL
        };
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.InputLayout = { inputElementDescs.data(), (uint32_t) inputElementDescs.size() };
    psoDesc.pRootSignature = m_RootSignature.Get();
    psoDesc.VS.pShaderBytecode = m_ShaderVert.GetShaderData().data();
    psoDesc.VS.BytecodeLength = m_ShaderVert.GetShaderData().size();
    psoDesc.PS.pShaderBytecode = m_ShaderFrag.GetShaderData().data();
    psoDesc.PS.BytecodeLength = m_ShaderFrag.GetShaderData().size();
    psoDesc.RasterizerState = rasterizerState;
    psoDesc.BlendState = blendState;
    psoDesc.DepthStencilState.DepthEnable = FALSE;
    psoDesc.DepthStencilState.StencilEnable = FALSE;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc.Count = 1;
    if (S_OK != dx12Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_PipelineState)))
        return false;
    return true;
}

//-----------------------------------------------------------------------------
bool Application::InitCommandBuffers()
//-----------------------------------------------------------------------------
{
    auto* const pGfxApi = GetDx12();

    return m_CommandList.Initialize(pGfxApi, "Main", CommandList::Type::Direct, 0, m_PipelineState.Get());
}

//-----------------------------------------------------------------------------
bool Application::UpdateUniforms(float fltDiffTime)
//-----------------------------------------------------------------------------
{
    auto*const pGfxApi = GetDx12();

    // Projection Matrix
    float aspect = (float)gRenderWidth / (float)gRenderHeight;

    // These are labeled as global because they should go to config file
    float gFOV = PI_DIV_4;
    float gNearPlane = 1.0f;
    float gFarPlane = 100.0f;

    m_ProjectionMatrix = glm::perspectiveRH(gFOV, aspect, gNearPlane, gFarPlane);

    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    m_ViewMatrix = glm::lookAtRH(m_CurrentCameraPos, m_CurrentCameraLook, up);

    // Handle object rotation
    m_TotalRotation += m_RotationSpeed * fltDiffTime;
    if (m_TotalRotation > PI_MUL_2)
        m_TotalRotation -= PI_MUL_2;


    glm::mat4 OneModel = glm::mat4(1.0f);

    // ********************************
    // Test Vert Uniform 
    // ********************************
    OneModel = glm::translate(glm::mat4(1.0f), m_ObjectWorldPos);
    OneModel = glm::scale(OneModel, glm::vec3(m_ObjectScale, m_ObjectScale, m_ObjectScale));
    OneModel = glm::rotate(OneModel, m_TotalRotation, glm::vec3(0.0f, 1.0f, 0.0f));

    TestVertUB vertUB{};
    vertUB.MVPMatrix = m_ProjectionMatrix * m_ViewMatrix * OneModel;
    vertUB.ModelMatrix = OneModel;
    UpdateUniformBuffer(pGfxApi, m_VertUniform, vertUB);

    // ********************************
    // Test Frag Uniform 
    // ********************************

    TestFragUB fragUB{};
    float gSpecularExponent = 256.0f;
    glm::vec4 gLightDirection = glm::vec4(-0.5f, -1.0f, -1.0f, 0.0);

    fragUB.Color = glm::vec4(0.9f, 0.9f, 0.9f, 1.0f);  // White by default
    fragUB.EyePos = glm::vec4(m_CurrentCameraPos.x, m_CurrentCameraPos.y, m_CurrentCameraPos.z, 1.0f);
    fragUB.LightDir = normalize(gLightDirection);
    fragUB.LightColor = glm::vec4(1.0f, 1.0f, 1.0f, gSpecularExponent);  // White by default;
    UpdateUniformBuffer(pGfxApi, m_FragUniform, fragUB);

    return true;
}

//-----------------------------------------------------------------------------
bool Application::BuildCmdBuffers(uint32_t whichFrame)
//-----------------------------------------------------------------------------
{
    LOGI("Creating command buffers...");

    auto* const pGfxApi = GetDx12();

    if (!pGfxApi->FrameInit(whichFrame))
        return false;

    if (!m_CommandList.Reset())
        return false;

    m_CommandList.Begin(m_PipelineState.Get());

    pGfxApi->BackbufferRenderSetup(whichFrame, m_CommandList.Get());

    pGfxApi->SetDescriptorHeaps(m_CommandList.Get());
    m_CommandList->SetGraphicsRootSignature(m_RootSignature.Get());

    auto srvRootDescriptorTable = pGfxApi->AllocateShaderResourceViewDescriptors(1, whichFrame);
    auto samplerRootDescriptorTable = pGfxApi->AllocateSamplerDescriptors(1, whichFrame);

    pGfxApi->GetDevice()->CreateShaderResourceView(m_Texture.GetResource(), &m_Texture.GetResourceViewDesc(), srvRootDescriptorTable.GetCpuHandle(0));
    pGfxApi->GetDevice()->CreateSampler(&m_SamplerRepeat.GetDesc(), samplerRootDescriptorTable.GetCpuHandle(0));

    // Draw the model
    m_CommandList->SetGraphicsRootConstantBufferView(0, m_VertUniform.buf.GetResource()->GetGPUVirtualAddress());
    m_CommandList->SetGraphicsRootConstantBufferView(1, m_FragUniform.buf.GetResource()->GetGPUVirtualAddress());
    m_CommandList->SetGraphicsRootDescriptorTable(2, srvRootDescriptorTable.GetGpuHandle());
    //m_CommandList->SetGraphicsRootDescriptorTable(3, samplerRootDescriptorTable.GetGpuHandle());

    m_CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    m_CommandList->IASetVertexBuffers(0, 1, &m_TestMesh.m_VertexBuffers[0].GetVertexBufferView());
    m_CommandList->DrawInstanced(m_TestMesh.m_NumVertices, 1, 0, 0);

    // Transition backbuffer to be 'presentable'
    pGfxApi->BackbufferPresentSetup(whichFrame, m_CommandList.Get());

    return m_CommandList.End();
}


//-----------------------------------------------------------------------------
void Application::Render(float fltDiffTime)
//-----------------------------------------------------------------------------
{
    LOGI("Render() Entered...");

    Dx12* const pDx12 = GetDx12();

    // Obtain the next swap chain image for the next frame.
    auto currentBackBuffer = pDx12->SetNextBackBuffer();
    uint32_t whichBuffer = currentBackBuffer.idx;

    UpdateUniforms(fltDiffTime);

    BuildCmdBuffers(whichBuffer);

    // Execute on gpu
    auto* const pGfxApi = GetDx12();
    pGfxApi->CommandListExecute(m_CommandList.Get());

    pGfxApi->PresentSwapchain();


    // Grab the vulkan wrapper
//    Vulkan* pVulkan = GetVulkan();

    // Obtain the next swap chain image for the next frame.
//    auto currentBuffer = pVulkan->SetNextBackBuffer();

    // ********************************
    // Application Draw() - Begin
    // ********************************

    // Update uniform buffers with latest data
//    UpdateUniforms(fltDiffTime);

//    m_CommandBuffers[currentBuffer.idx].QueueSubmit(currentBuffer.semaphore, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, pVulkan->m_RenderCompleteSemaphore, currentBuffer.fence);

    // ********************************
    // Application Draw() - End
    // ********************************

//    pVulkan->PresentQueue( pVulkan->m_RenderCompleteSemaphore, currentBuffer.swapchainPresentIdx );
}



