// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//                  Copyright (c) 2021 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

///
/// @file frameTestAppDx12.hpp
/// @brief Application demonstrating use of DX12 Framework to load and render a simple object.
/// 

#include "main/applicationHelperBaseDx12.hpp"
#include "dx12/dx12.hpp"
#include "dx12/commandList.hpp"
#include "material/dx12/shaderModule.hpp"
#include "material/shader.hpp"
#include "memory/dx12/bufferObject.hpp"
#include "memory/dx12/indexBufferObject.hpp"
#include "memory/dx12/uniform.hpp"
#include "memory/dx12/vertexBufferObject.hpp"
#include "mesh/mesh.hpp"


class Application : public ApplicationHelperBase
{
public:
    Application();
    ~Application() override;

    // Override FrameworkApplicationBase
    bool    Initialize(uintptr_t windowHandle, uintptr_t instanceHandle) override;
    void    Destroy() override;

    bool    SetWindowSize(uint32_t width, uint32_t height) override;
    void    Render(float fltDiffTime) override;

    // Setup
    bool    LoadMeshObjects();
    bool    LoadTextures();
    bool    LoadShaders();
    bool    InitUniforms();
    bool    InitMaterials();
    bool    InitCommandBuffers();
    // Render
    bool    UpdateUniforms(float fltDiffTime);
    bool    BuildCmdBuffers(uint32_t whichFrame);

private:

    // Meshes
    Mesh                m_TestMesh;

    // Textures
    TextureDx12         m_Texture;
    SamplerDx12         m_SamplerRepeat;

    // Shader
    ShaderModule<Dx12> m_ShaderVert;
    ShaderModule<Dx12> m_ShaderFrag;

    // Constant Buffers
    struct TestVertUB
    {
        glm::mat4       MVPMatrix;
        glm::mat4       ModelMatrix;
    };

    struct TestFragUB
    {
        glm::vec4       Color;
        glm::vec4       EyePos;
        glm::vec4       LightDir;
        glm::vec4       LightColor;
    };

    UniformT<TestVertUB> m_VertUniform;
    UniformT<TestFragUB> m_FragUniform;

    // Object rotation
    float               m_TotalRotation;
    float               m_RotationSpeed;

    // Camera
    glm::vec3           m_CurrentCameraPos;
    glm::vec3           m_CurrentCameraLook;

    // Matrices
    glm::mat4           m_ProjectionMatrix;
    glm::mat4           m_ViewMatrix;

    // The Test Object
    float               m_ObjectScale;
    glm::vec3           m_ObjectWorldPos;

    // MaterialBase
    ComPtr<ID3D12RootSignature> m_RootSignature;
    ComPtr<ID3D12PipelineState> m_PipelineState;

    // Commandlist
    CommandList         m_CommandList;
};
