// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2020 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

///
/// @file FrameTestApp.hpp
/// @brief Application demonstrating use of Framework to load and render a simple object.
/// 

#include "main/applicationHelperBase.hpp"

#include "memory/vulkan/indexBufferObject.hpp"
#include "memory/vulkan/uniform.hpp"
#include "memory/vulkan/vertexBufferObject.hpp"
#include "mesh/mesh.hpp"
#include "texture/vulkan/texture.hpp"
#include "vulkan/vulkan.hpp"
#include "vulkan/commandBuffer.hpp"
#include "vulkan/vulkan_support.hpp"

// MaterialBase Descriptions
#include "Materials.h"


class Application : public ApplicationHelperBase
{
public:
    Application();
    ~Application() override;

    // Override FrameworkApplicationBase
    bool    Initialize(uintptr_t windowHandle, uintptr_t instanceHandle) override;
    void    Destroy() override;

    void    Render(float fltDiffTime) override;

    void    TouchDownEvent(int iPointerID, float xPos, float yPos) override;
    void    TouchMoveEvent(int iPointerID, float xPos, float yPos) override;
    void    TouchUpEvent(int iPointerID, float xPos, float yPos) override;

    bool    LoadMeshObjects();
    bool    LoadTextures();
    bool    LoadShaders();
    bool    InitUniforms();
    bool    UpdateUniforms(float fltDiffTime);
    bool    InitMaterials();
    void    InitOneLayout(MaterialVk* pMaterial);
    void    InitOneDescriptorSet(MaterialVk* pMaterial);
    void    UpdateOneDescriptorSet(MaterialVk* pMaterial);
    bool    InitRenderPass();
    bool    InitPipelines();
    bool    InitCommandBuffers();

    bool    BuildCmdBuffers();
    void    AddMeshRenderToCmdBuffer(CommandListVulkan& cmdBuffer, Mesh* pMesh, MaterialVk* pMaterial);


private:
    // Meshes
    Mesh                m_TestMesh;

    // Textures
    TextureVulkan       m_TestTexture;

    // Shaders
    ShaderInfo          m_TestShader;

    // Uniforms
    UniformVulkan       m_TestVertUniform;
    TestVertUB          m_TestVertUniformData;
    UniformVulkan       m_TestFragUniform;
    TestFragUB          m_TestFragUniformData;

    MaterialVk          m_TestMaterial;

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

    // Render Pass
    VkRenderPass        m_RenderPass;

    // Command Buffers
    std::array<CommandListVulkan, NUM_VULKAN_BUFFERS> m_CommandBuffers;
};
