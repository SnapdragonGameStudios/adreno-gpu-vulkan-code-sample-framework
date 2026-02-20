//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "runtime_shader.hpp"
#include "main/applicationEntrypoint.hpp"
#include "camera/cameraController.hpp"
#include "camera/cameraControllerTouch.hpp"
#include "camera/cameraData.hpp"
#include "camera/cameraGltfLoader.hpp"
#include "gui/imguiVulkan.hpp"
#include "material/drawable.hpp"
#include "material/vulkan/shaderModule.hpp"
#include "material/shaderManagerT.hpp"
#include "material/materialManager.hpp"
#include "material/vulkan/specializationConstantsLayout.hpp"
#include "mesh/meshHelper.hpp"
#include "mesh/meshLoader.hpp"
#include "system/math_common.hpp"
#include "texture/textureManager.hpp"
#include "vulkan/extensionHelpers.hpp"
#include "imgui.h"
#include <../external/glslang/glslang/Include/glslang_c_interface.h>
#include <../external/glslang/glslang/Public/resource_limits_c.h>

#include <random>
#include <iostream>
#include <filesystem>

bool RuntimeShader::Build(const std::string& glsl_code,
    VkDevice device,
    const char* entry_point,
    glslang_stage_t stage)
{
    m_is_valid = false;

    m_spirv = CompileGLSLToSPIRV(glsl_code, entry_point, stage, m_defines);
    if (m_spirv.empty())
    {
        LOGE("Runtime Shader failed to compile GLSL into SPIRV blob");
        return false;
    }

    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = m_spirv.size() * sizeof(uint32_t);
    create_info.pCode = m_spirv.data();

    if (vkCreateShaderModule(device, &create_info, nullptr, &m_shader_module) != VK_SUCCESS)
    {
        LOGE("Runtime Shader failed to create vk shader module");
        return false;
    }

    m_is_valid = true;

    return true;
}

std::vector<uint32_t> RuntimeShader::CompileGLSLToSPIRV(
    const std::string&                                   glsl_source,
    const char*                                          entry_name,
    glslang_stage_t                                      stage,
    std::span<const std::pair<std::string, std::string>> defines)
{
    ////////////////////
    // COMPOSE SHADER //
    ////////////////////


    size_t version_string_index = glsl_source.find_first_of("version");
    if (version_string_index == std::string::npos)
    {
        LOGE("Shader compilation failed -> Could not locate 'version' string on shader code");
        return {};
    }
    version_string_index += std::string_view("version").length();
    size_t line_under_version_string_index = glsl_source.find_first_of('\n', version_string_index);
    if (line_under_version_string_index == std::string::npos)
    {
        LOGE("Shader compilation failed -> Could not locate 'version' string on shader code");
        return {};
    }
    line_under_version_string_index += 1;

    std::string composed_shader_code = glsl_source;
    for (auto& [define_text, value_text] : defines)
    {
        composed_shader_code.insert(line_under_version_string_index, "#define " + define_text + " " + value_text + "\n");
    }

#if 1
    glslang_input_t input = {
        .language = GLSLANG_SOURCE_GLSL,
        .stage = stage,
        .client = GLSLANG_CLIENT_VULKAN,
        .client_version = GLSLANG_TARGET_VULKAN_1_3,
        .target_language = GLSLANG_TARGET_SPV,
        .target_language_version = GLSLANG_TARGET_SPV_1_6,
        .code = composed_shader_code.c_str(),
        .default_version = 100,
        .default_profile = GLSLANG_NO_PROFILE,
        .force_default_version_and_profile = false,
        .forward_compatible = false,
        .messages = GLSLANG_MSG_DEFAULT_BIT,
//        .resource = s_slslang_built_in_resource,
        .resource = glslang_default_resource(),
//         .resource = nullptr,
    };

    if (!glslang_shader_create(&input)) // initialize internally
    {
        LOGE("Failed to create shader\n");
        return {};
    }

    glslang_shader_t* shader = glslang_shader_create(&input);

    if (!glslang_shader_preprocess(shader, &input))
    {
        LOGE("Preprocessing failed:\n%s\n", glslang_shader_get_info_log(shader));
        glslang_shader_delete(shader);
        return {};
    }

    if (!glslang_shader_parse(shader, &input))
    {
        LOGE("Parsing failed:\n%s\n", glslang_shader_get_info_log(shader));
        glslang_shader_delete(shader);
        return {};
    }

    glslang_program_t* program = glslang_program_create();
    glslang_program_add_shader(program, shader);

    if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT))
    {
        LOGE("Linking failed:\n%s\n", glslang_program_get_info_log(program));
        glslang_program_delete(program);
        glslang_shader_delete(shader);
        return {};
    }

    glslang_program_SPIRV_generate(program, stage);

    const auto* words = glslang_program_SPIRV_get_ptr(program);
    const auto size = glslang_program_SPIRV_get_size(program);

    std::vector<uint32_t> spirv(size);
    std::memcpy(spirv.data(), words, size * sizeof(uint32_t));

    glslang_program_delete(program);
    glslang_shader_delete(shader);

    return spirv;
#else

    ///////////////
    // SLANG API //
    ///////////////

    SlangSession* session = spCreateSession(nullptr);
    SlangCompileRequest* request = spCreateCompileRequest(&m_global_session);

    spSetCodeGenTarget(request, SLANG_SPIRV);
    spSetTargetProfile(request, 0, spFindProfile(&m_global_session, "vk_1_3"));  // Vulkan 1.3 compatibility

    int translationUnitIndex = spAddTranslationUnit(request, SlangSourceLanguage::SLANG_SOURCE_LANGUAGE_GLSL, nullptr);
    spAddTranslationUnitSourceString(request, translationUnitIndex, nullptr, composed_shader_code.c_str());

    spAddEntryPoint(
        request,
        translationUnitIndex,
        entry_name,
        stage);

    int compileResult = spCompile(request);
    if (SLANG_FAILED(compileResult))
    {
        const char* diagnosticOutput = spGetDiagnosticOutput(request);
        spDestroyCompileRequest(request);
        LOGE("Shader compilation failed -> Compilation failed");
        return {};
    }

    size_t spvSize = 0;
    const void* spvData = spGetEntryPointCode(request, 0, &spvSize);
    if (!spvData || spvSize == 0)
    {
        spDestroyCompileRequest(request);
        LOGE("Shader compilation failed -> Failed to retrieve entrypoint from compiled code");
        return {};
    }

    std::vector<uint32_t> spirv(spvSize / 4);
    std::memcpy(spirv.data(), spvData, spvSize);
    return spirv;
#endif
}