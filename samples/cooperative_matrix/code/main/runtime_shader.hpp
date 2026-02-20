//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include <vulkan/vulkan.hpp>
#include <cstdint>
#include <vector>
#include <string>
#include <span>
#include <unordered_map>
#include <../external/glslang/glslang/Include/glslang_c_interface.h>
#include <../external/glslang/glslang/Public/resource_limits_c.h>

////////////////////////////////////////////////////////////////////////////////
// Class name: RuntimeShader
////////////////////////////////////////////////////////////////////////////////
class RuntimeShader
{
public:

    /*
    * Adds a preprocessor definition to be used during shader compilation.
    * @param name : Name of the macro
    * @param value : Value of the macro
    */
    template<typename Arg, typename OverrideType = Arg>
    inline void AddDefine(const std::string& name, const Arg& arg)
    {
        if constexpr (std::is_same_v<OverrideType, std::string> || std::is_same_v<OverrideType, const char*>)
        {
            m_defines.emplace_back(name, std::string(arg));
        }
        else
        {
            m_defines.emplace_back(name, std::to_string(static_cast<const OverrideType&>(arg)));
        }
    }


    /*
    * Builds the shader by compiling GLSL to SPIR-V and creating a Vulkan shader module.
    * @param glsl_code : GLSL source code as a string
    * @param device : Vulkan logical device used to create the shader module
    * @param entry_point : Entry point name in the GLSL code (e.g., "main")
    * @param stage : Shader stage (e.g., SLANG_STAGE_VERTEX)
    * @return true if compilation and module creation succeeded
    * @note If compilation fails, m_is_valid will be false
    */
    bool Build(
        const std::string& glsl_code,
        VkDevice device,
        const char* entry_point,
        glslang_stage_t stage);

    /*
    * Returns the Vulkan shader module.
    * @return VkShaderModule handle
    */
    inline VkShaderModule GetShaderModule() const
    {
        return m_shader_module;
    }

    /*
    * Checks if the shader was successfully built.
    * @return true if valid
    */
    inline bool IsValid() const
    {
        return m_is_valid;
    }

private:

    std::vector<uint32_t> CompileGLSLToSPIRV(
        const std::string&                                   glsl_source,
        const char*                                          entry_name,
        glslang_stage_t                                      stage,
        std::span<const std::pair<std::string, std::string>> defines = {});

private:
    std::vector<std::pair<std::string, std::string>> m_defines;
    std::vector<uint32_t>                            m_spirv;
    VkShaderModule                                   m_shader_module = VK_NULL_HANDLE;
    bool                                             m_is_valid      = false;
};