
#
# Build shaders
# Add everything with .frag .vert .comp .json (etc) extension from the shaders/ directory and build using Vulkan shader compiler.
# If a sample needs more fine-grained control over this it can 'include(CompileShadersHelper)' and call compile_glsl etc manually.
#
include(CompileShadersHelper)

function(scan_for_shaders)
    cmake_parse_arguments(SCAN_FOR_SHADERS "DEBUG_INFO" "" "" ${ARGN})

    # Optional destination path for compiled shaders
    set(SHADER_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Media/Shaders")
    if(DEFINED SHADER_DESTINATION)
        set(SHADER_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_DESTINATION}")
    endif()

    # Use project name to generate unique target names
    set(target_prefix "${PROJECT_NAME}")

    # Scan through shaders directory looking for shader source files and generate build commands for them
    file(GLOB glsl_files "shaders/*.vert" "shaders/*.frag" "shaders/*.comp")
    compile_glsl("${glsl_files}" "vulkan1.1" "${SHADER_OUTPUT_PATH}" "${target_prefix}_GLSL" "${SCAN_FOR_SHADERS_DEBUG_INFO}")

    # Ray Tracing and Mesh shaders need to target Vulkan 1.2
    file(GLOB rt_files "shaders/*.rgen" "shaders/*.rint" "shaders/*.rahit" "shaders/*.rchit" "shaders/*.rmiss" "shaders/*.rcall" "shaders/*.mesh" "shaders/*.task")
    compile_glsl("${rt_files}" "spirv1.4" "${SHADER_OUTPUT_PATH}" "${target_prefix}_RTMESH" "${SCAN_FOR_SHADERS_DEBUG_INFO}")

    # HLSL files (compiled to SPIR-V). Entry point assumed to be "main".
    file(GLOB hlsl_files "shaders/*.comp.hlsl")
    compile_hlsl("${hlsl_files}" "vulkan1.1" "${SHADER_OUTPUT_PATH}" "${target_prefix}_HLSL")

    # JSON configuration files
    file(GLOB json_files "shaders/*.json")
    copy_json("${json_files}" "json" "${SHADER_OUTPUT_PATH}" "${target_prefix}_JSON")

    # Add shaders (sources) into a 'Shaders' folder for Visual Studio
    source_group("shaders" FILES ${SHADERS_SRC})
endfunction()
