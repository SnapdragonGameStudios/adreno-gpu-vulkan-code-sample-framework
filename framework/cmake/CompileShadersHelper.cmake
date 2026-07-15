#
# Helper for building shaders
# Implements function "compile_glsl" which should be used to compile a list of shaders
# Implements function "compile_hlsl" which should be used to compile a list of shaders
# Implements function "copy_json" which copies a list of shader json configs
# Implements function "compile_alias" which compiles a list of shader .alias files
#

# Make sure we have the Vulkan compiler
find_program(
	GLSL_VALIDATOR
	glslangValidator
	DOC "Vulkan Shader Compiler (glslangValidator) (is Vulkan SDK installed?)"
	REQUIRED
)

# Make sure we have the DXC compiler.  See if it is installed alngside Vulkan first.
if(DEFINED ENV{VULKAN_SDK})
    cmake_path(SET VULKAN_SDK_PATH NORMALIZE $ENV{VULKAN_SDK})
    find_program(
        DXC_EXE
        dxc
        HINTS ${VULKAN_SDK_PATH}
        PATH_SUFFIXES Bin
        NO_DEFAULT_PATH
        DOC "Microsoft Shader compiler (dxc) (is Vulkan SDK installed?)"
        OPTIONAL
    )
endif()

# We couldnt find dxc installed with Vulkan, look for it on the path (should find the Windows SDK version)
find_program(
    DXC_EXE
    dxc
    DOC "Microsoft Shader compiler (dxc) (is Vulkan SDK installed?)"
    REQUIRED
)

# Runs the command to get the pluginval version (more recent versions of dcx support --version, older ones dont and print the version inside -help)
if (NOT DEFINED ENV{DXC_VERSION})
    message("DCX compiler found at: ${DXC_EXE}")
    execute_process(COMMAND ${DXC_EXE} --help
                    OUTPUT_VARIABLE DXC_VERSION_RAW
                    ERROR_VARIABLE DXC_VERSION_RAW)
    string(REGEX MATCH "Version: ([^\r\n]+)[\r\n]"
           DXC_VERSION "${DXC_VERSION_RAW}")
    string(REGEX REPLACE "Version: (.*)"
           "\\1"
           DXC_VERSION "${DXC_VERSION}")
    message( "DXC version: ${DXC_VERSION}" )
    set(ENV{DXC_VERSION} ${DXC_VERSION})
endif()


# Custom shader include direcotry
if(DEFINED SHADER_INCLUDE)
  list(JOIN SHADER_INCLUDE ";" SHADER_INCLUDE_TMP)
  cmake_path(CONVERT "${SHADER_INCLUDE_TMP}" TO_NATIVE_PATH_LIST SHADER_INCLUDE)
  list(TRANSFORM SHADER_INCLUDE PREPEND "-I")
else()
  set(SHADER_INCLUDE "-I.")
endif()

# Ensure we have a place to put the .d dependency files emitted by the compiler
set(DEPENDS_PATH ${CMAKE_CURRENT_BINARY_DIR}/Media/Shaders/)
file(MAKE_DIRECTORY ${DEPENDS_PATH})
if (NOT DEFINED SHADERS_SRC)
  set(SHADERS_SRC "")
endif()

function(compile_glsl files targetenv dst_dir target_name)
    set(output_files "")
    set(glsl_debug_flags "")
    if(ARGC GREATER 4 AND ARGV4)
        set(glsl_debug_flags "-gVS")
    endif()
    foreach(file ${files})
        get_filename_component(output_filename ${file} NAME)
        set(output_shader "${dst_dir}/${output_filename}.spv")
        set(output_shader_dep "${DEPENDS_PATH}/${output_filename}.spv.d")

        add_custom_command(
            OUTPUT ${output_shader}
            MAIN_DEPENDENCY ${file}
            DEPFILE ${output_shader_dep}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${dst_dir}
            COMMAND ${GLSL_VALIDATOR} ${SHADER_INCLUDE} -I. -V ${file} -o ${output_filename}.spv --target-env ${targetenv} --quiet ${glsl_debug_flags} --depfile ${output_shader_dep}
            COMMAND ${CMAKE_COMMAND} -E rename ${output_filename}.spv ${output_shader}
        )
        list(APPEND output_files ${output_shader})
        list(APPEND SHADERS_SRC ${file})
    endforeach()

    add_custom_target(${target_name} ALL DEPENDS ${output_files})
    set_target_properties(${target_name} PROPERTIES FOLDER "shaders")
    if(TARGET ${PROJECT_NAME})
        add_dependencies(${PROJECT_NAME} ${target_name})
    elseif(DEFINED TARGET_NAME AND TARGET ${TARGET_NAME})
        add_dependencies(${TARGET_NAME} ${target_name})
    endif()
    set(SHADERS_SRC ${SHADERS_SRC} PARENT_SCOPE)
endfunction()

function(compile_hlsl files targetenv dst_dir target_name)
    set(output_files "")
    foreach(file ${files})
        get_filename_component(output_filename ${file} NAME)
        set(output_shader "${dst_dir}/${output_filename}.spv")
        set(output_shader_dep "${DEPENDS_PATH}/${output_filename}.spv.d")

        add_custom_command(
            OUTPUT ${output_shader}
            MAIN_DEPENDENCY ${file}
            DEPFILE ${output_shader_dep}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${dst_dir}
            COMMAND ${DXC_EXE} ${SHADER_INCLUDE} -T cs_6_7 -spirv -Wno-conversion -fspv-target-env=${targetenv} -enable-16bit-types -E main ${file} -Fo ${output_filename}.spv -MF ${output_shader_dep}
            COMMAND ${CMAKE_COMMAND} -E rename ${output_filename}.spv ${output_shader}
        )
        list(APPEND output_files ${output_shader})
        list(APPEND SHADERS_SRC ${file})
    endforeach()

    add_custom_target(${target_name} ALL DEPENDS ${output_files})
    set_target_properties(${target_name} PROPERTIES FOLDER "shaders")
    if(TARGET ${PROJECT_NAME})
        add_dependencies(${PROJECT_NAME} ${target_name})
    elseif(DEFINED TARGET_NAME AND TARGET ${TARGET_NAME})
        add_dependencies(${TARGET_NAME} ${target_name})
    endif()
    set(SHADERS_SRC ${SHADERS_SRC} PARENT_SCOPE)
endfunction()

function(copy_json files targetenv dst_dir target_name)
    set(output_files "")
    foreach(file ${files})
        get_filename_component(output_filename ${file} NAME)
        set(output_json "${dst_dir}/${output_filename}")

        add_custom_command(
            OUTPUT ${output_json}
            MAIN_DEPENDENCY ${file}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${dst_dir}
            COMMAND ${CMAKE_COMMAND} -E copy ${file} ${output_json}
        )
        list(APPEND output_files ${output_json})
        list(APPEND SHADERS_SRC ${file})
    endforeach()

    add_custom_target(${target_name} ALL DEPENDS ${output_files})
    set_target_properties(${target_name} PROPERTIES FOLDER "shaders")
    if(TARGET ${PROJECT_NAME})
        add_dependencies(${PROJECT_NAME} ${target_name})
    elseif(DEFINED TARGET_NAME AND TARGET ${TARGET_NAME})
        add_dependencies(${TARGET_NAME} ${target_name})
    endif()
    set(SHADERS_SRC ${SHADERS_SRC} PARENT_SCOPE)
endfunction()

function(compile_alias files)
    foreach(file ${files})
        set(INPUT_ALIAS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
        get_filename_component(OUTPUT_FILENAME ${file} NAME_WLE)
        set(OUTPUT_SHADER ${CMAKE_CURRENT_SOURCE_DIR}/Media/Shaders/${OUTPUT_FILENAME}.spv)
        set(OUTPUT_SHADER_DEP ${DEPENDS_PATH}${OUTPUT_FILENAME}.spv.d)
        cmake_path(NATIVE_PATH INPUT_ALIAS NORMALIZE INPUT_ALIAS_NATIVE)
        cmake_path(NATIVE_PATH OUTPUT_SHADER NORMALIZE OUTPUT_SHADER_NATIVE)
        cmake_path(NATIVE_PATH OUTPUT_SHADER_DEP NORMALIZE OUTPUT_SHADER_DEP_NATIVE)

        add_custom_command(
            OUTPUT ${OUTPUT_SHADER}
            MAIN_DEPENDENCY ${INPUT_ALIAS}
            DEPFILE ${OUTPUT_SHADER_DEP}
            COMMENT "Aliasing ... ${INPUT_ALIAS} to ${OUTPUT_SHADER} (dependency file ${OUTPUT_SHADER_DEP})"
            #COMMAND "echo Aliasing ... ${INPUT_ALIAS} to ${OUTPUT_SHADER} (dependency file ${OUTPUT_SHADER_DEP})"
            COMMAND ${CMAKE_COMMAND} -DINPUT_ALIAS=${INPUT_ALIAS} -DOUTPUT_SHADER=${OUTPUT_SHADER} -DOUTPUT_SHADER_DEP=${OUTPUT_SHADER_DEP} -DGLSL_VALIDATOR=${GLSL_VALIDATOR} -DDXC_EXE=${DXC_EXE} -DSHADER_INCLUDE=${SHADER_INCLUDE} -P ${CMAKE_CURRENT_LIST_DIR}/CompileAlias.cmake
        )
        list(APPEND SHADERS_SRC ${file})

        unset(OUTPUT_JSON)
    endforeach()
    set(SHADERS_SRC ${SHADERS_SRC} PARENT_SCOPE)
endfunction()
