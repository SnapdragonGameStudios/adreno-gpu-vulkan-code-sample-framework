
cmake_minimum_required (VERSION 3.21)

project (fully_fused_mlp C CXX)
set(CMAKE_CXX_STANDARD 20)

#
# Source files included in this application.
#
set(CPP_SRC
    code/main/application.cpp
    code/main/application.hpp
    code/main/fully_fused_mlp_runner.cpp
    code/main/fully_fused_mlp_runner.hpp
    code/main/runtime_shader.cpp
    code/main/runtime_shader.hpp
)

#
# Setup the module path to include the 'project directory' (project/windows or project/android)
#
if(NOT DEFINED PROJECT_ROOT_DIR)
    # Windows can use CMAKE_SOURCE_DIR; Android's gradle passes -DPROJECT_ROOT_DIR=${project.rootDir}
    set(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR})
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_ROOT_DIR}/cmake)

#
# Do all the build steps for a Framework application.
# (Defines ${TARGET_NAME} inside the helper; do NOT modify the helper that links .a files.)
#
include(FrameworkApplicationHelper)

# ------------------------------------------------------------------------------
# Half (float16) dependency under ../../framework/external
# Build as a normal CMake target and link it to ${TARGET_NAME}.
# Used by the CPU reference path to convert fp16 <-> fp32 exactly like Vulkan_MLP.
#
# 'half' is a shared external target: several samples (e.g. cooperative_matrix)
# may define it. When more than one such sample is enabled in the same CMake
# configure, only the first should create the target - guard with NOT TARGET so
# we reuse the existing one instead of colliding ("another target with the same
# name already exists").
# ------------------------------------------------------------------------------

if(NOT TARGET half)
    add_library(half STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../framework/external/half/half.cpp
    )

    # Expose headers to dependents (your app target will inherit this include path)
    target_include_directories(half
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/../../framework/external
    )

    # Nicer grouping in IDEs
    source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/../../framework/external" PREFIX "external" FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../../framework/external/half/half.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../framework/external/half/half.h
    )
endif()

# Link half into the application/library target defined by the helper
target_link_libraries(${TARGET_NAME} half)

#
# Setup asset source and target folders
#

# cmake will use our GameSampleAssets (default for no parameter) as root directory for any asset request
# (see FrameworkApplicationHelper.cmake for more info)
inject_root_asset_path()

# Register local variables for asset request, while also defining them in the C++ code for easy access
# Here we use the default destination paths, all defined at FrameworkApplicationHelper.cmake
register_local_asset_path(SHADER_DESTINATION  "${DEFAULT_LOCAL_SHADER_DESTINATION}")

#
# Add in the contents of 'shaders' directory
#
include(AddShadersDir)

# Search and include all project shaders
scan_for_shaders()
