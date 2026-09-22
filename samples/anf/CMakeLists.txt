# Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required (VERSION 3.21)

project (anf C CXX)
set(CMAKE_CXX_STANDARD 20)

#
# Source files included in this application.
#

set(CPP_SRC code/main/application.cpp
            code/main/application.hpp
            code/main/anf_gpu_profiler.hpp
            code/main/anf_gpu_profiler.cpp
            code/main/anf_dispatch_timing.hpp
            code/main/anf_dispatch_timing.cpp
            code/main/anf_sample_ui.hpp
            code/main/anf_sample_ui.cpp
            code/main/anf_runtime_controls.hpp
            code/main/anf_runtime_controls.cpp
            code/main/anf_interface.hpp
            code/main/anf_interface.cpp
            code/main/anf_sdk_backend.hpp
            code/main/anf_sdk_backend.cpp
            code/main/halton_jitter.hpp
)
set(FRAMEWORK_LIB framework_vulkan)

#
# Setup the module path to include the 'project directory' (project/windows or project/android)
#
if(NOT DEFINED PROJECT_ROOT_DIR)
    set(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR})   # Windows can use CMAKE_SOURCE_DIR, Android needs build.gradle needs "-DPROJECT_ROOT_DIR=${project.rootDir}" in call to cmake set since there is not a 'top' cmakefile (gradle is top level)
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_ROOT_DIR}/cmake ${FRAMEWORK_DIR}/cmake)

#
# Do all the build steps for a Framework application.
# needs Framework_dir and project_name variables.
#
include(FrameworkApplicationHelper)

#
# Use the SDK downloaded by Configure.py for this sample.
#
set(ANF_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../external/anf_sdk" CACHE PATH "ANF SDK directory")
foreach(ANF_HEADER anf.h anf_sr.h anf_fg.h anf_types.h anf_types_vk.h)
    if(NOT EXISTS "${ANF_SDK_ROOT}/public/${ANF_HEADER}")
        message(FATAL_ERROR "ANF SDK header missing: ${ANF_SDK_ROOT}/public/${ANF_HEADER}. Run python Configure.py with the anf sample selected.")
    endif()
endforeach()
target_include_directories(${TARGET_NAME} PRIVATE "${ANF_SDK_ROOT}/public")

if(ANDROID)
    if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
        message(FATAL_ERROR "ANF supports Android arm64-v8a only (requested ${ANDROID_ABI}).")
    endif()
    set(ANF_LIBRARY "${ANF_SDK_ROOT}/lib/${ANDROID_ABI}/libanf.so")
    if(NOT EXISTS "${ANF_LIBRARY}")
        message(FATAL_ERROR "ANF SDK library missing: ${ANF_LIBRARY}. Run python Configure.py with the anf sample selected.")
    endif()
    add_library(anf_sdk SHARED IMPORTED)
    set_target_properties(anf_sdk PROPERTIES
        IMPORTED_LOCATION "${ANF_LIBRARY}"
    )
    # Gradle packages the imported library in the APK automatically.
    target_link_libraries(${TARGET_NAME} c++_static anf_sdk)

    # Keep the upstream binary and header licenses with redistributed APKs.
    foreach(ANF_LICENSE LICENSE.txt LICENSE-BSD-3-Clause.txt)
        configure_file("${ANF_SDK_ROOT}/${ANF_LICENSE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/build/Media/Licenses/anf_sdk/${ANF_LICENSE}" COPYONLY)
    endforeach()
else()
    message(STATUS "ANF SDK runtime is Android-only; building the sample with SR/FG unavailable.")
endif()
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/build/Media/Licenses/anf/LICENSE.txt" COPYONLY)

#
# Setup asset source and target folders
#

# cmake will use our GameSampleAssets (default for no parameter) as root directory for any asset request (see FrameworkApplicationHelper.cmake for more info)
inject_root_asset_path()

# Register local variables for asset request, while also defining them in the C++ code for easy access
# Here we use the default destionation paths, all defined at FrameworkApplicationHelper.cmake
register_local_asset_path(SHADER_DESTINATION  "${DEFAULT_LOCAL_SHADER_DESTINATION}")
register_local_asset_path(MESH_DESTINATION    "${DEFAULT_LOCAL_MESH_DESTINATION}")
register_local_asset_path(TEXTURE_DESTINATION "${DEFAULT_LOCAL_TEXTURE_DESTINATION}")

#
# Add in the contents of 'shaders' directory
#
include(AddShadersDir)

# Search and include all project shaders
scan_for_shaders()

#
# Copy required models to local folders
#
include(ModelPackager)

# Scene GLTF
add_gltf(scenes/SteamPunkSauna/SteamPunkSauna.gltf)

#
# Convert and copy textures to local folders
#
include(TexturePackager)

# Scene Textures
add_textures_from_path(scenes/SteamPunkSauna UASTC)

# Supporting Textures
add_textures_from_path(textures)
