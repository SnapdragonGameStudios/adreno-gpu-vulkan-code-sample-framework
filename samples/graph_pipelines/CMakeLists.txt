cmake_minimum_required (VERSION 3.21)

project (graph_pipelines C CXX)
set(CMAKE_CXX_STANDARD 20)

#
# Source files included in this application.
#

set(CPP_SRC code/main/application.cpp
            code/main/application.hpp
			code/main/ml/GraphPipelineTypes.hpp
			code/main/ml/DataGraphPipeline.hpp
			code/main/ml/DataGraphPipeline.cpp
			code/main/ml/GraphDispatch.hpp
			code/main/ml/GraphDispatch.cpp
			code/main/ml/QcomDataGraphModel.hpp
			code/main/ml/QcomDataGraphModel.cpp
			code/main/ml/TensorResources.hpp
			code/main/ml/TensorResources.cpp
)

#
# Setup the module path to include the 'project directory' (project/windows or project/android)
#
if(NOT DEFINED PROJECT_ROOT_DIR)
    set(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR})   # Windows can use CMAKE_SOURCE_DIR, Android needs build.gradle needs "-DPROJECT_ROOT_DIR=${project.rootDir}" in call to cmake set since there is not a 'top' cmakefile (gradle is top level)
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_ROOT_DIR}/cmake)

#
# Do all the build steps for a Framework application.
# needs Framework_dir and project_name variables.
#
include(FrameworkApplicationHelper)

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
register_local_asset_path(MISC_DESTINATION    "${DEFAULT_LOCAL_MISC_DESTINATION}")

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