cmake_minimum_required (VERSION 3.21)

project (opencl_interop C CXX)
set(CMAKE_CXX_STANDARD 20)

if(NOT ANDROID)
    message(FATAL_ERROR "opencl_interop sample is Android-only.")
endif()

#
# Source files included in this application.
#
set(CPP_SRC 
    code/main/application.cpp
    code/main/application.hpp
    code/main/interop_common.cpp
    code/main/interop_common.hpp
    code/main/interop_buffer.cpp
    code/main/interop_buffer.hpp
    code/main/interop_image_linear.cpp
    code/main/interop_image_linear.hpp
    code/main/interop_image_optimal.cpp
    code/main/interop_image_optimal.hpp
    code/open_cl_common/open_cl_utils.cpp
    code/open_cl_common/open_cl_utils.h
)

#
# Setup the module path to include the 'project directory' (project/windows or project/android)
#
if(NOT DEFINED PROJECT_ROOT_DIR)
    set(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR})   # Windows can use CMAKE_SOURCE_DIR, Android needs build.gradle needs "-DPROJECT_ROOT_DIR=${project.rootDir}" in call to cmake set since there is not a 'top' cmakefile (gradle is top level)
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_ROOT_DIR}/cmake ${FRAMEWORK_DIR}/cmake ${CMAKE_CURRENT_LIST_DIR}/cmake)

#
# Do all the build steps for a Framework application.
# needs Framework_dir and project_name variables.
#
include(FrameworkApplicationHelper)

# OpenCL dependency under code/open_cl_common
# Build as a normal CMake target and link it to ${TARGET_NAME}.
add_library(opencl INTERFACE)
set(OPENCL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/code/open_cl_common)

set(REQUIRED_OPENCL_HEADERS
    ${OPENCL_INCLUDE_DIR}/CL/opencl.h
    ${OPENCL_INCLUDE_DIR}/CL/cl.h
    ${OPENCL_INCLUDE_DIR}/CL/cl_platform.h
    ${OPENCL_INCLUDE_DIR}/CL/cl_version.h
    ${OPENCL_INCLUDE_DIR}/CL/cl_ext.h
    ${OPENCL_INCLUDE_DIR}/CL/cl_ext_qcom.h
    ${OPENCL_INCLUDE_DIR}/CL/cl_gl.h
)

foreach(OPENCL_HEADER ${REQUIRED_OPENCL_HEADERS})
    if(NOT EXISTS ${OPENCL_HEADER})
        message(FATAL_ERROR
            "Missing OpenCL SDK header: ${OPENCL_HEADER}\n"
            "Please download the Qualcomm Adreno OpenCL SDK from "
            "https://softwarecenter.qualcomm.com/catalog/item/Adreno_OpenCL_SDK "
            "and copy opencl-sdk/inc/CL to "
            "samples/opencl_interop/code/open_cl_common/CL. "
            "See samples/opencl_interop/README.md for details.")
    endif()
endforeach()

target_sources(opencl INTERFACE ${OPENCL_INCLUDE_DIR}/CL/opencl.h)
target_include_directories(opencl SYSTEM INTERFACE ${OPENCL_INCLUDE_DIR})
target_link_libraries(${TARGET_NAME} opencl)

#
# Setup asset source and target folders
#

# cmake will use our GameSampleAssets (default for no parameter) as root directory for any asset request
# (see FrameworkApplicationHelper.cmake for more info)
inject_root_asset_path()

# Register local variables for asset request, while also defining them in the C++ code for easy access
# Here we use the default destination paths, all defined at FrameworkApplicationHelper.cmake
register_local_asset_path(SHADER_DESTINATION "${DEFAULT_LOCAL_SHADER_DESTINATION}")
register_local_asset_path(KERNEL_DESTINATION "build/Media/Kernels")

#
# Add in the contents of 'shaders' directory
#
include(AddShadersDir)

# Search and include all project shaders
scan_for_shaders()

#
# Add in the contents of 'kernels' directory
#
include(AddKernelsDir)

# Search and include all project OpenCL kernels
scan_for_kernels()
