//============================================================================================================
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#define ENABLE_QCOM_EXT // Enable if you want to use QCOM API extensions
#include "open_cl_utils.h"
#include "system/os_common.h"

#ifndef __ANDROID__
#   error "opencl_interop sample is Android-only."
#endif

#include <dlfcn.h>

namespace
{
    void* g_opencl_library = nullptr;

    template <typename T>
    bool LoadOpenClSymbol(const char* name, T& out)
    {
        out = reinterpret_cast<T>(dlsym(g_opencl_library, name));
        if (!out)
        {
            LOGE("Failed to load OpenCL symbol: %s", name);
            return false;
        }
        return true;
    }

    template <typename T>
    bool LoadOpenClExtension(cl_platform_id platform, const char* name, T& out)
    {
        out = reinterpret_cast<T>(clGetExtensionFunctionAddressForPlatform_ptr(platform, name));
        if (!out)
        {
            LOGE("Failed to load OpenCL extension function: %s", name);
            return false;
        }
        return true;
    }

    void ResetOpenClFunctions()
    {
#define RESET_CL_FUNCTION(func_name) func_name##_ptr = nullptr;

        OPENCL_CORE_FUNCTIONS(RESET_CL_FUNCTION)
        OPENCL_EXTENSION_FUNCTIONS(RESET_CL_FUNCTION)

#undef RESET_CL_FUNCTION
    }

    bool LoadCoreOpenClFunctions()
    {
        bool loaded = true;

        // Example expansion: 
        // loaded &= LoadOpenClSymbol("clCreateContext", clCreateContext_ptr);
#define LOAD_CL_FUNCTION(func_name) loaded &= LoadOpenClSymbol(#func_name, func_name##_ptr);

        OPENCL_CORE_FUNCTIONS(LOAD_CL_FUNCTION)

#undef LOAD_CL_FUNCTION

        return loaded;
    }

    bool LoadOpenClExtensionFunctions(cl_platform_id platform)
    {
        bool loaded = true;

        // Example expansion: 
        // loaded &= LoadOpenClExtension(platform, "clEnqueueAcquireExternalMemObjectsKHR", clEnqueueAcquireExternalMemObjectsKHR_ptr);
#define LOAD_CL_EXTENSION_FUNCTION(func_name) loaded &= LoadOpenClExtension(platform, #func_name, func_name##_ptr);

        OPENCL_EXTENSION_FUNCTIONS(LOAD_CL_EXTENSION_FUNCTION)

#undef LOAD_CL_EXTENSION_FUNCTION

        return loaded;
    }
}

// Example expansion: 
// decltype(&clCreateContext) clCreateContext_ptr = nullptr;
#define DEFINE_CL_FUNCTION(func_name) decltype(&func_name) func_name##_ptr = nullptr;

OPENCL_CORE_FUNCTIONS(DEFINE_CL_FUNCTION)
OPENCL_EXTENSION_FUNCTIONS(DEFINE_CL_FUNCTION)

#undef DEFINE_CL_FUNCTION

cl_platform_id load_opencl()
{
    if (g_opencl_library)
    {
        unload_opencl();
    }

    g_opencl_library = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (g_opencl_library == nullptr)
    {
        LOGE("Failed to open libOpenCL.so: %s", dlerror());
        return nullptr;
    }

    if (!LoadCoreOpenClFunctions())
    {
        unload_opencl();
        return nullptr;
    }

    cl_platform_id platform_id   = nullptr;
    cl_uint        num_platforms = 0;
    const cl_int   result        = clGetPlatformIDs_ptr(1, &platform_id, &num_platforms);
    if (result != CL_SUCCESS || platform_id == nullptr || num_platforms == 0)
    {
        LOGE("clGetPlatformIDs failed: result=%d, num_platforms=%u", result, num_platforms);
        unload_opencl();
        return nullptr;
    }

    if (!LoadOpenClExtensionFunctions(platform_id))
    {
        unload_opencl();
        return nullptr;
    }

    return platform_id;
}

void unload_opencl()
{
    if (g_opencl_library)
    {
        dlclose(g_opencl_library);
        g_opencl_library = nullptr;
    }

    ResetOpenClFunctions();
}
