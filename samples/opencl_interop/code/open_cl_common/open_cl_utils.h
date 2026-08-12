//============================================================================================================
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#pragma once

#define CL_TARGET_OPENCL_VERSION 300

#include <CL/opencl.h>
#include <CL/cl_ext.h>
#include <CL/cl_ext_qcom.h>

// Function lists reused by declaration/definition/loading macros.
#define OPENCL_CORE_FUNCTIONS(X) \
    X(clCreateContext) \
    X(clGetDeviceIDs) \
    X(clGetPlatformIDs) \
    X(clCreateBuffer) \
    X(clReleaseMemObject) \
    X(clCreateProgramWithSource) \
    X(clBuildProgram) \
    X(clCreateKernel) \
    X(clSetKernelArg) \
    X(clEnqueueNDRangeKernel) \
    X(clFlush) \
    X(clFinish) \
    X(clCreateCommandQueueWithProperties) \
    X(clReleaseContext) \
    X(clGetPlatformInfo) \
    X(clGetExtensionFunctionAddressForPlatform) \
    X(clCreateImageWithProperties) \
    X(clCreateBufferWithProperties) \
    X(clGetDeviceInfo) \
    X(clWaitForEvents) \
    X(clGetEventInfo) \
    X(clGetProgramBuildInfo) \
    X(clGetProgramInfo) \
    X(clGetSupportedImageFormats) \
    X(clReleaseCommandQueue) \
    X(clReleaseEvent) \
    X(clReleaseKernel) \
    X(clReleaseProgram)

#define OPENCL_EXTENSION_FUNCTIONS(X) \
    X(clGetSemaphoreHandleForTypeKHR) \
    X(clReImportSemaphoreSyncFdKHR) \
    X(clEnqueueWaitSemaphoresKHR) \
    X(clEnqueueSignalSemaphoresKHR) \
    X(clCreateSemaphoreWithPropertiesKHR) \
    X(clReleaseSemaphoreKHR) \
    X(clEnqueueAcquireExternalMemObjectsKHR) \
    X(clEnqueueReleaseExternalMemObjectsKHR)

// Example expansion: 
// extern decltype(&clCreateContext) clCreateContext_ptr;
#define DECLARE_CL_FUNCTION(func_name) extern decltype(&func_name) func_name##_ptr;

OPENCL_CORE_FUNCTIONS(DECLARE_CL_FUNCTION)
OPENCL_EXTENSION_FUNCTIONS(DECLARE_CL_FUNCTION)

#undef DECLARE_CL_FUNCTION

cl_platform_id load_opencl();
void           unload_opencl();

// Expands calls from clCreateContext(...) to clCreateContext_ptr(...).
// Keep this explicit list in sync with OPENCL_CORE_FUNCTIONS and OPENCL_EXTENSION_FUNCTIONS;
// macros cannot generate #define directives.
#ifdef OPENCL_ENABLE_FUNCTION_REMAP

#   define clCreateContext clCreateContext_ptr
#   define clGetDeviceIDs clGetDeviceIDs_ptr
#   define clGetPlatformIDs clGetPlatformIDs_ptr
#   define clCreateBuffer clCreateBuffer_ptr
#   define clReleaseMemObject clReleaseMemObject_ptr
#   define clCreateProgramWithSource clCreateProgramWithSource_ptr
#   define clBuildProgram clBuildProgram_ptr
#   define clCreateKernel clCreateKernel_ptr
#   define clSetKernelArg clSetKernelArg_ptr
#   define clEnqueueNDRangeKernel clEnqueueNDRangeKernel_ptr
#   define clFlush clFlush_ptr
#   define clFinish clFinish_ptr
#   define clCreateCommandQueueWithProperties clCreateCommandQueueWithProperties_ptr
#   define clReleaseContext clReleaseContext_ptr
#   define clGetPlatformInfo clGetPlatformInfo_ptr
#   define clGetExtensionFunctionAddressForPlatform clGetExtensionFunctionAddressForPlatform_ptr
#   define clCreateImageWithProperties clCreateImageWithProperties_ptr
#   define clCreateBufferWithProperties clCreateBufferWithProperties_ptr
#   define clGetDeviceInfo clGetDeviceInfo_ptr
#   define clWaitForEvents clWaitForEvents_ptr
#   define clGetEventInfo clGetEventInfo_ptr
#   define clGetProgramBuildInfo clGetProgramBuildInfo_ptr
#   define clGetProgramInfo clGetProgramInfo_ptr
#   define clGetSupportedImageFormats clGetSupportedImageFormats_ptr
#   define clReleaseCommandQueue clReleaseCommandQueue_ptr
#   define clReleaseEvent clReleaseEvent_ptr
#   define clReleaseKernel clReleaseKernel_ptr
#   define clReleaseProgram clReleaseProgram_ptr

#   define clGetSemaphoreHandleForTypeKHR clGetSemaphoreHandleForTypeKHR_ptr
#   define clReImportSemaphoreSyncFdKHR clReImportSemaphoreSyncFdKHR_ptr
#   define clEnqueueWaitSemaphoresKHR clEnqueueWaitSemaphoresKHR_ptr
#   define clEnqueueSignalSemaphoresKHR clEnqueueSignalSemaphoresKHR_ptr
#   define clCreateSemaphoreWithPropertiesKHR clCreateSemaphoreWithPropertiesKHR_ptr
#   define clReleaseSemaphoreKHR clReleaseSemaphoreKHR_ptr
#   define clEnqueueAcquireExternalMemObjectsKHR clEnqueueAcquireExternalMemObjectsKHR_ptr
#   define clEnqueueReleaseExternalMemObjectsKHR clEnqueueReleaseExternalMemObjectsKHR_ptr

#endif