//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "cooperative_matrix_tester.hpp"
#include "vulkan/extensionHelpers.hpp"
#include "vulkan/extensionLib.hpp"
#include <../external/glslang/glslang/Include/glslang_c_interface.h>
#include <../external/glslang/glslang/Public/resource_limits_c.h>

// Runtime shaders
#include "runtime_shaders/MxM_Basic.hpp"
#include "runtime_shaders/MxM_VecToMat.hpp"
#include "runtime_shaders/Conv.hpp"

#pragma push_macro("BOOL")
#define BOOL HALF_BOOL
#include "half/half.h"
#pragma pop_macro("BOOL")

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 1
#endif

#include "imgui.h"
#include "cooperative_matrix_conversion.hpp"
#include "system/config.h"

VAR(bool, gCoopAutoRun, false, kVariableNonpersistent);
VAR(bool, gCoopValidate, false, kVariableNonpersistent);
VAR(int, gCoopRepeats, 32, kVariableNonpersistent);
VAR(int, gCoopTest, 0, kVariableNonpersistent);
VAR(bool, gCoopLegacyLayouts, false, kVariableNonpersistent);
VAR(int, gCoopKBlocks, 128, kVariableNonpersistent);

#include <random>
#include <iostream>
#include <filesystem>
#include <sstream>

#define CHECK_VK(cmd)                                                                           \
    {                                                                                           \
        VkResult local_result = cmd;                                                            \
        if(local_result == VK_SUCCESS){}                                                        \
        else if (local_result == VK_NOT_READY || local_result == VK_TIMEOUT ||                  \
                 local_result == VK_EVENT_SET || local_result == VK_EVENT_RESET ||              \
                 local_result == VK_INCOMPLETE)                                                 \
        {                                                                                       \
            LOGW("CHECK_VK: Warning - %s returned %d", #cmd, static_cast<int>(local_result));   \
        }                                                                                       \
        else                                                                                    \
        {                                                                                       \
            LOGE("CHECK_VK: Error - %s returned %d", #cmd, static_cast<int>(local_result));     \
            assert(false);                                                                      \
        }                                                                                       \
    }


#define CHECK_BOOL(expr)                                                                        \
    {                                                                                           \
        bool local_result = (expr);                                                             \
        if (!local_result)                                                                      \
        {                                                                                       \
            LOGE("CHECK_BOOL: Error - %s evaluated to false", #expr);                           \
        }                                                                                       \
    }

namespace
{
    enum gpu_vendors
    {
        VK_VENDOR_ID_UNKNOWN = 0,
        VK_VENDOR_ID_NVIDIA = 0x10de,
        VK_VENDOR_ID_QUALCOMM = 0x5143,
        VK_VENDOR_ID_AMD = 0x1002,
        VK_VENDOR_ID_INTEL = 0x8086,
        VK_VENDOR_ID_APPLE = 0x106b
    };

    enum gpu_tiers
    {
        TIER_UNKNOWN = 0,
        QCOM_TIER_1 = 0x44050000,
        QCOM_TIER_2 = 0x44050A30,
        QCOM_TIER_3 = 0x44070040,
        QCOM_TIER_4 = 0x36334630,
        QCOM_TIER_5 = 0x44051430,
        OTHER = 0x1F14
    };

    const char* GetMatrixTypeName(VkComponentTypeKHR component_type)
    {
        switch (component_type)
        {
            case VK_COMPONENT_TYPE_FLOAT16_KHR: return "FLOAT16";
            case VK_COMPONENT_TYPE_FLOAT32_KHR: return "FLOAT32";
            case VK_COMPONENT_TYPE_FLOAT64_KHR: return "FLOAT64";
            case VK_COMPONENT_TYPE_SINT8_KHR: return "SINT8";
            case VK_COMPONENT_TYPE_SINT16_KHR: return "SINT16";
            case VK_COMPONENT_TYPE_SINT32_KHR: return "SINT32";
            case VK_COMPONENT_TYPE_SINT64_KHR: return "SINT64";
            case VK_COMPONENT_TYPE_UINT8_KHR: return "UINT8";
            case VK_COMPONENT_TYPE_UINT16_KHR: return "UINT16";
            case VK_COMPONENT_TYPE_UINT32_KHR: return "UINT32";
            case VK_COMPONENT_TYPE_UINT64_KHR: return "UINT64";
            case VK_COMPONENT_TYPE_BFLOAT16_KHR: return "BFLOAT16";
            case VK_COMPONENT_TYPE_SINT8_PACKED_NV: return "SINT8_PACKED";
            case VK_COMPONENT_TYPE_UINT8_PACKED_NV: return "UINT8_PACKED";
            case VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT: return "FLOAT8_E4M3";
            case VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT: return "FLOAT8_E5M2";
            default: return "UNKNOWN TYPE";
        }
    }

    const char* GetMatrixComponentTypeName(VkComponentTypeKHR type)
    {
        switch (type)
        {
            case VK_COMPONENT_TYPE_FLOAT64_KHR: return "FP64";
            case VK_COMPONENT_TYPE_FLOAT32_KHR: return "FP32";
            case VK_COMPONENT_TYPE_FLOAT16_KHR: return "FP16";
            case VK_COMPONENT_TYPE_SINT8_KHR:   return "INT8";
            case VK_COMPONENT_TYPE_SINT16_KHR:  return "INT16";
            case VK_COMPONENT_TYPE_SINT32_KHR:  return "INT32";
            case VK_COMPONENT_TYPE_SINT64_KHR:  return "INT64";
            default:                            return "UNKNOWN";
        }
    }

    bool FindMatrixProperty(
        std::span<VkCooperativeMatrixPropertiesKHR> cooperativeMatrixProperties,
        VkCooperativeMatrixPropertiesKHR &cooperativeMatrixProps, 
        uint32_t MSize, 
        uint32_t NSize, 
        uint32_t KSize,
        VkComponentTypeKHR AType, 
        VkComponentTypeKHR BType, 
        VkComponentTypeKHR CType, 
        VkComponentTypeKHR RType)
    {
        bool valid_testtypes = false;

        int32_t matrixprop;
        for(matrixprop = 0; matrixprop < cooperativeMatrixProperties.size() && !valid_testtypes; ++matrixprop)
        {
            if (cooperativeMatrixProperties[matrixprop].scope == VK_SCOPE_SUBGROUP_KHR &&
                !cooperativeMatrixProperties[matrixprop].saturatingAccumulation &&
                (cooperativeMatrixProperties[matrixprop].ResultType == RType) &&
                (cooperativeMatrixProperties[matrixprop].CType       == CType) &&
                (cooperativeMatrixProperties[matrixprop].BType       == BType) &&
                (cooperativeMatrixProperties[matrixprop].AType       == AType) &&
                (MSize != 0 ? cooperativeMatrixProperties[matrixprop].MSize == MSize : true) &&
                (NSize != 0 ? cooperativeMatrixProperties[matrixprop].NSize == NSize : true) &&
                (KSize != 0 ? cooperativeMatrixProperties[matrixprop].KSize == KSize : true) )
            {
                valid_testtypes = true;
                cooperativeMatrixProps = cooperativeMatrixProperties[matrixprop];
            }
        }

        return valid_testtypes;
    }


    static const char* ShaderPaths[]
    {
        Test01_MxM_Basic,
        Test02_MxM_VecToMat,
        Test03_CONV,
    };

    struct TestCase
    {
        TestType testType;
        VkComponentTypeKHR inputType;
        VkComponentTypeKHR outputType;

        // TOTAL_M, TOTAL_N, TOTAL_K is the size of the full R=AxB+C matrix multiply
        uint32_t TOTAL_M;
        uint32_t TOTAL_N;
        uint32_t TOTAL_K;

        // Each cooperative matrix multiply is R[TILE_M, TILE_N] = A[TILE_M, TILE_K] x B[TILE_K, TILE_N] + C[TILE_M, TILE_N]
        uint32_t TILE_M;
        uint32_t TILE_N;
        uint32_t TILE_K;

        bool layoutA_Mfirst;
        bool layoutB_Kfirst;
        bool layoutB_Nfirst;
        bool layoutA_TiledKfirst;
        bool layoutB_TiledKfirst;
        bool layoutC_Mfirst;
        bool layoutR_Mfirst;

        uint32_t strideAinElements;
        uint32_t strideBinElements;
        uint32_t strideCinElements;
        uint32_t strideRinElements;
    };

    struct sComponentTypeInfo
    {
        const char* typeName;
        uint32_t bits;
    };

    struct sComponentTypeInfo ComponentTypeInfo[] =
    {                       // From vulkan_core.h
        { "float16",  16 }, // VK_COMPONENT_TYPE_FLOAT16_KHR = 0,
        { "float32",  32 }, // VK_COMPONENT_TYPE_FLOAT32_KHR = 1,
        { "float64",  64 }, // VK_COMPONENT_TYPE_FLOAT64_KHR = 2,
        { "int8",     8 },  // VK_COMPONENT_TYPE_SINT8_KHR = 3,
        { "int16",    16 }, // VK_COMPONENT_TYPE_SINT16_KHR = 4,
        { "int32",    32 }, // VK_COMPONENT_TYPE_SINT32_KHR = 5,
        { "int64",    64 }, // VK_COMPONENT_TYPE_SINT64_KHR = 6,
        { "uint8",    8 },  // VK_COMPONENT_TYPE_UINT8_KHR = 7,
        { "uint16",   16 }, // VK_COMPONENT_TYPE_UINT16_KHR = 8,
        { "uint32",   32 }, // VK_COMPONENT_TYPE_UINT32_KHR = 9,
        { "uint64",   64 }, // VK_COMPONENT_TYPE_UINT64_KHR = 10,
    };

    const char* scopeString[] = {
        "invalid",
        "device",
        "workgroup",
        "subgroup",
        "invalid",
        "queuefamily",
    };

    const char* GetLayoutName(MatrixLayout layout)
    {
        switch (layout)
        {
            case MatrixLayout::K_FIRST: return "K-first";
            case MatrixLayout::M_FIRST: return "M-first";
            case MatrixLayout::N_FIRST: return "N-first";
            case MatrixLayout::TILED_K_FIRST: return "TiledK-first";
            default: return "Unknown";
        }
    }

    struct MatrixDesc
    {
        struct
        {
            uint32_t rows, cols;
        } dims;
        VkComponentTypeKHR dataType;
        size_t elementSize;
        VkDeviceSize bufferSize;
        uint32_t totalElements;

        // Create a host- and device-local buffer for each input and output.
        // Descriptors point at the device buffers.
        VkBuffer hostBuffer;
        VkDeviceMemory hostMemory;
        VkBuffer deviceBuffer;
        VkDeviceMemory deviceMemory;
        void* ptr;

        bool isFloatType() const
        {
            switch (dataType)
            {
            default:
                return false;
            case VK_COMPONENT_TYPE_FLOAT16_KHR:
            case VK_COMPONENT_TYPE_FLOAT32_KHR:
            case VK_COMPONENT_TYPE_FLOAT64_KHR:
                return true;
            }
        }

        void setDataFloat(uint32_t i, float value)
        {
            if (dataType == VK_COMPONENT_TYPE_FLOAT32_KHR)
            {
                ((float*)ptr)[i] = value;
            }
            else
            {
                uint32_t asInt = *(uint32_t*)&value;
                int sign = (asInt & 0x80000000) >> 31;
                int exp = ((asInt & 0x7f800000) >> 23) - 127;
                int mantissa = (asInt & 0x7FFFFF);

                sign = sign << 15;
                exp = (exp + 15) << 10;
                mantissa = mantissa >> (23 - 10);

                if (asInt != 0) {
                    asInt = sign | exp | mantissa;
                }

                ((uint16_t*)ptr)[i] = asInt;
            }
        }

        float getDataFloat(uint32_t i) const
        {
            if (dataType == VK_COMPONENT_TYPE_FLOAT32_KHR)
            {
                return ((float*)ptr)[i];
            }
            else
            {
                uint32_t asInt = ((uint16_t*)ptr)[i];
                int sign = (asInt & 0x8000) >> 15;
                int exp = ((asInt & 0x7c00) >> 10) - 15;
                int mantissa = (asInt & 0x3FF);

                sign = sign << 31;
                exp = (exp + 127) << 23;
                mantissa = mantissa << (23 - 10);

                if (asInt != 0) {
                    asInt = sign | exp | mantissa;
                }

                return *(float*)&asInt;
            }
        }

        float getDataFloat(int m, int n, bool colMajor) const
        {
            return getDataFloat(colMajor ? (n * dims.rows + m) : (m * dims.cols + n));
        }

        void setDataInt(uint32_t i, uint32_t value)
        {
            assert(ComponentTypeInfo[dataType].bits == 8 || ComponentTypeInfo[dataType].bits == 32);
            switch (dataType) {
            default: assert(0); // fallthrough
            case VK_COMPONENT_TYPE_UINT8_KHR:    ((uint8_t*)ptr)[i] = (uint8_t)value; break;
            case VK_COMPONENT_TYPE_UINT32_KHR:   ((uint32_t*)ptr)[i] = (uint32_t)value; break;
            case VK_COMPONENT_TYPE_SINT8_KHR:    ((int8_t*)ptr)[i] = (int8_t)value; break;
            case VK_COMPONENT_TYPE_SINT32_KHR:   ((int32_t*)ptr)[i] = (int32_t)value; break;
            }
        }

        uint32_t getDataInt(uint32_t i) const
        {
            assert(ComponentTypeInfo[dataType].bits == 8 || ComponentTypeInfo[dataType].bits == 32);
            switch (dataType) {
            default: assert(0); // fallthrough
            case VK_COMPONENT_TYPE_UINT8_KHR:	return ((uint8_t*)ptr)[i];
            case VK_COMPONENT_TYPE_UINT32_KHR:	return ((uint32_t*)ptr)[i];
            case VK_COMPONENT_TYPE_SINT8_KHR:	return ((int8_t*)ptr)[i];
            case VK_COMPONENT_TYPE_SINT32_KHR:	return ((int32_t*)ptr)[i];
            }
        }

        uint32_t getDataInt(int m, int n, bool colMajor) const
        {
            return getDataInt(colMajor ? (n * dims.rows + m) : (m * dims.cols + n));
        }
    };


    template<typename T>
    void InitMatrix(T* matrix, unsigned int mrows, unsigned int mcols, unsigned int stride, FillDataType init, unsigned int set_num_decimals=2)
    {
        struct MatrixKey
        {
            unsigned int mrows;
            unsigned int mcols;
            unsigned int stride;
            FillDataType init;
            unsigned int set_num_decimals;

            bool operator==(const MatrixKey& other) const
            {
                return mrows == other.mrows &&
                       mcols == other.mcols &&
                       stride == other.stride &&
                       init == other.init &&
                       set_num_decimals == other.set_num_decimals;
            }
        };

        struct MatrixKeyHasher
        {
            std::size_t operator()(const MatrixKey& key) const
            {
                std::size_t h1 = std::hash<unsigned int>{}(key.mrows);
                std::size_t h2 = std::hash<unsigned int>{}(key.mcols);
                std::size_t h3 = std::hash<unsigned int>{}(key.stride);
                std::size_t h4 = std::hash<int>{}(key.init);
                std::size_t h5 = std::hash<unsigned int>{}(key.set_num_decimals);
                return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
            }
        };

        static std::unordered_map<MatrixKey, std::vector<T>, MatrixKeyHasher> cache;

        MatrixKey key{ mrows, mcols, stride, init, set_num_decimals };

        auto it = cache.find(key);
        if (it != cache.end())
        {
            std::memcpy(matrix, it->second.data(), mrows * stride * sizeof(T));
            return;
        }

        std::vector<T> temp_matrix(mrows * stride, T(0));

        float r, rr;

        float flow  = 0.0f;
        float fhigh = 1.0f;
        int range   = 3; // 3 -> -1, 0 and 1
        static int  counter = 0;
        float const_init = 1.0f;
        int sequence = 3;
        if (sizeof(T) == 1) sequence = 255;

        static unsigned seed = 3;
        std::srand(seed++); // srand seed doesn't work with time(0)
        std::cout << "Initializing ROWxCOL=" << mrows << "x" << mcols << " matrix (stride=" << stride << ") with init option = " << init << " and using " << set_num_decimals << " number of decimals\n";

        // Set the buffer to '0' in case mcols < stride, init only mrows*mcols elements, 
        memset((void*)matrix, 0, size_t(mrows * stride));

        //	unsigned int counter=1; // for debugging purpose
        for (unsigned int row = 0; row < mrows; row++) // y
        {
            for (unsigned int col = 0; col < mcols; col++) // x
            {
                switch (init)
                {
                case FILL_WITH_ZERO:
                    r = 0;
                    break;
                case FILL_WITH_CONSTANTS:
                    r = const_init; // default const_init=1.0f
                    break;
                case FILL_WITH_RANDON_UINT:
                    r = float(std::rand() % range); // defualt range=3 -> init_matrix will be 0, 1, 2
                    break;
                case FILL_WITH_RANDON_INT:
                    r = float(std::rand() % range - ((range - 1) / 2)); // defualt range=3 -> init_matrix will be - 1, 0 and 1 -> guarantee average 0 for dot products preventing float16 going out of range
                    break;
                case FILL_SEQUENCE_INT:
                    r = float(counter++ % sequence);// + const_init;
                    break;
                case FILL_WITH_RANDOM_LOW_HIGH_INT:
                    r = T(std::rand() % int(fhigh)) + int(flow);
                    break;
                case FILL_WITH_RANDOM_FLOAT:
                    r = flow + float(rand()) / ((float(RAND_MAX) / (fhigh - flow)));
                    break;
                case FILL_WITH_RANDOM_PLUS1_MINUS1_FLOAT:
                    //r = float(rand());
                    r = rand() > RAND_MAX/2 ? float(1.0) : float(-1.0);
                    break;
                default:
                    LOGE("Invalid InitMatrix(...) initialization option '-i:%d'", init);
                }
                // Force to fixed number of decimals based on user input
                std::ostringstream o;
                o << std::setprecision(set_num_decimals) << std::fixed << r;
                rr = std::stof(o.str());
                // load the matrix
                temp_matrix[row * stride + col] = T(rr);
            }
        }

        std::memcpy(matrix, temp_matrix.data(), mrows* stride * sizeof(T));
        cache[key] = std::move(temp_matrix);
    }

    template<typename T>
    void TransposeMatrix(T* matrix, const unsigned int& mrows, const unsigned int& mcols, const char *info)
    {
        std::cout << "\nTransposing MxM(" << info << ") on CPU, input type '" << typeid(matrix).name() << "', number of rows: '" << mrows << "', number of columns: '" << mcols << "', IT'LL TAKE SOME TIME!!!\n\n";

        unsigned int count = mcols * mrows;

        for (unsigned int col = 0; col < mcols; ++col)
        {
            unsigned int count_adjustment = mcols - col - 1;

            for (unsigned int row = 0, step = 1; row < mrows; ++row, step += count_adjustment)
            {
                unsigned int last = count - (row + col * mrows);
                unsigned int first = last - step;

                std::rotate(matrix + first, matrix + first + 1, matrix + last);
            }
        }

        //std::swap(mrows, mcols);
        std::cout << "\nFinished Transposing MxM on CPU\n";
    }

    template<typename T>
    void TransposeMatrix(T* matrix, const unsigned int& mrows, const unsigned int& mcols, T* matrixOut)
    {
        std::cout << "\nTransposing MxM on CPU, input type '" << typeid(matrix).name() << "', output type '" << typeid(matrixOut).name() << "', number of rows: '" << mrows << "', number of columns: '" << mcols << "', IT'LL TAKE SOME TIME!!!\n\n";

        unsigned int count = mcols * mrows;

        for (unsigned int col = 0; col < mcols; col++)
            for (unsigned int row = 0; row < mrows; row++)
                matrixOut[col*mrows + row] = matrix[row*mcols + col];

        std::cout << "\nFinished Transposing MxM on CPU\n";
    }

    // Transform a matrix from row-major (K-first) layout to Tiled-K-first layout.
    // Applied to matrices A and B before uploading to GPU for TT_MXM_BASIC.
    // Do NOT apply to matrices C or D (result).
    //
    // The tileK parameter must be set to:
    //   - TILE_K (cooperativeMatrixProps.KSize) for matrix A
    //   - TILE_N (cooperativeMatrixProps.NSize) for matrix B
    //
    // Input  layout: matrix[mm * k + kk]
    // Output layout: matrixOut[(kk / tileK) * tileK * m + kk % tileK + mm * tileK]
    template<typename T>
    void TransformMatrixToTiledKfirst(const T* matrix, const uint32_t m, const uint32_t k, T* matrixOut, uint32_t tileK)
    {
        std::cout << "\nTransforming " << m << "x" << k
                  << " matrix to Tiled-K-first layout (tileK=" << tileK << ")\n";

        for (uint32_t kk = 0; kk < k; kk++)
        {
            for (uint32_t mm = 0; mm < m; mm++)
            {
                matrixOut[(kk / tileK) * tileK * m + kk % tileK + mm * tileK] =
                    matrix[mm * k + kk];
            }
        }

        std::cout << "Finished Tiled-K-first transform\n";
    }
} // anonymous namespace

CooperativeMatrixRunner::CooperativeMatrixRunner(Vulkan& vulkan_instance)
    : m_vulkan_instance(vulkan_instance)
{
    glslang_initialize_process();
}

CooperativeMatrixRunner::~CooperativeMatrixRunner()
{
    glslang_finalize_process();
}

bool CooperativeMatrixRunner::InitializeRunner()
{
    if (!m_vulkan_instance.HasLoadedVulkanDeviceExtension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
    {
        LOGE("Required Extension not supported %s", VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        LOGE("Platform does not support Cooperative Matrices. Cannot test.\n");

        return false;
    }

    auto cooperativeMatrixEXT = m_vulkan_instance.GetExtension<ExtensionLib::Ext_VK_KHR_cooperative_matrix>();
    if (!cooperativeMatrixEXT)
    {
        LOGE("Ext_VK_KHR_cooperative_matrix potentially unresolved!");
        return false;
    }

    // select supported cooperative matrix types/sizes
    uint32_t nCoopMatrixPropCount = 0;
    CHECK_VK(cooperativeMatrixEXT->m_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
        m_vulkan_instance.m_VulkanGpu,
        &nCoopMatrixPropCount,
        NULL
    ));

    if (nCoopMatrixPropCount == 0) return false;
    m_hFoundCooperativeMatrices.resize(nCoopMatrixPropCount);
    for (auto& matrixProp : m_hFoundCooperativeMatrices)
    {
        matrixProp.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        matrixProp.pNext = nullptr;
    }

    CHECK_VK(cooperativeMatrixEXT->m_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
            m_vulkan_instance.m_VulkanGpu,
            &nCoopMatrixPropCount,
            m_hFoundCooperativeMatrices.data()
    ));

    LOGI("Found Cooperative Matrices:\n");
    for (auto& cm : m_hFoundCooperativeMatrices)
    {
        LOGI("\tMxNxK: %ux%ux%u\n", cm.MSize, cm.NSize, cm.KSize);
        LOGI("\tA: %s | ", GetMatrixTypeName(cm.AType));
        LOGI("B: %s | ", GetMatrixTypeName(cm.BType));
        LOGI("C: %s | ", GetMatrixTypeName(cm.CType));
        LOGI("D: %s\n",  GetMatrixTypeName(cm.ResultType));
        LOGI("\tSaturating Accumulation: %u | Scope: %u\n\n", cm.saturatingAccumulation, cm.scope);
    }

    // Setup the test templates
    m_test_group_templates.push_back(TestGroupTemplateDescription{
        VK_COMPONENT_TYPE_FLOAT32_KHR ,
        VK_COMPONENT_TYPE_FLOAT32_KHR ,
        {
            {8,  6, 128, // SizeInBlocks
             64, 64, 8}, // Size (tile)

            {8,  12, 128,
             64, 32, 8},

            {8,  24, 128,
             64, 16, 8}
        } });

    m_test_group_templates.push_back(TestGroupTemplateDescription{
        VK_COMPONENT_TYPE_FLOAT16_KHR ,
        VK_COMPONENT_TYPE_FLOAT16_KHR ,
        {
            {8, 6, 128, // SizeInBlocks
             0, 64, 0}, // Size (tile)

            {8, 12, 128,
             0, 32, 0},

            {8, 24, 128,
             0, 16, 0}
        } });

    m_test_group_templates.push_back(TestGroupTemplateDescription{
        VK_COMPONENT_TYPE_SINT8_KHR ,
        VK_COMPONENT_TYPE_SINT32_KHR ,
        {
            {8, 6, 128, // SizeInBlocks
             0, 64, 0}, // Size (tile)

            {8, 12, 128,
             0, 32, 0},

            {8, 24, 128,
             0, 16, 0}
        } });

    auto unsignedTemplate = m_test_group_templates.back();
    unsignedTemplate.input_type = VK_COMPONENT_TYPE_UINT8_KHR;
    unsignedTemplate.output_type = VK_COMPONENT_TYPE_UINT32_KHR;
    m_test_group_templates.push_back(unsignedTemplate);
    if (gCoopAutoRun)
    {
        m_test_repeats = std::clamp(gCoopRepeats, 1, 1024);
        m_validate_matrix_result = gCoopValidate;
        m_test_type = static_cast<TestType>(std::clamp(gCoopTest, 0, int(TT_COUNT) - 1));
        m_legacy_layouts = gCoopLegacyLayouts;
        for (auto& t : m_test_group_templates)
            for (auto& size : t.size_configurations)
                size.KSizeInBlocks = std::clamp(gCoopKBlocks, 1, 128);
        PrepareTestSession();
        LOGI("COOP_SESSION_BEGIN tests=%u", m_total_tests);
    }
    return true;
}

bool CooperativeMatrixRunner::TriggerPendingTests()
{
    if (!m_is_processing_tests)
    {
        return true;
    }

    for (auto& test_group : m_test_groups)
    {
        for (auto& test_entry : test_group.test_entries)
        {
            if (test_entry.test_descriptions.size() != test_entry.test_results.size())
            {
                for (const auto& test_description : test_entry.test_descriptions)
                {
                    const auto test_result = RunTest(test_description);
                    if (test_result)
                    {
                        test_entry.test_results.push_back(test_result.value());
                    }
                    else
                    {
                        test_entry.test_results.push_back(TestResult());
                    } 

                    m_total_processed_tests++;
                }

                // Process a single test entry per frame (so we can display progress on the UI)
                return true;
            }
        }
    }

    m_is_processing_tests = false;
    LOGI("COOP_SESSION_DONE tests=%u", m_total_processed_tests);

    return true;
}

void CooperativeMatrixRunner::RenderUI()
{
    const bool disable_ui = m_is_processing_tests;
    ImGui::BeginDisabled(disable_ui);
    ImGui::BeginGroup();

    if (ImGui::CollapsingHeader("Test Configuration", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::DragInt("Test Repeats", &m_test_repeats, 1.0f, 0, 100);
        static const char* benchmark_mode_names[] = {
            "Layout Comparison",
            "Custom Matrix Layouts",
            "Legacy Tests",
        };

        int benchmark_mode_current_index = static_cast<int>(m_benchmark_mode);
        if (ImGui::BeginCombo("Benchmark Mode", benchmark_mode_names[benchmark_mode_current_index]))
        {
            for (int i = 0; i < IM_ARRAYSIZE(benchmark_mode_names); ++i)
            {
                const bool is_selected = benchmark_mode_current_index == i;
                if (ImGui::Selectable(benchmark_mode_names[i], is_selected))
                {
                    m_benchmark_mode = static_cast<BenchmarkMode>(i);
                }

                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        static const char* test_case_names[] = {
            "MxM Basic",
            "MxM Vector To Matrix",
            "CONV",
        };

        int test_type_current_index = static_cast<int>(m_test_type);

        ImGui::Text("Note: Not all tests are compatible with all devices!");
        ImGui::Text("Check shader instruction set for compatibility if testing other than MXM_BASIC");

        if (ImGui::BeginCombo("Test Case", test_case_names[test_type_current_index]))
        {
            // NOTE: Temporarily disabled other tests, new test template coming on the next patch
            for (int i = 0; i < static_cast<int>(TestType::TT_COUNT); ++i)
            {
                const bool is_selected = (test_type_current_index == i);
                if (ImGui::Selectable(test_case_names[i], is_selected))
                {
                    m_test_type = static_cast<TestType>(i);
                }

                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        ImGui::BeginDisabled(m_test_type != TestType::TT_CONV);
        ImGui::DragInt("Conv Width", &m_input_width, 1.0f, 1, 256);
        ImGui::DragInt("Conv Height", &m_input_height, 1.0f, 1, 256);
        ImGui::Checkbox("Normalize Inputs", &m_normalize_inputs);
        ImGui::EndDisabled();

        ImGui::Separator();
        ImGui::Checkbox("Legacy K-first comparison", &m_legacy_layouts);
        ImGui::Checkbox("Show Peak Percentage", &m_show_peak_percentage);
        ImGui::BeginDisabled(!m_show_peak_percentage);
        ImGui::DragFloat("Peak Frequency MHz", &m_peak_frequency_mhz, 1.0f, 1.0f, 3000.0f, "%.0f");
        ImGui::InputFloat("FP32 peak TOPS at frequency", &m_peak_fp32);
        ImGui::InputFloat("FP16 peak TOPS at frequency", &m_peak_fp16);
        ImGui::InputFloat("INT8 peak TOPS at frequency", &m_peak_int8);
        ImGui::EndDisabled();
    }

    if (ImGui::CollapsingHeader("Matrix Configuration", ImGuiTreeNodeFlags_None))
    {
        static const char* fill_type_labels[] = {
                "Fill with Zero",
                "Fill with Constants",
                "Fill with Random UInt",
                "Fill with Random Int",
                "Fill Sequence Int",
                "Fill with Random Low/High Int",
                "Fill with Random Float",
                "Fill with Random +/-1 Float"
        };

        int fill_data_current_index = static_cast<int>(m_fill_data_type);

        if (ImGui::Combo("Fill Data Type", &fill_data_current_index, fill_type_labels, IM_ARRAYSIZE(fill_type_labels)))
        {
            m_fill_data_type = static_cast<FillDataType>(fill_data_current_index);
        }

        ImGui::Separator();

        if (m_benchmark_mode == BenchmarkMode::CUSTOM_LAYOUTS)
        {
            static const char* a_layout_labels[] = { "TiledK-first", "M-first", "K-first" };
            static const MatrixLayout a_layout_values[] = { MatrixLayout::TILED_K_FIRST, MatrixLayout::M_FIRST, MatrixLayout::K_FIRST };
            static const char* b_layout_labels[] = { "TiledK-first", "N-first", "K-first" };
            static const MatrixLayout b_layout_values[] = { MatrixLayout::TILED_K_FIRST, MatrixLayout::N_FIRST, MatrixLayout::K_FIRST };
            static const char* r_layout_labels[] = { "N-first", "M-first" };
            static const MatrixLayout r_layout_values[] = { MatrixLayout::N_FIRST, MatrixLayout::M_FIRST };

            auto layoutCombo = [](const char* label, MatrixLayout& value, const char* const* labels, const MatrixLayout* values, int count)
            {
                int current_index = 0;
                for (int i = 0; i < count; ++i)
                {
                    if (values[i] == value)
                    {
                        current_index = i;
                        break;
                    }
                }

                if (ImGui::Combo(label, &current_index, labels, count))
                {
                    value = values[current_index];
                }
            };

            layoutCombo("A Layout", m_custom_layout_a, a_layout_labels, a_layout_values, IM_ARRAYSIZE(a_layout_values));
            layoutCombo("B Layout", m_custom_layout_b, b_layout_labels, b_layout_values, IM_ARRAYSIZE(b_layout_values));
            layoutCombo("Output Layout", m_custom_layout_r, r_layout_labels, r_layout_values, IM_ARRAYSIZE(r_layout_values));
        }
        else if (m_benchmark_mode == BenchmarkMode::LEGACY_TESTS)
        {
            static const char* option_labels[] = { "True", "False", "Variable" };
            static const char* matrix_labels[] = { "A", "B", "C", "R"};

            for (std::size_t i = 0; i < NUM_MATS; ++i)
            {
                int current_index = static_cast<int>(m_matrix_transpose_options[i]);

                char label[32];
                std::snprintf(label, sizeof(label), "Transpose Matrix %s", matrix_labels[i]);

                if (ImGui::Combo(label, &current_index, option_labels, IM_ARRAYSIZE(option_labels)))
                {
                    m_matrix_transpose_options[i] = static_cast<MatrixTransposeOption>(current_index);
                }
            }
        }
        else
        {
            ImGui::TextDisabled("The default table uses fixed layout rows.");
        }

        ImGui::Checkbox("Validate Result", &m_validate_matrix_result);
    }

    ImGui::Separator();

    if (ImGui::Button("Run Tests"))
    {
        PrepareTestSession();
    }

    if (m_is_processing_tests)
    {
        ImGui::SameLine();
        ImGui::EndDisabled();
        ImGui::Text("Processing Test [%d] of [%d]", m_total_processed_tests, m_total_tests);
        ImGui::SameLine();
        ImGui::ProgressBar(static_cast<float>(m_total_processed_tests) / static_cast<float>(std::max(0u, m_total_tests)));
        ImGui::BeginDisabled(disable_ui);
    }

    if (!m_test_groups.empty())
    {
        for (int i=0; i< m_test_groups.size(); i++)
        {
            const auto& test_group = m_test_groups[i];

            // Quick table exit if none of its entries are valid/supported
            if (!test_group.test_entries.empty() && !test_group.test_entries.back().test_results.empty())
            {
                bool is_any_result_valid = false;
                for (const auto& test_result : test_group.test_entries.back().test_results)
                {
                    is_any_result_valid |= test_result.is_valid;
                }

                if (!is_any_result_valid)
                {
                    continue;
                }
            }

            std::string collapsing_header_title = std::string("Test #").append(std::to_string(i)) +
                std::string(" - ") + GetMatrixComponentTypeName(test_group.template_description.input_type) +
                std::string(" input / ") +
                GetMatrixComponentTypeName(test_group.template_description.output_type) +
                std::string(" output");

            if (ImGui::CollapsingHeader(collapsing_header_title.c_str()))
            {
                ImGuiStyle& style                   = ImGui::GetStyle();
                const float original_scrollbar_size = style.ScrollbarSize;
                style.ScrollbarSize                 = 40.0f;

                ImGui::BeginChild("##test_results");
                if (ImGui::BeginTable("TestResultTable", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable))
                {
                    ImGui::TableSetupColumn("A Layout", ImGuiTableColumnFlags_WidthFixed, 100.0f);
                    ImGui::TableSetupColumn("B Layout", ImGuiTableColumnFlags_WidthFixed, 100.0f);
                    ImGui::TableSetupColumn("Output Layout", ImGuiTableColumnFlags_WidthFixed, 120.0f);

                    for (const auto& size_configuration : test_group.template_description.size_configurations)
                    {
                        ImGui::TableSetupColumn(("NTile=" + std::to_string(size_configuration.NSize)).c_str());
                    }

                    ImGui::TableHeadersRow();

                    // Each test entry will be a table row
                    for (int test_entry_index = 0; test_entry_index < test_group.test_entries.size(); test_entry_index++)
                    {
                        const auto& test_entry   = test_group.test_entries[test_entry_index];
                        int current_column_index = 0;

                        ImGui::TableNextRow();

                        // Transpose flags
                        ImGui::TableSetColumnIndex(current_column_index++);
                        ImGui::Text("%s", GetLayoutName(test_entry.layoutA));

                        ImGui::TableSetColumnIndex(current_column_index++);
                        ImGui::Text("%s", GetLayoutName(test_entry.layoutB));

                        ImGui::TableSetColumnIndex(current_column_index++);
                        ImGui::Text("%s", GetLayoutName(test_entry.layoutR));

                        // For each of the NSize configs
                        for (int test_result_index = 0; test_result_index < test_entry.test_results.size(); test_result_index++)
                        {
                            ImGui::TableSetColumnIndex(current_column_index++);

                            const auto& test_description = test_entry.test_descriptions[test_result_index];
                            const auto& test_result      = test_entry.test_results[test_result_index];

                            if (test_result.is_valid)
                            {
                                auto GetPercentageColor = [](float value) -> ImVec4
                                {
                                    value = std::clamp(value, 0.0f, 1.0f);

                                    if (value < 0.5f)
                                    {
                                        float t = value / 0.5f;
                                        return ImVec4(1.0f, t, 0.0f, 1.0f);
                                    }
                                    else
                                    {
                                        float t = (value - 0.5f) / 0.5f;
                                        return ImVec4(1.0f - t, 1.0f, 0.0f, 1.0f);
                                    }
                                };

                                ImGui::Text("[Time]: %.2fus", test_result.time_total);
                                ImGui::Text("[TOPS]: %.2f", test_result.TOPS);

                                if (test_result.validation_pass.has_value())
                                {
                                    const bool vpass = *test_result.validation_pass;
                                    ImGui::PushStyleColor(ImGuiCol_Text, vpass ? ImVec4(0.2f, 1.0f, 0.2f, 1.0f)
                                                                                : ImVec4(1.0f, 0.2f, 0.2f, 1.0f));
                                    ImGui::Text(vpass ? "[VAL]: PASS" : "[VAL]: FAIL");
                                    ImGui::PopStyleColor();
                                }

                                if (m_test_type == TT_CONV)
                                    ImGui::TextDisabled("WxH = %dx%d", test_description.inputWidth, test_description.inputHeight);

                                if (m_show_peak_percentage && test_result.percentage > 0.0)
                                {
                                    ImVec4 color = GetPercentageColor(test_result.percentage / 100.0f);
                                    ImGui::PushStyleColor(ImGuiCol_Text, color);
                                    ImGui::Text("[%% @%.0fMHz]: %.2f", m_peak_frequency_mhz, test_result.percentage);
                                    ImGui::PopStyleColor();
                                }
                                else if (m_show_peak_percentage)
                                {
                                    ImGui::TextDisabled("[%% @%.0fMHz]: N/A", m_peak_frequency_mhz);
                                }
                            }
                            else
                            {
                                ImGui::Text("N/A - Not Supported");
                            }
                        }
                    }

                    ImGui::EndTable();
                }
                ImGui::EndChild();

                style.ScrollbarSize = original_scrollbar_size;
            }
        }
    }

    ImGui::EndGroup();
    ImGui::EndDisabled();
}

void CooperativeMatrixRunner::PrepareTestSession()
{
    m_vulkan_instance.WaitUntilIdle();

    m_test_groups.clear();
    m_total_tests           = 0;
    m_total_processed_tests = 0;

    struct LayoutCombination
    {
        MatrixLayout layoutA;
        MatrixLayout layoutB;
        MatrixLayout layoutR;
        bool layoutC_Mfirst = false;
    };

    auto GenerateLayoutCombinations = [&]() -> std::vector<LayoutCombination>
    {
        if (m_test_type == TT_CONV)
            return { { MatrixLayout::K_FIRST, MatrixLayout::K_FIRST, MatrixLayout::N_FIRST, false } };

        if (m_benchmark_mode == BenchmarkMode::LAYOUT_COMPARISON)
        {
            const auto kLayout = m_legacy_layouts ? MatrixLayout::K_FIRST : MatrixLayout::TILED_K_FIRST;
            return {
                { kLayout, kLayout, MatrixLayout::N_FIRST, false },
                { MatrixLayout::M_FIRST,       kLayout, MatrixLayout::N_FIRST, false },
                { kLayout, MatrixLayout::N_FIRST,       MatrixLayout::N_FIRST, false },
                { MatrixLayout::M_FIRST,       MatrixLayout::N_FIRST,       MatrixLayout::N_FIRST, false },
                { kLayout, kLayout, MatrixLayout::M_FIRST, true },
                { MatrixLayout::M_FIRST,       kLayout, MatrixLayout::M_FIRST, true },
                { kLayout, MatrixLayout::N_FIRST,       MatrixLayout::M_FIRST, true },
                { MatrixLayout::M_FIRST,       MatrixLayout::N_FIRST,       MatrixLayout::M_FIRST, true },
            };
        }

        if (m_benchmark_mode == BenchmarkMode::CUSTOM_LAYOUTS)
        {
            return { { m_custom_layout_a, m_custom_layout_b, m_custom_layout_r, m_custom_layout_r == MatrixLayout::M_FIRST } };
        }

        std::vector<LayoutCombination> combinations;

        std::vector<std::size_t> variable_indices;
        std::vector<bool> fixed_values(NUM_MATS);

        for (std::size_t i = 0; i < NUM_MATS; ++i)
        {
            switch (m_matrix_transpose_options[i])
            {
                case MatrixTransposeOption::ALWAYS_TRUE:
                    fixed_values[i] = true;
                    break;
                case MatrixTransposeOption::ALWAYS_FALSE:
                    fixed_values[i] = false;
                    break;
                case MatrixTransposeOption::VARIABLE:
                    variable_indices.push_back(i);
                    break;
            }
        }

        std::size_t num_combinations = 1ULL << variable_indices.size();
        combinations.reserve(num_combinations);

        for (std::size_t combo = 0; combo < num_combinations; ++combo)
        {
            std::vector<bool> current(NUM_MATS);

            for (std::size_t i = 0; i < NUM_MATS; ++i)
            {
                current[i] = fixed_values[i];
            }

            for (std::size_t bit = 0; bit < variable_indices.size(); ++bit)
            {
                std::size_t index = variable_indices[bit];
                current[index] = (combo >> bit) & 1;
            }

            combinations.push_back(LayoutCombination{
                current[MAT_A] ? MatrixLayout::M_FIRST : MatrixLayout::K_FIRST,
                current[MAT_B] ? MatrixLayout::K_FIRST : MatrixLayout::N_FIRST,
                current[MAT_R] ? MatrixLayout::M_FIRST : MatrixLayout::N_FIRST,
                current[MAT_C],
            });
        }

        return combinations;
    };

    const auto layout_combinations = GenerateLayoutCombinations();

    for (const auto& test_template_description : m_test_group_templates)
    {
        TestGroup new_test_group;
        new_test_group.template_description = test_template_description;

        TestDescription new_test_description;

        new_test_description.fill_data_type = m_fill_data_type;
        new_test_description.test_type      = m_test_type;

        new_test_description.inputWidth  = m_input_width;
        new_test_description.inputHeight = m_input_height;

        new_test_description.input_type  = test_template_description.input_type;
        new_test_description.output_type = test_template_description.output_type;

        new_test_description.perf_loop = static_cast<uint32_t>(std::clamp(m_test_repeats, 1, 1024));

        for (auto& layoutCombination : layout_combinations)
        {
            TestGroup::TestRowEntry test_entry;

            new_test_description.layoutA = layoutCombination.layoutA;
            new_test_description.layoutB = layoutCombination.layoutB;
            new_test_description.layoutR = layoutCombination.layoutR;
            new_test_description.layoutC_Mfirst = layoutCombination.layoutC_Mfirst;

            test_entry.layoutA = new_test_description.layoutA;
            test_entry.layoutB = new_test_description.layoutB;
            test_entry.layoutR = new_test_description.layoutR;
            test_entry.layoutC_Mfirst = new_test_description.layoutC_Mfirst;

            for (auto& size_configuration : test_template_description.size_configurations)
            {
                new_test_description.MSizeInBlocks = size_configuration.MSizeInBlocks;
                new_test_description.NSizeInBlocks = size_configuration.NSizeInBlocks;
                new_test_description.KSizeInBlocks = size_configuration.KSizeInBlocks;
                new_test_description.MSize         = size_configuration.MSize;
                new_test_description.NSize         = size_configuration.NSize;
                new_test_description.KSize         = size_configuration.KSize;

                test_entry.test_descriptions.push_back(new_test_description);
                m_total_tests++;
            }

            new_test_group.test_entries.push_back(test_entry);
        }

        m_test_groups.push_back(new_test_group);
    }

    m_is_processing_tests = true;
}

std::optional<CooperativeMatrixRunner::TestResult> CooperativeMatrixRunner::RunTest(const TestDescription& test_description)
{
    TestResult test_result = {};
    test_result.is_valid = true;

    VkResult result;

    int MSize = test_description.MSize;
    int NSize = test_description.NSize;
    int KSize = test_description.KSize;
    int MSizeInBlocks = test_description.MSizeInBlocks;
    int NSizeInBlocks = test_description.NSizeInBlocks;
    int KSizeInBlocks = test_description.KSizeInBlocks;

    uint32_t perf_loop = test_description.perf_loop;

    bool layoutA_Mfirst = test_description.layoutA == MatrixLayout::M_FIRST;
    bool layoutA_TiledKfirst = test_description.layoutA == MatrixLayout::TILED_K_FIRST;
    bool layoutB_Nfirst = test_description.layoutB == MatrixLayout::N_FIRST;
    bool layoutB_Kfirst = test_description.layoutB == MatrixLayout::K_FIRST;
    bool layoutB_TiledKfirst = test_description.layoutB == MatrixLayout::TILED_K_FIRST;
    bool layoutC_Mfirst = test_description.layoutC_Mfirst;
    bool layoutR_Mfirst = test_description.layoutR == MatrixLayout::M_FIRST;

    int inputWidth  = test_description.inputWidth;
    int inputHeight = test_description.inputHeight;

    uint32_t tt = static_cast<uint32_t>(test_description.test_type);
    int init    = test_description.fill_data_type;

    auto command_pool_queue_family_index = m_vulkan_instance.m_VulkanQueues[Vulkan::QueueIndex::eGraphicsQueue].QueueFamilyIndex;
    auto submission_queue                = m_vulkan_instance.m_VulkanQueues[Vulkan::QueueIndex::eGraphicsQueue].Queue;

    // Not optimal at all but we are drawing the UI and running the test in the same queue
    m_vulkan_instance.QueueWaitIdle(Vulkan::QueueIndex::eGraphicsQueue);

    const auto subgroup_size = m_vulkan_instance.GetExtension<ExtensionLib::Vulkan_SubgroupPropertiesHook>()->Properties.subgroupSize;
    const auto gpuvendor_id = static_cast<gpu_vendors>(m_vulkan_instance.GetGpuProperties().Base.properties.vendorID);
    const auto gputier_id   = static_cast<gpu_tiers>(m_vulkan_instance.GetGpuProperties().Base.properties.deviceID);

    const auto device_limits = m_vulkan_instance.GetGpuProperties().Base.properties.limits;

    // Query matrix properties and see if the test is supported for the given GPU
    bool valid_testtypes = false;
    VkCooperativeMatrixPropertiesKHR cooperativeMatrixProps = {};
    if (!FindMatrixProperty(m_hFoundCooperativeMatrices, cooperativeMatrixProps, MSize, NSize, KSize, test_description.input_type, test_description.input_type, test_description.output_type, test_description.output_type))
    {
        LOGI("Skipping test: tile/type combination is not supported.");
        return std::nullopt;
    }

    const auto* conversion = m_vulkan_instance.GetExtension<QcomCoopMatConversionExtension>();
    if (tt != TT_MXM_BASIC && (!conversion || !conversion->RequestedFeatures.cooperativeMatrixConversion
        || subgroup_size != 64 || cooperativeMatrixProps.MSize != subgroup_size))
    {
        LOGI("Skipping conversion test: requires QCOM conversion and one lane per tile row.");
        return std::nullopt;
    }
    if (perf_loop == 0 || MSizeInBlocks <= 0 || NSizeInBlocks <= 0 || KSizeInBlocks <= 0)
        return std::nullopt;
    if (tt == TT_CONV && (layoutA_Mfirst || layoutA_TiledKfirst || !layoutB_Kfirst || layoutR_Mfirst))
        return std::nullopt;

    if (m_normalize_inputs)
    {
        int required_area = MSizeInBlocks * cooperativeMatrixProps.MSize;

        // Start with inputWidth as-is, compute height to match required_area
        if (inputWidth <= 0) inputWidth = 1; // safety
        inputHeight = required_area / inputWidth;

        // If division leaves remainder, just force height to match
        if (inputWidth * inputHeight != required_area)
        {
            inputHeight = required_area / inputWidth;
            if (inputWidth * inputHeight != required_area)
            {
                // Last resort: set width = required_area, height = 1
                inputWidth = required_area;
                inputHeight = 1;
            }
        }
    }

    // Set local_size (workgroup size) based on GPU/Tier, and datatype (fp32, fp16, etc)
    // Default for 'unknown' or gpu/tier not recohgnized is local_size(64,2,2) for all datatyes
    uint32_t local_size_x = 0, local_size_y = 0, local_size_z = 0;

    switch (gpuvendor_id)
    {
        case VK_VENDOR_ID_NVIDIA:
            local_size_x = subgroup_size;
            local_size_y = 1;
            local_size_z = 1;
            break;
        case VK_VENDOR_ID_AMD:
            local_size_x = subgroup_size;
            local_size_y = 1;
            local_size_z = 1;
            break;
        case VK_VENDOR_ID_INTEL:
            local_size_x = subgroup_size;
            local_size_y = 1;
            local_size_z = 1;
            break;
        case VK_VENDOR_ID_APPLE:
            local_size_x = subgroup_size;
            local_size_y = 1;
            local_size_z = 1;
            break;
        case VK_VENDOR_ID_QUALCOMM:
            local_size_x = subgroup_size;
            local_size_y = 2;
            local_size_z = 2;
            break;
        default: // unknown, including gpu option not part of the map
            printf("\nUnknown GPU");
            local_size_x = subgroup_size;
            local_size_y = 2;
            local_size_z = 2;
            break;
    }

    const auto* subgroupControl = m_vulkan_instance.GetExtension<ExtensionLib::Ext_VK_EXT_subgroup_size_control>();
    if (!subgroupControl || !subgroupControl->RequestedFeatures.subgroupSizeControl
        || !(subgroupControl->Properties.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT)
        || subgroup_size < subgroupControl->Properties.minSubgroupSize || subgroup_size > subgroupControl->Properties.maxSubgroupSize
        || local_size_x > device_limits.maxComputeWorkGroupSize[0]
        || local_size_y > device_limits.maxComputeWorkGroupSize[1] || local_size_z > device_limits.maxComputeWorkGroupSize[2]
        || local_size_x * local_size_y * local_size_z > device_limits.maxComputeWorkGroupInvocations
        || (MSizeInBlocks + local_size_y - 1) / local_size_y > device_limits.maxComputeWorkGroupCount[1]
        || (NSizeInBlocks + local_size_z - 1) / local_size_z > device_limits.maxComputeWorkGroupCount[2])
    {
        LOGI("Skipping test: subgroup or dispatch requirements exceed device support.");
        return std::nullopt;
    }

    if (tt == TT_CONV && (inputWidth * inputHeight != MSizeInBlocks * cooperativeMatrixProps.MSize))
    {
        LOGE("Convolution ConvInputWidth * ConvInputHeight (%d) must equal MSizeInBlocks * MSize (%d) for current datatype",
            (inputWidth * inputHeight), (MSizeInBlocks * cooperativeMatrixProps.MSize));
        return std::nullopt;
    }

    RuntimeShader runtime_shader;

    // Set compiler options
    std::vector<const char*> compiler_options;
    int bytesPerInput;
    int bytesPerOutput;

    if (test_description.input_type == VK_COMPONENT_TYPE_FLOAT32_KHR)
    {
        runtime_shader.AddDefine("A_TYPE", std::string("float"));
        runtime_shader.AddDefine("R_TYPE", std::string("float"));
        runtime_shader.AddDefine("NUM_PACK", std::string("1"));
        bytesPerInput = 4;
        bytesPerOutput = 4;
    }
    else
    if (test_description.input_type == VK_COMPONENT_TYPE_FLOAT16_KHR)
    {
        runtime_shader.AddDefine("A_TYPE", std::string("float16_t"));
        runtime_shader.AddDefine("R_TYPE", std::string("float16_t"));
        runtime_shader.AddDefine("NUM_PACK", std::string("2"));
        bytesPerInput = 2;
        bytesPerOutput = 2;
    }
    else
    if (test_description.input_type == VK_COMPONENT_TYPE_UINT8_KHR)
    {
        runtime_shader.AddDefine("A_TYPE", std::string("uint8_t"));
        runtime_shader.AddDefine("R_TYPE", std::string("uint32_t"));
        runtime_shader.AddDefine("NUM_PACK", std::string("4"));
        bytesPerInput = 1;
        bytesPerOutput = 4;
    }
    else
    if (test_description.input_type == VK_COMPONENT_TYPE_SINT8_KHR)
    {
        runtime_shader.AddDefine("A_TYPE", std::string("int8_t"));
        runtime_shader.AddDefine("R_TYPE", std::string("int32_t"));
        runtime_shader.AddDefine("NUM_PACK", std::string("4"));
        bytesPerInput = 1;
        bytesPerOutput = 4;
    }
    else
    {
        return std::nullopt;
    }

    // Use packed words for INT8 conversion and contiguous FP16 input.
    // Unsupported tile packing and M-first FP16 input retain typed loads.
    if (tt == TT_MXM_VecToMat)
        runtime_shader.AddDefine("PACKED_A", std::string((bytesPerInput == 1 || (!layoutA_Mfirst && bytesPerInput == 2))
            && cooperativeMatrixProps.KSize % (4 / bytesPerInput) == 0 ? "1" : "0"));

    if (!runtime_shader.Build(ShaderPaths[tt], m_vulkan_instance.m_VulkanDevice, "main", glslang_stage_t::GLSLANG_STAGE_COMPUTE))
    {
        LOGE("Failed to compile test shader");
        return std::nullopt;
    }

    VkShaderModule shaderModule = runtime_shader.GetShaderModule();

    // Create descriptor set and descriptor set layout for our A,B,C,R matrices (buffers)
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSet descriptorSet;
    VkDescriptorPool descriptorPool;

    auto create_buffers_desc_set = [](VkDevice device, VkDescriptorSetLayout & descriptorSetLayout, VkDescriptorSet & descriptorSet, VkDescriptorPool &descriptorPool, const uint32_t num_buffers)
    {
        VkResult result;

        VkDescriptorPoolSize* poolSizes = new VkDescriptorPoolSize[num_buffers];
        for (uint32_t i = 0; i < num_buffers; i++)
            poolSizes[i] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.pNext = NULL;
        descriptorPoolCreateInfo.maxSets = 1;
        descriptorPoolCreateInfo.poolSizeCount = num_buffers;
        descriptorPoolCreateInfo.pPoolSizes = poolSizes;

        result = vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &descriptorPool);
        CHECK_VK(result);

        VkDescriptorSetLayoutBinding* layoutBindings = new VkDescriptorSetLayoutBinding[num_buffers];
        for (uint32_t i = 0; i < num_buffers; i++)
        {
            layoutBindings[i].binding = i;
            layoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            layoutBindings[i].descriptorCount = 1;
            layoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            layoutBindings[i].pImmutableSamplers = nullptr;
        }

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.pNext = nullptr;
        descriptorSetLayoutCreateInfo.flags = 0;
        descriptorSetLayoutCreateInfo.bindingCount = num_buffers;
        descriptorSetLayoutCreateInfo.pBindings = layoutBindings;

        result = vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout);
        CHECK_VK(result);

        VkDescriptorSetAllocateInfo setAllocateInfo = {};
        setAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        setAllocateInfo.pNext = nullptr;
        setAllocateInfo.descriptorPool = descriptorPool;
        setAllocateInfo.descriptorSetCount = 1; // Use only 1 set for all descriptors
        setAllocateInfo.pSetLayouts = &descriptorSetLayout;

        result = vkAllocateDescriptorSets(device, &setAllocateInfo, &descriptorSet);
        CHECK_VK(result);

        delete[] poolSizes;
        delete[] layoutBindings;
    };

    create_buffers_desc_set(m_vulkan_instance.m_VulkanDevice, descriptorSetLayout, descriptorSet, descriptorPool, NUM_MATS);

    // Create command pool
    VkCommandPoolCreateInfo commandPoolCreateInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, (uint32_t)command_pool_queue_family_index };
    VkCommandPool commandPool;
    result = vkCreateCommandPool(m_vulkan_instance.m_VulkanDevice, &commandPoolCreateInfo, NULL, &commandPool);
    CHECK_VK(result);

    // Create command buffer
    //
    // The command buffers, one for initializing buffers, one for compute, one
    // for reading back the results. This lets us time the compute work more
    // precisely.
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 3 };
    VkCommandBuffer commandBuffers[3];
    result = vkAllocateCommandBuffers(m_vulkan_instance.m_VulkanDevice, &commandBufferAllocateInfo, commandBuffers);
    CHECK_VK(result);

    // Creat Pipeline layout
    // Use only 1 set for all descriptors
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &descriptorSetLayout, 0, nullptr };
    VkPipelineLayout pipelineLayout;
    result = vkCreatePipelineLayout(m_vulkan_instance.m_VulkanDevice, &pipelineLayoutCreateInfo, NULL, &pipelineLayout);
    CHECK_VK(result);

    int filterWidth  = 3;
    int filterHeight = 3;
    int dilation = 1;
    int stride   = 1;

    TestCase testCase = {};

    testCase.testType   = (TestType)tt;
    testCase.inputType  = cooperativeMatrixProps.AType;
    testCase.outputType = cooperativeMatrixProps.ResultType;

    // MxNxK is the size of the full matrix multiply
    testCase.TOTAL_M = cooperativeMatrixProps.MSize * MSizeInBlocks;
    testCase.TOTAL_N = cooperativeMatrixProps.NSize * NSizeInBlocks;
    testCase.TOTAL_K = cooperativeMatrixProps.KSize * KSizeInBlocks;

    int mA_paddedM = testCase.TOTAL_M;
    int mA_paddedK = testCase.TOTAL_K;
    int mB_paddedN = testCase.TOTAL_N;
    int mB_paddedK = testCase.TOTAL_K;
    int mC_paddedM = testCase.TOTAL_M;
    int mC_paddedN = testCase.TOTAL_N;
    int mR_paddedM = testCase.TOTAL_M;
    int mR_paddedN = testCase.TOTAL_N;

    if (!layoutA_TiledKfirst)
    {
        if (layoutA_Mfirst) mA_paddedM += (mA_paddedM % (128 / bytesPerInput))  ? 0 : 64 / bytesPerInput;  else  mA_paddedK += (mA_paddedK % (128 / bytesPerInput)) ? 0 : 64 / bytesPerInput;
    }
    if (!layoutB_TiledKfirst)
    {
        if (layoutB_Kfirst) mB_paddedK += (mB_paddedK % (128 / bytesPerInput))  ? 0 : 64 / bytesPerInput;  else  mB_paddedN += (mB_paddedN % (128 / bytesPerInput)) ? 0 : 64 / bytesPerInput;
    }
    if (layoutC_Mfirst) mC_paddedM += (mC_paddedM % (128 / bytesPerOutput)) ? 0 : 64 / bytesPerOutput; else  mC_paddedN += (mC_paddedN % (128 / bytesPerOutput)) ? 0 : 64 / bytesPerOutput;
    if (layoutR_Mfirst) mR_paddedM += (mR_paddedM % (128 / bytesPerOutput)) ? 0 : 64 / bytesPerOutput; else  mR_paddedN += (mR_paddedN % (128 / bytesPerOutput)) ? 0 : 64 / bytesPerOutput;

    // Each cooperative matrix multiply is R[TILE_M, TILE_N] = A[TILE_M, TILE_K] x B[TILE_K, TILE_N] + C[TILE_M, TILE_N]
    testCase.TILE_M = cooperativeMatrixProps.MSize;
    testCase.TILE_N = cooperativeMatrixProps.NSize;
    testCase.TILE_K = cooperativeMatrixProps.KSize;

    testCase.layoutA_Mfirst = (uint32_t)layoutA_Mfirst;
    testCase.layoutB_Kfirst = (uint32_t)layoutB_Kfirst;
    testCase.layoutB_Nfirst = (uint32_t)layoutB_Nfirst;
    testCase.layoutA_TiledKfirst = (uint32_t)layoutA_TiledKfirst;
    testCase.layoutB_TiledKfirst = (uint32_t)layoutB_TiledKfirst;
    testCase.layoutC_Mfirst = (uint32_t)layoutC_Mfirst;
    testCase.layoutR_Mfirst = (uint32_t)layoutR_Mfirst;

    testCase.strideAinElements = layoutA_TiledKfirst ? testCase.TILE_K : (layoutA_Mfirst ? mA_paddedM : mA_paddedK);
    testCase.strideBinElements = layoutB_TiledKfirst ? testCase.TILE_K : (layoutB_Kfirst ? mB_paddedK : mB_paddedN);
    testCase.strideCinElements = (layoutC_Mfirst ? mC_paddedM : mC_paddedN);
    testCase.strideRinElements = (layoutR_Mfirst ? mR_paddedM : mR_paddedN);

    // Specialize the shader with the matrix sizes, strides, and constants.
    // Also, work-group sizes
    const uint32_t specDataMxMBasic[] = {
        local_size_x,
        local_size_y,
        local_size_z,
        testCase.TOTAL_M,
        testCase.TOTAL_N,
        testCase.TOTAL_K,
        testCase.TILE_M,
        testCase.TILE_N,
        testCase.TILE_K,
        testCase.layoutA_Mfirst,
        testCase.layoutB_Nfirst,
        testCase.layoutA_TiledKfirst,
        testCase.layoutB_TiledKfirst,
        testCase.layoutC_Mfirst,
        testCase.layoutR_Mfirst,
        testCase.strideAinElements,
        testCase.strideBinElements,
        testCase.strideCinElements,
        testCase.strideRinElements
    };

    const uint32_t specDataCONV[] = {   // pass to shader_name.comp
        local_size_x,               // layout(constant_id = 0) const uint local_size_x;
        local_size_y,               // layout(constant_id = 1) const uint local_size_y;
        local_size_z,               // layout(constant_id = 2) const uint local_size_z;
        testCase.TOTAL_M,           // layout(constant_id = 3) const uint TOTAL_M = 1;
        testCase.TOTAL_N,           // layout(constant_id = 4) const uint TOTAL_N = 1;
        testCase.TOTAL_K,           // layout(constant_id = 5) const uint TOTAL_K = 1;
        testCase.TILE_M,            // layout(constant_id = 6) const uint TILE_M = 1;
        testCase.TILE_N,            // layout(constant_id = 7) const uint TILE_N = 1;
        testCase.TILE_K,            // layout(constant_id = 8) const uint TILE_K = 1;
        (uint32_t)inputWidth,       // layout(constant_id = 9) const uint INPUT_W = 1;
        (uint32_t)inputHeight,      // layout(constant_id =10) const uint INPUT_H = 1;
        (uint32_t)filterWidth,      // layout(constant_id =11) const uint FILTER_W = 1;
        (uint32_t)filterHeight,     // layout(constant_id =12) const uint FILTER_H = 1;
        (uint32_t)dilation,         // layout(constant_id =13) const uint DILATION = 1;
        (uint32_t)stride,           // layout(constant_id =14) const uint STRIDE  = 1;
        testCase.strideAinElements, // layout(constant_id =15) const uint strideAinElements = 1;
        testCase.strideBinElements, // layout(constant_id =16) const uint strideBinElements = 1;
        testCase.strideCinElements, // layout(constant_id =17) const uint strideCinElements = 1;
        testCase.strideRinElements  // layout(constant_id =18) const uint strideRinElements = 1;
    };

    auto fill_specialized_map_entries = [](VkSpecializationMapEntry entries[], uint32_t num_entries, uint32_t sizeof_entry)
    {
        for (uint32_t i = 0; i < num_entries; i++)
            entries[i] = { i, sizeof_entry * i, sizeof_entry };
    };

#define ARRAY_LENGTH(x) (sizeof(x) / sizeof(x[0]))

    VkSpecializationMapEntry entriesMxMBasic[ARRAY_LENGTH(specDataMxMBasic)];
    fill_specialized_map_entries(entriesMxMBasic, ARRAY_LENGTH(specDataMxMBasic), sizeof(uint32_t));

    VkSpecializationMapEntry entriesCONV[ARRAY_LENGTH(specDataCONV)];
    fill_specialized_map_entries(entriesCONV, ARRAY_LENGTH(specDataCONV), sizeof(uint32_t)); // {0, sizeof(uint32_t) * 0, sizeof(uint32_t)}, ...,//{end,  sizeof(uint32_t) * end, sizeof(uint32_t)}

    VkSpecializationInfo specInfo;
    switch (tt)
    {
    case TT_CONV:
        specInfo = { ARRAY_LENGTH(specDataCONV), entriesCONV, sizeof(specDataCONV), specDataCONV, };
        break;
    case TT_MXM_BASIC:
        specInfo = { ARRAY_LENGTH(specDataMxMBasic), entriesMxMBasic, sizeof(specDataMxMBasic), specDataMxMBasic, };
        break;
    case TT_MXM_VecToMat:
        specInfo = { ARRAY_LENGTH(specDataMxMBasic), entriesMxMBasic, sizeof(specDataMxMBasic), specDataMxMBasic, };
        break;
    default:
        LOGE("Unknown use case(%d), can't sent specialized constantas to shader!", tt);
    }

#undef ARRAY_LENGTH

    // Create pipeline with a desired subgroup size (e.g., AMD supports two subgroup sizes)
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSizeInfo = {};
    subgroupSizeInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroupSizeInfo.requiredSubgroupSize = subgroup_size; // Must be between min and max

    // SPIR-V 1.6 does not require REQUIRE_FULL_SUBGROUPS. X still equals the pinned subgroup size.
    VkPipelineShaderStageCreateInfo shaderCreateInfo   = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, &subgroupSizeInfo, 0, VK_SHADER_STAGE_COMPUTE_BIT, shaderModule, "main", &specInfo};
    VkComputePipelineCreateInfo     pipelineCreateInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, shaderCreateInfo, pipelineLayout, VK_NULL_HANDLE, 0 };

    // Create the query pool
    VkQueryPool query_pool_timestamps = VK_NULL_HANDLE;       // A query pool is required to use GPU time stamps
    std::vector<uint64_t> time_stamps((size_t)perf_loop*2, 0);// We will get timestamps for the beginning and end of each of the compute passes
                                                              // GPU time stamps will be stored in a vector
    // VK_QUERY_TYPE_TIMESTAMP: We need to specify the query type for this pool, which in our case is for time stamps
    // time_stamps: Set the no. of queries in this pool
    VkQueryPoolCreateInfo query_pool_info = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0, VK_QUERY_TYPE_TIMESTAMP, static_cast<uint32_t>(time_stamps.size()), 0 };
    result = vkCreateQueryPool(m_vulkan_instance.m_VulkanDevice, &query_pool_info, nullptr, &query_pool_timestamps);
    CHECK_VK(result);

    std::cout << "\nExecuting vkCreateComputePipelines(...) (takes a while!)\n";
    VkPipeline pipeline = VK_NULL_HANDLE;
    result = vkCreateComputePipelines(m_vulkan_instance.m_VulkanDevice, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &pipeline);
    CHECK_VK(result);

    if (result != VK_SUCCESS)
    {
        LOGE("Skipping test: compute pipeline creation failed (%d).", int(result));
        vkDestroyQueryPool(m_vulkan_instance.m_VulkanDevice, query_pool_timestamps, nullptr);
        vkDestroyCommandPool(m_vulkan_instance.m_VulkanDevice, commandPool, nullptr);
        vkDestroyDescriptorPool(m_vulkan_instance.m_VulkanDevice, descriptorPool, nullptr);
        vkDestroyPipelineLayout(m_vulkan_instance.m_VulkanDevice, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(m_vulkan_instance.m_VulkanDevice, descriptorSetLayout, nullptr);
        vkDestroyShaderModule(m_vulkan_instance.m_VulkanDevice, shaderModule, nullptr);
        return std::nullopt;
    }

    auto FindProperties = [](const VkPhysicalDeviceMemoryProperties* pMemoryProperties,
        uint32_t memoryTypeBitsRequirement, VkMemoryPropertyFlags requiredProperties) -> int32_t
    {
        const uint32_t memoryCount = pMemoryProperties->memoryTypeCount;
        for (uint32_t memoryIndex = 0; memoryIndex < memoryCount; ++memoryIndex) {
            const uint32_t memoryTypeBits = (1 << memoryIndex);
            const bool isRequiredMemoryType = memoryTypeBitsRequirement & memoryTypeBits;

            const VkMemoryPropertyFlags properties =
                pMemoryProperties->memoryTypes[memoryIndex].propertyFlags;
            const bool hasRequiredProperties =
                (properties & requiredProperties) == requiredProperties;

            if (isRequiredMemoryType && hasRequiredProperties)
                return static_cast<int32_t>(memoryIndex);
        }

        // failed to find memory type
        return -1;
    };

    auto CreateMatrixDesc = [&](
        VkDevice device, 
        VkPhysicalDeviceMemoryProperties& memory_properties,
        MatrixDesc& m, 
        VkComponentTypeKHR dt, 
        int rows, 
        int cols)
    {
        VkResult result;

        m.dims.rows = rows;
        m.dims.cols = cols;
        m.dataType = dt;
        m.elementSize = ComponentTypeInfo[m.dataType].bits / 8;
        m.totalElements = m.dims.cols * m.dims.rows;
        m.bufferSize = m.totalElements * m.elementSize;

        VkBufferCreateInfo bufferCreateInfo = {
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            NULL,
            0,
            m.bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT,
            VK_SHARING_MODE_EXCLUSIVE,
            0u,
            NULL,
        };

        result = vkCreateBuffer(device, &bufferCreateInfo, NULL, &m.hostBuffer);
        CHECK_VK(result);
        result = vkCreateBuffer(device, &bufferCreateInfo, NULL, &m.deviceBuffer);
        CHECK_VK(result);

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, m.hostBuffer, &memReqs);

        int32_t hostIndex = FindProperties(&memory_properties, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        int32_t deviceIndex = FindProperties(&memory_properties, memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        VkMemoryAllocateFlagsInfo memAllocateFlagsInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL,VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
        VkMemoryAllocateInfo memAllocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &memAllocateFlagsInfo, memReqs.size, (uint32_t)hostIndex};

        result = vkAllocateMemory(device, &memAllocateInfo, NULL, &m.hostMemory);
        CHECK_VK(result);

        memAllocateInfo.memoryTypeIndex = deviceIndex;
        result = vkAllocateMemory(device, &memAllocateInfo, NULL, &m.deviceMemory);
        CHECK_VK(result);

        result = vkBindBufferMemory(device, m.hostBuffer, m.hostMemory, 0);
        CHECK_VK(result);

        result = vkBindBufferMemory(device, m.deviceBuffer, m.deviceMemory, 0);
        CHECK_VK(result);

        result = vkMapMemory(device, m.hostMemory, 0, m.bufferSize, 0, &m.ptr);
        CHECK_VK(result);
    };

    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(m_vulkan_instance.m_VulkanGpu, &memory_properties);

    MatrixDesc matrices[NUM_MATS];

    CreateMatrixDesc(m_vulkan_instance.m_VulkanDevice, memory_properties, matrices[MAT_A], cooperativeMatrixProps.AType, mA_paddedM, mA_paddedK);
    if (tt == TT_CONV) CreateMatrixDesc(m_vulkan_instance.m_VulkanDevice, memory_properties, matrices[MAT_B], cooperativeMatrixProps.AType, filterHeight*filterWidth*mB_paddedN, mB_paddedK);
    else               CreateMatrixDesc(m_vulkan_instance.m_VulkanDevice, memory_properties, matrices[MAT_B], cooperativeMatrixProps.AType, mB_paddedK, mB_paddedN);
    CreateMatrixDesc(m_vulkan_instance.m_VulkanDevice, memory_properties, matrices[MAT_C], cooperativeMatrixProps.CType, mC_paddedM, mC_paddedN);
    CreateMatrixDesc(m_vulkan_instance.m_VulkanDevice, memory_properties, matrices[MAT_R], cooperativeMatrixProps.ResultType, mR_paddedM, mR_paddedN);

    auto update_buffer_descriptor_set = [](VkDevice device, MatrixDesc * matrices, uint32_t num_matrices, VkDescriptorSet & descriptorSet)
    {
        VkDescriptorBufferInfo* bufferDescriptor = new VkDescriptorBufferInfo[num_matrices];

        for (uint32_t i = 0; i < num_matrices; i++)
        {
            bufferDescriptor[i].buffer = matrices[i].deviceBuffer;
            bufferDescriptor[i].offset = 0;
            bufferDescriptor[i].range = matrices[i].bufferSize;
        }

        VkWriteDescriptorSet* writeDescriptorset = new VkWriteDescriptorSet[num_matrices];

        for (uint32_t i = 0; i < num_matrices; i++)
        {
            writeDescriptorset[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorset[i].pNext = nullptr;
            writeDescriptorset[i].dstSet = descriptorSet;
            writeDescriptorset[i].dstBinding = i;
            writeDescriptorset[i].dstArrayElement = 0;
            writeDescriptorset[i].descriptorCount = 1;
            writeDescriptorset[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writeDescriptorset[i].pImageInfo = nullptr;
            writeDescriptorset[i].pBufferInfo = &bufferDescriptor[i];
            writeDescriptorset[i].pTexelBufferView = nullptr;
        }

        vkUpdateDescriptorSets(device, num_matrices, writeDescriptorset, 0, NULL);

        delete[] bufferDescriptor;
        delete[] writeDescriptorset;
    };

    update_buffer_descriptor_set(m_vulkan_instance.m_VulkanDevice, matrices, NUM_MATS, descriptorSet);

    float*    matrixR_CPU_fp32   = new float[matrices[MAT_R].dims.rows * matrices[MAT_R].dims.cols]();
    FLOAT16*  matrixR_CPU_fp16   = new FLOAT16[matrices[MAT_R].dims.rows * matrices[MAT_R].dims.cols]();
    int32_t*  matrixR_CPU_sint32 = new int32_t[matrices[MAT_R].dims.rows * matrices[MAT_R].dims.cols]();
    uint32_t* matrixR_CPU_uint32 = new uint32_t[matrices[MAT_R].dims.rows * matrices[MAT_R].dims.cols]();

    // ToDo: Think in how to use templates!
    if ((tt == TT_CONV) && (test_description.input_type == VK_COMPONENT_TYPE_FLOAT32_KHR)) // CONV test case, input/output data Type Float 32?
    {
        InitMatrix((float*)matrices[MAT_A].ptr, testCase.TOTAL_M, testCase.TOTAL_K, matrices[MAT_A].dims.cols, (FillDataType)init, 2);
        InitMatrix((float*)matrices[MAT_B].ptr, filterHeight*filterWidth*testCase.TOTAL_N, testCase.TOTAL_K, matrices[MAT_B].dims.cols, (FillDataType)init, 2);
        InitMatrix((float*)matrices[MAT_C].ptr, testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_C].dims.cols, FILL_WITH_ZERO, 2);
        InitMatrix((float*)matrices[MAT_R].ptr, testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_R].dims.cols, FILL_WITH_ZERO, 2);
    }
    else if ((tt == TT_CONV) && (test_description.input_type == VK_COMPONENT_TYPE_FLOAT16_KHR)) // CONV test case, input/output data Type Float 16?
    {
        InitMatrix((FLOAT16*)matrices[MAT_A].ptr, testCase.TOTAL_M, testCase.TOTAL_K, matrices[MAT_A].dims.cols, (FillDataType)init, 2);
        InitMatrix((FLOAT16*)matrices[MAT_B].ptr, filterHeight*filterWidth*testCase.TOTAL_N, testCase.TOTAL_K, matrices[MAT_B].dims.cols, (FillDataType)init, 2);
        InitMatrix((FLOAT16*)matrices[MAT_C].ptr, testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_C].dims.cols, FILL_WITH_ZERO, 2);
        InitMatrix((FLOAT16*)matrices[MAT_R].ptr, testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_R].dims.cols, FILL_WITH_ZERO, 2);
    }
    else if ((tt == TT_CONV) && (test_description.input_type == VK_COMPONENT_TYPE_SINT8_KHR)) // CONV test case, Input data Type signed int8, output data type signed int 32?
    {
        InitMatrix((int8_t*)matrices[MAT_A].ptr, testCase.TOTAL_M, testCase.TOTAL_K, matrices[MAT_A].dims.cols, (FillDataType)init, 2);
        InitMatrix((int8_t*)matrices[MAT_B].ptr, filterHeight*filterWidth*testCase.TOTAL_N, testCase.TOTAL_K, matrices[MAT_B].dims.cols, (FillDataType)init, 2);
        InitMatrix((int32_t*)matrices[MAT_C].ptr,testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_C].dims.cols, FILL_WITH_ZERO, 2);
        InitMatrix((int32_t*)matrices[MAT_R].ptr,testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_R].dims.cols, FILL_WITH_ZERO, 2);
    }
    else if ((tt == TT_CONV) && (test_description.input_type == VK_COMPONENT_TYPE_UINT8_KHR)) // CONV test case, Input data Type signed int8, output data type signed int 32?
    {
        InitMatrix((uint8_t*)matrices[MAT_A].ptr, testCase.TOTAL_M, testCase.TOTAL_K, matrices[MAT_A].dims.cols, (FillDataType)init, 2);
        InitMatrix((uint8_t*)matrices[MAT_B].ptr, filterHeight*filterWidth*testCase.TOTAL_N, testCase.TOTAL_K, matrices[MAT_B].dims.cols, (FillDataType)init, 2);
        InitMatrix((uint32_t*)matrices[MAT_C].ptr,testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_C].dims.cols, FILL_WITH_ZERO, 2);
        InitMatrix((uint32_t*)matrices[MAT_R].ptr,testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_R].dims.cols, FILL_WITH_ZERO, 2);
    }
    else if (test_description.input_type == VK_COMPONENT_TYPE_FLOAT32_KHR) // Input/output data Type Float 32?
    {
        InitMatrix((float*)matrices[MAT_A].ptr, testCase.TOTAL_M, testCase.TOTAL_K, matrices[MAT_A].dims.cols, (FillDataType)init, 2);
        InitMatrix((float*)matrices[MAT_B].ptr, testCase.TOTAL_K, testCase.TOTAL_N, matrices[MAT_B].dims.cols, (FillDataType)init, 2);
        InitMatrix((float*)matrices[MAT_C].ptr, testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_C].dims.cols, FILL_WITH_ZERO, 2);  
        InitMatrix((float*)matrices[MAT_R].ptr, testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_R].dims.cols, FILL_WITH_ZERO, 2);

    }
    else
    if (test_description.input_type == VK_COMPONENT_TYPE_FLOAT16_KHR) // Input/output data Type Float 16?
    {
        InitMatrix((FLOAT16*)matrices[MAT_A].ptr, testCase.TOTAL_M, testCase.TOTAL_K, matrices[MAT_A].dims.cols, (FillDataType)init, 2);
        InitMatrix((FLOAT16*)matrices[MAT_B].ptr, testCase.TOTAL_K, testCase.TOTAL_N, matrices[MAT_B].dims.cols, (FillDataType)init, 2);
        InitMatrix((FLOAT16*)matrices[MAT_C].ptr, testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_C].dims.cols, FILL_WITH_ZERO, 2);
        InitMatrix((FLOAT16*)matrices[MAT_R].ptr, testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_R].dims.cols, FILL_WITH_ZERO, 2);

    }
    else
    if (test_description.input_type == VK_COMPONENT_TYPE_SINT8_KHR) // Input data Type signed int8, output data type signed int 32?
    {
        InitMatrix((int8_t*)matrices[MAT_A].ptr, testCase.TOTAL_M, testCase.TOTAL_K, matrices[MAT_A].dims.cols, (FillDataType)init, 2);
        InitMatrix((int8_t*)matrices[MAT_B].ptr, testCase.TOTAL_K, testCase.TOTAL_N, matrices[MAT_B].dims.cols, (FillDataType)init, 2);
        InitMatrix((int32_t*)matrices[MAT_C].ptr,testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_C].dims.cols, FILL_WITH_ZERO, 2);
        InitMatrix((int32_t*)matrices[MAT_R].ptr,testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_R].dims.cols, FILL_WITH_ZERO, 2);

    }
    else
    if (test_description.input_type == VK_COMPONENT_TYPE_UINT8_KHR) // Data Type input unsigned int 8, data type output unsigned int 32?
    {
        InitMatrix((uint8_t*)matrices[MAT_A].ptr, testCase.TOTAL_M, testCase.TOTAL_K, matrices[MAT_A].dims.cols, (FillDataType)init, 2);
        InitMatrix((uint8_t*)matrices[MAT_B].ptr, testCase.TOTAL_K, testCase.TOTAL_N, matrices[MAT_B].dims.cols, (FillDataType)init, 2);
        InitMatrix((uint32_t*)matrices[MAT_C].ptr,testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_C].dims.cols, FILL_WITH_ZERO, 2);
        InitMatrix((uint32_t*)matrices[MAT_R].ptr,testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_R].dims.cols, FILL_WITH_ZERO, 2);

    }
    else
    {
        return std::nullopt;
    }

    // Save original (pre-transform) A and B for validation before layout conversion.
    std::vector<uint8_t> savedA, savedB;
    if (m_validate_matrix_result)
    {
        savedA.assign((const uint8_t*)matrices[MAT_A].ptr, (const uint8_t*)matrices[MAT_A].ptr + matrices[MAT_A].bufferSize);
        savedB.assign((const uint8_t*)matrices[MAT_B].ptr, (const uint8_t*)matrices[MAT_B].ptr + matrices[MAT_B].bufferSize);
    }

    if (tt != TT_CONV)
    {
        auto applyTransform = [&](auto* ptrA, auto* ptrB)
        {
            using T = std::remove_pointer_t<decltype(ptrA)>;

            if (layoutA_TiledKfirst)
            {
                std::vector<T> tempA(testCase.TOTAL_M * testCase.TOTAL_K);
                TransformMatrixToTiledKfirst(ptrA, testCase.TOTAL_M, testCase.TOTAL_K, tempA.data(), testCase.TILE_K);
                std::memcpy(matrices[MAT_A].ptr, tempA.data(), tempA.size() * sizeof(T));
            }
            else if (layoutA_Mfirst)
            {
                std::vector<T> tempA((size_t)matrices[MAT_A].dims.rows * matrices[MAT_A].dims.cols, T{});
                for (uint32_t mm = 0; mm < testCase.TOTAL_M; ++mm)
                    for (uint32_t kk = 0; kk < testCase.TOTAL_K; ++kk)
                        tempA[kk * testCase.strideAinElements + mm] = ptrA[mm * matrices[MAT_A].dims.cols + kk];
                std::memcpy(matrices[MAT_A].ptr, tempA.data(), tempA.size() * sizeof(T));
            }

            if (layoutB_TiledKfirst)
            {
                std::vector<T> tempB(testCase.TOTAL_K * testCase.TOTAL_N);
                std::vector<T> tempBT(testCase.TOTAL_K * testCase.TOTAL_N);
                for (uint32_t kk = 0; kk < testCase.TOTAL_K; kk++)
                    for (uint32_t nn = 0; nn < testCase.TOTAL_N; nn++)
                        tempBT[nn * testCase.TOTAL_K + kk] = ptrB[kk * matrices[MAT_B].dims.cols + nn];
                TransformMatrixToTiledKfirst(tempBT.data(), testCase.TOTAL_N, testCase.TOTAL_K, tempB.data(), testCase.TILE_K);
                std::memcpy(matrices[MAT_B].ptr, tempB.data(), tempB.size() * sizeof(T));
            }
            else if (layoutB_Kfirst)
            {
                std::vector<T> tempB((size_t)matrices[MAT_B].dims.rows * matrices[MAT_B].dims.cols, T{});
                for (uint32_t kk = 0; kk < testCase.TOTAL_K; ++kk)
                    for (uint32_t nn = 0; nn < testCase.TOTAL_N; ++nn)
                        tempB[nn * testCase.strideBinElements + kk] = ptrB[kk * matrices[MAT_B].dims.cols + nn];
                std::memcpy(matrices[MAT_B].ptr, tempB.data(), tempB.size() * sizeof(T));
            }
        };

        if      (test_description.input_type == VK_COMPONENT_TYPE_FLOAT32_KHR) applyTransform((float*)   matrices[MAT_A].ptr, (float*)   matrices[MAT_B].ptr);
        else if (test_description.input_type == VK_COMPONENT_TYPE_FLOAT16_KHR) applyTransform((FLOAT16*) matrices[MAT_A].ptr, (FLOAT16*) matrices[MAT_B].ptr);
        else if (test_description.input_type == VK_COMPONENT_TYPE_SINT8_KHR)   applyTransform((int8_t*)  matrices[MAT_A].ptr, (int8_t*)  matrices[MAT_B].ptr);
        else if (test_description.input_type == VK_COMPONENT_TYPE_UINT8_KHR)   applyTransform((uint8_t*) matrices[MAT_A].ptr, (uint8_t*) matrices[MAT_B].ptr);
    }

    VkCommandBufferBeginInfo commandBufferBeginInfo{};
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    // Download input buffers to device memory.
    result = vkBeginCommandBuffer(commandBuffers[0], &commandBufferBeginInfo); // Begin command buffer recording
    CHECK_VK(result);

    for (uint32_t i = 0; i < NUM_MATS; ++i) {
        MatrixDesc &m = matrices[i];
        VkBufferCopy copy = { 0, 0, m.bufferSize };
        vkCmdCopyBuffer(commandBuffers[0], m.hostBuffer, m.deviceBuffer, 1, &copy);
    }

    VkMemoryBarrier uploadBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
    vkCmdPipelineBarrier(commandBuffers[0], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &uploadBarrier, 0, nullptr, 0, nullptr);
    result = vkEndCommandBuffer(commandBuffers[0]); // End command buffer recording
    CHECK_VK(result);

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL,1, &commandBuffers[0], 0,  NULL};

    submitInfo.pCommandBuffers = &commandBuffers[0];
    result = vkQueueSubmit(submission_queue, 1, &submitInfo, VK_NULL_HANDLE);
    CHECK_VK(result);
    result = vkQueueWaitIdle(submission_queue);
    CHECK_VK(result);

    uint32_t groupCountX = 1;
    uint32_t groupCountY = (testCase.TOTAL_M / testCase.TILE_M + (local_size_y - 1)) / local_size_y;
    uint32_t groupCountZ = (testCase.TOTAL_N / testCase.TILE_N + (local_size_z - 1)) / local_size_z;

    result = vkBeginCommandBuffer(commandBuffers[1], &commandBufferBeginInfo); // Begin command buffer recording
    CHECK_VK(result);

    vkCmdBindPipeline(commandBuffers[1], VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffers[1], VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0u, 1, &descriptorSet, 0u, NULL);

	// Reset the timestamp query pool, so we can start fetching new values into it
    vkCmdResetQueryPool(commandBuffers[1], query_pool_timestamps, 0, static_cast<uint32_t>(time_stamps.size()));

    perf_loop = time_stamps.size()/2; // Both should have the same value, but just in case...

    VkMemoryBarrier computeBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT };
    vkCmdDispatch(commandBuffers[1], groupCountX, groupCountY, groupCountZ); // untimed warmup
    for (size_t loop = 0; loop < perf_loop; loop++)
    {
        vkCmdPipelineBarrier(commandBuffers[1], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &computeBarrier, 0, nullptr, 0, nullptr);
        // Start after the preceding compute dependency, including the warmup.
        vkCmdWriteTimestamp( commandBuffers[1], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool_timestamps, loop*2 );
        vkCmdDispatch(       commandBuffers[1], groupCountX, groupCountY, groupCountZ);                                // Dispacth work
        vkCmdWriteTimestamp( commandBuffers[1], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_timestamps,loop*2+1); // Stop timer...
    }

    VkMemoryBarrier downloadBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT };
    vkCmdPipelineBarrier(commandBuffers[1], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &downloadBarrier, 0, nullptr, 0, nullptr);

    result = vkEndCommandBuffer(commandBuffers[1]); // End command buffer recording
    CHECK_VK(result);

    submitInfo.pCommandBuffers = &commandBuffers[1];
    result = vkQueueSubmit(submission_queue, 1, &submitInfo, VK_NULL_HANDLE); // Here is the actual work!
    CHECK_VK(result);
    result = vkQueueWaitIdle(submission_queue);
    CHECK_VK(result);

    result = vkGetQueryPoolResults(m_vulkan_instance.m_VulkanDevice, query_pool_timestamps, 0,	time_stamps.size(), time_stamps.size() * sizeof(uint64_t), time_stamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    CHECK_VK(result);

    double ms = 0.0, min_ms = DBL_MAX, delta_in_ms = 0.0;
    for (size_t loop = 0; loop < perf_loop; loop++)
    {
        delta_in_ms = double(time_stamps[loop*2+1] - time_stamps[loop*2]) * double(device_limits.timestampPeriod) / 1000000.0;
        min_ms = (delta_in_ms < min_ms ? delta_in_ms : min_ms);
        ms += delta_in_ms;
    }

    if(gpuvendor_id == VK_VENDOR_ID_QUALCOMM )
    {
        uint64_t total_ops = 0;
        if (tt == TT_CONV)
        {
            total_ops = static_cast<uint64_t>(testCase.TOTAL_M) *
                static_cast<uint64_t>(testCase.TOTAL_N) *
                static_cast<uint64_t>(testCase.TOTAL_K) *
                static_cast<uint64_t>(filterHeight) *
                static_cast<uint64_t>(filterWidth) * 2;
        }
        else
        {
            total_ops = static_cast<uint64_t>(testCase.TOTAL_M) *
                static_cast<uint64_t>(testCase.TOTAL_N) *
                static_cast<uint64_t>(testCase.TOTAL_K) * 2;
        }

        ms /= double(perf_loop);

        test_result.time_total = ms * 1000;
        test_result.TOPS       = static_cast<double>(total_ops) / (ms / 1000.0) / 1e12;
        test_result.percentage = 0.0;

        if (m_show_peak_percentage && tt == TT_MXM_BASIC && m_peak_frequency_mhz > 0.0f)
        {
            const double peak_tops = test_description.input_type == VK_COMPONENT_TYPE_FLOAT32_KHR ? m_peak_fp32
                : test_description.input_type == VK_COMPONENT_TYPE_FLOAT16_KHR ? m_peak_fp16 : m_peak_int8;
            if (peak_tops > 0.0)
                test_result.percentage = test_result.TOPS / peak_tops * 100.0;
        }
    }
    else
    {
        ms /= double(perf_loop);
        std::cout << "MxM kernel time, average of " << perf_loop << " run(s): " << ms * 1000 << "us\n";
        std::cout << "MxM kernel time, min of     " << perf_loop << " run(s): " << min_ms * 1000 << "us\n";

        test_result.time_total = ms * 1000;
        test_result.TOPS       = 0.0;
        test_result.percentage = 0.0;
    }

    // Upload the result from device memory.
    result = vkBeginCommandBuffer(commandBuffers[2], &commandBufferBeginInfo); // Begin command buffer recording
    CHECK_VK(result);
    {
        MatrixDesc &m = matrices[MAT_R];
        VkBufferCopy copy = { 0, 0, m.bufferSize };
        vkCmdCopyBuffer(commandBuffers[2], m.deviceBuffer, m.hostBuffer, 1, &copy);
    }
    VkMemoryBarrier readbackBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT };
    vkCmdPipelineBarrier(commandBuffers[2], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &readbackBarrier, 0, nullptr, 0, nullptr);
    result = vkEndCommandBuffer(commandBuffers[2]); // End command buffer recording
    CHECK_VK(result);

    submitInfo.pCommandBuffers = &commandBuffers[2];
    result = vkQueueSubmit(submission_queue, 1, &submitInfo, VK_NULL_HANDLE);
    CHECK_VK(result);
    result = vkQueueWaitIdle(submission_queue);
    CHECK_VK(result);

    // Compare GPU output with the CPU reference for the selected kernel.
    if (m_validate_matrix_result && !savedA.empty())
    {
        const uint32_t M  = testCase.TOTAL_M;
        const uint32_t N  = testCase.TOTAL_N;
        const uint32_t K  = testCase.TOTAL_K;
        const uint32_t RC = M * N;

        auto accumulateReference = [&](auto* A, auto* B, auto* output)
        {
            using Acc = std::remove_pointer_t<decltype(output)>;
            for (uint32_t m = 0; m < M; ++m)
                for (uint32_t k = 0; k < K; ++k)
                    for (int fy = 0; fy < (tt == TT_CONV ? filterHeight : 1); ++fy)
                        for (int fx = 0; fx < (tt == TT_CONV ? filterWidth : 1); ++fx)
                        {
                            int pixel = int(m);
                            if (tt == TT_CONV)
                            {
                                const int y = int(m / inputWidth) * stride + dilation * (fy - filterHeight / 2);
                                const int x = int(m % inputWidth) * stride + dilation * (fx - filterWidth / 2);
                                if (y < 0 || y >= inputHeight || x < 0 || x >= inputWidth) continue;
                                pixel = y * inputWidth + x;
                            }
                            const Acc a = Acc(A[pixel * matrices[MAT_A].dims.cols + k]);
                            for (uint32_t n = 0; n < N; ++n)
                            {
                                const size_t bIndex = tt == TT_CONV
                                    ? ((n * filterHeight + fy) * filterWidth + fx) * matrices[MAT_B].dims.cols + k
                                    : k * matrices[MAT_B].dims.cols + n;
                                output[m * N + n] += a * Acc(B[bIndex]);
                            }
                        }
        };

        // Compute reference result into matrixR_CPU_fp32.
        // Loop order (m, k outer; n inner) is cache-friendly for row-major A and B.
        auto cpuRef = [&](auto* A, auto* B)
        {
            std::fill(matrixR_CPU_fp32, matrixR_CPU_fp32 + RC, 0.0f);
            accumulateReference(A, B, matrixR_CPU_fp32);
        };

        auto compare = [&](auto* gpu, float tol, const char* label) -> bool
        {
            // Use 2D indexing to account for row-stride padding and layoutR_Mfirst.
            // layoutR_Mfirst=false (N-first/row-major): R[m][n] = gpu[m * strideR + n]
            // layoutR_Mfirst=true  (M-first/col-major): R[m][n] = gpu[n * strideR + m]
            const uint32_t strideR = testCase.strideRinElements;
            float maxErr = 0.0f;
            uint32_t maxErrM = 0, maxErrN = 0;
            for (uint32_t mi = 0; mi < M; ++mi)
            {
                for (uint32_t ni = 0; ni < N; ++ni)
                {
                    const uint32_t gpu_idx = layoutR_Mfirst
                        ? (ni * strideR + mi)
                        : (mi * strideR + ni);
                    const float gpu_val = (float)gpu[gpu_idx];
                    const float cpu_val = matrixR_CPU_fp32[mi * N + ni];
                    if (!std::isfinite(gpu_val) || !std::isfinite(cpu_val)) return false;
                    const float diff    = fabsf(gpu_val - cpu_val);
                    const float ref     = fabsf(cpu_val);
                    const float relErr  = ref > 1.0f ? diff / ref : diff;
                    if (relErr > maxErr) { maxErr = relErr; maxErrM = mi; maxErrN = ni; }
                }
            }
            const bool pass = maxErr <= tol;
            LOGI("Validation [%s]: %s  max_err=%.5f  tol=%.5f  at[%u,%u]\n",
                 label, pass ? "PASS" : "FAIL", maxErr, tol, maxErrM, maxErrN);
            return pass;
        };

        // Exact integer CPU reference: accumulate in int64 to avoid float32 rounding
        // for large K (e.g. INT8 with K=4096 can reach ~66M, past float32 exact range).
        // GPU computes exact int32 — so we compare directly with zero tolerance.
        std::vector<int64_t> matrixR_CPU_i64;
        auto cpuRefInt = [&](auto* A, auto* B)
        {
            matrixR_CPU_i64.assign(RC, 0);
            accumulateReference(A, B, matrixR_CPU_i64.data());
        };

        // Exact comparison for integer types: GPU result must match CPU i64 exactly.
        auto compareExact = [&](auto* gpu, const char* label) -> bool
        {
            const uint32_t strideR = testCase.strideRinElements;
            int64_t maxErr = 0;
            uint32_t maxErrM = 0, maxErrN = 0;
            for (uint32_t mi = 0; mi < M; ++mi)
            {
                for (uint32_t ni = 0; ni < N; ++ni)
                {
                    const uint32_t gpu_idx = layoutR_Mfirst
                        ? (ni * strideR + mi)
                        : (mi * strideR + ni);
                    const int64_t diff = llabs((int64_t)gpu[gpu_idx] - matrixR_CPU_i64[mi * N + ni]);
                    if (diff > maxErr) { maxErr = diff; maxErrM = mi; maxErrN = ni; }
                }
            }
            const bool pass = (maxErr == 0);
            LOGI("Validation [%s]: %s  max_err=%lld  at[%u,%u]\n",
                 label, pass ? "PASS" : "FAIL", (long long)maxErr, maxErrM, maxErrN);
            return pass;
        };

        if (test_description.input_type == VK_COMPONENT_TYPE_FLOAT32_KHR)
        {
            cpuRef((float*)savedA.data(), (float*)savedB.data());
            test_result.validation_pass = compare((float*)matrices[MAT_R].ptr, 1e-3f, "FP32");
        }
        else if (test_description.input_type == VK_COMPONENT_TYPE_FLOAT16_KHR)
        {
            cpuRef((FLOAT16*)savedA.data(), (FLOAT16*)savedB.data());
            test_result.validation_pass = compare((FLOAT16*)matrices[MAT_R].ptr, 5e-2f, "FP16");
        }
        else if (test_description.input_type == VK_COMPONENT_TYPE_SINT8_KHR)
        {
            cpuRefInt((int8_t*)savedA.data(), (int8_t*)savedB.data());
            test_result.validation_pass = compareExact((int32_t*)matrices[MAT_R].ptr, "SINT8");
        }
        else if (test_description.input_type == VK_COMPONENT_TYPE_UINT8_KHR)
        {
            cpuRefInt((uint8_t*)savedA.data(), (uint8_t*)savedB.data());
            test_result.validation_pass = compareExact((uint32_t*)matrices[MAT_R].ptr, "UINT8");
        }
    }

    auto destroyMatrixDesc = [](VkDevice device, MatrixDesc & m)
    {
        vkDestroyBuffer(device, m.hostBuffer, NULL);
        vkDestroyBuffer(device, m.deviceBuffer, NULL);
        vkFreeMemory(device, m.hostMemory, NULL);
        vkFreeMemory(device, m.deviceMemory, NULL);
    };

    // Free the memory/buffers/pipeline for this iteration.
    for (int i = 0; i < NUM_MATS; ++i) 
    {
        destroyMatrixDesc(m_vulkan_instance.m_VulkanDevice, matrices[i]);
    }

    vkDestroyQueryPool(m_vulkan_instance.m_VulkanDevice, query_pool_timestamps, nullptr);
    vkDestroyCommandPool(m_vulkan_instance.m_VulkanDevice, commandPool, nullptr);
    vkDestroyDescriptorPool(m_vulkan_instance.m_VulkanDevice, descriptorPool, nullptr);
    vkDestroyPipelineLayout(m_vulkan_instance.m_VulkanDevice, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_vulkan_instance.m_VulkanDevice, descriptorSetLayout, nullptr);
    vkDestroyPipeline(m_vulkan_instance.m_VulkanDevice, pipeline, NULL);

    vkDestroyShaderModule(m_vulkan_instance.m_VulkanDevice, shaderModule, NULL);

    delete[] matrixR_CPU_fp32;
    delete[] matrixR_CPU_fp16;
    delete[] matrixR_CPU_sint32;
    delete[] matrixR_CPU_uint32;

    LOGI("COOP_RESULT test=%u type=%s A=%s B=%s C=%s M=%u N=%u K=%u tile=%ux%ux%u repeats=%u us=%.6f tops=%.6f validation=%s",
        tt, GetMatrixComponentTypeName(test_description.input_type), GetLayoutName(test_description.layoutA),
        GetLayoutName(test_description.layoutB), GetLayoutName(test_description.layoutR),
        testCase.TOTAL_M, testCase.TOTAL_N, testCase.TOTAL_K, testCase.TILE_M, testCase.TILE_N, testCase.TILE_K,
        perf_loop, test_result.time_total, test_result.TOPS,
        test_result.validation_pass.has_value() ? (*test_result.validation_pass ? "PASS" : "FAIL") : "OFF");
    return test_result;
}
