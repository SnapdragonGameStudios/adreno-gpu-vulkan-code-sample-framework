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
            if ((cooperativeMatrixProperties[matrixprop].ResultType == RType) &&
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
}

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
    }

    // select supported cooperative matrix types/sizes
    uint32_t nCoopMatrixPropCount = 0;
    CHECK_VK(cooperativeMatrixEXT->m_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
        m_vulkan_instance.m_VulkanGpu,
        &nCoopMatrixPropCount,
        NULL
    ));
    
    m_hFoundCooperativeMatrices.resize(nCoopMatrixPropCount);
    for (auto& matrixProp : m_hFoundCooperativeMatrices)
    {
        matrixProp.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        matrixProp.pNext = nullptr;
    }

    CHECK_VK(cooperativeMatrixEXT->m_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
            m_vulkan_instance.m_VulkanGpu,
            &nCoopMatrixPropCount,
            &m_hFoundCooperativeMatrices[0]
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
             64, 32, 16},

            {8,  24, 128,
             64, 16, 32}
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

        static const char* test_case_names[] = {
            "MxM Basic",
            "MxM Vector To Matrix",
            "CONV",
        };

        int test_type_current_index = static_cast<int>(m_test_type);
        bool changed = false;

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
                    changed     = true;
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

        // NOTE: Validation (and its transpose option) will be added in a future path
        ImGui::BeginDisabled();
        if (m_validate_matrix_result)
        {
            ImGui::BeginDisabled();
            static bool always_true = true;
            ImGui::Checkbox("Transpose When Needed", &always_true);
            ImGui::EndDisabled();
        }
        else
        {
            ImGui::Checkbox("Transpose When Needed", &m_transpose_when_needed);
        }

        ImGui::Checkbox("Validate Result", &m_validate_matrix_result);
        ImGui::EndDisabled();
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

            const bool show_matrix_d = false;

            if (ImGui::CollapsingHeader(collapsing_header_title.c_str()))
            {
                ImGuiStyle& style                   = ImGui::GetStyle();
                const float original_scrollbar_size = style.ScrollbarSize;
                style.ScrollbarSize                 = 40.0f;

                ImGui::BeginChild("##test_results");
                if (ImGui::BeginTable("TestResultTable", (NUM_MATS - (show_matrix_d ? 0 : 1)) + 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable))
                {
                    ImGui::TableSetupColumn("A", ImGuiTableColumnFlags_WidthFixed, 100.0f);
                    ImGui::TableSetupColumn("B", ImGuiTableColumnFlags_WidthFixed, 100.0f);
                    ImGui::TableSetupColumn("C", ImGuiTableColumnFlags_WidthFixed, 100.0f);

                    if (show_matrix_d)
                    {
                        ImGui::TableSetupColumn("D", ImGuiTableColumnFlags_WidthFixed, 100.0f);
                    }

                    for (const auto& size_configuration : test_group.template_description.size_configurations)
                    {
                        ImGui::TableSetupColumn(("NSize=" + std::to_string(size_configuration.NSize)).c_str());
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
                        ImGui::Text("%s", test_entry.layoutA_Mfirst ? "M-first" : "K-first");

                        ImGui::TableSetColumnIndex(current_column_index++);
                        ImGui::Text("%s", test_entry.layoutB_Nfirst ? "N-first" : "K-first");

                        ImGui::TableSetColumnIndex(current_column_index++);
                        ImGui::Text("%s", test_entry.layoutC_Mfirst ? "M-first" : "N-first");

                        if (show_matrix_d)
                        {
                            ImGui::TableSetColumnIndex(current_column_index++);
                            ImGui::Text("%s", test_entry.layoutR_Mfirst ? "M-first" : "N-first");
                        }

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

                                if (m_test_type == TT_CONV)
                                    ImGui::TextDisabled("WxH = %dx%d", test_description.inputWidth, test_description.inputHeight);

                                ImVec4 color = GetPercentageColor(test_result.percentage / 100.0f);
                                // ImGui::PushStyleColor(ImGuiCol_Text, color);
                                // ImGui::Text("[%%]: %.2f", test_result.percentage);
                                // ImGui::PopStyleColor();
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

    auto GenerateTransposeCombinations = [&]() -> std::vector<std::vector<bool>>
    {
        std::vector<std::vector<bool>> combinations;

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

            combinations.push_back(std::move(current));
        }

        return combinations;
    };

    const auto transpose_combinations = GenerateTransposeCombinations();

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

        new_test_description.perf_loop = static_cast<uint32_t>(m_test_repeats);

        for (auto& transposeCombination : transpose_combinations)
        {
            TestGroup::TestRowEntry test_entry;

            new_test_description.layoutA_Mfirst = transposeCombination[0];
            new_test_description.layoutB_Nfirst = transposeCombination[1];
            new_test_description.layoutC_Mfirst = transposeCombination[2];
            new_test_description.layoutR_Mfirst = transposeCombination[3];

            test_entry.layoutA_Mfirst = new_test_description.layoutA_Mfirst;
            test_entry.layoutB_Nfirst = new_test_description.layoutB_Nfirst;
            test_entry.layoutC_Mfirst = new_test_description.layoutC_Mfirst;
            test_entry.layoutR_Mfirst = new_test_description.layoutR_Mfirst;

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
    
    bool layoutA_Mfirst = test_description.layoutA_Mfirst;
    bool layoutB_Kfirst = !test_description.layoutB_Nfirst;
    bool layoutC_Mfirst = test_description.layoutC_Mfirst;
    bool layoutR_Mfirst = test_description.layoutR_Mfirst;

    int inputWidth  = test_description.inputWidth;
    int inputHeight = test_description.inputHeight;

    uint32_t tt = static_cast<uint32_t>(test_description.test_type);
    int init    = test_description.fill_data_type;

    auto command_pool_queue_family_index = m_vulkan_instance.m_VulkanQueues[Vulkan::QueueIndex::eGraphicsQueue].QueueFamilyIndex;
    auto submission_queue                = m_vulkan_instance.m_VulkanQueues[command_pool_queue_family_index].Queue;

    // Not optimal at all but we are drawing the UI and running the test in the same queue
    m_vulkan_instance.QueueWaitIdle(Vulkan::QueueIndex::eGraphicsQueue);

    const auto subgroup_size = m_vulkan_instance.GetExtension<ExtensionLib::Vulkan_SubgroupPropertiesHook>()->Properties.subgroupSize;
    const auto gpuvendor_id = static_cast<gpu_vendors>(m_vulkan_instance.GetGpuProperties().Base.properties.vendorID);
    const auto gputier_id   = static_cast<gpu_tiers>(m_vulkan_instance.GetGpuProperties().Base.properties.deviceID);
    
    const auto device_limits = m_vulkan_instance.GetGpuProperties().Base.properties.limits;

    // Create descriptor set and descriptor set layout for our A,B,C,R matrices (buffers)
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSet descriptorSet;

    auto create_buffers_desc_set = [](VkDevice device, VkDescriptorSetLayout & descriptorSetLayout, VkDescriptorSet & descriptorSet, const uint32_t num_buffers)
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

        VkDescriptorPool descriptorPool;
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
    
    create_buffers_desc_set(m_vulkan_instance.m_VulkanDevice, descriptorSetLayout, descriptorSet, NUM_MATS);

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

    // Query matrix properties and see if the test is supported for the given GPU
    bool valid_testtypes = false;
    VkCooperativeMatrixPropertiesKHR cooperativeMatrixProps = {};
    if (!FindMatrixProperty(m_hFoundCooperativeMatrices, cooperativeMatrixProps, MSize, NSize, KSize, test_description.input_type, test_description.input_type, test_description.output_type, test_description.output_type))
    {
        return std::nullopt;
    }

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
            local_size_x = 64;
            local_size_y = 2;
            local_size_z = 2;
            break;
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

    if (!runtime_shader.Build(ShaderPaths[tt], m_vulkan_instance.m_VulkanDevice, "main", glslang_stage_t::GLSLANG_STAGE_COMPUTE))
    {
        LOGE("Failed to compile test shader");
        return std::nullopt;
    }
    
    VkShaderModule shaderModule = runtime_shader.GetShaderModule();

    if (tt == TT_CONV && (inputWidth * inputHeight != MSizeInBlocks * cooperativeMatrixProps.MSize))
    {
        LOGE("Convolution ConvInputWidth * ConvInputHeight (%d) must equal MSizeInBlocks * MSize (%d) for current datatype",
            (inputWidth * inputHeight), (MSizeInBlocks * cooperativeMatrixProps.MSize));
        return std::nullopt;
    }

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

    if (layoutA_Mfirst) mA_paddedM += (mA_paddedM % (128 / bytesPerInput))  ? 0 : 64 / bytesPerInput;  else  mA_paddedK += (mA_paddedK % (128 / bytesPerInput)) ? 0 : 64 / bytesPerInput;
    if (layoutB_Kfirst) mB_paddedK += (mB_paddedK % (128 / bytesPerInput))  ? 0 : 64 / bytesPerInput;  else  mB_paddedN += (mB_paddedN % (128 / bytesPerInput)) ? 0 : 64 / bytesPerInput;
    if (layoutC_Mfirst) mC_paddedM += (mC_paddedM % (128 / bytesPerOutput)) ? 0 : 64 / bytesPerOutput; else  mC_paddedN += (mC_paddedN % (128 / bytesPerOutput)) ? 0 : 64 / bytesPerOutput;
    if (layoutR_Mfirst) mR_paddedM += (mR_paddedM % (128 / bytesPerOutput)) ? 0 : 64 / bytesPerOutput; else  mR_paddedN += (mR_paddedN % (128 / bytesPerOutput)) ? 0 : 64 / bytesPerOutput;

    // Each cooperative matrix multiply is R[TILE_M, TILE_N] = A[TILE_M, TILE_K] x B[TILE_K, TILE_N] + C[TILE_M, TILE_N]
    testCase.TILE_M = cooperativeMatrixProps.MSize;
    testCase.TILE_N = cooperativeMatrixProps.NSize;
    testCase.TILE_K = cooperativeMatrixProps.KSize;

    testCase.layoutA_Mfirst = (uint32_t)layoutA_Mfirst;
    testCase.layoutB_Kfirst = (uint32_t)layoutB_Kfirst;
    testCase.layoutC_Mfirst = (uint32_t)layoutC_Mfirst;
    testCase.layoutR_Mfirst = (uint32_t)layoutR_Mfirst;

    testCase.strideAinElements = (layoutA_Mfirst ? mA_paddedM : mA_paddedK);
    testCase.strideBinElements = (layoutB_Kfirst ? mB_paddedK : mB_paddedN);
    testCase.strideCinElements = (layoutC_Mfirst ? mC_paddedM : mC_paddedN);
    testCase.strideRinElements = (layoutR_Mfirst ? mR_paddedM : mR_paddedN);

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
    if (tt == TT_CONV) CreateMatrixDesc(m_vulkan_instance.m_VulkanDevice, memory_properties, matrices[MAT_B], cooperativeMatrixProps.AType, filterWidth*filterWidth*mB_paddedN, mB_paddedK);
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

        if (m_transpose_when_needed || m_validate_matrix_result)
        {
            if (layoutA_Mfirst) // Matrix A M-First?
                TransposeMatrix((float*)matrices[MAT_A].ptr, matrices[MAT_A].dims.rows, matrices[MAT_A].dims.cols, "layoutA_Mfirst");
            if (layoutB_Kfirst) // Matrix B K-First?
                TransposeMatrix((float*)matrices[MAT_B].ptr, matrices[MAT_B].dims.rows, matrices[MAT_B].dims.cols, "layoutB_Kfirst");
            if (layoutC_Mfirst) // Matrix C M-First?
                TransposeMatrix((float*)matrices[MAT_C].ptr, matrices[MAT_C].dims.rows, matrices[MAT_C].dims.cols, "layoutC_Mfirst");
        }
    }
    else
    if (test_description.input_type == VK_COMPONENT_TYPE_FLOAT16_KHR) // Input/output data Type Float 16?
    {
        InitMatrix((FLOAT16*)matrices[MAT_A].ptr, testCase.TOTAL_M, testCase.TOTAL_K, matrices[MAT_A].dims.cols, (FillDataType)init, 2);
        InitMatrix((FLOAT16*)matrices[MAT_B].ptr, testCase.TOTAL_K, testCase.TOTAL_N, matrices[MAT_B].dims.cols, (FillDataType)init, 2);
        InitMatrix((FLOAT16*)matrices[MAT_C].ptr, testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_C].dims.cols, FILL_WITH_ZERO, 2);
        InitMatrix((FLOAT16*)matrices[MAT_R].ptr, testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_R].dims.cols, FILL_WITH_ZERO, 2);

        if (m_transpose_when_needed || m_validate_matrix_result)
        {
            if (layoutA_Mfirst) // Matrix A M-First?
                TransposeMatrix((FLOAT16*)matrices[MAT_A].ptr, matrices[MAT_A].dims.rows, matrices[MAT_A].dims.cols, "layoutA_Mfirst");
            if (layoutB_Kfirst) // Matrix B K-First?
                TransposeMatrix((FLOAT16*)matrices[MAT_B].ptr, matrices[MAT_B].dims.rows, matrices[MAT_B].dims.cols, "layoutB_Kfirst");
            if (layoutC_Mfirst) // Matrix C M-First?
                TransposeMatrix((FLOAT16*)matrices[MAT_C].ptr, matrices[MAT_C].dims.rows, matrices[MAT_C].dims.cols, "layoutC_Mfirst");
        }
    }
    else
    if (test_description.input_type == VK_COMPONENT_TYPE_SINT8_KHR) // Input data Type signed int8, output data type signed int 32?
    {
        InitMatrix((int8_t*)matrices[MAT_A].ptr, testCase.TOTAL_M, testCase.TOTAL_K, matrices[MAT_A].dims.cols, (FillDataType)init, 2);
        InitMatrix((int8_t*)matrices[MAT_B].ptr, testCase.TOTAL_K, testCase.TOTAL_N, matrices[MAT_B].dims.cols, (FillDataType)init, 2);
        InitMatrix((int32_t*)matrices[MAT_C].ptr,testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_C].dims.cols, FILL_WITH_ZERO, 2);
        InitMatrix((int32_t*)matrices[MAT_R].ptr,testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_R].dims.cols, FILL_WITH_ZERO, 2);

        if (m_transpose_when_needed || m_validate_matrix_result)
        {
            if (layoutA_Mfirst) // Matrix A M-First?
                TransposeMatrix((int8_t*)matrices[MAT_A].ptr, matrices[MAT_A].dims.rows, matrices[MAT_A].dims.cols, "layoutA_Mfirst");
            if (layoutB_Kfirst) // Matrix B K-First?
                TransposeMatrix((int8_t*)matrices[MAT_B].ptr, matrices[MAT_B].dims.rows, matrices[MAT_B].dims.cols, "layoutB_Kfirst");
            if (layoutC_Mfirst) // Matrix C M-First?
                TransposeMatrix((int32_t*)matrices[MAT_C].ptr, matrices[MAT_C].dims.rows, matrices[MAT_C].dims.cols, "layoutC_Mfirst");
        }
    }
    else
    if (test_description.input_type == VK_COMPONENT_TYPE_UINT8_KHR) // Data Type input unsigned int 8, data type output unsigned int 32?
    {
        InitMatrix((uint8_t*)matrices[MAT_A].ptr, testCase.TOTAL_M, testCase.TOTAL_K, matrices[MAT_A].dims.cols, (FillDataType)init, 2);
        InitMatrix((uint8_t*)matrices[MAT_B].ptr, testCase.TOTAL_K, testCase.TOTAL_N, matrices[MAT_B].dims.cols, (FillDataType)init, 2);
        InitMatrix((uint32_t*)matrices[MAT_C].ptr,testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_C].dims.cols, FILL_WITH_ZERO, 2);
        InitMatrix((uint32_t*)matrices[MAT_R].ptr,testCase.TOTAL_M, testCase.TOTAL_N, matrices[MAT_R].dims.cols, FILL_WITH_ZERO, 2);

        if (m_transpose_when_needed || m_validate_matrix_result)
        {
            if (layoutA_Mfirst) // Matrix A M-First?
                TransposeMatrix((uint8_t*)matrices[MAT_A].ptr, matrices[MAT_A].dims.rows, matrices[MAT_A].dims.cols, "layoutA_Mfirst");
            if (layoutB_Kfirst) // Matrix B K-First?
                TransposeMatrix((uint8_t*)matrices[MAT_B].ptr, matrices[MAT_B].dims.rows, matrices[MAT_B].dims.cols, "layoutB_Kfirst");
            if (layoutC_Mfirst) // Matrix C M-First?
                TransposeMatrix((uint32_t*)matrices[MAT_C].ptr, matrices[MAT_C].dims.rows, matrices[MAT_C].dims.cols, "layoutC_Mfirst");
        }
    }
    else
    {
        return std::nullopt;
    }

    // Specialize the shader with the matrix sizes, strides, and constants.
    // Also, work-group sizes
    const uint32_t specDataMxM[] = {   // pass to shader_name.comp
        local_size_x,               // layout(constant_id = 0) const uint local_size_x;
        local_size_y,               // layout(constant_id = 1) const uint local_size_y;
        local_size_z,               // layout(constant_id = 2) const uint local_size_z;
        testCase.TOTAL_M,           // layout(constant_id = 3) const uint TOTAL_M = 1;
        testCase.TOTAL_N,           // layout(constant_id = 4) const uint TOTAL_N = 1;
        testCase.TOTAL_K,           // layout(constant_id = 5) const uint TOTAL_K = 1;
        testCase.TILE_M,            // layout(constant_id = 6) const uint TILE_M = 1;
        testCase.TILE_N,            // layout(constant_id = 7) const uint TILE_N = 1;
        testCase.TILE_K,            // layout(constant_id = 8) const uint TILE_K = 1;
        testCase.layoutA_Mfirst,    // layout(constant_id = 9) const bool layoutA_Mfirst = false;
        testCase.layoutB_Kfirst,    // layout(constant_id =10) const bool layoutB_Kfirst = false;
        testCase.layoutC_Mfirst,    // layout(constant_id =11) const bool layoutC_Mfirst = false;
        testCase.layoutR_Mfirst,    // layout(constant_id =12) const bool layoutR_Mfirst = false;
        testCase.strideAinElements, // layout(constant_id =13) const uint strideAinElements = 1;
        testCase.strideBinElements, // layout(constant_id =14) const uint strideBinElements = 1;
        testCase.strideCinElements, // layout(constant_id =15) const uint strideCinElements = 1;
        testCase.strideRinElements  // layout(constant_id =16) const uint strideRinElements = 1;
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

    VkSpecializationMapEntry entriesMxM[ARRAY_LENGTH(specDataMxM)];
    fill_specialized_map_entries(entriesMxM, ARRAY_LENGTH(specDataMxM), sizeof(uint32_t)); // {0,  sizeof(uint32_t) * 0, sizeof(uint32_t)},...,//{end,  sizeof(uint32_t) * end, sizeof(uint32_t)}

    VkSpecializationMapEntry entriesCONV[ARRAY_LENGTH(specDataCONV)];
    fill_specialized_map_entries(entriesCONV, ARRAY_LENGTH(specDataCONV), sizeof(uint32_t)); // {0, sizeof(uint32_t) * 0, sizeof(uint32_t)}, ...,//{end,  sizeof(uint32_t) * end, sizeof(uint32_t)}

    VkSpecializationInfo specInfo;
    switch (tt)
    {
    case TT_CONV:
        specInfo = { ARRAY_LENGTH(specDataCONV), entriesCONV, sizeof(specDataCONV), specDataCONV, };
        break;
    case TT_MXM_BASIC:
    case TT_MXM_VecToMat:
        specInfo = { ARRAY_LENGTH(specDataMxM), entriesMxM, sizeof(specDataMxM), specDataMxM, };
        break;
    default:
        LOGE("Unknown use case(%d), can't sent specialized constantas to shader!", tt);
    }

#undef ARRAY_LENGTH

    // Create pipeline with a desired subgroup size (e.g., AMD supports two subgroup sizes)
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSizeInfo = {};
    subgroupSizeInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroupSizeInfo.requiredSubgroupSize = subgroup_size; // Must be between min and max

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
    VkPipeline pipeline;
    result = vkCreateComputePipelines(m_vulkan_instance.m_VulkanDevice, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &pipeline);
    CHECK_VK(result);

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

    for (size_t loop = 0; loop < perf_loop; loop++)
    {
        vkCmdPipelineBarrier(commandBuffers[1], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 0, nullptr);
        vkCmdWriteTimestamp( commandBuffers[1], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,   query_pool_timestamps, loop*2  ); // Start timer...
        vkCmdDispatch(       commandBuffers[1], groupCountX, groupCountY, groupCountZ);                                // Dispacth work
        vkCmdWriteTimestamp( commandBuffers[1], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_timestamps,loop*2+1); // Stop timer...
    }

    vkCmdPipelineBarrier(commandBuffers[1], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 0, nullptr);

    result = vkEndCommandBuffer(commandBuffers[1]); // End command buffer recording
    CHECK_VK(result);

    submitInfo.pCommandBuffers = &commandBuffers[1];
    result = vkQueueSubmit(submission_queue, 1, &submitInfo, VK_NULL_HANDLE); // Here is the actual work!
    CHECK_VK(result);
    result = vkQueueWaitIdle(submission_queue);
    CHECK_VK(result);

    vkGetQueryPoolResults(m_vulkan_instance.m_VulkanDevice, query_pool_timestamps, 0,	time_stamps.size(), time_stamps.size() * sizeof(uint64_t), time_stamps.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    
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
    result = vkEndCommandBuffer(commandBuffers[2]); // End command buffer recording
    CHECK_VK(result);

    submitInfo.pCommandBuffers = &commandBuffers[2];
    result = vkQueueSubmit(submission_queue, 1, &submitInfo, VK_NULL_HANDLE);
    CHECK_VK(result);
    result = vkQueueWaitIdle(submission_queue);
    CHECK_VK(result);

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

    vkDestroyPipeline(m_vulkan_instance.m_VulkanDevice, pipeline, NULL);

    vkDestroyShaderModule(m_vulkan_instance.m_VulkanDevice, shaderModule, NULL);

    return test_result;
}
