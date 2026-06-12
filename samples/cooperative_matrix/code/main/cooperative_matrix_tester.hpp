//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include <vulkan/vulkan.hpp>
#include "runtime_shader.hpp"
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>
#include <glm/glm.hpp>

enum { MAT_A = 0, MAT_B = 1, MAT_C = 2, MAT_R = 3, NUM_MATS = 4 };
enum TestType
{
    TT_MXM_BASIC = 0,
    TT_MXM_VecToMat = 1,
    TT_CONV = 2,
    TT_COUNT,
};

enum MatrixTransposeOption
{
    ALWAYS_TRUE,
    ALWAYS_FALSE,
    VARIABLE,
};

enum FillDataType { FILL_WITH_ZERO = 0, FILL_WITH_CONSTANTS, FILL_WITH_RANDON_UINT, FILL_WITH_RANDON_INT, FILL_SEQUENCE_INT, FILL_WITH_RANDOM_LOW_HIGH_INT, FILL_WITH_RANDOM_FLOAT, FILL_WITH_RANDOM_PLUS1_MINUS1_FLOAT };

class CooperativeMatrixRunner
{
    struct TestDescription
    {
        TestType test_type = TT_MXM_BASIC;
        FillDataType fill_data_type = FILL_WITH_RANDON_INT;

        VkComponentTypeKHR input_type;
        VkComponentTypeKHR output_type;

        int MSize;
        int NSize;
        int KSize;
        int MSizeInBlocks;
        int NSizeInBlocks;
        int KSizeInBlocks;
        uint32_t perf_loop;

        bool layoutA_Mfirst = false;
        bool layoutB_Nfirst = false;
        bool layoutC_Mfirst = false;
        bool layoutR_Mfirst = false;

        int inputWidth = 32;
        int inputHeight = 16;
    };

    struct TestResult
    {
        bool   is_valid = false;
        double time_total;
        double TOPS;
        double percentage;
        std::optional<bool> validation_pass; // set when validate checkbox is on
    };

    struct SizeConfiguration
    {
        int MSizeInBlocks;
        int NSizeInBlocks;
        int KSizeInBlocks;

        int MSize;
        int NSize;
        int KSize;
    };

    struct TestGroupTemplateDescription
    {
        VkComponentTypeKHR input_type;
        VkComponentTypeKHR output_type;

        std::vector<SizeConfiguration> size_configurations;
    };

    struct TestGroup
    {
        struct TestRowEntry
        {
            std::vector<TestDescription> test_descriptions;
            std::vector<TestResult>      test_results;

            bool layoutA_Mfirst = false;
            bool layoutB_Nfirst = false;
            bool layoutC_Mfirst = false;
            bool layoutR_Mfirst = false;
        };

        TestGroupTemplateDescription template_description;
        std::vector<TestRowEntry>    test_entries; // One per size_in_block_configuration from the template description
    };

public:

    CooperativeMatrixRunner(Vulkan& vulkan_instance);
    ~CooperativeMatrixRunner();

    bool InitializeRunner();

    bool TriggerPendingTests();
    void RenderUI();

private:

    void PrepareTestSession();
    std::optional<TestResult> RunTest(const TestDescription& test_description);

private:

    Vulkan& m_vulkan_instance;

    std::vector<VkCooperativeMatrixPropertiesKHR> m_hFoundCooperativeMatrices;

    TestType     m_test_type           = TT_MXM_BASIC;
    FillDataType m_fill_data_type      = FILL_WITH_RANDON_INT;

    MatrixTransposeOption m_matrix_transpose_options[NUM_MATS] = { VARIABLE , VARIABLE , VARIABLE , ALWAYS_FALSE };

    int  m_test_repeats          = 1;
    bool m_transpose_when_needed = false;
    bool m_validate_matrix_result = false;

    bool m_normalize_inputs = true;
    int m_input_width = 32;
    int m_input_height = 16;

    bool     m_is_processing_tests   = false;
    uint32_t m_total_tests           = 0;
    uint32_t m_total_processed_tests = 0;
    std::vector<TestGroupTemplateDescription> m_test_group_templates;
    std::vector<TestGroup> m_test_groups;

};