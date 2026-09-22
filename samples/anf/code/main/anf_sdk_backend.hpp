//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "vulkan/vulkan.hpp"

// ANF SDK headers
#include "anf.h"
#include "anf_sr.h"
#include "anf_fg.h"
#include "anf_types.h"
#include "anf_types_vk.h"

#undef ERROR_INVALID_PARAMETER

namespace Anf
{
    ////////////////////////////////////////////////////////////////////////////////
    // Enum name: AnfBackendResult
    ////////////////////////////////////////////////////////////////////////////////
    enum class AnfBackendResult : uint32_t
    {
        SUCCESS = 0,
        ERROR_INVALID_PARAMETER,
        ERROR_ANFEROR,
        ERROR_NOT_INITIALIZED,
        ERROR_UNSUPPORTED
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Struct name: AnfBackendCaps
    ////////////////////////////////////////////////////////////////////////////////
    struct AnfBackendCaps
    {
        bool     sr_supported    = false;
        uint32_t sr_quality_modes = 0;
        bool     fg_supported    = false;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Struct name: AnfSrFormats
    ////////////////////////////////////////////////////////////////////////////////
    struct AnfSrFormats
    {
        AnfFormat input_color_format  = ANF_FORMAT_UNKNOWN;
        AnfFormat depth_format        = ANF_FORMAT_UNKNOWN;
        AnfFormat motion_format       = ANF_FORMAT_UNKNOWN;
        AnfFormat output_color_format = ANF_FORMAT_UNKNOWN;
        VkImageUsageFlags input_color_usage  = 0;
        VkImageUsageFlags depth_usage        = 0;
        VkImageUsageFlags motion_usage       = 0;
        VkImageUsageFlags output_color_usage = 0;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Struct name: AnfFgFormats
    ////////////////////////////////////////////////////////////////////////////////
    struct AnfFgFormats
    {
        AnfFormat input_color_format  = ANF_FORMAT_UNKNOWN;
        AnfFormat depth_format        = ANF_FORMAT_UNKNOWN;
        AnfFormat motion_format       = ANF_FORMAT_UNKNOWN;
        AnfFormat output_color_format = ANF_FORMAT_UNKNOWN;
        VkImageUsageFlags input_color_usage  = 0;
        VkImageUsageFlags depth_usage        = 0;
        VkImageUsageFlags motion_usage       = 0;
        VkImageUsageFlags output_color_usage = 0;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Struct name: AnfSceneFormats
    ////////////////////////////////////////////////////////////////////////////////
    struct AnfSceneFormats
    {
        AnfFormat        color_format  = ANF_FORMAT_B10G11R11_UFLOAT;
        AnfFormat        depth_format  = ANF_FORMAT_D32_FLOAT;
        AnfFormat        motion_format = ANF_FORMAT_R16G16_FLOAT;
        VkImageUsageFlags color_usage   = 0;
        VkImageUsageFlags depth_usage   = 0;
        VkImageUsageFlags motion_usage  = 0;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Struct name: AnfInstanceConfig
    //
    // Passed to InitializeInstance().  Holds the Vulkan handles and dispatch mode
    // that are shared by all techniques created on this backend.
    ////////////////////////////////////////////////////////////////////////////////
    struct AnfInstanceConfig
    {
        VkInstance          instance;
        VkPhysicalDevice    physical_device;
        VkDevice            device;
        uint32_t            queue_family_index = 0;
        VkQueue             queue              = VK_NULL_HANDLE;
        bool                dispatch_immediate = true;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Struct name: AnfSrConfig
    //
    // Technique-specific parameters for SR.  Vulkan handles are no longer here;
    // they are provided once via AnfInstanceConfig / InitializeInstance().
    ////////////////////////////////////////////////////////////////////////////////
    struct AnfSrConfig
    {
        VkExtent2D          input_extent       = {};
        VkExtent2D          output_extent      = {};
        AnfSRQualityMode   quality_mode       = ANF_SR_QUALITY_MODE_PERFORMANCE;
        uint32_t            max_in_flight      = 2;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Struct name: AnfFgConfig
    //
    // Technique-specific parameters for FG.  Vulkan handles and dispatch mode are
    // provided once via AnfInstanceConfig / InitializeInstance().
    ////////////////////////////////////////////////////////////////////////////////
    struct AnfFgConfig
    {
        VkExtent2D  extent       = {};
        uint32_t    max_in_flight = 2;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Struct name: AnfSrFrameParams
    ////////////////////////////////////////////////////////////////////////////////
    struct AnfSrFrameParams
    {
        // In immediate mode, cmd must be VK_NULL_HANDLE and semaphores are used.
        // In indirect mode, cmd must be a valid VkCommandBuffer and semaphores are ignored.
        VkCommandBuffer              cmd                        = VK_NULL_HANDLE;
        bool                         reset                      = false;
        float                        jitter_x                   = 0.0f;
        float                        jitter_y                   = 0.0f;
        std::span<const VkSemaphore> wait_semaphores;
        std::span<const VkSemaphore> signal_semaphores;
        uint32_t                     temp_queue_family_index_vk = 0;
        VkQueue                      temp_queue_vk              = VK_NULL_HANDLE;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Struct name: AnfFgFrameParams
    ////////////////////////////////////////////////////////////////////////////////
    struct AnfFgFrameParams
    {
        // Same immediate/indirect semantics as AnfSrFrameParams.
        VkCommandBuffer              cmd                        = VK_NULL_HANDLE;
        bool                         reset                      = false;
        std::span<const VkSemaphore> wait_semaphores;
        std::span<const VkSemaphore> signal_semaphores;
        uint32_t                     temp_queue_family_index_vk = 0;
        VkQueue                      temp_queue_vk              = VK_NULL_HANDLE;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Struct name: AnfSrResourcesVk
    ////////////////////////////////////////////////////////////////////////////////
    struct AnfSrResourcesVk
    {
        VkImage       input_color        = VK_NULL_HANDLE;
        VkExtent2D    input_color_ext    = {};
        VkImageLayout input_color_layout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkImage       depth              = VK_NULL_HANDLE;
        VkExtent2D    depth_ext          = {};
        VkImageLayout depth_layout       = VK_IMAGE_LAYOUT_UNDEFINED;

        VkImage       motion_vectors     = VK_NULL_HANDLE;
        VkExtent2D    motion_ext         = {};
        VkImageLayout motion_layout      = VK_IMAGE_LAYOUT_UNDEFINED;

        VkImage       output_color       = VK_NULL_HANDLE;
        VkExtent2D    output_ext         = {};
        VkImageLayout output_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Struct name: AnfFgResourcesVk
    ////////////////////////////////////////////////////////////////////////////////
    struct AnfFgResourcesVk
    {
        VkImage       input_color        = VK_NULL_HANDLE;
        VkExtent2D    input_color_ext    = {};
        VkImageLayout input_color_layout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkImage       depth              = VK_NULL_HANDLE;
        VkExtent2D    depth_ext          = {};
        VkImageLayout depth_layout       = VK_IMAGE_LAYOUT_UNDEFINED;

        VkImage       motion_vectors     = VK_NULL_HANDLE;
        VkExtent2D    motion_ext         = {};
        VkImageLayout motion_layout      = VK_IMAGE_LAYOUT_UNDEFINED;

        VkImage       output_color       = VK_NULL_HANDLE;
        VkExtent2D    output_ext         = {};
        VkImageLayout output_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Class name: AnfSdkBackend
    //
    // Wraps the raw ANF C API. Manages the ANF instance and one technique per
    // supported feature (SR and FG). Call InitializeSr first; InitializeFg reuses
    // the same ANF instance.
    ////////////////////////////////////////////////////////////////////////////////
    class AnfSdkBackend
    {
    public:
        AnfSdkBackend()  = default;
        ~AnfSdkBackend() = default;

        /*
        * Creates the ANF instance, loads functions, and queries technique capabilities
        * and formats.  Must be called before InitializeSr() or InitializeFg().
        * @param config : Vulkan handles and dispatch mode shared by all techniques.
        * @return Backend result code.
        */
        AnfBackendResult InitializeInstance(const AnfInstanceConfig& config);

        /*
        * Creates the SR technique.  InitializeInstance() must have been called first.
        * @param config : SR extents, quality mode, and in-flight count.
        * @return Backend result code.
        */
        AnfBackendResult InitializeSr(const AnfSrConfig& config);

        /*
        * Creates the FG technique.  InitializeInstance() must have been called first.
        * @param config : FG extent and in-flight count.
        * @return Backend result code.
        */
        AnfBackendResult InitializeFg(const AnfFgConfig& config);

        /*
        * Destroys all techniques and the ANF instance.
        * @return Backend result code.
        */
        AnfBackendResult Shutdown();

        /*
        * Dispatches the SR technique.
        * In immediate mode the backend submits work directly; cmd must be VK_NULL_HANDLE.
        * In indirect mode ANF records into cmd; the caller is responsible for submission.
        */
        AnfBackendResult DispatchSr(const AnfSrFrameParams& frame, const AnfSrResourcesVk& resources);

        /*
        * Dispatches the FG technique.
        * Same immediate/indirect semantics as DispatchSr.
        */
        AnfBackendResult DispatchFg(const AnfFgFrameParams& frame, const AnfFgResourcesVk& resources);

        inline const AnfBackendCaps& GetCaps()      const { return m_caps; }
        inline const AnfSrFormats&   GetSrFormats() const { return m_sr_formats; }
        inline const AnfFgFormats&   GetFgFormats() const { return m_fg_formats; }
        inline const AnfSceneFormats& GetSceneFormats() const { return m_scene_formats; }
        inline const std::vector<AnfFormat>& GetSrOutputFormatCandidates() const { return m_sr_output_format_candidates; }
        inline const std::vector<AnfFormat>& GetFgOutputFormatCandidates() const { return m_fg_output_format_candidates; }
        inline bool IsValid()            const { return m_is_valid; }
        inline bool IsSrValid()          const { return m_sr_technique != nullptr; }
        inline bool IsFgValid()          const { return m_fg_technique != nullptr; }
        inline bool IsDispatchImmediate() const { return m_dispatch_immediate; }
        bool SetSrOutputFormat(AnfFormat format);
        bool SetFgOutputFormat(AnfFormat format);

    private:
        AnfBackendResult QueryCapsAndFormats();
        struct QueriedResourceInfo
        {
            std::vector<AnfFormat> supported_formats;
            VkImageUsageFlags       image_usage = 0;
        };

        bool QueryResourceInfo(AnfTechniqueId technique_id, AnfResourceLabel label, QueriedResourceInfo& out_info) const;
        void LoadFunctions();

        // Shared helper that fills an AnfResourceParamDesc array for a 4-resource dispatch.
        void BuildResourceParams(
            AnfResourceParamDesc    params[4],
            const AnfSrResourcesVk& resources,
            const AnfSrFormats&     formats);

        void BuildFgResourceParams(
            AnfResourceParamDesc    params[4],
            const AnfFgResourcesVk& resources,
            const AnfFgFormats&     formats);

    private:
        AnfFunctions m_anf            = {};
        bool          m_functions_loaded = false;

        bool              m_is_valid           = false;
        bool              m_dispatch_immediate = true;
        AnfInstance      m_instance           = nullptr;
        AnfInstanceConfig m_instance_config   = {};

        // SR
        AnfTechnique            m_sr_technique    = nullptr;
        AnfTechniqueCreateFlags m_sr_create_flags = 0;
        AnfSrConfig             m_sr_config       = {};
        AnfSrFormats            m_sr_formats      = {};

        // FG
        AnfTechnique            m_fg_technique    = nullptr;
        AnfTechniqueCreateFlags m_fg_create_flags = 0;
        AnfFgConfig             m_fg_config       = {};
        AnfFgFormats            m_fg_formats      = {};
        AnfSceneFormats         m_scene_formats   = {};
        std::vector<AnfFormat>  m_sr_output_format_candidates;
        std::vector<AnfFormat>  m_fg_output_format_candidates;

        AnfBackendCaps m_caps = {};
    };

} // namespace Anf
