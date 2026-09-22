//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#include "anf_sdk_backend.hpp"

#include <algorithm>
#include <initializer_list>
#include <string>

namespace Anf
{
    namespace
    {
        constexpr AnfFormat kFallbackSrInputColor  = ANF_FORMAT_B10G11R11_UFLOAT;
        constexpr AnfFormat kFallbackSrDepth       = ANF_FORMAT_D32_FLOAT;
        constexpr AnfFormat kFallbackSrMotion      = ANF_FORMAT_R16G16_FLOAT;
        constexpr AnfFormat kFallbackSrOutputColor = ANF_FORMAT_R8G8B8A8_UNORM;

        constexpr AnfFormat kFallbackFgInputColor  = ANF_FORMAT_R8G8B8A8_UNORM;
        constexpr AnfFormat kFallbackFgDepth       = ANF_FORMAT_D32_FLOAT;
        constexpr AnfFormat kFallbackFgMotion      = ANF_FORMAT_R16G16_FLOAT;
        constexpr AnfFormat kFallbackFgOutputColor = ANF_FORMAT_R8G8B8A8_UNORM;

        const char* TechniqueName(AnfTechniqueId technique_id)
        {
            switch (technique_id)
            {
                case ANF_TECHNIQUE_ID_SR_TEMPORAL: return "SR";
                case ANF_TECHNIQUE_ID_FG_TEMPORAL: return "FG";
                default:                            return "Unknown";
            }
        }

        const char* ResourceLabelName(AnfResourceLabel label)
        {
            switch (label)
            {
                case ANF_RESOURCE_LABEL_DEPTH:          return "Depth";
                case ANF_RESOURCE_LABEL_MOTION_VECTORS: return "MotionVectors";
                case ANF_RESOURCE_LABEL_INPUT_COLOR:    return "InputColor";
                case ANF_RESOURCE_LABEL_OUTPUT_COLOR:   return "OutputColor";
                default:                                 return "Unknown";
            }
        }

        const char* AnfFormatName(AnfFormat format)
        {
            switch (format)
            {
                case ANF_FORMAT_UNKNOWN:               return "UNKNOWN";
                case ANF_FORMAT_R8G8B8A8_UNORM:        return "R8G8B8A8_UNORM";
                case ANF_FORMAT_B10G11R11_UFLOAT:      return "B10G11R11_UFLOAT";
                case ANF_FORMAT_D32_FLOAT:             return "D32_FLOAT";
                case ANF_FORMAT_D24S8_UNORM:           return "D24S8_UNORM";
                case ANF_FORMAT_R16G16_FLOAT:          return "R16G16_FLOAT";
                case ANF_FORMAT_R16G16B16A16_FLOAT:    return "R16G16B16A16_FLOAT";
                default:                                return "UNRECOGNIZED";
            }
        }

        AnfFormat GetFallbackFormat(AnfTechniqueId technique_id, AnfResourceLabel label)
        {
            if (technique_id == ANF_TECHNIQUE_ID_SR_TEMPORAL)
            {
                switch (label)
                {
                    case ANF_RESOURCE_LABEL_INPUT_COLOR:    return kFallbackSrInputColor;
                    case ANF_RESOURCE_LABEL_DEPTH:          return kFallbackSrDepth;
                    case ANF_RESOURCE_LABEL_MOTION_VECTORS: return kFallbackSrMotion;
                    case ANF_RESOURCE_LABEL_OUTPUT_COLOR:   return kFallbackSrOutputColor;
                    default:                                 return ANF_FORMAT_UNKNOWN;
                }
            }

            if (technique_id == ANF_TECHNIQUE_ID_FG_TEMPORAL)
            {
                switch (label)
                {
                    case ANF_RESOURCE_LABEL_INPUT_COLOR:    return kFallbackFgInputColor;
                    case ANF_RESOURCE_LABEL_DEPTH:          return kFallbackFgDepth;
                    case ANF_RESOURCE_LABEL_MOTION_VECTORS: return kFallbackFgMotion;
                    case ANF_RESOURCE_LABEL_OUTPUT_COLOR:   return kFallbackFgOutputColor;
                    default:                                 return ANF_FORMAT_UNKNOWN;
                }
            }

            return ANF_FORMAT_UNKNOWN;
        }

        bool ContainsFormat(const std::vector<AnfFormat>& formats, AnfFormat format)
        {
            return std::find(formats.begin(), formats.end(), format) != formats.end();
        }

        AnfFormat ChoosePreferredFormat(
            const std::vector<AnfFormat>& available_formats,
            std::initializer_list<AnfFormat> preferred_order,
            AnfFormat fallback)
        {
            for (const AnfFormat preferred : preferred_order)
            {
                if (ContainsFormat(available_formats, preferred))
                {
                    return preferred;
                }
            }

            if (!available_formats.empty())
            {
                return available_formats.front();
            }

            return fallback;
        }

        std::vector<AnfFormat> BuildPreferredFormatOrder(
            const std::vector<AnfFormat>& available_formats,
            std::initializer_list<AnfFormat> preferred_order,
            AnfFormat fallback)
        {
            std::vector<AnfFormat> ordered;
            ordered.reserve(available_formats.size() + 1);

            for (const AnfFormat preferred : preferred_order)
            {
                if (ContainsFormat(available_formats, preferred) && !ContainsFormat(ordered, preferred))
                {
                    ordered.push_back(preferred);
                }
            }

            for (const AnfFormat available : available_formats)
            {
                if (!ContainsFormat(ordered, available))
                {
                    ordered.push_back(available);
                }
            }

            if (ordered.empty())
            {
                ordered.push_back(fallback);
            }

            return ordered;
        }

        AnfFormat ChooseCompatibleFormat(
            const std::vector<AnfFormat>& preferred_formats,
            const std::vector<AnfFormat>& secondary_formats,
            AnfFormat                     fallback)
        {
            if (!preferred_formats.empty() && !secondary_formats.empty())
            {
                for (const AnfFormat format : preferred_formats)
                {
                    if (ContainsFormat(secondary_formats, format))
                    {
                        return format;
                    }
                }
            }

            if (!preferred_formats.empty())
            {
                return preferred_formats.front();
            }

            if (!secondary_formats.empty())
            {
                return secondary_formats.front();
            }

            return fallback;
        }

        void LogUsageFlags(const char* prefix, VkImageUsageFlags usage)
        {
            LOGI("%s usage: colorAttachment=%d sampled=%d storage=%d transferSrc=%d transferDst=%d depthStencil=%d",
                 prefix,
                 (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0,
                 (usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0,
                 (usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0,
                 (usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0,
                 (usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0,
                 (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0);
        }

        void LogFormats(const char* prefix, const std::vector<AnfFormat>& formats)
        {
            if (formats.empty())
            {
                LOGW("%s supported formats: <none>", prefix);
                return;
            }

            std::string joined;
            for (size_t index = 0; index < formats.size(); ++index)
            {
                if (!joined.empty())
                {
                    joined += ", ";
                }
                joined += AnfFormatName(formats[index]);
            }

            LOGI("%s supported formats: %s", prefix, joined.c_str());
        }
    }

    static inline AnfStructHeader MakeHeader(AnfStructType type)
    {
        AnfStructHeader hdr{};
        hdr.type  = type;
        hdr.pNext = nullptr;
        return hdr;
    }

    // -------------------------------------------------------------------------
    // LoadFunctions
    // -------------------------------------------------------------------------

    void AnfSdkBackend::LoadFunctions()
    {
        if (m_functions_loaded)
        {
            return;
        }

#if defined(OS_ANDROID)
        // Stage the API table in padded storage before copying its declared entries.
        struct AnfFunctionsGuarded
        {
            AnfFunctions known;
            uint8_t       overflow_guard[128];  // absorbs up to 16 extra function pointers
        } buf = {};

        GetAnfFunctions(&buf.known);
        m_anf = buf.known;
        m_functions_loaded = true;
#endif
    }

    // -------------------------------------------------------------------------
    // InitializeInstance
    // -------------------------------------------------------------------------

    AnfBackendResult AnfSdkBackend::InitializeInstance(const AnfInstanceConfig& config)
    {
        LOGI("ANF - Loading functions");
        LoadFunctions();
        if (!m_functions_loaded)
        {
            return AnfBackendResult::ERROR_UNSUPPORTED;
        }

        m_instance_config    = config;
        m_dispatch_immediate = config.dispatch_immediate;

        auto InstanceLog = [](AnfLogLevel logLevel, const char* pMessage)
        {
            if (!pMessage) return;
            switch (logLevel)
            {
                case ANF_LOG_LEVEL_ERROR:   LOGE("ANF - %s", pMessage); break;
                case ANF_LOG_LEVEL_WARNING: LOGW("ANF - %s", pMessage); break;
                default:                     LOGI("ANF - %s", pMessage); break;
            }
        };

        AnfInstanceCreateInfo instance_ci{};
        instance_ci.header             = MakeHeader(ANF_STYPE_INSTANCE_CREATE_INFO);
        instance_ci.logMessageCallback = InstanceLog;
        instance_ci.logLevel           = ANF_LOG_LEVEL_WARNING;
        instance_ci.flags              = 0;
        instance_ci.clientApi          = ANF_CLIENT_API_VULKAN;

        if (m_anf.CreateInstance(&instance_ci, &m_instance) != ANF_RESULT_SUCCESS)
        {
            return AnfBackendResult::ERROR_ANFEROR;
        }

        AnfAdapterInfoVulkan adapterInfo{};
        adapterInfo.header         = MakeHeader(ANF_STYPE_ADAPTER_INFO_VULKAN);
        adapterInfo.instance       = config.instance;
        adapterInfo.physicalDevice = config.physical_device;
        adapterInfo.device         = config.device;

        AnfQueueInfoVulkan queueInfo{};
        if (m_dispatch_immediate)
        {
            if (config.queue == VK_NULL_HANDLE)
            {
                Shutdown();
                return AnfBackendResult::ERROR_INVALID_PARAMETER;
            }

            queueInfo.header              = MakeHeader(ANF_STYPE_QUEUE_INFO_VULKAN);
            queueInfo.numQueues           = 1;
            queueInfo.pQueues             = const_cast<VkQueue*>(&config.queue);
            queueInfo.pQueueFamilyIndices = const_cast<uint32_t*>(&config.queue_family_index);
            adapterInfo.header.pNext      = reinterpret_cast<AnfStructHeader*>(&queueInfo);
        }

        if (!m_anf.SetInstanceClientAPIAdapterInfo ||
            m_anf.SetInstanceClientAPIAdapterInfo(m_instance, reinterpret_cast<const AnfStructHeader*>(&adapterInfo)) != ANF_RESULT_SUCCESS)
        {
            Shutdown();
            return AnfBackendResult::ERROR_ANFEROR;
        }

        if (QueryCapsAndFormats() != AnfBackendResult::SUCCESS)
        {
            Shutdown();
            return AnfBackendResult::ERROR_ANFEROR;
        }

        // Instance is valid; SR and FG techniques can now be created independently.
        m_is_valid = true;
        return AnfBackendResult::SUCCESS;
    }

    // -------------------------------------------------------------------------
    // InitializeSr
    // -------------------------------------------------------------------------

    AnfBackendResult AnfSdkBackend::InitializeSr(const AnfSrConfig& config)
    {
        // The ANF instance must already exist (created by InitializeInstance).
        if (!m_is_valid || m_instance == nullptr)
        {
            return AnfBackendResult::ERROR_NOT_INITIALIZED;
        }

        if (!m_caps.sr_supported)
        {
            return AnfBackendResult::ERROR_UNSUPPORTED;
        }

        const uint32_t requested_mode_bit = (1u << static_cast<uint32_t>(config.quality_mode));
        if ((m_caps.sr_quality_modes & requested_mode_bit) == 0)
        {
            LOGW("ANF SR quality mode %u is unsupported by this SDK/device", static_cast<uint32_t>(config.quality_mode));
            return AnfBackendResult::ERROR_UNSUPPORTED;
        }

        m_sr_config = config;

        AnfSRCreateInfo sr_ci{};
        sr_ci.header     = MakeHeader(ANF_STYPE_SR_CREATE_INFO);
        sr_ci.inputSize  = { config.input_extent.width,  config.input_extent.height };
        sr_ci.outputSize = { config.output_extent.width, config.output_extent.height };

        AnfTechniqueVulkanCreateInfo vk_ci{};
        vk_ci.header = MakeHeader(ANF_STYPE_TECHNIQUE_VULKAN_CREATE_INFO);

        // When dispatch_immediate is true the technique owns its own submission.
        // When false, the caller records ANF commands into a provided VkCommandBuffer.
        m_sr_create_flags = m_dispatch_immediate ? ANF_TECHNIQUE_CREATE_FLAG_DISPATCH_IMMEDIATE : 0;

        AnfTechniqueCreateInfo tech_ci{};
        tech_ci.header                    = MakeHeader(ANF_STYPE_TECHNIQUE_CREATE_INFO);
        tech_ci.techniqueId               = ANF_TECHNIQUE_ID_SR_TEMPORAL;
        tech_ci.flags                     = m_sr_create_flags;
        tech_ci.maxInFlight               = config.max_in_flight;
        tech_ci.pClientApiCreateInfo      = reinterpret_cast<AnfStructHeader*>(&vk_ci);
        tech_ci.pTechniqueGroupCreateInfo = reinterpret_cast<AnfStructHeader*>(&sr_ci);

        if (m_anf.CreateTechnique(m_instance, &tech_ci, &m_sr_technique) != ANF_RESULT_SUCCESS)
        {
            m_sr_technique = nullptr;
            return AnfBackendResult::ERROR_ANFEROR;
        }

        return AnfBackendResult::SUCCESS;
    }

    // -------------------------------------------------------------------------
    // InitializeFg
    // -------------------------------------------------------------------------

    AnfBackendResult AnfSdkBackend::InitializeFg(const AnfFgConfig& config)
    {
        // The ANF instance must already exist (created by InitializeInstance).
        if (!m_is_valid || m_instance == nullptr)
        {
            return AnfBackendResult::ERROR_NOT_INITIALIZED;
        }

        if (!m_caps.fg_supported)
        {
            return AnfBackendResult::ERROR_UNSUPPORTED;
        }

        m_fg_config = config;

        AnfFGCreateInfo fg_ci{};
        fg_ci.header    = MakeHeader(ANF_STYPE_FG_CREATE_INFO);
        fg_ci.inputSize = { config.extent.width, config.extent.height };

        AnfTechniqueVulkanCreateInfo vk_ci{};
        vk_ci.header = MakeHeader(ANF_STYPE_TECHNIQUE_VULKAN_CREATE_INFO);

        m_fg_create_flags = m_dispatch_immediate ? ANF_TECHNIQUE_CREATE_FLAG_DISPATCH_IMMEDIATE : 0;

        AnfTechniqueCreateInfo tech_ci{};
        tech_ci.header                    = MakeHeader(ANF_STYPE_TECHNIQUE_CREATE_INFO);
        tech_ci.techniqueId               = ANF_TECHNIQUE_ID_FG_TEMPORAL;
        tech_ci.flags                     = m_fg_create_flags;
        tech_ci.maxInFlight               = config.max_in_flight;
        tech_ci.pClientApiCreateInfo      = reinterpret_cast<AnfStructHeader*>(&vk_ci);
        tech_ci.pTechniqueGroupCreateInfo = reinterpret_cast<AnfStructHeader*>(&fg_ci);

        if (m_anf.CreateTechnique(m_instance, &tech_ci, &m_fg_technique) != ANF_RESULT_SUCCESS)
        {
            m_fg_technique = nullptr;
            return AnfBackendResult::ERROR_ANFEROR;
        }

        return AnfBackendResult::SUCCESS;
    }

    // -------------------------------------------------------------------------
    // Shutdown
    // -------------------------------------------------------------------------

    AnfBackendResult AnfSdkBackend::Shutdown()
    {
        // Release techniques together with their owning instance after GPU work completes.
        if (m_instance_config.device != VK_NULL_HANDLE)
            vkDeviceWaitIdle(m_instance_config.device);

        if (m_fg_technique != nullptr && m_instance != nullptr)
        {
            LOGI("ANF - releasing FG technique with its parent instance");
            m_fg_technique = nullptr;
        }

        if (m_sr_technique != nullptr && m_instance != nullptr)
        {
            LOGI("ANF - releasing SR technique with its parent instance");
            m_sr_technique = nullptr;
        }

        if (m_instance != nullptr)
        {
            m_anf.DestroyInstance(m_instance);
            m_instance = nullptr;
        }

        m_is_valid       = false;
        m_caps           = {};
        m_sr_formats     = {};
        m_fg_formats     = {};
        m_sr_config      = {};
        m_fg_config      = {};
        m_instance_config = {};
        m_scene_formats  = {};
        m_sr_output_format_candidates.clear();
        m_fg_output_format_candidates.clear();

        return AnfBackendResult::SUCCESS;
    }

    // -------------------------------------------------------------------------
    // QueryCapsAndFormats
    // -------------------------------------------------------------------------

    AnfBackendResult AnfSdkBackend::QueryCapsAndFormats()
    {
        const auto query_support = [&](AnfTechniqueId technique_id)
        {
            if (!m_anf.IsTechniqueSupported)
            {
                LOGW("ANF %s - IsTechniqueSupported unavailable, assuming supported", TechniqueName(technique_id));
                return true;
            }

            AnfAdapterInfoVulkan adapter_info{};
            adapter_info.header         = MakeHeader(ANF_STYPE_ADAPTER_INFO_VULKAN);
            adapter_info.instance       = m_instance_config.instance;
            adapter_info.physicalDevice = m_instance_config.physical_device;

            const AnfResult result = m_anf.IsTechniqueSupported(
                m_instance,
                technique_id,
                reinterpret_cast<const AnfStructHeader*>(&adapter_info));

            if (result == ANF_RESULT_SUCCESS)
            {
                LOGI("ANF %s - technique supported", TechniqueName(technique_id));
                return true;
            }

            if (result == ANF_RESULT_NOT_SUPPORTED)
            {
                LOGI("ANF %s - technique not supported", TechniqueName(technique_id));
                return false;
            }

            LOGW("ANF %s - IsTechniqueSupported failed with result %u", TechniqueName(technique_id), static_cast<uint32_t>(result));
            return false;
        };

        m_caps.sr_supported = query_support(ANF_TECHNIQUE_ID_SR_TEMPORAL);
        m_caps.fg_supported = query_support(ANF_TECHNIQUE_ID_FG_TEMPORAL);

        QueriedResourceInfo sr_input_color{};
        QueriedResourceInfo sr_depth{};
        QueriedResourceInfo sr_motion{};
        QueriedResourceInfo sr_output_color{};
        QueriedResourceInfo fg_input_color{};
        QueriedResourceInfo fg_depth{};
        QueriedResourceInfo fg_motion{};
        QueriedResourceInfo fg_output_color{};

        const auto log_technique_requirements = [&](AnfTechniqueId technique_id)
        {
            const AnfTechniqueRequirements* p_tech_req = nullptr;
            if (!m_anf.QueryTechniqueRequirements ||
                m_anf.QueryTechniqueRequirements(m_instance, technique_id, &p_tech_req) != ANF_RESULT_SUCCESS)
            {
                LOGW("ANF %s - QueryTechniqueRequirements failed", TechniqueName(technique_id));
                if (technique_id == ANF_TECHNIQUE_ID_SR_TEMPORAL)
                {
                    m_caps.sr_quality_modes = (1u << static_cast<uint32_t>(ANF_SR_QUALITY_MODE_PERFORMANCE));
                }
                return;
            }

            if (!p_tech_req)
            {
                LOGW("ANF %s - no technique requirements returned", TechniqueName(technique_id));
                if (technique_id == ANF_TECHNIQUE_ID_SR_TEMPORAL)
                {
                    m_caps.sr_quality_modes = (1u << static_cast<uint32_t>(ANF_SR_QUALITY_MODE_PERFORMANCE));
                }
                return;
            }

            for (uint32_t index = 0; index < p_tech_req->numRequiredResources; ++index)
            {
                LOGI("ANF %s - required resource: %s", TechniqueName(technique_id), ResourceLabelName(p_tech_req->pRequiredResources[index]));
            }

            for (uint32_t index = 0; index < p_tech_req->numOptionalResources; ++index)
            {
                LOGI("ANF %s - optional resource: %s", TechniqueName(technique_id), ResourceLabelName(p_tech_req->pOptionalResources[index]));
            }

            if (p_tech_req->pTechniqueGroupRequirements &&
                p_tech_req->pTechniqueGroupRequirements->type == ANF_STYPE_SR_REQUIREMENTS)
            {
                const auto* sr_req = reinterpret_cast<const AnfSRRequirements*>(p_tech_req->pTechniqueGroupRequirements);
                m_caps.sr_quality_modes = sr_req->supportedQualityModes;
                LOGI("ANF SR - supported quality mode mask: 0x%08x", m_caps.sr_quality_modes);
            }

            if (p_tech_req->pClientApiRequirements == nullptr ||
                p_tech_req->pClientApiRequirements->type != ANF_STYPE_TECHNIQUE_VULKAN_API_REQUIREMENTS)
            {
                return;
            }

            const auto* vk_req = reinterpret_cast<const AnfTechniqueVulkanApiRequirements*>(
                p_tech_req->pClientApiRequirements);
            for (uint32_t ext_index = 0; ext_index < vk_req->numDeviceExtensions; ++ext_index)
            {
                LOGI("ANF %s - required device extension: %s", TechniqueName(technique_id), vk_req->ppDeviceExtensions[ext_index]);
            }
            for (uint32_t ext_index = 0; ext_index < vk_req->numInstanceExtensions; ++ext_index)
            {
                LOGI("ANF %s - required instance extension: %s", TechniqueName(technique_id), vk_req->ppInstanceExtensions[ext_index]);
            }
        };

        if (m_caps.sr_supported)
        {
            log_technique_requirements(ANF_TECHNIQUE_ID_SR_TEMPORAL);
            QueryResourceInfo(ANF_TECHNIQUE_ID_SR_TEMPORAL, ANF_RESOURCE_LABEL_INPUT_COLOR, sr_input_color);
            QueryResourceInfo(ANF_TECHNIQUE_ID_SR_TEMPORAL, ANF_RESOURCE_LABEL_DEPTH, sr_depth);
            QueryResourceInfo(ANF_TECHNIQUE_ID_SR_TEMPORAL, ANF_RESOURCE_LABEL_MOTION_VECTORS, sr_motion);
            QueryResourceInfo(ANF_TECHNIQUE_ID_SR_TEMPORAL, ANF_RESOURCE_LABEL_OUTPUT_COLOR, sr_output_color);

            m_sr_formats.input_color_format = ChoosePreferredFormat(
                sr_input_color.supported_formats,
                { ANF_FORMAT_B10G11R11_UFLOAT, ANF_FORMAT_R8G8B8A8_UNORM, ANF_FORMAT_R16G16B16A16_FLOAT },
                kFallbackSrInputColor);
            m_sr_formats.depth_format = ChoosePreferredFormat(
                sr_depth.supported_formats,
                { ANF_FORMAT_D32_FLOAT, ANF_FORMAT_D24S8_UNORM },
                kFallbackSrDepth);
            m_sr_formats.motion_format = ChoosePreferredFormat(
                sr_motion.supported_formats,
                { ANF_FORMAT_R16G16_FLOAT },
                kFallbackSrMotion);
            m_sr_output_format_candidates = BuildPreferredFormatOrder(
                sr_output_color.supported_formats,
                { ANF_FORMAT_R8G8B8A8_UNORM, ANF_FORMAT_B10G11R11_UFLOAT, ANF_FORMAT_R16G16B16A16_FLOAT },
                kFallbackSrOutputColor);
            m_sr_formats.output_color_format = m_sr_output_format_candidates.front();
            m_sr_formats.input_color_usage   = sr_input_color.image_usage;
            m_sr_formats.depth_usage         = sr_depth.image_usage;
            m_sr_formats.motion_usage        = sr_motion.image_usage;
            m_sr_formats.output_color_usage  = sr_output_color.image_usage;

            LOGI("ANF SR - selected formats: input=%s depth=%s motion=%s output=%s",
                 AnfFormatName(m_sr_formats.input_color_format),
                 AnfFormatName(m_sr_formats.depth_format),
                 AnfFormatName(m_sr_formats.motion_format),
                 AnfFormatName(m_sr_formats.output_color_format));
        }
        else
        {
            m_caps.sr_quality_modes = 0;
            m_sr_output_format_candidates = { kFallbackSrOutputColor };
        }

        if (m_caps.fg_supported)
        {
            log_technique_requirements(ANF_TECHNIQUE_ID_FG_TEMPORAL);
            QueryResourceInfo(ANF_TECHNIQUE_ID_FG_TEMPORAL, ANF_RESOURCE_LABEL_INPUT_COLOR, fg_input_color);
            QueryResourceInfo(ANF_TECHNIQUE_ID_FG_TEMPORAL, ANF_RESOURCE_LABEL_DEPTH, fg_depth);
            QueryResourceInfo(ANF_TECHNIQUE_ID_FG_TEMPORAL, ANF_RESOURCE_LABEL_MOTION_VECTORS, fg_motion);
            QueryResourceInfo(ANF_TECHNIQUE_ID_FG_TEMPORAL, ANF_RESOURCE_LABEL_OUTPUT_COLOR, fg_output_color);

            m_fg_formats.input_color_format = ChoosePreferredFormat(
                fg_input_color.supported_formats,
                { ANF_FORMAT_R8G8B8A8_UNORM, ANF_FORMAT_B10G11R11_UFLOAT, ANF_FORMAT_R16G16B16A16_FLOAT },
                kFallbackFgInputColor);
            m_fg_formats.depth_format = ChoosePreferredFormat(
                fg_depth.supported_formats,
                { ANF_FORMAT_D32_FLOAT, ANF_FORMAT_D24S8_UNORM },
                kFallbackFgDepth);
            m_fg_formats.motion_format = ChoosePreferredFormat(
                fg_motion.supported_formats,
                { ANF_FORMAT_R16G16_FLOAT },
                kFallbackFgMotion);
            m_fg_output_format_candidates = BuildPreferredFormatOrder(
                fg_output_color.supported_formats,
                { ANF_FORMAT_R8G8B8A8_UNORM, ANF_FORMAT_B10G11R11_UFLOAT, ANF_FORMAT_R16G16B16A16_FLOAT },
                kFallbackFgOutputColor);
            m_fg_formats.output_color_format = m_fg_output_format_candidates.front();
            m_fg_formats.input_color_usage   = fg_input_color.image_usage;
            m_fg_formats.depth_usage         = fg_depth.image_usage;
            m_fg_formats.motion_usage        = fg_motion.image_usage;
            m_fg_formats.output_color_usage  = fg_output_color.image_usage;

            LOGI("ANF FG - selected formats: input=%s depth=%s motion=%s output=%s",
                 AnfFormatName(m_fg_formats.input_color_format),
                 AnfFormatName(m_fg_formats.depth_format),
                 AnfFormatName(m_fg_formats.motion_format),
                 AnfFormatName(m_fg_formats.output_color_format));
        }
        else
        {
            m_fg_output_format_candidates = { kFallbackFgOutputColor };
        }

        m_scene_formats.color_format = m_caps.sr_supported
            ? m_sr_formats.input_color_format
            : kFallbackSrInputColor;
        m_scene_formats.color_usage = m_caps.sr_supported
            ? m_sr_formats.input_color_usage
            : (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

        m_scene_formats.depth_format = ChooseCompatibleFormat(
            sr_depth.supported_formats,
            fg_depth.supported_formats,
            kFallbackSrDepth);
        m_scene_formats.motion_format = ChooseCompatibleFormat(
            sr_motion.supported_formats,
            fg_motion.supported_formats,
            kFallbackSrMotion);
        m_scene_formats.depth_usage = sr_depth.image_usage | fg_depth.image_usage;
        m_scene_formats.motion_usage = sr_motion.image_usage | fg_motion.image_usage;

        if (m_caps.sr_supported && m_caps.fg_supported)
        {
            if (!sr_depth.supported_formats.empty() &&
                !fg_depth.supported_formats.empty() &&
                !ContainsFormat(fg_depth.supported_formats, m_scene_formats.depth_format))
            {
                LOGW("ANF scene depth picked %s without SR/FG intersection; FG-only may be constrained",
                     AnfFormatName(m_scene_formats.depth_format));
            }

            if (!sr_motion.supported_formats.empty() &&
                !fg_motion.supported_formats.empty() &&
                !ContainsFormat(fg_motion.supported_formats, m_scene_formats.motion_format))
            {
                LOGW("ANF scene motion picked %s without SR/FG intersection; FG-only may be constrained",
                     AnfFormatName(m_scene_formats.motion_format));
            }
        }

        return AnfBackendResult::SUCCESS;
    }

    bool AnfSdkBackend::SetSrOutputFormat(AnfFormat format)
    {
        if (!ContainsFormat(m_sr_output_format_candidates, format))
        {
            return false;
        }
        m_sr_formats.output_color_format = format;
        return true;
    }

    bool AnfSdkBackend::SetFgOutputFormat(AnfFormat format)
    {
        if (!ContainsFormat(m_fg_output_format_candidates, format))
        {
            return false;
        }
        m_fg_formats.output_color_format = format;
        return true;
    }

    // -------------------------------------------------------------------------
    // QueryResourceInfo
    // -------------------------------------------------------------------------

    bool AnfSdkBackend::QueryResourceInfo(
        AnfTechniqueId  technique_id,
        AnfResourceLabel label,
        QueriedResourceInfo& out_info) const
    {
        out_info = {};

        const char* prefix = TechniqueName(technique_id);
        const char* resource_name = ResourceLabelName(label);
        const AnfResourceRequirements* p_req = nullptr;
        if (!m_anf.QueryTechniqueResourceRequirements ||
            m_anf.QueryTechniqueResourceRequirements(m_instance, technique_id, label, &p_req) != ANF_RESULT_SUCCESS)
        {
            LOGW("ANF %s - QueryTechniqueResourceRequirements failed for %s, using fallback %s",
                 prefix,
                 resource_name,
                 AnfFormatName(GetFallbackFormat(technique_id, label)));
            out_info.supported_formats.push_back(GetFallbackFormat(technique_id, label));
            return false;
        }

        if (p_req == nullptr || p_req->numSupportedFormats == 0 || p_req->pSupportedFormats == nullptr)
        {
            LOGW("ANF %s - no supported formats returned for %s, using fallback %s",
                 prefix,
                 resource_name,
                 AnfFormatName(GetFallbackFormat(technique_id, label)));
            out_info.supported_formats.push_back(GetFallbackFormat(technique_id, label));
            return false;
        }

        out_info.supported_formats.assign(p_req->pSupportedFormats, p_req->pSupportedFormats + p_req->numSupportedFormats);

        if (p_req->pClientApiRequirements != nullptr &&
            p_req->pClientApiRequirements->type == ANF_STYPE_RESOURCE_VULKAN_API_REQUIREMENTS)
        {
            const auto* vk_req = reinterpret_cast<const AnfResourceVulkanApiRequirements*>(
                p_req->pClientApiRequirements);
            out_info.image_usage = vk_req->imageUsageFlags;
        }

        std::string log_prefix = std::string("ANF ") + prefix + " - " + resource_name;
        LogFormats(log_prefix.c_str(), out_info.supported_formats);
        LogUsageFlags(log_prefix.c_str(), out_info.image_usage);
        return true;
    }

    // -------------------------------------------------------------------------
    // BuildResourceParams  (SR)
    // -------------------------------------------------------------------------

    void AnfSdkBackend::BuildResourceParams(
        AnfResourceParamDesc    params[4],
        const AnfSrResourcesVk& resources,
        const AnfSrFormats&     formats)
    {
        auto fill = [&](int idx, AnfResourceLabel label, AnfFormat fmt,
                        VkImage image, VkExtent2D ext, VkImageLayout layout)
        {
            params[idx].header                          = MakeHeader(ANF_STYPE_RESOURCE_PARAM_DESC);
            params[idx].resourceLabel                   = label;
            params[idx].resourceDesc.header             = MakeHeader(ANF_STYPE_RESOURCE_DESC);
            params[idx].resourceDesc.type               = ANF_RESOURCE_TYPE_TEXTURE2D;
            params[idx].resourceDesc.format             = fmt;
            params[idx].resourceDesc.textureDims.width  = ext.width;
            params[idx].resourceDesc.textureDims.height = ext.height;
            params[idx].resourceDesc.textureDims.numArrayLayers = 1;
            params[idx].resourceDesc.textureDims.mipCount       = 1;
            params[idx].resourceDesc.resource           = reinterpret_cast<AnfHandle>(image);
        };

        fill(0, ANF_RESOURCE_LABEL_INPUT_COLOR,   formats.input_color_format,  resources.input_color,   resources.input_color_ext,  resources.input_color_layout);
        fill(1, ANF_RESOURCE_LABEL_DEPTH,          formats.depth_format,         resources.depth,          resources.depth_ext,         resources.depth_layout);
        fill(2, ANF_RESOURCE_LABEL_MOTION_VECTORS, formats.motion_format,        resources.motion_vectors, resources.motion_ext,        resources.motion_layout);
        fill(3, ANF_RESOURCE_LABEL_OUTPUT_COLOR,   formats.output_color_format, resources.output_color,  resources.output_ext,        resources.output_layout);
    }

    // -------------------------------------------------------------------------
    // BuildFgResourceParams  (FG)
    // -------------------------------------------------------------------------

    void AnfSdkBackend::BuildFgResourceParams(
        AnfResourceParamDesc    params[4],
        const AnfFgResourcesVk& resources,
        const AnfFgFormats&     formats)
    {
        auto fill = [&](int idx, AnfResourceLabel label, AnfFormat fmt,
                        VkImage image, VkExtent2D ext, VkImageLayout layout)
        {
            params[idx].header                          = MakeHeader(ANF_STYPE_RESOURCE_PARAM_DESC);
            params[idx].resourceLabel                   = label;
            params[idx].resourceDesc.header             = MakeHeader(ANF_STYPE_RESOURCE_DESC);
            params[idx].resourceDesc.type               = ANF_RESOURCE_TYPE_TEXTURE2D;
            params[idx].resourceDesc.format             = fmt;
            params[idx].resourceDesc.textureDims.width  = ext.width;
            params[idx].resourceDesc.textureDims.height = ext.height;
            params[idx].resourceDesc.textureDims.numArrayLayers = 1;
            params[idx].resourceDesc.textureDims.mipCount       = 1;
            params[idx].resourceDesc.resource           = reinterpret_cast<AnfHandle>(image);
        };

        fill(0, ANF_RESOURCE_LABEL_INPUT_COLOR,   formats.input_color_format,  resources.input_color,   resources.input_color_ext,  resources.input_color_layout);
        fill(1, ANF_RESOURCE_LABEL_DEPTH,          formats.depth_format,         resources.depth,          resources.depth_ext,         resources.depth_layout);
        fill(2, ANF_RESOURCE_LABEL_MOTION_VECTORS, formats.motion_format,        resources.motion_vectors, resources.motion_ext,        resources.motion_layout);
        fill(3, ANF_RESOURCE_LABEL_OUTPUT_COLOR,   formats.output_color_format, resources.output_color,  resources.output_ext,        resources.output_layout);
    }

    // -------------------------------------------------------------------------
    // DispatchSr
    // -------------------------------------------------------------------------

    AnfBackendResult AnfSdkBackend::DispatchSr(
        const AnfSrFrameParams& frame,
        const AnfSrResourcesVk& resources)
    {
        if (!m_is_valid || m_instance == nullptr || m_sr_technique == nullptr)
        {
            return AnfBackendResult::ERROR_NOT_INITIALIZED;
        }

        // In indirect mode a valid command buffer must be provided.
        if (!m_dispatch_immediate && frame.cmd == VK_NULL_HANDLE)
        {
            return AnfBackendResult::ERROR_INVALID_PARAMETER;
        }

        AnfResourceParamDesc params[4] = {};
        BuildResourceParams(params, resources, m_sr_formats);

        AnfSRDispatch sr_dispatch{};
        sr_dispatch.header       = MakeHeader(ANF_STYPE_SR_DISPATCH);
        sr_dispatch.jitterOffset = { frame.jitter_x, frame.jitter_y };

        AnfTechniqueDispatchInfo dispatch{};
        dispatch.header                      = MakeHeader(ANF_STYPE_TECHNIQUE_DISPATCH_INFO);
        dispatch.commandList                 = reinterpret_cast<AnfClientCommandList>(frame.cmd);
        dispatch.pResources                  = params;
        dispatch.numResources                = 4;
        dispatch.reset                       = frame.reset ? ANF_TRUE : ANF_FALSE;
        dispatch.pTechniqueGroupDispatchInfo = reinterpret_cast<const AnfStructHeader*>(&sr_dispatch);

        AnfTechniqueDispatchImmediateInfo immediate_dispatch{};
        if (m_dispatch_immediate)
        {
            immediate_dispatch.header                 = MakeHeader(ANF_STYPE_TECHNIQUE_DISPATCH_IMMEDIATE_INFO);
            immediate_dispatch.pWaitSyncObjects      = reinterpret_cast<const AnfClientSyncObject*>(frame.wait_semaphores.data());
            immediate_dispatch.numWaitSyncObjects    = static_cast<uint32_t>(frame.wait_semaphores.size());
            immediate_dispatch.pSignalSyncObjects    = reinterpret_cast<const AnfClientSyncObject*>(frame.signal_semaphores.data());
            immediate_dispatch.numSignalSyncObjects  = static_cast<uint32_t>(frame.signal_semaphores.size());
            dispatch.header.pNext                     = reinterpret_cast<AnfStructHeader*>(&immediate_dispatch);
        }

        if (m_anf.DispatchTechnique(m_sr_technique, &dispatch) != ANF_RESULT_SUCCESS)
        {
            return AnfBackendResult::ERROR_ANFEROR;
        }

        return AnfBackendResult::SUCCESS;
    }

    // -------------------------------------------------------------------------
    // DispatchFg
    // -------------------------------------------------------------------------

    AnfBackendResult AnfSdkBackend::DispatchFg(
        const AnfFgFrameParams& frame,
        const AnfFgResourcesVk& resources)
    {
        if (!m_is_valid || m_instance == nullptr || m_fg_technique == nullptr)
        {
            return AnfBackendResult::ERROR_NOT_INITIALIZED;
        }

        if (!m_dispatch_immediate && frame.cmd == VK_NULL_HANDLE)
        {
            return AnfBackendResult::ERROR_INVALID_PARAMETER;
        }

        AnfResourceParamDesc params[4] = {};
        BuildFgResourceParams(params, resources, m_fg_formats);

        AnfFGDispatch fg_dispatch{};
        fg_dispatch.header = MakeHeader(ANF_STYPE_FG_DISPATCH);

        AnfTechniqueDispatchInfo dispatch{};
        dispatch.header                      = MakeHeader(ANF_STYPE_TECHNIQUE_DISPATCH_INFO);
        dispatch.commandList                 = reinterpret_cast<AnfClientCommandList>(frame.cmd);
        dispatch.pResources                  = params;
        dispatch.numResources                = 4;
        dispatch.reset                       = frame.reset ? ANF_TRUE : ANF_FALSE;
        dispatch.pTechniqueGroupDispatchInfo = reinterpret_cast<const AnfStructHeader*>(&fg_dispatch);

        AnfTechniqueDispatchImmediateInfo immediate_dispatch{};
        if (m_dispatch_immediate)
        {
            immediate_dispatch.header                 = MakeHeader(ANF_STYPE_TECHNIQUE_DISPATCH_IMMEDIATE_INFO);
            immediate_dispatch.pWaitSyncObjects      = reinterpret_cast<const AnfClientSyncObject*>(frame.wait_semaphores.data());
            immediate_dispatch.numWaitSyncObjects    = static_cast<uint32_t>(frame.wait_semaphores.size());
            immediate_dispatch.pSignalSyncObjects    = reinterpret_cast<const AnfClientSyncObject*>(frame.signal_semaphores.data());
            immediate_dispatch.numSignalSyncObjects  = static_cast<uint32_t>(frame.signal_semaphores.size());
            dispatch.header.pNext                     = reinterpret_cast<AnfStructHeader*>(&immediate_dispatch);
        }

        if (m_anf.DispatchTechnique(m_fg_technique, &dispatch) != ANF_RESULT_SUCCESS)
        {
            return AnfBackendResult::ERROR_ANFEROR;
        }

        return AnfBackendResult::SUCCESS;
    }

} // namespace Anf
