//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#include "anf_interface.hpp"

#include <cassert>

namespace Anf
{
    namespace
    {
        inline StatusCode ParseResult(AnfBackendResult result)
        {
            return (result == AnfBackendResult::SUCCESS) ? StatusCode::SUCCESS : StatusCode::ERROR_FAILED;
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

        inline VkExtent2D ComputeInputExtentForQuality(VkExtent2D output, AnfSRQualityMode mode)
        {
            // The sample renders at half the output resolution.
            (void)mode;
            return { std::max(1u, output.width / 2), std::max(1u, output.height / 2) };
        }

        std::optional<TextureFormat> RequireTextureFormat(const char* label, AnfFormat format)
        {
            const auto converted = AnfInterface::ToTextureFormat(format);
            if (!converted.has_value())
            {
                LOGE("ANF %s format is unresolved (AnfFormat=%d)", label, static_cast<int>(format));
            }
            return converted;
        }

        bool ValidateAnfOwnedTexture(
            const char* label,
            const TextureVulkan& texture,
            TextureFormat expectedFormat,
            VkImageUsageFlags requiredUsage,
            VkImageAspectFlags expectedAspectMask)
        {
            bool ok = true;
            if (!texture)
            {
                LOGE("ANF resource compatibility failed for %s: texture was not created", label);
                assert(false);
                return false;
            }

            if (expectedFormat != TextureFormat::UNDEFINED && texture.Format != expectedFormat)
            {
                LOGE("ANF resource compatibility failed for %s: format=%d expected=%d", label, static_cast<int>(texture.Format), static_cast<int>(expectedFormat));
                assert(texture.Format == expectedFormat);
                ok = false;
            }

            const auto& props = texture.GetProperties();
            if (requiredUsage != 0 && (props.Usage & requiredUsage) != requiredUsage)
            {
                LOGE("ANF resource compatibility failed for %s: usage=0x%08x required=0x%08x", label, props.Usage, requiredUsage);
                assert((props.Usage & requiredUsage) == requiredUsage);
                ok = false;
            }

            if (expectedAspectMask != 0 && (props.AspectMask & expectedAspectMask) != expectedAspectMask)
            {
                LOGE("ANF resource compatibility failed for %s: aspect=0x%08x expected=0x%08x", label, props.AspectMask, expectedAspectMask);
                assert((props.AspectMask & expectedAspectMask) == expectedAspectMask);
                ok = false;
            }

            if (props.Samples != VK_SAMPLE_COUNT_1_BIT)
            {
                LOGE("ANF resource compatibility failed for %s: samples=0x%08x expected=0x%08x", label, props.Samples, VK_SAMPLE_COUNT_1_BIT);
                assert(props.Samples == VK_SAMPLE_COUNT_1_BIT);
                ok = false;
            }

            if (props.Tiling != VK_IMAGE_TILING_OPTIMAL)
            {
                LOGE("ANF resource compatibility failed for %s: tiling=%d expected=%d", label, props.Tiling, VK_IMAGE_TILING_OPTIMAL);
                assert(props.Tiling == VK_IMAGE_TILING_OPTIMAL);
                ok = false;
            }

            return ok;
        }
    }

    AnfInterface::AnfInterface(Vulkan& vulkan)
        : m_vulkan(vulkan)
    {
    }

    // -------------------------------------------------------------------------
    // InitializeInstance
    // -------------------------------------------------------------------------

    StatusCode AnfInterface::InitializeInstance(
        uint32_t   num_frames_in_flight,
        VkExtent2D output_extent,
        bool       dispatch_immediate)
    {
        m_backend = std::make_unique<AnfSdkBackend>();
        m_num_frames_in_flight = num_frames_in_flight;
        m_output_extent        = output_extent;

        AnfInstanceConfig config{};
        config.instance          = m_vulkan.GetVulkanInstance();
        config.physical_device   = m_vulkan.m_VulkanGpu;
        config.device            = m_vulkan.m_VulkanDevice;
        config.queue_family_index = static_cast<uint32_t>(m_vulkan.m_VulkanQueues[Vulkan::eGraphicsQueue].QueueFamilyIndex);
        config.queue              = m_vulkan.m_VulkanQueues[Vulkan::eGraphicsQueue].Queue;
        config.dispatch_immediate = dispatch_immediate;

        const auto result = m_backend->InitializeInstance(config);
        if (result != AnfBackendResult::SUCCESS)
        {
            m_backend.reset();
            return StatusCode::ERROR_FAILED;
        }

        m_sr_output_format_candidates = m_backend->GetSrOutputFormatCandidates();
        m_fg_output_format_candidates = m_backend->GetFgOutputFormatCandidates();
        m_sr_output_format_index = 0;
        m_fg_output_format_index = 0;

        return StatusCode::SUCCESS;
    }

    // -------------------------------------------------------------------------
    // InitializeSR
    // -------------------------------------------------------------------------

    StatusCode AnfInterface::InitializeSR(AnfSRQualityMode quality_mode)
    {
        if (!m_backend || !m_backend->IsValid())
        {
            return StatusCode::ERROR_FAILED;
        }

        AnfSrConfig config{};
        config.output_extent = m_output_extent;
        config.input_extent  = ComputeInputExtentForQuality(m_output_extent, quality_mode);
        config.quality_mode  = quality_mode;
        config.max_in_flight = m_num_frames_in_flight;

        const auto& candidates = m_sr_output_format_candidates;
        if (candidates.empty())
        {
            LOGE("ANF SR has no output format candidates");
            return StatusCode::ERROR_FAILED;
        }

        for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index)
        {
            const AnfFormat candidate = candidates[candidate_index];
            if (!SelectSrOutputFormatCandidate(candidate_index, true))
            {
                continue;
            }

            LOGI("ANF SR init attempting output format candidate[%zu]=%s",
                 candidate_index,
                 AnfFormatName(candidate));

            const auto result = m_backend->InitializeSr(config);
            if (result == AnfBackendResult::SUCCESS)
            {
                return StatusCode::SUCCESS;
            }

            ReleaseSrOutputTextures();
            if (result == AnfBackendResult::ERROR_UNSUPPORTED)
            {
                return StatusCode::ERROR_NOT_IMPLEMENTED;
            }
        }

        return StatusCode::ERROR_FAILED;
    }

    // -------------------------------------------------------------------------
    // RecreateSR
    // -------------------------------------------------------------------------

    StatusCode AnfInterface::RecreateSR(AnfSRQualityMode quality_mode)
    {
        ReleaseSrOutputTextures();
        if (m_backend)
        {
            // Destroy only the SR technique; keep the instance and FG alive.
            // For now, full shutdown + re-init is the safe path.
            m_backend->Shutdown();
            m_backend.reset();
        }
        // Re-initialize the instance and SR technique.
        const auto inst_result = InitializeInstance(m_num_frames_in_flight, m_output_extent,
            m_backend ? m_backend->IsDispatchImmediate() : true);
        if (inst_result != StatusCode::SUCCESS)
            return inst_result;
        return InitializeSR(quality_mode);
    }

    // -------------------------------------------------------------------------
    // InitializeFG
    // -------------------------------------------------------------------------

    StatusCode AnfInterface::InitializeFG(VkExtent2D extent)
    {
        if (!m_backend || !m_backend->IsValid())
        {
            return StatusCode::ERROR_FAILED;
        }

        if (extent.width == 0 || extent.height == 0)
        {
            extent = m_output_extent;
        }
        m_fg_output_extent = extent;

        AnfFgConfig config{};
        config.extent       = extent;
        config.max_in_flight = m_num_frames_in_flight;

        const auto& candidates = m_fg_output_format_candidates;
        if (candidates.empty())
        {
            LOGE("ANF FG has no output format candidates");
            return StatusCode::ERROR_FAILED;
        }

        for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index)
        {
            const AnfFormat candidate = candidates[candidate_index];
            if (!SelectFgOutputFormatCandidate(candidate_index, true))
            {
                continue;
            }

            LOGI("ANF FG init attempting output format candidate[%zu]=%s",
                 candidate_index,
                 AnfFormatName(candidate));

            const auto result = m_backend->InitializeFg(config);
            if (result == AnfBackendResult::SUCCESS)
            {
                return StatusCode::SUCCESS;
            }

            ReleaseFgOutputTextures();
            if (result == AnfBackendResult::ERROR_UNSUPPORTED)
            {
                return StatusCode::ERROR_NOT_IMPLEMENTED;
            }
        }

        return StatusCode::ERROR_FAILED;
    }

    // -------------------------------------------------------------------------
    // Release
    // -------------------------------------------------------------------------

    StatusCode AnfInterface::Release()
    {
        if (m_backend)
        {
            m_backend->Shutdown();
            m_backend.reset();
        }

        ReleaseSrOutputTextures();
        ReleaseFgOutputTextures();
        m_num_frames_in_flight = 0;
        m_output_extent        = {};
        m_fg_output_extent     = {};
        m_sr_output_format_candidates.clear();
        m_fg_output_format_candidates.clear();
        m_sr_output_format_index = 0;
        m_fg_output_format_index = 0;
        return StatusCode::SUCCESS;
    }

    void AnfInterface::AbandonSdkForAndroidTeardown()
    {
        ReleaseSrOutputTextures();
        ReleaseFgOutputTextures();
        m_backend.release();
        m_num_frames_in_flight = 0;
        m_output_extent        = {};
        m_fg_output_extent     = {};
        m_sr_output_format_candidates.clear();
        m_fg_output_format_candidates.clear();
        m_sr_output_format_index = 0;
        m_fg_output_format_index = 0;
    }

    // -------------------------------------------------------------------------
    // IsSrValid / IsFgValid / IsDispatchImmediate
    // -------------------------------------------------------------------------

    bool AnfInterface::IsSrValid() const
    {
        return m_backend && m_backend->IsSrValid();
    }

    bool AnfInterface::IsFgValid() const
    {
        return m_backend && m_backend->IsFgValid() && m_fg_output_targets.size() > 0;
    }

    bool AnfInterface::IsDispatchImmediate() const
    {
        return !m_backend || m_backend->IsDispatchImmediate();
    }

    AnfSrFormats AnfInterface::GetSrFormats() const
    {
        return m_backend ? m_backend->GetSrFormats() : AnfSrFormats{};
    }

    AnfFgFormats AnfInterface::GetFgFormats() const
    {
        return m_backend ? m_backend->GetFgFormats() : AnfFgFormats{};
    }

    AnfSceneFormats AnfInterface::GetSceneFormats() const
    {
        return m_backend ? m_backend->GetSceneFormats() : AnfSceneFormats{};
    }

    uint32_t AnfInterface::GetSupportedSrQualityModes() const
    {
        return m_backend ? m_backend->GetCaps().sr_quality_modes : 0;
    }

    std::optional<TextureFormat> AnfInterface::ToTextureFormat(AnfFormat format)
    {
        switch (format)
        {
            case ANF_FORMAT_R8G8B8A8_UNORM:      return TextureFormat::R8G8B8A8_UNORM;
            case ANF_FORMAT_B10G11R11_UFLOAT:    return TextureFormat::B10G11R11_UFLOAT_PACK32;
            case ANF_FORMAT_D32_FLOAT:           return TextureFormat::D32_SFLOAT;
            case ANF_FORMAT_D24S8_UNORM:         return TextureFormat::D24_UNORM_S8_UINT;
            case ANF_FORMAT_R16G16_FLOAT:        return TextureFormat::R16G16_SFLOAT;
            case ANF_FORMAT_R16G16B16A16_FLOAT:  return TextureFormat::R16G16B16A16_SFLOAT;
            default:                              return std::nullopt;
        }
    }

    // -------------------------------------------------------------------------
    // GetSrOutputTexture / GetFgOutputTexture
    // -------------------------------------------------------------------------

    std::pair<TextureVulkan*, VkImageLayout> AnfInterface::GetSrOutputTexture(uint32_t frame_index)
    {
        if (frame_index >= m_sr_output_targets.size())
        {
            return { nullptr, VK_IMAGE_LAYOUT_UNDEFINED };
        }
        return { &m_sr_output_targets[frame_index], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    }

    std::pair<TextureVulkan*, VkImageLayout> AnfInterface::GetFgOutputTexture(uint32_t frame_index)
    {
        if (frame_index >= m_fg_output_targets.size())
        {
            return { nullptr, VK_IMAGE_LAYOUT_UNDEFINED };
        }
        return { &m_fg_output_targets[frame_index], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    }

    // -------------------------------------------------------------------------
    // RenderSR
    // -------------------------------------------------------------------------

    StatusCode AnfInterface::RenderSR(
        uint32_t                     frame_index,
        VkCommandBuffer              cmd,
        UpscalingInitData            props,
        std::span<const VkSemaphore> wait_semaphores,
        std::span<const VkSemaphore> signal_semaphores,
        uint32_t                     queue_family_index,
        VkQueue                      queue,
        bool                         reset)
    {
        if (!IsSrValid())
        {
            return StatusCode::ERROR_FAILED;
        }

        const auto& [out_tex, out_layout] = GetSrOutputTexture(frame_index);
        if (!out_tex || !props.ColorImage || !props.DepthImage || !props.MotionVectorImage)
        {
            return StatusCode::ERROR_FAILED;
        }

        AnfSrFrameParams frame{};
        frame.cmd                        = cmd;
        frame.reset                      = reset;
        frame.jitter_x                   = props.JitterOffset.x;
        frame.jitter_y                   = props.JitterOffset.y;
        frame.wait_semaphores            = wait_semaphores;
        frame.signal_semaphores          = signal_semaphores;
        frame.temp_queue_family_index_vk = queue_family_index;
        frame.temp_queue_vk              = queue;

        AnfSrResourcesVk res{};
        res.input_color        = GetVkImageFromTexture(*props.ColorImage);
        res.input_color_ext    = GetExtentFromTexture(*props.ColorImage);
        res.input_color_layout = props.ColorLayout;

        res.depth              = GetVkImageFromTexture(*props.DepthImage);
        res.depth_ext          = GetExtentFromTexture(*props.DepthImage);
        res.depth_layout       = props.DepthLayout;

        res.motion_vectors     = GetVkImageFromTexture(*props.MotionVectorImage);
        res.motion_ext         = GetExtentFromTexture(*props.MotionVectorImage);
        res.motion_layout      = props.MotionVectorLayout;

        res.output_color       = GetVkImageFromTexture(*out_tex);
        res.output_ext         = GetExtentFromTexture(*out_tex);
        res.output_layout      = out_layout;

        AnfBackendResult dispatch_result = m_backend->DispatchSr(frame, res);
        if (dispatch_result == AnfBackendResult::SUCCESS)
        {
            return StatusCode::SUCCESS;
        }

        // If SR rejects the currently selected output format during dispatch, try
        // the next output format candidate and retry once per candidate.
        if (dispatch_result == AnfBackendResult::ERROR_ANFEROR)
        {
            const bool can_retry_now = m_backend->IsDispatchImmediate() && frame.cmd == VK_NULL_HANDLE;

            for (size_t candidate_index = m_sr_output_format_index + 1;
                 candidate_index < m_sr_output_format_candidates.size();
                 ++candidate_index)
            {
                if (!SelectSrOutputFormatCandidate(candidate_index, true))
                {
                    continue;
                }

                LOGW("ANF SR dispatch switching to output format candidate[%zu]=%s",
                     candidate_index,
                     AnfFormatName(m_sr_output_format_candidates[candidate_index]));

                if (!can_retry_now)
                {
                    return StatusCode::ERROR_FAILED;
                }

                const auto& [retry_out_tex, retry_out_layout] = GetSrOutputTexture(frame_index);
                if (!retry_out_tex)
                {
                    return StatusCode::ERROR_FAILED;
                }
                res.output_color  = GetVkImageFromTexture(*retry_out_tex);
                res.output_ext    = GetExtentFromTexture(*retry_out_tex);
                res.output_layout = retry_out_layout;

                dispatch_result = m_backend->DispatchSr(frame, res);
                if (dispatch_result == AnfBackendResult::SUCCESS)
                {
                    return StatusCode::SUCCESS;
                }
            }
        }

        return ParseResult(dispatch_result);
    }

    // -------------------------------------------------------------------------
    // RenderFG
    // -------------------------------------------------------------------------

    StatusCode AnfInterface::RenderFG(
        uint32_t                     frame_index,
        VkCommandBuffer              cmd,
        FrameGenInitData             props,
        std::span<const VkSemaphore> wait_semaphores,
        std::span<const VkSemaphore> signal_semaphores,
        uint32_t                     queue_family_index,
        VkQueue                      queue,
        bool                         reset)
    {
        if (!IsFgValid())
        {
            return StatusCode::ERROR_FAILED;
        }

        const auto& [out_tex, out_layout] = GetFgOutputTexture(frame_index);
        if (!out_tex || !props.ColorImage || !props.DepthImage || !props.MotionVectorImage)
        {
            return StatusCode::ERROR_FAILED;
        }

        const VkExtent2D color_extent  = GetExtentFromTexture(*props.ColorImage);
        const VkExtent2D depth_extent  = GetExtentFromTexture(*props.DepthImage);
        const VkExtent2D motion_extent = GetExtentFromTexture(*props.MotionVectorImage);
        const VkExtent2D output_extent = GetExtentFromTexture(*out_tex);
        if (color_extent.width  != depth_extent.width  || color_extent.height  != depth_extent.height ||
            color_extent.width  != motion_extent.width || color_extent.height  != motion_extent.height ||
            color_extent.width  != output_extent.width || color_extent.height  != output_extent.height)
        {
            LOGE("ANF FG extent mismatch: color=%ux%u depth=%ux%u motion=%ux%u output=%ux%u",
                 color_extent.width, color_extent.height,
                 depth_extent.width, depth_extent.height,
                 motion_extent.width, motion_extent.height,
                 output_extent.width, output_extent.height);
            return StatusCode::ERROR_FAILED;
        }

        AnfFgFrameParams frame{};
        frame.cmd                        = cmd;
        frame.reset                      = reset;
        frame.wait_semaphores            = wait_semaphores;
        frame.signal_semaphores          = signal_semaphores;
        frame.temp_queue_family_index_vk = queue_family_index;
        frame.temp_queue_vk              = queue;

        AnfFgResourcesVk res{};
        res.input_color        = GetVkImageFromTexture(*props.ColorImage);
        res.input_color_ext    = color_extent;
        res.input_color_layout = props.ColorLayout;

        res.depth              = GetVkImageFromTexture(*props.DepthImage);
        res.depth_ext          = depth_extent;
        res.depth_layout       = props.DepthLayout;

        res.motion_vectors     = GetVkImageFromTexture(*props.MotionVectorImage);
        res.motion_ext         = motion_extent;
        res.motion_layout      = props.MotionVectorLayout;

        res.output_color       = GetVkImageFromTexture(*out_tex);
        res.output_ext         = output_extent;
        res.output_layout      = out_layout;

        AnfBackendResult dispatch_result = m_backend->DispatchFg(frame, res);
        if (dispatch_result == AnfBackendResult::SUCCESS)
        {
            return StatusCode::SUCCESS;
        }

        if (dispatch_result == AnfBackendResult::ERROR_ANFEROR)
        {
            const bool can_retry_now = m_backend->IsDispatchImmediate() && frame.cmd == VK_NULL_HANDLE;

            for (size_t candidate_index = m_fg_output_format_index + 1;
                 candidate_index < m_fg_output_format_candidates.size();
                 ++candidate_index)
            {
                if (!SelectFgOutputFormatCandidate(candidate_index, true))
                {
                    continue;
                }

                LOGW("ANF FG dispatch switching to output format candidate[%zu]=%s",
                     candidate_index,
                     AnfFormatName(m_fg_output_format_candidates[candidate_index]));

                if (!can_retry_now)
                {
                    return StatusCode::ERROR_FAILED;
                }

                const auto& [retry_out_tex, retry_out_layout] = GetFgOutputTexture(frame_index);
                if (!retry_out_tex)
                {
                    return StatusCode::ERROR_FAILED;
                }
                res.output_color  = GetVkImageFromTexture(*retry_out_tex);
                res.output_ext    = GetExtentFromTexture(*retry_out_tex);
                res.output_layout = retry_out_layout;

                dispatch_result = m_backend->DispatchFg(frame, res);
                if (dispatch_result == AnfBackendResult::SUCCESS)
                {
                    return StatusCode::SUCCESS;
                }
            }
        }

        return ParseResult(dispatch_result);
    }

    // -------------------------------------------------------------------------
    // Texture allocation helpers
    // -------------------------------------------------------------------------

    bool AnfInterface::SelectSrOutputFormatCandidate(size_t candidate_index, bool allocate_textures)
    {
        if (!m_backend || candidate_index >= m_sr_output_format_candidates.size())
        {
            return false;
        }

        const size_t previous_index = m_sr_output_format_index;
        const AnfFormat candidate = m_sr_output_format_candidates[candidate_index];
        if (!m_backend->SetSrOutputFormat(candidate))
        {
            return false;
        }

        if (allocate_textures && !AllocateSrOutputTextures(m_num_frames_in_flight, m_output_extent))
        {
            if (previous_index < m_sr_output_format_candidates.size())
            {
                m_backend->SetSrOutputFormat(m_sr_output_format_candidates[previous_index]);
                AllocateSrOutputTextures(m_num_frames_in_flight, m_output_extent);
            }
            return false;
        }

        m_sr_output_format_index = candidate_index;
        return true;
    }

    bool AnfInterface::SelectFgOutputFormatCandidate(size_t candidate_index, bool allocate_textures)
    {
        if (!m_backend || candidate_index >= m_fg_output_format_candidates.size())
        {
            return false;
        }

        const size_t previous_index = m_fg_output_format_index;
        const AnfFormat candidate = m_fg_output_format_candidates[candidate_index];
        if (!m_backend->SetFgOutputFormat(candidate))
        {
            return false;
        }

        if (allocate_textures && !AllocateFgOutputTextures(m_num_frames_in_flight, m_fg_output_extent))
        {
            if (previous_index < m_fg_output_format_candidates.size())
            {
                m_backend->SetFgOutputFormat(m_fg_output_format_candidates[previous_index]);
                AllocateFgOutputTextures(m_num_frames_in_flight, m_fg_output_extent);
            }
            return false;
        }

        m_fg_output_format_index = candidate_index;
        return true;
    }

    bool AnfInterface::AllocateSrOutputTextures(uint32_t num_frames_in_flight, VkExtent2D output_extent)
    {
        ReleaseSrOutputTextures();
        m_sr_output_targets.resize(num_frames_in_flight);
        const auto formats = GetSrFormats();
        const auto output_format_opt = RequireTextureFormat("SR output", formats.output_color_format);
        if (!output_format_opt.has_value())
        {
            return false;
        }
        const TextureFormat output_format = *output_format_opt;

        for (uint32_t i = 0; i < num_frames_in_flight; ++i)
        {
            CreateTexObjectInfo info{};
            info.uiWidth  = output_extent.width;
            info.uiHeight = output_extent.height;
            info.Format   = output_format;
            info.TexType  = TT_COMPUTE_TARGET;  // SR output requires VK_IMAGE_USAGE_STORAGE_BIT for compute write
            info.pName    = "ANF SR Output";

            m_sr_output_targets[i] = std::move(CreateTextureObject<Vulkan>(m_vulkan, info));
            if (!ValidateAnfOwnedTexture("SR output color", m_sr_output_targets[i], output_format, formats.output_color_usage, VK_IMAGE_ASPECT_COLOR_BIT))
                return false;
        }

        return true;
    }

    bool AnfInterface::AllocateFgOutputTextures(uint32_t num_frames_in_flight, VkExtent2D extent)
    {
        ReleaseFgOutputTextures();
        m_fg_output_targets.resize(num_frames_in_flight);
        const auto formats = GetFgFormats();
        const auto output_format_opt = RequireTextureFormat("FG output", formats.output_color_format);
        if (!output_format_opt.has_value())
        {
            return false;
        }
        const TextureFormat output_format = *output_format_opt;

        for (uint32_t i = 0; i < num_frames_in_flight; ++i)
        {
            CreateTexObjectInfo info{};
            info.uiWidth  = extent.width;
            info.uiHeight = extent.height;
            info.Format   = output_format;
            info.TexType  = TT_COMPUTE_TARGET;              // FG output requires STORAGE usage
            info.pName    = "ANF FG Output";

            m_fg_output_targets[i] = std::move(CreateTextureObject<Vulkan>(m_vulkan, info));
            if (!ValidateAnfOwnedTexture("FG output color", m_fg_output_targets[i], output_format, formats.output_color_usage, VK_IMAGE_ASPECT_COLOR_BIT))
                return false;
        }

        return true;
    }

    void AnfInterface::ReleaseSrOutputTextures()
    {
        for (auto& rt : m_sr_output_targets)
        {
            if (rt) rt.Release(&m_vulkan);
        }
        m_sr_output_targets.clear();
    }

    void AnfInterface::ReleaseFgOutputTextures()
    {
        for (auto& rt : m_fg_output_targets)
        {
            if (rt) rt.Release(&m_vulkan);
        }
        m_fg_output_targets.clear();
    }

    VkImage AnfInterface::GetVkImageFromTexture(const TextureVulkan& tex)
    {
        return tex.GetVkImage();
    }

    VkExtent2D AnfInterface::GetExtentFromTexture(const TextureVulkan& tex)
    {
        return { tex.Width, tex.Height };
    }

} // namespace Anf
