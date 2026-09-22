//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "vulkan/vulkan.hpp"

// Framework
#include "main/applicationHelperBase.hpp"
#include "texture/vulkan/texture.hpp"
#include "vulkan/renderPass.hpp"

// ANF backend
#include "anf_sdk_backend.hpp"

namespace Anf
{
    enum class StatusCode : int32_t
    {
        ERROR_UNKNOWN         = std::numeric_limits<int32_t>::lowest(),
        ERROR_NOT_IMPLEMENTED = -404,
        ERROR_FAILED          = -1,
        SUCCESS               = 0,
        PARTIAL_SUCCESS       = 1,
    };

    // Resources passed to RenderSR each frame.
    struct UpscalingInitData
    {
        TextureVulkan* ColorImage         = nullptr;
        TextureVulkan* DepthImage         = nullptr;
        TextureVulkan* MotionVectorImage  = nullptr;
        VkImageLayout  ColorLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout  DepthLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout  MotionVectorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        glm::vec2      JitterOffset;
    };

    // Resources passed to RenderFG each frame.
    struct FrameGenInitData
    {
        TextureVulkan* ColorImage         = nullptr;
        TextureVulkan* DepthImage         = nullptr;
        TextureVulkan* MotionVectorImage  = nullptr;
        VkImageLayout  ColorLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout  DepthLayout        = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout  MotionVectorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    ////////////////////////////////////////////////////////////////////////////////
    // Class name: AnfInterface
    //
    // High-level wrapper around AnfSdkBackend. Owns per-frame output textures for
    // SR and FG and translates framework types into ANF SDK types.
    ////////////////////////////////////////////////////////////////////////////////
    class AnfInterface
    {
    public:
        AnfInterface(Vulkan& vulkan);
        ~AnfInterface() = default;

        /*
        * Creates the ANF instance and queries technique capabilities.
        * Must be called before InitializeSR() or InitializeFG().
        * @param num_frames_in_flight : Number of frames/buffers used by the app.
        * @param output_extent        : The output resolution used by SR and FG.
        * @param dispatch_immediate   : When true ANF submits work internally (immediate mode).
        *                              When false the caller provides a VkCommandBuffer and
        *                              handles submission (indirect mode).
        * @return StatusCode::SUCCESS on success.
        */
        StatusCode InitializeInstance(
            uint32_t   num_frames_in_flight,
            VkExtent2D output_extent,
            bool       dispatch_immediate = true);

        /*
        * Creates the SR technique and allocates per-frame SR output textures.
        * InitializeInstance() must be called first.
        * @param quality_mode : SR quality mode selected by the user.
        * @return StatusCode::SUCCESS on success, StatusCode::ERROR_NOT_IMPLEMENTED if SR is
        *         not supported by the SDK.
        */
        StatusCode InitializeSR(AnfSRQualityMode quality_mode);

        /*
        * Recreates SR technique and output textures (call after vkDeviceWaitIdle).
        */
        StatusCode RecreateSR(AnfSRQualityMode quality_mode);

        /*
        * Creates the FG technique and allocates per-frame FG output textures.
        * InitializeInstance() must be called first.
        * @return StatusCode::SUCCESS on success, StatusCode::ERROR_NOT_IMPLEMENTED if FG is
        *         unavailable on the selected SDK/device.
        */
        StatusCode InitializeFG(VkExtent2D extent = {});

        /*
        * Releases all ANF resources.
        */
        StatusCode Release();

        // Releases framework-owned output textures without calling the SDK teardown path.
        // Used during final Android teardown; SDK resources are left for process cleanup.
        void AbandonSdkForAndroidTeardown();

        /*
        * Dispatches SR.
        *
        * Immediate mode  (dispatch_immediate == true):
        *   - Pass VK_NULL_HANDLE for cmd.
        *   - Provide wait_semaphores / signal_semaphores; ANF handles submission.
        *
        * Indirect mode (dispatch_immediate == false):
        *   - Pass a valid, already-begun VkCommandBuffer for cmd.
        *   - wait_semaphores / signal_semaphores are ignored; the caller submits cmd.
        */
        StatusCode RenderSR(
            uint32_t                     frame_index,
            VkCommandBuffer              cmd,
            UpscalingInitData            props,
            std::span<const VkSemaphore> wait_semaphores,
            std::span<const VkSemaphore> signal_semaphores,
            uint32_t                     queue_family_index,
            VkQueue                      queue,
            bool                         reset);

        /*
        * Dispatches FG. Same immediate/indirect semantics as RenderSR.
        * The input color is typically the SR output for the current frame.
        */
        StatusCode RenderFG(
            uint32_t                     frame_index,
            VkCommandBuffer              cmd,
            FrameGenInitData             props,
            std::span<const VkSemaphore> wait_semaphores,
            std::span<const VkSemaphore> signal_semaphores,
            uint32_t                     queue_family_index,
            VkQueue                      queue,
            bool                         reset);

        bool IsSrValid() const;
        bool IsFgValid() const;

        // Returns the SR output texture and its expected layout for a given frame index.
        std::pair<TextureVulkan*, VkImageLayout> GetSrOutputTexture(uint32_t frame_index);

        // Returns the FG output texture and its expected layout for a given frame index.
        std::pair<TextureVulkan*, VkImageLayout> GetFgOutputTexture(uint32_t frame_index);

        // Returns the supported SR quality modes bitfield (bit index = AnfSRQualityMode value).
        uint32_t GetSupportedSrQualityModes() const;

        // Returns true when the backend is in immediate dispatch mode.
        bool IsDispatchImmediate() const;

        AnfSrFormats GetSrFormats() const;
        AnfFgFormats GetFgFormats() const;
        AnfSceneFormats GetSceneFormats() const;
        static std::optional<TextureFormat> ToTextureFormat(AnfFormat format);

    private:
        bool AllocateSrOutputTextures(uint32_t num_frames_in_flight, VkExtent2D output_extent);
        bool AllocateFgOutputTextures(uint32_t num_frames_in_flight, VkExtent2D extent);
        bool SelectSrOutputFormatCandidate(size_t candidate_index, bool allocate_textures);
        bool SelectFgOutputFormatCandidate(size_t candidate_index, bool allocate_textures);
        void ReleaseSrOutputTextures();
        void ReleaseFgOutputTextures();

        static VkImage    GetVkImageFromTexture(const TextureVulkan& tex);
        static VkExtent2D GetExtentFromTexture(const TextureVulkan& tex);

    private:
        Vulkan&                         m_vulkan;
        std::unique_ptr<AnfSdkBackend> m_backend;
        std::vector<Texture<Vulkan>>    m_sr_output_targets;
        std::vector<Texture<Vulkan>>    m_fg_output_targets;
        std::vector<AnfFormat>         m_sr_output_format_candidates;
        std::vector<AnfFormat>         m_fg_output_format_candidates;
        size_t                          m_sr_output_format_index = 0;
        size_t                          m_fg_output_format_index = 0;

        // Stored from InitializeInstance() so InitializeSR/FG can use them.
        uint32_t   m_num_frames_in_flight = 0;
        VkExtent2D m_output_extent        = {};
        VkExtent2D m_fg_output_extent     = {};
    };

} // namespace Anf
