// Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "anf_runtime_controls.hpp"

#include "anf_gpu_profiler.hpp"
#include "system/os_common.h"

#include <algorithm>
#include <cctype>

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

namespace
{
#if defined(__ANDROID__)
    bool EqualsCaseInsensitive(const char* lhs, const char* rhs)
    {
        if (!lhs || !rhs)
            return lhs == rhs;

        while (*lhs != '\0' && *rhs != '\0')
        {
            if (std::tolower(static_cast<unsigned char>(*lhs)) !=
                std::tolower(static_cast<unsigned char>(*rhs)))
            {
                return false;
            }

            ++lhs;
            ++rhs;
        }

        return *lhs == '\0' && *rhs == '\0';
    }

    bool ParseBoolPropertyValue(const char* value)
    {
        if (!value || value[0] == '\0')
            return false;

        if (value[1] == '\0')
            return value[0] == '1';

        if (EqualsCaseInsensitive(value, "yes") || EqualsCaseInsensitive(value, "true") || EqualsCaseInsensitive(value, "on"))
            return true;
        if (EqualsCaseInsensitive(value, "no") || EqualsCaseInsensitive(value, "false") || EqualsCaseInsensitive(value, "off"))
            return false;
        return atoi(value) != 0;
    }
#endif
}

AnfRuntimeControls::AnfRuntimeControls(const AnfSampleUiBindings& bindings)
    : m_Bindings(bindings)
{
}

void AnfRuntimeControls::ApplyStartupProperties()
{
#if defined(__ANDROID__)
    if (m_StartupPropertiesApplied)
        return;

    m_StartupPropertiesApplied = true;
    if (!m_Bindings.DispatchImmediate)
        return;

    char value[PROP_VALUE_MAX] = {};
    if (__system_property_get("debug.anf.start_dispatch_immediate", value) > 0)
    {
        *m_Bindings.DispatchImmediate = ParseBoolPropertyValue(value);
        LOGI("ANF startup property applied: debug.anf.start_dispatch_immediate=%s -> dispatch=%s",
             value,
             *m_Bindings.DispatchImmediate ? "immediate" : "indirect");
    }
#endif
}

AnfRuntimeControlActions AnfRuntimeControls::UpdatePerFrame(float deltaSeconds)
{
    AnfRuntimeControlActions actions{};

#if defined(__ANDROID__)
    m_AdbPollAccumSeconds += std::max(0.0f, deltaSeconds);
    if (m_AdbPollAccumSeconds >= 1.0f)
    {
        m_AdbPollAccumSeconds = 0.0f;
        PollAdbProperties();
    }
#else
    (void)deltaSeconds;
#endif

    const bool srRequested = IsSrRequested();
    if (srRequested != m_CiSrEnabledLastFrame)
    {
        actions.RequestSrReset = srRequested;
        m_CiSrEnabledLastFrame = srRequested;
        m_CiSrAwaitingConfirmation = srRequested;
    }

    const bool fgRequested = IsFgRequested();
    if (fgRequested != m_CiFgEnabledLastFrame)
    {
        actions.RequestFgReset = fgRequested;
        m_CiFgEnabledLastFrame = fgRequested;
        m_CiFgAwaitingConfirmation = fgRequested;
    }

    return actions;
}

AnfRuntimeControlActions AnfRuntimeControls::UpdateUi(const AnfRuntimeControlsUiState& uiState)
{
    AnfRuntimeControlActions actions{};

    AnfSampleUiModel model{};
    model.SrSupported              = uiState.SrSupported;
    model.FgSupported              = uiState.FgSupported;
    model.ProfilingReady           = uiState.ProfilingReady;
    model.ActiveAnfFramesInFlight = uiState.ActiveAnfFramesInFlight;
    model.MaxFramesInFlightLimit   = uiState.MaxFramesInFlightLimit;
    model.HasCalculatedFps         = uiState.HasCalculatedFps;
    model.CalculatedRealFps        = uiState.CalculatedRealFps;
    model.CalculatedPresentedFps   = uiState.CalculatedPresentedFps;
    model.Profiler                 = uiState.Profiler;

    const auto uiResult = DrawAnfSampleUi(model, m_Bindings);
    actions.RequestSrReset = uiResult.RequestSrReset;
    actions.RequestFgReset = uiResult.RequestFgReset;
    return actions;
}

bool AnfRuntimeControls::IsSrRequested() const
{
    return m_Bindings.UpscalingEnabled && *m_Bindings.UpscalingEnabled;
}

bool AnfRuntimeControls::IsFgRequested() const
{
    return m_Bindings.FrameGenerationEnabled && *m_Bindings.FrameGenerationEnabled;
}

bool AnfRuntimeControls::ShouldUseFgOnSrMode() const
{
    return IsSrRequested() && IsFgRequested();
}

void AnfRuntimeControls::ConfirmSrRanOnce()
{
    if (m_CiSrAwaitingConfirmation)
    {
        LOGI("ANF_CI_SR_CONFIRMED");
        m_CiSrAwaitingConfirmation = false;
    }
}

void AnfRuntimeControls::ConfirmFgRanOnce()
{
    if (m_CiFgAwaitingConfirmation)
    {
        LOGI("ANF_CI_FG_CONFIRMED");
        m_CiFgAwaitingConfirmation = false;
    }
}

void AnfRuntimeControls::PollAdbProperties()
{
#if defined(__ANDROID__)
    if ((m_Bindings.IgnoreAdbRuntimeCommands && *m_Bindings.IgnoreAdbRuntimeCommands) ||
        !m_Bindings.UpscalingEnabled || !m_Bindings.FrameGenerationEnabled)
        return;

    char value[PROP_VALUE_MAX] = {};

    if (__system_property_get("debug.anf.dynamic_cmds", value) == 0 || value[0] != '1')
        return;

    if (__system_property_get("debug.anf.sr", value) > 0)
        *m_Bindings.UpscalingEnabled = (value[0] == '1');

    if (__system_property_get("debug.anf.fg", value) > 0)
        *m_Bindings.FrameGenerationEnabled = (value[0] == '1');
#endif
}
