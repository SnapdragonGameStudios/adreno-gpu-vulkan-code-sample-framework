// Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "anf_sample_ui.hpp"

class AnfGpuProfiler;

struct AnfRuntimeControlActions
{
    bool RequestSrReset = false;
    bool RequestFgReset = false;
};

struct AnfRuntimeControlsUiState
{
    bool                   SrSupported              = false;
    bool                   FgSupported              = false;
    bool                   ProfilingReady           = false;
    uint32_t               ActiveAnfFramesInFlight = 1;
    uint32_t               MaxFramesInFlightLimit   = 1;
    bool                   HasCalculatedFps         = false;
    double                 CalculatedRealFps        = 0.0;
    double                 CalculatedPresentedFps   = 0.0;
    const AnfGpuProfiler* Profiler                 = nullptr;
};

class AnfRuntimeControls
{
public:
    explicit AnfRuntimeControls(const AnfSampleUiBindings& bindings);

    void ApplyStartupProperties();
    AnfRuntimeControlActions UpdatePerFrame(float deltaSeconds);
    AnfRuntimeControlActions UpdateUi(const AnfRuntimeControlsUiState& uiState);

    bool IsSrRequested() const;
    bool IsFgRequested() const;
    bool ShouldUseFgOnSrMode() const;

    void ConfirmSrRanOnce();
    void ConfirmFgRanOnce();

private:
    void PollAdbProperties();

private:
    AnfSampleUiBindings m_Bindings{};
    bool                 m_CiSrEnabledLastFrame     = false;
    bool                 m_CiFgEnabledLastFrame     = false;
    bool                 m_CiSrAwaitingConfirmation = false;
    bool                 m_CiFgAwaitingConfirmation = false;
    float                m_AdbPollAccumSeconds      = 0.0f;
    bool                 m_StartupPropertiesApplied = false;
};
