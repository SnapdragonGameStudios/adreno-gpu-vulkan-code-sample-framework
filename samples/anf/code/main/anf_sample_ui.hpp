// Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>

class AnfGpuProfiler;

struct AnfSampleUiModel
{
    bool                  SrSupported               = false;
    bool                  FgSupported               = false;
    bool                  ProfilingReady            = false;
    uint32_t              ActiveAnfFramesInFlight  = 1;
    uint32_t              MaxFramesInFlightLimit    = 1;
    bool                  HasCalculatedFps          = false;
    double                CalculatedRealFps         = 0.0;
    double                CalculatedPresentedFps    = 0.0;
    const AnfGpuProfiler* Profiler                 = nullptr;
};

struct AnfSampleUiBindings
{
    bool* DispatchImmediate = nullptr;
    int*  MaxFramesInFlight = nullptr;
    bool* UpscalingEnabled  = nullptr;
    bool* FrameGenerationEnabled = nullptr;
    bool* IgnoreAdbRuntimeCommands = nullptr;
    bool* UseInverseDepth  = nullptr;
    int*  AnfSrQualityMode = nullptr;
    bool* AnimationPaused   = nullptr;
    bool* DebugForceZeroMv  = nullptr;
    bool* DebugForceDisableJitter = nullptr;
    bool* DebugForceWaitForIdle   = nullptr;
};

struct AnfSampleUiResult
{
    bool RequestSrReset = false;
    bool RequestFgReset = false;
};

AnfSampleUiResult DrawAnfSampleUi(const AnfSampleUiModel& model, const AnfSampleUiBindings& bindings);
