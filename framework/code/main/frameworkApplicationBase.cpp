//============================================================================================================
//
//
//                  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "frameworkApplicationBase.hpp"

#include "gui/gui.hpp"
#include "graphicsApi/graphicsApiBase.hpp"
#include "system/os_common.h"

// Bring in the timestamp (and assign to a variable)
//#include "../../project/buildtimestamp.h"
//const char* const FrameworkApplicationBase::sm_BuildTimestamp = BUILD_TIMESTAMP;
const char* const FrameworkApplicationBase::sm_BuildTimestamp = "";


//#########################################################
// Config options - Begin
//#########################################################
// ************************************
// General Settings
// ************************************
VAR(uint32_t, gSurfaceWidth, 1280, kVariableNonpersistent);
VAR(uint32_t, gSurfaceHeight, 720, kVariableNonpersistent);

VAR(uint32_t, gRenderWidth, 1280, kVariableNonpersistent);
VAR(uint32_t, gRenderHeight, 720, kVariableNonpersistent);

VAR(uint32_t, gReflectMapWidth, 1280/2, kVariableNonpersistent);
VAR(uint32_t, gReflectMapHeight, 720/2, kVariableNonpersistent);

VAR(uint32_t, gShadowMapWidth, 1024, kVariableNonpersistent);
VAR(uint32_t, gShadowMapHeight, 1024, kVariableNonpersistent);

VAR(uint32_t, gHudRenderWidth, 1280, kVariableNonpersistent);
VAR(uint32_t, gHudRenderHeight, 720, kVariableNonpersistent);

VAR(float,    gFixedFrameRate, 0.0f, kVariableNonpersistent);

VAR(uint32_t, gFramesToRender, 0/*0=loop forever*/, kVariableNonpersistent);
VAR(bool,     gRunOnHLM, false, kVariableNonpersistent);
VAR(int,      gHLMDumpFrame, -1, kVariableNonpersistent);       // start dumping on this frame
VAR(int,      gHLMDumpFrameCount, 1, kVariableNonpersistent );  // dump for this number of frames
VAR(char*,    gHLMDumpFile, "output", kVariableNonpersistent);

VAR(bool,     gFifoPresentMode, false, kVariableNonpersistent); // enable to use FIFO present mode (locks app to refresh rate)


//#########################################################
// Config options - End
//#########################################################

//-----------------------------------------------------------------------------
FrameworkApplicationBase::FrameworkApplicationBase()
//-----------------------------------------------------------------------------
{
    m_AssetManager = std::make_unique<AssetManager>();
    m_ConfigFilename = "app_config.txt";

    // FPS and frametime Handling
    m_CurrentFPS = 0.0f;
    m_FpsFrameCount = 0;
    m_FrameCount = 0;
    m_LastUpdateTimeUS = OS_GetTimeUS();
    m_LastFpsCalcTimeMS = (uint32_t) (m_LastUpdateTimeUS / 1000);
    m_LastFpsLogTimeMS = m_LastFpsCalcTimeMS;

    m_WindowWidth = 0;
    m_WindowHeight = 0;

    //LOGI("Application build time: %s", sm_BuildTimestamp);
}

//-----------------------------------------------------------------------------
FrameworkApplicationBase::~FrameworkApplicationBase()
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
void FrameworkApplicationBase::SetAndroidAssetManager(AAssetManager* pAAssetManager)
//-----------------------------------------------------------------------------
{
    m_AssetManager->SetAAssetManager(pAAssetManager);
}

//-----------------------------------------------------------------------------
void FrameworkApplicationBase::SetAndroidExternalFilesDir(const std::string& androidExternalFilesDir)
//-----------------------------------------------------------------------------
{
    m_AssetManager->SetAndroidExternalFilesDir(androidExternalFilesDir);
}

//-----------------------------------------------------------------------------
void FrameworkApplicationBase::SetConfigFilename(const std::string& filename)
//-----------------------------------------------------------------------------
{
    m_ConfigFilename = filename;
}

//-----------------------------------------------------------------------------
bool FrameworkApplicationBase::LoadConfigFile()
//-----------------------------------------------------------------------------
{
    const std::string ConfigFileFallbackPath = std::string("build\\Media\\") + m_ConfigFilename;
    LOGI("Loading Configuration File: %s", m_ConfigFilename.c_str());
    std::string configFile;
    if (!m_AssetManager->LoadFileIntoMemory(m_ConfigFilename, configFile ) &&
        !m_AssetManager->LoadFileIntoMemory(ConfigFileFallbackPath, configFile))
    {
        LOGE("Error loading Configuration file: %s (and fallback: %s)", m_ConfigFilename.c_str(), ConfigFileFallbackPath.c_str());
    }
    else
    {
        LOGI("Parsing Variable Buffer: ");
        // LogInfo("%s", pConfigFile);
        LoadVariableBuffer(configFile.c_str());
    }

    return true;
}

//-----------------------------------------------------------------------------
void FrameworkApplicationBase::PreInitializeSetApplicationConfiguration(ApplicationConfig&)
//-----------------------------------------------------------------------------
{
    return;
}

//-----------------------------------------------------------------------------
bool FrameworkApplicationBase::Initialize(uintptr_t windowHandle, uintptr_t instanceHandle)
//-----------------------------------------------------------------------------
{
    return true;
}

//-----------------------------------------------------------------------------
bool FrameworkApplicationBase::PostInitialize()
//-----------------------------------------------------------------------------
{
    m_LastUpdateTimeUS = OS_GetTimeUS();
    return true;
}

//-----------------------------------------------------------------------------
bool FrameworkApplicationBase::ReInitialize(uintptr_t windowHandle, uintptr_t instanceHandle)
//-----------------------------------------------------------------------------
{
    return true;
}

//-----------------------------------------------------------------------------
void FrameworkApplicationBase::Destroy()
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
bool FrameworkApplicationBase::SetWindowSize(uint32_t width, uint32_t height)
//-----------------------------------------------------------------------------
{
    LOGI("SetWindowSize: window size %u x %u", width, height);

    m_WindowWidth = width;
    m_WindowHeight = height;

    return true;
}

//-----------------------------------------------------------------------------
bool FrameworkApplicationBase::Render()
//-----------------------------------------------------------------------------
{
    //
    // Gather pre-render timing statistics
    //
    m_FpsFrameCount++;

    uint64_t TimeNowUS = OS_GetTimeUS();
    uint32_t TimeNowMS = (uint32_t) (TimeNowUS / 1000);
    {
        float DiffTime = (float)(TimeNowMS - m_LastFpsCalcTimeMS) * 0.001f;     // Time in seconds
        if (DiffTime > m_FpsEvaluateInterval)
        {
            m_CurrentFPS = (float)m_FpsFrameCount / DiffTime;

            m_FpsFrameCount = 0;
            m_LastFpsCalcTimeMS = TimeNowMS;
        }
    }

    float fltDiffTimeMs = (float)(TimeNowUS - m_LastUpdateTimeUS) * 0.001f;

    if (gFixedFrameRate > 0.0f)
    {
        // Fixing the frame rate, add sleeps to keep the app running at the requested interval.
        // Frame time step reported to app Update etc is calculated directly from the requested fps (no longer the actual frame time).
        float targetFrameTimeMs = 1000.0f / gFixedFrameRate;
        int extraSleep = int(targetFrameTimeMs - fltDiffTimeMs + m_FixedFramerateCorrectionMs);
        if (extraSleep > 0)
        {
            OS_SleepMs((uint32_t) extraSleep);
        }

        // Calculate how close the extra sleep got us to the target fps.
        TimeNowUS = OS_GetTimeUS();
        fltDiffTimeMs = (float)(TimeNowUS - m_LastUpdateTimeUS) * 0.001f;
        // Apply a correction to the next frame (limited to 2x the target frame time).
        m_FixedFramerateCorrectionMs += targetFrameTimeMs - fltDiffTimeMs;
        if (m_FixedFramerateCorrectionMs < -targetFrameTimeMs * 2.0f)
            m_FixedFramerateCorrectionMs = -targetFrameTimeMs * 2.0f;
        else if (m_FixedFramerateCorrectionMs > targetFrameTimeMs * 2.0f)
            m_FixedFramerateCorrectionMs = targetFrameTimeMs * 2.0f;
        m_FixedFramerateCorrectionMs = m_FixedFramerateCorrectionMs < 1000.0f ? (m_FixedFramerateCorrectionMs > -1000.0f ? m_FixedFramerateCorrectionMs : -1000.0f) : 1000.0f;

        // Force the 'frame time' seen by the application to be locked to the fixed fps
        fltDiffTimeMs = targetFrameTimeMs;
    }

    float fltDiffTime = fltDiffTimeMs * 0.001f;     // Time in seconds
    m_LastUpdateTimeUS = TimeNowUS;


    //
    // Call in to the derived application class
    //
    Render(fltDiffTime);

    //
    // Gather post render timing statistics
    //
    if (m_LogFPS)
    {
        // Log FPS is configured
        uint32_t TimeNow = OS_GetTimeMS();
        uint32_t DiffTime = TimeNow - m_LastFpsLogTimeMS;
        if (DiffTime > 1000)
        {
            LogFps();
            m_LastFpsLogTimeMS = TimeNow;
        }
    }

    m_FrameCount++;

    //
    // Determine if app should exit (reached requested number of frames)
    //
    if (gFramesToRender != 0 && m_FrameCount >= gFramesToRender)
    {
        m_gfxBase->WaitUntilIdle();
        return false;   // Exit
    }
    return true;        // Ok
}

//-----------------------------------------------------------------------------
void FrameworkApplicationBase::LogFps()
//-----------------------------------------------------------------------------
{
    LOGI("FPS: %0.2f", m_CurrentFPS);
}

//-----------------------------------------------------------------------------
void FrameworkApplicationBase::KeyDownEvent(uint32_t key)
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
void FrameworkApplicationBase::KeyRepeatEvent(uint32_t key)
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
void FrameworkApplicationBase::KeyUpEvent(uint32_t key)
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
void FrameworkApplicationBase::TouchDownEvent(int iPointerID, float xPos, float yPos)
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
void FrameworkApplicationBase::TouchMoveEvent(int iPointerID, float xPos, float yPos)
//-----------------------------------------------------------------------------
{
}

//-----------------------------------------------------------------------------
void FrameworkApplicationBase::TouchUpEvent(int iPointerID, float xPos, float yPos)
//-----------------------------------------------------------------------------
{
}
