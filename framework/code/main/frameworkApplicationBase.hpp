//============================================================================================================
//
//
//                  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include <memory>
#include <span>
#include "system/assetManager.hpp"
#include "system/glm_common.hpp"    // must be before config.h
#include "system/config.h"

// Forward declares
class GraphicsApiBase;
class Gui;

//#########################################################
// Config options - Begin
//#########################################################
// ************************************
// General Settings
// ************************************
EXTERN_VAR(uint32_t, gSurfaceWidth);
EXTERN_VAR(uint32_t, gSurfaceHeight);

EXTERN_VAR(uint32_t, gRenderWidth);
EXTERN_VAR(uint32_t, gRenderHeight);

EXTERN_VAR(uint32_t, gReflectMapWidth);
EXTERN_VAR(uint32_t, gReflectMapHeight);

EXTERN_VAR(uint32_t, gShadowMapWidth);
EXTERN_VAR(uint32_t, gShadowMapHeight);

EXTERN_VAR(uint32_t, gHudRenderWidth);
EXTERN_VAR(uint32_t, gHudRenderHeight);

EXTERN_VAR(uint32_t, gFramesToRender);

EXTERN_VAR(bool,     gRunOnHLM);
EXTERN_VAR(int,      gHLMDumpFrame);
EXTERN_VAR( int,     gHLMDumpFrameCount);
EXTERN_VAR(char*,    gHLMDumpFile);

// Vulkan binding locations for the 'default' layouts when using InitOneLayout
#define SHADER_VERT_UBO_LOCATION            0
#define SHADER_FRAG_UBO_LOCATION            1
// Texture Locations
#define SHADER_BASE_TEXTURE_LOC             2

class ApplicationConfig
{
public:
    enum class Requirement
    {
        Optional = 0,
        Required,
        NotUsed,
    };

    Requirement assetFolderRequirement = Requirement::Required;
    Requirement configFileRequirement =  Requirement::Optional;
};

//#########################################################
// Config options - End
//#########################################################

/// Base class from which Applications must derive.
/// Application should implement a function "Application_ConstructApplication" that returns a new pointer to their main implemention class that derives from this class.
class FrameworkApplicationBase
{
    FrameworkApplicationBase& operator=(const FrameworkApplicationBase&) = delete;
    FrameworkApplicationBase(const FrameworkApplicationBase&) = delete;

public:
    FrameworkApplicationBase();
    virtual ~FrameworkApplicationBase();

    /// Set the AAssetManager (on android, does nothing on windows).  Do before LoadConfigFile().
    void            SetAndroidAssetManager(AAssetManager* pAAssetManager);

    /// Set the Android ExternalFilesDir (on android, does nothing on windows).  This is the location where the app is 'allowed' to read/write files including the app_config.txt.  Do before LoadConfigFile().
    void            SetAndroidExternalFilesDir(const std::string& pAndroidExternalFilesDir);

    /// Override the (default) filename used by LoadConfigFile
    virtual void    SetConfigFilename(const std::string& filename);

    /// May be called before initialize (from main(), before window creation).
    /// @param configuration (if untouched, Application will be setup using defaults).
    virtual void    PreInitializeSetApplicationConfiguration(ApplicationConfig&);

    /// Load config file before Initialize()
    /// This will go to derived class that can then set global parameters
    virtual bool    LoadConfigFile();

    /// Initialize application.
    /// This is the main entry point for your Application code!
    virtual bool    Initialize(uintptr_t windowHandle, uintptr_t instanceHandle);

    /// Post-Initialize application.
    /// This is called immediately after Initialize and before first Render, should be lightweight (reset timers etc)
    virtual bool    PostInitialize();

    /// ReInitialize application.
    /// Will only be called if the app was initialized and the underlying platform requires the app be re-initialized (due to Vulkan window changing)
    virtual bool    ReInitialize(uintptr_t windowHandle, uintptr_t instanceHandle);

    /// Destroy application (opposite of Initialize, must get back to state that is re-Initializable)
    virtual void    Destroy();

    /// Called by framework whenever the screen (swap chain) size has changed.  This is the WINDOW size (not necissarily the buffer size etc), mouse/touch input is expected to be in this coordinate space)
    virtual bool    SetWindowSize( uint32_t width, uint32_t height );

    /// Return the WINDOW size (not necissarily the buffer size etc), mouse/touch input is expected to be in this coordinate space)
    std::pair<uint32_t, uint32_t> GetWindowSize() const { return {m_WindowWidth, m_WindowHeight}; };

    /// Application Main thread 'render' loop (eg ALooper_pollAll loop on Android)
    /// Called every frame.
    /// Application should derive from this and issue their renderring commands from this function.
    virtual void    Render(float frameTimeSeconds) = 0;

    /// Logging function for FPS (frames per second) that is periodically called and can be overridden for other profile/performance logging
    virtual void    LogFps();

    /// Keyboard Input 'down' event call (application should override to recieve these events)
    virtual void    KeyDownEvent(uint32_t key);
    /// Keyboard Input 'repeat' event call (application should override to recieve these events)
    virtual void    KeyRepeatEvent(uint32_t key);
    /// Keyboard Input 'up' event call (application should override to recieve these events)
    virtual void    KeyUpEvent(uint32_t key);
    /// Touch (or mouse) input 'down' event call (application should override to recieve these events)
    virtual void    TouchDownEvent(int iPointerID, float xPos, float yPos);
    /// Touch (or mouse) input 'move' event call (application should override to recieve these events)
    virtual void    TouchMoveEvent(int iPointerID, float xPos, float yPos);
    /// Touch (or mouse) input 'up' event call (application should override to recieve these events)
    virtual void    TouchUpEvent(int iPointerID, float xPos, float yPos);

    /// Main thread 'render' loop (eg ALooper_pollAll loop on Android)
    /// Wraps 'Render(float frameTimeSeconds)' with frame timers.  Applications should derive from Render(float frameTimeSeconds).
    /// @returns false if Render fails or is done (requesting to exit app).
    bool            Render();

    // Accessors
    GraphicsApiBase*GetGraphicsApiBase() const  { return m_gfxBase.get(); }
    Gui*            GetGui() const              { return m_Gui.get(); }
    uint32_t        GetFrameCount() const       { return m_FrameCount; }

public:
    // Frame timings
    bool                    m_LogFPS = true;
    float                   m_CurrentFPS = 0.0f;

    /// @brief Timestamp string filled in by cmake build process (time this application was build at).
    static const char* const sm_BuildTimestamp;
protected:
    std::unique_ptr<GraphicsApiBase> m_gfxBase;
    std::unique_ptr<Gui>    m_Gui;
    std::unique_ptr<AssetManager> m_AssetManager;
    std::string             m_ConfigFilename;

    uint32_t                m_WindowWidth = 0;              ///< Window width in pixels.  MAY not be the resolution of the render buffer or the rendering/backbuffer surface.  In an Android app MAY not be the full screeen device size.  DOES match the mouse/touch co-oordinates (mouse 0,0 is the edge of this window area)
    uint32_t                m_WindowHeight = 0;             ///< Window width in pixels.  MAY not be the resolution of the render buffer or the rendering/backbuffer surface.  In an Android app MAY not be the full screeen device size.  DOES match the mouse/touch co-oordinates (mouse 0,0 is the edge of this window area)
    float                   m_FpsEvaluateInterval = 1.0f;   // In seconds

private:
    uint64_t                m_LastUpdateTimeUS = 0;         // In microseconds.
    uint32_t                m_FrameCount = 0;               // Number of times Render called.

    uint32_t                m_LastFpsCalcTimeMS = 0;        // In milliseconds.
    uint32_t                m_FpsFrameCount = 0;
    uint32_t                m_LastFpsLogTimeMS = 0;
    float                   m_FixedFramerateCorrectionMs = 0.0f;
};
