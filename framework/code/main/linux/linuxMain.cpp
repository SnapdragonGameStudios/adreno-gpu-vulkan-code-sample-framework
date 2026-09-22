//============================================================================================================
//
//
//                  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

/// @file unixMain.cpp
/// @brief Unix 'main' entry point and event handler.
/// 
/// Implements Unix specific wrapping of frameworkApplicationBase.
//  There should be the minimum (possible) amount of code in here.

#include <string>
#include <fcntl.h>
#include <iostream>
#include <filesystem>

#include "system/os_common.h"

#include "main/frameworkApplicationBase.hpp"
#include "main/applicationEntrypoint.hpp"
#include "gui/gui.hpp"

#include <GLFW/glfw3.h>



// Based on config file loading, there is a window between the application being created,
// and it is intialized.  A WM_PAINT can come in at this point and will crash because
// nothing has been initialized.
bool gAppInitialized = false;

// Flag to indicate if the gui is currently 'eating' all the mouse input events (if false the events are passed on to the application)
static bool gGuiMouseActive = false;

// If the mouse is down and being handled by the application (not the gui)
static bool gAppMouseActive = false;

//------------------------------------------------------------------------------
/// Create Window for rendering into
//------------------------------------------------------------------------------
GLFWwindow* CreateWindow(int width, int height){
	GLFWwindow* window;

        // Initialize the library
        if (!glfwInit())
            return nullptr;

	// Tel GLFW to not attach to OpenGL
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        // Create a windowed mode window
        window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
        if (!window)
        {
            glfwTerminate();
            return nullptr;
        }

        /* Make the window's context current */
        glfwMakeContextCurrent(window);

	return window;
}

//------------------------------------------------------------------------------
// Destroy the Window
//------------------------------------------------------------------------------
void DestroyWindow(GLFWwindow* window)
{
    glfwDestroyWindow(window);
}



/// This is the main entry point of the Unix app.
/// initializes Vulkan and loops processing windows events (until the app is closed).
//-----------------------------------------------------------------------------
int main(int argc, const char*const* argv)
//-----------------------------------------------------------------------------
{
    // Check for a media directy (fatal to not have one).
    constexpr auto mediaPath = "build/Media";
    if (!std::filesystem::exists(mediaPath) || !std::filesystem::is_directory(mediaPath))
    {
        std::string errorMessage = "Cannot find 'build/Media' folder.\n  You're likely not running from the correct directory or are missing files from the Media folder (check this sample's README.md for instructions).\n";
        LOGE(errorMessage.c_str());
        return EXIT_FAILURE;
    }

    // Create the application
    auto* gpApplication = Application_ConstructApplication();

    // Do a very simple parse of the cmd line...
    std::string sConfigFilenameOverride;
    if (argc > 1)
        gpApplication->SetConfigFilename(argv[1]);

    // Load the config file
    // Need this here in order to get the window sizes
    gpApplication->LoadConfigFile();

    if (gSurfaceWidth == 0)
    {
        LOGI("gSurfaceWidth => %d", gRenderWidth);
        gSurfaceWidth = gRenderWidth;
    }
    if (gSurfaceHeight == 0)
    {
        LOGI("gSurfaceHeight => %d", gRenderHeight);
        gSurfaceHeight = gRenderHeight;
    }

    auto* pWindow = CreateWindow(gSurfaceWidth, gSurfaceHeight);

    // Initialize the application class
    if (!gpApplication->Initialize((uintptr_t)pWindow, (uintptr_t)0))    {
        LOGE("Application initialization failed!!");
        return EXIT_FAILURE;
    }

    gAppInitialized = true;

    if (!gpApplication->PostInitialize())
        return EXIT_FAILURE;

    // Loop on signals
    bool  fDone = false;
    while (!fDone && !glfwWindowShouldClose(pWindow))
    {
	glfwPollEvents();

	if (!gpApplication->Render())
	    fDone = true;

        OS_SleepMs(1);
    }
    // Release the application
    if (gpApplication)
    {
        gpApplication->Destroy();

        delete gpApplication;
        gpApplication = NULL;
    }

    DestroyWindow(pWindow);
    pWindow = nullptr;

    glfwTerminate();

    return EXIT_SUCCESS;
}