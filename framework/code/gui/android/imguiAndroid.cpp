//============================================================================================================
//
//
//                  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "imguiAndroid.hpp"
#define NOMINMAX
#if defined(OS_ANDROID)
#include <android/native_window.h>
#endif
#include <imgui/imgui.h>


//
// Implementation of GuiImguiPlatform (for Android)
//

GuiImguiPlatform::GuiImguiPlatform()
{}

GuiImguiPlatform::~GuiImguiPlatform()
{}


bool GuiImguiPlatform::Initialize(uintptr_t windowHandle, TextureFormat renderFormat, uint32_t deviceWidth, uint32_t deviceHeight, uint32_t renderWidth, uint32_t renderHeight)
{
    if (!GuiImguiBase::Initialize(windowHandle, renderFormat, renderWidth, renderHeight))
    {
        return false;
    }

    // Android touch coordinates are reported in native window coordinates.
    // Keep ImGui's logical DisplaySize in that same coordinate space, while
    // DisplayFramebufferScale maps draw data to the actual render target.
#if defined(OS_ANDROID)
    if (windowHandle != 0)
    {
        ANativeWindow* window = reinterpret_cast<ANativeWindow*>(windowHandle);
        const int32_t nativeWidth = ANativeWindow_getWidth(window);
        const int32_t nativeHeight = ANativeWindow_getHeight(window);
        if (nativeWidth > 0 && nativeHeight > 0)
        {
            deviceWidth = static_cast<uint32_t>(nativeWidth);
            deviceHeight = static_cast<uint32_t>(nativeHeight);
        }
    }
#endif

    ImGuiIO& io = ImGui::GetIO();
    io.DeltaTime = 1.0f / 60.0f;                // set the time elapsed since the previous frame (in seconds)
    io.DisplaySize.x = float(deviceWidth);     // set the current display width
    io.DisplaySize.y = float(deviceHeight);    // set the current display height here
    io.DisplayFramebufferScale.x = (float)renderWidth / io.DisplaySize.x;
    io.DisplayFramebufferScale.y = (float)renderHeight / io.DisplaySize.y;

    float SCALE = 4.0f;
    ImFontConfig cfg {};
    cfg.SizePixels = 13 * SCALE;
    ImGui::GetIO().Fonts->AddFontDefault(&cfg);

    return true;
}

//-----------------------------------------------------------------------------

void GuiImguiPlatform::Update()
{
    GuiImguiBase::Update();
}

//-----------------------------------------------------------------------------

bool GuiImguiPlatform::TouchDownEvent(int iPointerID, float xPos, float yPos)
{
    if (iPointerID == 0)
    {
        ImGuiIO& io = ImGui::GetIO();
        io.MouseDown[0] = true; // left button down
        io.MousePos = { xPos, yPos };
        return io.WantCaptureMouse;
    }
    return false;
}

//-----------------------------------------------------------------------------

bool GuiImguiPlatform::TouchMoveEvent(int iPointerID, float xPos, float yPos)
{
    if (iPointerID == 0)
    {
        ImGuiIO& io = ImGui::GetIO();
        io.MouseDown[0] = true;
        io.MousePos = { xPos,yPos };
        return io.WantCaptureMouse;
    }
    return false;
}

//-----------------------------------------------------------------------------

bool GuiImguiPlatform::TouchUpEvent(int iPointerID, float xPos, float yPos)
{
    if (iPointerID == 0)
    {
        ImGuiIO& io = ImGui::GetIO();
        io.MouseDown[0] = false; // left button up
        return io.WantCaptureMouse;
    }
    return false;
}
