// SPDX-License-Identifier: BSD-3-Clause

//============================================================================================================
//
//
//                  Copyright (c) 2025 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//============================================================================================================

#include "imguiLinux.hpp"
#include <imgui/imgui.h>
#include "imgui/backends/imgui_impl_glfw.h"

//
// Implementation of GuiImguiPlatform (for glfw)
//

GuiImguiPlatform::GuiImguiPlatform()
{}

GuiImguiPlatform::~GuiImguiPlatform()
{
}

bool GuiImguiPlatform::Initialize(uintptr_t windowHandle, TextureFormat renderFormat, uint32_t renderWidth, uint32_t renderHeight)
{
    if (!GuiImguiBase::Initialize(windowHandle, renderFormat, renderWidth, renderHeight))
    {
        return false;
    }

    if (!ImGui_ImplGlfw_InitForVulkan((GLFWwindow*)windowHandle, true/*callbacks*/))
    {
        return false;
    }

    // Disable the IME callback, if IME is needed then we can revisit but this fixes a hang when entering text into an entry box.
    // Remove if/when we upgrade IMGUI past 1.88.  https://github.com/ocornut/imgui/issues/5535
    ImGui::GetIO().SetPlatformImeDataFn = nullptr;

    return true;
}

//-----------------------------------------------------------------------------

void GuiImguiPlatform::Update()
{
    ImGui_ImplGlfw_NewFrame();
    GuiImguiBase::Update();
}

//-----------------------------------------------------------------------------
// Windows inputs are all handled by ImGui_ImplGlfw_NewFrame
// No need to pass these events on to imGui (unlike on Android for instance!)
//-----------------------------------------------------------------------------

bool GuiImguiPlatform::TouchDownEvent(int iPointerID, float xPos, float yPos) { return false; }
bool GuiImguiPlatform::TouchMoveEvent(int iPointerID, float xPos, float yPos) { return false; }
bool GuiImguiPlatform::TouchUpEvent(int iPointerID, float xPos, float yPos) { return false; }
