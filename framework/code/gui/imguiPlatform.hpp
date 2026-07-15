//============================================================================================================
//
//
//                  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include "imguiBase.hpp"

// forward declarations
class Vulkan;

///
/// @brief Platform specific imgui class declaration
/// Implementation derives from this and is in the Windows / Android (or whatever) subdirectory.
/// Does the input/screen (but not rendering) for whichever platform we are running on
/// @ingroup GUI
///
class GuiImguiPlatform : public GuiImguiBase
{
public:
    GuiImguiPlatform();
    ~GuiImguiPlatform();
    virtual bool Initialize(uintptr_t windowHandle, TextureFormat renderFormat, uint32_t deviceWidth, uint32_t deviceHeight, uint32_t renderWidth, uint32_t renderHeight);
    void Update() override;

    bool TouchDownEvent(int iPointerID, float xPos, float yPos) override;
    bool TouchMoveEvent(int iPointerID, float xPos, float yPos) override;
    bool TouchUpEvent(int iPointerID, float xPos, float yPos) override;
};

///
/// @brief Graphics API specific imgui rendering class template
/// Actual implementations are implemented as a specialization of this class.
/// @ingroup GUI
///
template<typename T_GFXAPI>
class GuiImguiGfx : public GuiImguiPlatform
{
    static_assert(sizeof( GuiImguiGfx<T_GFXAPI> ) > sizeof( GuiImguiPlatform ));   // Ensure this class template is specialized (and not used as-is)
};

