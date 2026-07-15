//============================================================================================================
//
//
//                  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
#pragma once

#include "system/glm_common.hpp"


class CameraControllerBase
{
public:
    CameraControllerBase();
    virtual ~CameraControllerBase();

    virtual bool Initialize(uint32_t width, uint32_t height) = 0;

    // Inputs
    virtual void KeyDownEvent(uint32_t key) {}
    virtual void KeyUpEvent(uint32_t key) {}
    virtual void TouchDownEvent(int iPointerID, float xPos, float yPos) {}
    virtual void TouchMoveEvent(int iPointerID, float xPos, float yPos) {}
    virtual void TouchUpEvent(int iPointerID, float xPos, float yPos) {}
    virtual void TouchDoubleClickEvent(int iPointerID) {}

    /// Set screen size (for mouse events)
    void    SetSize(uint32_t width, uint32_t height);
    /// Set movement speed (units per second)
    void    SetMoveSpeed(float speed);
    /// Set rotation speed (units per second)
    void    SetRotateSpeed(float speed);
    /// Set reverse rotation axis (boolean)
    void    SetReverseRotation(bool reverseVertical, bool reverseHorizontal);
    /// Set world up vector (assumed normalized).  Tested with y-up and z-up
    void    SetWorldUp(glm::vec3 up);

    /// Update the camera controller and modify the output position/rotation accordingly
    /// @param frameTime time in seconds since last Update.
    /// @param position in - current camera position, out - camera position after applying controller
    /// @param rot in - current camera rotation, out - camera rotation after applying controller
    /// @param cut in - current camera cut, out - camera cut after applying controller
    virtual void Update(float frameTime, glm::vec3& position, glm::quat& rot, bool& cut) = 0;

protected:
    enum KeysDownBits {
        eNone = 0,
        eForward = 0x1,
        eBackward = 0x2,
        eLeft = 0x4,
        eRight = 0x8,
        eUp = 0x10,
        eDown = 0x20,
        eShift = 0x40
    };
    typedef std::underlying_type_t<KeysDownBits> KeysDownBits_base;
    static enum KeysDownBits KeyToBits(uint32_t key);

protected:
    glm::vec2       m_ScreenSize;;
    glm::vec3       m_WorldUp;
    float           m_MoveSpeed;
    float           m_RotateSpeed;
    bool            m_ReverseVerticalRotateAxis;
    bool            m_ReverseHorizontalRotateAxis;
};


///
/// Basic camera controller
/// Mouse (or touch) to look around and keyboard 'wasd' to move.
/// @ingroup Camera
/// 
class CameraController : public CameraControllerBase
{
    CameraController(const CameraController&) = delete;
    CameraController& operator=(const CameraController&) = delete;
public:
    CameraController();

    bool Initialize(uint32_t width, uint32_t height) override;

    // Inputs
    void KeyDownEvent(uint32_t key) override;
    void KeyUpEvent(uint32_t key) override;
    void TouchDownEvent(int iPointerID, float xPos, float yPos) override;
    void TouchMoveEvent(int iPointerID, float xPos, float yPos) override;
    void TouchUpEvent(int iPointerID, float xPos, float yPos) override;

    /// Update the camera controller and modify the output position/rotation accordingly
    /// @param frameTime time in seconds since last Update.
    /// @param position in - current camera position, out - camera position after applying controller
    /// @param rot in - current camera rotation, out - camera rotation after applying controller
    /// @param cut in - current camera cut, out - camera cut after applying controller
    void    Update(float frameTime, glm::vec3& position, glm::quat& rot, bool& cut ) override;

protected:
    glm::vec2       m_LastMousePosition;
    glm::vec2       m_CurrentMousePosition;
    bool            m_touchDown;

    KeysDownBits    m_KeysDown = KeysDownBits::eNone;
};

