//============================================================================================================
//
//
//                  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "cameraController.hpp"
#include <glm/gtc/quaternion.hpp>

static const float cMouseRotSpeed = 0.5f;

// Helpers
constexpr glm::vec3 cVecViewRight = glm::vec3(1.0f, 0.0f, 0.0f);    // x-direction (vector pointing to right of screen)!
constexpr glm::vec3 cVecViewForward = glm::vec3(0.0f, 0.0f, -1.0f); // z-direction (vector pointing into screen)

//-----------------------------------------------------------------------------

CameraControllerBase::CameraControllerBase()
: m_ScreenSize(1920.f, 1080.f)
, m_WorldUp(glm::vec3(0.0f, 1.0f, 0.0f))
, m_MoveSpeed(1.0f)
, m_RotateSpeed(cMouseRotSpeed)
, m_ReverseVerticalRotateAxis(false)
, m_ReverseHorizontalRotateAxis(false)
{}

//-----------------------------------------------------------------------------

CameraControllerBase::~CameraControllerBase()
{}

//-----------------------------------------------------------------------------

bool CameraControllerBase::Initialize(uint32_t width, uint32_t height)
{
    SetSize(width, height);
    return true;
}

//-----------------------------------------------------------------------------

void CameraControllerBase::SetMoveSpeed(float speed)
{
    m_MoveSpeed = speed;
}

//-----------------------------------------------------------------------------

void CameraControllerBase::SetRotateSpeed(float speed)
{
    m_RotateSpeed = speed;
}

//-----------------------------------------------------------------------------

void CameraControllerBase::SetSize(uint32_t width, uint32_t height)
{
    m_ScreenSize = glm::vec2(width, height);
}

//-----------------------------------------------------------------------------

void CameraControllerBase::SetWorldUp(glm::vec3 up)
{
    m_WorldUp = up;
}

//-----------------------------------------------------------------------------

enum CameraController::KeysDownBits CameraControllerBase::KeyToBits(uint32_t key)
{
    switch (key)
    {
    case 'W':
        return KeysDownBits::eForward;
        break;
    case 'A':
        return KeysDownBits::eLeft;
        break;
    case 'S':
        return KeysDownBits::eBackward;
        break;
    case 'D':
        return KeysDownBits::eRight;
        break;
    case 'Q':
        return KeysDownBits::eDown;
        break;
    case 'E':
        return KeysDownBits::eUp;
        break;
    case 16:
        return KeysDownBits::eShift;
        break;
    default:
        return KeysDownBits::eNone;
        break;
    }
}

//-----------------------------------------------------------------------------

CameraController::CameraController() : CameraControllerBase()
    , m_LastMousePosition(0.0f)
    , m_CurrentMousePosition(0.0f)
    , m_touchDown(false)
{
}

//-----------------------------------------------------------------------------

bool CameraController::Initialize(uint32_t width, uint32_t height)
{
    return CameraControllerBase::Initialize(width, height);
}

//-----------------------------------------------------------------------------

void CameraController::KeyDownEvent(uint32_t key)
{
    KeysDownBits keyDown = KeyToBits(key);
    m_KeysDown = KeysDownBits(m_KeysDown | keyDown );
}

//-----------------------------------------------------------------------------

void CameraController::KeyUpEvent(uint32_t key)
{
    KeysDownBits keyUp = KeyToBits(key);
    m_KeysDown = KeysDownBits(m_KeysDown & ~keyUp);
}

//-----------------------------------------------------------------------------

void CameraController::TouchDownEvent(int iPointerID, float xPos, float yPos)
{
    //assert(m_touchDown==false);
    m_touchDown = true;
    m_LastMousePosition = { xPos, yPos };
    m_CurrentMousePosition = m_LastMousePosition;
}

//-----------------------------------------------------------------------------

void CameraController::TouchMoveEvent(int iPointerID, float xPos, float yPos)
{
    m_CurrentMousePosition = { xPos, yPos };
}

//-----------------------------------------------------------------------------

void CameraController::TouchUpEvent(int iPointerID, float xPos, float yPos)
{
    m_touchDown = false;
    m_CurrentMousePosition = { xPos, yPos };
}

//-----------------------------------------------------------------------------

void CameraController::Update(float frameTime, glm::vec3& position, glm::quat& rot, bool& cut)
{
    cut = false;
    if (m_touchDown)
    {
        auto mouseDiff = m_LastMousePosition - m_CurrentMousePosition;
        auto angleChange = mouseDiff * frameTime * m_RotateSpeed;

        m_LastMousePosition = m_CurrentMousePosition;

        float horizonalRotation = m_ReverseHorizontalRotateAxis ? -1.0f : 1.0f;
        float verticalRotation  = m_ReverseVerticalRotateAxis   ? -1.0f : 1.0f;
        // one (mouse) rotation axis is relative to the view direction, other is relative to world - prevents camera from 'twisting' although does introduce gimbal when looking along the UP axis and rotationg left/right.
        rot = glm::angleAxis(angleChange.x * horizonalRotation, m_WorldUp) * rot * glm::angleAxis(angleChange.y * verticalRotation, cVecViewRight);
        rot = glm::normalize(rot);
    }


    const float moveSpeed = m_MoveSpeed * ((m_KeysDown & KeysDownBits::eShift) == 0 ? 1.0f : 4.0f);
    if (m_KeysDown != KeysDownBits::eNone)
    {
        // Position change is relative to the camera rotation/direction.
        if (m_KeysDown & (KeysDownBits::eLeft | KeysDownBits::eRight))
        {
            float direction = (m_KeysDown & KeysDownBits::eLeft) ? -1.0f : 1.0f;
            position += rot * cVecViewRight * frameTime * moveSpeed * direction;
        }
        if (m_KeysDown & (KeysDownBits::eForward | KeysDownBits::eBackward))
        {
            float direction = (m_KeysDown & KeysDownBits::eBackward) ? -1.0f : 1.0f;
            position += rot * cVecViewForward * frameTime * moveSpeed * direction;
        }
        if (m_KeysDown & KeysDownBits::eUp)
        {
            glm::vec3 VecUp = glm::vec3(0.0f, 1.0f, 0.0f);
            position += VecUp * frameTime * moveSpeed;
        }
        if (m_KeysDown & KeysDownBits::eDown)
        {
            glm::vec3 VecUp = glm::vec3(0.0f, -1.0f, 0.0f);
            position += VecUp * frameTime * moveSpeed;
        }
    }
}

