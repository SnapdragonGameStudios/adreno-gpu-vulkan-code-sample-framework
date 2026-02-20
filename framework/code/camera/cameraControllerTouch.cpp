//============================================================================================================
//
//
//                  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "cameraControllerTouch.hpp"

static const float cMouseRotSpeed = 0.1f;
static const float cTouchMoveSpeedMultipler = 0.001f;

// Helpers
constexpr glm::vec3 cVecViewRight = glm::vec3( 1.0f, 0.0f, 0.0f );    // x-direction (vector pointing to right of screen)!
constexpr glm::vec3 cVecViewForward = glm::vec3( 0.0f, 0.0f, -1.0f ); // z-direction (vector pointing into screen)


//-----------------------------------------------------------------------------

CameraControllerTouch::CameraControllerTouch() : CameraControllerBase()
    , m_LastMovementTouchPosition(0.0f)
    , m_CurrentMovementTouchPosition(0.0f)
    , m_LastLookaroundTouchPosition(0.0f)
    , m_CurrentLookaroundTouchPosition(0.0f)
    , m_MovementTouchId(-1)
    , m_LookaroundTouchId(-1)
{
}

//-----------------------------------------------------------------------------

CameraControllerTouch::~CameraControllerTouch()
{}

//-----------------------------------------------------------------------------

bool CameraControllerTouch::Initialize(uint32_t width, uint32_t height)
{
    return CameraControllerBase::Initialize(width, height);
}

//-----------------------------------------------------------------------------


void CameraControllerTouch::TouchDownEvent(int iPointerID, float xPos, float yPos)
{
    if (xPos >= m_ScreenSize.x * 0.5f && m_LookaroundTouchId == -1)
    {
        m_LookaroundTouchId = iPointerID;
        m_LastLookaroundTouchPosition = { xPos, yPos };
        m_CurrentLookaroundTouchPosition = m_LastLookaroundTouchPosition;
        m_LookDeltaPixelsAccum = glm::vec2(0.0f);
    }
    else if (xPos < m_ScreenSize.x * 0.5f && m_MovementTouchId == -1)
    {
        m_MovementTouchId = iPointerID;
        m_LastMovementTouchPosition = { xPos, yPos };
        m_CurrentMovementTouchPosition = m_LastMovementTouchPosition;
    }
}

//-----------------------------------------------------------------------------

void CameraControllerTouch::TouchMoveEvent(int iPointerID, float xPos, float yPos)
{
    if (iPointerID == m_LookaroundTouchId)
    {
        const glm::vec2 new_pos = { xPos, yPos };
        const glm::vec2 delta = new_pos - m_CurrentLookaroundTouchPosition;

        m_CurrentLookaroundTouchPosition = new_pos;
        m_LookDeltaPixelsAccum += delta;
    }
    else if (iPointerID == m_MovementTouchId)
    {
        m_CurrentMovementTouchPosition = { xPos, yPos };
    }
}

//-----------------------------------------------------------------------------

void CameraControllerTouch::TouchUpEvent(int iPointerID, float xPos, float yPos)
{
    if (iPointerID == m_LookaroundTouchId)
    {
        m_LookaroundTouchId = -1;
        m_CurrentLookaroundTouchPosition = { xPos, yPos };
        m_LookDeltaPixelsAccum = glm::vec2(0.0f);
    }
    else if (iPointerID == m_MovementTouchId)
    {
        m_MovementTouchId = -1;
        m_CurrentMovementTouchPosition = { xPos, yPos };
    }
}

//-----------------------------------------------------------------------------

void CameraControllerTouch::Update(float frameTime, glm::vec3& position, glm::quat& rot, bool& cut)
{
    cut = false;

    if (m_LookaroundTouchId != -1)
    {
        const float tau = glm::max(0.0001f, m_LookSmoothTauSec);
        const float alpha = 1.0f - glm::exp(-frameTime / tau); // 0..1

        const glm::vec2 applyPixels = m_LookDeltaPixelsAccum * alpha;
        m_LookDeltaPixelsAccum -= applyPixels;

        const glm::vec2 ndcDelta =
        {
            applyPixels.x / glm::max(1.0f, m_ScreenSize.x),
            applyPixels.y / glm::max(1.0f, m_ScreenSize.y)
        };

        const float yaw = -ndcDelta.x * glm::pi<float>() * m_RotateSpeed;
        const float pitch = -ndcDelta.y * glm::pi<float>() * m_RotateSpeed;

        const glm::vec3 viewRight = rot * cVecViewRight;

        rot = glm::angleAxis(yaw, m_WorldUp) * rot;
        rot = glm::angleAxis(pitch, viewRight) * rot;
        rot = glm::normalize(rot);
    }
    else
    {
        m_LookDeltaPixelsAccum = glm::vec2(0.0f);
    }

    if (m_MovementTouchId != -1)
    {
#if 1
        const auto mouseDiff = m_LastMovementTouchPosition - m_CurrentMovementTouchPosition;
        const auto directionChange = mouseDiff * frameTime * m_MoveSpeed * cTouchMoveSpeedMultipler;

        position -= rot * cVecViewRight * directionChange.x;
        position += rot * cVecViewForward * directionChange.y;
#else
        const glm::vec2 mouseDiff = m_LastMovementTouchPosition - m_CurrentMovementTouchPosition;
        const glm::vec2 directionChange = mouseDiff * frameTime * m_MoveSpeed * cTouchMoveSpeedMultipler;

        position -= rot * cVecViewRight * directionChange.x;
        position += rot * cVecViewForward * directionChange.y;

        m_LastMovementTouchPosition = m_CurrentMovementTouchPosition;
#endif
    }
}

