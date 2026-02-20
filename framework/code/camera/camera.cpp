//============================================================================================================
//
//
//                  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#include "camera.hpp"
#include "cameraData.hpp"
#include "system/math_common.hpp"

Camera::Camera()
    : m_Orthographic(false)
    , m_Cut(false)
    , m_Aspect(1.0)
    , m_FOV(PI*0.25f)
    , m_NearClip(1.0f)
    , m_FarClip(100.0f)
    , m_CurrentCameraPos()
    , m_CurrentCameraRot()
    , m_Jitter()
    , m_ProjectionMatrix(1.0)
    , m_ViewMatrix()
    , m_ViewMatrixPreTranslation()
    , m_InverseViewProjection()
{
    UpdateMatrices();   // set sensible proj/view matrices
}

//-----------------------------------------------------------------------------
void Camera::SetAspect(float aspect)
//-----------------------------------------------------------------------------
{
    m_Aspect = aspect;
}

//-----------------------------------------------------------------------------
void Camera::SetFov(float fovRadians)
//-----------------------------------------------------------------------------
{
    m_FOV = fovRadians;
}

//-----------------------------------------------------------------------------
void Camera::SetClipPlanes(float near, float far)
//-----------------------------------------------------------------------------
{
    m_NearClip = near;
    m_FarClip = far;
}

//-----------------------------------------------------------------------------
void Camera::SetPosition(const glm::vec3& position, const glm::quat& rotation)
//-----------------------------------------------------------------------------
{
    m_CurrentCameraPos = position;
    m_CurrentCameraRot = rotation;
}

//-----------------------------------------------------------------------------
void Camera::Set(const CameraData& data)
{
    m_CurrentCameraPos = data.Position;
    m_CurrentCameraRot = data.Orientation;
}

//-----------------------------------------------------------------------------
void Camera::SetJitter(const glm::vec2 jitter)
//-----------------------------------------------------------------------------
{
    m_Jitter = jitter;
}

//-----------------------------------------------------------------------------
void Camera::SetCut(bool cut)
//-----------------------------------------------------------------------------
{
    m_Cut = cut;
}

//-----------------------------------------------------------------------------
void Camera::UpdateMatrices()
//-----------------------------------------------------------------------------
{
    // Update camera position, view matrix, etc.
    if (m_Orthographic)
    {
        // Orthographic camera
        m_ProjectionMatrixNoJitter = glm::orthoRH( -1.0f, 1.0f, -1.0f, 1.0f, m_NearClip, m_FarClip );   ///TODO: pass/use correct worth left/right/top/bottom values
        m_ProjectionMatrix = m_ProjectionMatrixNoJitter;    // no jitter applied for now
    }
    else
    {
        // Perspective camera
        m_ProjectionMatrixNoJitter = glm::perspectiveRH( m_FOV, m_Aspect, m_NearClip, m_FarClip );
        m_ProjectionMatrix = GetProjectionWithJitter( glm::vec3( m_Jitter, 0.0f ) );

        m_ViewMatrix = glm::mat4_cast( glm::inverse( m_CurrentCameraRot ) );
        //m_ViewMatrix = glm::translate(m_ViewMatrix, -m_CurrentCameraPos);

        auto translation = glm::mat4( 1.0f );
        m_ViewMatrixPreTranslation = m_ViewMatrix;
        translation = glm::translate( translation, -m_CurrentCameraPos );
        m_ViewMatrix = m_ViewMatrix * translation;

        auto viewProj = m_ProjectionMatrix * m_ViewMatrix;
        m_InverseViewProjection = glm::inverse( viewProj );
    }
}

//-----------------------------------------------------------------------------
glm::mat4 Camera::GetProjectionWithJitter(const glm::vec3 jitter) const
//-----------------------------------------------------------------------------
{
#if 0
    glm::mat4 jm = glm::translate(jitter);
    return jm * m_ProjectionMatrixNoJitter;
#else

    // Assumes jitts is in NDC (not pixel).
    // Remember to flip jitter y, if needed.
    glm::mat4 jitteredProj = m_ProjectionMatrixNoJitter;
    for (int col = 0; col < 4; ++col)
    {
        jitteredProj[col][0] += jitter.x * jitteredProj[col][3];
        jitteredProj[col][1] += jitter.y * jitteredProj[col][3];
    }
    return jitteredProj;
#endif
}
