// SPDX-License-Identifier: BSD-3-Clause

//=============================================================================
//
//
//                  Copyright (c) 2020 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================
#pragma once

#include "vulkan/vulkan.hpp"

#include "system/os_common.h"
#include "system/glm_common.hpp"


//=============================================================================
// Uniform Buffers
//=============================================================================
// ************************************
// Object
// ************************************
typedef struct _ObjectVertUB
{
    glm::mat4   MVPMatrix;
    glm::mat4   ModelMatrix;
    glm::mat4   ShadowMatrix;
} ObjectVertUB;

typedef struct _ObjectFragUB
{
    glm::vec4   Color;

    // X: Normal Height
    // Y: Normal Mirror Reflect Amount
    // Z: Reflect Fresnel Min
    // W: Reflect Fresnel Max
    glm::vec4   NormalHeight;

} ObjectFragUB;


// ************************************
// Skybox
// ************************************
typedef struct _SkyboxVertUB
{
    glm::mat4   MVPMatrix;
    glm::mat4   ModelMatrix;
    glm::vec4   Color;
} SkyboxVertUB;

// ************************************
// Floor
// ************************************
typedef struct _FloorVertUB
{
    glm::mat4   MVPMatrix;
    glm::mat4   ModelMatrix;
    glm::mat4   ShadowMatrix;
} FloorVertUB;

typedef struct _FloorFragUB
{
    glm::vec4   Color;
    glm::vec4   EyePos;
    glm::vec4   LightDir;
    glm::vec4   LightColor;

    // X: Normal Height
    // Y: Normal Mirror Reflect Amount
    // Z: Reflect Fresnel Min
    // W: Reflect Fresnel Max
    glm::vec4   NormalHeight;

    glm::vec4   MirrorAmount;

    // X: Screen Width
    // Y: Screen Height
    // Z: One Width Pixel
    // W: One Height Pixel
    glm::vec4   ScreenSize;

    // X: Shadow Width
    // Y: Shadow Height
    // Z: One Width Pixel
    // W: One Height Pixel
    glm::vec4   ShadowSize;

    // X: Mip Level
    // Y: Reflect Amount
    // Z: Reflect Fresnel Min
    // W: Reflect Fresnel Max
    glm::vec4   ReflectParams;

    // X: Mip Level
    // Y: IBL Amount
    // Z: Not Used
    // W: Not Used
    glm::vec4   IrradianceParams;
} FloorFragUB;

// ************************************
// Blit
// ************************************
typedef struct _BlitVertUB
{
    glm::mat4   MVPMatrix;
    glm::mat4   ModelMatrix;
    glm::vec4   Scale;
    glm::vec4   Offset;
} BlitVertUB;

typedef struct _BlitFragUB
{
    glm::vec4   Color;

    // X: U Scale
    // Y: V Scale
    // Z: U Offset
    // W: V Offset
    glm::vec4   ScaleOffset;

    // X: Not Used
    // Y: Not Used
    // Z: Not Used
    // W: Not Used
    glm::vec4   BlitParams;
} BlitFragUB;
