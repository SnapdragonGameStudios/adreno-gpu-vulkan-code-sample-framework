//============================================================================================================
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

// Varying's
layout (location = 0) in vec2   v_TexCoord;

// Output: color only (R8G8B8A8_UNORM)
layout (location = 0) out vec4 FragColor;

void main()
{
    // UV-based gradient as placeholder scene content
    FragColor = vec4(v_TexCoord.x, v_TexCoord.y, 0.5, 1.0);
}