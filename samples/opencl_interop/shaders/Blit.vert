//============================================================================================================
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#define SHADER_ATTRIB_LOC_POSITION          0
#define SHADER_ATTRIB_LOC_TEXCOORD0         1
#define SHADER_ATTRIB_LOC_COLOR             2

layout (location = SHADER_ATTRIB_LOC_POSITION ) in vec4 a_Position;
layout (location = SHADER_ATTRIB_LOC_TEXCOORD0) in vec2 a_TexCoord;

// Varying's
layout (location = 0) out vec2    v_TexCoord;

void main()
{
    // Position and text coord are simple (Except Y in inverted on screen compared to OpenGL)
    vec4 TempPos = vec4(a_Position.xyz, 1.0); 
    gl_Position = vec4(TempPos.x, -TempPos.y, TempPos.z, TempPos.w);
    v_TexCoord = vec2(a_TexCoord.xy);
}
