//============================================================================================================
//
//
//                  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#define SHADER_ATTRIB_LOC_POSITION          0
#define SHADER_ATTRIB_LOC_NORMAL            1
#define SHADER_ATTRIB_LOC_TEXCOORD0         2
#define SHADER_ATTRIB_LOC_COLOR             3
#define SHADER_ATTRIB_LOC_TANGENT           4

layout (location = SHADER_ATTRIB_LOC_POSITION ) in vec4 a_Position;
layout (location = SHADER_ATTRIB_LOC_NORMAL   ) in vec3 a_Normal;
layout (location = SHADER_ATTRIB_LOC_TEXCOORD0) in vec2 a_TexCoord;
layout (location = SHADER_ATTRIB_LOC_COLOR    ) in vec4 a_Color;
layout (location = SHADER_ATTRIB_LOC_TANGENT  ) in vec4 a_Tangent;

// Varying's
layout (location = 0) out vec2    v_TexCoord;
layout (location = 1) out vec4    v_VertColor;

void main()
{
    // Position and text coord are simple (Except Y in inverted on screen compared to OpenGL)
    vec4 TempPos = vec4(a_Position.xyz, 1.0); 
    gl_Position = vec4(TempPos.x, -TempPos.y, TempPos.z, TempPos.w);
    v_TexCoord = vec2(a_TexCoord.xy);

    // Color is simple attribute color
    v_VertColor.xyzw = vec4(a_Color.xyz, 1.0);
}
