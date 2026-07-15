//============================================================================================================
//
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#version 400

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

// Buffer binding locations
#define SHADER_OVERLAY_TEXTURE_LOC          0

layout(set = 0, binding = SHADER_OVERLAY_TEXTURE_LOC) uniform sampler2D u_OverlayTex;

// Varying's
layout (location = 0) in vec2   v_TexCoord;
layout (location = 1) in vec4   v_VertColor;

// Finally, the output color
layout (location = 0) out vec4 FragColor;

//-----------------------------------------------------------------------------
void main()
//-----------------------------------------------------------------------------
{
    vec2 LocalTexCoord = vec2(v_TexCoord.xy);
    vec4 OverlayColor = texture( u_OverlayTex, LocalTexCoord.xy );
    FragColor = OverlayColor;
}
