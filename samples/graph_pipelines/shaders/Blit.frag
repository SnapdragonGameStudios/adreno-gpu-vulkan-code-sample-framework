//============================================================================================================
//
//
//                  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#version 400

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

// Buffer binding locations
#define SHADER_DIFFUSE_TEXTURE_LOC          0
#define SHADER_OVERLAY_TEXTURE_LOC          1
#define SHADER_FRAG_UBO_LOCATION            2

layout(set = 0, binding = SHADER_DIFFUSE_TEXTURE_LOC) uniform sampler2D u_DiffuseTex;
layout(set = 0, binding = SHADER_OVERLAY_TEXTURE_LOC) uniform sampler2D u_OverlayTex;

// Varying's
layout (location = 0) in vec2   v_TexCoord;
layout (location = 1) in vec4   v_VertColor;

// Finally, the output color
layout (location = 0) out vec4 FragColor;

// Uniform Constant Buffer
layout(std140, set = 0, binding = SHADER_FRAG_UBO_LOCATION) uniform Params 
{
    bool IsUpscalingActive;

} FragCB;


//-----------------------------------------------------------------------------
void main()
//-----------------------------------------------------------------------------
{
    vec2 LocalTexCoord = vec2(v_TexCoord.xy);

    // ********************************
    // Texture Colors
    // ********************************
    // Get base color from the color texture
    vec4 DiffuseColor = texture( u_DiffuseTex, LocalTexCoord.xy );

    // Multiply by vertex color.
    DiffuseColor.xyzw *= v_VertColor.xyzw;

    // Apply darkening/lightening control
    // float lerp01 = min(1,FragCB.Diffuse);
    // float lerp12 = max(0,FragCB.Diffuse-1);
    // DiffuseColor = DiffuseColor * lerp01 + lerp12 - lerp12 * DiffuseColor;

    // Get the Overlay value
    vec4 OverlayColor = texture( u_OverlayTex, LocalTexCoord.xy );

    // ********************************
    // Alpha Blending
    // ********************************
    FragColor.rgb = DiffuseColor.rgb *(1.0-OverlayColor.a) + OverlayColor.rgb;
    FragColor.a = 1.0;

    // DEBUG! DEBUG! DEBUG! DEBUG! DEBUG! DEBUG! DEBUG! DEBUG! DEBUG! DEBUG! DEBUG! 
    // FragColor.xyz = (1.0 - OverlayColor.a) * DiffuseColor.xyz + OverlayColor.a * OverlayColor.xyz;
    // FragColor = vec4(DiffuseColor.xyz, DiffuseColor.a);
    // FragColor = vec4(OverlayColor.xyz, OverlayColor.a);
    // FragColor.a = 1.0;
    // FragColor = vec4(0.8, 0.2, 0.8, 1.0);
}

