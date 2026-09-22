
//============================================================================================================
//
//                  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
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
#define NUM_SHADER_ATTRIB_LOCATIONS         6

#define SHADER_VERT_UBO_LOCATION            0
#define SHADER_MODEL_UBO_LOCATION           1
#define SHADER_FRAG_UBO_LOCATION            2
#define SHADER_LIGHT_UBO_LOCATION           3

layout (location = SHADER_ATTRIB_LOC_POSITION ) in vec4 a_Position;
layout (location = SHADER_ATTRIB_LOC_NORMAL   ) in vec3 a_Normal;
layout (location = SHADER_ATTRIB_LOC_TEXCOORD0) in vec2 a_TexCoord;
layout (location = SHADER_ATTRIB_LOC_COLOR    ) in vec4 a_Color;
layout (location = SHADER_ATTRIB_LOC_TANGENT  ) in vec4 a_Tangent;

// Uniform Constant Buffer
layout(std140, set = 0, binding = SHADER_VERT_UBO_LOCATION) uniform VertConstantsBuff
{
    mat4   CurrViewProj;         // Jittered VP (raster)
    mat4   PrevViewProj;         // (unused)
    mat4   CurrViewProjNoJitter; // No-jitter VP (motion vectors)
    mat4   PrevViewProjNoJitter; // No-jitter VP (motion vectors)
    mat4   ModelMatrix;
    mat4   PrevModelMatrix;
    mat4   ShadowMatrix;
    vec2   CurrentJitter;
    vec2   PrevJitter;
} VertCB;

const mat4 biasMat = mat4(
    0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.5, 0.5, 0.0, 1.0
);

// Varyings
layout (location = 0) out vec2    v_TexCoord;
layout (location = 1) out vec3    v_WorldPos;
layout (location = 2) out vec3    v_WorldNorm;
layout (location = 3) out vec3    v_WorldTan;
layout (location = 4) out vec3    v_WorldBitan;
layout (location = 5) out vec4    v_ShadowCoord;
layout (location = 6) out vec4    v_VertColor;
layout (location = 7) out vec4    v_CurrClip;
layout (location = 8) out vec4    v_PrevClip;
layout (location = 9) flat out vec2    v_CurrentJitter;
layout (location = 10) flat out vec2   v_PrevJitter;

void main()
{
    v_TexCoord = a_TexCoord.xy;

    vec4 objectPos = vec4(a_Position.xyz, 1.0);

    vec4 currWorldPos4 = VertCB.ModelMatrix     * objectPos;
    vec4 prevWorldPos4 = VertCB.PrevModelMatrix * objectPos;

    v_WorldPos = currWorldPos4.xyz;

    // Raster (jittered)
    gl_Position = VertCB.CurrViewProj * currWorldPos4;

    // MV (no-jitter)
    vec4 currClipMv = VertCB.CurrViewProjNoJitter * currWorldPos4;
    vec4 prevClipMv = VertCB.PrevViewProjNoJitter * prevWorldPos4; // Ignoring static motion for model v2
    // vec4 prevClipMv = VertCB.CurrViewProjNoJitter * prevWorldPos4;

    v_CurrClip = currClipMv;
    v_PrevClip = prevClipMv;

    v_CurrentJitter = VertCB.CurrentJitter;
    v_PrevJitter    = VertCB.PrevJitter;

    v_ShadowCoord = biasMat * VertCB.ShadowMatrix * vec4(v_WorldPos, 1.0);

    v_WorldNorm  = normalize((VertCB.ModelMatrix * vec4(a_Normal.xyz,  0.0)).xyz);
    v_WorldTan   = normalize((VertCB.ModelMatrix * vec4(a_Tangent.xyz, 0.0)).xyz);
    v_WorldBitan = normalize(cross(v_WorldNorm, v_WorldTan) * a_Tangent.w);

    v_VertColor = vec4(a_Color.xyz, 1.0);
}
