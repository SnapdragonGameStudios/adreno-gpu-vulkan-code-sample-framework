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


// Uniform buffer locations
#define SHADER_VERT_UBO_LOCATION            0
#define SHADER_FRAG_UBO_LOCATION            1
#define SHADER_LIGHT_UBO_LOCATION           2

// Texture Locations
#define SHADER_DIFFUSE_TEXTURE_LOC          3
#define SHADER_NORMAL_TEXTURE_LOC           4

// Uniform Constant Buffer
layout(std140, set = 0, binding = SHADER_FRAG_UBO_LOCATION) uniform FragConstantsBuff 
{
    vec4    Color;

    // X: Normal Height
    // Y: Normal Mirror Reflect Amount
    // Z: Not Used
    // W: Not Used
    vec4    NormalHeight;

} FragCB;

// Light uniform
layout(std140, set = 0, binding = SHADER_LIGHT_UBO_LOCATION) uniform LightConstantsBuff
{
    mat4 ProjectionInv;
    mat4 ViewInv;
    mat4 ViewProjectionInv; // ViewInv * ProjectionInv
    // mat4 WorldToShadow;
    vec4 ProjectionInvW;    // w components of ProjectionInv
    vec4 CameraPos;

    vec4 LightDirection;
    vec4 LightColor;

    vec4 AmbientColor;

} LightCB;

#define NORMAL_HEIGHT           FragCB.NormalHeight.x
#define NORMAL_MIRROR_AMOUNT    FragCB.NormalHeight.y

// Textures
layout(set = 0, binding = SHADER_DIFFUSE_TEXTURE_LOC) uniform sampler2D u_DiffuseTex;
layout(set = 0, binding = SHADER_NORMAL_TEXTURE_LOC) uniform sampler2D  u_NormalTex;

// Varying's
layout (location = 0) in vec2   v_TexCoord;
layout (location = 1) in vec3   v_WorldPos;
layout (location = 2) in vec3   v_WorldNorm;
layout (location = 3) in vec3   v_WorldTan;
layout (location = 4) in vec3   v_WorldBitan;
layout (location = 5) in vec4   v_ShadowCoord;
layout (location = 6) in vec4   v_VertColor;

// Output color
layout (location = 0) out vec4 FragColor;

//-----------------------------------------------------------------------------
vec3 ScreenToWorld(vec2 ScreenCoord/*0-1 range*/, float Depth/*0-1*/)
//-----------------------------------------------------------------------------
{
    // Faster ScreenToWorld does one dotproduct with the inverse projection matrix to the perspective divisor and does one full (xyz) matrix multiply which is then perspective divided
    // Thanks to David McAllister for pointing this out.
    vec4 ClipSpacePosition = vec4((ScreenCoord * 2.0) - vec2(1.0), Depth, 1.0);
    ClipSpacePosition.y = -ClipSpacePosition.y;

    //  Just one dp4 to calculate w, so 3(4-1) dp4 calculations can be saved from previous mat4*vec4
    float ViewSpacePositionW = dot(LightCB.ProjectionInvW, ClipSpacePosition);
    
    vec3 WorldSpacePosition = (LightCB.ViewProjectionInv * ClipSpacePosition).xyz;
    return WorldSpacePosition.xyz/ViewSpacePositionW;
}

//-----------------------------------------------------------------------------
void main()
//-----------------------------------------------------------------------------
{
    vec2 LocalTexCoord = vec2(v_TexCoord.xy);

    // ********************************
    // Base (albedo) color
    // ********************************
    // Get color from the color texture

    vec4 AlbedoColor = texture( u_DiffuseTex, v_TexCoord.xy );
    AlbedoColor.xyzw *= FragCB.Color.xyzw;

    // Adjust by vertex color.
    AlbedoColor.xyzw *= v_VertColor.xyzw;

    // Get base normal from the bump texture
    vec4 NormTexValue = texture( u_NormalTex, v_TexCoord.xy );
    vec3 N = NormTexValue.xyz * 2.0 - 1.0;

    //N.xy *= NORMAL_HEIGHT;
    N = normalize(N);

    // Need matrix to convert to tangent space
    // vec3 binormal = cross(v_WorldNorm, v_WorldTan);
    // mat3 WorldToTan = mat3(normalize(v_WorldTan), normalize(binormal), normalize(v_WorldNorm));
    mat3 WorldToTan = mat3(normalize(v_WorldTan), normalize(v_WorldBitan), normalize(v_WorldNorm));
    
    // Convert the bump normal to tangent space
    vec3 BumpNormal = normalize(WorldToTan * N);

    // Setup the color and put depth value in the alpha channel
    AlbedoColor = vec4(AlbedoColor.rgb, FragCB.Color.a);

    // Setup the Normal
    vec3 Normal = BumpNormal.xyz;
    float Depth = gl_FragCoord.z; /*1.0 - NormalWithDepth.w;*/

    // ********************************
    // Ambient Occlusion
    // ********************************
    FragColor.rgb = AlbedoColor.rgb;
    FragColor.a = 1.0;

    // Determine World position of pixel
    vec3 WorldPos = ScreenToWorld( LocalTexCoord, Depth );

    // Calculate ambient (fixed value with darkening by Ambient Occlusion)
    vec3 Ambient = LightCB.AmbientColor.rgb;

    // ********************************
    // Light
    // ********************************
    vec3 EyeDir = normalize(LightCB.CameraPos.xyz - WorldPos);

    vec3 LightAmt = Ambient;

    {
        vec3 WorldToLightVec = LightCB.LightDirection.xyz;
        float WorldToLightDist2 = dot(WorldToLightVec, WorldToLightVec);
        vec3 WorldToLightNorm = normalize(WorldToLightVec);

        float SpotFalloffAng = dot(vec3(0.0, 1.0, 0.0), WorldToLightNorm);
		float SpotFalloff =  clamp((SpotFalloffAng - 0.8) / 0.2, 0.0, 1.0);

        float LightAng = max(0.0, dot( WorldToLightNorm, Normal));

        // Spec (blinn-phong)
        vec3 LightDir = WorldToLightNorm;
        vec3 HalfVector = normalize(LightDir+EyeDir);
        float Spec = pow(max(dot(Normal,HalfVector),0.0), 100) * 1.5;

        LightAmt += SpotFalloff * LightCB.LightColor.rgb * (Spec + LightAng) * LightCB.LightColor.w / (1.0 + WorldToLightDist2);   
    }

    FragColor.rgb = AlbedoColor.rgb * LightAmt;
    FragColor.a = FragCB.Color.a;
}

