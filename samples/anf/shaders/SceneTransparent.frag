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


// Uniform buffer locations
#define SHADER_VERT_UBO_LOCATION            0
#define SHADER_FRAG_UBO_LOCATION            1
#define SHADER_LIGHT_UBO_LOCATION           2

// Texture Locations
#define SHADER_DIFFUSE_TEXTURE_LOC            3
#define SHADER_NORMAL_TEXTURE_LOC             4
#define SHADER_EMISSIVE_TEXTURE_LOC           5
#define SHADER_METALLIC_ROUGHNESS_TEXTURE_LOC 6

#define NUM_SPOT_LIGHTS (4)

// Uniform Constant Buffer
layout(std140, set = 0, binding = SHADER_FRAG_UBO_LOCATION) uniform FragConstantsBuff
{
    vec4 Color;
    vec4 ORM;

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

    // Spotlight data
    vec4 SpotLights_pos[NUM_SPOT_LIGHTS];
    vec4 SpotLights_dir[NUM_SPOT_LIGHTS];
    vec4 SpotLights_color[NUM_SPOT_LIGHTS];

    vec4 AmbientColor;

    float AmbientOcclusionScale;
    int Width;
    int Height;

} LightCB;

#ifndef PI
#define PI (3.14159265359)
#endif

// Textures
layout(set = 0, binding = SHADER_DIFFUSE_TEXTURE_LOC)            uniform sampler2D u_DiffuseTex;
layout(set = 0, binding = SHADER_NORMAL_TEXTURE_LOC)             uniform sampler2D u_NormalTex;
layout(set = 0, binding = SHADER_EMISSIVE_TEXTURE_LOC)           uniform sampler2D u_EmissiveTex;
layout(set = 0, binding = SHADER_METALLIC_ROUGHNESS_TEXTURE_LOC) uniform sampler2D u_MetallicRoughnessTex;

// Varying's
layout (location = 0) in vec2   v_TexCoord;
layout (location = 1) in vec3   v_WorldPos;
layout (location = 2) in vec3   v_WorldNorm;
layout (location = 3) in vec3   v_WorldTan;
layout (location = 4) in vec3   v_WorldBitan;
layout (location = 5) in vec4   v_ShadowCoord;
layout (location = 6) in vec4   v_VertColor;

// Output color
layout (location = 0) out vec4 SceneColor;
layout (location = 1) out vec2 SceneVelocity;

//-----------------------------------------------------------------------------
vec4 ScreenToView(vec2 ScreenCoord/*0-1 range*/, float Depth/*0-1*/)
//-----------------------------------------------------------------------------
{
    vec4 ClipSpacePosition = vec4((ScreenCoord * 2.0) - vec2(1.0), Depth, 1.0);
    ClipSpacePosition.y = -ClipSpacePosition.y;
    vec4 ViewSpacePosition = LightCB.ProjectionInv * ClipSpacePosition;

    // Perspective division
    ViewSpacePosition /= vec4(ViewSpacePosition.w);

    return ViewSpacePosition;
}

//-----------------------------------------------------------------------------
vec3 ScreenToWorld(vec2 ScreenCoord/*0-1 range*/, float Depth/*0-1*/)
//-----------------------------------------------------------------------------
{
    vec4 ViewSpacePosition = ScreenToView(ScreenCoord, Depth);

    vec4 WorldSpacePosition = LightCB.ViewInv * ViewSpacePosition;
    return WorldSpacePosition.xyz;
}

//-----------------------------------------------------------------------------
float FSchlick(float f0, float f90, float u)
//-----------------------------------------------------------------------------
{
    return f0 + (f90 - f0) * pow(1.0 - u, 5.0);
}

//-----------------------------------------------------------------------------
vec3 FSchlick(vec3 f0, float f90, float u)
//-----------------------------------------------------------------------------
{
    return f0 + (f90 - f0) * pow(1.0 - u, 5.0);
}

//-----------------------------------------------------------------------------
void CalcBRDF(vec3 EyeDir, vec3 Normal, vec3 LightDir, vec3 AlbedoColor, float Roughness, float Metallic, out vec3 f_diffuse, out vec3 f_specular, out vec3 f0)
//-----------------------------------------------------------------------------
{
    vec3 H = normalize(LightDir + EyeDir);
    //float NL = max(0.0, dot(Normal, LightDir));
    float NV = max(0.0, dot(Normal, EyeDir));
    float LH = max(0.0, dot(LightDir, H));
    float NH = max(0.0, dot(Normal, H));
    float VH = max(0.0, dot(EyeDir, H));

    float gltfDielectricSpecular = 0.04;

    vec3 c_diff = mix(AlbedoColor.rgb * (1 - gltfDielectricSpecular), vec3(0.0)/*diffuse color is black in a metallic material*/, Metallic);
    f0 = mix(vec3(0.04, 0.04, 0.04), AlbedoColor.rgb, Metallic);
    float alpha = Roughness * Roughness;

    vec3 F = FSchlick(f0, 1.0/*f90*/, VH);
    f_diffuse = (1 - F) * (1 / PI) * c_diff;

    // D - GGX microfacet distribution
    float alphaSqr = alpha * alpha;
    float denom = NH * NH * (alphaSqr - 1.0) + 1.0f;
    float D = alphaSqr / (PI * denom * denom);

    // V
    float k = alpha / 2.0f;
    float k2 = k * k;
    float invK2 = 1.0f - k2;
    float Vis = 1.0 / (LH * LH * invK2 + k2);

    f_specular = F * Vis * D;
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

    vec4 DiffuseColor = texture( u_DiffuseTex, v_TexCoord.xy );
    DiffuseColor.xyzw *= FragCB.Color.xyzw;

    if(DiffuseColor.a < 0.5)
    {
        discard;
    }

    // Adjust by vertex color.
    DiffuseColor.xyzw *= v_VertColor.xyzw;

    vec4 Emissive = texture( u_EmissiveTex, v_TexCoord.xy );
    vec4 MetallicRoughness = texture( u_MetallicRoughnessTex, v_TexCoord.xy );

    // Get base normal from the bump texture
    vec3 Normal = texture( u_NormalTex, v_TexCoord.xy ).rgb;
    Normal = Normal * 2.0 - 1.0;

	mat3 TBN = mat3(normalize(v_WorldTan), normalize(v_WorldBitan), normalize(v_WorldNorm));
    Normal = normalize(TBN * Normal);

    float Depth = gl_FragCoord.z;

    // Determine World position of pixel
    vec3 WorldPos = ScreenToWorld( LocalTexCoord, Depth );
    vec3 EyeDir = normalize(LightCB.CameraPos.xyz - v_WorldPos.xyz);

    vec3 L = -LightCB.LightDirection.xyz;

    vec3 f_diffuse  = vec3(0.0);
    vec3 f_specular = vec3(0.0);
    vec3 f0         = vec3(0.0);

    CalcBRDF(
        EyeDir,
        Normal,
        L,
        DiffuseColor.rgb,
        MetallicRoughness.g * FragCB.ORM.g,
        MetallicRoughness.b * FragCB.ORM.b,
        f_diffuse,
        f_specular,
        f0);

    // Spot lights
    vec3 spot_diffuse  = vec3(0.0);
    vec3 spot_specular = vec3(0.0);
    for(int l=0;l<NUM_SPOT_LIGHTS;++l)
    {
        // vec3 LightDir = normalize(v_WorldPos.xyz - LightCB.SpotLights_pos[l].xyz);

        // Light direction is from world position to light position
        vec3 LightDir = normalize(v_WorldPos.xyz - LightCB.SpotLights_pos[l].xyz);
        float LightFalloff = dot( LightDir, LightCB.SpotLights_dir[l].xyz );
        LightFalloff = smoothstep(0.5, 0.75, LightFalloff);

        vec3 diffuse  = vec3(0.0);
        vec3 specular = vec3(0.0);
        vec3 f0       = vec3(0.0);

        CalcBRDF(
            EyeDir,
            Normal,
            -LightCB.SpotLights_dir[l].xyz,
            DiffuseColor.rgb,
            MetallicRoughness.g * FragCB.ORM.g,
            MetallicRoughness.b * FragCB.ORM.b,
            diffuse,
            specular,
            f0);

        float NL = max(0.0, dot(Normal, -LightCB.SpotLights_dir[l].xyz));

        spot_diffuse += diffuse * LightCB.SpotLights_color[l].xyz * LightCB.SpotLights_color[l].a * NL * LightFalloff;
        spot_specular += specular * LightCB.SpotLights_color[l].xyz * LightCB.SpotLights_color[l].a * NL * LightFalloff;
    }

    float NL = max(0.0, dot(Normal, L));

    f_diffuse  = spot_diffuse + f_diffuse * LightCB.LightColor.rgb * LightCB.LightColor.a * NL;
    f_specular = spot_specular * LightCB.LightColor.rgb * LightCB.LightColor.a * NL;

    // Calculate ambient
    vec3 Ambient = LightCB.AmbientColor.rgb;

    // Light the material.
    vec3 LitColor = Ambient * DiffuseColor.rgb + (f_specular + f_diffuse);

    SceneColor       = vec4(LitColor, DiffuseColor.a);
    SceneVelocity    = vec2(0.0, 0.0); // Entire scene is static, for now
}