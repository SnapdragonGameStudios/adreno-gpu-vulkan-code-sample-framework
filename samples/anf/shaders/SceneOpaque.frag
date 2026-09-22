
//============================================================================================================
//
//                  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#define SHADER_VERT_UBO_LOCATION              0
#define SHADER_FRAG_UBO_LOCATION              1
#define SHADER_LIGHT_UBO_LOCATION             2

#define SHADER_DIFFUSE_TEXTURE_LOC            3
#define SHADER_NORMAL_TEXTURE_LOC             4
#define SHADER_EMISSIVE_TEXTURE_LOC           5
#define SHADER_METALLIC_ROUGHNESS_TEXTURE_LOC 6

#define NUM_SPOT_LIGHTS (4)

layout(std140, set = 0, binding = SHADER_FRAG_UBO_LOCATION) uniform FragConstantsBuff
{
    vec4 Color;
    vec4 ORM;
} FragCB;

layout(std140, set = 0, binding = SHADER_LIGHT_UBO_LOCATION) uniform LightConstantsBuff
{
    mat4 ProjectionInv;
    mat4 ViewInv;
    mat4 ViewProjectionInv;
    vec4 ProjectionInvW;
    vec4 CameraPos;

    int   Width;
    int   Height;
    int Debug_MVEnabled; // Just for simplicity we use the light UB for this
    int Debug_MVInvertX; // Just for simplicity we use the light UB for this
    int Debug_MVInvertY; // Just for simplicity we use the light UB for this
    float Debug_MVCompensate;
    int Debug_MVAxisInvertX;
    int Debug_MVAxisInvertY;

    vec4 LightDirection;
    vec4 LightColor;

    vec4 SpotLights_pos[NUM_SPOT_LIGHTS];
    vec4 SpotLights_dir[NUM_SPOT_LIGHTS];
    vec4 SpotLights_color[NUM_SPOT_LIGHTS];

    vec4 AmbientColor;

    float AmbientOcclusionScale;

} LightCB;

#ifndef PI
#define PI (3.14159265359)
#endif

layout(set = 0, binding = SHADER_DIFFUSE_TEXTURE_LOC)            uniform sampler2D u_DiffuseTex;
layout(set = 0, binding = SHADER_NORMAL_TEXTURE_LOC)             uniform sampler2D u_NormalTex;
layout(set = 0, binding = SHADER_EMISSIVE_TEXTURE_LOC)           uniform sampler2D u_EmissiveTex;
layout(set = 0, binding = SHADER_METALLIC_ROUGHNESS_TEXTURE_LOC) uniform sampler2D u_MetallicRoughnessTex;

// Varyings
layout (location = 0) in vec2   v_TexCoord;
layout (location = 1) in vec3   v_WorldPos;
layout (location = 2) in vec3   v_WorldNorm;
layout (location = 3) in vec3   v_WorldTan;
layout (location = 4) in vec3   v_WorldBitan;
layout (location = 5) in vec4   v_ShadowCoord;
layout (location = 6) in vec4   v_VertColor;
layout (location = 7) in vec4   v_CurrClip;
layout (location = 8) in vec4   v_PrevClip;
layout (location = 9) flat in vec2   v_CurrentJitter;
layout (location = 10) flat in vec2  v_PrevJitter;

layout (location = 0) out vec4 SceneColor;
layout (location = 1) out vec2 SceneVelocity;
layout (location = 2) out float SceneInvDepth;

vec4 ScreenToView(vec2 ScreenCoord, float Depth)
{
    vec4 ClipSpacePosition = vec4((ScreenCoord * 2.0) - vec2(1.0), Depth, 1.0);
    vec4 ViewSpacePosition = LightCB.ProjectionInv * ClipSpacePosition;
    ViewSpacePosition /= vec4(ViewSpacePosition.w);
    return ViewSpacePosition;
}

vec3 ScreenToWorld(vec2 ScreenCoord, float Depth)
{
    vec4 ViewSpacePosition = ScreenToView(ScreenCoord, Depth);
    vec4 WorldSpacePosition = LightCB.ViewInv * ViewSpacePosition;
    return WorldSpacePosition.xyz;
}

float FSchlick(float f0, float f90, float u)
{
    return f0 + (f90 - f0) * pow(1.0 - u, 5.0);
}

vec3 FSchlick(vec3 f0, float f90, float u)
{
    return f0 + (f90 - f0) * pow(1.0 - u, 5.0);
}

vec3 SafeNormalize(vec3 v, vec3 fallback_dir)
{
    float len2 = dot(v, v);
    if (len2 > 1e-12)
    {
        return v * inversesqrt(len2);
    }
    return fallback_dir;
}

void CalcBRDF(vec3 EyeDir, vec3 Normal, vec3 LightDir, vec3 AlbedoColor, float Roughness, float Metallic,
              out vec3 f_diffuse, out vec3 f_specular, out vec3 f0)
{
    vec3 H = SafeNormalize(LightDir + EyeDir, EyeDir);
    float NV = max(0.0, dot(Normal, EyeDir));
    float LH = max(0.0, dot(LightDir, H));
    float NH = max(0.0, dot(Normal, H));
    float VH = max(0.0, dot(EyeDir, H));

    float gltfDielectricSpecular = 0.04;

    vec3 c_diff = mix(AlbedoColor.rgb * (1 - gltfDielectricSpecular), vec3(0.0), Metallic);
    f0 = mix(vec3(0.04), AlbedoColor.rgb, Metallic);
    float RoughnessClamped = max(Roughness, 0.045);
    float alpha = RoughnessClamped * RoughnessClamped;

    vec3 F = FSchlick(f0, 1.0, VH);
    f_diffuse = (1 - F) * (1 / PI) * c_diff;

    float alphaSqr = alpha * alpha;
    float denom = NH * NH * (alphaSqr - 1.0) + 1.0;
    float D = alphaSqr / (PI * denom * denom);

    float k = alpha / 2.0;
    float k2 = k * k;
    float invK2 = 1.0 - k2;
    float Vis = 1.0 / (LH * LH * invK2 + k2);

    f_specular = F * Vis * D;
}

vec2 GetScreenUV()
{
    vec2 rt_size = vec2(float(LightCB.Width), float(LightCB.Height));
    return (gl_FragCoord.xy + vec2(0.5)) / rt_size;
}

bool WasOffscreen(vec4 prev_clip)
{
    vec3 clamped_pos = clamp(prev_clip.xyz,
                             vec3(-prev_clip.w, -prev_clip.w, 0.0),
                             vec3( prev_clip.w,  prev_clip.w, prev_clip.w));

    vec3 diff = abs(clamped_pos - prev_clip.xyz);

    // Guard band: scale epsilon by w so it's stable across depth
    float eps = 1e-4 * prev_clip.w;

    return any(greaterThan(diff, vec3(eps)));
}

vec2 CalcScreenSpaceDisplacement(vec4 CurrClip, vec4 PrevClip)
{
    if (WasOffscreen(PrevClip))
    {
        return vec2(0.0, 0.0);
    }

    vec2 new_ndc = CurrClip.xy / CurrClip.w;
    vec2 old_ndc = PrevClip.xy / PrevClip.w;

    vec2 delta_ndc = (new_ndc - old_ndc) ;

    // Convert NDC delta -> UV delta:
    //   NDC spans [-1..1] so multiply by 0.5 to get UV-sized displacement,
    //   and flip Y because screen Y is typically opposite of clip/NDC Y.
    return 0.5 * vec2(delta_ndc.x, -delta_ndc.y);
}

vec2 SafeMvUv(vec2 mv_uv)
{
    const float max_uv = 0.5;
    mv_uv = clamp(mv_uv, vec2(-max_uv), vec2(max_uv));

    if (any(isnan(mv_uv)) || any(isinf(mv_uv)))
        mv_uv = vec2(0.0);

    return mv_uv;
}

void main()
{
    vec4 DiffuseColor = texture(u_DiffuseTex, v_TexCoord.xy);
    DiffuseColor *= FragCB.Color;

    ///////////////////////////////////////////
    // EARLY OUT
    ///////////////////////////////////////////

    if (DiffuseColor.a < 0.5)
    {
        discard;
    }

    ///////////////////////////////////////////
    // BRDF
    ///////////////////////////////////////////

    DiffuseColor *= v_VertColor;

    vec4 Emissive = texture(u_EmissiveTex, v_TexCoord.xy);
    vec4 MetallicRoughness = texture(u_MetallicRoughnessTex, v_TexCoord.xy);

    vec3 Normal = texture(u_NormalTex, v_TexCoord.xy).rgb;
    Normal = Normal * 2.0 - 1.0;

    mat3 TBN = mat3(normalize(v_WorldTan), normalize(v_WorldBitan), normalize(v_WorldNorm));
    Normal = TBN * Normal;
    Normal = SafeNormalize(Normal, vec3(0.0, 0.0, 1.0));

    float Depth = gl_FragCoord.z;

    vec2 ScreenUV = GetScreenUV();
    vec3 WorldPos = ScreenToWorld(ScreenUV, Depth);
    vec3 EyeDir   = normalize(LightCB.CameraPos.xyz - v_WorldPos.xyz);

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
        f0
    );

    vec3 spot_diffuse  = vec3(0.0);
    vec3 spot_specular = vec3(0.0);

    for (int l = 0; l < NUM_SPOT_LIGHTS; ++l)
    {
        vec3 LightDir = normalize(v_WorldPos.xyz - LightCB.SpotLights_pos[l].xyz);
        float LightFalloff = dot(LightDir, LightCB.SpotLights_dir[l].xyz);
        LightFalloff = smoothstep(0.5, 0.75, LightFalloff);

        vec3 diffuse  = vec3(0.0);
        vec3 specular = vec3(0.0);
        vec3 tmpf0    = vec3(0.0);

        CalcBRDF(
            EyeDir,
            Normal,
            -LightCB.SpotLights_dir[l].xyz,
            DiffuseColor.rgb,
            MetallicRoughness.g * FragCB.ORM.g,
            MetallicRoughness.b * FragCB.ORM.b,
            diffuse,
            specular,
            tmpf0
        );

        float NL = max(0.0, dot(Normal, -LightCB.SpotLights_dir[l].xyz));

        spot_diffuse  += diffuse  * LightCB.SpotLights_color[l].xyz * LightCB.SpotLights_color[l].a * NL * LightFalloff;
        spot_specular += specular * LightCB.SpotLights_color[l].xyz * LightCB.SpotLights_color[l].a * NL * LightFalloff;
    }

    float NL = max(0.0, dot(Normal, L));

    f_diffuse  = spot_diffuse + f_diffuse * LightCB.LightColor.rgb * LightCB.LightColor.a * NL;
    f_specular = spot_specular * LightCB.LightColor.rgb * LightCB.LightColor.a * NL;

    vec3 Ambient = LightCB.AmbientColor.rgb;
    vec3 LitColor = Ambient * DiffuseColor.rgb + (f_specular + f_diffuse);

    ///////////////////////////////////////////
    // SANITIZE COLOR
    ///////////////////////////////////////////

    vec3 lit = LitColor;

    // Remove negatives (your RT is UFLOAT; negatives are invalid anyway)
    lit = max(lit, vec3(0.0));

    // Kill NaN/Inf
    if (any(isnan(lit)) || any(isinf(lit)))
    {
        lit = vec3(0.0);
    }

    // Optional: clamp HDR range to avoid absurd spikes (tune)
    lit = min(lit, vec3(1e4)); // start with something generous

    SceneColor = vec4(lit, FragCB.Color.a);

    ///////////////////////////////////////////
    // INVERSE DEPTH
    ///////////////////////////////////////////

    SceneInvDepth = 1.0 - Depth;

    ///////////////////////////////////////////
    // Motion Vectors (ANF-friendly)
    ///////////////////////////////////////////

    SceneVelocity = SafeMvUv(CalcScreenSpaceDisplacement(v_CurrClip, v_PrevClip));

    if(LightCB.Debug_MVEnabled < 1.0)
    {
        SceneVelocity = vec2(0.0, 0.0);
    }
}