//============================================================================================================
//
//
//                  Copyright (c) 2023, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================


// Uniform Constant Buffer
cbuffer VertCB : register( b0 )
{
    float4x4 MVPMatrix;
    float4x4 ModelMatrix;
    float4x4 ShadowMatrix;
};

// Uniform Constant Buffer
cbuffer FragCB : register( b1 )
{
    float4    Color;

    // X: Normal Height
    // Y: Normal Mirror Reflect Amount
    // Z: Not Used
    // W: Not Used
    float4    NormalHeight;
};

// Light uniform
cbuffer LightCB : register( b2 )
{
    float4x4 ProjectionInv;
    float4x4 ViewInv;
    float4x4 ViewProjectionInv; // ViewInv * ProjectionInv
    float4x4 WorldToShadow;
    float4 ProjectionInvW;    // w components of ProjectionInv
    float4 CameraPos;

    float4 LightDirection;
    float4 LightColor;

    float4 AmbientColor;

    float AmbientOcclusionScale;
    int Width;
    int Height;
};

#define NORMAL_HEIGHT           NormalHeight.x
#define NORMAL_MIRROR_AMOUNT    NormalHeight.y

// Textures
Texture2D u_DiffuseTex : register( t0, space1);
Texture2D u_NormalTex  : register( t1, space1);
sampler s_Linear : register(s0);


// Varyings
struct PSInput
{
    float4 Position : SV_POSITION;
//    float4 v_WorldPos : SV_POSITION;
    float2 UV : TEXCOORD;
    float4 Color : COLOR;
};
//layout (location = 0) in float2   v_TexCoord;
//layout (location = 1) in float3   v_WorldPos;
//layout (location = 2) in float3   v_WorldNorm;
//layout (location = 3) in float3   v_WorldTan;
//layout (location = 4) in float3   v_WorldBitan;
//layout (location = 6) in float4   v_VertColor;


PSInput VSMain(float4 position : POSITION, float2 uv : TEXCOORD, float4 color : COLOR)
{
    PSInput result;

    //result.Position = mul( MVPMatrix, float4(position.xyz, 1.0) );
    result.Position = mul( MVPMatrix, float4(position.xyz, 1.0) );
    //result.Position.y = -result.Position.y;
    //result.Position.z = -result.Position.z;
    result.UV = uv;

    // Need Position in world space
//    result.v_WorldPos = (ModelMatrix * float4(position.xyz, 1.0)).xyz;

//    // Get shadow texture coordinate while we have world position
//    // Expanded out since trying to handle shadows in reflection
//    v_ShadowCoord = biasMat * ShadowMatrix * float4(result.v_WorldPos.xyz, 1.0);

    // Need  Normal, Tangent, and Bitangent in world space
//    v_WorldNorm = (ModelMatrix * float4(a_Normal.xyz, 0.0)).xyz;
//    v_WorldTan = (ModelMatrix * float4(a_Tangent.xyz, 0.0)).xyz;
//    v_WorldBitan = cross(v_WorldNorm, v_WorldTan);

    // Color is simple attribute color
    result.Color.xyzw = float4(color.xyz, 1.0);

    return result;
}



//-----------------------------------------------------------------------------
float3 ScreenToWorld(float2 ScreenCoord/*0-1 range*/, float Depth/*0-1*/)
//-----------------------------------------------------------------------------
{
    // Faster ScreenToWorld does one dotproduct with the inverse projection matrix to the perspective divisor and does one full (xyz) matrix multiply which is then perspective divided
    // Thanks to David McAllister for pointing this out.
    float4 ClipSpacePosition = float4((ScreenCoord * 2.0) - float2(1.0,1.0), Depth, 1.0);
    ClipSpacePosition.y = -ClipSpacePosition.y;

    //  Just one dp4 to calculate w, so 3(4-1) dp4 calculations can be saved from previous float4x4*float4
    float ViewSpacePositionW = dot(ProjectionInvW, ClipSpacePosition);
    
    float3 WorldSpacePosition = mul(ViewProjectionInv, ClipSpacePosition).xyz;
    return WorldSpacePosition.xyz/ViewSpacePositionW;
}

//-----------------------------------------------------------------------------
float4 PSMain(PSInput input) : SV_TARGET
//-----------------------------------------------------------------------------
{
    // ********************************
    // Base (albedo) color
    // ********************************
    // Get color from the color texture

    float4 AlbedoColor = u_DiffuseTex.Sample(s_Linear, input.UV);
    AlbedoColor.xyzw *= input.Color.xyzw;

#if 0
    // Adjust by vertex color.
    AlbedoColor.xyzw *= v_VertColor.xyzw;

    // Get base normal from the bump texture
    float4 NormTexValue = u_NormalTex.Sample(s_Linear, input.UV);
    float3 N = NormTexValue.xyz * 2.0 - 1.0;

    N.xy *= NORMAL_HEIGHT;
    N = normalize(N);

    //// Need matrix to convert to tangent space
    //mat3 WorldToTan = mat3(normalize(v_WorldTan), normalize(v_WorldBitan), normalize(v_WorldNorm));
    
    // Convert the bump normal to tangent space
    float3 BumpNormal = normalize(WorldToTan * N);

    // Setup the color and put depth value in the alpha channel
    AlbedoColor = float4(AlbedoColor.rgb, Color.a);

    // Setup the Normal
    float3 Normal = BumpNormal.xyz;
    float Depth = gl_FragCoord.z; /*1.0 - NormalWithDepth.w;*/

    // ********************************
    // Ambient Occlusion
    // ********************************

    // Determine World position of pixel
    float3 WorldPos = ScreenToWorld( LocalTexCoord, Depth );

    // Calculate ambient (fixed value with darkening by Ambient Occlusion)
    float3 Ambient = AmbientColor.rgb;

    // ********************************
    // Light
    // ********************************
    float3 EyeDir = normalize(CameraPos.xyz - WorldPos);

    float3 LightAmt = Ambient;

    {
        float3 WorldToLightVec = LightDirection.xyz;
        float WorldToLightDist2 = dot(WorldToLightVec, WorldToLightVec);
        float3 WorldToLightNorm = normalize(WorldToLightVec);

        float SpotFalloffAng = dot(float3(0.0, 1.0, 0.0), WorldToLightNorm);
		float SpotFalloff =  clamp((SpotFalloffAng - 0.8) / 0.2, 0.0, 1.0);

        float LightAng = max(0.0, dot( WorldToLightNorm, Normal));

        // Spec (blinn-phong)
        float3 LightDir = WorldToLightNorm;
        float3 HalfVector = normalize(LightDir+EyeDir);
        float Spec = pow(max(dot(Normal,HalfVector),0.0), 100) * 1.5;

        LightAmt += SpotFalloff * LightColor.rgb * (Spec + LightAng) * LightColor.w / (1.0 + WorldToLightDist2);   
    }
#else
    float LightAmt = 1.0;
#endif

    float4 OutputColor;
    OutputColor.rgb = AlbedoColor.rgb * LightAmt;
    OutputColor.a = Color.a;
    return OutputColor;
}

