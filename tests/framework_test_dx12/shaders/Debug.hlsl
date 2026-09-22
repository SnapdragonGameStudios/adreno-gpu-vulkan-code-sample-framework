//=============================================================================
//
//
//                  Copyright (c) 2022 QUALCOMM Technologies Inc.
//                              All Rights Reserved.
//
//==============================================================================

// Debug.frag

// Uniform Constant Buffer
cbuffer VertConstantsBuff : register( b0 )
{
    float4x4 MVPMatrix;
    float4x4 ModelMatrix;
};

// Uniform Constant Buffer
cbuffer FragConstantsBuff : register( b1 )
{
    float4  Color;
    float4  EyePos;
    float4  LightDir;
    float4  LightColor;
};

// Textures
Texture2D u_DiffuseTex : register( t0 );
sampler s_Linear : register(s0);

// Varying's
struct PSInput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD;
};


//-----------------------------------------------------------------------------
PSInput VSMain(float4 position : POSITION, float2 uv : TEXCOORD)
//-----------------------------------------------------------------------------
{
    PSInput result;

    result.position = mul(MVPMatrix, float4(position.xyz, 1.0));
    result.uv = float2(uv.x, uv.y);

    //// Need Position in world space
    //v_WorldPos = (VertCB.ModelMatrix * vec4(a_Position.xyz, 1.0)).xyz;

    //// Need  Normal in world space
    //v_WorldNorm = (VertCB.ModelMatrix * vec4(a_Normal.xyz, 0.0)).xyz;

    //// Color is simple attribute color
    //v_VertColor.xyzw = vec4(a_Color.xyz, 1.0);

    return result;
}

//-----------------------------------------------------------------------------
float4 PSMain(PSInput input) : SV_TARGET
//-----------------------------------------------------------------------------
{
#if 0
    // ********************************
    // Base Lighting
    // ********************************
    // Get base color from the color texture
    float4 DiffuseColor = texture( u_DiffuseTex, input.uv.xy );
    // float4 DiffuseColor = float4(0.8, 0.8, 0.8, 1.0);
    DiffuseColor.xyzw *= FragCB.Color.xyzw;

    // Adjust by vertex color.
    DiffuseColor.xyzw *= v_VertColor.xyzw;

    // Need the normal
    vec3 BumpNormal = normalize(v_WorldNorm.xyz);

    // Light direction is from world position to light position
    // vec3 LightDir = normalize(FragCB.LightPos.xyz - v_WorldPos.xyz);
    
    // Can now figure out the diffuse amount
    float DiffuseAmount = max(0.25, dot(BumpNormal.xyz, -FragCB.LightDir.xyz));
    
    // For specular, we need Half vector
    vec3 EyeDir = normalize(FragCB.EyePos.xyz - v_WorldPos.xyz);
    vec3 Half = normalize(EyeDir - FragCB.LightDir.xyz);
    float Specular = max(0.0, dot(BumpNormal.xyz, Half));
    Specular = pow(Specular, FragCB.LightColor.w); 

    // ********************************
    // Start adding in the colors
    // ********************************

    vec3 LightTotal = DiffuseAmount * DiffuseColor.xyz * FragCB.LightColor.xyz;

    // TODO: This equation may no longer be correct since removed many parts
    vec3 FinalColor =   LightTotal + Specular * FragCB.LightColor.xyz;

    // Write out the color and put depth value in the alpha channel
    FragColor = vec4(FinalColor.rgb, FragCB.Color.a);

    // DEBUG! DEBUG! DEBUG! DEBUG! DEBUG! DEBUG! DEBUG! DEBUG! DEBUG! 
    FragColor = vec4(DiffuseColor.xyz, 1.0);
#else

    float4 DiffuseColor = u_DiffuseTex.Sample(s_Linear, input.uv.xy);
    return float4(DiffuseColor.xyz, 1.0);

#endif
}

