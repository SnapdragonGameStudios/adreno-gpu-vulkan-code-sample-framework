//============================================================================================================
//
//
//                  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

// Textures
Texture2D u_DiffuseTex : register( t0, space1 );
Texture2D u_OverlayTex : register( t1, space1 );
sampler s_Linear : register(s0);

// Varyings
struct PSInput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD;
    float4 color : COLOR;
};


//-----------------------------------------------------------------------------
PSInput VSMain(float4 position : POSITION, float2 uv : TEXCOORD, float4 color : COLOR)
//-----------------------------------------------------------------------------
{
    PSInput result;

    result.position = float4(position.xyz, 1.0);
    result.uv = float2(uv.x, uv.y);
    result.color = float4(color.xyz, 1.0);

    return result;
}


//-----------------------------------------------------------------------------
float4 PSMain(PSInput input) : SV_TARGET
//-----------------------------------------------------------------------------
{
    float2 LocalTexCoord = input.uv;

    // ********************************
    // Texture Colors
    // ********************************
    // Get base color from the color texture
    float4 DiffuseColor = u_DiffuseTex.Sample(s_Linear, LocalTexCoord);

    // Multiply by vertex color.
    DiffuseColor *= input.color;

    // Apply darkening/lightening control
    // float lerp01 = min(1,FragCB.Diffuse);
    // float lerp12 = max(0,FragCB.Diffuse-1);
    // DiffuseColor = DiffuseColor * lerp01 + lerp12 - lerp12 * DiffuseColor;

    // Get the Overlay value
    float4 OverlayColor = u_OverlayTex.Sample(s_Linear, LocalTexCoord);

    // ********************************
    // Alpha Blending
    // ********************************
    float4 OutputColor;
    OutputColor.rgb = DiffuseColor.rgb *(1.0-OverlayColor.a) + OverlayColor.rgb;
    OutputColor.a = 1.0;
    return OutputColor;
}

