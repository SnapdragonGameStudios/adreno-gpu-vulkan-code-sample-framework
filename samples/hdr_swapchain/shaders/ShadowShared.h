// Copyright (c) Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// ShadowShared.h

//
// Shared Implementation of Shadow mapping code.
//

//-----------------------------------------------------------------------------
float GetInShadow(vec4 LocalShadowCoord)
//-----------------------------------------------------------------------------
{
    float fBias = 0.001;
    float NormalizedCoord = (LocalShadowCoord.z / LocalShadowCoord.w) - fBias;

    // Flag for in shadow
    // Step returns 0.0 of second parameter is less that first parameter.  Returns 1.0 otherwise

//    float InShadow = (LocalShadowCoord.x >= 0.0 && LocalShadowCoord.x < 1.0 && LocalShadowCoord.y >= 0.0 && LocalShadowCoord.y < 1.0) ? 1 : 0;
    float InShadow = ( texture( u_ShadowMap, LocalShadowCoord.xy ) ).r;
    InShadow = min(1.0, step(0.0, InShadow - NormalizedCoord) + smoothstep(0.7, 1.0, InShadow));
    float RetVal = InShadow;

    return RetVal;
}

#ifdef USE_VARIANCE_SHADOWMAP
//-----------------------------------------------------------------------------
float GetInShadowVSM(vec4 LocalShadowCoord)
//-----------------------------------------------------------------------------
{
    // Variance Shadow Map.
    float NormalizedCoord = LocalShadowCoord.z + 0.02;

    vec2 DepthAndDepthSq = texture(u_ShadowMap, LocalShadowCoord.xy).xy;
    float Variance = DepthAndDepthSq.y - (DepthAndDepthSq.x * DepthAndDepthSq.x);
    Variance = max( Variance, 0.01 );   // Clamp variance to help with self-shadowing

    float edgeFade = smoothstep(0.40, 0.50, max(abs(LocalShadowCoord.x - 0.5), abs(LocalShadowCoord.y - 0.5))); //1 outside 'edge' of shadow map, 0 inside he map area
    float Distance = NormalizedCoord - DepthAndDepthSq.x;
	if (Distance < 0.0)
		return 1.0;//not shadowed
    return max(edgeFade, Variance / (Variance + (Distance * Distance)));
}
#endif // USE_VARIANCE_SHADOWMAP

#ifdef USE_PCF_SAMPLING
//-----------------------------------------------------------------------------
float GetInShadowPCF(vec4 LocalShadowCoord)
//-----------------------------------------------------------------------------
{
    float RetVal = 0.0;

    // Set up the blur coordinates...
    vec4 blurTexCoords[16];
    blurTexCoords[  0] = LocalShadowCoord + vec4(-1.5 * FragCB.ShadowSize.z, -1.5 * FragCB.ShadowSize.w, 0.0, 0.0);
    blurTexCoords[  1] = LocalShadowCoord + vec4(-0.5 * FragCB.ShadowSize.z, -1.5 * FragCB.ShadowSize.w, 0.0, 0.0);
    blurTexCoords[  2] = LocalShadowCoord + vec4( 0.5 * FragCB.ShadowSize.z, -1.5 * FragCB.ShadowSize.w, 0.0, 0.0);
    blurTexCoords[  3] = LocalShadowCoord + vec4( 1.5 * FragCB.ShadowSize.z, -1.5 * FragCB.ShadowSize.w, 0.0, 0.0);

    blurTexCoords[  4] = LocalShadowCoord + vec4(-1.5 * FragCB.ShadowSize.z, -0.5 * FragCB.ShadowSize.w, 0.0, 0.0);
    blurTexCoords[  5] = LocalShadowCoord + vec4(-0.5 * FragCB.ShadowSize.z, -0.5 * FragCB.ShadowSize.w, 0.0, 0.0);
    blurTexCoords[  6] = LocalShadowCoord + vec4( 0.5 * FragCB.ShadowSize.z, -0.5 * FragCB.ShadowSize.w, 0.0, 0.0);
    blurTexCoords[  7] = LocalShadowCoord + vec4( 1.5 * FragCB.ShadowSize.z, -0.5 * FragCB.ShadowSize.w, 0.0, 0.0);

    blurTexCoords[  8] = LocalShadowCoord + vec4(-1.5 * FragCB.ShadowSize.z,  0.5 * FragCB.ShadowSize.w, 0.0, 0.0);
    blurTexCoords[  9] = LocalShadowCoord + vec4(-0.5 * FragCB.ShadowSize.z,  0.5 * FragCB.ShadowSize.w, 0.0, 0.0);
    blurTexCoords[ 10] = LocalShadowCoord + vec4( 0.5 * FragCB.ShadowSize.z,  0.5 * FragCB.ShadowSize.w, 0.0, 0.0);
    blurTexCoords[ 11] = LocalShadowCoord + vec4( 1.5 * FragCB.ShadowSize.z,  0.5 * FragCB.ShadowSize.w, 0.0, 0.0);

    blurTexCoords[ 12] = LocalShadowCoord + vec4(-1.5 * FragCB.ShadowSize.z,  1.5 * FragCB.ShadowSize.w, 0.0, 0.0);
    blurTexCoords[ 13] = LocalShadowCoord + vec4(-0.5 * FragCB.ShadowSize.z,  1.5 * FragCB.ShadowSize.w, 0.0, 0.0);
    blurTexCoords[ 14] = LocalShadowCoord + vec4( 0.5 * FragCB.ShadowSize.z,  1.5 * FragCB.ShadowSize.w, 0.0, 0.0);
    blurTexCoords[ 15] = LocalShadowCoord + vec4( 1.5 * FragCB.ShadowSize.z,  1.5 * FragCB.ShadowSize.w, 0.0, 0.0);


    // ...and then read the values
    RetVal += GetInShadow(blurTexCoords[  0]);
    RetVal += GetInShadow(blurTexCoords[  1]);
    RetVal += GetInShadow(blurTexCoords[  2]);
    RetVal += GetInShadow(blurTexCoords[  3]);

    RetVal += GetInShadow(blurTexCoords[  4]);
    RetVal += GetInShadow(blurTexCoords[  5]);
    RetVal += GetInShadow(blurTexCoords[  6]);
    RetVal += GetInShadow(blurTexCoords[  7]);

    RetVal += GetInShadow(blurTexCoords[  8]);
    RetVal += GetInShadow(blurTexCoords[  9]);
    RetVal += GetInShadow(blurTexCoords[ 10]);
    RetVal += GetInShadow(blurTexCoords[ 11]);

    RetVal += GetInShadow(blurTexCoords[ 12]);
    RetVal += GetInShadow(blurTexCoords[ 13]);
    RetVal += GetInShadow(blurTexCoords[ 14]);
    RetVal += GetInShadow(blurTexCoords[ 15]);

    // Just took 16 samples so divide to get average			
	RetVal /= 16.0 ;

    return RetVal;
}
#endif // USE_PCF_SAMPLING

//-----------------------------------------------------------------------------
float GetShadowAmount(vec4 ShadowCoord)
//-----------------------------------------------------------------------------
{
    // Shadow Calculations (textureProj returns 1.0 for pass (not in shadow) and 0.0 for fail (in shadow))
#if defined(USE_VARIANCE_SHADOWMAP)
    float ShadowAmount = GetInShadowVSM(ShadowCoord);
#elif defined(USE_PCF_SAMPLING)
    float ShadowAmount = GetInShadowPCF(ShadowCoord);
#else
    float ShadowAmount = GetInShadow(ShadowCoord);
#endif

    //float MinShadow = 0.5f;
    //ShadowAmount.w = max(ShadowAmount.w, MinShadow);

    ShadowAmount = mix(0.5, 1.0, ShadowAmount);
    return ShadowAmount;
}

