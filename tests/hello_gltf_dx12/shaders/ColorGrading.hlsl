//============================================================================================================
//
//
//                  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

// Textures
Texture2D u_DiffuseTex : register( t0, space1 );
sampler s_Linear : register(s0);
RWTexture2D<float2> g_output : register(u0, space1);

[numthreads(8, 8, 1)]
void CSMain(uint2 dispatch_thread_id : SV_DispatchThreadID)
{
   g_output[dispatch_thread_id] = 0.42;
}

