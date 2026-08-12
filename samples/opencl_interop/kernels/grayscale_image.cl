//============================================================================================================
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

// Mode 2/3 interop kernel: reads RGBA color from a shared image2d, outputs grayscale to a shared image2d.
__kernel void grayscale_image(
    __read_only  image2d_t input_color,
    __write_only image2d_t output_color
)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    int2 coord = (int2)(get_global_id(0), get_global_id(1));

    float4 c = read_imagef(input_color, sampler, coord);
    float  g = 0.299f * c.x + 0.587f * c.y + 0.114f * c.z;
    write_imagef(output_color, coord, (float4)(g, g, g, 1.0f));
}