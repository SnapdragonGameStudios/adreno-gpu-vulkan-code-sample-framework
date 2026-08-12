//============================================================================================================
//
//                  Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================

// Mode 1 interop kernel: reads RGBA color from a shared buffer, outputs grayscale.
__kernel void grayscale_buffer(
    __global const uchar4* input_color,
    __global uchar4*       output_color,
    uint                   width,
    uint                   height
)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    uchar4 c = input_color[x + width * y];
    uchar  g = (uchar)(0.299f * (float)c.x + 0.587f * (float)c.y + 0.114f * (float)c.z);
    output_color[x + width * y] = (uchar4)(g, g, g, 255);
}
