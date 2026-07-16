# Subpass Sample

For mobile tile-based GPUs, subpasses are an important way to save memory bandwidth, improve power efficiency, and help performance.

This sample demonstrates Vulkan subpasses by optionally running a filmic tonemapping operator as a subpass of the main scene pass. The on-screen UI can enable or disable the subpass path so the impact can be measured with external profiling tools.

When the subpass path is enabled, the tonemap work can consume the scene color data while it is still tile-local instead of forcing an additional off-chip store and reload. In prior Snapdragon Profiler captures for this sample, enabling the subpass path reduced the number of intermediate surfaces and lowered total read/write bandwidth.

## Running

- If you haven't already, setup the framework and build the code [instructions here](../../README.md#configuring)
- Running this sample has no special additional requirements [instructions here](../../README.md#running)
