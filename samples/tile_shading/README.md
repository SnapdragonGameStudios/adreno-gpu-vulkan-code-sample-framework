# Tile Shading Sample

![Screenshot](img/screenshot.png)

This sample demonstrates a tile-based shading technique using Vulkan, with support for the *VK_QCOM_tile_memory_heap* extension.

The extension enables the application to allocate and manage tile-local memory, which is scoped to the duration of a command buffer submission and optimized for high-bandwidth, low-latency access within a tile.

The sample implements a forward rendering pipeline where shading computations are performed per tile, rather than per pixel or per fragment. This approach leverages the tiling architecture of Adreno™ GPUs to reduce memory traffic and improve cache efficiency.

By using tile memory, the sample avoids costly round-trips to global memory for intermediate shading data. Instead, lighting calculations and material evaluations are performed directly in tile-local memory, which is faster and more power-efficient.

The technique is particularly well-suited for mobile GPUs, where bandwidth and power are constrained. It demonstrates how Vulkan applications can take advantage of Qualcomm™-specific extensions to optimize rendering workloads and achieve better performance on Snapdragon™ platforms.