# Tile Memory Heap Sample

![Screenshot](img/screenshot.png)

This sample demonstrates a light clustering algorithm using Vulkan, with specific support for the *VK_QCOM_tile_memory_heap* extension.
This extension allows the application to allocate and manage tile memory, which is used for efficient memory management within a command buffer submission batch.

The sample showcases how tile memory can be used to optimize rendering performance by reducing memory bandwidth and improving cache locality. It implements a forward rendering pipeline with clustered lighting, where lights are grouped based on screen-space tiles. These tiles are processed using tile-local memory allocations, enabling fast access and minimizing global memory usage.

The rendering technique is designed to highlight the benefits of tile memory in scenarios with many dynamic lights, demonstrating how Vulkan applications can leverage Qualcomm™-specific extensions to achieve better performance on Adreno™ GPUs.