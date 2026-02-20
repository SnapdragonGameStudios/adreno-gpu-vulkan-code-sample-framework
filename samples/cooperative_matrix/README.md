# Cooperative Matrix Sample

![Screenshot](img/screenshot.png)

This sample demonstrates the use of the *VK_KHR_cooperative_matrix* extension in Vulkan to run matrix operations using GPU‑accelerated cooperative matrix arithmetic.

The extension enables the application to query supported matrix tile sizes and data types, allocate the required buffers, and dispatch compute workloads that take advantage of hardware‑level cooperative matrix execution.

The sample highlights how cooperative matrices can significantly improve the performance of operations such as matrix multiplication and convolution, showcasing how Vulkan applications can leverage Qualcomm™-specific GPU capabilities to achieve higher throughput on Adreno™ GPUs.
