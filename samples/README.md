# Samples

Unless noted, all samples run on Windows and Android.

## [Cooperative Matrix](cooperative_matrix)
Demonstrates **VK_KHR_cooperative_matrix** for high‑throughput matrix operations such as GEMM and convolution on Adreno™ GPUs.

## [Graph Pipelines](graph_pipelines)
Shows how to use **VK_ARM_tensors**, **VK_ARM_data_graph**, and **VK_QCOM_data_graph_model** to run ML‑backed image processing using Vulkan Data Graph pipelines.

## [HDR Swapchain](hdr_swapchain)
Creates and presents to an **HDR‑capable** Vulkan swapchain, selecting HDR formats/color spaces and falling back to SDR when needed.

## [Image Processing](image_processing)
Implements a bloom effect using **VK_QCOM_image_processing**, with a toggle to compare the extension path against a standard downsample/blur pipeline.

## [Rotated Copy](rotated_copy)
Demonstrates **VK_QCOM_rotated_copy_commands** to perform rotated image copies on devices without rotated‑swapchain support.

## [SGSR](sgsr)
Integrates **Snapdragon™ Game Super Resolution**, with toggles for activation and optional edge‑direction processing.

## [SGSR 2](sgsr2)
Showcases **Snapdragon™ Game Super Resolution 2**, featuring the temporal upscaling **compute 3‑pass** variant optimized for Adreno.

## [Sub Pass](sub_pass)
Highlights multi‑subpass rendering workflows, including MSAA resolve/tonemap performed inside a subpass.

## [Tile Memory](tile_memory)
Explores tile‑local memory usage to reduce external bandwidth and improve on‑chip efficiency.

## [Tile Shading](tile_shading)
Implements tile‑friendly shading techniques designed to maximize performance on tile‑based GPU architectures.