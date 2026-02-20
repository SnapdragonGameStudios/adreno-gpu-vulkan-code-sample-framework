# Graph Pipelines

![Screenshot](img/screenshot.png)

This sample demonstrates how to use the **VK_ARM_tensors**, **VK_ARM_data_graph**, and **VK_QCOM_data_graph_model** Vulkan extensions to run an ML‑powered image‑upscaling pipeline on supported hardware.

The app renders the scene at a lower resolution, copies the result into tensor objects, dispatches a **Data Graph** pipeline (optionally backed by Qualcomm’s model‑import path), and writes the upscaled output back into an image for display. When these extensions are unavailable or disabled, the sample falls back to a standard GPU blit.

The sample highlights how Vulkan applications can integrate tensor operations, graph pipelines, and Qualcomm™‑specific model execution paths to achieve real‑time upscaling performance on Adreno™ GPUs.
