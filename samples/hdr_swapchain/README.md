# HDR swapchain sample

![HDR swapchain sample output](img/screenshot.png)

Queries surface formats and color spaces and presents a scene through an HDR-capable Vulkan swapchain. On Adreno™ GPUs and other supported devices, an HDR display and supported surface format are required to inspect the HDR path.

The sample also uses `VK_QCOM_render_pass_transform` when available. Check the selected swapchain format and color space when comparing displays.

## Build and run

Follow the [framework setup](../../README.md#configuring), select `hdr_swapchain`, and build the target platform. Use the [run instructions](../../README.md#running) for installation, working directories, and configuration.
