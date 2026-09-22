# Cooperative matrix sample

![Cooperative matrix sample output](img/screenshot.png)

Runs matrix operations on supported Adreno™ GPUs using `VK_KHR_cooperative_matrix`. The application queries supported tile sizes and component types before creating a workload.

The default layout comparison measures alternative input layouts. The UI also exposes custom layouts and legacy tests. Validate GPU output against the CPU reference before interpreting timings. Available cases, including Qualcomm™-specific paths, depend on device capabilities.

## Build and run

Follow the [framework setup](../../README.md#configuring), select `cooperative_matrix`, and build the target platform. Use the [run instructions](../../README.md#running) for installation, working directories, and configuration.
