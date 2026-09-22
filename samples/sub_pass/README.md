# Subpass sample

![Subpass sample output](project/img/screenshot.png)

Applies a filmic tonemapping operator either as a subpass of the scene render pass or through a separate path. Use the on-screen control to compare them.

A subpass can consume an attachment while the data remains in tile memory. Measure bandwidth and frame time on the target device; results depend on the workload and driver.

## Build and run

Follow the [framework setup](../../README.md#configuring), select `sub_pass`, and build the target platform. Use the [run instructions](../../README.md#running) for installation, working directories, and configuration.
