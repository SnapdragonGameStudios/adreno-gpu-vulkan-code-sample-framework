# Graph pipelines sample

Runs an image-processing workload through Vulkan data-graph pipelines using `VK_ARM_tensors`, `VK_ARM_data_graph`, and `VK_QCOM_data_graph_model`. The graph path requires compatible device and driver support.

## Execution flow

1. Load the precompiled model pipeline cache and create the data-graph pipeline.
2. Create a pipeline session.
3. Query session memory requirements, allocate memory, and bind it.
4. Bind the graph pipeline and descriptors in a command buffer.
5. Dispatch the graph and synchronize its output before use.

## Model pipeline cache

The graph path needs a `PipelineCache.bin` generated for the model and target driver with a compatible model compiler. The framework build does not generate this cache from an ONNX model.

Place the file at:

```text
samples/graph_pipelines/Media/Misc/PipelineCache.bin
```

If the cache is missing, invalid, or cannot be loaded, the sample keeps rendering with graph upscaling disabled and shows a warning. Supply a valid cache, rebuild the package, and run again to enable graph dispatch.

## Build and run

Follow the [framework setup](../../README.md#configuring), select `graph_pipelines`, and build for the target device. See the [run instructions](../../README.md#running).
