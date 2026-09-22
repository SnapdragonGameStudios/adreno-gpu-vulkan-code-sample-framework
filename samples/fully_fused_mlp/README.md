# Fully fused MLP sample

Runs a multilayer perceptron forward pass as a Vulkan compute workload. It compares a fused cooperative-matrix dispatch, an ALU implementation, and a baseline that dispatches each layer separately.

The cooperative-matrix paths use `VK_KHR_cooperative_matrix` and the Qualcomm™ extension [`VK_QCOM_cooperative_matrix_conversion`](https://github.com/KhronosGroup/GLSL/blob/main/extensions/qcom/GLSL_QCOM_cooperative_matrix_conversion.txt).

## Networks and execution modes

| Network | Shape | Width choices |
|---|---|---|
| RGBA | Input and hidden layers use the selected width; output has four channels | 16, 32, 64 |
| Wide-IO | 12 inputs, one hidden layer, 10 outputs | 16, 64 |

Hidden layers can use ReLU. The output layer is linear.

| Mode | Behavior |
|---|---|
| ALU | Runs an FP16 compute kernel without cooperative matrices |
| Coopmat | Fuses the forward pass using cooperative matrices |
| Unfused | Dispatches each layer separately and stores intermediate activations in global memory |

Coopmat offers GPR, local, and global fusing strategies. GPR fusing keeps hidden state in registers and supports width 16 only. Local fusing uses shared memory. Global fusing stores intermediate state in global memory.

## Controls and validation

Use the UI to select the network, mode, width, activation, bias, batch size, repeat count, warm-up iterations, validation count, and tolerance. Changing shader configuration recompiles the compute shader through the embedded glslang compiler.

Inputs, weights, and biases are deterministic. Compare GPU results with the CPU reference before using the timing table. Warm-up iterations are excluded from steady-state timing. Unfused timing sums the layer dispatches for one inference.

## Requirements

Coopmat and Unfused need both extensions listed above on compatible Adreno™ GPUs. The UI disables unavailable modes and widths. ALU still requires the Vulkan and FP16 capabilities used by its shader; it is not supported on every Vulkan device.

## Build and run

Follow the [framework setup](../../README.md#configuring), select `fully_fused_mlp`, and use the [run instructions](../../README.md#running).

Windows is the previously documented test target. Android needs the same device capabilities. Check Android resources and build output before deployment.
