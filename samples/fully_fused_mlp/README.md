# Fully Fused MLP Sample

This sample runs a fully‑fused multi‑layer‑perceptron (MLP) forward pass on the
GPU as a Vulkan compute workload. 

It demonstrates how an MLP inference can be *fused* into a single compute
dispatch using the *[VK_KHR_cooperative_matrix](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html)*
extension together with the Qualcomm™ *VK_QCOM_cooperative_matrix_conversion*
extension *[link](https://github.com/KhronosGroup/GLSL/blob/main/extensions/qcom/GLSL_QCOM_cooperative_matrix_conversion.txt)* (`vectorToCoopmatQCOM` / `coopmatToVectorQCOM`), and compares that
against a plain‑ALU implementation and an unfused per‑layer baseline.

Every configuration is switchable **at run time**: each change recompiles the
compute shader through the embedded glslang runtime compiler and re‑dispatches,
while the window keeps presenting and shows the live FPS — exactly like the
`cooperative_matrix` sample. Weights, inputs and biases are filled
deterministically so the GPU result can be validated against a CPU reference,
and a results table shows per‑sample CPU‑vs‑GPU values with pass/fail and the
steady‑state timing.

## Networks

Two network topologies are selectable from the **Network** dropdown:

- **RGBA** — 1 input + 2 hidden + 1 output, `in == hidden == width`, producing
  **4** (RGBA) output channels. Width selectable **16 / 32 / 64**.
- **Wide‑IO** — a fixed **12 → W → 10** network: 12 input features, one hidden
  layer of width **W**, and 10 output channels. Hidden width **16 / 64**.

Hidden layers are ReLU‑able; the output layer is always linear (so the network
can emit signed values).

## Modes

- **Execution mode**
  - **ALU** — no cooperative matrices; a native‑fp16 compute kernel (for RGBA the
    single‑fiber, tile‑interleaved `forward_single_fiber_tile_interleaved`
    kernel).
  - **Coopmat** — the whole forward pass fused into one dispatch using
    `VK_KHR_cooperative_matrix`. Three fusing strategies:
    - **GPR fusing** — fiber‑vector coopmat; hidden state kept in registers.
      16‑wide only.
    - **Local fusing** — input/hidden state cached in shared (LDS) memory.
    - **Global fusing** — hidden state round‑trips through global memory.
  - **Unfused (baseline)** — the same coopmat math, but one dispatch **per
    layer** with the intermediate activations written to and read back from
    global memory between dispatches. It exists as a baseline to measure the cost
    that fusion saves; its reported timing is the **summed per‑inference latency**
    over the layer dispatches.
- **Width** — 16 / 32 / 64 for RGBA (16 / 64 for Wide‑IO; GPR fusing is 16‑wide
  only).
- **Activation** — None / ReLU (hidden layers only).
- **Bias** — Zero / Random.
- **Batch size**, perf‑loop count, warm‑up iterations (excluded from
  steady‑state timing), validation sample count, results rows shown, and
  validation epsilon.

## Capability gating

The coopmat kernels use the QCOM cooperative‑matrix conversion ops for their
output store, so both the **Coopmat** and **Unfused** modes require **both**
`VK_KHR_cooperative_matrix` **and** `VK_QCOM_cooperative_matrix_conversion`.

- If either extension is **not** supported, the **Coopmat** and **Unfused** modes
  are disabled in the UI (with an explanatory note) and only **ALU** mode runs.
- GPR fusing is additionally restricted to 16‑wide.

Capabilities are detected at startup and the unavailable modes/strategies/widths
are greyed out, so the sample runs on any device — falling back to the ALU path
where cooperative matrices are unavailable.

## Running

- If you haven't already, setup the framework and build the code
  [instructions here](../../README.md#configuring)
- Running this sample has no special additional requirements
  [instructions here](../../README.md#running)
- Coopmat and Unfused modes require a device exposing both
  `VK_KHR_cooperative_matrix` and `VK_QCOM_cooperative_matrix_conversion`
  (Qualcomm Adreno™); otherwise the sample runs the ALU path.

## Android note

Windows is the primary, tested target. For an Android APK build, copy the
launcher icons from another sample into this sample's resource folders (they are
binary PNGs not included here):

```
samples/cooperative_matrix/project/android/res/mipmap-*/ic_launcher.png
  ->  samples/fully_fused_mlp/project/android/res/mipmap-*/ic_launcher.png
```
