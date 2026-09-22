<!--
Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
-->

# Adreno Neural Fusion Vulkan sample

This sample integrates Adreno Neural Fusion into an Android Vulkan renderer. It renders an animated scene and provides controls for two techniques:

- **Super Resolution (SR)** reconstructs a higher-resolution image from a lower-resolution render.
- **Frame Generation (FG)** creates intermediate frames between rendered frames.

The [official ANF SDK](https://github.com/SnapdragonGameStudios/adreno-neural-fusion) provides the API, integration documentation, and [device requirements](https://github.com/SnapdragonGameStudios/adreno-neural-fusion#requirements).

## Build and run

ANF requires an Android device with Snapdragon™ 8 Elite Gen 6 or higher. Broader platform support is planned. The Android sample builds for `arm64-v8a`.

1. Follow the [framework setup instructions](../../README.md) to install the build tools, Android SDK/NDK, and Vulkan SDK.
2. From the repository root, run `python Configure.py` or `01_Configure.bat`. Select the `anf` sample and the **Android** build target. Keep the required **Tools** target selected.
3. Choose **Save And Begin Processing**. Configure downloads the SDK and framework dependencies, then builds the sample.
4. Connect your device with USB debugging enabled and run `samples\anf\install_apk.bat`. Open **SGS ANF** on the device.

For subsequent builds, run `python Configure.py --build`. The Debug APK is written to `build/android/anf/outputs/apk/debug/anf-debug.apk`.

You can also select **Windows x64** to explore the scene and rendering code on a PC. The Windows sample renders without ANF; SR and FG are available on supported Android devices.

## Controls

SR and FG start disabled. Use the **ANF Sample Controls** panel to enable either feature and compare the result with the original scene. The panel shows whether each feature is supported and displays GPU timing information.

You can also switch between **Immediate** dispatch, where ANF submits its own work, and **Indirect** dispatch, where the application records and submits the commands. Availability depends on the device and driver.

The sample keeps rendering when ANF is unavailable. OpenCL is optional for APK installation.

## How the integration works

The renderer supplies scene color, depth, motion vectors, and camera jitter to ANF. SR reconstructs the output image, and optional FG produces intermediate frames. The UI is composited afterward so it stays separate from the reconstructed or generated scene.

Start with [application.cpp](code/main/application.cpp) for the frame loop, [anf_interface.cpp](code/main/anf_interface.cpp) for resource handling, and [anf_sdk_backend.cpp](code/main/anf_sdk_backend.cpp) for SDK calls.

For integration details, see the SDK's [integration guide](https://github.com/SnapdragonGameStudios/adreno-neural-fusion/blob/main/INTEGRATION-GUIDE.md) and [debug overlay guide](https://github.com/SnapdragonGameStudios/adreno-neural-fusion/blob/main/DEBUG-OVERLAY.md).

## License

This sample is licensed under [BSD-3-Clause](LICENSE.txt). The ANF SDK has separate licenses for its [public headers](https://github.com/SnapdragonGameStudios/adreno-neural-fusion/blob/main/LICENSE-BSD-3-Clause.txt) and [runtime library](https://github.com/SnapdragonGameStudios/adreno-neural-fusion/blob/main/LICENSE.txt). Android builds include these notices in the APK.
