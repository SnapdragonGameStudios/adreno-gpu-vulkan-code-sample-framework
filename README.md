# Adreno™ GPU Vulkan code sample framework

The framework provides C++ components and sample applications for Vulkan rendering on Adreno™ GPUs. It supports Windows and Android build targets. Individual samples may require device-specific Vulkan extensions, SDKs, or driver support.

## Contents

- [Resources](#resources)
- [Requirements](#requirements)
- [Configuring](#configuring)
- [Building](#building)
- [Running](#running)
- [Repository layout](#repository-layout)
- [Contributing](#contributing)
- [License](#license)

## Resources

| Resource | Use it for |
|---|---|
| [Samples](samples/README.md) | Choosing a rendering or compute example |
| [Framework](framework/README.md) | Understanding the application entry point and shared components |
| [Test applications](tests/README.md) | Checking a build or starting a small application |

## Requirements

Install Git, Python, CMake, and the [Vulkan SDK](https://vulkan.lunarg.com/). Put Python and CMake on `PATH`. The existing setup has used Python 3.10.9 and CMake 3.30 or newer. Use a Vulkan SDK that supplies the extensions required by the selected sample.

| Target | Additional requirements |
|---|---|
| Windows | Visual Studio with C++ build tools. Scripts detect Visual Studio 2019, 2022, or 2026. Use a CMake version that supports the selected generator. |
| Android | Android SDK, NDK, Java JDK, Ninja, and an `arm64-v8a` device with the required Vulkan capabilities. Gradle currently selects NDK `26.0.10792818`. |
| Linux | CMake and a compatible C++/Vulkan environment. Linux support has been exercised under WSL; inspect `project/linux/` for its build configuration. |

For Android, use the Java runtime required by the selected Android Gradle Plugin. Set `JAVA_HOME` to that installation. A compatible Android Studio installation includes a suitable Java runtime.

If Gradle cannot locate the Android SDK or CMake, create `project/android/local.properties` with paths for your machine:

```properties
sdk.dir=C:/Android/Sdk
cmake.dir=C:/Tools/CMake
```

The Android configuration requires CMake 3.25 or newer. Keep local paths out of source control.

## Configuring

From the repository root, run:

```powershell
python Configure.py
```

On Windows, `01_Configure.bat` runs the same command. The shell wrapper is `01_Configure.sh`.

Select the samples or tests and target platforms. Expand a submenu with the right-arrow key. Keep required dependencies and the Tools target selected, then choose **Save And Begin Processing**.

The script saves the selection in `ConfigLocal.*`, downloads dependencies, builds asset tools, generates platform projects, and starts the selected builds. Samples and dependencies differ by branch; `Config.txt` defines the available choices.

Check the [sample guides](samples/README.md) for dependencies that require a separate download or setup step.

## Building

Rebuild the saved selection from the repository root:

```powershell
python Configure.py --build
```

The equivalent wrappers are `02_Build.bat` and `02_Build.sh`.

Windows scripts generate `SampleFramework.sln` under `project/windows/solution/` or the selected architecture's solution directory. The default Windows batch build uses Debug. Build other configurations from the generated solution.

For example, rebuild the `empty` test after selecting it in the configuration:

```powershell
.\project\android\build.bat empty
```

Replace `empty` with the selected sample or test target. Calling the script without a name builds all configured Android targets.

Android Studio can open `project/android`, and VS Code can use the supplied CMake settings template. Reproduce IDE build failures with the repository scripts before reporting them.

## Running

### Android

Debug APKs are written to `build/android/<target>/outputs/apk/debug/`. Use the target's `install_apk.bat` when supplied, or install the APK with `adb`:

```powershell
adb install -r build/android/empty/outputs/apk/debug/empty-debug.apk
```

Enable USB debugging and select the intended device before installing. Use the target's `install_config.bat` to copy `app_config.txt` when supplied.

### Windows

Run each executable with its sample or test directory as the working directory so it can locate assets and configuration. Generated Visual Studio projects set that directory. Executables are under the solution's `samples/<target>/<configuration>/` or `tests/<target>/<configuration>/` directory.

Some samples read an optional `app_config.txt`. Use the settings documented by that sample.

## Repository layout

| Path | Contents |
|---|---|
| `framework/` | Shared application, rendering, resource, and platform code |
| `samples/` | Rendering and compute examples |
| `tests/` | Small framework test applications |
| `project/` | Platform builds, asset tools, and CMake helpers |
| `Config.txt` | Selectable targets and dependency downloads |
| `framework/external/`, `samples/external/` | Downloaded dependencies |

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) for validation and sign-off requirements. Participation follows the [code of conduct](CODE-OF-CONDUCT.md).

## License

The Adreno™ GPU Vulkan framework source uses the [BSD 3-Clause License](LICENSE.txt). Bundled dependencies, sample SDKs, and media may have separate terms. Preserve their licenses and attribution.
