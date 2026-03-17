# Vulkan Sample Framework

The Adreno™ GPU Vulkan Code Sample Framework is a lightweight collection of C++ classes and sample projects to demonstrate Vulkan rendering features on the Qualcomm Snapdragon Adreno™ GPU.

Both Android and Microsoft Windows build targets are supported (some samples may require/target Qualcomm Adreno™ specific Vulkan extensions).

## Prerequisites

- Git https://git-scm.com/downloads
- Python (tested against 3.10.9)
- CMake (tested against 3.30+) https://cmake.org/download/
- Vulkan SDK (1.3 or later) https://vulkan.lunarg.com/

### Windows

- Visual Studio 2022

### Android

- Android Studio (install NDK and SDK)
- Ninja https://github.com/ninja-build/ninja/releases <br>
    After installation ninja.exe should be copied in to the same directory as cmake.exe - this works around a current [July2022] open bug with gradle where it expects ninja to be in the same directory as cmake!)
- Java JDK<br>
    Ensure JAVA_HOME environment variable points to a current version of Java.  While a 'standalone' Java can be used it is highly recommended to use the Java build shipped inside Android Studio as the Android build system and gradle are very sensitive to Java versions.  Eg. `set JAVA_HOME=c:\Program Files\Android\Android Studio\jbr`

### Linux

Tested using WSL running Ubuntu.


## Build Setup

### Windows

Ensure CMake and python are in the Windows PATH

### Android

Ensure you have CMake and a recent version of Android SDK installed (Android Studio can install both).
Android builds require CMake version 3.25 and above, if when building Android projects gradle is throwing arrors about cmake or Android SDK/NDK versions you can override the defaults by creating `project/android/local.properties` and adding the location of your local Android SDK and Cmake installs eg:
<pre><code>sdk.dir=C\:\\Users\\yournamehere\\AppData\\Local\\Android\\Sdk
cmake.dir=c\:\\Program Files\\CMake</code></pre>


## Configuring

In the root folder run the configuration/build script:
`01_Configure.bat` (source `01_Configure.sh` on Linux)

Select the build targets and solutions you are interested in building (sub-menus can be opened with the 'right' arrow).
Then select 'Save And Begin Processing' which:
- saves your selections.
- creates configuration files for building (`ConfigLocal.cmake` / `ConfigLocal.properties`).
- cleans any previous builds (if targets to clean are selected)
- downloads any external project dependencies (eg tinygltfloader, glm, KTX-Software).
- runs project\tools\build.bat to build tools required for building.
- runs project\windows\build.bat and/or buildArm64.bat (if windows target is selected)
	- which creates the MSVC solution containing your selected projects; project\windows\solution\SampleFramework.sln
	- attempts to build this solution for both debug and release.
- runs project\android\build.bat (if Android target is selected)
	- which runs gradle to build the apk targets for your selected projects.

## Subsequent builds

01_Configure.bat can be used to re-run builds after changes.

Alternately the Windows solution can be used to make changes and re-build Windows targets
	- project\windows\solution\SampleFramework.sln
	- project\windowsArm64\solution\SampleFramework.sln
and Android builds can be re-ran with
	- project\android\build.bat

If desired you can also open/build the SampleFramework using Android Studio.  In Android Studio open the project/android folder (initial load of the projects takes a while, subsequent opens are fast).  Using Android Studio to build is untested and not supported!  If you are having problems building framework samples please test using the batch file build scripts before opening any support requests.

## Running

See the [Samples](samples) folder for instructions on building assets and running individual samples.

Most samples also support a configuration file (`app_config.txt`) placed in the base of the sample's directory (Windows) or pushed loose to the sample's install folder (Android).

### Android

Android apk are written to `build\android\<samplename>\outputs\apk\debug\`.
Config files may be pushed to the device using the `sample/<samplename>/install_config.bat` script provided, if available.
Most sample apks can be installed with the `sample/<samplename>/install_apk.bat` script provided, if available.
If the Android apk was correctly built (see note above about binary assets) it can be installed with `apk install <apk>` and run from Android.

### Windows

Executables are written to `project\windows\solution\samples\<samplename>\debug\` or `project\windowsArm64\solution\samples\<samplename>\debug\`.
When running sample executables from command line ensure you are running from the relevant sample directory (not the location of the built exe).  The Visual Studio solution is already configured to launch samples from the correct directory.

## Directory Structure

- [/framework](framework)
Contains the sample framework code.
- [/framework/external](framework/external)
Contains external projects downloaded and extracted by Configure.py
- [/project](project)
Platform specific top level build files and build tools.
- [/samples](samples)
Samples that use the framework.
- [/samples/external](samples/external)
External projects shared between samples, downloaded and extracted by Configure.py

## License
Adreno™ GPU Vulkan Code Sample Framework is licensed under the BSD 3-clause “New” or “Revised” License. Check out the [LICENSE](LICENSE) for more details.
