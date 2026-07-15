@echo off
cd /D "%~dp0"
mkdir solution
pushd solution

echo Looking for Visual Studio...
"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [18.0,19.0] -format value -property displayName |findstr 2026
if %ERRORLEVEL%==0 goto :FOUND2026
"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [17.0,18.0] -format value -property displayName |findstr 2022
if %ERRORLEVEL%==0 goto :FOUND2022
"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [16.0,17.0] -format value -property displayName |findstr 2019
if %ERRORLEVEL%==0 goto :FOUND2019
if exist "%ProgramFiles%\Microsoft Visual Studio\18\Insiders\Common7\IDE\devenv.exe" goto :FOUND2026
echo Not found VS2019, VS2022, or VS2026.  Attempting to create VS2019 solution anyways.

:FOUND2019
cmake.exe -G "Visual Studio 16 2019" ..
if %ERRORLEVEL% ==0  goto :BUILD
popd
echo.
echo Could not build Visual Studio .sln files.  Check above errors (Visual Studio Pro 2022, 2026 or 2019 supported)
goto :EOF

:FOUND2022
cmake.exe -G "Visual Studio 17 2022" ..
if %ERRORLEVEL% ==0  goto :BUILD
popd
echo.
echo Could not build Visual Studio .sln files.  Check above errors (Visual Studio Pro 2022, 2026 or 2019 supported)
goto :EOF

:FOUND2026
cmake.exe -G "Visual Studio 18 2026" ..
if %ERRORLEVEL% ==0  goto :BUILD
popd
echo.
echo Could not build Visual Studio .sln files.  Check above errors (Visual Studio Pro 2022, 2026 or 2019 supported)
goto :EOF

:BUILD
echo Starting debug build.
echo .
cmake.exe --build . --config Debug
rem echo Starting release build.
rem echo .
rem cmake.exe --build . --config Release
popd

echo.
echo Visual Studio solution written to %CD%\solution\SampleFramework.sln 
echo.
