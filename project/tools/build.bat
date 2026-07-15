@echo off
cd /D "%~dp0"
mkdir solution
pushd ..\..
cd framework\external\KTX-Software
git fetch origin 'refs/tags/*:refs/tags/*'
popd
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
echo Could not build Visual Studio 2019 .sln files.  Check above errors (Visual Studio Pro 2022, 2026 or 2019 supported)
goto :EOF

:FOUND2022
cmake.exe -G "Visual Studio 17 2022" ..
if %ERRORLEVEL% ==0  goto :BUILD
echo Could not build Visual Studio 2022 .sln files.  Check above errors.
popd
goto :EOF

:FOUND2026
cmake.exe -G "Visual Studio 18 2026" ..
if %ERRORLEVEL% ==0  goto :BUILD
echo Could not build Visual Studio 2026 .sln files.  Check above errors.
popd
goto :EOF

:BUILD
cmake.exe --build . --config Release
@popd
