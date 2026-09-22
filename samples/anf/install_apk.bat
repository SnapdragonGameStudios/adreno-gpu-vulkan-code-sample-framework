REM Copyright (c) 2026, Qualcomm Innovation Center, Inc. All rights reserved.
REM SPDX-License-Identifier: BSD-3-Clause

@echo off
cd /D "%~dp0"

:: Get the name of the current folder (assumed to be the project name)
for %%I in ("%~dp0.") do set "project_name=%%~nxI"

@echo.
@echo ****************************************
@echo Installing APK for project: %project_name%
@echo ****************************************

set "apk_path=..\..\build\android\%project_name%\outputs\apk\debug\%project_name%-debug.apk"

call adb install -r -t "%apk_path%"

@echo.
@echo ****************************************
@echo Done!
@echo ****************************************

IF "%~dpnx0"=="%0" PAUSE
