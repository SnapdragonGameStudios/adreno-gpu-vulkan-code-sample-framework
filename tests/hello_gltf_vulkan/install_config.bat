
@echo off
cd /D "%~dp0"

:: Get the name of the current folder (assumed to be the project name)
for %%I in ("%~dp0.") do set "project_name=%%~nxI"

:: Check if app_config.txt exists
if exist "app_config.txt" (
    @echo.
    @echo ****************************************
    @echo Pushing app_config.txt to: /sdcard/Android/data/com.quic.%project_name%/files/
    @echo ****************************************
    adb push ./app_config.txt /sdcard/Android/data/com.quic.%project_name%/files/app_config.txt

    @echo.
    @echo ****************************************
    @echo Done!
    @echo ****************************************
) else (
    @echo.
    @echo ****************************************
    @echo No app_config.txt was found.
    @echo It's not necessary for the app, but it can be used to override application settings.
    @echo If such functionality is desired, please create the file and override the global variables
    @echo according to how they are defined in the project.
    @echo ****************************************
)

:: Pause only if run directly
IF "%~dpnx0"=="%0" PAUSE