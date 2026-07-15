@echo off
cd /D "%~dp0"

if "%1"=="" (
  echo Building all projects [build.bat can be invoved with project name argument to build just one project, if desired]
  call "%~dp0gradlew.bat" build -Dorg.gradle.warning.mode=none --no-daemon -Dorg.gradle.jvmargs=-Xmx4096m --continue -x test
) else (
  echo Building project : %1
  call "%~dp0gradlew.bat" :%1:buildNeeded -Dorg.gradle.warning.mode=none --no-daemon -Dorg.gradle.jvmargs=-Xmx4096m --configure-on-demand -x test
)

