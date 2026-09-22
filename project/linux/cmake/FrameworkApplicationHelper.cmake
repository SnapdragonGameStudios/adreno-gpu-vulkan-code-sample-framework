#
# Build the application, linking against the framework.
#

# Input:
#  CPP_SRC       - list of cpp/hpp files
#  SHADERS_SRC   - list of 'shader' (.vert, .frag, .comp) source files
#  FRAMEWORK_LIB    - name of the framework to link against (eg framework_vulkan)
#  NATVIS_SCHEMA - list of application specific Visual Studio visualization schemas (for debuffer)
#  PROJECT_NAME  - name of the application being compiled (from the 'project(...)' command)

# Windows and Android differ in terms of output.
# Windows generates an executable, Android generates a library (that the AndroidManifest references)
if(WIN32)
    set( TARGET_NAME ${PROJECT_NAME} )
    add_executable( ${TARGET_NAME} WIN32 ${CPP_SRC} ${SHADERS_SRC} ${NATVIS_SCHEMA})
    target_compile_definitions( ${TARGET_NAME} PRIVATE OS_WINDOWS;_CRT_SECURE_NO_WARNINGS )
    set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
elseif(ANDROID)
    set( TARGET_NAME native-lib )
    add_library( ${TARGET_NAME} SHARED ${CPP_SRC} ${SHADERS_SRC} )
    target_compile_definitions(${TARGET_NAME} PRIVATE OS_ANDROID)

    # Android also needs app-glue generating (and linking before the framework)
    add_library( app-glue STATIC ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c )
    target_link_libraries( ${TARGET_NAME} app-glue )
    # Export ANativeActivity_onCreate(), see https://github.com/android-ndk/ndk/issues/381.
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")
    # FRAMEWORK_DIR is set by the calling gradle script (needed for Android)
    add_subdirectory( ${FRAMEWORK_DIR} framework )
elseif(UNIX)
    set( TARGET_NAME ${PROJECT_NAME} )
    add_executable( ${TARGET_NAME} ${CPP_SRC} ${SHADERS_SRC} ${NATVIS_SCHEMA})
    target_compile_definitions( ${TARGET_NAME} PRIVATE OS_LINUX)
else()
    message(FATAL_ERROR "(currently) Unsupported platform")
endif()

if(NOT DEFINED FRAMEWORK_LIB)
    set(FRAMEWORK_LIB framework_vulkan)
endif()

#
# Link the framework (to our application)
#
target_link_libraries( ${TARGET_NAME} ${FRAMEWORK_LIB} )

include_directories(.)
