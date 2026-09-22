cmake_minimum_required (VERSION 3.25)

project (SampleFramework)

# Include top level configuration
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/../../ConfigLocal.cmake)
  message(FATAL_ERROR "ConfigLocal.cmake does not exits - run \"python configure.py\" in the top level directory to generate this file.")
endif()
include(../../ConfigLocal.cmake)


# Turn exceptions off (msvc)
# Done in this top-level class as it needs to apply to libraries and applications globally.
# Even in 2020 exception support removes compiler/optimizer oppertunities to remove 'redundant' instructions (eg look at dissassembly for move constructors).
# Also output debug information on release (Windows) builds so the Release build is debuggable.
if(false)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  string(REGEX REPLACE "/EH[a-z]+" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE} /ZI")
  set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
  set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
  add_definitions(-D_HAS_EXCEPTIONS=0)
endif()
endif()

# Determine which graphics API we can support and set appropriate defines
set(FRAMEWORK_ENABLE_VULKAN OFF)
set(FRAMEWORK_ENABLE_DX12 OFF)
set(FRAMEWORK_WINDOWS_ARM64 OFF)
if(ANDROID)
  # Android target (Vulkan only)
  set(FRAMEWORK_ENABLE_VULKAN ON)
elseif(WIN32)
  # Windows target (Dx12 for sure, maybe Vulkan too)
  set(FRAMEWORK_ENABLE_DX12 ON)
  #if (NOT CMAKE_GENERATOR_PLATFORM MATCHES "ARM64")
    set(FRAMEWORK_ENABLE_VULKAN ON)
  #endif()
  if (CMAKE_GENERATOR_PLATFORM MATCHES "ARM64")
    set(FRAMEWORK_WINDOWS_ARM64 ON)
  endif()
elseif(UNIX)
  set(FRAMEWORK_ENABLE_VULKAN ON)
endif()

message("FRAMEWORK_ENABLE_VULKAN - ${FRAMEWORK_ENABLE_VULKAN}") 
message("FRAMEWORK_ENABLE_DX12   - ${FRAMEWORK_ENABLE_DX12}")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Add in all the child subdirectories
set(FRAMEWORK_DIR ../../framework)
set(FRAMEWORK_DOWNLOAD_EXTERNALS On CACHE BOOL "Pull down the framework external dependencies" FORCE)
add_subdirectory( ${FRAMEWORK_DIR} framework )
set(FRAMEWORK_DOWNLOAD_EXTERNALS Off CACHE BOOL "Pull down the framework external dependencies" FORCE)

add_subdirectory( ../../samples samples)
