cmake_minimum_required (VERSION 3.25 FATAL_ERROR)  # 3.7 added FindVulkan, 3.10 supported MSVC c++17 (without manually setting flags), 3.19 (maybe before) supported C++20 on Win/Android, 3.21 added Android NDK support with NDK r23 although we are not currently using it!, 3.25 for fetchontent features
include(FindVulkan)

project (framework C CXX)

set(CMAKE_CXX_STANDARD 20)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
define_property(TARGET PROPERTY FOLDER INHERITED BRIEF_DOCS "Set the folder name." FULL_DOCS  "Use to organize targets in an IDE.")
set_property(DIRECTORY "." PROPERTY FOLDER "framework/")

include("../ConfigLocal.cmake")

# Pull down all the external dependencies needed by the Framework
# we only do this if we are a 'top level' framework project such as with Windows (Android gradle will include the framework in each application's project but we dont want to re-get externals)
option(FRAMEWORK_DOWNLOAD_EXTERNALS "Set download/fetch external repositiories" No)
include(cmake/ExternalDependencies.cmake)

set(FRAMEWORK_LIB_OUTPUT "" CACHE STRING "Location to output framework libraries (use on Android to output static libraries from gradle)")

# Base functionality for Framework applications (including command-line executables)
set(CPP_BASE_SRC
    code/main/frameworkApplicationBase.cpp
    code/main/frameworkApplicationBase.hpp
    code/mesh/instanceGenerator.cpp
    code/mesh/instanceGenerator.hpp
    code/mesh/meshLoader.cpp
    code/mesh/meshLoader.hpp
    code/mesh/meshIntermediate.cpp
    code/mesh/meshIntermediate.hpp
    code/mesh/octree.cpp
    code/mesh/octree.hpp
    code/system/config.cpp
    code/system/config.h
    code/system/containers.cpp
    code/system/containers.h
    code/system/crc32c.hpp
    code/system/glm_common.hpp
    code/system/math_common.hpp
    code/system/os_common.cpp
    code/system/os_common.h
)

# Graphics API agnostic framework code
set(CPP_GENERIC_SRC
    code/allocator/threadBufferResource.hpp
    code/allocator/threadBufferResourceHelper.hpp
    code/allocator/threadManagedBufferResourceAllocator.hpp
    code/allocator/threadMonotonicBufferResourceAllocator.hpp
    code/animation/animation.cpp
    code/animation/animation.hpp
    code/animation/animationData.hpp
    code/animation/animationGltfLoader.cpp
    code/animation/animationGltfLoader.hpp
    code/animation/skeleton.cpp
    code/animation/skeleton.hpp
    code/animation/skeletonData.cpp
    code/animation/skeletonData.hpp
    code/animation/skeletonGltfLoader.cpp
    code/animation/skeletonGltfLoader.hpp
    code/animation/skin.cpp
    code/animation/skin.hpp
    code/animation/skinData.cpp
    code/animation/skinData.hpp
    code/animation/skinGltfLoader.cpp
    code/animation/skinGltfLoader.hpp
    code/camera/camera.cpp
    code/camera/camera.hpp
    code/camera/cameraController.cpp
    code/camera/cameraController.hpp
    code/camera/cameraControllerAnim.cpp
    code/camera/cameraControllerAnim.hpp
    code/camera/cameraControllerTouch.cpp
    code/camera/cameraControllerTouch.hpp
    code/camera/cameraData.hpp
    code/camera/cameraGltfLoader.cpp
    code/camera/cameraGltfLoader.hpp
    code/graphicsApi/commandList.hpp
    code/graphicsApi/graphicsApiBase.cpp
    code/graphicsApi/graphicsApiBase.hpp
    code/graphicsApi/pipeline.hpp
    code/graphicsApi/renderContext.hpp
    code/graphicsApi/renderPass.hpp
    code/graphicsApi/renderTarget.hpp
    code/gui/gui.hpp
    code/gui/imguiBase.cpp
    code/gui/imguiBase.hpp
    code/gui/imguiPlatform.hpp
    code/light/light.cpp
    code/light/light.hpp
    code/light/lightData.hpp
    code/light/lightList.cpp
    code/light/lightList.hpp
    code/light/lightListGui.cpp
    code/light/lightListGui.hpp
    code/light/lightGltfLoader.cpp
    code/light/lightGltfLoader.hpp
    code/light/lightLoader.cpp
    code/light/lightLoader.hpp
    code/material/computable.cpp
    code/material/computable.hpp
    code/material/descriptorSetLayout.hpp
    code/material/drawable.hpp
    code/material/drawableLoader.cpp
    code/material/drawableLoader.hpp
    code/material/material.cpp
    code/material/material.hpp
    code/material/materialT.hpp
    code/material/materialPass.hpp
    code/material/materialManagerT.hpp
    code/material/pipeline.hpp
    code/material/pipelineLayout.hpp
    code/material/pipelineVertexInputState.hpp
    code/material/shader.hpp
    code/material/shaderDescription.cpp
    code/material/shaderDescription.hpp
    code/material/shaderModule.hpp
    code/material/shaderManager.cpp
    code/material/shaderManager.hpp
    code/material/shaderManagerT.hpp
    code/material/specializationConstants.hpp
    code/material/specializationConstantsLayout.hpp
    code/memory/buffer.hpp
    code/memory/drawIndirectBuffer.hpp
    code/memory/memory.hpp
    code/memory/memoryManager.hpp
    code/memory/memoryMapped.hpp
    code/memory/indexBuffer.hpp
    code/memory/vertexBuffer.hpp
    code/memory/uniform.hpp
    code/mesh/instanceGenerator.cpp
    code/mesh/instanceGenerator.hpp
    code/mesh/mesh.hpp
    code/mesh/meshHelper.cpp
    code/mesh/meshHelper.hpp
    code/shadow/shadow.cpp
    code/shadow/shadow.hpp
    code/system/assetManager.hpp
    code/system/profile.cpp
    code/system/profile.h
    code/system/timer.cpp
    code/system/timer.hpp
    code/system/Worker.cpp
    code/system/Worker.h
    code/texture/loaderKtx.cpp
    code/texture/loaderKtx.hpp
    code/texture/loaderPpm.cpp
    code/texture/loaderPpm.hpp
    code/texture/sampler.cpp
    code/texture/sampler.hpp
    code/texture/texture.hpp
    code/texture/textureFormat.cpp
    code/texture/textureFormat.hpp
    code/texture/textureManager.cpp
    code/texture/textureManager.hpp
)

# OS independant (Vulkan targetted) source here
set(CPP_VULKAN_SRC
    code/main/applicationHelperBase.cpp
    code/main/applicationHelperBase.hpp
    code/gui/imguiVulkan.cpp
    code/gui/imguiVulkan.hpp
    code/helper/postProcess.hpp
    code/helper/postProcessStandard.cpp
    code/helper/postProcessStandard.hpp
    code/helper/postProcessSMAA.cpp
    code/helper/postProcessSMAA.hpp
    code/helper/zbufferReduce.cpp
    code/helper/zbufferReduce.hpp
    code/material/vulkan/computable.cpp
    code/material/vulkan/computable.hpp
    code/material/vulkan/drawable.cpp
    code/material/vulkan/drawable.hpp
    code/material/descriptorSetDescription.hpp
    code/material/vulkan/descriptorSetLayout.cpp
    code/material/vulkan/descriptorSetLayout.hpp
    code/material/materialManager.hpp
    code/material/materialProps.cpp
    code/material/materialProps.h
    code/material/materialShaderDefinition.cpp
    code/material/materialShaderDefinition.hpp
    code/material/vulkan/material.cpp
    code/material/vulkan/material.hpp
    code/material/vulkan/materialManager.cpp
    code/material/vulkan/materialManager.hpp
    code/material/vulkan/materialPass.cpp
    code/material/vulkan/materialPass.hpp
    code/material/vulkan/pipeline.cpp
    code/material/vulkan/pipeline.hpp
    code/material/vulkan/pipelineLayout.cpp
    code/material/vulkan/pipelineLayout.hpp
    code/material/vulkan/pipelineVertexInputState.cpp
    code/material/vulkan/pipelineVertexInputState.hpp
    code/material/vulkan/shader.cpp
    code/material/vulkan/shader.hpp
    code/material/vulkan/shaderManager.hpp
    code/material/vulkan/shaderModule.cpp
    code/material/vulkan/shaderModule.hpp
    code/material/specializationConstantDescription.hpp
    code/material/vulkan/specializationConstants.cpp
    code/material/vulkan/specializationConstants.hpp
    code/material/vulkan/specializationConstantsLayout.cpp
    code/material/vulkan/specializationConstantsLayout.hpp
    code/material/vulkan/vertexDescription.cpp
    code/material/vulkan/vertexDescription.hpp
    code/material/vertexFormat.hpp
    code/memory/vulkan/bufferObject.cpp
    code/memory/vulkan/bufferObject.hpp
    code/memory/vulkan/drawIndirectBufferObject.cpp
    code/memory/vulkan/drawIndirectBufferObject.hpp
    code/memory/vulkan/indexBufferObject.cpp
    code/memory/vulkan/indexBufferObject.hpp
    code/memory/vulkan/memoryManager.cpp
    code/memory/vulkan/memoryManager.hpp
    code/memory/vulkan/memoryMapped.hpp
    code/memory/vulkan/uniform.cpp
    code/memory/vulkan/uniform.hpp
    code/memory/vulkan/vertexBufferObject.cpp
    code/memory/vulkan/vertexBufferObject.hpp
    code/shadow/shadow.cpp
    code/shadow/shadow.hpp
    code/shadow/shadowVsm.cpp
    code/shadow/shadowVsm.hpp
    code/shadow/shadowVulkan.cpp
    code/shadow/shadowVulkan.hpp
    code/texture/vulkan/loaderKtx.cpp
    code/texture/vulkan/loaderKtx.hpp
    code/texture/vulkan/sampler.cpp
    code/texture/vulkan/sampler.hpp
    code/texture/vulkan/texture.cpp
    code/texture/vulkan/texture.hpp
    code/texture/vulkan/textureManager.cpp
    code/texture/vulkan/textureManager.hpp
    code/vulkan/commandBuffer.cpp
    code/vulkan/commandBuffer.hpp
    code/vulkan/extension.cpp
    code/vulkan/extension.hpp
    code/vulkan/extensionHelpers.cpp
    code/vulkan/extensionHelpers.hpp
    code/vulkan/extensionLib.cpp
    code/vulkan/extensionLib.hpp
    code/vulkan/framebuffer.cpp
    code/vulkan/framebuffer.hpp
    code/vulkan/MeshObject.cpp
    code/vulkan/MeshObject.h
    code/vulkan/renderContext.cpp
    code/vulkan/renderContext.hpp
    code/vulkan/renderPass.cpp
    code/vulkan/renderPass.hpp
    code/vulkan/refHandle.hpp
    code/vulkan/renderTarget.cpp
    code/vulkan/renderTarget.hpp
    code/vulkan/semaphore.cpp
    code/vulkan/semaphore.hpp
    code/vulkan/TextureFuncts.cpp
    code/vulkan/TextureFuncts.h
    code/vulkan/vulkan.cpp
    code/vulkan/vulkan.hpp
    code/vulkan/vulkanDebugCallback.cpp
    code/vulkan/vulkanDebugCallback.hpp
    code/vulkan/timerPool.cpp
    code/vulkan/timerPool.hpp
    code/vulkan/timerSimple.cpp
    code/vulkan/timerSimple.hpp
    code/vulkan/vulkan_support.cpp
    code/vulkan/vulkan_support.hpp
)

# DX12 specific source files here
set(CPP_DX12_SRC
    code/main/applicationHelperBaseDx12.cpp
    code/main/applicationHelperBaseDx12.hpp
    code/dx12/dx12.cpp
    code/dx12/dx12.hpp
    code/dx12/commandList.cpp
    code/dx12/commandList.hpp
    code/dx12/descriptorHeapManager.cpp
    code/dx12/descriptorHeapManager.hpp
    code/dx12/renderPass.cpp
    code/dx12/renderPass.hpp
    code/dx12/renderContext.cpp
    code/dx12/renderContext.hpp
    code/dx12/renderTarget.cpp
    code/dx12/renderTarget.hpp
    code/gui/imguiDx12.cpp
    code/gui/imguiDx12.hpp
    code/material/dx12/computable.cpp
    code/material/dx12/computable.hpp
    code/material/dx12/descriptorSetLayout.hpp
    code/material/dx12/descriptorSetLayout.cpp
    code/material/dx12/drawableDx12.cpp
    code/material/dx12/drawableDx12.hpp
    code/material/dx12/material.cpp
    code/material/dx12/material.hpp
    code/material/dx12/materialManager.cpp
    code/material/dx12/materialManager.hpp
    code/material/dx12/materialPass.hpp
    code/material/dx12/materialPass.cpp
    code/material/dx12/pipeline.cpp
    code/material/dx12/pipeline.hpp
    code/material/dx12/pipelineLayout.cpp
    code/material/dx12/pipelineLayout.hpp
    code/material/dx12/pipelineVertexInputState.cpp
    code/material/dx12/pipelineVertexInputState.hpp
    code/material/dx12/shader.hpp
    code/material/dx12/shaderModule.cpp
    code/material/dx12/shaderModule.hpp
    code/material/dx12/specializationConstantsLayout.hpp
    code/material/dx12/vertexDescription.cpp
    code/material/dx12/vertexDescription.hpp
    code/memory/dx12/bufferObject.cpp
    code/memory/dx12/bufferObject.hpp
    code/memory/dx12/indexBufferObject.cpp
    code/memory/dx12/indexBufferObject.hpp
    code/memory/dx12/memoryMapped.hpp
    code/memory/dx12/memoryManager.cpp
    code/memory/dx12/memoryManager.hpp
    code/memory/dx12/uniform.cpp
    code/memory/dx12/uniform.hpp
    code/memory/dx12/vertexBufferObject.cpp
    code/memory/dx12/vertexBufferObject.hpp
    code/texture/dx12/loaderKtx.cpp
    code/texture/dx12/loaderKtx.hpp
    code/texture/dx12/sampler.cpp
    code/texture/dx12/sampler.hpp
    code/texture/dx12/texture.cpp
    code/texture/dx12/texture.hpp
    code/texture/dx12/textureManager.cpp
    code/texture/dx12/textureManager.hpp
)

# Platform Specific source files here
if(WIN32)
  set(CPP_BASE_SRC ${CPP_BASE_SRC}
      code/system/windows/windowsAssetManager.cpp
      code/main/windows/winMain.cpp
  )
  set(CPP_GENERIC_SRC ${CPP_GENERIC_SRC}
      code/gui/windows/imguiWindows.cpp
  )
elseif(ANDROID)
  set(CPP_BASE_SRC ${CPP_BASE_SRC}
      code/system/android/androidAssetManager.cpp
      code/main/android/androidMain.cpp
  )
  set(CPP_GENERIC_SRC ${CPP_GENERIC_SRC}
      code/memory/androidHardwareBuffer.cpp
      code/memory/androidHardwareBuffer.hpp
      code/gui/android/imguiAndroid.cpp
  )
  set(CPP_VULKAN_SRC ${CPP_VULKAN_SRC}
  )
  include_directories( ${ANDROID_NDK}/sources/android/native_app_glue/ )
elseif(UNIX)
  set(CPP_BASE_SRC ${CPP_BASE_SRC}
      code/system/linux/linuxAssetManager.cpp
      code/main/linux/linuxMain.cpp
  )
  set(CPP_GENERIC_SRC ${CPP_GENERIC_SRC}
      code/gui/linux/imguiLinux.cpp
  )
endif()

# Framework support for Ray Tracing/Query
if(TRUE)
  set(CPP_VULKAN_SRC ${CPP_VULKAN_SRC}
      code/rayTracing/accelerationStructure.hpp
      code/vulkanRT/accelerationStructure.cpp
      code/vulkanRT/accelerationStructure.hpp
      code/vulkanRT/accelerationInstanceBufferObject.cpp
      code/vulkanRT/accelerationInstanceBufferObject.hpp
      code/vulkanRT/extensionLibRT.hpp
      code/vulkanRT/meshObjectRT.cpp
      code/vulkanRT/meshObjectRT.hpp
      code/vulkanRT/meshUpdateRT.cpp
      code/vulkanRT/meshUpdateRT.hpp
      code/vulkanRT/sceneRT.cpp
      code/vulkanRT/sceneRT.hpp
      code/vulkanRT/traceable.cpp
      code/vulkanRT/traceable.hpp
      code/vulkanRT/vulkanRT.cpp
      code/vulkanRT/vulkanRT.hpp
    )
endif()

# Any externals we need to compile as part of framework here
set(EXTERNAL_BASE_SRC
    external/tinyobjloader/tiny_obj_loader.cc
)
set(EXTERNAL_GENERIC_SRC
    external/imgui/imgui.h
    external/imgui/imgui.cpp
    external/imgui/imgui_demo.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/implot/implot.h
    external/implot/implot.cpp
    external/implot/implot_items.cpp
    external/implot/implot_internal.h
    external/implot/implot_demo.cpp
)
set(EXTERNAL_VULKAN_SRC
    external/imgui/backends/imgui_impl_vulkan.cpp
    external/imgui/backends/imgui_impl_vulkan.h
)
set(EXTERNAL_DX12_SRC
    external/D3D12MemoryAllocator/src/D3D12MemAlloc.cpp
    external/D3D12MemoryAllocator/include/D3D12MemAlloc.h
    external/imgui/backends/imgui_impl_dx12.cpp
    external/imgui/backends/imgui_impl_dx12.h
)
# Platform Specific external files to compile here
if(WIN32)
  set(EXTERNAL_GENERIC_SRC ${EXTERNAL_GENERIC_SRC}
      external/imgui/backends/imgui_impl_win32.cpp
      external/imgui/backends/imgui_impl_win32.h
  )
elseif(ANDROID)
    set(EXTERNAL_GENERIC_SRC ${EXTERNAL_GENERIC_SRC}
    )
elseif(UNIX)
  set(EXTERNAL_GENERIC_SRC ${EXTERNAL_GENERIC_SRC}
      external/imgui/backends/imgui_impl_glfw.cpp
      external/imgui/backends/imgui_impl_glfw.h
  )
endif()

# Json schemas here (so they are shown in the Solution Explorer)
set(JSON_SCHEMA
    schema/lightsSchema.json
    schema/shaderSchema.json
)

# Visualizers (Visual Studio)
set(BASE_NATVIS_SCHEMA
    external/glm/util/glm.natvis
    external/json/nlohmann_json.natvis
)
set(NATVIS_GENERIC_SCHEMA
    external/imgui/misc/debuggers/imgui.natvis
)
set(NATVIS_VULKAN_SCHEMA
    external/VulkanMemoryAllocator/src/vk_mem_alloc.natvis
)
set(NATVIS_DX12_SCHEMA
    external/D3D12MemoryAllocator/src/D3D12MemAlloc.natvis
)

# Create the gfx api agnostic framework libraries
if (FRAMEWORK_framework_base)
  add_library(framework_base STATIC ${CPP_BASE_SRC} ${EXTERNAL_BASE_SRC} ${BASE_NATVIS_SCHEMA})
  target_include_directories(framework_base     PUBLIC code)
  target_include_directories(framework_base     PUBLIC external)
  target_include_directories(framework_base     PUBLIC external/glm)  # so code can do #include "glm/mat3x3.hpp" etc
  target_include_directories(framework_base     PUBLIC external/json/single_include)
  if(ANDROID)
    target_link_libraries(framework_base        PUBLIC android log)
    target_compile_options(framework_base       PRIVATE -Wno-nullability-completeness)
    target_compile_options(framework_base       PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:DEBUG>:-O3>")
    target_compile_definitions(framework_base   PRIVATE OS_ANDROID)
  elseif(WIN32)
    target_compile_definitions(framework_base   PRIVATE OS_WINDOWS;_CRT_SECURE_NO_WARNINGS)
  elseif(UNIX)
    target_compile_definitions(framework_base   PRIVATE OS_LINUX)
    find_package(glfw3 REQUIRED)
    target_link_libraries(framework_base glfw)
  else()
	  message(FATAL_ERROR "Need to have either ANDROID or WIN32 or UNIX defined in cmake")
  endif()
  
  # MSVC hierachy
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/code"      FILES ${CPP_BASE_SRC})
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/external"  PREFIX "external" FILES ${EXTERNAL_BASE_SRC})
endif()



if(FRAMEWORK_framework_generic)
  add_library(framework      STATIC ${CPP_GENERIC_SRC} ${EXTERNAL_GENERIC_SRC} ${JSON_SCHEMA} ${NATVIS_GENERIC_SCHEMA})
  target_include_directories(framework          PUBLIC code)
  target_include_directories(framework          PUBLIC external)
  target_include_directories(framework          PUBLIC external/glm)  # so code can do #include "glm/mat3x3.hpp" etc
  target_include_directories(framework          PUBLIC external/json/single_include)
  target_include_directories(framework          PUBLIC external/imgui)
  target_include_directories(framework          PUBLIC external/implot)
  if(ANDROID)
    target_compile_options(framework            PRIVATE -Wno-nullability-completeness;-Wno-deprecated-volatile;-Wno-deprecated-anon-enum-enum-conversion)
    target_compile_options(framework            PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:DEBUG>:-O3>")
    target_compile_definitions(framework        PRIVATE OS_ANDROID)
  elseif(WIN32)
    target_compile_definitions(framework        PRIVATE OS_WINDOWS;_CRT_SECURE_NO_WARNINGS)
  elseif(UNIX)
    target_compile_definitions(framework        PRIVATE OS_LINUX)
  endif()

  # Vulkan backend or not, set no prototypes
  target_compile_definitions(framework PUBLIC IMGUI_IMPL_VULKAN_NO_PROTOTYPES)
  
  # MSVC hierachy
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/code"      FILES ${CPP_GENERIC_SRC})
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/external"  PREFIX "external" FILES ${EXTERNAL_GENERIC_SRC})
endif()

# Make sure our graphics api flags are set (Android in particular may/will not have gone through a top level CMakeLists.txt to set these)
if(NOT DEFINED FRAMEWORK_ENABLE_VULKAN)
  set(FRAMEWORK_ENABLE_VULKAN ON)
  set(FRAMEWORK_ENABLE_DX12 OFF)
endif()

# Vulkan framework support
if(FRAMEWORK_ENABLE_VULKAN AND FRAMEWORK_framework_vulkan)

  if(ANDROID)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_ANDROID_KHR)
  elseif(WIN32)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
  endif()

  # Add vulkan framework dependency libraries
  set(ENABLE_OPT OFF CACHE BOOL "Disable SPIRV-Tools optimizer" FORCE)
  set(ENABLE_C_INTERFACE ON CACHE BOOL "" FORCE)
  add_subdirectory(external/volk)
  add_subdirectory(external/SPIRV-Cross)
  add_subdirectory(external/glslang)
  #add_subdirectory(external/slang)

  add_library(framework_vulkan STATIC ${CPP_VULKAN_SRC} ${EXTERNAL_VULKAN_SRC})

  # Make sure framework_vulkan builds after glslang libs
  add_dependencies(framework_vulkan
    glslang
    glslang-default-resource-limits
    SPIRV
  )

  target_include_directories(framework_vulkan   PUBLIC code)
  target_include_directories(framework_vulkan   PUBLIC external)
  target_include_directories(framework_vulkan   PUBLIC external/glm)  # so code can do #include "glm/mat3x3.hpp" etc
  target_include_directories(framework_vulkan   PUBLIC external/json/single_include)
  target_include_directories(framework_vulkan   PUBLIC external/imgui)
  target_include_directories(framework_vulkan   PUBLIC external/implot)

  target_link_libraries(framework_vulkan PUBLIC framework)

  target_include_directories(framework_vulkan   PUBLIC external/Vulkan-Headers/include)        # point the framework to the Vulkan includes that we have as a submodule
  
  #do not add the system Vulkan SDK headers, since we use the Vulkan-Headers submodule for consistency and don't want to shadow it
  #get_target_property(VulkanHeaders_INCLUDE_DIRS Vulkan::Headers INTERFACE_INCLUDE_DIRECTORIES)
  #target_include_directories(framework_vulkan PUBLIC "${VulkanHeaders_INCLUDE_DIRS}")
  
  target_include_directories(framework_vulkan   PUBLIC external/volk)                          # point the framework to the volk includes that we have as a submodule
  target_include_directories(framework_vulkan   PUBLIC external/SPIRV-Cross/include)
  target_include_directories(framework_vulkan   PUBLIC external/glslang)
  target_include_directories(framework_vulkan   PUBLIC external/slang/include)

  # Link vulkan framework library dependencies
  target_link_libraries(
    framework_vulkan
    PUBLIC
    glslang
    glslang-default-resource-limits
    spirv-cross-core
    spirv-cross-cpp
    spirv-cross-glsl
    spirv-cross-hlsl
    spirv-cross-msl
    spirv-cross-reflect
    spirv-cross-util
    SPIRV
  )
  
  target_compile_definitions(framework_vulkan PUBLIC VK_ENABLE_BETA_EXTENSIONS)

  if(ANDROID)
    target_compile_options(framework_vulkan     PRIVATE -Wno-nullability-completeness)
    target_compile_options(framework_vulkan     PRIVATE "$<$<CONFIG:RELEASE>:-O3>" "$<$<CONFIG:DEBUG>:-O3>")
    target_compile_definitions(framework_vulkan PRIVATE OS_ANDROID)
    target_compile_definitions(framework_vulkan PUBLIC VK_USE_PLATFORM_ANDROID_KHR)
  elseif(WIN32)
    target_compile_definitions(framework_vulkan PRIVATE OS_WINDOWS;_CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(framework_vulkan PUBLIC VK_USE_PLATFORM_WIN32_KHR)
    if (FRAMEWORK_WINDOWS_ARM64)
      target_link_directories(framework_vulkan BEFORE INTERFACE ../project/windows/libs_arm64)
    endif()
  elseif(UNIX)
    target_compile_definitions(framework_vulkan PRIVATE OS_LINUX)
  endif()
  
  # MSVC hierachy
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/code"      PREFIX "code" FILES ${CPP_VULKAN_SRC})
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/external"  PREFIX "external" FILES ${EXTERNAL_VULKAN_SRC})
endif()

# DX12 framework support
if(FRAMEWORK_ENABLE_DX12 AND FRAMEWORK_framework_dx12)
  add_library(framework_dx12 STATIC ${CPP_DX12_SRC} ${EXTERNAL_DX12_SRC})
  target_include_directories(framework_dx12     PUBLIC code)
  target_include_directories(framework_dx12     PUBLIC external)
  target_include_directories(framework_dx12     PUBLIC external/glm)  # so code can do #include "glm/mat3x3.hpp" etc
  target_include_directories(framework_dx12     PUBLIC external/json/single_include)
  target_include_directories(framework_dx12     PUBLIC external/imgui)
  target_include_directories(framework_dx12     PUBLIC external/implot)
  target_include_directories(framework_dx12     PUBLIC external/D3D12MemoryAllocator/include)

  target_link_libraries(framework_dx12 framework)
  target_link_libraries(framework_dx12 d3d12.lib)
  target_link_libraries(framework_dx12 dxgi.lib)
  target_link_libraries(framework_dx12 dxguid.lib)

  if(ANDROID)
    message(FATAL_ERROR "Android build does not support DirectX12")
  else()
    target_compile_definitions(framework_dx12   PRIVATE OS_WINDOWS;_CRT_SECURE_NO_WARNINGS)
  endif()

  # MSVC hierachy
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/code"      PREFIX "code" FILES ${CPP_DX12_SRC})
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/external"  PREFIX "external" FILES ${EXTERNAL_DX12_SRC})
endif()

# KTX library
if(FRAMEWORK_framework_generic AND FRAMEWORK_framework_external_KTX-Software)
  target_link_libraries(framework     PUBLIC ktx_read)
endif()

if(ANDROID)
  # Setup Android native_app_glue
  #include_directories( ${ANDROID_NDK}/sources/android/native_app_glue/ )
  #add_library( app-glue STATIC ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c )
  #target_link_libraries( framework app-glue )
  message("ndk at ${ANDROID_NDK}")

  # Export ANativeActivity_onCreate(),
  # Refer to: https://github.com/android-ndk/ndk/issues/381.
  set(CMAKE_SHARED_LINKER_FLAGS
      "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")
  set(CMAKE_SHARED_LINKER_FLAGS
      "${CMAKE_SHARED_LINKER_FLAGS} -u android_main")
endif()

# framework links framework_base
if(FRAMEWORK_framework_generic)
  target_link_libraries(framework PUBLIC framework_base)
endif()



# Potentially build shared library versions too and copy the static library into a more easily accessable location for potential use by other projects.
# We likely will do this for the 'framework' project only (projects that use the framework dont need to re-build the shared library)
if(FRAMEWORK_LIB_OUTPUT)
	add_library(framework_base_shared SHARED code/main/framework_static_empty.cpp)
	target_link_libraries(framework_base_shared PUBLIC framework_base)
	add_custom_command(TARGET framework_base POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:framework_base> ${FRAMEWORK_LIB_OUTPUT}/${CMAKE_BUILD_TYPE}/$<TARGET_FILE_NAME:framework_base>)

	add_library(framework_shared SHARED code/main/framework_static_empty.cpp)
	if(FRAMEWORK_framework_generic)
    	target_link_libraries(framework_shared PUBLIC framework)
		add_custom_command(TARGET framework POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:framework> ${FRAMEWORK_LIB_OUTPUT}/${CMAKE_BUILD_TYPE}/$<TARGET_FILE_NAME:framework>)
		if(FRAMEWORK_framework_external_KTX-Software)
			add_custom_command(TARGET framework POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:ktx_read> ${FRAMEWORK_LIB_OUTPUT}/${CMAKE_BUILD_TYPE}/$<TARGET_FILE_NAME:ktx_read>)
		endif()
	endif()

    if (FRAMEWORK_ENABLE_VULKAN AND FRAMEWORK_framework_vulkan)
	    add_library(framework_vulkan_shared SHARED code/main/framework_static_empty.cpp)
	    target_link_libraries(framework_vulkan_shared PUBLIC framework_vulkan)
	    add_custom_command(TARGET framework_vulkan POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:framework_vulkan> ${FRAMEWORK_LIB_OUTPUT}/${CMAKE_BUILD_TYPE}/$<TARGET_FILE_NAME:framework_vulkan>)
    endif()
    if (FRAMEWORK_ENABLE_DX12)
	    add_library(framework_dx12_shared SHARED code/main/framework_static_empty.cpp)
	    target_link_libraries(framework_dx12_shared PUBLIC framework_dx12)
    endif()


  # Copy external libraries using a local target's post-build step
  add_custom_command(TARGET framework_vulkan POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:glslang>
            ${FRAMEWORK_LIB_OUTPUT}/${CMAKE_BUILD_TYPE}/$<TARGET_FILE_NAME:glslang>)
  add_custom_command(TARGET framework_vulkan POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:glslang-default-resource-limits>
            ${FRAMEWORK_LIB_OUTPUT}/${CMAKE_BUILD_TYPE}/$<TARGET_FILE_NAME:glslang-default-resource-limits>)
  add_custom_command(TARGET framework_vulkan POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:SPIRV>
            ${FRAMEWORK_LIB_OUTPUT}/${CMAKE_BUILD_TYPE}/$<TARGET_FILE_NAME:SPIRV>)

endif()


# MSVC hierachy
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/schema"  PREFIX "schema" FILES ${JSON_SCHEMA})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}"         PREFIX "natvis" FILES ${NATVIS_SCHEMA})
